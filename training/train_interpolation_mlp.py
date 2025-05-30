import argparse
import json
import os
import time
import torch
import wandb

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from dataclasses import dataclass
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Resize
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor, get_cosine_schedule_with_warmup
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings, CLIPVisionConfig, CLIPVisionModel
from tqdm.auto import tqdm

from qlip.clip_models import convert_to_qlip_CLIP, QLIPVisionTransformer
from qlip.config_templates import config_template

def collate_fn(examples, processor, max_image_size):
    max_width = max([example["image"].size[0] for example in examples])
    max_height = max([example["image"].size[1] for example in examples])

    batched_tensor = torch.zeros(len(examples), 3, max_height, max_width)
    square_tensor = torch.zeros(len(examples), 3, 336, 336)
    mask_tensor = torch.zeros(len(examples), max_height, max_width)

    input_ids = []
    attention_masks = []

    for i, example in enumerate(examples):
        img = example["image"].convert("RGB")

        img = resize_too_small(img, 28)
        img = resize_too_big(img, max_image_size)

        if img.size[0] < 28 or img.size[1] < 28:
            print(f"Image size too small: {img.size}")
            continue

        processor.image_processor.do_resize = False
        processor.image_processor.do_center_crop = False

        inputs = processor(
            images=[img],
            text=["This is some text"],
            return_tensors="pt",
            padding=True,
            is_split_into_words=True,
        )

        image = inputs["pixel_values"][0]
        width, height = image.shape[-1], image.shape[-2]
        batched_tensor[i, :, :height, :width] = image
        mask_tensor[i, :height, :width] = 1.0

        input_ids.append(inputs["input_ids"][0])
        attention_masks.append(inputs["attention_mask"][0])

        processor.image_processor.do_resize = True
        processor.image_processor.do_center_crop = True

        inputs = processor(
            images=[example["image"]],
            text=["This is some text"],
            return_tensors="pt",
            padding=True,
            is_split_into_words=True,
        )

        square_tensor[i, :, :, :] = inputs["pixel_values"][0]

    input_ids = torch.stack(input_ids, dim=0)
    attention_mask = torch.stack(attention_masks, dim=0)
    pixel_values = torch.cat([batched_tensor, mask_tensor.unsqueeze(1)], dim=1)

    return {
        "pixel_values": pixel_values, 
        "pixel_values_square": square_tensor,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

@dataclass
class TrainingRun:
    n_epochs: int
    lr: float
    gamma: float
    accumulation_steps: int
    max_image_size: int
    log_to_wandb: bool
    wandb_project: str
    freeze_vision_model: bool
    checkpoint_path: str
    max_alpha: float
    run_name: str = None
    batch_size: int = 128
    mlp_depth: int = 2
    num_fourier_features: int = 16
    use_l1_loss: bool = False

    def as_dict(self):
        return {
            "n_epochs": self.n_epochs,
            "lr": self.lr,
            "gamma": self.gamma,
            "accumulation_steps": self.accumulation_steps,
            "max_image_size": self.max_image_size,
            "log_to_wandb": self.log_to_wandb,
            "wandb_project": self.wandb_project,
            "freeze_vision_model": self.freeze_vision_model,
            "checkpoint_path": self.checkpoint_path,
            "max_alpha": self.max_alpha,
            "run_name": self.run_name,
            "file": "yolo_min_cs_batched.py",
            "batch_size": self.batch_size,
            "mlp_depth": self.mlp_depth,
            "num_fourier_featurs": self.num_fourier_features,
            "use_l1_loss": self.use_l1_loss,
        }

torch.manual_seed(42)

def center_crop_image(image, max_image_size, min_image_size=48):
    width, height = image.size
    if width < min_image_size or height < min_image_size:
        resize_size = max(width, height, min_image_size)
        transform = Resize(resize_size)
        image = transform(image)
    if width > max_image_size or height > max_image_size:
        crop_size = min(width, height, max_image_size)
        transform = CenterCrop(crop_size)
        image = transform(image)
    return image

def resize_too_small(image, min_image_size: int = 14):
    width, height = image.size
    if width < min_image_size:
        new_width = min_image_size
        new_height = int((min_image_size / width) * height)
    else:
        new_width = width
        new_height = height

    if new_height < min_image_size:
        new_height = min_image_size
        new_width = int((min_image_size / new_height) * new_width)

    transform = Resize((new_height, new_width))
    image = transform(image)
    return image

def resize_too_big(image, max_image_size):
    width, height = image.size
    if width > max_image_size:
        new_width = max_image_size
        new_height = int((max_image_size / width) * height)
    else:
        new_width = width
        new_height = height

    if new_height > max_image_size:
        new_height = max_image_size
        new_width = int((max_image_size / new_height) * new_width)
    
    transform = Resize((new_height, new_width))
    image = transform(image)

    return image

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, grid):
        out = self.model(**inputs) 
        mlp_out = self.model.vision_model.embeddings.mlp(
            grid.flatten(0, 1)
        ).reshape(24, 24, -1).flatten(0, 1)
        return (out, mlp_out)

def train(
    device: str,
    n_epochs: int = 25,
    lr: float = 3.5e-5,
    gamma: float = 750.0,
    accumulation_steps: int = 16,
    max_image_size: int = 896,
    log_to_wandb: bool = True,
    wandb_project: str = "qlip_mlp_training",
    freeze_vision_model: bool = True,
    checkpoint_path: str = ".",
    max_alpha: float = 3.0,
    batch_size: int = 128,
    mlp_depth: int = 2,
    num_fourier_features: int = 16,
    use_l1_loss: bool = False,
):
    print(f"training on {device}")
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )

    if accelerator.is_main_process and log_to_wandb:
        wandb.init(project=wandb_project)
        run_name = wandb.run.name

        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_path, run_name)
        os.makedirs(checkpoint_path, exist_ok=True)

        config = TrainingRun(
            n_epochs=n_epochs,
            lr=lr,
            gamma=gamma,
            accumulation_steps=accumulation_steps,
            max_image_size=max_image_size,
            log_to_wandb=log_to_wandb,
            wandb_project=wandb_project,
            freeze_vision_model=freeze_vision_model,
            checkpoint_path=checkpoint_path,
            max_alpha=max_alpha,
            run_name=run_name,
            batch_size=batch_size,
            mlp_depth=mlp_depth,
            num_fourier_features=num_fourier_features,
            use_l1_loss=use_l1_loss,
        )

        wandb.config.update(config.as_dict())

        with open(os.path.join(checkpoint_path, f"{run_name}.json"), "w") as f:
            json.dump(config.as_dict(), f, indent=4)

    accelerator.init_trackers(
        project_name=wandb_project,
        config=wandb.config,
        init_kwargs={"wandb": {"entity": "my-wandb-team"}}
    )

    processor = AutoProcessor.from_pretrained(
        "openai/clip-vit-large-patch14-336", use_fast=False
    )
    
    model = AutoModelForZeroShotImageClassification.from_pretrained(
        "openai/clip-vit-large-patch14-336"
    )

    model.vision_model = QLIPVisionTransformer.from_CLIPVisionTransformer(model.vision_model)
    
    model = convert_to_qlip_CLIP(
        model,
        mode="random",
        interpolation_mode="mlp",
        flops_file=None,
        alpha=max_alpha,
        alpha_is_max_alpha=True,
        debug=False,
        mlp_weights=None,
        save_pos_enc=True,
        mlp_depth=mlp_depth,
        num_fourier_features=num_fourier_features
    ).to("cuda")        
    
    ds = load_dataset("frgfm/imagenette", "full_size")

    dataloader = DataLoader(
        ds["train"],
        batch_size=batch_size,
        collate_fn=lambda x: collate_fn(x, processor, max_image_size),
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    x_coords = torch.linspace(-1, 1, 24)
    y_coords = torch.linspace(-1, 1, 24)

    grid = torch.stack(torch.meshgrid(y_coords, x_coords), dim=-1).to(device)

    embeddings = model.vision_model.embeddings
    base_embeddings = torch.clone(embeddings.position_embedding.weight).to(device)

    model = ModelWrapper(model)

    # Freeze all the weights in model except for the weights in the mlp
    for param in model.parameters():
        param.requires_grad = False
    for param in model.model.vision_model.embeddings.mlp.parameters():
        param.requires_grad = True

    if not freeze_vision_model:
        for param in model.model.vision_model.patch_embedding.parameters():
            param.requires_grad = True

    optimizer = torch.optim.Adam(model.model.vision_model.embeddings.mlp.parameters(), lr=lr)

    n_gpus = accelerator.state.num_processes
    num_training_steps = (len(dataloader) * n_epochs)
    num_warmup_steps = int(0.01 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )

    model, optimizer, dataloader, processor, scheduler, grid = accelerator.prepare(
        model, optimizer, dataloader, processor, scheduler, grid,
    )

    global_step = 0

    for epoch in range(n_epochs):
        for ii, sample in (pbar := tqdm(enumerate(dataloader))):
            model.module.model.vision_model.embeddings.mode = "bicubic"
            model.module.model.vision_model.embeddings.alpha = 0.0
            model.module.model.vision_model.embeddings.zero_mode = True

            inputs = {
                'pixel_values': sample["pixel_values_square"].to(device),
                'input_ids': sample["input_ids"].to(device),
                'attention_mask': sample["attention_mask"].to(device),
            }

            with torch.no_grad():
                output, _ = model(inputs, grid)

            image_embeds_gt = output.image_embeds.detach()

            model.module.model.vision_model.embeddings.mode = "mlp"
            model.module.model.vision_model.embeddings.alpha = max_alpha
            model.module.model.vision_model.embeddings.zero_mode = False

            # with accelerator.accumulate(model):
            inputs = {
                'pixel_values': sample["pixel_values"].to(device),
                'input_ids': sample["input_ids"].to(device),
                'attention_mask': sample["attention_mask"].to(device),
            }

            output, mlp_out = model(inputs, grid)

            image_embeds_qlip = output.image_embeds

            loss1 = 1 - torch.nn.functional.cosine_similarity(
                image_embeds_qlip, image_embeds_gt, dim=-1
            ).mean()

            loss_q = torch.nn.functional.mse_loss(image_embeds_qlip, image_embeds_gt)

            loss_l2 = torch.nn.functional.mse_loss(mlp_out, base_embeddings[1:])
            loss_l1 = torch.nn.functional.l1_loss(mlp_out, base_embeddings[1:])

            if use_l1_loss:
                loss2 = loss_l1

            loss = loss_q + gamma * loss2
            accelerator.backward(loss)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            grad_tensor_norm = model.module.model.vision_model.embeddings.last_pos_enc.grad.norm(p=2).item()

            pbar.set_description(f"Loss: {loss.item()}.")
            if log_to_wandb and accelerator.is_main_process:
                accelerator.log({
                    "epoch": epoch,
                    "loss": loss.item() * 16,
                    "loss1": loss1.item(),
                    "loss2": loss2.item(),
                    "loss_l1": loss_l1.item(),
                    "loss_l2": loss_l2.item(),
                    "loss_q": loss_q.item(),
                    "grad_norm": grad_tensor_norm,
                    "lr": scheduler.get_last_lr()[0],
                }, step=global_step)

            torch.cuda.empty_cache()

        print(f"Saving checkpoint {epoch + 1}...")
        save_path = os.path.join(checkpoint_path, f"_{epoch}.pt")
        torch.save(model.to('cpu').module.model.vision_model.embeddings.mlp.state_dict(), save_path)
        model = model.to(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=16 * 3.5e-5)
    parser.add_argument("--gamma", type=float, default=1250.0)
    parser.add_argument("--accumulation_steps", type=int, default=256)
    parser.add_argument("--log_to_wandb", type=bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="qlip_mlp_training")
    parser.add_argument("--max_image_size", type=int, default=896)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--freeze_vision_model", type=bool, default=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--max_alpha", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--mlp_depth", type=int, default=2)
    parser.add_argument("--num_fourier_features", type=int, default=32)
    parser.add_argument("--use_l1_loss", default=False, action="store_true")

    args = parser.parse_args()

    train(
        device=args.device,
        n_epochs=args.n_epochs,
        lr=args.lr,
        gamma=args.gamma,
        accumulation_steps=args.accumulation_steps,
        max_image_size=args.max_image_size,
        log_to_wandb=args.log_to_wandb,
        wandb_project=args.wandb_project,
        freeze_vision_model=args.freeze_vision_model,
        checkpoint_path=args.checkpoint_path,
        max_alpha=args.max_alpha,
        batch_size=args.batch_size,
        use_l1_loss=args.use_l1_loss,
        mlp_depth=args.mlp_depth,
        num_fourier_features=args.num_fourier_features,
    )