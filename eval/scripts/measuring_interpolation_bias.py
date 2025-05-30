"""
Measure
1. Interpolation bias crop
2. Interpolation bias native
3. 
"""
import os
import json
import random
import time
import torch

from datasets import load_dataset
from PIL import Image
from transformers import (
    AutoModelForZeroShotImageClassification, 
    AutoProcessor
)
from tqdm.auto import tqdm

from qlip.clip_models import convert_to_qlip_CLIP
from qlip.utils import seed_everything

BENCHMARK_FOLDER = None # Read the VSTAR_BENCHMARK_FOLDER environment variable

BENCHMARK_FOLDER = os.getenv("VSTAR_BENCHMARK_FOLDER", None)
if BENCHMARK_FOLDER is None:
    raise ValueError("Please set the VSTAR_BENCHMARK_FOLDER environment variable to the path of the V* Benchmark folder.")

def run(
    n_patches: int,
    mlp_weights: str,
    interpolation_mode: str,
    crop: bool,
):
    seed_everything(42)
    SAMPLES = 30

    processor = AutoProcessor.from_pretrained(
        "openai/clip-vit-large-patch14-336", use_fast=False
    )
    model = AutoModelForZeroShotImageClassification.from_pretrained(
        "openai/clip-vit-large-patch14-336"
    )
    
    model = convert_to_qlip_CLIP(
        model,
        mode="random", # We use random mode so that we never split
        interpolation_mode=interpolation_mode,
        flops_file=None,
        alpha=0.0,
        alpha_is_max_alpha=False,
        debug=False,
        mlp_weights=mlp_weights,
        save_pos_enc=True,
    ).to("cuda")

    opt = torch.optim.Adam(model.parameters(), lr=1e-5)

    dataset = load_dataset("craigwu/vstar_bench")["test"]

    indexes = random.sample(range(0, len(dataset)), SAMPLES)

    images = []
    labels = []

    for i in indexes:
        img_path = os.path.join(BENCHMARK_FOLDER, dataset[i]["image"])
        images.append(Image.open(img_path).convert("RGB"))
        labels.append(dataset[i]["text"])

    avg_cos_sim = 0
    avg_grad_norm = 0

    opt.zero_grad()

    for i, (img, txt) in tqdm(enumerate(zip(images, labels)), total=len(images)):
        processor.image_processor.do_center_crop = True
        processor.image_processor.crop_size = (336, 336)
         # This ensures that we use the origina CLIP embeddings, since the 
         # bicubic interpolation of the original embeddings are themselves
        model.vision_model.embeddings.mode = "bicubic"

        with torch.no_grad():
            inputs = processor(
                images=[img],
                text=[txt],
                return_tensors="pt",
                padding=True,
                is_split_into_words=True,
            ).to("cuda")

            output = model(**inputs)

        # Returns the pooled output
        image_embeds_gt = output.image_embeds.detach()

        processor.image_processor.do_center_crop = crop
        processor.image_processor.do_resize = True
        processor.image_processor.size = {"shortest_edge": 224 + 28 * n_patches}
        processor.image_processor.crop_size = (224 + 28 * n_patches, 224 + 28 * n_patches)
        model.vision_model.embeddings.mode = "mlp"

        opt.zero_grad()

        inputs = processor(
            images=[img],
            text=[txt],
            return_tensors="pt",
            padding=True,
            is_split_into_words=True,
        ).to("cuda")

        output = model(**inputs)

        image_embeds_anyres = output.image_embeds

        cos_sim = 1 - torch.nn.functional.cosine_similarity(
            image_embeds_gt,
            image_embeds_anyres,
            dim=-1,
        )

        cos_sim.backward()
        pos_enc_anyres = model.vision_model.embeddings.last_pos_enc
        pos_emb_grad = pos_enc_anyres.grad
        pos_emb_grad_norm = pos_emb_grad.cpu().norm(p=2).item()

        avg_grad_norm += pos_emb_grad_norm
        avg_cos_sim += cos_sim.item()      

    avg_cos_sim = avg_cos_sim / len(images)
    avg_grad_norm = avg_grad_norm / len(images)

    return avg_cos_sim, avg_grad_norm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp_checkpoint", type=str, required=True)
    parser.add_argument("--interpolation_mode", type=str, default="mlp", choices=["bicubic", "mlp"])
    parser.add_argument("--crop", action="store_true")
    args = parser.parse_args()

    data = {}

    for i in range(22):
        img_size = 224 + 28 * i
        avg_cos_sim, avg_grad_norm = run(
            n_patches=i,
            mlp_weights=args.mlp_checkpoint,
            interpolation_mode=args.interpolation_mode,
            crop=args.crop,
        )
        data[img_size] = {
            "avg_cos_sim": 1 - avg_cos_sim,
            "avg_grad_norm": avg_grad_norm,
        }
        print(f"{img_size}: {1 - avg_cos_sim}, {avg_grad_norm}")

        os.makedirs("./eval/data/interpolation_bias", exist_ok=True)

        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(f"./eval/data/interpolation_bias/{now}_intmode_{args.interpolation_mode}_crop_{args.crop}.json", "w") as f:
            json.dump(data, f, indent=4)