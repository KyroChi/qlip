#!/bin/bash
# I've set the default parameters to match what was used in the paper.
# Note that we are using four GPUs by defualt, numbered 0, 1, 2, and 3.
# You will probably need to change this depending on your hardware setup.

NOW=$(date +"%m-%d-%Y_%H:%M:%S")
EXPERIMENT_NAME=qlip_mlp_training
CHECKPOINT_BASE=./interpolation_mlp_checkpoints

# create the checkpoint directory if it does not exist
mkdir -p ${CHECKPOINT_BASE}
echo Checkpoint directory: ${CHECKPOINT_BASE}

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --main_process_port 29501 \
    --multi_gpu \
    --num_machines 1 \
    --num_processes 4 \
    training/train_interpolation_mlp.py \
    --checkpoint_path ${CHECKPOINT_BASE}/${EXPERIMENT_NAME}_${NOW} \
    --wandb_project ${EXPERIMENT_NAME} \
    --max_alpha 0.1 \
    --n_epochs 100 \
    --gamma 1.0 \
    --max_image_size 560 \
    --batch_size 14 \
    --mlp_depth 4 \
    --num_fourier_features 48 \
    --accumulation_steps 1 \
    --lr 7.5e-5 \
    --use_l1_loss
