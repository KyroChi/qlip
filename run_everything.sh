#!/bin/bash
# Running this script will fully reproduce all of the results in our QLIP paper.
# We train the interpolation MLP, use the weights to measure the improvement
# in mesoscopic bias and interpolation bias, evaluation QLIP on the vision benchmarks,
# and finally, generate all of the figures and tables that were included in the paper.

# Train the MLP. The training script has as its default parameters
# the hyperparameters which we used in our paper.
# ./training/train_interpolation_mlp.sh

# Look in the mlp_runs folder for the latest checkpoint and set the 
# environment variable
LATEST_CHECKPOINT_FOLDER=$(pwd)/interpolation_mlp_checkpoints/$( ls -t $(pwd)/interpolation_mlp_checkpoints | head -n 1 )
echo "Using MLP checkpoint folder: ${LATEST_CHECKPOINT_FOLDER}"
LATEST_MLP_CHECKPOINT=${LATEST_CHECKPOINT_FOLDER}/$(ls -t "${LATEST_CHECKPOINT_FOLDER}" | head -n 1)
echo "Using MLP checkpoint: ${LATEST_MLP_CHECKPOINT}.json"

# Download V* and set the VSTAR_BENCHMARK_FOLDER environment variable
# HuggingFace Dataset doesn't seem to include the actual data files by default?

# Check that git-lfs is installed and install it if not
git lfs install

# Check that the vstar_folder exists, if not, clone the dataset
if [ ! -d "$(pwd)/vstar_bench" ]; then
    echo "Cloning V*STAR benchmark dataset..."
    git clone https://huggingface.co/datasets/craigwu/vstar_bench
    echo "V*STAR benchmark dataset cloned to $(pwd)/vstar_bench"
fi

export VSTAR_BENCHMARK_FOLDER=$(pwd)/vstar_bench

# Gather the data for measuring the mesoscopic bias
INTERPOLATION_BIAS_RESULTS_FOLDER=$(pwd)/eval/data/interpolation_bias

python eval/scripts/measuring_interpolation_bias.py \
    --mlp_checkpoint ${LATEST_MLP_CHECKPOINT} \
    --interpolation_mode bicubic
BICUBIC_NATIVE_INTERPOLATION_BIAS=${INTERPOLATION_BIAS_RESULTS_FOLDER}/$(ls -t ${INTERPOLATION_BIAS_RESULTS_FOLDER} | head -n 1)

python eval/scripts/measuring_interpolation_bias.py \
    --mlp_checkpoint ${LATEST_MLP_CHECKPOINT} \
    --interpolation_mode mlp
MLP_NATIVE_INTERPOLATION_BIAS=${INTERPOLATION_BIAS_RESULTS_FOLDER}/$(ls -t ${INTERPOLATION_BIAS_RESULTS_FOLDER} | head -n 1)

python eval/scripts/measuring_interpolation_bias.py \
    --mlp_checkpoint ${LATEST_MLP_CHECKPOINT} \
    --interpolation_mode bicubic \
    --crop 
BICUBIC_CROP_INTERPOLATION_BIAS=${INTERPOLATION_BIAS_RESULTS_FOLDER}/$(ls -t ${INTERPOLATION_BIAS_RESULTS_FOLDER} | head -n 1)

python eval/scripts/measuring_interpolation_bias.py \
    --mlp_checkpoint ${LATEST_MLP_CHECKPOINT} \
    --interpolation_mode mlp \
    --crop 
MLP_CROP_INTERPOLATION_BIAS=${INTERPOLATION_BIAS_RESULTS_FOLDER}/$(ls -t ${INTERPOLATION_BIAS_RESULTS_FOLDER} | head -n 1)

# Plot the results of the interpolation bias measurements

# Run all of the evaluations. This takes a long time.

# Find the latest evaluations folder and set the environment variable

# Generate the figures and tables from the evaluation results

# Random other things: 
# - Quadtree patchification of Junie