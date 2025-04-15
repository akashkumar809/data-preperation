#!/bin/bash

# Run Gemma-3-12b-it Unsupervised Fine-Tuning with 8-bit quantization
# This script assumes you have set up the NVIDIA NeMo environment correctly

# Set your Hugging Face token for accessing Gemma-3 models
# You can get a token from https://huggingface.co/settings/tokens
export HF_TOKEN="your_huggingface_token_here"  # Replace with your actual token

# Set environment variables for NVIDIA NeMo
export CUDA_VISIBLE_DEVICES=0,1  # Adjust based on your GPU setup
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Path to your Python script and config file
SCRIPT_PATH="./gemma_uft_nemo.py"
CONFIG_PATH="./gemma_uft_config.yaml"

# Run the training script using NeMo's launcher
python $SCRIPT_PATH --config-file $CONFIG_PATH