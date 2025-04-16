#!/bin/bash

# Create a virtual environment
python -m venv gemma_uft_env

# Activate the virtual environment
source gemma_uft_env/bin/activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt

# Verify NeMo installation
python -c "import nemo; print(f'NeMo version: {nemo.__version__}')"

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Number of GPUs: {torch.cuda.device_count()}')"

echo "Setup complete. Activate the environment with: source gemma_uft_env/bin/activate"