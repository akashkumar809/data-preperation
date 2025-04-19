python3.9 -m venv gemma_uft_hf_env
. gemma_uft_hf_env/bin/activate

# Install PyTorch for your CUDA version first (e.g., cu121 or cu126)
# cu126 was having some os library version mismatch eventhough the cuda version was 12.6, I used cu121 and din't gave any error
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the rest
pip install transformers accelerate datasets bitsandbytes peft tensorboard scipy packaging huggingface_hub

# login to huggingface_cli, needed once
# huggingface-cli login

# run the script
python run_simple_uft_gemma_8bit_peft.py
