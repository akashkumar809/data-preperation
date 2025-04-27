python3.9 -m venv gemma_sft_hf_env
. gemma_sft_hf_env/bin/activate

# Install PyTorch for your CUDA version first (e.g., cu121 or cu126)
# cu126 was having some os library version mismatch eventhough the cuda version was 12.6, I used cu121 and din't gave any error
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the rest
pip install -r uft_sft_requirements.txt

python run_simple_sft_gemma_8bit_peft.py