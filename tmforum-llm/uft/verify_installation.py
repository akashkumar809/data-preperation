import torch
import transformers
import huggingface_hub
import nemo

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
print(f"GPU names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
print(f"NeMo version: {nemo.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Hugging Face Hub version: {huggingface_hub.__version__}")