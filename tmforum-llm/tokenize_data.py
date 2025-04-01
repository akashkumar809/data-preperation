from transformers import AutoTokenizer
from datasets import load_from_disk

# Choose a model tokenizer
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token
# Load dataset
dataset = load_from_disk("processed-datasets")

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=2048)

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.save_to_disk("tokenized_data")

print("✅ Tokenization complete. Data saved at tokenized_data")
