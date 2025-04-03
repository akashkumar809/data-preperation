import os
from transformers import AutoTokenizer
from datasets import load_from_disk

PROCESSED_DATA_PATH = "processed-datasets"
OUTPUT_PATH = "tokenized_data"

def tokenize_dataset():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_PATH)
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


    MODEL_NAME = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenizer.pad_token = tokenizer.eos_token
    # Load dataset
    dataset = load_from_disk(PROCESSED_DATA_PATH)

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=2048)

    # Tokenize dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.save_to_disk(OUTPUT_PATH)

    print("✅ Tokenization complete. Data saved at tokenized_data")

if __name__ == "__main__":
    tokenize_dataset()