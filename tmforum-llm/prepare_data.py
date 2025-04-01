from datasets import Dataset
import glob
import os

# Define input/output paths
DATA_PATH = "datasets"
OUTPUT_DATASET_PATH = "processed-datasets"

# Read all text files
text_files = glob.glob(os.path.join(DATA_PATH, "*.txt"))

data = []
for file in text_files:
    with open(file, "r", encoding="utf-8") as f:
        data.append({"text": f.read()})

# Convert to Hugging Face dataset and save
dataset = Dataset.from_list(data)
dataset.save_to_disk(OUTPUT_DATASET_PATH)

print(f"✅ Dataset saved at {OUTPUT_DATASET_PATH}")
