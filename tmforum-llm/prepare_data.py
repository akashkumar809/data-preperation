from datasets import Dataset
import glob
import os
import json
import pandas as pd

DATA_PATH = "target/jsonls"
OUTPUT_DATASET_PATH = "processed-datasets"


def load_jsonls():
    json_files = glob.glob(os.path.join(DATA_PATH, "*.jsonl"))
    data = []

    for file in json_files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))

    return data


def create_hf_daaset() :
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_DATASET_PATH)
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


    data = load_jsonls()
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(OUTPUT_DATASET_PATH)

    print(f"✅ Dataset saved at {OUTPUT_DATASET_PATH}")


if __name__ == "__main__":
    create_hf_daaset()