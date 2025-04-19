import os
import glob
from pathlib import Path
import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig, # For 8-bit config
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
# PEFT imports
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# --- 1. Hardcoded Parameters ---

# -- Paths --
# !! IMPORTANT: Set these paths correctly for your environment !!
DATASET_PATH = "../target/jsonls"  # Directory containing all your .jsonl files
OUTPUT_DIR = "../target/output/gemma-3-12b-it-uft-output"      # Where the adapter checkpoint will be saved

# -- Model & Tokenizer --
MODEL_NAME = "google/gemma-3-12b-it" # Or "google/gemma-7b-it" etc.
TRUST_REMOTE_CODE = True

# -- Quantization --
QUANTIZATION_BITS = 8 # Using 8-bit as requested

# -- PEFT/LoRA Config --
LORA_R = 16          # LoRA rank (dimension)
LORA_ALPHA = 32      # LoRA scaling factor
LORA_DROPOUT = 0.05  # Dropout probability for LoRA layers
# Target modules for Gemma 2 (verify if needed by printing model architecture)
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj",
    "o_proj", "gate_proj", "up_proj", "down_proj"
]

# -- Training Params --
BLOCK_SIZE = 2048                # Sequence length for training (adjust based on VRAM)
PER_DEVICE_TRAIN_BATCH_SIZE = 2  # Adjust based on VRAM (8-bit uses more than 4-bit)
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = N_GPUS * batch_size * grad_accum
LEARNING_RATE = 1e-4             # Common starting point for LoRA
NUM_TRAIN_EPOCHS = 1             # Number of passes through the data
WEIGHT_DECAY = 0.01
BF16 = True                      # Use bfloat16 for H100 performance
GRADIENT_CHECKPOINTING = True    # Recommended to save memory
SEED = 42

# -- Logging/Saving --
LOGGING_STEPS = 10
SAVE_STEPS = 100                 # Save adapter checkpoint every N steps
SAVE_TOTAL_LIMIT = 2             # Keep only the last N checkpoints


# --- 2. Data Loading and Preparation Function ---
# (This function remains largely the same, handles loading/tokenizing/grouping)
def load_and_prepare_dataset(data_path, tokenizer, block_size):
    """Loads JSONL dataset, tokenizes, and groups texts."""
    if os.path.isdir(data_path):
        files = glob.glob(os.path.join(data_path, "*.jsonl"))
        print(f"Found {len(files)} JSONL files in {data_path}")
    elif os.path.isfile(data_path):
        files = [data_path]
        print(f"Using single JSONL file: {data_path}")
    else:
        raise ValueError(f"Invalid data_path: {data_path}. Must be a directory or a .jsonl file.")

    if not files:
        raise ValueError(f"No .jsonl files found in {data_path}")

    # Specify cache_dir to potentially avoid issues on certain systems
    # Using a relative path here, adjust if needed
    cache_dir = Path("./.cache/huggingface/datasets")
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using dataset cache dir: {cache_dir}")

    raw_datasets = load_dataset("json", data_files=files, split="train", cache_dir=str(cache_dir))
    print(f"Loaded dataset with {len(raw_datasets)} examples.")

    if "text" not in raw_datasets.column_names:
        raise ValueError("Dataset must contain a 'text' column.")

    raw_datasets = raw_datasets.filter(lambda example: example.get("text") and len(example["text"].strip()) > 0)
    print(f"Dataset size after filtering empty text: {len(raw_datasets)}")
    if len(raw_datasets) == 0: raise ValueError("Dataset is empty after filtering.")

    column_names = raw_datasets.column_names
    text_column_name = "text"

    def tokenize_function(examples):
        texts_to_tokenize = [str(t) if t is not None else "" for t in examples[text_column_name]]
        try:
            return tokenizer(texts_to_tokenize, truncation=False)
        except Exception as e:
            print(f"Error tokenizing batch: {e}. Skipping batch.")
            return {k: [] for k in tokenizer("").keys()} # Return empty structure

    tokenized_datasets = raw_datasets.map(
        tokenize_function, batched=True, remove_columns=column_names, desc="Running tokenizer"
    )
    tokenized_datasets = tokenized_datasets.filter(lambda example: len(example.get('input_ids', [])) > 0)
    print(f"Dataset size after tokenization filtering: {len(tokenized_datasets)}")
    if len(tokenized_datasets) == 0: raise ValueError("Dataset empty after tokenization.")

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts, batched=True, desc=f"Grouping texts into chunks of {block_size}"
    )
    print(f"Processed dataset with {len(lm_datasets)} sequences of length {block_size}.")
    if len(lm_datasets) == 0: raise ValueError("Dataset empty after grouping.")

    return lm_datasets

# --- 3. Main Script Logic ---

def main():
    print("--- Starting 8-bit PEFT/LoRA Training ---")
    print(f"Using model: {MODEL_NAME}")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")

    # --- Load Tokenizer ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE
    )
    # Add padding token if missing
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Using EOS token ({tokenizer.eos_token}) as pad token.")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("Added '[PAD]' as pad token.")

    # --- Load Dataset ---
    print("Loading and preparing dataset...")
    lm_dataset = load_and_prepare_dataset(DATASET_PATH, tokenizer, BLOCK_SIZE)

    # --- Configure 8-bit Quantization ---
    print(f"Configuring {QUANTIZATION_BITS}-bit quantization...")
    compute_dtype = torch.bfloat16 if BF16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        # Optional: Specify compute dtype for mixed precision matrix multiplications if needed
        # Often handled by Trainer args, but can be explicit:
        # bnb_4bit_compute_dtype=compute_dtype # Note: This arg name might be misleading for 8bit, check docs if issues
    )

    # --- Load Quantized Model ---
    print(f"Loading base model '{MODEL_NAME}' quantized to {QUANTIZATION_BITS}-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto", # Let accelerate place layers automatically
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=compute_dtype # Set compute dtype for non-quantized parts/operations
    )
    print("Base model loaded.")

    # --- Prepare for PEFT + Gradient Checkpointing ---
    # Important for training quantized models with PEFT and GC
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=GRADIENT_CHECKPOINTING)
    if GRADIENT_CHECKPOINTING:
        print("Model prepared for k-bit training with gradient checkpointing enabled.")
    else:
        print("Model prepared for k-bit training (gradient checkpointing disabled).")


    # --- Define LoRA Configuration ---
    print(f"Applying LoRA to modules: {LORA_TARGET_MODULES}")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # --- Apply PEFT to Model ---
    print("Wrapping model with PEFT LoraConfig...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() # Show how few parameters are trained

    # Resize embeddings if pad token was added (needs to happen *after* get_peft_model?)
    # Let's try resizing *before* get_peft_model, often safer.
    # if tokenizer.pad_token_id == len(tokenizer) - 1 and hasattr(tokenizer, 'added_tokens_encoder') and tokenizer.pad_token in tokenizer.added_tokens_encoder:
    #     print(f"Resizing model token embeddings to {len(tokenizer)}.")
    #     model.resize_token_embeddings(len(tokenizer)) # Resize base model before PEFT wrap

    # --- Configure Training Arguments ---
    print("Configuring Training Arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        bf16=BF16,
        fp16=not BF16, # Use fp16 only if bf16 is False
        logging_dir=os.path.join(OUTPUT_DIR, 'logs'),
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        seed=SEED,
        report_to="tensorboard",
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        # Required for GC compatibility when using Trainer
        gradient_checkpointing_kwargs={'use_reentrant': False} if GRADIENT_CHECKPOINTING else None,
        # Optimizer choice: Default AdamW works. Paged optimizers are mainly for 4-bit.
        # optim="paged_adamw_8bit", # Usually not needed/recommended for 8-bit
    )

    # --- Initialize Trainer ---
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,        # The PEFT model
        args=training_args,
        train_dataset=lm_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    model.config.use_cache = False # Disable cache for training

    # --- Train ---
    print("Starting PEFT training...")
    train_result = trainer.train()

    # --- Save Adapter ---
    print("Training finished. Saving PEFT adapter...")
    # trainer.save_model() saves only the adapter weights and config
    trainer.save_model(OUTPUT_DIR)

    # --- Log Metrics ---
    print("Logging and saving metrics...")
    metrics = train_result.metrics
    try:
        metrics["train_samples"] = len(lm_dataset)
    except TypeError:
        metrics["train_samples"] = -1
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print("--- Training Complete ---")
    print(f"PEFT adapter saved to: {OUTPUT_DIR}")

# --- Run Main ---
if __name__ == "__main__":
    # Before running, make sure you have installed dependencies:
    # pip install torch torchvision torchaudio --index-url <YOUR_CUDA_URL>
    # pip install transformers accelerate datasets bitsandbytes peft tensorboard scipy packaging huggingface_hub
    # And configured huggingface login:
    # huggingface-cli login
    # And potentially configured accelerate (though not strictly needed for device_map="auto" on single machine):
    # accelerate config

    # Set the dataset path correctly!
    if DATASET_PATH == "/path/to/your/jsonl/data_dir":
       print("ERROR: Please update the global variable 'DATASET_PATH' in the script before running.")
    else:
       main()
