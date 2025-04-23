import os
import glob
from pathlib import Path
import torch
import json # To load the Q&A JSON

from datasets import load_dataset, Dataset # To load JSON and create Dataset object
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    # Trainer, # We'll use SFTTrainer instead
)
# PEFT imports
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, TaskType
# TRL import for SFT
from trl import SFTTrainer

# --- 1. Hardcoded Parameters ---

# -- Paths --
# !! IMPORTANT: Set these paths correctly for your environment !!
SFT_DATASET_PATH = "../generate-answers/optimized_question_answer_pairs.json" # Your Q&A JSON file path
BASE_MODEL_ADAPTER_PATH = "../target/output/gemma-3-12b-it-uft-output" # Path to your PREVIOUS UFT adapter output
OUTPUT_DIR = "../target/output/gemma-3-12b-it-sft-output"             # Where the NEW SFT adapter checkpoint will be saved

# -- Model & Tokenizer --
MODEL_NAME = "google/gemma-3-12b-it" # Base model used for UFT
TRUST_REMOTE_CODE = True

# -- Quantization --
QUANTIZATION_BITS = 8 # Must match the base model adapter

# -- PEFT/LoRA Config (for the NEW SFT adapter) --
# You can adjust these if desired for the SFT layer
SFT_LORA_R = 8           # Rank for SFT adapter (can be same or different from UFT)
SFT_LORA_ALPHA = 16      # Alpha for SFT adapter
SFT_LORA_DROPOUT = 0.05
# Target modules should generally match the UFT adapter's targets
SFT_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj",
    "o_proj", "gate_proj", "up_proj", "down_proj"
]

# -- Training Params --
BLOCK_SIZE = 2048                # Max sequence length (SFTTrainer handles packing/padding)
PER_DEVICE_TRAIN_BATCH_SIZE = 2  # Adjust based on VRAM (8-bit + 2 adapters)
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 5e-5             # SFT LR might be slightly different, adjust as needed (e.g., 1e-4, 5e-5)
NUM_TRAIN_EPOCHS = 1             # Start with 1-3 epochs for SFT
WEIGHT_DECAY = 0.01
BF16 = True
GRADIENT_CHECKPOINTING = True
SEED = 42
MAX_SEQ_LENGTH = BLOCK_SIZE      # Parameter for SFTTrainer

# -- Logging/Saving --
LOGGING_STEPS = 10
SAVE_STEPS = 50                  # Save SFT adapter checkpoint more frequently if needed
SAVE_TOTAL_LIMIT = 2

# --- 2. Dataset Loading and Formatting ---

# Gemma Instruct Chat Template Structure:
# <start_of_turn>user
# {question}<end_of_turn>
# <start_of_turn>model
# {answer}<end_of_turn>

def format_instruction(sample):
    """Applies the Gemma chat template to a Question/Answer pair."""
    return f"<start_of_turn>user\n{sample['question']}<end_of_turn>\n<start_of_turn>model\n{sample['answer']}<end_of_turn>"

# --- 3. Main Script Logic ---

def main():
    print("--- Starting 8-bit SFT (on existing Adapter) ---")
    print(f"Base Model: {MODEL_NAME}")
    print(f"Loading UFT Adapter from: {BASE_MODEL_ADAPTER_PATH}")
    print(f"SFT Dataset: {SFT_DATASET_PATH}")
    print(f"New SFT Adapter Output Dir: {OUTPUT_DIR}")

    # --- Load Tokenizer ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=TRUST_REMOTE_CODE)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Using EOS token ({tokenizer.eos_token}) as pad token.")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("Added '[PAD]' as pad token.")
            # Important: Resize needed if token added BEFORE model loading
            # resize_needed = True
    # else:
    # resize_needed = False
    tokenizer.padding_side = 'right' # SFT requires proper padding side

    # --- Load Dataset ---
    print("Loading SFT dataset...")
    with open(SFT_DATASET_PATH, 'r') as f:
        sft_data = json.load(f)
    # Convert list of dicts to Hugging Face Dataset object
    sft_dataset = Dataset.from_list(sft_data)
    print(f"Loaded {len(sft_dataset)} Q&A pairs.")

    # Optional: Split dataset if needed (e.g., 90% train, 10% eval)
    sft_dataset = sft_dataset.train_test_split(test_size=0.05)
    train_dataset = sft_dataset["train"]
    eval_dataset = sft_dataset["test"]
    # train_dataset = sft_dataset # Using all data for training in this example


    # --- Configure 8-bit Quantization ---
    print(f"Configuring {QUANTIZATION_BITS}-bit quantization...")
    compute_dtype = torch.bfloat16 if BF16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    # --- Load Base Model (Quantized) ---
    print(f"Loading base model '{MODEL_NAME}' quantized to {QUANTIZATION_BITS}-bit...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto", # Distribute across GPUs
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=compute_dtype
    )
    print("Base model loaded.")

    # --- Load the FIRST (UFT) Adapter onto the base model ---
    print(f"Loading existing UFT adapter from {BASE_MODEL_ADAPTER_PATH}...")
    # This loads the adapter and merges it (logically) - by default it's not trainable
    model = PeftModel.from_pretrained(base_model, BASE_MODEL_ADAPTER_PATH, is_trainable=False)
    print("UFT adapter loaded.")

    # --- Prepare model for k-bit training (needed before adding the *next* adapter) ---
    # Ensures compatibility with PEFT + quantization + GC
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=GRADIENT_CHECKPOINTING)
    if GRADIENT_CHECKPOINTING:
        print("Model prepared for k-bit training with gradient checkpointing enabled.")
    else:
        print("Model prepared for k-bit training (gradient checkpointing disabled).")

    # --- Define NEW LoRA Configuration (for SFT) ---
    print(f"Defining NEW SFT LoRA config (R={SFT_LORA_R}, Alpha={SFT_LORA_ALPHA})")
    sft_lora_config = LoraConfig(
        r=SFT_LORA_R,
        lora_alpha=SFT_LORA_ALPHA,
        target_modules=SFT_LORA_TARGET_MODULES,
        lora_dropout=SFT_LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # --- Apply the NEW SFT adapter ---
    # This adds the second adapter layer, which WILL be trainable
    print("Applying NEW SFT adapter config...")
    model = get_peft_model(model, sft_lora_config)
    print("Trainable SFT adapter added.")
    model.print_trainable_parameters() # Verify only the SFT adapter params are trainable

    # --- Configure Training Arguments ---
    print("Configuring Training Arguments for SFT...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        bf16=BF16,
        fp16=not BF16,
        logging_dir=os.path.join(OUTPUT_DIR, 'logs'),
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        seed=SEED,
        report_to="tensorboard",
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={'use_reentrant': False} if GRADIENT_CHECKPOINTING else None,
        evaluation_strategy="steps",    # Evaluate every `eval_steps`
        eval_steps=50,                  # How often to evaluate (adjust based on dataset size/save_steps)
        # ddp_find_unused_parameters=False, # Uncomment if needed for multi-GPU issues
    )

    # --- Initialize SFTTrainer ---
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,                      # Pass the PEFT model (with UFT + SFT adapters)
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,    # Optional: Pass eval dataset
        tokenizer=tokenizer,
        peft_config=sft_lora_config,      # Pass the SFT config (ensures correct adapter is trained)
        formatting_func=format_instruction, # Function to apply chat template
        max_seq_length=MAX_SEQ_LENGTH,    # Max sequence length
        # packing=True,                   # Optional: Pack short sequences together (can speed up)
    )
    model.config.use_cache = False # Disable cache for training

    # --- Train ---
    print("Starting SFT training...")
    train_result = trainer.train()

    # --- Save SFT Adapter ---
    print("Training finished. Saving SFT adapter...")
    # This saves only the trained SFT adapter weights
    trainer.save_model(OUTPUT_DIR)

    # --- Log Metrics ---
    print("Logging and saving metrics...")
    metrics = train_result.metrics
    try:
        num_train_samples = len(train_dataset)
        metrics["train_samples"] = num_train_samples
    except TypeError:
        metrics["train_samples"] = -1
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print("--- SFT Training Complete ---")
    print(f"SFT adapter saved to: {OUTPUT_DIR}")

# --- Run Main ---
if __name__ == "__main__":
    # Before running:
    # 1. Ensure dependencies are installed (`trl` is needed)
    # 2. Ensure you are logged into huggingface (`huggingface-cli login`)
    # 3. !! CRITICAL: Update SFT_DATASET_PATH and BASE_MODEL_ADAPTER_PATH !!

    if not os.path.exists(SFT_DATASET_PATH):
        print(f"ERROR: SFT Dataset not found at '{SFT_DATASET_PATH}'. Please update the path.")
    elif not os.path.exists(os.path.join(BASE_MODEL_ADAPTER_PATH, "adapter_config.json")):
        print(f"ERROR: Base UFT adapter not found at '{BASE_MODEL_ADAPTER_PATH}'. Please update the path.")
    else:
        main()
