import argparse
import os
import glob
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig # For 8-bit config
)
from accelerate import Accelerator # To get device placement info

# Function to load and prepare datasets
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

    raw_datasets = load_dataset("json", data_files=files, split="train")
    print(f"Loaded dataset with {len(raw_datasets)} examples.")

    # Basic check for 'text' column
    if "text" not in raw_datasets.column_names:
        raise ValueError("Dataset must contain a 'text' column.")

    # Remove examples with empty text
    raw_datasets = raw_datasets.filter(lambda example: example.get("text") and len(example["text"].strip()) > 0)
    print(f"Dataset size after filtering empty text: {len(raw_datasets)}")
    if len(raw_datasets) == 0:
        raise ValueError("Dataset is empty after filtering.")


    column_names = raw_datasets.column_names
    text_column_name = "text"

    def tokenize_function(examples):
        # Process text field, handling potential None values
        texts_to_tokenize = [str(t) if t is not None else "" for t in examples[text_column_name]]
        # Don't truncate yet, we'll group texts later. Handle potential errors during tokenization.
        try:
            return tokenizer(texts_to_tokenize, truncation=False)
        except Exception as e:
            print(f"Error tokenizing batch: {e}")
            # Return empty dict or handle error appropriately
            # This example skips problematic batches, adjust if needed
            return {k: [] for k in tokenizer("").keys()}


    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    # Filter out empty tokenized results if error handling above creates them
    tokenized_datasets = tokenized_datasets.filter(lambda example: len(example.get('input_ids', [])) > 0)
    print(f"Dataset size after tokenization and filtering empty: {len(tokenized_datasets)}")
    if len(tokenized_datasets) == 0:
        raise ValueError("Dataset is empty after tokenization, check input data and tokenization process.")


    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # Create labels for causal LM
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts into chunks of {block_size}",
    )

    print(f"Processed dataset with {len(lm_datasets)} sequences of length {block_size}.")
    if len(lm_datasets) == 0:
         raise ValueError(f"Dataset is empty after grouping. Check block_size ({block_size}), input text length, and tokenization.")

    return lm_datasets

def main():
    parser = argparse.ArgumentParser(description="Unsupervised Fine-Tuning Script for Gemma")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="google/gemma-2-9b-it", # Recommend starting with this
        help="Hugging Face model ID (e.g., google/gemma-2-9b-it)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the directory containing .jsonl files or a single .jsonl file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gemma-uft-output",
        help="Directory to save the fine-tuned model and logs.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=2048, # Gemma 2 can handle longer context; adjust based on VRAM
        help="Block size for tokenization.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4, # STARTING POINT for H100 - Adjust based on VRAM
        help="Batch size per GPU.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4, # STARTING POINT - Adjust effective batch size (2 GPUs * 4 * 4 = 32)
        help="Number of steps to accumulate gradients before updating weights.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5, # Lower LR often better for larger models/UFT
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1, # Start with 1 epoch for UFT, adjust as needed
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log training metrics every N steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100, # Save checkpoints periodically
        help="Save model checkpoint every N steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save VRAM at the cost of speed.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True, # Default to True for H100
        help="Use BF16 mixed precision (recommended for Ampere GPUs and newer).",
    )
    parser.add_argument(
        "--no_bf16",
        action="store_false",
        dest="bf16",
        help="Disable BF16 mixed precision.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True, # Often needed for newer models like Gemma 2
        help="Trust remote code execution from model authors."
    )


    args = parser.parse_args()

    print(f"Script arguments: {args}")

    # --- Accelerator Initialization ---
    # Handled by Trainer with TrainingArguments

    # --- Load Tokenizer ---
    print(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code
    )
    # Add padding token if missing (common practice)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Tokenizer missing pad token. Using EOS token ({tokenizer.eos_token}) as pad token.")
        else:
            added_token = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print(f"Tokenizer missing pad token and EOS token. Added '[PAD]' as pad token. Added tokens: {added_token}")
            # If we added a token, model resizing is needed later


    # --- Load Dataset ---
    print(f"Loading and preparing dataset from: {args.dataset_path}")
    lm_dataset = load_and_prepare_dataset(args.dataset_path, tokenizer, args.block_size)

    # --- Load Model (with 8-bit Quantization) ---
    print(f"Loading model '{args.model_name_or_path}' with 8-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        # Optional: specify compute dtype if needed, but often okay by default with bf16
        # bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Determine torch dtype based on bf16 flag
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
    print(f"Using compute dtype: {compute_dtype}")

    # device_map="auto" lets accelerate handle device placement
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config,
        device_map="auto", # Automatically distribute across available GPUs
        torch_dtype=compute_dtype, # Use determined dtype
        trust_remote_code=args.trust_remote_code
    )

    # Resize token embeddings if a new pad token was added
    if tokenizer.pad_token_id == len(tokenizer) - 1 and hasattr(tokenizer, 'added_tokens_encoder') and tokenizer.pad_token in tokenizer.added_tokens_encoder:
         print(f"Resizing model token embeddings to {len(tokenizer)} due to added pad token.")
         model.resize_token_embeddings(len(tokenizer))

    # Important settings for training
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")
    else:
        print("Gradient checkpointing disabled.")

    # Disable cache during training for efficiency
    model.config.use_cache = False
    print("Model cache disabled for training.")


    # --- Data Collator ---
    # Standard collator for language modeling. It handles padding.
    # `mlm=False` means causal language modeling (predict next token).
    print("Setting up Data Collator for Language Modeling.")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- Training Arguments ---
    print("Configuring Training Arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bf16=args.bf16, # Use bf16 if specified and available
        fp16=not args.bf16, # Use fp16 if bf16 is not used (redundant if bf16=True)
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps", # Save based on steps
        save_steps=args.save_steps,
        save_total_limit=2, # Keep only the last 2 checkpoints
        seed=args.seed,
        report_to="tensorboard", # Log to tensorboard
        gradient_checkpointing=args.gradient_checkpointing,
        # Use paged optimizer for 8-bit training - generally recommended
        optim="paged_adamw_8bit",
        # Required for gradient checkpointing compatibility with HF Trainer + PEFT/Quantization
        gradient_checkpointing_kwargs={'use_reentrant': False} if args.gradient_checkpointing else None,
        # ddp_find_unused_parameters=False, # Uncomment if you encounter specific DDP errors
    )

    # --- Initialize Trainer ---
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        # eval_dataset=eval_dataset, # Add evaluation dataset if you have one
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- Train ---
    print("Starting training loop...")
    train_result = trainer.train()

    # --- Save ---
    print("Training finished. Saving final model and state...")
    # Saves the quantized model correctly when using Trainer + BitsAndBytesConfig
    trainer.save_model()
    trainer.save_state() # Saves optimizer state etc.
    tokenizer.save_pretrained(args.output_dir) # Save tokenizer

    # --- Log Metrics ---
    metrics = train_result.metrics
    try:
        metrics["train_samples"] = len(lm_dataset)
    except TypeError: # Handle cases where dataset might not have len after processing
         metrics["train_samples"] = -1 # Indicate unknown sample count
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print(f"Model, tokenizer, and metrics saved to {args.output_dir}")

if __name__ == "__main__":
    main()