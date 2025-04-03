import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk

# Load pre-trained model
model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Apply QLoRA (low-rank adapters)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load tokenized dataset
dataset = load_from_disk("tokenized_data")
train_dataset = dataset

# Define function to add labels to the dataset
def add_labels(examples):
    examples["labels"] = examples["input_ids"].copy()
    return examples

# Add labels to the dataset
train_dataset = train_dataset.map(add_labels, batched=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir="tmforum_finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    save_steps=1000,
    logging_steps=100,
    learning_rate=2e-5,
    fp16=False,
    bf16=True,
    optim="adamw_torch",
    eval_strategy="no",
    eval_steps=500,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    push_to_hub=False
)

# Train model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
