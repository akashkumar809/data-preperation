import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- Configuration ---
base_model_name = "google/gemma-3-12b-it"  # Or your specific base model
uft_adapter_path = "../target/output/gemma-3-12b-it-uft-output"   # <<<--- UPDATE THIS PATH (to the UFT adapter dir)
sft_adapter_path = "../target/output/gemma-3-12b-it-sft-output"   # <<<--- UPDATE THIS PATH (to the SFT adapter dir)

uft_adapter_name = "uft_adapter" # Name for the first adapter
sft_adapter_name = "sft_adapter" # Name for the second adapter

# --- Prompts ---
system_prompt = "You are an expert assistant specializing in TM Forum standards and telecommunications APIs. Provide clear, concise, and accurate information based on TM Forum specifications."
user_question = "Explain the purpose of the TMF620 Product Catalog Management API according to TM Forum standards. What are its key resources?"
# --- --- --- --- --- --- --- --- ---


# Optional: Quantization config (MUST match how BOTH adapters were trained/loaded)
#bnb_config = BitsAndBytesConfig(
#    load_in_8bit=True,
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
#)

# --- 1. Load Base Model ---
print(f"Loading base model: {base_model_name}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
#   quantization_config=bnb_config, # Uncomment if using quantization
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # Important for generation

# --- 2. Load First (UFT) Adapter ---
print(f"Loading first (UFT) adapter from: {uft_adapter_path}")
# This creates the PeftModel, merging the first adapter
model = PeftModel.from_pretrained(
    base_model,
    uft_adapter_path,
    adapter_name=uft_adapter_name, # Assign the name
    is_trainable=False
)
print(f"Adapter '{uft_adapter_name}' loaded.")

# --- 3. Load Second (SFT) Adapter ---
print(f"Loading second (SFT) adapter from: {sft_adapter_path}")
# Use load_adapter on the *existing* PeftModel
model.load_adapter(
    sft_adapter_path,
    adapter_name=sft_adapter_name, # Assign the name
    is_trainable=False
)
print(f"Adapter '{sft_adapter_name}' loaded.")

# --- 4. Set BOTH Adapters Active ---
print(f"Activating adapters: {sft_adapter_name}")
# Pass a list of names to set_adapter to activate them simultaneously
model.set_adapter(sft_adapter_name)
# model.set_adapter([uft_adapter_name, sft_adapter_name])
# Verify active adapters (optional)
# print(f"Currently active adapters: {model.active_adapters}")

# --- 5. Perform Inference ---
# IMPORTANT: Use the prompt format expected by the FINAL (SFT) stage
#            since that's the task you want the combined model to perform.
messages = [
    {"role": "user", "content": f"{system_prompt}\n\n{user_question}"}, # Combine system+user for Gemma's typical user turn
    # Alternatively, if the base model/tokenizer *explicitly* supports a 'system' role in its template:
    # {"role": "system", "content": system_prompt},
    # {"role": "user", "content": user_question},
]

prompt_string = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, # We want the string, not token IDs yet
    add_generation_prompt=True 
)

# prompt = "### Instruction:\nPlease explain the concept of supervised fine-tuning.\n\n### Response:\n" # Example SFT prompt

print("\nGenerating response with combined adapters...")
inputs = tokenizer(prompt_string, return_tensors="pt").to(model.device)

# Generation settings
outputs = model.generate(
    **inputs,
    max_new_tokens=500, # Adjust as needed
    do_sample=True,
    temperature=0.7,
    # top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id # Ensure pad token is set if needed
)

# Decode the output
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n--- Model Response ---")
print(response)
print("----------------------")
