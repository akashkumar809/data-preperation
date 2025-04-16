# Ensure you are in the directory containing run_uft_gemma.py and uft_hf_requirements.txt

accelerate launch --num_processes=2 run_uft_gemma.py \
    --model_name_or_path "google/gemma-3-12b-it" \
    --dataset_path "../target/jsonls" `# IMPORTANT: Replace with actual path` \
    --output_dir "../target/output/gemma-3-12b-it-uft-output" `# Adjust if needed` \
    --block_size 2048 `# Can adjust, affects VRAM` \
    --per_device_train_batch_size 4 `# START HERE - Adjust based on H100 VRAM usage` \
    --gradient_accumulation_steps 4 `# START HERE - Adjust effective batch size` \
    --learning_rate 1e-5 `# Good starting point for UFT` \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 100 `# Save checkpoints every 100 steps` \
    --bf16 `# Use BrainFloat16 for H100` \
    --trust_remote_code `# Often needed for Gemma models` \
    # --gradient_checkpointing # UNCOMMENT this line if you run into Out-Of-Memory errors