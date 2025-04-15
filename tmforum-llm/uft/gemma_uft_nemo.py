#!/usr/bin/env python3

import os
import json
import glob
import argparse
import logging
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
import nemo
from nemo.utils import logging as nemo_logging
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
import huggingface_hub

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Unsupervised Fine-Tuning for Gemma-3-12b-it with 8-bit quantization")

    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing JSONL files for training')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save model checkpoints and logs')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size per GPU for training')
    parser.add_argument('--grad_accum_steps', type=int, default=8,
                        help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--max_seq_length', type=int, default=2048,
                        help='Maximum sequence length')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Number of warmup steps')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='Evaluate every N steps')

    # Model parameters
    parser.add_argument('--model_name_or_path', type=str,
                        default='google/gemma-3-12b-it',
                        help='Model path or name on HuggingFace Hub')
    parser.add_argument('--hf_token', type=str,
                        default=None,
                        help='Hugging Face token for downloading gated models')
    parser.add_argument('--model_cache_dir', type=str,
                        default=None,
                        help='Directory to cache downloaded models')
    parser.add_argument('--tp_size', type=int, default=4,
                        help='Tensor parallelism size')
    parser.add_argument('--pp_size', type=int, default=1,
                        help='Pipeline parallelism size')

    # Quantization parameters
    parser.add_argument('--quantization', type=str, default='int8',
                        choices=['int8', 'none'],
                        help='Quantization type for model weights')

    # TensorBoard parameters
    parser.add_argument('--tensorboard_dir', type=str, default=None,
                        help='Directory for TensorBoard logs. If None, uses output_dir/tensorboard')

    # Gemma conversion/download parameters
    parser.add_argument('--download_model', action='store_true',
                        help='Download model from Hugging Face Hub')
    parser.add_argument('--convert_to_nemo', action='store_true',
                        help='Convert HF model to NeMo format')
    parser.add_argument('--nemo_model_path', type=str, default=None,
                        help='Path to pre-converted NeMo model file')

    return parser.parse_args()

def setup_directories(args):
    """Create necessary directories for outputs and logs."""
    os.makedirs(args.output_dir, exist_ok=True)

    if args.tensorboard_dir is None:
        args.tensorboard_dir = os.path.join(args.output_dir, 'tensorboard')
    os.makedirs(args.tensorboard_dir, exist_ok=True)

    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.model_cache_dir:
        os.makedirs(args.model_cache_dir, exist_ok=True)

    return ckpt_dir

def download_model_from_hub(args):
    """Download the model from HuggingFace Hub with authentication."""
    logger.info(f"Downloading model {args.model_name_or_path} from Hugging Face Hub")

    # Check if we have a token either from args or environment variable
    token = args.hf_token
    if token is None:
        token = os.environ.get('HF_TOKEN', None)

    if token is None:
        logger.warning("No Hugging Face token provided. This might fail for gated models like Gemma-3.")
        logger.warning("Set HF_TOKEN environment variable or use --hf_token argument.")
        logger.warning("You can get a token from https://huggingface.co/settings/tokens")
    else:
        # Login to Hugging Face
        huggingface_hub.login(token=token)
        logger.info("Logged in to Hugging Face Hub")

    # Set cache directory for model downloads if specified
    cache_dir = args.model_cache_dir

    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

    # Download tokenizer and config first
    logger.info("Downloading tokenizer and configuration...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=cache_dir,
        token=token,
    )

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=cache_dir,
        token=token,
    )

    # Download the model with 8-bit quantization if specified
    logger.info("Downloading model weights...")
    if args.quantization == 'int8':
        logger.info("Using 8-bit quantization for model download")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=cache_dir,
            token=token,
            device_map="auto",
            load_in_8bit=True,  # This enables INT8 quantization during loading
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=cache_dir,
            token=token,
            device_map="auto",
        )

    logger.info(f"Model successfully downloaded: {args.model_name_or_path}")

    # Return information needed for conversion to NeMo format
    return {
        "model": model,
        "tokenizer": tokenizer,
        "config": config,
        "model_path": args.model_name_or_path,
        "cache_dir": cache_dir
    }

def convert_hf_to_nemo(args, model_info=None):
    """Convert HuggingFace model to NeMo format."""
    logger.info("Converting HuggingFace model to NeMo format")

    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

    # If model_info is not provided, get model path or name
    model_name_or_path = model_info["model_path"] if model_info else args.model_name_or_path
    cache_dir = model_info["cache_dir"] if model_info else args.model_cache_dir

    # Create base configuration for conversion
    config = {
        "name": "megatron_gemma_3",
        "tensor_model_parallel_size": 1,  # For conversion, we use TP size 1
        "pipeline_model_parallel_size": 1,  # For conversion, we use PP size 1
        "encoder_seq_length": args.max_seq_length,
        "max_position_embeddings": args.max_seq_length,
        "position_embedding_type": "rope",
        "tokenizer": {
            "library": "huggingface",
            "type": "auto",
            "model_name": model_name_or_path,
            "use_fast": True
        }
    }

    # Convert configuration to OmegaConf
    config = OmegaConf.create(config)

    # Use the NeMo model conversion utility
    nemo_model_path = os.path.join(args.output_dir, "gemma_3_12b_nemo.nemo")

    logger.info("Starting conversion process... this may take some time")

    # Token is needed for accessing the model during conversion
    token = args.hf_token
    if token is None:
        token = os.environ.get('HF_TOKEN', None)

    # Convert HF model to NeMo
    # We use MegatronGPTModel.from_pretrained_checkpoint for Gemma conversion
    nemo_model = MegatronGPTModel.from_pretrained(
        model_name=model_name_or_path,
        save_restore_connector=NLPSaveRestoreConnector(),
        hf_model_loading_mirror_map=None,
        config=config,
        trainer=None,
        modify_config_fn=None,
    )

    # Save the converted model to NeMo format
    nemo_model.save_to(nemo_model_path)

    logger.info(f"Conversion complete. NeMo model saved to {nemo_model_path}")

    return nemo_model_path

def process_jsonl_files(data_dir, max_seq_length):
    """Process all JSONL files in the given directory and extract text data."""
    logger.info(f"Processing JSONL files from {data_dir}")
    jsonl_files = glob.glob(os.path.join(data_dir, "*.jsonl"))

    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {data_dir}")

    logger.info(f"Found {len(jsonl_files)} JSONL files")

    # List to store all text samples
    all_texts = []

    for file_path in jsonl_files:
        logger.info(f"Processing file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    # Parse the JSON line
                    data = json.loads(line)

                    # Extract text from the 'text' field
                    if 'text' in data:
                        text = data['text']
                        # Truncate if longer than max_seq_length
                        if len(text) > max_seq_length * 4:  # Rough character estimation
                            text = text[:max_seq_length * 4]
                        all_texts.append(text)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON line in {file_path}")
                except KeyError:
                    logger.warning(f"'text' field not found in JSON line in {file_path}")

    logger.info(f"Extracted {len(all_texts)} text samples from JSONL files")

    return all_texts

def prepare_training_data(texts, output_dir, max_seq_length):
    """Prepare training data for NeMo format."""
    data_file = os.path.join(output_dir, "training_data.jsonl")

    with open(data_file, 'w', encoding='utf-8') as f:
        for text in texts:
            # NeMo expects a specific format for pretraining data
            # For simplicity, we're using a direct text format
            f.write(json.dumps({"text": text}) + "\n")

    logger.info(f"Prepared training data saved to {data_file}")
    return data_file

def create_nemo_config(args, train_data_path):
    """Create NeMo configuration for training."""
    config = {
        "trainer": {
            "devices": 1,
            "num_nodes": 1,
            "precision": 16,  # Mixed precision training
            "logger": True,
            "enable_checkpointing": True,
            "replace_sampler_ddp": False,
            "max_epochs": args.epochs,
            "accelerator": "gpu",
            "strategy": {
                "class_path": "nemo.collections.nlp.parts.nlp_overrides.NLPDDPStrategy",
                "init_args": {
                    "find_unused_parameters": False,
                    "bucket_cap_mb": 125,
                    "gradient_as_bucket_view": True,
                }
            },
            "num_sanity_val_steps": 0,
        },
        "model": {
            "micro_batch_size": args.batch_size,
            "global_batch_size": args.batch_size * args.grad_accum_steps,
            "tensor_model_parallel_size": args.tp_size,
            "pipeline_model_parallel_size": args.pp_size,
            "virtual_pipeline_model_parallel_size": None,
            "megatron_amp_O2": True,
            "seed": 42,
            "use_cpu_initialization": False,
            "apex_transformer_log_level": 30,
            "gradient_accumulation_fusion": False,

            # Gemma-3 specific settings
            "encoder_seq_length": args.max_seq_length,
            "max_position_embeddings": args.max_seq_length,
            "position_embedding_type": "rope",
            "tokenizer": {
                "library": "huggingface",
                "type": "auto",
                "model_name": args.model_name_or_path,
            },

            # Optimization settings
            "optim": {
                "name": "distributed_fused_adam",
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
                "betas": [0.9, 0.95],
                "sched": {
                    "name": "CosineAnnealing",
                    "warmup_steps": args.warmup_steps,
                    "constant_steps": 0,
                    "min_lr": 1e-5,
                },
            },

            # Data settings
            "data": {
                "data_impl": "jsonl",
                "splits_string": "100,0,0",
                "seq_length": args.max_seq_length,
                "skip_warmup": True,
                "num_workers": 2,
                "dataloader_type": "single",
                "reset_position_ids": False,
                "reset_attention_mask": False,
                "eod_mask_loss": False,
                "train_data_paths": [train_data_path],
                "val_data_paths": [],
                "test_data_paths": [],
            },

            # Quantization settings if enabled
            "quantization": {
                "enabled": args.quantization == "int8",
                "quantize_weight_in_forward": True,
                "calibrate": True,
                "use_int8": True,
            } if args.quantization == "int8" else {"enabled": False},
        }
    }

    # Create OmegaConf object from dict
    return OmegaConf.create(config)

def setup_tensorboard(args):
    """Set up TensorBoard writer."""
    # Create a timestamp for unique logging directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.tensorboard_dir, f"run_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)

    logger.info(f"TensorBoard logs will be written to {log_dir}")
    return writer

def load_and_prepare_model(args, config, nemo_model_path=None):
    """Load and prepare the model for training."""
    # If a pre-converted NeMo model is provided, load it directly
    if nemo_model_path or args.nemo_model_path:
        model_path = nemo_model_path if nemo_model_path else args.nemo_model_path
        logger.info(f"Loading pre-converted NeMo model from {model_path}")
        model = MegatronGPTModel.restore_from(
            restore_path=model_path,
            trainer=None,
            override_config_path=config
        )
    else:
        # This path should not be reached as we should have a NeMo model by now
        # But just in case, we'll handle loading from HuggingFace
        logger.info(f"Loading model from HuggingFace: {args.model_name_or_path}")
        model = MegatronGPTModel.from_pretrained(
            model_name=args.model_name_or_path,
            config_override=config,
            trainer=None,
        )

    # Apply 8-bit quantization if specified (through config parameters)
    if args.quantization == "int8":
        logger.info("Using 8-bit quantization for training")

    logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    return model

def train_model(model, config, args, tensorboard_writer, ckpt_dir):
    """Train the model with the specified configuration."""
    logger.info("Starting model training...")

    # Build a trainer using the NeMo Megatron trainer builder
    trainer = MegatronTrainerBuilder(config).create_trainer()
    model.set_trainer(trainer)

    # Setup training callbacks
    callback_params = {
        "save_nemo_on_train_end": True,
        "save_nemo_interval": args.save_interval,
        "save_best_model": True,
    }

    # Set callbacks for the trainer
    callbacks = [
        nemo.collections.nlp.callbacks.MegatronCheckpointCallback(
            folder=ckpt_dir,
            **callback_params
        )
    ]

    for callback in callbacks:
        trainer.callbacks.append(callback)

    # Add custom TensorBoard logging
    original_log_batch_step = model.log_batch_step

    def log_batch_step_with_tensorboard(batch_idx, data_iter_step, scheduler_iter_step):
        # Call the original log batch step method
        metrics = original_log_batch_step(batch_idx, data_iter_step, scheduler_iter_step)

        if metrics:
            # Log metrics to TensorBoard
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.item()
                tensorboard_writer.add_scalar(f"train/{metric_name}", metric_value, scheduler_iter_step)

            # Log learning rate
            lr = model.lr_scheduler.get_last_lr()[0]
            tensorboard_writer.add_scalar("train/learning_rate", lr, scheduler_iter_step)

        return metrics

    # Replace the log batch step method with our custom one
    model.log_batch_step = log_batch_step_with_tensorboard

    # Start training
    logger.info("Training started")
    trainer.fit(model)

    # Save the final model
    model_save_path = os.path.join(args.output_dir, "gemma_3_12b_ft_final.nemo")
    model.save_to(model_save_path)
    logger.info(f"Final model saved to {model_save_path}")

    return model_save_path

def main():
    # Parse command-line arguments
    args = parse_args()

    # Setup directories
    ckpt_dir = setup_directories(args)

    # Download model from HuggingFace Hub if requested
    model_info = None
    if args.download_model:
        model_info = download_model_from_hub(args)

    # Convert HuggingFace model to NeMo format if requested
    nemo_model_path = None
    if args.convert_to_nemo:
        nemo_model_path = convert_hf_to_nemo(args, model_info)
    elif args.nemo_model_path:
        nemo_model_path = args.nemo_model_path

    # Process JSONL files
    texts = process_jsonl_files(args.data_dir, args.max_seq_length)

    # Prepare training data
    train_data_path = prepare_training_data(texts, args.output_dir, args.max_seq_length)

    # Create NeMo configuration
    config = create_nemo_config(args, train_data_path)

    # Setup TensorBoard
    tensorboard_writer = setup_tensorboard(args)

    # Load and prepare model
    model = load_and_prepare_model(args, config, nemo_model_path)

    # Train model
    model_save_path = train_model(model, config, args, tensorboard_writer, ckpt_dir)

    # Final log
    logger.info(f"Unsupervised fine-tuning completed. Model saved to {model_save_path}")
    tensorboard_writer.close()

if __name__ == "__main__":
    main()