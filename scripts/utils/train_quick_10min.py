"""
Quick Training Script - Train model in ~10 minutes
Uses small dataset and optimized settings for fast training
"""

import os
import sys
import yaml
import json
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
import numpy as np

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class QuickParaphraseDataset(Dataset):
    """Dataset class for quick training"""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item.get('input', '')
        target_text = item.get('target', '')
        tone = item.get('tone', 'academic')
        strength = item.get('strength', 'medium')
        
        # Build input with control tokens
        prefix = f"paraphrase <tone={tone}> <strength={strength}>:"
        input_text = f"{prefix} {input_text}"
        
        # Tokenize (return lists, not tensors)
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors=None
        )
        
        # Use text_target parameter for modern tokenization
        target_encoding = self.tokenizer(
            text_target=target_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors=None
        )
        
        return {
            'input_ids': input_encoding['input_ids'],
            'attention_mask': input_encoding['attention_mask'],
            'labels': target_encoding['input_ids']
        }


def quick_train():
    """Quick training function - optimized for ~10 minutes"""
    
    print("="*60)
    print("Quick Training - Target: ~10 minutes")
    print("="*60)
    
    # Load config
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Quick training settings (smaller dataset for speed)
    train_file = "data/processed/train_quick_2k.json"
    val_file = "data/processed/val_quick_500.json"
    
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print(f"Error: Quick training files not found!")
        print(f"Expected: {train_file}, {val_file}")
        return
    
    # Load data
    print(f"\nLoading data...")
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(val_file, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    print(f"Train: {len(train_data):,} samples")
    print(f"Val: {len(val_data):,} samples")
    
    # FORCE GPU USAGE - NO CPU FALLBACK
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! GPU is required for training. Please check your CUDA installation.")
    
    device = torch.device('cuda')
    
    # Print GPU information
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    print("="*60)
    print("GPU Configuration:")
    print(f"  Device: {device}")
    print(f"  GPU Name: {gpu_name}")
    print(f"  GPU Memory: {gpu_memory:.2f} GB")
    print(f"  CUDA Version: {torch.version.cuda}")
    print("="*60)
    
    # Load model and tokenizer
    base_model_name = config['model']['base_model']
    print(f"\nLoading model: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    
    # Add special tokens
    style_config = config.get('style', {})
    tone_tokens = [f"<tone={t}>" for t in style_config.get('tones', [])]
    strength_tokens = [f"<strength={s}>" for s in style_config.get('strengths', [])]
    special_tokens = list({*tone_tokens, *strength_tokens})
    if special_tokens:
        tokenizer.add_tokens(special_tokens, special_tokens=False)
        model.resize_token_embeddings(len(tokenizer))
    
    # Setup LoRA (minimal for speed)
    print("\nSetting up LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=4,  # Very small r for fastest training
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q", "v"],  # T5 modules
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Create datasets (max_length = 512 as requested, or lower for speed)
    print("\nCreating datasets...")
    max_seq_length = 512  # As requested, or use 256 for faster training
    train_dataset = QuickParaphraseDataset(train_data, tokenizer, max_length=max_seq_length)
    val_dataset = QuickParaphraseDataset(val_data, tokenizer, max_length=max_seq_length)
    print(f"Train dataset: {len(train_dataset):,} samples")
    print(f"Val dataset: {len(val_dataset):,} samples")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8
    )
    
    # Training arguments - OPTIMIZED FOR GPU WITH FP16
    print("\nSetting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./models/checkpoints_quick",
        num_train_epochs=1,  # Only 1 epoch
        per_device_train_batch_size=16,  # Optimized for GPU memory
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,  # Effective batch size = 32
        learning_rate=5e-4,  # Higher LR for faster convergence
        warmup_steps=10,  # Minimal warmup
        max_steps=50,  # Fewer steps for quick test
        logging_steps=10,
        save_steps=50,
        save_total_limit=1,
        eval_strategy="no",  # Disable evaluation to save time (transformers 4.57.3 uses eval_strategy)
        save_strategy="steps",
        dataloader_num_workers=0,  # 0 workers on Windows (multiprocessing can slow down)
        dataloader_pin_memory=True,  # Pin memory for faster GPU transfer
        fp16=True,  # FORCE fp16 mixed precision for GPU
        bf16=False,  # Use fp16, not bf16
        report_to=None,  # Disable wandb/tensorboard
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        optim="adamw_torch",  # Use PyTorch optimizer
        max_grad_norm=1.0,  # Gradient clipping
    )
    
    # Move model to GPU explicitly and verify
    model = model.to(device)
    # Verify model is on GPU
    next_param = next(model.parameters())
    actual_device = next_param.device
    if actual_device.type != 'cuda':
        raise RuntimeError(f"Model is not on GPU! Expected cuda, got {actual_device.type}")
    
    # torch.compile disabled temporarily - can cause issues with LoRA on Windows
    # Uncomment below to enable after confirming training works well
    # try:
    #     model = torch.compile(model, mode="reduce-overhead")
    #     print("✅ Model compiled with torch.compile for faster training")
    # except Exception as e:
    #     print(f"⚠️  torch.compile not available or failed: {e}")
    #     print("   Continuing without compilation (still fast)")
    
    print(f"✅ Model moved to {device} (verified: {actual_device})")
    
    # Enable cuDNN benchmark for faster training
    torch.backends.cudnn.benchmark = True
    print("✅ cuDNN benchmark enabled for faster training")
    
    # Fix: Add generation_config attribute to TrainingArguments for Seq2SeqTrainer compatibility
    # Seq2SeqTrainer checks for this attribute even if it's None
    if not hasattr(training_args, 'generation_config'):
        training_args.generation_config = None
    
    # Create trainer
    print("\nInitializing Seq2SeqTrainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        # No eval_dataset to reduce overhead
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting quick training (target: ~10 minutes)...")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Save model
    print("\n" + "="*60)
    print("Training completed! Saving model...")
    print("="*60)
    
    output_dir = "./models/final_quick"
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✅ Model saved to: {output_dir}")
    print("="*60)
    print("Quick training completed!")
    print("="*60)


if __name__ == "__main__":
    quick_train()

