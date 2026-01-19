"""
Training Module - GPU-Optimized
Fine-tunes T5/BART model with LoRA for paraphrase generation
Uses pre-tokenization to eliminate CPU bottleneck
"""

import os
import yaml
import json
import torch
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset, load_from_disk
from pathlib import Path
from typing import Dict, Any


# Enable TF32 for faster training on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


class ModelTrainer:
    """Handles model training with LoRA - GPU-optimized with pre-tokenization"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.style_config = self.config.get('style', {})
        
        # Ensure learning_rate is float
        if isinstance(self.training_config.get('learning_rate'), str):
            self.training_config['learning_rate'] = float(self.training_config['learning_rate'])
        
        # FORCE GPU USAGE - NO CPU FALLBACK
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! GPU is required for training. Please check your CUDA installation.")
        
        self.device = torch.device('cuda')
        
        # Print GPU information
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print("="*60)
        print("GPU Configuration:")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        print(f"  Device: {self.device}")
        print(f"  GPU Name: {gpu_name}")
        print(f"  GPU Memory: {gpu_memory:.2f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
        print("="*60)
        
        # Initialize model and tokenizer
        # Check if there's a previously trained model to continue from
        final_model_dir = self.config['paths']['final_model_dir']
        self.base_model_name = self.model_config['base_model']
        
        # Check if we should resume from a previously trained model
        is_resuming = os.path.exists(final_model_dir) and (
            os.path.exists(os.path.join(final_model_dir, "adapter_config.json")) or
            os.path.exists(os.path.join(final_model_dir, "config.json"))
        )
        
        if is_resuming:
            print(f"\n🔄 Found previously trained model at {final_model_dir}")
            print("   Continuing training from this model...")
            model_path = final_model_dir
        else:
            print(f"\n🆕 Starting fresh from base model: {self.base_model_name}")
            model_path = self.base_model_name
        
        print(f"Loading tokenizer from: {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Get special tokens that should be in the model
        self.tone_tokens = [f"<tone={t}>" for t in self.style_config.get('tones', [])]
        self.strength_tokens = [f"<strength={s}>" for s in self.style_config.get('strengths', [])]
        self.special_tokens = list({*self.tone_tokens, *self.strength_tokens})
        
        # Check which special tokens are missing (if any)
        missing_tokens = []
        if self.special_tokens:
            for token in self.special_tokens:
                if token not in self.tokenizer.get_vocab():
                    missing_tokens.append(token)
        
        # Add missing tokens if starting fresh, or if resuming and tokens are missing
        if missing_tokens:
            if is_resuming:
                print(f"⚠️  Warning: Some special tokens are missing: {missing_tokens}")
                print("   Adding missing tokens...")
            else:
                print(f"Adding {len(missing_tokens)} special tokens...")
            
            self.tokenizer.add_tokens(missing_tokens, special_tokens=False)
            # Resize embeddings BEFORE loading model weights
            vocab_size_before = len(self.tokenizer) - len(missing_tokens)
            vocab_size_after = len(self.tokenizer)
            print(f"   Vocabulary size: {vocab_size_before} → {vocab_size_after}")
        
        # Load model - if resuming, we need to load base model first, then adapter
        if is_resuming:
            print(f"Loading base model from: {self.base_model_name}...")
            # Load base model first
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.base_model_name,
                ignore_mismatched_sizes=True  # Allow size mismatch for embeddings
            )
            
            # Resize embeddings to match tokenizer if needed
            if missing_tokens or self.model.get_input_embeddings().num_embeddings != len(self.tokenizer):
                print(f"Resizing model embeddings to match tokenizer ({len(self.tokenizer)} tokens)...")
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Load the adapter weights
            print(f"Loading adapter weights from: {model_path}...")
            # Handle compatibility with older PEFT versions
            try:
                self.model = PeftModel.from_pretrained(self.model, model_path)
            except TypeError as e:
                if "unexpected keyword argument" in str(e):
                    print(f"⚠️  Compatibility issue with adapter config. Fixing...")
                    # Load and clean adapter config
                    import json
                    adapter_config_path = os.path.join(model_path, "adapter_config.json")
                    if os.path.exists(adapter_config_path):
                        with open(adapter_config_path, 'r') as f:
                            adapter_config = json.load(f)
                        
                        # Remove unsupported parameters (keep only essential ones)
                        # Based on current PEFT version supported parameters
                        essential_params = {
                            'base_model_name_or_path',
                            'bias',  # Use this instead of lora_bias
                            'fan_in_fan_out',
                            'inference_mode',
                            'init_lora_weights',
                            'lora_alpha',
                            'lora_dropout',  # Note: lora_bias is deprecated, use 'bias' instead
                            'modules_to_save',
                            'peft_type',
                            'r',
                            'target_modules',
                            'task_type',
                            # Optional parameters that might be present
                            'auto_mapping',
                            'revision',
                            'layers_to_transform',
                            'layers_pattern',
                            'rank_pattern',
                            'alpha_pattern',
                            'megatron_config',
                            'megatron_core',
                            'loftq_config',
                        }
                        
                        cleaned_config = {k: v for k, v in adapter_config.items() 
                                        if k in essential_params}
                        
                        # Save cleaned config back
                        with open(adapter_config_path, 'w') as f:
                            json.dump(cleaned_config, f, indent=2)
                        
                        print("✅ Cleaned adapter config. Retrying...")
                        # Try loading again with cleaned config
                        self.model = PeftModel.from_pretrained(self.model, model_path)
                    else:
                        raise e
                else:
                    raise e
            
            # CRITICAL: Enable training mode and ensure adapter is trainable
            self.model.train()  # Set to training mode
            # Enable adapter for training
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower() or 'adapter' in name.lower():
                    param.requires_grad = True
            
            print("✅ Model and adapter loaded successfully (trainable mode enabled)")
        else:
            print(f"Loading model from: {model_path}...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            
            # Resize embeddings if we added tokens
            if missing_tokens:
                self.model.resize_token_embeddings(len(self.tokenizer))
                print(f"✅ Model embeddings resized to {len(self.tokenizer)} tokens")
        
        # Set target_modules for LoRA based on model architecture
        if 't5' in self.base_model_name.lower():
            self.lora_target_modules = ["q", "v"]
        elif 'bart' in self.base_model_name.lower():
            self.lora_target_modules = ["q_proj", "v_proj"]
        elif 'pegasus' in self.base_model_name.lower():
            self.lora_target_modules = ["q", "v"]
        else:
            self.lora_target_modules = ["q", "v"]
        
        # Setup LoRA if enabled
        # Only setup LoRA if we're starting from base model (not resuming)
        if self.model_config.get('use_lora', True):
            if not is_resuming:
                # Starting fresh - setup LoRA
                self._setup_lora()
            else:
                # Resuming from trained model - LoRA already loaded via PeftModel.from_pretrained
                print("✅ LoRA adapter already loaded from trained model")
        
        # Move model to GPU explicitly and verify
        self.model = self.model.to(self.device)
        
        # Ensure model is in training mode (especially important when resuming)
        self.model.train()
        
        # Verify model is on GPU and has trainable parameters
        next_param = next(self.model.parameters())
        actual_device = next_param.device
        if actual_device.type != 'cuda':
            raise RuntimeError(f"Model is not on GPU! Expected cuda, got {actual_device.type}")
        
        # Check if model has trainable parameters (critical for LoRA)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if trainable_params == 0:
            print("⚠️  WARNING: No trainable parameters found!")
            print("   Attempting to enable training for LoRA adapter...")
            # Try to enable training for all LoRA parameters
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower() or 'adapter' in name.lower():
                    param.requires_grad = True
                    print(f"   Enabled training for: {name}")
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            if trainable_params == 0:
                raise RuntimeError("No trainable parameters found! Cannot train the model.")
            print(f"✅ Found {trainable_params:,} trainable parameters")
        else:
            print(f"✅ Model has {trainable_params:,} trainable parameters")
        
        # torch.compile disabled temporarily - can cause issues with LoRA on Windows
        # Uncomment below to enable after confirming training works well
        # try:
        #     self.model = torch.compile(self.model, mode="reduce-overhead")
        #     print("✅ Model compiled with torch.compile for faster training")
        # except Exception as e:
        #     print(f"⚠️  torch.compile not available or failed: {e}")
        #     print("   Continuing without compilation (still fast)")
        
        # Print model device and GPU memory
        gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"✅ Model moved to {self.device} (verified: {actual_device})")
        print(f"✅ GPU Memory Allocated: {gpu_mem_allocated:.2f} GB")
        print(f"✅ Model device: {next(self.model.parameters()).device}")
        
        # Log model type and path
        if is_resuming:
            print(f"\n📦 Model Type: LoRA Adapter (Resuming from {final_model_dir})")
            print(f"📦 Base Model: {self.base_model_name}")
        else:
            print(f"\n📦 Model Type: LoRA Adapter (Starting fresh from {self.base_model_name})")
        
        # Check if LoRA is actually applied
        if hasattr(self.model, 'peft_config'):
            print(f"✅ LoRA Configuration:")
            for key, config in self.model.peft_config.items():
                print(f"   - r: {config.r}, alpha: {config.lora_alpha}, dropout: {config.lora_dropout}")
                print(f"   - target_modules: {config.target_modules}")
    
    def _setup_lora(self):
        """Configure and apply LoRA"""
        print("\nSetting up LoRA...")
        
        target_modules = getattr(self, 'lora_target_modules', None) or \
                        self.model_config.get('target_modules', ["q", "v"])
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=self.model_config['lora_r'],
            lora_alpha=self.model_config['lora_alpha'],
            lora_dropout=self.model_config['lora_dropout'],
            target_modules=target_modules,
            bias="none",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def _preprocess_function(self, examples, max_source_length=256, max_target_length=256):
        """Preprocess function for batched tokenization"""
        try:
            default_tone = self.style_config.get('tones', ['academic'])[0]
            default_strength = self.style_config.get('strengths', ['medium'])[0]
            
            # Get input texts - ensure they are lists
            if isinstance(examples['input'], list):
                input_texts = examples['input']
            else:
                input_texts = [examples['input']]
            
            if isinstance(examples['target'], list):
                target_texts = examples['target']
            else:
                target_texts = [examples['target']]
            
            # Ensure same length
            batch_size = len(input_texts)
            if len(target_texts) != batch_size:
                raise ValueError(f"Mismatch: {len(input_texts)} inputs vs {len(target_texts)} targets")
            
            # Build input texts - use data as-is if it already has prompt prefix
            inputs = []
            for i, input_text in enumerate(input_texts):
                # Ensure input_text is string
                if not isinstance(input_text, str):
                    input_text = str(input_text)
                
                # Check if input already has a prompt prefix (from preprocessing)
                if input_text.startswith("humanize: ") or input_text.startswith("rewrite to sound natural and human: "):
                    # Data already has prompt, use as-is
                    inputs.append(input_text)
                else:
                    # No prompt prefix, add one with tone/strength control tokens
                    if 'tone' in examples:
                        tones = examples['tone'] if isinstance(examples['tone'], list) else [examples['tone']]
                        tone = tones[i] if i < len(tones) else default_tone
                    else:
                        tone = default_tone
                    
                    if 'strength' in examples:
                        strengths = examples['strength'] if isinstance(examples['strength'], list) else [examples['strength']]
                        strength = strengths[i] if i < len(strengths) else default_strength
                    else:
                        strength = default_strength
                    
                    # Add prompt with control tokens
                    if tone != default_tone or strength != default_strength:
                        prefix = f"humanize <tone={tone}> <strength={strength}>:"
                        inputs.append(f"{prefix} {input_text}")
                    else:
                        inputs.append(f"humanize: {input_text}")
            
            # Ensure target_texts are strings
            target_texts = [str(t) if not isinstance(t, str) else t for t in target_texts]
            
            # Tokenize inputs (NO padding - DataCollator will handle dynamic padding)
            model_inputs = self.tokenizer(
                inputs,
                max_length=max_source_length,
                truncation=True,
                padding=False,  # Dynamic padding by DataCollator
            )
            
            # Tokenize targets using text_target parameter (modern approach, no as_target_tokenizer)
            labels = self.tokenizer(
                text_target=target_texts,
                max_length=max_target_length,
                truncation=True,
                padding=False,  # Dynamic padding by DataCollator
            )
            
            # DataCollatorForSeq2Seq will handle padding and -100 for labels automatically
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        except Exception as e:
            print(f"Error in _preprocess_function: {e}")
            print(f"Input type: {type(examples.get('input'))}")
            print(f"Target type: {type(examples.get('target'))}")
            raise
    
    def clear_cache(self, cache_dir: str = "./data/cache"):
        """Clear tokenization cache"""
        import shutil
        train_cache_path = os.path.join(cache_dir, "train_tokenized")
        val_cache_path = os.path.join(cache_dir, "val_tokenized")
        cache_marker_file = os.path.join(cache_dir, "cache_marker.txt")
        
        cleared = False
        if os.path.exists(train_cache_path):
            shutil.rmtree(train_cache_path)
            cleared = True
        if os.path.exists(val_cache_path):
            shutil.rmtree(val_cache_path)
            cleared = True
        if os.path.exists(cache_marker_file):
            os.remove(cache_marker_file)
            cleared = True
        
        if cleared:
            print(f"✅ Cache cleared: {cache_dir}")
        else:
            print(f"ℹ️  No cache found to clear")
    
    def load_and_preprocess_data(self, train_file: str, val_file: str, cache_dir: str = "./data/cache", clear_cache_first: bool = False):
        """Load JSON data and pre-tokenize using HuggingFace datasets"""
        print(f"\n{'='*60}")
        print("Loading and Preprocessing Data")
        print(f"{'='*60}")
        
        # Clear cache if requested
        if clear_cache_first:
            print("🗑️  Clearing cache as requested...")
            self.clear_cache(cache_dir)
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        train_cache_path = os.path.join(cache_dir, "train_tokenized")
        val_cache_path = os.path.join(cache_dir, "val_tokenized")
        
        # Check if cached tokenized datasets exist
        # Note: Cache is based on file paths, so different train files will have different caches
        # For safety, we'll regenerate cache if train file path changes
        cache_marker_file = os.path.join(cache_dir, "cache_marker.txt")
        cache_valid = False
        
        if os.path.exists(train_cache_path) and os.path.exists(val_cache_path) and os.path.exists(cache_marker_file):
            try:
                with open(cache_marker_file, 'r') as f:
                    cached_train_file = f.read().strip()
                if cached_train_file == train_file:
                    cache_valid = True
                    print(f"✅ Cache found for: {train_file}")
                else:
                    print(f"⚠️  Cache mismatch! Cached: {cached_train_file}, Current: {train_file}")
                    print(f"   Regenerating cache...")
            except Exception as e:
                print(f"⚠️  Error reading cache marker: {e}")
                print(f"   Regenerating cache...")
        
        if cache_valid:
            print("Loading pre-tokenized datasets from cache...")
            try:
                train_dataset = load_from_disk(train_cache_path)
                val_dataset = load_from_disk(val_cache_path)
                print(f"✅ Train: {len(train_dataset):,} samples")
                print(f"✅ Val: {len(val_dataset):,} samples")
                return train_dataset, val_dataset
            except Exception as e:
                print(f"⚠️  Cache loading failed: {e}")
                print("Regenerating cache...")
        
        # Load raw JSON data
        print(f"Loading raw data from {train_file} and {val_file}...")
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(val_file, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        
        print(f"Raw data loaded - Train: {len(train_data):,}, Val: {len(val_data):,}")
        
        # Subsample validation set if too large (32k -> 3k max for faster evaluation)
        if len(val_data) > 3000:
            print(f"⚠️  Validation set is large ({len(val_data):,}). Subsampling to 3,000 samples for faster evaluation...")
            val_data = val_data[:3000]
        
        # Convert to HuggingFace Dataset
        print("Converting to HuggingFace Dataset format...")
        train_hf_dataset = Dataset.from_list(train_data)
        val_hf_dataset = Dataset.from_list(val_data)
        
        # Set sequence lengths for 6GB GPU
        # Unified: Both input and output use 256 tokens for consistency
        max_source_length = 256  # Input length
        max_target_length = 256  # Output length (same as input for consistency)
        
        print(f"\nPre-tokenizing datasets (max_source={max_source_length}, max_target={max_target_length})...")
        print("This may take a few minutes but will speed up training significantly...")
        
        # Pre-tokenize with batched processing
        # Use num_proc=1 on Windows if multiprocessing causes issues
        import sys
        num_proc = 4 if sys.platform != 'win32' else 1
        
        # Use larger batch size for faster tokenization
        batch_size = 1000  # Process 1000 examples at a time
        
        print(f"Using batch_size={batch_size} for tokenization (num_proc={num_proc})...")
        
        try:
            train_dataset = train_hf_dataset.map(
                lambda x: self._preprocess_function(x, max_source_length, max_target_length),
                batched=True,
                batch_size=batch_size,
                num_proc=num_proc,
                remove_columns=train_hf_dataset.column_names,  # Remove raw text columns
                desc="Tokenizing train dataset"
            )
            
            val_dataset = val_hf_dataset.map(
                lambda x: self._preprocess_function(x, max_source_length, max_target_length),
                batched=True,
                batch_size=batch_size,
                num_proc=num_proc,
                remove_columns=val_hf_dataset.column_names,
                desc="Tokenizing val dataset"
            )
        except Exception as e:
            print(f"\n❌ Error during tokenization: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise
        
        # Save tokenized datasets to cache
        print(f"\nSaving tokenized datasets to cache...")
        train_dataset.save_to_disk(train_cache_path)
        val_dataset.save_to_disk(val_cache_path)
        
        # Save cache marker with train file path
        with open(cache_marker_file, 'w') as f:
            f.write(train_file)
        
        print(f"✅ Cache saved to {cache_dir}")
        
        print(f"\n✅ Pre-tokenization complete!")
        print(f"   Train: {len(train_dataset):,} samples")
        print(f"   Val: {len(val_dataset):,} samples")
        
        return train_dataset, val_dataset
    
    def _check_data_quality(self, train_file: str, sample_size: int = 10):
        """Check data quality by comparing input vs target differences"""
        print(f"\n{'='*60}")
        print("Checking Data Quality")
        print(f"{'='*60}")
        
        try:
            with open(train_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"Total samples in file: {len(data)}")
            print(f"Checking first {min(sample_size, len(data))} samples...\n")
            
            import difflib
            total_word_changes = 0
            total_length_diff = 0
            samples_with_small_diff = 0
            
            for i, sample in enumerate(data[:sample_size]):
                input_text = sample.get('input', '').replace('humanize: ', '').strip()
                target_text = sample.get('target', '').strip()
                
                input_words = input_text.split()
                target_words = target_text.split()
                
                # Calculate word-level differences
                diff = list(difflib.unified_diff(input_words, target_words, lineterm='', n=0))
                word_changes = len([d for d in diff if (d.startswith('+') or d.startswith('-')) 
                                   and not d.startswith('+++') and not d.startswith('---')])
                
                length_diff = abs(len(input_words) - len(target_words))
                change_ratio = word_changes / max(len(input_words), 1)
                
                total_word_changes += word_changes
                total_length_diff += length_diff
                
                if change_ratio < 0.1:  # Less than 10% change
                    samples_with_small_diff += 1
                
                if i < 3:  # Show first 3 samples in detail
                    print(f"Sample {i+1}:")
                    print(f"  Input words: {len(input_words)}, Target words: {len(target_words)}")
                    print(f"  Word changes: {word_changes} ({change_ratio*100:.1f}%)")
                    print(f"  Input preview: {input_text[:100]}...")
                    print(f"  Target preview: {target_text[:100]}...")
                    print()
            
            avg_word_changes = total_word_changes / min(sample_size, len(data))
            avg_length_diff = total_length_diff / min(sample_size, len(data))
            
            print(f"Average word changes per sample: {avg_word_changes:.1f}")
            print(f"Average length difference: {avg_length_diff:.1f} words")
            print(f"Samples with <10% change: {samples_with_small_diff}/{min(sample_size, len(data))}")
            
            if samples_with_small_diff > sample_size * 0.5:
                print(f"\n⚠️  WARNING: More than 50% of samples have minimal changes!")
                print(f"   This may cause the model to learn copying instead of paraphrasing.")
            else:
                print(f"\n✅ Data quality looks good - sufficient variation between input and target")
            
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"⚠️  Could not check data quality: {e}")
            print()
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation (lightweight to avoid CPU bottleneck)
        
        Now includes diversity metrics to encourage actual paraphrasing instead of copying.
        """
        predictions, labels = eval_pred
        
        # Clean predictions: replace invalid values (-100, negative, or out of range) with pad_token_id
        # This prevents OverflowError when decoding
        try:
            predictions = np.asarray(predictions, dtype=np.int32)
        except (ValueError, TypeError):
            # If conversion fails, try to handle as list/tuple
            predictions = np.array(predictions, dtype=np.int32)
        
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        # Replace invalid token IDs (negative, -100, or >= vocab_size) with pad_token_id
        vocab_size = len(self.tokenizer)
        # Clip values to valid range [0, vocab_size-1] and replace -100
        predictions_clean = np.clip(predictions, 0, vocab_size - 1)
        predictions_clean = np.where(predictions_clean == -100, pad_token_id, predictions_clean)
        predictions_clean = np.where(predictions_clean < 0, pad_token_id, predictions_clean)
        
        # Convert to list of lists for batch_decode (handles 2D arrays)
        if predictions_clean.ndim == 2:
            predictions_list = predictions_clean.tolist()
        else:
            predictions_list = [predictions_clean.tolist()]
        
        # Decode predictions (this is CPU-bound but necessary)
        try:
            decoded_preds = self.tokenizer.batch_decode(predictions_list, skip_special_tokens=True)
        except Exception as e:
            # Fallback: decode one by one if batch_decode fails
            print(f"Warning: batch_decode failed, decoding individually: {e}")
            decoded_preds = []
            for pred in predictions_list:
                try:
                    decoded = self.tokenizer.decode(pred, skip_special_tokens=True)
                    decoded_preds.append(decoded)
                except Exception:
                    decoded_preds.append("")  # Empty string if decoding fails
        
        # Replace -100 in labels with pad_token_id for decoding
        labels = np.asarray(labels, dtype=np.int32)
        labels = np.where(labels != -100, labels, pad_token_id)
        labels = np.clip(labels, 0, vocab_size - 1)
        
        if labels.ndim == 2:
            labels_list = labels.tolist()
        else:
            labels_list = [labels.tolist()]
        
        try:
            decoded_labels = self.tokenizer.batch_decode(labels_list, skip_special_tokens=True)
        except Exception as e:
            print(f"Warning: batch_decode for labels failed: {e}")
            decoded_labels = []
            for label in labels_list:
                try:
                    decoded = self.tokenizer.decode(label, skip_special_tokens=True)
                    decoded_labels.append(decoded)
                except Exception:
                    decoded_labels.append("")
        
        # Simple metrics: average length difference (CPU-bound but fast)
        pred_lengths = [len(p.split()) for p in decoded_preds]
        label_lengths = [len(l.split()) for l in decoded_labels]
        
        avg_pred_length = np.mean(pred_lengths)
        avg_label_length = np.mean(label_lengths)
        length_diff = abs(avg_pred_length - avg_label_length)
        
        # Calculate diversity: word overlap ratio (lower is better for paraphrasing)
        # This encourages the model to use different words while preserving meaning
        import re
        word_overlaps = []
        for pred, label in zip(decoded_preds, decoded_labels):
            pred_words = set(re.findall(r'\b\w+\b', pred.lower()))
            label_words = set(re.findall(r'\b\w+\b', label.lower()))
            if len(label_words) > 0:
                overlap = len(pred_words & label_words) / len(label_words)
                word_overlaps.append(overlap)
        
        avg_word_overlap = np.mean(word_overlaps) if word_overlaps else 0.0
        
        # Calculate number preservation score (simple heuristic)
        # Count how many numbers from labels are preserved in predictions
        number_preservation_scores = []
        for pred, label in zip(decoded_preds, decoded_labels):
            label_numbers = set(re.findall(r'\b\d+(?:\.\d+)?(?:%|²|³)?\b', label))
            pred_numbers = set(re.findall(r'\b\d+(?:\.\d+)?(?:%|²|³)?\b', pred))
            if len(label_numbers) > 0:
                preserved = len(pred_numbers & label_numbers) / len(label_numbers)
                number_preservation_scores.append(preserved)
        
        avg_number_preservation = np.mean(number_preservation_scores) if number_preservation_scores else 1.0
        
        return {
            "avg_pred_length": float(avg_pred_length),
            "avg_label_length": float(avg_label_length),
            "length_diff": float(length_diff),
            "word_overlap": float(avg_word_overlap),  # Lower is better (more diverse)
            "number_preservation": float(avg_number_preservation),  # Higher is better
        }
    
    def train(self, train_file: str, val_file: str, clear_cache: bool = False):
        """Main training function with GPU optimization
        
        Args:
            train_file: Path to training JSON file
            val_file: Path to validation JSON file
            clear_cache: If True, clear tokenization cache before training
        """
        # Check data quality first
        self._check_data_quality(train_file, sample_size=10)
        
        # Load and pre-tokenize data
        train_dataset, val_dataset = self.load_and_preprocess_data(train_file, val_file, clear_cache_first=clear_cache)
        
        # Log which files are being used
        print(f"\n{'='*60}")
        print("Training Configuration")
        print(f"{'='*60}")
        # Convert to absolute paths for clarity
        train_file_abs = os.path.abspath(train_file)
        val_file_abs = os.path.abspath(val_file)
        print(f"📁 Train file: {train_file}")
        print(f"   Absolute path: {train_file_abs}")
        print(f"   File exists: {os.path.exists(train_file)}")
        print(f"📁 Validation file: {val_file}")
        print(f"   Absolute path: {val_file_abs}")
        print(f"   File exists: {os.path.exists(val_file)}")
        print(f"📊 Train samples: {len(train_dataset):,}")
        print(f"📊 Validation samples: {len(val_dataset):,}")
        print(f"💾 Output directory: {os.path.abspath(self.training_config['output_dir'])}")
        print(f"{'='*60}\n")
        
        # Data collator - handles padding efficiently for GPU
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            pad_to_multiple_of=8  # Optimize for GPU tensor cores
        )
        
        # Training arguments - OPTIMIZED FOR 6GB GPU (faster batch processing)
        per_device_batch_size = 4  # Increased batch for better GPU utilization
        per_device_eval_batch_size = 16  # Larger batch for faster evaluation (no gradients needed)
        gradient_accumulation_steps = 8  # Effective batch size = 32 (same as before, but faster)
        
        # Calculate eval steps based on dataset size
        steps_per_epoch = len(train_dataset) // (per_device_batch_size * gradient_accumulation_steps)
        eval_steps = max(500, steps_per_epoch // 4)  # Evaluate 4 times per epoch
        
        # IMPORTANT: save_steps must be a multiple of eval_steps when load_best_model_at_end=True
        # Calculate save_steps as a multiple of eval_steps
        desired_save_interval = max(1000, steps_per_epoch // 2)  # Desired save interval
        # Round up to nearest multiple of eval_steps
        save_steps = ((desired_save_interval + eval_steps - 1) // eval_steps) * eval_steps
        
        # Print calculated steps for debugging
        print(f"📊 Training Steps Calculation:")
        print(f"   Steps per epoch: {steps_per_epoch:,}")
        print(f"   Eval steps: {eval_steps:,} (evaluate 4 times per epoch)")
        print(f"   Save steps: {save_steps:,} (save 2 times per epoch, multiple of eval_steps)")
        print(f"   ✅ save_steps ({save_steps}) is a multiple of eval_steps ({eval_steps}): {save_steps % eval_steps == 0}")
        print()
        
        training_args = TrainingArguments(
            output_dir=self.training_config['output_dir'],
            num_train_epochs=self.training_config['num_epochs'],
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,  # Larger batch for faster evaluation
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=self.training_config['learning_rate'],
            warmup_steps=self.training_config['warmup_steps'],
            logging_steps=100,  # More frequent logging for better monitoring
            eval_steps=eval_steps,  # Dynamic eval steps based on dataset size
            save_steps=save_steps,  # Dynamic save steps
            save_total_limit=self.training_config['save_total_limit'],
            eval_strategy="steps",  # Enable evaluation during training
            save_strategy="steps",
            load_best_model_at_end=True,  # Load best model based on eval_loss
            metric_for_best_model="eval_loss",  # Use eval_loss to determine best model
            greater_is_better=False,  # Lower loss is better
            report_to=[],  # Disable external logging (empty list)
            fp16=True,  # Mixed precision
            bf16=False,
            dataloader_num_workers=0,  # 0 workers on Windows (multiprocessing can slow down)
            dataloader_pin_memory=True,  # Pin memory for faster GPU transfer
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            gradient_checkpointing=False,  # Disable for speed
            optim="adamw_torch",
            max_grad_norm=1.0,
        )
        
        # Fix: Add generation_config and generation_max_length attributes for Seq2SeqTrainer compatibility
        # Seq2SeqTrainer checks for these attributes during evaluation (at eval_steps)
        # This is required to avoid AttributeError when evaluation runs
        # Always set these attributes to avoid AttributeError during evaluation
        setattr(training_args, 'generation_config', None)
        # ENABLE predict_with_generate for better model selection
        # Generation-based evaluation is slower but better for paraphrasing tasks
        # It helps select models that actually paraphrase instead of copying
        setattr(training_args, 'predict_with_generate', True)  # Enable generation for better evaluation
        # Set generation params anyway (in case predict_with_generate is enabled later)
        setattr(training_args, 'generation_max_length', 256)
        setattr(training_args, 'generation_num_beams', 1)
        
        # Verify the fix was applied
        if not hasattr(training_args, 'generation_max_length'):
            raise RuntimeError("Failed to set generation_max_length attribute!")
        print(
            f"✅ Evaluation mode: generation-based (predict_with_generate=True) - Better for paraphrasing"
        )
        
        # Initialize trainer with evaluation
        print("\n" + "="*60)
        print("Initializing Seq2SeqTrainer...")
        print("="*60)
        print(f"✅ Evaluation enabled: eval_strategy='steps', eval_steps={eval_steps}")
        print(f"✅ Best model selection: metric_for_best_model='eval_loss'")
        print(f"✅ Save steps: {save_steps}")
        print("="*60)
        
        # Enable compute_metrics for generation-based evaluation
        # This helps select models that actually paraphrase instead of copying
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,  # Enable evaluation dataset
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,  # Enable metrics for better model selection
        )
        
        # Print GPU memory before training
        gpu_mem_before = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"\nGPU Memory Before Training: {gpu_mem_before:.2f} GB")
        print(f"Model Device: {next(self.model.parameters()).device}")
        
        # Calculate training steps
        total_steps = len(train_dataset) // (per_device_batch_size * gradient_accumulation_steps) * self.training_config['num_epochs']
        print(f"\nTraining Configuration:")
        print(f"  Batch size per device: {per_device_batch_size}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps}")
        print(f"  Effective batch size: {per_device_batch_size * gradient_accumulation_steps}")
        print(f"  Total training steps: ~{total_steps}")
        print(f"  Epochs: {self.training_config['num_epochs']}")
        
        # Enable cuDNN benchmark for faster training (finds optimal algorithms)
        torch.backends.cudnn.benchmark = True
        print("✅ cuDNN benchmark enabled for faster training")
        
        # Train
        print("\n" + "="*60)
        print("Starting Training...")
        print("="*60 + "\n")
        
        trainer.train()
        
        # Print GPU memory after training
        gpu_mem_after = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"\nGPU Memory After Training: {gpu_mem_after:.2f} GB")
        
        # Save final model
        final_model_dir = self.config['paths']['final_model_dir']
        os.makedirs(final_model_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("Saving Final Model")
        print(f"{'='*60}")
        print(f"📁 Saving to: {final_model_dir}")
        
        # Check if best model was loaded
        if hasattr(trainer.state, 'best_metric'):
            print(f"✅ Best model loaded (eval_loss: {trainer.state.best_metric:.4f})")
        
        self.model.save_pretrained(final_model_dir)
        self.tokenizer.save_pretrained(final_model_dir)
        
        # Verify what was saved
        adapter_path = os.path.join(final_model_dir, "adapter_config.json")
        if os.path.exists(adapter_path):
            print(f"✅ LoRA adapter saved successfully")
            print(f"   - adapter_config.json: ✅")
            print(f"   - adapter_model.safetensors: ✅")
        else:
            print(f"⚠️  Warning: No adapter_config.json found - model may not be LoRA!")
        
        # Calculate model size
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(final_model_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        size_mb = total_size / (1024 * 1024)
        print(f"📦 Model size: {size_mb:.2f} MB")
        
        if size_mb > 50:
            print(f"⚠️  Warning: Model size is large ({size_mb:.2f} MB)")
            print(f"   Expected LoRA adapter size: ~5-15 MB")
            print(f"   This might indicate full model was saved instead of adapter only")
        
        print(f"{'='*60}")
        print("✅ Training completed!")
        print(f"{'='*60}\n")
        
        return trainer


if __name__ == "__main__":
    import sys
    
    trainer = ModelTrainer()
    
    # Get data paths from config
    processed_data_dir = trainer.config['paths']['processed_data_dir']
    
    # Prefer clean PAWS splits if present, then combined_raw, and skip MPC variants
    paws_train = os.path.join(processed_data_dir, "paws_train.json")
    paws_val = os.path.join(processed_data_dir, "paws_val.json")
    train_file = os.path.join(processed_data_dir, "combined_raw_train.json")
    val_file = os.path.join(processed_data_dir, "combined_raw_val.json")
    
    if len(sys.argv) >= 3:
        train_file = sys.argv[1]
        val_file = sys.argv[2]
    elif os.path.exists(paws_train) and os.path.exists(paws_val):
        train_file = paws_train
        val_file = paws_val
        print(f"Using PAWS training data: {train_file}")
    
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print(f"Error: Training files not found.")
        print(f"Expected: {train_file} and {val_file}")
        print("Please run preprocessor.py first.")
    else:
        trainer.train(train_file, val_file)
