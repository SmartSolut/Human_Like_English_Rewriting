"""
Test script for the model trained on Part 1/5
Tests the trained LoRA model for paraphrase generation
"""
import os
import sys
import torch
import re

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel


def load_trained_model(model_path=None):
    """Load the trained model from Part 1"""
    if model_path is None:
        # Try final model first, then latest checkpoint
        model_path = "./models/final"
        if not os.path.exists(model_path):
            # Try to find latest checkpoint
            checkpoints_dir = "./models/checkpoints"
            if os.path.exists(checkpoints_dir):
                checkpoints = [d for d in os.listdir(checkpoints_dir) 
                             if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoints_dir, d))]
                if checkpoints:
                    # Sort by checkpoint number
                    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                    model_path = os.path.join(checkpoints_dir, checkpoints[-1])
                    print(f"⚠️  Final model not found, using latest checkpoint: {model_path}")
    
    base_model_name = "t5-base"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print(f"\nAvailable checkpoints:")
        checkpoints_dir = "./models/checkpoints"
        if os.path.exists(checkpoints_dir):
            checkpoints = [d for d in os.listdir(checkpoints_dir) 
                         if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoints_dir, d))]
            for cp in sorted(checkpoints, key=lambda x: int(x.split("-")[1])):
                print(f"  - {os.path.join(checkpoints_dir, cp)}")
        return None, None, None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print("Loading Trained Model (Part 1/5)")
    print("="*60)
    print(f"📱 Device: {device}")
    print(f"📂 Model path: {model_path}")
    print(f"🔧 Base model: {base_model_name}")
    print()
    
    try:
        # Load tokenizer
        print("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✅ Tokenizer loaded (vocab size: {len(tokenizer)})")
        
        # Load base model
        print("📥 Loading base model...")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        
        # Resize embeddings to match tokenizer vocab size (IMPORTANT!)
        print("🔧 Resizing embeddings to match tokenizer...")
        base_model.resize_token_embeddings(len(tokenizer))
        print("✅ Base model loaded and embeddings resized")
        
        # Check if LoRA model
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            print("🔧 Loading LoRA adapter...")
            model = PeftModel.from_pretrained(base_model, model_path)
            print("✅ LoRA model loaded successfully!")
        else:
            model = base_model
            print("✅ Base model loaded (no LoRA adapter found)")
        
        # Move to device
        model = model.to(device)
        model.eval()
        
        # Print GPU memory if using CUDA
        if device.type == 'cuda':
            gpu_mem = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"✅ Model on GPU (Memory: {gpu_mem:.2f} GB)")
        
        print()
        return model, tokenizer, device
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def clean_output(text):
    """Clean the generated output"""
    # Remove control tokens and prefixes
    text = re.sub(r'<tone=[^>]+>', '', text)
    text = re.sub(r'<strength=[^>]+>', '', text)
    text = re.sub(r'^paraphrase:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^rewrite:\s*', '', text, flags=re.IGNORECASE)
    
    # Clean excessive dots/dashes
    text = re.sub(r'\.{3,}', '...', text)
    text = re.sub(r'-{3,}', '---', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    return text.strip()


def test_paraphrase(model, tokenizer, device, text, tone="academic", strength="medium", max_length=128, num_beams=4):
    """Test paraphrasing with the model"""
    # Prepare input - use same format as training (with tone and strength tokens)
    input_text = f"paraphrase <tone={tone}> <strength={strength}>: {text}"
    
    # Tokenize
    inputs = tokenizer(
        input_text,
        max_length=256,  # max_source_length from training
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    # Generate with different parameters to encourage variation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,  # Use **inputs to pass input_ids and attention_mask
            max_length=max_length,
            min_length=10,  # Minimum length to encourage generation
            num_beams=num_beams,
            repetition_penalty=2.0,  # Higher penalty to avoid repetition
            length_penalty=1.2,  # Slightly favor longer outputs
            no_repeat_ngram_size=3,  # Prevent 3-gram repetition
            early_stopping=True,
            do_sample=True,  # Enable sampling for more variation
            temperature=0.8,  # Lower temperature for more focused generation
            top_p=0.9,  # Nucleus sampling
            top_k=50  # Top-k sampling
        )
    
    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned = clean_output(generated)
    
    return cleaned


def main():
    print("="*60)
    print("Testing Model Trained on Part 1/5")
    print("="*60)
    print()
    
    # Allow custom model path from command line
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    # Load model
    model, tokenizer, device = load_trained_model(model_path)
    
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    print("="*60)
    print("Model loaded successfully! Testing...")
    print("="*60)
    print()
    
    # Test cases
    test_texts = [
        "The weather is very nice today.",
        "I want to learn machine learning.",
        "This is a simple test sentence.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Students need to study hard to succeed.",
        "The book was very interesting and well written.",
    ]
    
    print("📝 Test Results:\n")
    for i, text in enumerate(test_texts, 1):
        print(f"Test {i}:")
        print(f"  Original:    {text}")
        
        try:
            result = test_paraphrase(model, tokenizer, device, text)
            print(f"  Paraphrased: {result}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Additional test cases
    print("="*60)
    print("Additional Test Cases")
    print("="*60)
    print()
    
    additional_tests = [
        "Machine learning is a subset of artificial intelligence.",
        "The students worked hard on their assignments.",
        "Technology has changed the way we communicate.",
    ]
    
    for i, text in enumerate(additional_tests, 1):
        print(f"Additional Test {i}:")
        print(f"  Original:    {text}")
        
        try:
            result = test_paraphrase(model, tokenizer, device, text)
            print(f"  Paraphrased: {result}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print("="*60)
    print("✅ Testing completed!")
    print("="*60)
    print("\n💡 To test more examples, run this script and modify the test_texts list,")
    print("   or use the API: python -m uvicorn src.api.main:app --reload")


if __name__ == "__main__":
    main()

