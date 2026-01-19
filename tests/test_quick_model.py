"""
Test script for the quick-trained model
"""
import os
import sys
import torch
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

def load_quick_model():
    """Load the quick-trained model"""
    model_path = "./models/final_quick"
    base_model_name = "t5-base"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return None, None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    print(f"📂 Loading model from: {model_path}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✅ Tokenizer loaded")
        
        # Load base model
        print("📥 Loading base model...")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        
        # Resize embeddings to match tokenizer vocab size
        base_model.resize_token_embeddings(len(tokenizer))
        print("✅ Base model loaded and embeddings resized")
        
        # Check if LoRA model
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print("🔧 Loading LoRA adapter...")
            model = PeftModel.from_pretrained(base_model, model_path)
            print("✅ LoRA model loaded successfully!")
        else:
            model = base_model
            print("✅ Base model loaded")
        
        model = model.to(device)
        model.eval()
        
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


def test_paraphrase(model, tokenizer, device, text, tone="casual", strength="medium"):
    """Test paraphrasing with the model"""
    # Prepare input
    input_text = f"paraphrase: {text}"
    
    # Tokenize
    inputs = tokenizer(
        input_text,
        max_length=128,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=128,
            num_beams=4,
            repetition_penalty=1.5,
            length_penalty=1.0,
            no_repeat_ngram_size=2,
            early_stopping=True,
            do_sample=False
        )
    
    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned = clean_output(generated)
    
    return cleaned


def main():
    print("="*60)
    print("Testing Quick-Trained Model")
    print("="*60)
    print()
    
    # Load model
    model, tokenizer, device = load_quick_model()
    
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    print("\n" + "="*60)
    print("Model loaded successfully! Testing...")
    print("="*60 + "\n")
    
    # Test cases
    test_texts = [
        "The weather is very nice today.",
        "I want to learn machine learning.",
        "This is a simple test sentence.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world."
    ]
    
    print("📝 Test Results:\n")
    for i, text in enumerate(test_texts, 1):
        print(f"Test {i}:")
        print(f"  Original: {text}")
        
        try:
            result = test_paraphrase(model, tokenizer, device, text)
            print(f"  Paraphrased: {result}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print()
    
    # Interactive mode
    print("="*60)
    print("Interactive Mode (type 'exit' to quit)")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("Enter text to paraphrase: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("\n🔄 Processing...")
            result = test_paraphrase(model, tokenizer, device, user_input)
            print(f"✅ Result: {result}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    main()





