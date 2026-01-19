"""
Use trained model to paraphrase Book1 texts, then create training pairs
"""

import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load model
print("Loading trained model...")
model_path = "models/final"
base_model = "t5-base"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model_obj = AutoModelForSeq2SeqLM.from_pretrained(base_model)
base_model_obj.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(base_model_obj, model_path)
model.to(device)
model.eval()

print("Model loaded!")

# Load Book1 data
print("\nLoading Book1 data...")
with open("data/processed/book1_training.json", "r", encoding="utf-8") as f:
    book1_data = json.load(f)

print(f"Loaded {len(book1_data)} texts")

# Paraphrase each text
pairs = []
print("\nParaphrasing texts...")
for idx, item in enumerate(book1_data):
    original = item['input']
    print(f"\nProcessing {idx+1}/{len(book1_data)}...")
    
    # Prepare input
    input_text = f"humanize: {original}"
    inputs = tokenizer(
        input_text,
        max_length=256,
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    # Generate paraphrase
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            min_length=50,
            num_beams=3,
            early_stopping=True,
            repetition_penalty=2.5,
            length_penalty=1.1,
            do_sample=True,
            temperature=1.1,
            top_p=0.99,
            top_k=200,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    paraphrased = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    
    # Remove prompt if present
    if paraphrased.startswith("humanize:"):
        paraphrased = paraphrased.replace("humanize:", "").strip()
    
    # Create pair
    pair = {
        "input": original,
        "target": paraphrased
    }
    pairs.append(pair)
    
    print(f"  Original: {len(original)} chars")
    print(f"  Paraphrased: {len(paraphrased)} chars")

# Save
output_path = Path("data/processed/book1_paraphrased.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(pairs, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(pairs)} pairs to {output_path}")

