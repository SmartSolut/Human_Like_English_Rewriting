"""
Prepare Book1 data for training
Assumes: Column 0 = text, Column 1 = label (Ai/human)
For AI texts: use as input, need to create human version (or use model)
For human texts: these are the targets we want
"""

import pandas as pd
import json
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load Excel
print("Loading Book1.xlsx...")
df = pd.read_excel('Book1.xlsx')

print(f"Total rows: {len(df)}")

# Separate AI and human texts
ai_texts = []
human_texts = []

for idx, row in df.iterrows():
    text = str(row.iloc[0]).strip()
    label = str(row.iloc[1]).strip().lower()
    
    if 'ai' in label:
        ai_texts.append(text)
    elif 'human' in label:
        human_texts.append(text)

print(f"\nAI texts: {len(ai_texts)}")
print(f"Human texts: {len(human_texts)}")

# Strategy: Use human texts as targets
# For AI texts, we'll use them as input and the model will learn to paraphrase them
# OR: pair them if we can match them

pairs = []

# If counts match, pair them
if len(ai_texts) == len(human_texts):
    print("\nPairing AI and human texts...")
    for ai_text, human_text in zip(ai_texts, human_texts):
        pairs.append({
            "input": ai_text,
            "target": human_text
        })
else:
    print(f"\nCounts don't match. Creating pairs from available texts...")
    # Use all texts - assume human texts are the "target" versions
    # and AI texts are the "input" versions
    # For training, we want: input (AI) -> target (human)
    
    # Create pairs: use AI texts as input, human texts as target
    # If we have more AI than human, some AI texts won't have pairs
    # If we have more human than AI, some human texts won't have pairs
    
    min_count = min(len(ai_texts), len(human_texts))
    
    for i in range(min_count):
        pairs.append({
            "input": ai_texts[i],
            "target": human_texts[i]
        })
    
    print(f"Created {len(pairs)} pairs from {min_count} matching texts")
    
    # For remaining texts, we could:
    # 1. Use them as self-pairs (input = target) - not ideal
    # 2. Skip them
    # 3. Use model to generate pairs
    
    if len(ai_texts) > len(human_texts):
        print(f"Note: {len(ai_texts) - len(human_texts)} AI texts don't have human pairs")
    elif len(human_texts) > len(ai_texts):
        print(f"Note: {len(human_texts) - len(ai_texts)} human texts don't have AI pairs")

print(f"\nTotal pairs created: {len(pairs)}")

# Save
output_path = Path("data/processed/book1_training_pairs.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(pairs, f, ensure_ascii=False, indent=2)

print(f"Saved to: {output_path}")

# Show samples
if pairs:
    print("\nSample pairs:")
    for i in range(min(3, len(pairs))):
        print(f"\nPair {i+1}:")
        print(f"  Input (AI) - first 150 chars: {pairs[i]['input'][:150]}...")
        print(f"  Target (Human) - first 150 chars: {pairs[i]['target'][:150]}...")

