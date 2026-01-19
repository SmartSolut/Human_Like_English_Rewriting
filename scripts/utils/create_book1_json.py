"""
Create JSON from Book1.xlsx
Assuming: Column 0 = AI text, Column 1 = Human paraphrased text (or label)
"""

import pandas as pd
import json
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load Excel
print("Loading Book1.xlsx...")
df = pd.read_excel('Book1.xlsx')

print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nColumn 1 (labels) value counts:")
print(df.iloc[:, 1].value_counts())

# Check if column 1 is just labels or actual text
sample_col1 = str(df.iloc[0, 1])
print(f"\nSample Column 1 value: '{sample_col1}' (length: {len(sample_col1)})")

# If column 1 is just labels, we need to understand the structure
# User said: "نص ذكاء ومعاد صياغتة بشريا" = AI text and human paraphrased text

# Let's check: maybe column 0 is the human text and we need to find AI text?
# Or column 0 is AI and column 1 label indicates it's human?

# Based on user's description: "نص ذكاء ومعاد صياغتة بشريا"
# This means: AI text and human paraphrased text
# So we need pairs where:
# - input = AI text
# - target = human paraphrased text

# Since column 1 seems to be just labels, maybe the structure is:
# - Column 0 = text (could be AI or human)
# - Column 1 = label (Ai/human)

# Let's create pairs assuming:
# - If label is "Ai" or "AI": this is the AI text (input)
# - If label is "human" or "Human": this is the human text (target)

pairs = []
ai_texts = []
human_texts = []

for idx, row in df.iterrows():
    text = str(row.iloc[0])
    label = str(row.iloc[1]).strip().lower()
    
    if label in ['ai', 'ai ']:
        ai_texts.append((idx, text))
    elif label in ['human', 'human ']:
        human_texts.append((idx, text))

print(f"\nFound {len(ai_texts)} AI texts")
print(f"Found {len(human_texts)} human texts")

# If we have matching pairs, create them
# Otherwise, assume alternating or some other pattern

# Simple approach: if labels alternate, pair them
if len(ai_texts) == len(human_texts):
    print("\nPairing AI and human texts...")
    for (ai_idx, ai_text), (human_idx, human_text) in zip(ai_texts, human_texts):
        pairs.append({
            "input": ai_text.strip(),
            "target": human_text.strip()
        })
else:
    print("\nWarning: Unequal counts. Creating pairs from first column only...")
    # Maybe all texts in column 0 are AI, and we need human versions?
    # Or maybe they're already paired in order?
    # Let's assume they're already in pairs (alternating or sequential)
    for i in range(0, len(df) - 1, 2):
        if i + 1 < len(df):
            text1 = str(df.iloc[i, 0]).strip()
            text2 = str(df.iloc[i + 1, 0]).strip()
            label1 = str(df.iloc[i, 1]).strip().lower()
            label2 = str(df.iloc[i + 1, 1]).strip().lower()
            
            # Determine which is AI and which is human
            if 'ai' in label1:
                pairs.append({"input": text1, "target": text2})
            elif 'ai' in label2:
                pairs.append({"input": text2, "target": text1})
            elif 'human' in label1:
                pairs.append({"input": text2, "target": text1})
            elif 'human' in label2:
                pairs.append({"input": text1, "target": text2})
            else:
                # Default: first is AI, second is human
                pairs.append({"input": text1, "target": text2})

print(f"\nCreated {len(pairs)} pairs")

# Save
output_path = Path("data/processed/book1_final.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(pairs, f, ensure_ascii=False, indent=2)

print(f"Saved to: {output_path}")

# Show sample
if pairs:
    print("\nSample pair:")
    print(f"Input (first 200 chars): {pairs[0]['input'][:200]}...")
    print(f"Target (first 200 chars): {pairs[0]['target'][:200]}...")

