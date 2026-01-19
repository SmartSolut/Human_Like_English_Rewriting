"""Clean Part 3 data by removing bad patterns"""
import json
import re

# Load Part 3 data
print("Loading Part 3 data...")
with open('data/processed/splits_5_parts/train_part_3.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Original samples: {len(data)}")

# Bad patterns to filter
BAD_PATTERNS = [
    r'\bplus\s+(?:unable|to|seek|the|help|of|his|her|their|our|your|my|was|is|are|were|been|have|has|had|do|does|did|can|could|will|would|should|may|might)\b',
    r'\bplus\s+plus\b',
    r'\bplus\s+\w+\s+plus\b',
    r'\b(?:geographically|discretionary|prodigal|firmabrinac|outfitting)\b',
    r'\b\w+plus\b',  # Words ending with "plus"
    r'\bplus\s+\d+',  # "plus" followed by number
]

# Suspicious words that indicate bad paraphrasing
SUSPICIOUS_WORDS = {
    'firmabrinac', 'outfitting', 'geographical', 'discretionary', 'prodigal',
    'togetherness', 'troublesomeness', 'troubleshooting', 'troubleworthyness',
    'otherworldlinestrezziness', 'troubleshoving', 'quizziness', 'heckplus',
    'hellabrinism', 'homeplus', 'friendswear', 'lovedplus', 'friendslines',
    'builtpertaining', 'associateddication', 'associatedwith',
    'tenduousness', 'senselessness', 'latterness', 'firmness',
    'fixtures', 'outfitwear', 'accessory', 'pertaining', 'framework'
}

def is_bad_sample(item):
    """Check if a sample contains bad patterns"""
    target = item.get('target', '').lower()
    input_text = item.get('input', '').lower()
    
    # Check for excessive "plus" usage
    plus_count = target.count(' plus ') + target.count('plus ')
    if plus_count > 2:
        return True, "excessive_plus"
    
    # Check for bad patterns
    for pattern in BAD_PATTERNS:
        if re.search(pattern, target, re.IGNORECASE):
            return True, "bad_pattern"
    
    # Check for suspicious words
    for word in SUSPICIOUS_WORDS:
        if word in target:
            return True, "suspicious_word"
    
    # Check if target is too similar to input (not paraphrased)
    if target == input_text.replace('humanize:', '').strip().lower():
        return True, "identical"
    
    # Check for incomplete sentences
    if target.endswith('...') or target.endswith('[...]'):
        return True, "incomplete"
    
    return False, None

# Filter bad samples
cleaned_data = []
removed_count = 0
removed_reasons = {}

for item in data:
    is_bad, reason = is_bad_sample(item)
    if is_bad:
        removed_count += 1
        removed_reasons[reason] = removed_reasons.get(reason, 0) + 1
    else:
        cleaned_data.append(item)

print(f"\nCleaned samples: {len(cleaned_data)}")
print(f"Removed samples: {removed_count}")
print(f"\nRemoval reasons:")
for reason, count in removed_reasons.items():
    print(f"  {reason}: {count}")

# Save cleaned data
output_file = 'data/processed/splits_5_parts/train_part_3_cleaned.json'
print(f"\nSaving cleaned data to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"Cleaned data saved to {output_file}")
print(f"   Removed {removed_count}/{len(data)} samples ({removed_count/len(data)*100:.1f}%)")

