"""Clean ALL data parts (1-5) more aggressively"""
import json
import re
import os
from pathlib import Path

# Bad patterns to filter
BAD_PATTERNS = [
    r'\bplus\s+(?:unable|to|seek|the|help|of|his|her|their|our|your|my|was|is|are|were|been|have|has|had|do|does|did|can|could|will|would|should|may|might)\b',
    r'\bplus\s+plus\b',
    r'\bplus\s+\w+\s+plus\b',
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
    'fixtures', 'outfitwear', 'accessory', 'pertaining', 'framework',
    'particular', 'characteristic', 'associated', 'front', 'friends',
    'troublesome', 'troubleshooting', 'troublesomeness', 'troubleworthyness',
}

def is_bad_sample(item):
    """Check if a sample contains bad patterns"""
    target = item.get('target', '').lower()
    input_text = item.get('input', '').lower()
    
    # Check for excessive "plus" usage
    plus_count = target.count(' plus ') + target.count('plus ')
    if plus_count > 1:  # More aggressive: allow max 1 "plus"
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
    input_clean = input_text.replace('humanize:', '').strip().lower()
    if target == input_clean:
        return True, "identical"
    
    # Check for incomplete sentences
    if target.endswith('...') or target.endswith('[...]'):
        return True, "incomplete"
    
    # Check for very short targets
    if len(target.split()) < 20:
        return True, "too_short"
    
    return False, None

def clean_data_file(input_file, output_file):
    """Clean a single data file"""
    print(f"\nProcessing {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = len(data)
    
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
    
    print(f"  Original: {original_count}")
    print(f"  Cleaned: {len(cleaned_data)}")
    print(f"  Removed: {removed_count} ({removed_count/original_count*100:.1f}%)")
    if removed_reasons:
        print(f"  Reasons: {dict(removed_reasons)}")
    
    # Save cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    return len(cleaned_data), removed_count

# Main
if __name__ == "__main__":
    data_dir = Path("data/processed/splits_5_parts")
    
    print("="*60)
    print("Cleaning ALL data parts (1-5) - IN PLACE")
    print("="*60)
    print("Files will be cleaned directly in splits_5_parts")
    print("No backup copies will be created")
    print("="*60)
    
    total_original = 0
    total_cleaned = 0
    total_removed = 0
    
    for part_num in range(1, 6):
        input_file = data_dir / f"train_part_{part_num}.json"
        # Clean in place - same file, same name
        output_file = input_file
        
        if input_file.exists():
            # Create temporary file for cleaning
            temp_file = data_dir / f"train_part_{part_num}_temp.json"
            cleaned_count, removed_count = clean_data_file(input_file, temp_file)
            total_original += cleaned_count + removed_count
            total_cleaned += cleaned_count
            total_removed += removed_count
            
            # Replace original with cleaned version
            if temp_file.exists():
                import shutil
                shutil.move(str(temp_file), str(output_file))
                print(f"  Replaced: {input_file.name}")
        else:
            print(f"\nWarning: {input_file} not found, skipping...")
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Total original: {total_original}")
    print(f"  Total cleaned: {total_cleaned}")
    print(f"  Total removed: {total_removed} ({total_removed/total_original*100:.1f}%)")
    print(f"\nFiles cleaned in place: {data_dir}")
    print("="*60)

