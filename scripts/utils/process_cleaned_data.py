"""
Process cleaned MPC data using the updated preprocessor
"""

import os
import json
import sys
from src.data.preprocessor import DataPreprocessor

def combine_cleaned_files():
    """Combine cleaned train and test files"""
    print("="*60)
    print("Combining Cleaned MPC Data")
    print("="*60)
    
    train_file = "data/raw/mpc_train_pairs_clean.json"
    test_file = "data/raw/mpc_test_pairs_clean.json"
    
    combined_data = []
    
    # Load train data
    if os.path.exists(train_file):
        print(f"\nLoading {train_file}...")
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        print(f"Loaded {len(train_data):,} train samples")
        combined_data.extend(train_data)
    else:
        print(f"Warning: {train_file} not found!")
    
    # Load test data (we'll use it for training too, or split later)
    if os.path.exists(test_file):
        print(f"\nLoading {test_file}...")
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"Loaded {len(test_data):,} test samples")
        # Add test data to combined (will be split by preprocessor)
        combined_data.extend(test_data)
    else:
        print(f"Warning: {test_file} not found!")
    
    if not combined_data:
        print("\nError: No data found! Make sure cleaned files exist.")
        return None
    
    # Save combined file
    output_file = "data/raw/mpc_cleaned_combined.json"
    print(f"\nSaving combined data to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(combined_data):,} total samples")
    return output_file

def main():
    print("="*60)
    print("Processing Cleaned MPC Data with Enhanced Preprocessor")
    print("="*60)
    
    # Step 1: Combine cleaned files
    combined_file = combine_cleaned_files()
    if not combined_file:
        return
    
    # Step 2: Process with preprocessor
    print("\n" + "="*60)
    print("Processing with Enhanced Preprocessor")
    print("="*60)
    print("\nFiltering criteria:")
    print("  - Empty quotes: EXCLUDED")
    print("  - Incomplete sentences: EXCLUDED")
    print("  - Broken encoding: EXCLUDED")
    print("  - Match distance < 0.15: EXCLUDED")
    print("  - Less than 120 words: EXCLUDED")
    print("  - Prompt: 'humanize: {text}'")
    print("="*60)
    
    preprocessor = DataPreprocessor()
    train, val, test = preprocessor.process_file(combined_file)
    
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"\nFinal datasets:")
    print(f"  Train: {len(train):,} samples")
    print(f"  Validation: {len(val):,} samples")
    print(f"  Test: {len(test):,} samples")
    print(f"\nFiles saved in: {preprocessor.processed_data_dir}")

if __name__ == "__main__":
    main()



