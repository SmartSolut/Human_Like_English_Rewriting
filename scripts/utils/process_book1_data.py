"""
Process Book1 data and prepare for training
"""

import json
from pathlib import Path
from src.data.preprocessor import DataPreprocessor

# Load data (use training pairs)
print("Loading Book1 training pairs...")
with open("data/processed/book1_training_pairs.json", "r", encoding="utf-8") as f:
    book1_data = json.load(f)

print(f"Loaded {len(book1_data)} pairs from Book1")

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Process data
print("\nProcessing data...")
processed_data = preprocessor.preprocess(book1_data)

print(f"\nProcessed {len(processed_data)} pairs after filtering")

# Split data
print("\nSplitting data...")
train_data, val_data, test_data = preprocessor.split_data(processed_data)

print(f"Train: {len(train_data)}")
print(f"Val: {len(val_data)}")
print(f"Test: {len(test_data)}")

# Save processed data
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

train_path = output_dir / "book1_train.json"
val_path = output_dir / "book1_val.json"
test_path = output_dir / "book1_test.json"

with open(train_path, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(val_path, "w", encoding="utf-8") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

with open(test_path, "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print(f"\nSaved:")
print(f"  Train: {train_path}")
print(f"  Val: {val_path}")
print(f"  Test: {test_path}")

