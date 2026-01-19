"""
Merge book1_fixed.json with mpc_cleaned_combined_train.json
"""
import json
import os
import sys

# Fix encoding for Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def merge_training_files():
    """Merge book1 data with MPC combined training data"""
    
    # Get project root directory (parent of scripts/utils)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    book1_file = os.path.join(project_root, "data/raw/book1_fixed.json")
    mpc_file = os.path.join(project_root, "data/processed/mpc_cleaned_combined_train.json")
    output_file = os.path.join(project_root, "data/processed/mpc_cleaned_combined_train_with_book1.json")
    
    print("="*60)
    print("Merging Training Files")
    print("="*60)
    print()
    
    # Load book1 data
    if not os.path.exists(book1_file):
        print(f"[ERROR] File not found: {book1_file}")
        return False
    
    print(f"Loading {book1_file}...")
    with open(book1_file, 'r', encoding='utf-8') as f:
        book1_data = json.load(f)
    print(f"[OK] Loaded {len(book1_data)} entries from book1_fixed.json")
    
    # Load MPC data
    if not os.path.exists(mpc_file):
        print(f"[ERROR] File not found: {mpc_file}")
        return False
    
    print(f"Loading {mpc_file}...")
    print("(This may take a moment - file is large...)")
    with open(mpc_file, 'r', encoding='utf-8') as f:
        mpc_data = json.load(f)
    print(f"[OK] Loaded {len(mpc_data):,} entries from mpc_cleaned_combined_train.json")
    
    # Combine data
    print()
    print("Merging data...")
    combined_data = mpc_data + book1_data
    
    print(f"[OK] Combined: {len(mpc_data):,} + {len(book1_data)} = {len(combined_data):,} total entries")
    
    # Save merged file
    print()
    print(f"Saving merged data to {output_file}...")
    print("(This may take a moment - file is large...)")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] Saved {len(combined_data):,} entries to {output_file}")
    
    # Print summary
    print()
    print("="*60)
    print("Merge Complete!")
    print("="*60)
    print(f"Total entries: {len(combined_data):,}")
    print(f"  - MPC data: {len(mpc_data):,}")
    print(f"  - Book1 data: {len(book1_data)}")
    print(f"Output file: {output_file}")
    print()
    
    # Count by source
    source_counts = {}
    for entry in combined_data:
        source = entry.get('source', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print("Summary by source:")
    for source, count in sorted(source_counts.items()):
        print(f"  - {source}: {count:,}")
    
    return True

if __name__ == "__main__":
    success = merge_training_files()
    sys.exit(0 if success else 1)
