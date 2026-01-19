"""
Fix all file paths after project reorganization
"""
import os
import re
from pathlib import Path

# Get project root (parent of scripts/utils)
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

FIXES = [
    # Batch files that need path fixes
    {
        "file": PROJECT_ROOT / "scripts" / "train_with_book1.bat",
        "fixes": [
            ('python check_gpu.py', 'python scripts\\utils\\check_gpu.py'),
            ('python test_model_part1.py', 'python tests\\test_model_part1.py'),
        ]
    },
    {
        "file": PROJECT_ROOT / "scripts" / "verify_gpu_setup.bat",
        "fixes": [
            ('python check_gpu.py', 'python scripts\\utils\\check_gpu.py'),
        ]
    },
    {
        "file": PROJECT_ROOT / "scripts" / "install_pytorch_cuda.bat",
        "fixes": [
            ('python check_gpu.py', 'python scripts\\utils\\check_gpu.py'),
        ]
    },
    # Python files that need path fixes
    {
        "file": PROJECT_ROOT / "scripts" / "utils" / "fix_csv_properly.py",
        "fixes": [
            ('"book1_fixed.json"', '"data/raw/book1_fixed.json"'),
            ('"_template_data_collection.csv"', '"data/raw/_template_data_collection.csv"'),
        ]
    },
    {
        "file": PROJECT_ROOT / "scripts" / "utils" / "fix_and_convert_csv.py",
        "fixes": [
            ('"_template_data_collection.csv"', '"data/raw/_template_data_collection.csv"'),
            ('"book1_fixed.json"', '"data/raw/book1_fixed.json"'),
        ]
    },
    {
        "file": PROJECT_ROOT / "scripts" / "utils" / "convert_csv_to_json.py",
        "fixes": [
            ('template_data_collection.csv', 'data/raw/template_data_collection.csv'),
            ('book1_custom.json', 'data/raw/book1_custom.json'),
        ]
    },
]

def fix_file(file_path, fixes):
    """Fix paths in a single file"""
    if not file_path.exists():
        print(f"[SKIP] File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        for pattern, replacement in fixes:
            content = content.replace(pattern, replacement)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8', newline='\r\n' if file_path.suffix == '.bat' else '\n') as f:
                f.write(content)
            print(f"[OK] Fixed: {file_path.name}")
            return True
        else:
            print(f"[INFO] No changes needed: {file_path.name}")
            return True
    except Exception as e:
        print(f"[ERROR] Failed to fix {file_path.name}: {e}")
        return False

def main():
    print("="*60)
    print("FIXING FILE PATHS AFTER REORGANIZATION")
    print("="*60)
    print()
    print(f"Project root: {PROJECT_ROOT}")
    print()
    
    fixed = 0
    errors = 0
    
    for fix_config in FIXES:
        file_path = fix_config["file"]
        fixes = fix_config["fixes"]
        
        if fix_file(file_path, fixes):
            fixed += 1
        else:
            errors += 1
    
    print()
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Fixed files: {fixed}")
    print(f"Errors: {errors}")
    print()
    
    if errors == 0:
        print("[SUCCESS] All paths fixed successfully!")
    else:
        print("[WARNING] Some files had errors. Please check above.")
    
    return errors == 0

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
