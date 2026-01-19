"""
Fix adapter_config.json in any directory
Run this script from the project root or specify the model path
"""
import sys
import json
import os
import shutil
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def fix_adapter_config(model_path):
    """Remove unsupported parameters from adapter_config.json"""
    
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    
    if not os.path.exists(adapter_config_path):
        print(f"[ERROR] adapter_config.json not found at {adapter_config_path}")
        return False
    
    print(f"[INFO] Loading adapter config from {adapter_config_path}...")
    
    # Backup original
    backup_path = adapter_config_path + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy(adapter_config_path, backup_path)
        print(f"[OK] Backup created: {backup_path}")
    else:
        print(f"[INFO] Backup already exists: {backup_path}")
    
    # Load config
    with open(adapter_config_path, 'r', encoding='utf-8') as f:
        adapter_config = json.load(f)
    
    print(f"[INFO] Original config has {len(adapter_config)} parameters")
    
    # List of essential parameters to keep (based on current PEFT version)
    # These are the parameters actually supported by LoraConfig
    essential_params = {
        'base_model_name_or_path',
        'bias',
        'fan_in_fan_out',
        'inference_mode',
        'init_lora_weights',
        'lora_alpha',
        'lora_dropout',
        'modules_to_save',
        'peft_type',
        'r',
        'target_modules',
        'task_type',
        # Optional parameters that might be present
        'auto_mapping',
        'revision',
        'layers_to_transform',
        'layers_pattern',
        'rank_pattern',
        'alpha_pattern',
        'megatron_config',
        'megatron_core',
        'loftq_config',
    }
    
    # Remove unsupported parameters
    removed_params = []
    cleaned_config = {}
    
    for k, v in adapter_config.items():
        if k in essential_params:
            cleaned_config[k] = v
        else:
            removed_params.append(k)
    
    if removed_params:
        print(f"[INFO] Removed {len(removed_params)} unsupported parameters:")
        for param in removed_params:
            print(f"   - {param}")
        
        # Save cleaned config
        with open(adapter_config_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_config, f, indent=2)
        
        print(f"[OK] Cleaned config saved ({len(cleaned_config)} parameters)")
    else:
        print(f"[OK] Config is already clean ({len(cleaned_config)} parameters)")
    
    return True

if __name__ == "__main__":
    # Default to models/final in current directory
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Try to find models/final in current directory or parent
        current_dir = Path.cwd()
        possible_paths = [
            current_dir / "models" / "final",
            current_dir.parent / "models" / "final",
            Path("models") / "final",
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = str(path)
                break
        
        if not model_path:
            print("[ERROR] Could not find models/final directory")
            print("Usage: python fix_adapter_config_anywhere.py [path_to_models/final]")
            sys.exit(1)
    
    print("="*60)
    print("Fixing adapter_config.json")
    print("="*60)
    print(f"Model path: {model_path}")
    print()
    
    if fix_adapter_config(model_path):
        print()
        print("="*60)
        print("[OK] Fix complete!")
        print("="*60)
    else:
        print()
        print("="*60)
        print("[ERROR] Fix failed!")
        print("="*60)
        sys.exit(1)
