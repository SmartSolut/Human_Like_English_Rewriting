"""
Fix adapter_config.json to remove unsupported parameters
"""
import sys
import json
import os
import shutil
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def fix_adapter_config(model_path="./models/final"):
    """Remove unsupported parameters from adapter_config.json"""
    
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    
    if not os.path.exists(adapter_config_path):
        print(f"[ERROR] adapter_config.json not found at {adapter_config_path}")
        return False
    
    print(f"[INFO] Loading adapter config from {adapter_config_path}...")
    
    # Backup original
    backup_path = adapter_config_path + ".backup"
    shutil.copy(adapter_config_path, backup_path)
    print(f"[OK] Backup created: {backup_path}")
    
    # Load config
    with open(adapter_config_path, 'r') as f:
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
    
    print(f"[INFO] Removed {len(removed_params)} unsupported parameters:")
    for param in removed_params:
        print(f"   - {param}")
    
    # Save cleaned config
    with open(adapter_config_path, 'w') as f:
        json.dump(cleaned_config, f, indent=2)
    
    print(f"[OK] Cleaned config saved ({len(cleaned_config)} parameters)")
    print(f"[OK] Original backed up to: {backup_path}")
    
    return True

if __name__ == "__main__":
    import sys
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./models/final"
    
    print("="*60)
    print("Fixing adapter_config.json")
    print("="*60)
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

