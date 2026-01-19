"""
Check training status and history
"""
import os
import json
from pathlib import Path

print("="*60)
print("Training Status Check")
print("="*60)

# Check final model
final_path = Path("models/final")
if final_path.exists():
    adapter_config = final_path / "adapter_config.json"
    if adapter_config.exists():
        print("\n[OK] Final model exists (LoRA Adapter)")
        with open(adapter_config, 'r') as f:
            config = json.load(f)
        print(f"   Base model: {config.get('base_model_name_or_path', 'N/A')}")
        print(f"   LoRA rank (r): {config.get('r', 'N/A')}")
        print(f"   LoRA alpha: {config.get('lora_alpha', 'N/A')}")
    else:
        print("\n[INFO] Final model exists but no adapter config")

# Check checkpoints
checkpoints_path = Path("models/checkpoints")
if checkpoints_path.exists():
    checkpoints = [d for d in checkpoints_path.iterdir() if d.is_dir() and 'checkpoint' in d.name]
    print(f"\n[INFO] Found {len(checkpoints)} checkpoints:")
    for cp in sorted(checkpoints, key=lambda x: int(x.name.split('-')[-1]) if x.name.split('-')[-1].isdigit() else 0):
        trainer_state = cp / "trainer_state.json"
        if trainer_state.exists():
            try:
                with open(trainer_state, 'r') as f:
                    state = json.load(f)
                epoch = state.get('epoch', 'N/A')
                step = state.get('global_step', 'N/A')
                print(f"   - {cp.name}: step={step}, epoch={epoch:.2f}" if isinstance(epoch, float) else f"   - {cp.name}: step={step}, epoch={epoch}")
            except:
                print(f"   - {cp.name}: (unable to read state)")

# Check if training parts exist
splits_dir = Path("data/processed/splits_5_parts")
if splits_dir.exists():
    parts = list(splits_dir.glob("train_part_*.json"))
    print(f"\n[INFO] Found {len(parts)} training parts:")
    for part in sorted(parts):
        size = part.stat().st_size / (1024*1024)  # MB
        print(f"   - {part.name}: {size:.1f} MB")

print("\n" + "="*60)
print("Recommendation:")
print("="*60)
print("If you retrained Part 1 and model works well:")
print("  - Option 1: Continue training Parts 2-5 for consistency")
print("  - Option 2: Test current model first, then decide")
print("  - Option 3: Use current model if performance is acceptable")
print("="*60)

