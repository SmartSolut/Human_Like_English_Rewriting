"""
Delete all trained models and checkpoints to start training from scratch
"""
import os
import shutil
import sys

def delete_directory(path, name):
    """Delete a directory and all its contents"""
    if os.path.exists(path):
        try:
            print(f"Deleting {name}...")
            shutil.rmtree(path)
            print(f"[OK] Deleted {name}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to delete {name}: {e}")
            return False
    else:
        print(f"[INFO] {name} does not exist (already deleted)")
        return True

def main():
    print("="*60)
    print("RESET TRAINING ENVIRONMENT")
    print("="*60)
    print()
    print("Deleting all trained models and checkpoints...")
    print()
    
    success = True
    
    # Delete models/final
    final_path = "models/final"
    success = delete_directory(final_path, "models/final") and success
    
    # Delete models/checkpoints
    checkpoints_path = "models/checkpoints"
    success = delete_directory(checkpoints_path, "models/checkpoints") and success
    
    # Delete data/cache
    cache_path = "data/cache"
    success = delete_directory(cache_path, "data/cache") and success
    
    print()
    print("="*60)
    if success:
        print("Reset Complete!")
        print("="*60)
        print()
        print("All trained models and checkpoints have been deleted.")
        print("You can now start training from scratch using:")
        print("  scripts\\START_TRAINING.bat")
        print("  OR")
        print("  scripts\\train_with_book1.bat")
    else:
        print("Reset Complete with Errors!")
        print("="*60)
        print()
        print("Some files could not be deleted. Please check manually.")
    
    print()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
