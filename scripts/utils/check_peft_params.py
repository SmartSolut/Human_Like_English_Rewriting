"""Check supported PEFT parameters"""
from peft import LoraConfig
import inspect

sig = inspect.signature(LoraConfig.__init__)
print("Supported LoraConfig parameters:")
for param in sig.parameters:
    print(f"  - {param}")

