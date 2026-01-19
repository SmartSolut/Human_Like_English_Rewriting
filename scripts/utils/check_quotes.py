import json

# Load JSON file
with open('data/processed/mpc_cleaned_combined_test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Check first sample
sample = data[0]
print("Sample input (first 200 chars):")
print(sample['input'][:200])
print("\nSample target (first 200 chars):")
print(sample['target'][:200])
print("\nContains double quotes:", '"' in sample['input'] or '"' in sample['target'])
print("\nNote: \\\" in JSON file is just JSON escaping.")
print("When loaded in Python, it becomes regular \" quotes.")



