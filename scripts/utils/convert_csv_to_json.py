"""
Convert CSV data collection file to JSON format for training
"""
import csv
import json
import sys
import os

def convert_csv_to_json(csv_file, output_file=None):
    """Convert CSV to JSON format compatible with training"""
    
    if not os.path.exists(csv_file):
        print(f"[ERROR] File not found: {csv_file}")
        return False
    
    if output_file is None:
        # Generate output filename from input
        base_name = os.path.splitext(csv_file)[0]
        output_file = f"{base_name}.json"
    
    print(f"Reading CSV from: {csv_file}")
    print(f"Output JSON: {output_file}")
    print()
    
    data = []
    errors = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
            try:
                # Validate required fields
                if not row.get('input'):
                    errors.append(f"Row {row_num}: Missing 'input' field")
                    continue
                
                if not row.get('target'):
                    errors.append(f"Row {row_num}: Missing 'target' field")
                    continue
                
                # Build data entry
                entry = {
                    'input': row['input'].strip(),
                    'target': row['target'].strip(),
                    'source': row.get('source', 'custom').strip(),
                    'tone': row.get('tone', 'casual').strip(),
                    'strength': row.get('strength', 'medium').strip(),
                }
                
                # Validate tone
                valid_tones = ['formal', 'casual', 'academic']
                if entry['tone'] not in valid_tones:
                    print(f"[WARNING] Row {row_num}: Invalid tone '{entry['tone']}', using 'casual'")
                    entry['tone'] = 'casual'
                
                # Validate strength
                valid_strengths = ['light', 'medium', 'strong']
                if entry['strength'] not in valid_strengths:
                    print(f"[WARNING] Row {row_num}: Invalid strength '{entry['strength']}', using 'medium'")
                    entry['strength'] = 'medium'
                
                # Ensure input starts with "humanize: "
                if not entry['input'].startswith('humanize: '):
                    entry['input'] = f"humanize: {entry['input']}"
                
                data.append(entry)
                
            except Exception as e:
                errors.append(f"Row {row_num}: Error - {str(e)}")
                continue
    
    if errors:
        print("[WARNINGS/ERRORS]:")
        for error in errors:
            print(f"  - {error}")
        print()
    
    if not data:
        print("[ERROR] No valid data found!")
        return False
    
    # Save JSON
    print(f"Saving {len(data)} entries to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print()
    print("="*60)
    print("Conversion Complete!")
    print("="*60)
    print(f"Total entries: {len(data)}")
    print(f"Output file: {output_file}")
    print()
    
    # Print summary by tone and strength
    tone_counts = {}
    strength_counts = {}
    for entry in data:
        tone_counts[entry['tone']] = tone_counts.get(entry['tone'], 0) + 1
        strength_counts[entry['strength']] = strength_counts.get(entry['strength'], 0) + 1
    
    print("Summary:")
    print(f"  By tone: {tone_counts}")
    print(f"  By strength: {strength_counts}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_csv_to_json.py <input.csv> [output.json]")
        print()
        print("Example:")
        print("  python convert_csv_to_json.py data/raw/data/raw/template_data_collection.csv data/raw/data/raw/book1_custom.json")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = convert_csv_to_json(csv_file, output_file)
    sys.exit(0 if success else 1)
