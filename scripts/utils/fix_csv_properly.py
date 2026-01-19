"""
Properly fix and convert CSV with multi-line text fields
"""
import json
import sys
import re

def fix_csv_properly(csv_file, output_file=None):
    """Fix CSV with proper handling of multi-line quoted fields"""
    
    if output_file is None:
        output_file = "data/raw/book1_fixed.json"
    
    print(f"Reading CSV: {csv_file}")
    print(f"Output: {output_file}")
    print()
    
    # Read file with proper encoding
    with open(csv_file, 'r', encoding='latin-1', errors='ignore') as f:
        content = f.read()
    
    # Split into lines but preserve quoted multi-line fields
    lines = []
    current_line = ""
    in_quotes = False
    
    for char in content:
        if char == '"':
            # Toggle quote state
            if current_line and current_line[-1] == '\\':
                # Escaped quote
                current_line += char
            else:
                in_quotes = not in_quotes
                current_line += char
        elif char == '\n' and not in_quotes:
            # End of line (only if not in quotes)
            if current_line.strip():
                lines.append(current_line)
            current_line = ""
        else:
            current_line += char
    
    if current_line.strip():
        lines.append(current_line)
    
    print(f"Total lines read: {len(lines)}")
    
    # Parse CSV
    import csv
    from io import StringIO
    
    # Reconstruct CSV content
    csv_content = '\n'.join(lines)
    
    # Parse using csv module
    reader = csv.DictReader(StringIO(csv_content))
    
    data = []
    errors = []
    
    for row_num, row in enumerate(reader, start=2):
        try:
            # Get input and target
            input_text = row.get('input', '').strip()
            target_text = row.get('target', '').strip()
            
            # Validate
            if not input_text or not target_text:
                errors.append(f"Row {row_num}: Missing input or target")
                continue
            
            # Fix "humaniz:" to "humanize:"
            if input_text.startswith('humaniz:'):
                input_text = input_text.replace('humaniz:', 'humanize: ', 1)
            
            # Ensure starts with "humanize: "
            if not input_text.startswith('humanize: '):
                if input_text.startswith('humanize:'):
                    input_text = input_text.replace('humanize:', 'humanize: ', 1)
                else:
                    input_text = f"humanize: {input_text}"
            
            # Clean text
            input_text = re.sub(r'\s+', ' ', input_text).strip()
            target_text = re.sub(r'\s+', ' ', target_text).strip()
            
            # Skip if too short (probably broken)
            if len(input_text) < 20 or len(target_text) < 20:
                errors.append(f"Row {row_num}: Text too short (probably broken)")
                continue
            
            # Build entry
            entry = {
                'input': input_text,
                'target': target_text,
                'source': row.get('source', 'book1').strip() or 'book1',
                'tone': row.get('tone', 'casual').strip() or 'casual',
                'strength': row.get('strength', 'medium').strip() or 'medium',
            }
            
            # Validate tone
            if entry['tone'] not in ['formal', 'casual', 'academic']:
                entry['tone'] = 'casual'
            
            # Validate strength
            if entry['strength'] not in ['light', 'medium', 'strong']:
                entry['strength'] = 'medium'
            
            data.append(entry)
            
        except Exception as e:
            errors.append(f"Row {row_num}: Error - {str(e)}")
            continue
    
    if errors:
        print(f"[WARNINGS]: {len(errors)} issues found")
        for error in errors[:5]:  # Show first 5
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
        print()
    
    if not data:
        print("[ERROR] No valid data!")
        return False
    
    # Save JSON
    print(f"Saving {len(data)} valid entries...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print()
    print("="*60)
    print("Conversion Complete!")
    print("="*60)
    print(f"Valid entries: {len(data)}")
    print(f"Output: {output_file}")
    print()
    
    # Summary
    tone_counts = {}
    for entry in data:
        tone_counts[entry['tone']] = tone_counts.get(entry['tone'], 0) + 1
    
    print(f"Summary by tone: {tone_counts}")
    
    return True

if __name__ == "__main__":
    csv_file = "data/raw/_template_data_collection.csv"
    output_file = "data/raw/book1_fixed.json"
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    success = fix_csv_properly(csv_file, output_file)
    sys.exit(0 if success else 1)
