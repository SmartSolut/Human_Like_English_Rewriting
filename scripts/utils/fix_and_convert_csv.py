"""
Fix CSV issues and convert to JSON
Handles encoding issues and fixes common problems
"""
import csv
import json
import sys
import os
import re

def fix_and_convert_csv(csv_file, output_file=None):
    """Fix CSV issues and convert to JSON"""
    
    if not os.path.exists(csv_file):
        print(f"[ERROR] File not found: {csv_file}")
        return False
    
    if output_file is None:
        base_name = os.path.splitext(csv_file)[0]
        output_file = f"{base_name}_fixed.json"
    
    print(f"Reading CSV from: {csv_file}")
    print(f"Output JSON: {output_file}")
    print()
    
    # Try different encodings
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1256', 'windows-1252']
    data_lines = None
    used_encoding = None
    
    for enc in encodings:
        try:
            with open(csv_file, 'r', encoding=enc) as f:
                data_lines = f.readlines()
                used_encoding = enc
                print(f"Successfully read file with encoding: {enc}")
                break
        except Exception as e:
            continue
    
    if data_lines is None:
        print("[ERROR] Could not read file with any encoding!")
        return False
    
    # Parse CSV manually to handle multi-line fields
    print("Parsing CSV data...")
    data = []
    errors = []
    
    # Read header
    if not data_lines:
        print("[ERROR] File is empty!")
        return False
    
    header_line = data_lines[0].strip()
    headers = [h.strip() for h in header_line.split(',')]
    
    # Fix header if needed
    if 'input' not in [h.lower() for h in headers]:
        print("[ERROR] 'input' column not found in header!")
        return False
    
    # Parse data rows
    current_row = {}
    current_field = None
    in_quotes = False
    row_num = 1
    
    for line_idx, line in enumerate(data_lines[1:], start=2):
        line = line.rstrip('\n\r')
        
        # Handle multi-line fields in quotes
        if '"' in line:
            # Simple CSV parsing for quoted fields
            parts = []
            current_part = ""
            in_quotes = False
            
            i = 0
            while i < len(line):
                char = line[i]
                
                if char == '"':
                    if i + 1 < len(line) and line[i + 1] == '"':
                        # Escaped quote
                        current_part += '"'
                        i += 2
                    else:
                        # Toggle quote state
                        in_quotes = not in_quotes
                        i += 1
                elif char == ',' and not in_quotes:
                    parts.append(current_part)
                    current_part = ""
                    i += 1
                else:
                    current_part += char
                    i += 1
            
            if current_part:
                parts.append(current_part)
            
            # Ensure we have enough fields
            while len(parts) < len(headers):
                parts.append("")
            
            # Create row dictionary
            row = {}
            for i, header in enumerate(headers):
                row[header] = parts[i].strip('"').strip() if i < len(parts) else ""
            
            # Fix common issues
            if 'input' in row:
                input_text = row['input']
                # Fix "humaniz:" to "humanize:"
                if input_text.startswith('humaniz:'):
                    input_text = input_text.replace('humaniz:', 'humanize:', 1)
                    row['input'] = input_text
                
                # Ensure it starts with "humanize: "
                if not input_text.startswith('humanize: '):
                    if input_text.startswith('humanize:'):
                        input_text = input_text.replace('humanize:', 'humanize: ', 1)
                    else:
                        input_text = f"humanize: {input_text}"
                    row['input'] = input_text
            
            # Validate required fields
            if not row.get('input') or not row.get('target'):
                errors.append(f"Row {row_num}: Missing input or target")
                row_num += 1
                continue
            
            # Build entry
            entry = {
                'input': row['input'].strip(),
                'target': row['target'].strip(),
                'source': row.get('source', 'book1').strip() or 'book1',
                'tone': row.get('tone', 'casual').strip() or 'casual',
                'strength': row.get('strength', 'medium').strip() or 'medium',
            }
            
            # Validate tone
            valid_tones = ['formal', 'casual', 'academic']
            if entry['tone'] not in valid_tones:
                entry['tone'] = 'casual'
            
            # Validate strength
            valid_strengths = ['light', 'medium', 'strong']
            if entry['strength'] not in valid_strengths:
                entry['strength'] = 'medium'
            
            # Clean up text (remove special characters that might cause issues)
            entry['input'] = re.sub(r'[^\x20-\x7E\u00A0-\uFFFF\n\r\t]', '', entry['input'])
            entry['target'] = re.sub(r'[^\x20-\x7E\u00A0-\uFFFF\n\r\t]', '', entry['target'])
            
            data.append(entry)
            row_num += 1
            
        else:
            # Simple comma-separated line
            parts = [p.strip().strip('"') for p in line.split(',')]
            if len(parts) >= 2:
                row = {}
                for i, header in enumerate(headers):
                    row[header] = parts[i] if i < len(parts) else ""
                
                # Fix and validate same as above
                if row.get('input') and row.get('target'):
                    if row['input'].startswith('humaniz:'):
                        row['input'] = row['input'].replace('humaniz:', 'humanize: ', 1)
                    if not row['input'].startswith('humanize: '):
                        row['input'] = f"humanize: {row['input']}"
                    
                    entry = {
                        'input': row['input'].strip(),
                        'target': row['target'].strip(),
                        'source': row.get('source', 'book1').strip() or 'book1',
                        'tone': row.get('tone', 'casual').strip() or 'casual',
                        'strength': row.get('strength', 'medium').strip() or 'medium',
                    }
                    
                    if entry['tone'] not in ['formal', 'casual', 'academic']:
                        entry['tone'] = 'casual'
                    if entry['strength'] not in ['light', 'medium', 'strong']:
                        entry['strength'] = 'medium'
                    
                    data.append(entry)
                    row_num += 1
    
    if errors:
        print("[WARNINGS]:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
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
    
    # Print summary
    tone_counts = {}
    strength_counts = {}
    for entry in data:
        tone_counts[entry['tone']] = tone_counts.get(entry['tone'], 0) + 1
        strength_counts[entry['strength']] = strength_counts.get(entry['strength'], 0) + 1
    
    print("Summary:")
    print(f"  By tone: {tone_counts}")
    print(f"  By strength: {strength_counts}")
    print()
    
    # Show first entry as example
    if data:
        print("First entry preview:")
        print(f"  Input (first 100 chars): {data[0]['input'][:100]}...")
        print(f"  Target (first 100 chars): {data[0]['target'][:100]}...")
    
    return True

if __name__ == "__main__":
    csv_file = "data/raw/_template_data_collection.csv"
    output_file = "data/raw/book1_fixed.json"
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    success = fix_and_convert_csv(csv_file, output_file)
    sys.exit(0 if success else 1)
