import json
import os
import re

def convert_dlpro_to_json(text_file, json_file):
    data = []
    # Regex to match lines like "image_path": "correct sentence","wrong sentence"
    # Handles Windows paths with backslashes
    pattern = r'"([^"]+)":\s*"([^"]+)"\s*,\s*"([^"]+)"'
    
    with open(text_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            match = re.match(pattern, line)
            if match:
                image_path = match.group(1)
                correct_sentence = match.group(2)
                wrong_sentence = match.group(3)
                data.append({
                    'image_path': image_path,
                    'correct': correct_sentence,
                    'wrong': wrong_sentence
                })
            else:
                print(f"Skipping malformed line: {line}")
    
    # Ensure the assets directory exists
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    
    # Save to JSON file
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"JSON file saved to {json_file}")

# File paths
text_file = 'assets/dlpro.txt'
json_file = 'assets/dlpro.json'

# Run the conversion
convert_dlpro_to_json(text_file, json_file)