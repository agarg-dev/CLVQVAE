import json
from collections import defaultdict

def extract_balanced_subset(json_file, txt_file, samples_per_class=4000):
    """
    Extract a balanced subset from AG News dataset with specified samples per class.
    
    Args:
        json_file (str): Path to the JSON file
        txt_file (str): Path to the TXT file
        samples_per_class (int): Number of samples to extract per class
    """
    
    # Read JSON file
    print("Reading JSON file...")
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Read TXT file
    print("Reading TXT file...")
    with open(txt_file, 'r', encoding='utf-8') as f:
        txt_lines = f.readlines()
    
    # Verify that both files have the same number of entries
    if len(json_data) != len(txt_lines):
        raise ValueError(f"Mismatch: JSON has {len(json_data)} entries, TXT has {len(txt_lines)} lines")
    
    print(f"Total samples: {len(json_data)}")
    
    # Group samples by label
    label_groups = defaultdict(list)
    
    for idx, sample in enumerate(json_data):
        label = sample['label']
        label_groups[label].append(idx)
    
    # Print class distribution
    print("\nOriginal class distribution:")
    for label in sorted(label_groups.keys()):
        print(f"Label {label}: {len(label_groups[label])} samples")
    
    # Select indices for balanced subset
    selected_indices = []
    
    print(f"\nExtracting {samples_per_class} samples per class...")
    for label in sorted(label_groups.keys()):
        available_samples = len(label_groups[label])
        if available_samples < samples_per_class:
            print(f"Warning: Label {label} has only {available_samples} samples, taking all")
            selected_indices.extend(label_groups[label])
        else:
            selected_indices.extend(label_groups[label][:samples_per_class])
            print(f"Label {label}: Selected {samples_per_class} samples")
    
    # Sort indices to maintain original order
    selected_indices.sort()
    
    print(f"\nTotal selected samples: {len(selected_indices)}")
    
    # Create filtered datasets
    filtered_json = [json_data[idx] for idx in selected_indices]
    filtered_txt_lines = [txt_lines[idx] for idx in selected_indices]
    
    # Verify the filtered dataset balance
    print("\nFiltered dataset class distribution:")
    label_counts = defaultdict(int)
    for sample in filtered_json:
        label_counts[sample['label']] += 1
    
    for label in sorted(label_counts.keys()):
        print(f"Label {label}: {label_counts[label]} samples")
    
    # Save filtered JSON
    output_json_file = json_file.replace('.json', '_balanced_4k.json')
    print(f"\nSaving filtered JSON to: {output_json_file}")
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_json, f, indent=2, ensure_ascii=False)
    
    # Save filtered TXT
    output_txt_file = txt_file.replace('.txt', '_balanced_4k.txt')
    print(f"Saving filtered TXT to: {output_txt_file}")
    with open(output_txt_file, 'w', encoding='utf-8') as f:
        f.writelines(filtered_txt_lines)
    
    print("\nSubset extraction completed successfully!")
    
    return output_json_file, output_txt_file

# Example usage
if __name__ == "__main__":
    # Specify your file paths
    json_file_path = "dev.json"
    txt_file_path = "dev.txt"
    
    try:
        # Extract balanced subset
        output_json, output_txt = extract_balanced_subset(
            json_file_path, 
            txt_file_path, 
            samples_per_class=300
        )
        
        print(f"\nOutput files created:")
        print(f"- JSON: {output_json}")
        print(f"- TXT: {output_txt}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please make sure 'dev.json' and 'dev.txt' are in the same directory as this script.")
    except Exception as e:
        print(f"Error: {e}")