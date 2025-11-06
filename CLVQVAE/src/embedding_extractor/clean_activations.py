import json
import argparse
import sys
import os
import shutil

def clean_file_inplace(file_path):
    """
    Safely modifies a single JSONL activation file in-place to remove '<s>' tokens.

    This function reads the specified file, filters out the '<s>' token from the
    'features' list in each line, and overwrites the original file with the
    cleaned content. It uses a temporary file to ensure the operation is safe
    and does not corrupt data if an error occurs.

    Args:
        file_path (str): The exact path to the activation file to be cleaned.
    """
    # 1. Verify the target file exists before starting.
    if not os.path.isfile(file_path):
        print(f"Error: Input file not found at '{file_path}'", file=sys.stderr)
        sys.exit(1)

    # 2. Define a temporary file path in the same directory.
    temp_file_path = file_path + '.tmp'

    try:
        # 3. Open the original file for reading and the temporary file for writing.
        with open(file_path, 'r', encoding='utf-8') as infile, \
             open(temp_file_path, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                try:
                    # Load the JSON object from the current line.
                    data = json.loads(line)
                    
                    # Check if the 'features' key exists and is a list.
                    if 'features' in data and isinstance(data['features'], list):
                        # Create a new list, keeping all features except '<s>'.
                        filtered_features = [
                            feature for feature in data['features'] 
                            if feature.get('token') != '<s>'
                        ]
                        # Replace the old list with the new, filtered one.
                        data['features'] = filtered_features
                    
                    # Write the (potentially modified) JSON object to the temporary file.
                    outfile.write(json.dumps(data) + '\n')

                except json.JSONDecodeError:
                    # If a line is not valid JSON, write it back as-is to avoid data loss.
                    outfile.write(line)

        # 4. If the entire file is processed successfully, atomically replace the
        #    original file with the temporary file. This is the safe-overwrite step.
        shutil.move(temp_file_path, file_path)
        print(f"Successfully cleaned: {file_path}")

    except Exception as e:
        print(f"An error occurred during processing: {e}", file=sys.stderr)
        # If any error occurs, clean up by deleting the temporary file.
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        # Exit with a non-zero status code to signal failure.
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to safely remove the '<s>' (beginning-of-sentence) token "
                    "from a single JSONL activation file in-place."
    )
    parser.add_argument(
        "activation_file_path",
        help="The full, exact path to the activation file to be modified."
    )
    
    args = parser.parse_args()
    clean_file_inplace(args.activation_file_path)
