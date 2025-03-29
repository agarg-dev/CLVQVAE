import json
import argparse
import pandas as pd

def read_token_map(json_file_path):
    """Read token to vector index mapping file"""
    with open(json_file_path, 'r') as f:
        return json.load(f)


def read_explanation_file(file_path):
    """Read explanation file"""
    explanations = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            prediction = parts[0]
            position_index = parts[1]
            sentence_index = parts[2]
            
            explanations.append({
                'prediction': prediction,
                'position_index': position_index,
                'sentence_index': sentence_index
            })
    return explanations


def process_and_merge_data(token_map_path, explanation_file_path, output_file_path):
    """Process and merge data"""
    # Read mapping file
    token_map = read_token_map(token_map_path)
    
    # Read explanation file
    explanations = read_explanation_file(explanation_file_path)
    
    # Create data list
    data = []
    for exp in explanations:
        position_index = exp['position_index']
        sentence_index = exp['sentence_index']
        prediction = exp['prediction']
        
        # Build match pattern
        match_pattern = f"_{position_index}_{sentence_index}"
        
        # Find corresponding vector_idx and token from token_map
        vector_idx = "NA"
        token = "NA"
        for token_key, idx in token_map.items():
            if token_key.endswith(match_pattern):
                vector_idx = idx
                # Extract token part (everything before the last two underscores)
                token = token_key.rsplit('_', 2)[0]
                break
        
        # Add to data list
        data.append({
            'token': token,
            'position_index': position_index,
            'sentence_index': sentence_index,
            'vector_idx': vector_idx,
            'prediction': prediction
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV file
    df.to_csv(output_file_path, index=False)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Merge token mapping and explanation data')
    parser.add_argument('--token_map', type=str, required=True,
                        help='Path to token to vector index mapping file (JSON format)')
    parser.add_argument('--explanation', type=str, required=True,
                        help='Path to explanation file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output CSV file')
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    try:
        process_and_merge_data(args.token_map, args.explanation, args.output)
        print(f"Processing complete! Output file saved as: {args.output}")
    except Exception as e:
        print(f"Error occurred during processing: {str(e)}")


if __name__ == "__main__":
    main()
