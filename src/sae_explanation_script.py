import json
import argparse
import pandas as pd


def read_token_to_neuron_map(json_file_path):
    """Read token to neuron index mapping file"""
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


def process_and_merge_data_sae(token_map_path, explanation_file_path, output_file_path):
    """Process and merge data for SAE"""
    # Read token to neuron mapping file
    token_to_neuron_map = read_token_to_neuron_map(token_map_path)
    
    # Read explanation file
    explanations = read_explanation_file(explanation_file_path)
    
    # Create data list
    data = []
    for exp in explanations:
        position_index = exp['position_index']
        sentence_index = exp['sentence_index']
        prediction = exp['prediction']
        
        # Build match pattern (same as CLVQ-VAE)
        match_pattern = f"_{position_index}_{sentence_index}"
        
        # Find corresponding neuron_idx and token from token_to_neuron_map
        neuron_idx = "NA"
        token = "NA"
        for token_key, idx in token_to_neuron_map.items():
            if token_key.endswith(match_pattern):
                neuron_idx = idx
                # Extract token part (everything before the last two underscores)
                token = token_key.rsplit('_', 2)[0]
                break
        
        # Add to data list
        data.append({
            'token': token,
            'position_index': position_index,
            'sentence_index': sentence_index,
            'neuron_idx': neuron_idx,  # Changed from vector_idx to neuron_idx
            'prediction': prediction
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV file
    df.to_csv(output_file_path, index=False)
    
    print(f"SAE merged explanations saved to {output_file_path}")
    print(f"Total entries: {len(df)}")
    print(f"Entries with valid neuron mapping: {len(df[df['neuron_idx'] != 'NA'])}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Merge token to neuron mapping and explanation data for SAE')
    parser.add_argument('--token_map', type=str, required=True,
                        help='Path to token to neuron index mapping file (JSON format)')
    parser.add_argument('--explanation', type=str, required=True,
                        help='Path to explanation file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output CSV file')
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    try:
        process_and_merge_data_sae(args.token_map, args.explanation, args.output)
        print(f"Processing complete! Output file saved as: {args.output}")
    except Exception as e:
        print(f"Error occurred during processing: {str(e)}")


if __name__ == "__main__":
    main()