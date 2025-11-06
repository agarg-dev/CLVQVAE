import torch
import argparse



def extract_codebook(model_path, output_path):
    """
    Extract codebook vectors from a saved model checkpoint and save them as a PyTorch file.
    
    Args:
        model_path: Path to the saved model checkpoint (.pt file)
        output_path: Path to save the extracted codebook vectors (.pt file)
    """
    # Load the checkpoint
    print(f"Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    # Get the model state dict
    model_state = checkpoint['model_state_dict']
    # Extract codebook vectors from the VectorQuantizer's embedding weight
    codebook_key = '_VectorQuantizer._embedding.weight'
    codebook_vectors = model_state[codebook_key]
    # Create dictionary mapping cluster IDs to vectors as lists
    codebook_dict = {}
    for idx, vector in enumerate(codebook_vectors):
        codebook_dict[idx] = vector.tolist()
    # Save as PyTorch file
    torch.save(codebook_dict, output_path)
    print(f"Codebook vectors saved to {output_path}")
    print(f"Number of clusters: {len(codebook_dict)}")
    print(f"Vector dimension: {len(codebook_dict[0])}")




def main():
    parser = argparse.ArgumentParser(description='Extract codebook vectors from a saved model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the saved model checkpoint (.pt file)')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path to save the extracted codebook vectors (.pt file)')
    
    args = parser.parse_args()
    extract_codebook(args.model_path, args.output_path)




if __name__ == '__main__':
    main() 