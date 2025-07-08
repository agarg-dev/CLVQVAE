import torch
import argparse


def extract_sae_vectors(model_path, output_path):
    """
    Extract decoder vectors from a saved SAE model checkpoint and save them as a PyTorch file.
    
    Args:
        model_path: Path to the saved SAE model checkpoint (.pt file)
        output_path: Path to save the extracted decoder vectors (.pt file)
    """
    # Load the checkpoint
    print(f"Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Get the model state dict
    model_state = checkpoint['model_state_dict']
    
    # Check if we have pre-computed decoder vectors in checkpoint
    if 'decoder_vectors' in checkpoint:
        decoder_vectors = torch.tensor(checkpoint['decoder_vectors'])
        print("Using pre-computed decoder vectors from checkpoint")
    else:
        # Extract decoder vectors from the SAE's decoder weights
        decoder_key = 'decoder.weight'
        if decoder_key in model_state:
            decoder_weight = model_state[decoder_key]  # Shape: [output_dim, hidden_dim]
            decoder_vectors = decoder_weight.t()  # Transpose to [hidden_dim, output_dim]
            print("Extracted decoder vectors from decoder weights")
        else:
            raise ValueError(f"Expected decoder weight key '{decoder_key}' not found in model state dict")
    
    # Create dictionary mapping neuron IDs to vectors as lists (same format as CLVQ-VAE)
    decoder_dict = {}
    for idx, vector in enumerate(decoder_vectors):
        decoder_dict[idx] = vector.tolist()
    
    # Save as PyTorch file
    torch.save(decoder_dict, output_path)
    
    print(f"SAE decoder vectors saved to {output_path}")
    print(f"Number of neurons: {len(decoder_dict)}")
    print(f"Vector dimension: {len(decoder_dict[0])}")
    print(f"Decoder vectors shape: {decoder_vectors.shape}")


def main():
    parser = argparse.ArgumentParser(description='Extract decoder vectors from a saved SAE model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the saved SAE model checkpoint (.pt file)')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path to save the extracted decoder vectors (.pt file)')
    
    args = parser.parse_args()
    extract_sae_vectors(args.model_path, args.output_path)


if __name__ == '__main__':
    main()