import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from functools import partial
from models.sae_model import SparseAutoencoder
import torch.nn.functional as F


# Set device for computation (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class DualDataset(Dataset):
    """Dataset class for handling dual embedding data."""
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        meta, input_embedding, output_embedding = self.data[idx]
        return meta, input_embedding, output_embedding


def dual_collate_fn(batch, device=None):
    """Custom collate function to handle variable-length sequences with dual embeddings."""
    meta = [item[0] for item in batch]
    input_embeddings = [item[1] for item in batch]
    output_embeddings = [item[2] for item in batch]
    input_embeddings = [emb.squeeze(0) for emb in input_embeddings]
    output_embeddings = [emb.squeeze(0) for emb in output_embeddings]
    max_len = max(len(emb) for emb in input_embeddings)
    input_embedding_dim = input_embeddings[0].size(-1)
    output_embedding_dim = output_embeddings[0].size(-1)
    padded_input_embeddings = []
    padded_output_embeddings = []
    
    for input_emb, output_emb in zip(input_embeddings, output_embeddings):
        # Ensure both have the same sequence length
        assert len(input_emb) == len(output_emb), "Input and output embeddings must have the same sequence length"
        if len(input_emb) < max_len:
            # Pad inputs
            input_padding = torch.zeros(max_len - len(input_emb), input_embedding_dim)
            padded_input_emb = torch.cat([input_emb, input_padding], dim=0)
            # Pad outputs
            output_padding = torch.zeros(max_len - len(output_emb), output_embedding_dim)
            padded_output_emb = torch.cat([output_emb, output_padding], dim=0)
        else:
            padded_input_emb = input_emb
            padded_output_emb = output_emb
        padded_input_embeddings.append(padded_input_emb)
        padded_output_embeddings.append(padded_output_emb)
    
    stacked_input_embeddings = torch.stack(padded_input_embeddings)
    stacked_output_embeddings = torch.stack(padded_output_embeddings)
    
    if device:
        stacked_input_embeddings = stacked_input_embeddings.to(device)
        stacked_output_embeddings = stacked_output_embeddings.to(device)
    
    return meta, stacked_input_embeddings, stacked_output_embeddings


def load_continuousEmbedding(file_name):
    """Load continuous embeddings from a JSON file."""
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File not found: {file_name}")
    with open(file_name, 'r') as json_file:
        return json.load(json_file)


def add_seq_length_dimension(input_data, output_data=None):
    """Process embeddings to add sequence length dimension and handle padding."""
    meta_list = []
    input_embedding_list = []
    for idx, (meta, embedding) in enumerate(input_data):
        if meta is not None:
            meta_list.append(meta)
            input_embedding_list.append(embedding)
    
    df = pd.DataFrame(columns=['meta', 'input_embedding', 'sentence_idx'])
    df['meta'] = meta_list
    df['input_embedding'] = input_embedding_list
    
    for i, word in enumerate(meta_list):
        sentence_idx = int(word.split("|||")[2])
        df.loc[i, 'sentence_idx'] = sentence_idx
    
    new_input_embeddings = [group["input_embedding"].tolist() for _, group in df.groupby("sentence_idx")]
    new_meta = [group["meta"].tolist() for _, group in df.groupby("sentence_idx")]
    
    max_seq_length = max(len(sentence) for sentence in new_input_embeddings)
    input_embedding_dim = len(new_input_embeddings[0][0])
    
    padded_input_embeddings = []
    for sentence in new_input_embeddings:
        if len(sentence) < max_seq_length:
            padding = [[0.0] * input_embedding_dim] * (max_seq_length - len(sentence))
            padded_sentence = sentence + padding
        else:
            padded_sentence = sentence
        padded_input_embeddings.append(padded_sentence)
    
    new_input_embeddings_tensor = torch.tensor(padded_input_embeddings, dtype=torch.float)
    
    if output_data is not None:
        output_meta_list = []
        output_embedding_list = []
        for idx, (meta, embedding) in enumerate(output_data):
            if meta is not None:
                output_meta_list.append(meta)
                output_embedding_list.append(embedding)
        
        output_df = pd.DataFrame(columns=['meta', 'output_embedding', 'sentence_idx'])
        output_df['meta'] = output_meta_list
        output_df['output_embedding'] = output_embedding_list
        
        for i, word in enumerate(output_meta_list):
            sentence_idx = int(word.split("|||")[2])
            output_df.loc[i, 'sentence_idx'] = sentence_idx
        
        new_output_embeddings = [group["output_embedding"].tolist() for _, group in output_df.groupby("sentence_idx")]
        
        # Ensure input and output have the same number of sentences
        assert len(new_input_embeddings) == len(new_output_embeddings), "Input and output datasets must have the same number of sentences"
        
        # Ensure each sentence has the same sequence length in both datasets
        for i, (input_sen, output_sen) in enumerate(zip(new_input_embeddings, new_output_embeddings)):
            assert len(input_sen) == len(output_sen), f"Input and output sequence lengths must match for sentence {i}"
        
        output_embedding_dim = len(new_output_embeddings[0][0])
        
        padded_output_embeddings = []
        for sentence in new_output_embeddings:
            if len(sentence) < max_seq_length:
                padding = [[0.0] * output_embedding_dim] * (max_seq_length - len(sentence))
                padded_sentence = sentence + padding
            else:
                padded_sentence = sentence
            padded_output_embeddings.append(padded_sentence)
        
        new_output_embeddings_tensor = torch.tensor(padded_output_embeddings, dtype=torch.float)
        return new_meta, new_input_embeddings_tensor, new_output_embeddings_tensor

    return new_meta, new_input_embeddings_tensor


def split_data(new_meta, new_input_embeddings_tensor, new_output_embeddings_tensor=None, train_ratio=0.9):
    """Split data into training and validation sets."""
    if len(new_meta) != len(new_input_embeddings_tensor):
        raise ValueError(f"Length mismatch: meta ({len(new_meta)}) != input_embeddings ({len(new_input_embeddings_tensor)})")

    if new_output_embeddings_tensor is not None and len(new_input_embeddings_tensor) != len(new_output_embeddings_tensor):
        raise ValueError(f"Length mismatch: input_embeddings ({len(new_input_embeddings_tensor)}) != output_embeddings ({len(new_output_embeddings_tensor)})")

    df = pd.DataFrame(columns=['meta', 'input_embedding', 'output_embedding', 'sentence_idx'])
    df['meta'] = new_meta
    df['input_embedding'] = [tensor for tensor in new_input_embeddings_tensor]

    if new_output_embeddings_tensor is not None:
        df['output_embedding'] = [tensor for tensor in new_output_embeddings_tensor]

    for i, word_list in enumerate(df['meta']):
        sentence_idx = int(word_list[0].split("|||")[2])
        df.loc[i, 'sentence_idx'] = sentence_idx

    sentence_idx = df['sentence_idx'].unique()
    np.random.shuffle(sentence_idx)
    split_idx = int(len(sentence_idx) * train_ratio)
    train_idx = sentence_idx[:split_idx]
    val_idx = sentence_idx[split_idx:]

    train_data = []
    val_data = []
    for idx in train_idx:
        mask = df['sentence_idx'] == idx
        batch_meta = df[mask]['meta'].tolist()
        batch_input_embedding = torch.stack([emb.clone().detach() for emb in df[mask]['input_embedding']])
        batch_input_embedding = batch_input_embedding.to(device)
        if new_output_embeddings_tensor is not None:
            batch_output_embedding = torch.stack([emb.clone().detach() for emb in df[mask]['output_embedding']])
            batch_output_embedding = batch_output_embedding.to(device)
            train_data.append((batch_meta, batch_input_embedding, batch_output_embedding))
        else:
            train_data.append((batch_meta, batch_input_embedding))
            
    for idx in val_idx:
        mask = df['sentence_idx'] == idx
        batch_meta = df[mask]['meta'].tolist()
        batch_input_embedding = torch.stack([emb.clone().detach() for emb in df[mask]['input_embedding']])
        batch_input_embedding = batch_input_embedding.to(device)
        
        if new_output_embeddings_tensor is not None:
            batch_output_embedding = torch.stack([emb.clone().detach() for emb in df[mask]['output_embedding']])
            batch_output_embedding = batch_output_embedding.to(device)
            val_data.append((batch_meta, batch_input_embedding, batch_output_embedding))
        else:
            val_data.append((batch_meta, batch_input_embedding))
    
    return train_data, val_data


def map_tokens_to_neurons(meta, most_active_neurons):
    """Map tokens to their most active SAE neurons."""
    neuron_map = {}
    batch_size, seq_length = most_active_neurons.shape
    flattened_meta = [item[0] for item in meta]
    max_length = max(len(row) for row in flattened_meta)
    padded_meta = [row + [''] * (max_length - len(row)) for row in flattened_meta]
    meta_array = np.array(padded_meta)
    
    for i in range(batch_size):
        actual_length = len(meta_array[i])
        for j in range(actual_length):
            # Skip padding tokens (empty strings)
            if meta_array[i][j] == '' or not meta_array[i][j]:
                continue
            try:
                word_info = meta_array[i][j].split("|||")
                if len(word_info) >= 3:
                    word = word_info[0] + "_" + word_info[-1] + "_" + word_info[-2]
                    neuron_idx = most_active_neurons[i][j].item()  # ✅ Move inside
                    if neuron_idx not in neuron_map:
                        neuron_map[neuron_idx] = []
                    neuron_map[neuron_idx].append(word)  # ✅ Move inside
            except (IndexError, AttributeError) as e:
                continue

    return neuron_map


def training(train_data, val_data, model, num_training_updates, optimizer, scheduler, device, save_path, args, batch_size=32):
    """Train the SAE model."""
    train_dataset = DualDataset(train_data)
    val_dataset = DualDataset(val_data)
    collate_with_device = partial(dual_collate_fn, device=device)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        collate_fn=collate_with_device
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        collate_fn=collate_with_device
    )
    
    best_epoch = 0
    best_val_loss = float('inf')
    best_model_state = None
    best_optimizer_state = None
    best_neuron_map = None
    no_improvement_counter = 0
    whole_neuron_map = {}
    token_to_neuron_reverse = {} 
    for epoch in range(num_training_updates):
        print("\n" + "="*50)
        print(f"Epoch {epoch + 1}/{num_training_updates}, batches: {len(train_loader)}")
        print("="*50)

        model.reset_stats()
        model.train()

        train_total_loss_error = []
        train_reconstruct_loss_error = []
        train_sparsity_loss_error = []
        
        for idx, (meta, input_embedding, output_embedding) in enumerate(train_loader):
            input_embedding = input_embedding.to(device)
            output_embedding = output_embedding.to(device)
            
            if torch.isnan(input_embedding).any() or torch.isnan(output_embedding).any():
                continue
                
            optimizer.zero_grad()

            sae_output = model(input_embedding)
            reconstructed = sae_output["reconstructed"]
            sparsity_loss = sae_output["sparsity_loss"]
            hidden = sae_output["hidden"]

            # Get most active neurons for token mapping (reuse hidden activations)
            if meta is not None:
                most_active_neurons = torch.argmax(hidden, dim=-1)  # Add this line instead
                neuron_map = map_tokens_to_neurons(meta, most_active_neurons)
                whole_neuron_map = update_neuron_map(whole_neuron_map, neuron_map, token_to_neuron_reverse)

            # Create a padding mask to identify non-padding tokens
            padding_mask = torch.norm(output_embedding, dim=2) > 1e-6
            # Create mask for the reconstruction loss calculation
            mask_expanded = padding_mask.unsqueeze(-1).expand_as(output_embedding)
            # Calculate reconstruction error only on non-padding tokens
            output_valid = torch.masked_select(output_embedding, mask_expanded)
            recon_valid = torch.masked_select(reconstructed, mask_expanded)
            # Calculate reconstruction error for final loss
            recon_error = F.mse_loss(recon_valid, output_valid, reduction="mean")

            # Use the reconstruction error from target embeddings + sparsity loss
            final_loss = recon_error + args.sparsity_weight * sparsity_loss
            final_loss.backward()
            optimizer.step()
            
            train_total_loss_error.append(final_loss.item())
            train_reconstruct_loss_error.append(recon_error.item())
            train_sparsity_loss_error.append(sparsity_loss.item())

        # Print training metrics
        print("\n📊 TRAINING METRICS:")
        print(f"Train Total Loss: {np.mean(train_total_loss_error):.3f}, "
            f"Reconstruct Loss: {np.mean(train_reconstruct_loss_error):.3f}, "
            f"Sparsity Loss: {np.mean(train_sparsity_loss_error):.3f}")
        
        # Get SAE statistics
        sae_stats = model.get_activation_stats()
        print(f"  • SAE Active Neurons: {sae_stats['active_neurons']}/{sae_stats['total_neurons']} ({sae_stats['utilization_rate']*100:.1f}%)")

        # Validation
        model.eval()
        val_total_loss_error = []
        val_reconstruct_loss_error = []
        val_sparsity_loss_error = []

        with torch.no_grad():
            for idx, (meta, input_embedding, output_embedding) in enumerate(val_loader):
                input_embedding = input_embedding.to(device)
                output_embedding = output_embedding.to(device)

                sae_output = model(input_embedding)
                reconstructed = sae_output["reconstructed"]
                sparsity_loss = sae_output["sparsity_loss"]

                # Create a padding mask to identify non-padding tokens
                padding_mask = torch.norm(output_embedding, dim=2) > 1e-6
                # Create mask for the reconstruction loss calculation
                mask_expanded = padding_mask.unsqueeze(-1).expand_as(output_embedding)
                # Calculate reconstruction error only on non-padding tokens
                output_valid = torch.masked_select(output_embedding, mask_expanded)
                recon_valid = torch.masked_select(reconstructed, mask_expanded)
                # Calculate MSE loss on valid elements only
                recon_error = F.mse_loss(recon_valid, output_valid, reduction="mean")
                # Calculate total loss
                total_loss = recon_error + args.sparsity_weight * sparsity_loss

                val_total_loss_error.append(total_loss.item())
                val_reconstruct_loss_error.append(recon_error.item())
                val_sparsity_loss_error.append(sparsity_loss.item())

        # Print validation metrics
        print("-"*50)
        print("📊 VALIDATION METRICS:")
        print(f"Val Total Loss: {np.mean(val_total_loss_error):.3f}, "
            f"Val Reconstruct Loss: {np.mean(val_reconstruct_loss_error):.3f}, "
            f"Val Sparsity Loss: {np.mean(val_sparsity_loss_error):.3f}")
        
        # Update the learning rate based on validation loss
        val_loss_mean = np.mean(val_total_loss_error)
        scheduler.step(val_loss_mean)
        
        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
            best_optimizer_state = optimizer.state_dict()
            best_neuron_map = clean_unused_neurons(whole_neuron_map.copy())
            no_improvement_counter = 0

            # Get decoder vectors for saving
            decoder_vectors = model.get_decoder_vectors()
            sae_stats = model.get_activation_stats()

            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': best_optimizer_state,
                'best_val_loss': best_val_loss,
                'neuron_map': best_neuron_map,
                'input_dim': model.input_dim,
                'hidden_dim': model.hidden_dim,
                'output_dim': model.output_dim,
                'decoder_vectors': decoder_vectors.detach().cpu().numpy(),
                'sparsity_weight': model.sparsity_weight,
                'sae_stats': sae_stats,
            }, save_path)
            print("\n✅ Best model updated and saved at epoch", best_epoch)
        else:
            no_improvement_counter += 1
            
        if no_improvement_counter >= 15:
            print("Early stopping triggered.")
            break

    return best_neuron_map, best_model_state



def update_neuron_map(whole_neuron_map, neuron_map, token_to_neuron_reverse):
    """Update neuron mapping with proper token reassignment handling."""
    for neuron_idx, tokens in neuron_map.items():
        if neuron_idx not in whole_neuron_map:
            whole_neuron_map[neuron_idx] = []
        for token in tokens:
            old_neuron = token_to_neuron_reverse.get(token, None)
            if old_neuron is not None and old_neuron != neuron_idx:
                # Token was previously assigned to different neuron
                if token in whole_neuron_map[old_neuron]:
                    whole_neuron_map[old_neuron].remove(token)
            if token not in whole_neuron_map[neuron_idx]:
                whole_neuron_map[neuron_idx].append(token)
            token_to_neuron_reverse[token] = neuron_idx
    return whole_neuron_map


def clean_unused_neurons(neuron_map):
    """Remove neurons with no assigned tokens."""
    cleaned_map = {}
    for neuron_id, tokens in neuron_map.items():
        if tokens:  # Only keep neurons with assigned tokens
            cleaned_map[neuron_id] = tokens
    return cleaned_map

# Custom dataset and collate function for inference
class CustomDataset(Dataset):
    """Dataset class for handling embedding data."""
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        meta, embedding = self.data[idx]
        return meta, embedding


def custom_collate_fn(batch, device=None):
    """Custom collate function to handle variable-length sequences."""
    meta = [item[0] for item in batch]
    embeddings = [item[1] for item in batch]
    embeddings = [emb.squeeze(0) for emb in embeddings]
    max_len = max(len(emb) for emb in embeddings)
    embedding_dim = embeddings[0].size(-1)
    padded_embeddings = []
    for emb in embeddings:
        if len(emb) < max_len:
            padding = torch.zeros(max_len - len(emb), embedding_dim)
            padded_emb = torch.cat([emb, padding], dim=0)
        else:
            padded_emb = emb
        padded_embeddings.append(padded_emb)
    
    stacked_embeddings = torch.stack(padded_embeddings)
    return meta, stacked_embeddings.to(device) if device else stacked_embeddings


def inference(model, test_data, device, batch_size=64):
    """Perform inference using a trained SAE model."""
    model.eval()
    model.reset_stats()
    
    # For inference, we only need the input embeddings
    test_dataset = CustomDataset([
        (item[0], item[1]) for item in test_data
    ])
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(custom_collate_fn, device=device)
    )
    
    token_to_neuron = {}
    
    with torch.no_grad():
        for meta, embedding in test_loader:
            embedding = embedding.to(device)
            
            # Get most active neurons for each token
            most_active_neurons, _ = model.get_most_active_neuron(embedding)
            
            if meta is not None:
                neuron_map = map_tokens_to_neurons(meta, most_active_neurons)
                for neuron_idx, tokens in neuron_map.items():
                    for token in tokens:
                        token_to_neuron[token] = neuron_idx

    sae_stats = model.get_activation_stats()
    print("\nSAE Usage During Inference:")
    print(f"Active neurons: {sae_stats['active_neurons']}/{sae_stats['total_neurons']} ({sae_stats['utilization_rate']*100:.2f}%)")
    
    return token_to_neuron


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True,
                      help='Run mode: train or inference')
    parser.add_argument('--hidden_dim', type=int, default=2048,
                      help='SAE hidden dimension (should be larger than input_dim)')
    parser.add_argument('--sparsity_weight', type=float, default=0.01,
                      help='Weight for L1 sparsity loss')
    parser.add_argument('--input_layer_embedding', type=str, 
                      help='Path to input embeddings')
    parser.add_argument('--output_layer_embedding', type=str, 
                      help='Path to output embeddings')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--model_path', type=str, help='Path to saved model for inference')
    parser.add_argument('--input_layer', type=int, required=False,
                        help='Index of the input layer to use for training')
    parser.add_argument('--output_layer', type=int, required=False,
                        help='Index of the output layer to use for training')

    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.mode == 'train':
        if args.input_layer == args.output_layer:
            input_data = load_continuousEmbedding(args.input_layer_embedding)
            output_data = input_data.copy()
        else:
            # Load both input and output data
            input_data = load_continuousEmbedding(args.input_layer_embedding)
            output_data = load_continuousEmbedding(args.output_layer_embedding)

        # Get embedding dimensions
        input_embedding_dim = len(input_data[0][1])
        output_embedding_dim = len(output_data[0][1])

        print("input_embedding_dim:", input_embedding_dim)
        print("output_embedding_dim:", output_embedding_dim)
        print("input shape:", len(input_data))
        print("output shape:", len(output_data))
        
        # Process both datasets
        new_meta, new_input_embeddings_tensor, new_output_embeddings_tensor = add_seq_length_dimension(input_data, output_data)
        train_inputs, dev_inputs = split_data(new_meta, new_input_embeddings_tensor, new_output_embeddings_tensor)
        
        # Initialize SAE model
        model = SparseAutoencoder(
            input_dim=input_embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=output_embedding_dim,
            sparsity_weight=args.sparsity_weight
        ).to(device)
        
        # Reset statistics before training
        model.reset_stats()

        optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        best_neuron_map, best_model_state = training(
            train_inputs, 
            dev_inputs, 
            model,
            num_training_updates=100,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            save_path=f"{args.output_dir}/sae_model.pt",
            args=args,
            batch_size=128
        )
        
        with open(f"{args.output_dir}/neuron_map.json", 'w') as f:
            json.dump(best_neuron_map, f, indent=4)
            
    elif args.mode == 'inference':
        if not args.model_path:
            raise ValueError("Model path must be provided for inference mode")
        if not args.input_layer_embedding:
            raise ValueError("Test input data path must be provided for inference mode")
            
        # Load checkpoint
        checkpoint = torch.load(args.model_path, weights_only=False)
        
        model = SparseAutoencoder(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            output_dim=checkpoint['output_dim'],
            sparsity_weight=checkpoint.get('sparsity_weight', 0.01)
        ).to(device)
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Warning: Error loading model state dict: {e}")
            print("Attempting to load with strict=False...")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        print("\nPerforming inference on test data...")
        test_input_data = load_continuousEmbedding(args.input_layer_embedding)
        test_meta, test_input_embeddings_tensor = add_seq_length_dimension(test_input_data)
        test_inputs, _ = split_data(test_meta, test_input_embeddings_tensor, train_ratio=1.0)
        
        # Perform inference to get token mappings
        token_to_neuron_map = inference(model, test_inputs, device)
        
        inference_output_path = f"{args.output_dir}/token_to_neuron_map.json"
        with open(inference_output_path, 'w') as f:
            json.dump(token_to_neuron_map, f, indent=4)
        print(f"Token to neuron mapping saved to {inference_output_path}")


if __name__ == '__main__':
    main()