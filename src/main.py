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
from models.model import Model
from models.vector_quantizer import VectorQuantizerEMA
import torch.nn.functional as F

# Set device for computation (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class DualDataset(Dataset):
    """Dataset class for handling dual embedding data.
    
    Attributes:
        data: List of tuples containing (metadata, input_embedding, output_embedding) triples
    """
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        meta, input_embedding, output_embedding = self.data[idx]
        return meta, input_embedding, output_embedding



def dual_collate_fn(batch, device=None):
    """Custom collate function to handle variable-length sequences with dual embeddings.
    
    Args:
        batch: List of (metadata, input_embedding, output_embedding) tuples
        device: Device to move tensors to (optional)
    
    Returns:
        Tuple of (metadata_list, padded_input_embeddings, padded_output_embeddings)
    """
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
    """Load continuous embeddings from a JSON file.
    
    Args:
        file_name: Path to the JSON file containing embeddings
        
    Returns:
        List of embeddings loaded from the file
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File not found: {file_name}")
    with open(file_name, 'r') as json_file:
        return json.load(json_file)



def add_seq_length_dimension(input_data, output_data=None):
    """Process embeddings to add sequence length dimension and handle padding.
    
    Args:
        input_data: List of (metadata, embedding) tuples for input
        output_data: List of (metadata, embedding) tuples for output (optional)
        
    Returns:
        If output_data is None:
            Tuple of (processed_metadata, padded_input_embeddings_tensor)
        If output_data is provided:
            Tuple of (processed_metadata, padded_input_embeddings_tensor, padded_output_embeddings_tensor)
    """
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
    """Split data into training and validation sets.
    
    Args:
        new_meta: List of metadata
        new_input_embeddings_tensor: Tensor of input embeddings
        new_output_embeddings_tensor: Tensor of output embeddings (optional)
        train_ratio: Ratio of data to use for training (default: 0.9)
        
    Returns:
        If new_output_embeddings_tensor is None:
            Tuple of (train_data, val_data) where each item is a (meta, input_embedding) tuple
        If new_output_embeddings_tensor is provided:
            Tuple of (train_data, val_data) where each item is a (meta, input_embedding, output_embedding) tuple
    """
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



def map_discrete_idx_with_tokens(meta, indices):
    """Map discrete indices to their corresponding tokens.
    
    Args:
        meta: Metadata containing token information
        indices: Tensor of discrete indices
        
    Returns:
        Dictionary mapping vector indices to lists of tokens
    """
    vectors_map = {}
    batch_size, seq_length = indices.shape
    flattened_meta = [item[0] for item in meta]
    max_length = max(len(row) for row in flattened_meta)
    padded_meta = [row + [''] * (max_length - len(row)) for row in flattened_meta]
    meta = np.array(padded_meta)
    for i in range(batch_size):
        actual_length = len(meta[i])
        for j in range(actual_length):
            try:
                word_info = meta[i][j].split("|||")
                word = word_info[0] + "_" + word_info[-1] + "_" + word_info[-2]
                vector_idx = indices[i][j].item()
                if vector_idx not in vectors_map:
                    vectors_map[vector_idx] = []
                vectors_map[vector_idx].append(word)
            except (IndexError, AttributeError) as e:
                continue

    return vectors_map



def update_vector_map(whole_vector_map, vector_map, token_to_key_map):
    """Update the vector mapping with new token assignments.
    
    Args:
        whole_vector_map: Complete mapping of vectors to tokens
        vector_map: New mapping to incorporate
        token_to_key_map: Reverse mapping from tokens to vector indices
        
    Returns:
        Updated whole_vector_map
    """
    for key, tokens in vector_map.items():
        if key not in whole_vector_map:
            whole_vector_map[key] = []
        for token in tokens:
            old_key = token_to_key_map.get(token, None)
            if old_key is not None and old_key != key:
                whole_vector_map[old_key].remove(token)
            if token not in whole_vector_map[key]:
                whole_vector_map[key].append(token)
                token_to_key_map[token] = key
    return whole_vector_map

def clean_unused_vectors(best_vector_map):
    """Remove unused vectors from the codebook mapping.
    
    Args:
        best_vector_map: Vector mapping to clean
        
    Returns:
        Cleaned mapping with only used vectors
    """
    used_vectors = set()
    for cluster_vectors in best_vector_map.values():
        used_vectors.update(cluster_vectors)
    cleaned_map = {}
    for cluster_id, vectors in best_vector_map.items():
        cleaned_vectors = [v for v in vectors if v in used_vectors]
        if cleaned_vectors:
            cleaned_map[cluster_id] = cleaned_vectors
            
    return cleaned_map

def training(train_data, val_data, model, num_training_updates, optimizer, scheduler, device, save_path, args, batch_size=32):
    """Train the VQ-VAE model.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        model: VQ-VAE model instance
        num_training_updates: Number of training epochs
        optimizer: Optimizer instance
        device: Computation device
        save_path: Path to save the best model
        batch_size: Batch size for training (default: 32)
        
    Returns:
        Tuple of (best_vector_map, best_model_state)
    """

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
    best_vector_map = None
    no_improvement_counter = 0
    whole_vector_map = {}
    token_to_key_map = {}

    for epoch in range(num_training_updates):

        model.reset_codebook_usage()
        print(f"\nEpoch {epoch + 1}/{num_training_updates}, batches: {len(train_loader)}")
        model.train()
        train_loss_error = []
        train_reconstruct_loss_error = []
        train_vq_loss_error = []
        train_perplexity_loss_error = []
        for idx, (meta, input_embedding, output_embedding) in enumerate(train_loader):
            input_embedding = input_embedding.to(device)
            output_embedding = output_embedding.to(device)
            if torch.isnan(input_embedding).any() or torch.isnan(output_embedding).any():
                continue
            optimizer.zero_grad()
            model_output = model(input_embedding, device=device)
            z_e = model_output["z_e"]
            reconstructed = model_output["reconstructed"]
            vq_loss = model_output["loss"]
            if meta is not None:
                vector_map = map_discrete_idx_with_tokens(meta, model_output["indices"])
                whole_vector_map = update_vector_map(whole_vector_map, vector_map, token_to_key_map)
            # Create a padding mask to identify non-padding tokens
            padding_mask = torch.norm(output_embedding, dim=2) > 1e-6
            # Create mask for the reconstruction loss calculation
            mask_expanded = padding_mask.unsqueeze(-1).expand_as(output_embedding)
            # Calculate reconstruction error only on non-padding tokens
            output_valid = torch.masked_select(output_embedding, mask_expanded)
            recon_valid = torch.masked_select(reconstructed, mask_expanded)
            recon_error = F.mse_loss(recon_valid, output_valid, reduction="mean")
            loss = recon_error + vq_loss
            loss.backward()
            optimizer.step()
            if "perplexity_loss" in model_output:
                train_perplexity_loss_error.append(model_output["perplexity_loss"].item())
            train_loss_error.append(loss.item())
            train_reconstruct_loss_error.append(recon_error.item())
            train_vq_loss_error.append(vq_loss.item())
            if idx % 10 == 0:  # Print every 10 batches
                perplexity_loss_str = ""
                if train_perplexity_loss_error:
                    perplexity_loss_str = f", perplexity_loss={train_perplexity_loss_error[-1]:.4f}"
                print(f"  Training batch {idx}: loss={loss.item():.4f}, "
                    f"recon_error={recon_error.item():.4f}, "
                    f"vq_loss={vq_loss.item():.4f}, "
                    f"{perplexity_loss_str}")

        # Validation
        model.eval()
        val_loss_error = []
        val_reconstruct_loss_error = []
        val_vq_loss_error = []
        val_perplexity_loss_error = []
        with torch.no_grad():
            for idx, (meta, input_embedding, output_embedding) in enumerate(val_loader):
                input_embedding = input_embedding.to(device)
                output_embedding = output_embedding.to(device)
                model_output = model(input_embedding, device=device)
                z_e = model_output["z_e"]
                reconstructed = model_output["reconstructed"]
                vq_loss = model_output["loss"]
                # Create a padding mask to identify non-padding tokens
                padding_mask = torch.norm(output_embedding, dim=2) > 1e-6
                # Create mask for the reconstruction loss calculation
                mask_expanded = padding_mask.unsqueeze(-1).expand_as(output_embedding)
                # Calculate reconstruction error only on non-padding tokens
                output_valid = torch.masked_select(output_embedding, mask_expanded)
                recon_valid = torch.masked_select(reconstructed, mask_expanded)
                # Calculate MSE loss on valid elements only
                recon_error = F.mse_loss(recon_valid, output_valid, reduction="mean")
                loss = recon_error + vq_loss
                val_loss_error.append(loss.item())
                val_reconstruct_loss_error.append(recon_error.item())
                val_vq_loss_error.append(vq_loss.item())
                # Track perplexity loss for validation
                if "perplexity_loss" in model_output:
                    val_perplexity_loss_error.append(model_output["perplexity_loss"].item())
                if idx % 10 == 0:  # Print every 10 batches
                    perplexity_loss_str = ""
                    if "perplexity_loss" in model_output:
                        perplexity_loss_str = f", perplexity_loss={model_output['perplexity_loss'].item():.4f}"
                    print(f"  Validation batch {idx}: loss={loss.item():.4f}, "
                        f"recon_error={recon_error.item():.4f}, "
                        f"vq_loss={vq_loss.item():.4f}"
                        f"{perplexity_loss_str}")

        print(
            f"Epoch {epoch + 1}, "
            f"Train Loss: {np.mean(train_loss_error):.3f}, "
            f"Train Reconstruct Loss: {np.mean(train_reconstruct_loss_error):.3f}, "
            f"Train VQ Loss: {np.mean(train_vq_loss_error):.3f}, "
            f"Train Perplexity Loss: {np.mean(train_perplexity_loss_error):.3f}, "
            f"Dev Loss: {np.mean(val_loss_error):.3f}, "
            f"Dev Reconstruct Loss: {np.mean(val_reconstruct_loss_error):.3f}, "
            f"Dev VQ Loss: {np.mean(val_vq_loss_error):.3f}, "
            f"Dev Perplexity Loss: {np.mean(val_perplexity_loss_error):.3f}, "
        )
        
        val_loss_mean = np.mean(val_loss_error)
        scheduler.step(val_loss_mean)
        cosine_mean_similarity = model_output["similarity_metric"]["cosine_mean_similarity"]
        euclidean_mean_distance = model_output["similarity_metric"]["euclidean_mean_distance"]
        perplexity = model_output.get('perplexity')
        stats = model.get_codebook_usage()
        nonzero_counts = (stats['usage_count'] > 0).sum().item()
        min_count = stats['usage_count'].min().item() if nonzero_counts > 0 else 0
        max_count = stats['usage_count'].max().item() if nonzero_counts > 0 else 0

        print(f"Training Perplexity: {perplexity:.3f}")
        print(f"cosine_mean_similarity: {cosine_mean_similarity:.3f}" )
        print(f"euclidean_mean_distance: {euclidean_mean_distance:.3f}" )
        print(f"Codebook details: {nonzero_counts}/{stats['total_codes']} vectors used")
        print(f"Usage counts - Min: {min_count}, Max: {max_count}")
        
        if np.mean(val_loss_error) < best_val_loss:
            best_val_loss = np.mean(val_loss_error)
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
            best_optimizer_state = optimizer.state_dict()
            best_vector_map = clean_unused_vectors(whole_vector_map.copy())
            no_improvement_counter = 0
            # Get embedding weights based on VectorQuantizer type
            if isinstance(model._VectorQuantizer, VectorQuantizerEMA):
                embedding_weights = model._VectorQuantizer._embedding.data.cpu().numpy()
            else:
                embedding_weights = model._VectorQuantizer._embedding.weight.data.cpu().numpy()
            # Get codebook usage statistics
            codebook_usage = model.analyze_codebook()

            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': best_optimizer_state,
                'best_val_loss': best_val_loss,
                'whole_vector_map': best_vector_map,
                'embedding_dim': model._VectorQuantizer._embedding_dim,
                'num_embeddings': model._VectorQuantizer._num_embeddings,
                'output_dim': model._decoder.output_projection.out_features,
                'embedding_weights': embedding_weights,
                'use_ema': isinstance(model._VectorQuantizer, VectorQuantizerEMA),
                'codebook_usage': codebook_usage,
                'perplexity_weight': model._VectorQuantizer._perplexity_weight ,
                'use_adaptive_encoder': args.use_adaptive_encoder
            }, save_path)
            print(f"Best model updated and saved at epoch {best_epoch}")
        else:
            no_improvement_counter += 1
        if no_improvement_counter >= 15:
            print("Early stopping triggered.")
            break

    for key in list(best_vector_map.keys()):
        if len(best_vector_map[key]) == 0:
            del best_vector_map[key]

    return best_vector_map, best_model_state



def inference(model, test_data, device, batch_size=32):
    """Perform inference using a trained model.
    
    Args:
        model: Trained VQ-VAE model
        test_data: Test dataset containing (meta, input_embedding) pairs
        device: Computation device
        batch_size: Batch size for inference (default: 32)
        
    Returns:
        Dictionary mapping tokens to their discrete indices
    """
    model.eval()
    # Reset usage statistics before inference
    model.reset_codebook_usage()
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
    token_to_index = {}
    with torch.no_grad():
        for meta, embedding in test_loader:
            embedding = embedding.to(device)
            # For inference, we use the forward method without target_embedding
            output = model(embedding, target_embedding=None, device=device)
            if meta is not None:
                vector_map = map_discrete_idx_with_tokens(meta, output["indices"])
            for idx, tokens in vector_map.items():
                for token in tokens:
                    token_to_index[token] = idx
    
    codebook_stats = model.analyze_codebook()
    print("\nCodebook Usage During Inference:")
    print(f"Active codes: {codebook_stats['active_codes']}/{codebook_stats['total_codes']} ({codebook_stats['utilization_percentage']:.2f}%)")
    print(f"Unused codes: {codebook_stats['unused_codes']}")
    return token_to_index



# CustomDataset and custom_collate_fn for inference
class CustomDataset(Dataset):
    """Dataset class for handling embedding data.
    
    Attributes:
        data: List of tuples containing (metadata, embedding) pairs
    """
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        meta, embedding = self.data[idx]
        return meta, embedding



def custom_collate_fn(batch, device=None):
    """Custom collate function to handle variable-length sequences.
    
    Args:
        batch: List of (metadata, embedding) tuples
        device: Device to move tensors to (optional)
    
    Returns:
        Tuple of (metadata_list, padded_embeddings)
    """
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



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True,
                      help='Run mode: train or inference')
    parser.add_argument('--num_embeddings', type=int, default=400)
    parser.add_argument('--code_vectors', type=str)
    parser.add_argument('--input_layer_embedding', type=str, 
                      help='Path to input embeddings')
    parser.add_argument('--output_layer_embedding', type=str, 
                      help='Path to output embeddings')
    parser.add_argument('--test_fcinput_data', type=str, 
                      help='Path to input embeddings (input to FC layer)')
    parser.add_argument('--test_fcoutput_data', type=str, 
                      help='Path to output embeddings (output of FC layer)')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--model_path', type=str, help='Path to saved model for inference')
    parser.add_argument('--use_ema', action='store_true', help='Use EMA VectorQuantizer')
    parser.add_argument('--use_sampling', action='store_true', help='Use sampling instead of deterministic selection')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top candidates to consider for sampling')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature parameter for sampling')
    parser.add_argument('--use_adaptive_encoder', action='store_true',
                      help='Use adaptive residual encoder with normalization')
    parser.add_argument('--initialization', default='random',
                        help ='Enter the codebook initialization technique')

    return parser.parse_args()



def main():
    args = parse_args()
    
    if args.mode == 'train':
        # Import the initialization functions
        from utils.codebook_initializers import initialize_codebook_from_type
        
        # Load both input and output data
        input_data = load_continuousEmbedding(args.input_layer_embedding)
        output_data = load_continuousEmbedding(args.output_layer_embedding)
        # Get embedding dimensions
        input_embedding_dim = len(input_data[0][1])
        output_embedding_dim = len(output_data[0][1])
        # Process both datasets
        new_meta, new_input_embeddings_tensor, new_output_embeddings_tensor = add_seq_length_dimension(input_data, output_data)
        train_inputs, dev_inputs = split_data(new_meta, new_input_embeddings_tensor, new_output_embeddings_tensor)
        # Initialize model with input embedding dimension and sampling parameters
        model = Model(
            num_embeddings=args.num_embeddings,
            embedding_dim=input_embedding_dim,
            output_dim=output_embedding_dim,
            device=device,
            use_ema=args.use_ema,
            perplexity_weight=0.0,
            use_sampling=args.use_sampling,
            top_k=args.top_k,
            temperature=args.temperature,
            use_adaptive_encoder=args.use_adaptive_encoder
        ).to(device)
        # Reset codebook usage statistics before training
        model.reset_codebook_usage()
        # Initialize codebook
        initialize_codebook_from_type(model, new_input_embeddings_tensor, args.initialization, device)
        optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        best_vector_map, best_model_state = training(
            train_inputs, 
            dev_inputs, 
            model,
            num_training_updates=50,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            save_path=f"{args.output_dir}/model.pt",
            args=args,
            batch_size=128
        )
        with open(f"{args.output_dir}/vector_map.json", 'w') as f:
            json.dump(best_vector_map, f, indent=4)

    elif args.mode == 'inference':
        if not args.model_path:
            raise ValueError("Model path must be provided for inference mode")
        if not args.test_fcinput_data:
            raise ValueError("Test input data path must be provided for inference mode")
        # Load checkpoint
        checkpoint = torch.load(args.model_path)
        output_dim = checkpoint.get('output_dim', checkpoint['embedding_dim'])
        # Include sampling parameters if they were saved in the checkpoint
        use_sampling = checkpoint.get('use_sampling', args.use_sampling)
        top_k = checkpoint.get('top_k', args.top_k)
        temperature = checkpoint.get('temperature', args.temperature)
        use_adaptive_encoder = checkpoint.get('use_adaptive_encoder', False)
        model = Model(
            num_embeddings=checkpoint['num_embeddings'],
            embedding_dim=checkpoint['embedding_dim'],
            output_dim=output_dim,
            device=device,
            use_ema=checkpoint.get('use_ema', False),
            perplexity_weight=checkpoint.get('perplexity_weight', 0.0),
            use_sampling=use_sampling,
            top_k=top_k,
            temperature=temperature,
            use_adaptive_encoder=use_adaptive_encoder
        ).to(device)
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Warning: Error loading model state dict: {e}")
            print("Attempting to load with strict=False...")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        print("\nPerforming inference on test data...")
        test_input_data = load_continuousEmbedding(args.test_fcinput_data)
        test_meta, test_input_embeddings_tensor = add_seq_length_dimension(test_input_data)
        test_inputs, _ = split_data(test_meta, test_input_embeddings_tensor, train_ratio=1.0)
        # Perform inference to get token mappings
        token_to_index_map = inference(model, test_inputs, device)
        inference_output_path = f"{args.output_dir}/token_to_index_map.json"
        with open(inference_output_path, 'w') as f:
            json.dump(token_to_index_map, f, indent=4)
        print(f"Token to index mapping saved to {inference_output_path}")

if __name__ == '__main__':
    main()