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
from models.codebook_initialization import initialize_codebook_from_type
import random
import math
from transformers import get_cosine_schedule_with_warmup


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        clear_gpu_memory()
        return new_meta, new_input_embeddings_tensor, new_output_embeddings_tensor

    clear_gpu_memory()
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
        # batch_input_embedding = batch_input_embedding.to(device)
        if new_output_embeddings_tensor is not None:
            batch_output_embedding = torch.stack([emb.clone().detach() for emb in df[mask]['output_embedding']])
            # batch_output_embedding = batch_output_embedding.to(device)
            train_data.append((batch_meta, batch_input_embedding, batch_output_embedding))
        else:
            train_data.append((batch_meta, batch_input_embedding))
            
    for idx in val_idx:
        mask = df['sentence_idx'] == idx
        batch_meta = df[mask]['meta'].tolist()
        batch_input_embedding = torch.stack([emb.clone().detach() for emb in df[mask]['input_embedding']])
        # batch_input_embedding = batch_input_embedding.to(device)
        
        if new_output_embeddings_tensor is not None:
            batch_output_embedding = torch.stack([emb.clone().detach() for emb in df[mask]['output_embedding']])
            # batch_output_embedding = batch_output_embedding.to(device)
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

    scaler = torch.amp.GradScaler()

    for epoch in range(num_training_updates):

        print("\n" + "="*50)
        print(f"Epoch {epoch + 1}/{num_training_updates}, batches: {len(train_loader)}")
        print("="*50)

        model.reset_codebook_usage()
        model.train()

        train_total_loss_error = []
        train_reconstruct_loss_error = []
        train_commit_loss_error = []
        train_perplexity_loss_error = []

        train_perplexity_values = []
        train_cosine_similarity_values = []
        train_euclidean_distance_values = []
        
        for idx, (meta, input_embedding, output_embedding) in enumerate(train_loader):
            input_embedding = input_embedding.to(device)
            output_embedding = output_embedding.to(device)
            if torch.isnan(input_embedding).any() or torch.isnan(output_embedding).any():
                continue
            optimizer.zero_grad()


            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):

                model_output = model(input_embedding, device=device)
                z_e = model_output["z_e"]
                reconstructed = model_output["reconstructed"]
                commit_loss = model_output["commit_loss"]
                perplexity_loss = model_output["perplexity_loss"]

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
                # Calculate reconstruction error
                recon_error = F.mse_loss(recon_valid, output_valid, reduction="mean")
                # Calculate total loss
                total_loss = recon_error + commit_loss + perplexity_loss


            # Scale the loss and call backward() to create scaled gradients
            scaler.scale(total_loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            # Unscales gradients and calls or skips optimizer.step()
            scaler.step(optimizer)
            # Updates the scale for next iteration
            scaler.update()
            
            train_perplexity_loss_error.append(perplexity_loss.item())
            train_total_loss_error.append(total_loss.item())
            train_reconstruct_loss_error.append(recon_error.item())
            train_commit_loss_error.append(commit_loss.item())
            train_perplexity_values.append(model_output["perplexity"].item())
            train_cosine_similarity_values.append(model_output["similarity_metric"]["cosine_mean_similarity"])
            train_euclidean_distance_values.append(model_output["similarity_metric"]["euclidean_mean_distance"])


        # Print training metrics
        print("\n📊 TRAINING METRICS:")

        print(f"Train Total Loss: {np.mean(train_total_loss_error):.3f}, "
            f"Reconstruct Loss: {np.mean(train_reconstruct_loss_error):.3f}, "
            f"Commit Loss: {np.mean(train_commit_loss_error):.3f}, "
            f"Perplexity Loss: {np.mean(train_perplexity_loss_error):.3f}")
        print(f"  • Perplexity (avg): {np.mean(train_perplexity_values):.3f}")
        print(f"  • Cosine Similarity: {train_cosine_similarity_values[-1]:.3f}")
        print(f"  • Euclidean Distance: {train_euclidean_distance_values[-1]:.3f}")
        # Inside the training loop, after the last batch
        print(f"DEBUG: Min _ema_cluster_size: {model._VectorQuantizer._ema_cluster_size.min().item()}")
        print(f"DEBUG: Max _ema_cluster_size: {model._VectorQuantizer._ema_cluster_size.max().item()}")

        # Validation
        model.eval()
        val_total_loss_error = []
        val_reconstruct_loss_error = []
        val_commit_loss_error = []
        val_perplexity_loss_error = []

        val_perplexity_values = []
        val_cosine_similarity_values = []
        val_euclidean_distance_values = []

        with torch.no_grad():

            for idx, (meta, input_embedding, output_embedding) in enumerate(val_loader):
                input_embedding = input_embedding.to(device)
                output_embedding = output_embedding.to(device)

                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    model_output = model(input_embedding, device=device)

                    z_e = model_output["z_e"]
                    reconstructed = model_output["reconstructed"]
                    commit_loss = model_output["commit_loss"]
                    perplexity_loss = model_output["perplexity_loss"]

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
                    total_loss = recon_error + commit_loss + perplexity_loss

                val_total_loss_error.append(total_loss.item())
                val_reconstruct_loss_error.append(recon_error.item())
                val_commit_loss_error.append(commit_loss.item())
                val_perplexity_loss_error.append(perplexity_loss.item())
                val_perplexity_values.append(model_output["perplexity"].item())
                val_cosine_similarity_values.append(model_output["similarity_metric"]["cosine_mean_similarity"])
                val_euclidean_distance_values.append(model_output["similarity_metric"]["euclidean_mean_distance"])


        # Print validation metrics
        stats = model.get_codebook_usage()
        nonzero_counts = (stats['usage_count'] > 0).sum().item()
        min_count = stats['usage_count'].min().item() if nonzero_counts > 0 else 0
        max_count = stats['usage_count'].max().item() if nonzero_counts > 0 else 0

        # For validation metrics:
        print("-"*50)
        print("📊 VALIDATION METRICS:")
        
        print(f"Dev Total Loss: {np.mean(val_total_loss_error):.3f}, "
            f"Dev Reconstruct Loss: {np.mean(val_reconstruct_loss_error):.3f}, "
            f"Dev Commit Loss: {np.mean(val_commit_loss_error):.3f}, "
            f"Dev Perplexity Loss: {np.mean(val_perplexity_loss_error):.3f}")
        print(f"  • Perplexity (avg): {np.mean(val_perplexity_values):.3f}")
        print(f"  • Cosine Similarity: {val_cosine_similarity_values[-1]:.3f}")
        print(f"  • Euclidean Distance: {val_euclidean_distance_values[-1]:.3f}")
        print(f"  • Codebook details: {nonzero_counts}/{stats['total_codes']} vectors used")
        print(f"  • Usage counts - Min: {min_count:.1f}, Max: {max_count:.1f}")

        
        # ALPHA TRACKING CODE:
        if args.use_adaptive_encoder and hasattr(model._ContinuousEmbedding, 'alpha'):
            if hasattr(model._ContinuousEmbedding, 'is_fixed') and not model._ContinuousEmbedding.is_fixed:
                alpha_val = torch.sigmoid(model._ContinuousEmbedding.alpha).item() * 1
                print(f"  • Current α: {alpha_val:.4f}")
            elif hasattr(model._ContinuousEmbedding, 'is_fixed') and model._ContinuousEmbedding.is_fixed:
                alpha_val = model._ContinuousEmbedding.alpha.item() * 1
                print(f"  • Fixed α: {alpha_val:.4f}")

        # Update the learning rate based on scheduler type
        val_loss_mean = np.mean(val_total_loss_error)
        if args.scheduler_type == 'plateau':
            scheduler.step(val_loss_mean)
        elif args.scheduler_type == 'cosine_warmup':
            scheduler.step()

        
        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
            best_optimizer_state = optimizer.state_dict()
            best_vector_map = clean_unused_vectors(whole_vector_map.copy())
            no_improvement_counter = 0
            # Get embedding weights based on VectorQuantizer type
            if isinstance(model._VectorQuantizer, VectorQuantizerEMA):
                embedding_weights = model._VectorQuantizer._embedding.weight.data.cpu().numpy() 
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
            print("\n✅ Best model updated and saved at epoch", best_epoch)
        else:
            no_improvement_counter += 1
        if no_improvement_counter >= 15:
            print("Early stopping triggered.")
            break
        clear_gpu_memory()

    for key in list(best_vector_map.keys()):
        if len(best_vector_map[key]) == 0:
            del best_vector_map[key]

    return best_vector_map, best_model_state


def analyze_s_token(token_to_index_map):
    """
    Count how many times <s> token appears in each cluster.
    
    Args:
        token_to_index_map: Dictionary mapping tokens to their discrete indices
    """
    # Count each cluster's <s> tokens
    cluster_counts = {}
    
    for token_key, cluster_id in token_to_index_map.items():
        if token_key.startswith("<s>"):
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
    
    # Print results
    print("\n===== <s> Token Cluster Counts =====")
    
    if cluster_counts:
        print(f"<s> tokens appear in {len(cluster_counts)} different clusters:")
        for cluster_id, count in sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  Cluster {cluster_id}: {count} times")
    else:
        print("No <s> tokens found in any cluster.")



def inference(model, test_data, device, batch_size=64):
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
    perplexity_values = [] 
    with torch.no_grad():
        for meta, embedding in test_loader:
            embedding = embedding.to(device)
            # For inference, we use the forward method without target_embedding
            output = model(embedding, target_embedding=None, device=device)
            
            if 'perplexity' in output:
                perplexity_values.append(output['perplexity'].item())
            if meta is not None:
                vector_map = map_discrete_idx_with_tokens(meta, output["indices"])
            for idx, tokens in vector_map.items():
                for token in tokens:
                    token_to_index[token] = idx

    # Calculate average perplexity
    if perplexity_values:
        avg_perplexity = np.mean(perplexity_values)
        print(f"\nAverage Inference Perplexity: {avg_perplexity:.3f}")
    
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


def scale_tensor_log(tensor, avg_scaler):
    tensor_flat = tensor.view(-1, tensor.size(-1))
    scaled_flat = torch.zeros_like(tensor_flat)
    mask = torch.any(tensor_flat != 0, dim=1)
    active_vectors = tensor_flat[mask]
    
    if active_vectors.shape[0] > 0:
        original_norms = torch.norm(active_vectors, p=2, dim=1, keepdim=True)
        final_norms = torch.log1p(original_norms) / avg_scaler
        scaled_vectors = (active_vectors / (original_norms + 1e-8)) * final_norms
        scaled_flat[mask] = scaled_vectors
    
    return scaled_flat.view(tensor.shape)



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
    parser.add_argument('--random_vector_seed', default=42,
                        help ='seed for random vector initialization', type=int,required=False)
    parser.add_argument('--codebook_dir', type=str, default=None,
                  help='Directory to save/load codebook from')
    parser.add_argument('--perplexity_weight', type=float, default=0.0,
                        help='Weight for perplexity loss')
    parser.add_argument('--input_layer',type=int, required=False,
                        help='Index of the input layer to use for training')
    parser.add_argument('--output_layer',type=int, required=False,
                        help='Index of the output layer to use for training')
    parser.add_argument('--fixed_alpha', type=float, default=None,
                   help='Use fixed alpha value instead of learnable')
    parser.add_argument('--commitment_cost', type=float, default=0.1,
                        help='Commitment cost for vector quantization')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--dec_dim', type=int, default=1024,
                        help='Dimension of the bottleneck and decoder.')
    parser.add_argument('--encoder_weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer.')
    parser.add_argument('--decoder_weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer.')
    parser.add_argument('--use_adamw', action='store_true',
                        help='Use AdamW optimizer instead of Adam')
    parser.add_argument('--model_name', type=str, default='roberta',
                        help='Name of the embedding model')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--scheduler_type', type=str, default='plateau',
                        choices=['plateau', 'cosine_warmup'],
                        help='Learning rate scheduler type: plateau (ReduceLROnPlateau) or cosine_warmup (CosineAnnealingWarmup)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()



def main():
    args = parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # torch.autograd.set_detect_anomaly(True)
    if args.mode == 'train':
        # --- Data Loading ---
        if args.input_layer == args.output_layer:
            input_data = load_continuousEmbedding(args.input_layer_embedding)
            output_data = input_data.copy()
        else:
            input_data = load_continuousEmbedding(args.input_layer_embedding)
            output_data = load_continuousEmbedding(args.output_layer_embedding)

        input_embedding_dim = len(input_data[0][1])
        output_embedding_dim = len(output_data[0][1])

        new_meta, new_input_embeddings_tensor, new_output_embeddings_tensor = add_seq_length_dimension(input_data, output_data)
        train_inputs, dev_inputs = split_data(new_meta, new_input_embeddings_tensor, new_output_embeddings_tensor)

        tensor_for_init = new_input_embeddings_tensor

        # --- Model Creation ---
        model = Model(
            num_embeddings=args.num_embeddings,
            embedding_dim=input_embedding_dim,
            output_dim=output_embedding_dim,
            device=device,
            use_ema=args.use_ema,
            perplexity_weight=args.perplexity_weight,
            use_sampling=args.use_sampling,
            top_k=args.top_k,
            temperature=args.temperature,
            use_adaptive_encoder=args.use_adaptive_encoder,
            fixed_alpha=args.fixed_alpha,
            commitment_cost=args.commitment_cost
        ).to(device)
        
        initialize_codebook_from_type(model, tensor_for_init, args.initialization, args.random_vector_seed, args.input_layer, device, args.codebook_dir)

        # --- Optimizer, Scheduler, and Training ---
        num_epochs = args.num_epochs
        encoder_params, decoder_params, no_decay_params = [], [], []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if '._VectorQuantizer._embedding.weight' in name or name.endswith(".bias") or "layernorm" in name.lower():
                no_decay_params.append(param)
            elif '_ContinuousEmbedding' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)
        
        # Using separate encoder/decoder weight decay
        optim_groups = [
            {'params': encoder_params, 'weight_decay': args.encoder_weight_decay},
            {'params': decoder_params, 'weight_decay': args.decoder_weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        if args.use_adamw:
            optimizer = optim.AdamW(
                optim_groups,
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-6
            )
        else:
            optimizer = optim.Adam(
                optim_groups,
                lr=args.learning_rate
            )
            

        # Choose scheduler based on argument
        if args.scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        elif args.scheduler_type == 'cosine_warmup':
            # Calculate training steps for scheduler
            steps_per_epoch = len(train_inputs) // args.batch_size + (1 if len(train_inputs) % args.batch_size != 0 else 0)
            total_steps = num_epochs * steps_per_epoch
            warmup_steps = int(0.1 * total_steps)  # 10% warmup

            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                num_cycles=0.5, 
                last_epoch=-1
            )

        best_vector_map, best_model_state = training(
            train_inputs, 
            dev_inputs, 
            model,
            num_training_updates=num_epochs, 
            optimizer=optimizer, 
            scheduler=scheduler,
            device=device, 
            save_path=f"{args.output_dir}/model.pt", 
            args=args, 
            batch_size=args.batch_size
        )


        with open(f"{args.output_dir}/vector_map.json", 'w') as f:
            json.dump(best_vector_map, f, indent=4)

    elif args.mode == 'inference':
        # --- Model Loading ---
        if not args.model_path or not args.input_layer_embedding:
            raise ValueError("Model path and test input data must be provided for inference")
        
        checkpoint = torch.load(args.model_path, map_location=device)
        output_dim = checkpoint.get('output_dim', checkpoint['embedding_dim'])
        
        model = Model(
            num_embeddings=checkpoint['num_embeddings'],
            embedding_dim=checkpoint['embedding_dim'],
            output_dim=output_dim,
            device=device,
            use_ema=checkpoint.get('use_ema', False),
            perplexity_weight=checkpoint.get('perplexity_weight', 0.0),
            use_sampling=checkpoint.get('use_sampling', False),
            top_k=checkpoint.get('top_k', 10),
            temperature=checkpoint.get('temperature', 1.0),
            use_adaptive_encoder=checkpoint.get('use_adaptive_encoder', False)
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # --- Load and Scale Inference Data ---
        test_input_data = load_continuousEmbedding(args.input_layer_embedding)
        test_meta, test_input_embeddings_tensor = add_seq_length_dimension(test_input_data)
        
        tensor_for_inference = test_input_embeddings_tensor

        # --- Perform Inference ---
        test_inputs, _ = split_data(test_meta, tensor_for_inference, train_ratio=1.0)
        token_to_index_map = inference(model, test_inputs, device)
        
        inference_output_path = f"{args.output_dir}/token_to_index_map.json"
        with open(inference_output_path, 'w') as f:
            json.dump(token_to_index_map, f, indent=4)
        print(f"Token to index mapping saved to {inference_output_path}")
        analyze_s_token(token_to_index_map)


if __name__ == '__main__':
    main()