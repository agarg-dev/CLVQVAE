import torch
import torch.nn.functional as F
import numpy as np
import os
import random
import faiss

def _get_active_embeddings(embeddings_tensor, embedding_dim):
    """Helper to flatten, move to CPU, and filter zero vectors."""
    embeddings_cpu = embeddings_tensor.cpu()
    flat_embeddings = embeddings_cpu.view(-1, embedding_dim)
    mask = torch.any(flat_embeddings != 0, dim=1)
    active_embeddings = flat_embeddings[mask]
    return active_embeddings

def initialize_spherical_kmeans_plus_plus(model, embeddings_tensor, device=None):
    """
    Initialize the codebook using a highly optimized Faiss implementation of Spherical K-means.
    
    This version uses Faiss for a significant performance improvement, especially with 
    GPU acceleration. It optimizes for directional similarity and then scales centroids
    by the average magnitude of the points in their respective clusters.
    """
    target_device = device or next(model.parameters()).device
    print("Initializing codebook with Faiss Spherical K-means++...")
    num_embeddings = model._VectorQuantizer._num_embeddings
    embedding_dim = model._VectorQuantizer._embedding_dim
    
    active_embeddings = _get_active_embeddings(embeddings_tensor, embedding_dim)
    
    if active_embeddings.shape[0] < num_embeddings:
        print(f"Warning: Not enough unique vectors ({active_embeddings.shape[0]}) for clustering. Using available vectors and padding.")
        final_centroids = torch.zeros(num_embeddings, embedding_dim, device=target_device)
        final_centroids[:active_embeddings.shape[0], :] = active_embeddings.to(target_device)
    else:
        total_vectors = active_embeddings.size(0)
        print(f"Total input vectors for initialization: {total_vectors}")
        
        # Calculate magnitudes for scaling later, then normalize for clustering
        norms = torch.norm(active_embeddings, p=2, dim=1, keepdim=True)
        avg_norm = norms.mean().item()
        normalized_embeddings = active_embeddings / (norms + 1e-8)
        
        data_for_faiss = np.ascontiguousarray(normalized_embeddings.numpy(), dtype='float32')

        # Configure and run Faiss K-means
        kmeans = faiss.Kmeans(
            d=embedding_dim, k=num_embeddings, niter=30, nredo=10, verbose=True,
            gpu=torch.cuda.is_available()
        )
        kmeans.train(data_for_faiss)
        
        # Get centroids and cluster assignments
        normalized_centroids = kmeans.centroids
        _, cluster_assignments = kmeans.index.search(data_for_faiss, 1)
        cluster_assignments = cluster_assignments.flatten()
        
        # Scale each normalized centroid by the average magnitude of its cluster members
        print("Applying cluster-specific magnitude scaling...")
        scaled_centroids = np.zeros_like(normalized_centroids)
        original_np = active_embeddings.numpy()
        
        for i in range(num_embeddings):
            cluster_indices = np.where(cluster_assignments == i)[0]
            if len(cluster_indices) > 0:
                cluster_points = original_np[cluster_indices]
                avg_cluster_norm = np.linalg.norm(cluster_points, axis=1).mean()
                scaled_centroids[i] = normalized_centroids[i] * avg_cluster_norm
            else:
                # Fallback for empty clusters
                scaled_centroids[i] = normalized_centroids[i] * avg_norm
                print(f"Warning: Empty cluster {i} found, using global average magnitude.")

        final_centroids = torch.tensor(scaled_centroids, dtype=torch.float).to(target_device)

    # --- Update Model Weights ---
    print("Updating model codebook with new centroids...")
    with torch.no_grad():
        if hasattr(model._VectorQuantizer, '_ema_w'):
            model._VectorQuantizer._embedding.weight.data.copy_(final_centroids)
            model._VectorQuantizer._ema_w.data.copy_(final_centroids)
            model._VectorQuantizer._ema_cluster_size.data.fill_(1.0)
        else:
            model._VectorQuantizer._embedding.weight.data.copy_(final_centroids)
    print("Initialization with Spherical K-means++ complete.")


def initialize_codebook_kmeans_plus_plus(model, embeddings_tensor, device=None):
    """
    Initialize the codebook using a highly optimized Faiss implementation of K-means++.
    """
    target_device = device or next(model.parameters()).device
    print("Initializing codebook with Faiss K-means++...")
    num_embeddings = model._VectorQuantizer._num_embeddings
    embedding_dim = model._VectorQuantizer._embedding_dim
    
    active_embeddings = _get_active_embeddings(embeddings_tensor, embedding_dim)

    if active_embeddings.shape[0] < num_embeddings:
        print(f"Warning: Not enough unique vectors ({active_embeddings.shape[0]}) for clustering. Using available vectors and padding.")
        final_centroids = torch.zeros(num_embeddings, embedding_dim, device=target_device)
        final_centroids[:active_embeddings.shape[0], :] = active_embeddings.to(target_device)
    else:
        total_vectors = active_embeddings.size(0)
        print(f"Total input vectors for initialization: {total_vectors}")
        
        data_for_faiss = np.ascontiguousarray(active_embeddings.numpy(), dtype='float32')

        # Configure and run Faiss K-means
        kmeans = faiss.Kmeans(
            d=embedding_dim, k=num_embeddings, niter=30, nredo=10, verbose=True,
            gpu=torch.cuda.is_available()
        )
        kmeans.train(data_for_faiss)
        
        final_centroids = torch.tensor(kmeans.centroids, dtype=torch.float).to(target_device)

    # --- Update Model Weights ---
    print("Updating model codebook with new centroids...")
    with torch.no_grad():
        if hasattr(model._VectorQuantizer, '_ema_w'):
            model._VectorQuantizer._embedding.weight.data.copy_(final_centroids)
            model._VectorQuantizer._ema_w.data.copy_(final_centroids)
            model._VectorQuantizer._ema_cluster_size.data.fill_(1.0)
        else:
            model._VectorQuantizer._embedding.weight.data.copy_(final_centroids)
    print("Initialization with K-means++ complete.")


def initialize_codebook(model, embeddings_tensor, seed, device=None):
    """Initialize the codebook by randomly sampling from input embeddings."""
    gen = torch.Generator().manual_seed(seed)
    target_device = device or next(model.parameters()).device
    print("Initializing codebook vectors by random sampling...")
    num_embeddings = model._VectorQuantizer._num_embeddings
    embedding_dim = model._VectorQuantizer._embedding_dim
    
    active_embeddings = _get_active_embeddings(embeddings_tensor, embedding_dim)
    total_vectors = active_embeddings.size(0)
    print(f"Total input vectors for initialization: {total_vectors}")
    
    # Ensure we don't try to sample more vectors than are available
    k = min(num_embeddings, total_vectors)
    indices = torch.randperm(total_vectors, generator=gen)[:k]
    codebook = active_embeddings[indices].to(target_device)

    # If we sampled fewer than num_embeddings, pad with zeros
    if k < num_embeddings:
        padding = torch.zeros(num_embeddings - k, embedding_dim, device=target_device)
        codebook = torch.cat([codebook, padding], dim=0)
    
    with torch.no_grad():
        if hasattr(model._VectorQuantizer, '_ema_w'):
            model._VectorQuantizer._embedding.weight.data.copy_(codebook)
            model._VectorQuantizer._ema_w.data.copy_(codebook)
            model._VectorQuantizer._ema_cluster_size.data.fill_(1.0)
        else:
            model._VectorQuantizer._embedding.weight.data.copy_(codebook)
    print(f"Codebook initialized with {num_embeddings} vectors.")


def initialize_codebook_from_type(model, embeddings_tensor, initialization_type='random', 
                                random_vector_seed=42, input_layer=8, device=None, codebook_dir=None):
    """
    Dispatcher to initialize the codebook based on the specified type.
    """
    if codebook_dir is not None:
        os.makedirs(codebook_dir, exist_ok=True)
        codebook_path = os.path.join(
            codebook_dir, 
            f"codebook_input-layer_{input_layer}_init_{initialization_type}_K{model._VectorQuantizer._num_embeddings}_seed{random_vector_seed}.pt"
        )
        if os.path.exists(codebook_path):
            print(f"Loading pre-initialized codebook from {codebook_path}")
            codebook_checkpoint = torch.load(codebook_path, map_location='cpu')
            codebook = codebook_checkpoint['codebook'].to(device)
            
            with torch.no_grad():
                if hasattr(model._VectorQuantizer, '_ema_w'):
                    model._VectorQuantizer._embedding.weight.data.copy_(codebook)
                    model._VectorQuantizer._ema_w.data.copy_(codebook)
                    model._VectorQuantizer._ema_cluster_size.data.fill_(1.0)
                else:
                    model._VectorQuantizer._embedding.weight.data.copy_(codebook)
            print("Codebook loaded successfully.")
            return
    print(f"\n--- Starting Codebook Initialization (type: {initialization_type}) ---")
    if initialization_type == 'random':
        initialize_codebook(model, embeddings_tensor, random_vector_seed, device)
    elif initialization_type == 'kmean++':
        initialize_codebook_kmeans_plus_plus(model, embeddings_tensor, device)
    elif initialization_type == 'spherical':
        initialize_spherical_kmeans_plus_plus(model, embeddings_tensor, device)
    else:
        raise ValueError(f"Unknown initialization type: '{initialization_type}'. "
                         "Supported types: 'random', 'kmean++', 'spherical'")
    
    if codebook_dir is not None:
        codebook_path = os.path.join(
            codebook_dir, 
            f"codebook_input-layer_{input_layer}_init_{initialization_type}_K{model._VectorQuantizer._num_embeddings}_seed{random_vector_seed}.pt"
        )
        
        with torch.no_grad():
            if hasattr(model._VectorQuantizer, '_ema_w'):
                codebook = model._VectorQuantizer._embedding.weight.data.cpu()
            else:
                codebook = model._VectorQuantizer._embedding.weight.data.cpu()
        
        torch.save({'codebook': codebook}, codebook_path)
        print(f"Initialized codebook saved to {codebook_path}")