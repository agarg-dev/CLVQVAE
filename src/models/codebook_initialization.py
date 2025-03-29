import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


def initialize_codebook(model, embeddings_tensor, device=None):
    """Initialize the codebook by randomly sampling from input embeddings.
    
    This method initializes the codebook vectors by randomly sampling from the input
    embeddings, excluding zero vectors (padding). This can help start training from
    a better initialization than random.
    
    Args:
        model: The Model instance with a VectorQuantizer component
        embeddings_tensor (torch.Tensor): Tensor of embeddings to sample from
        device (torch.device, optional): Device to initialize codebook on
    """
    device = device or next(model.parameters()).device
    embeddings_tensor = embeddings_tensor.to(device)
    print("Initializing codebook vectors...")
    num_embeddings = model._VectorQuantizer._num_embeddings
    embedding_dim = model._VectorQuantizer._embedding_dim
    # Reshape to (-1, embedding_dim) to get all vectors
    flat_embeddings = embeddings_tensor.view(-1, embedding_dim)
    # Remove zero vectors (padding)
    mask = torch.any(flat_embeddings != 0, dim=1)
    flat_embeddings = flat_embeddings[mask]
    total_vectors = flat_embeddings.size(0)
    print(f"Total input vectors for initialization: {total_vectors}")
    # Randomly sample num_embeddings vectors from input features
    indices = torch.randperm(total_vectors)[:num_embeddings]
    codebook = flat_embeddings[indices]
    # Update embeddings based on quantizer type
    if hasattr(model._VectorQuantizer, '_ema_w'):  # Check if it's EMA quantizer
        model._VectorQuantizer._embedding.data = codebook.to(device)
        model._VectorQuantizer._ema_w.data = codebook.clone().to(device)
        model._VectorQuantizer._ema_cluster_size.data = torch.ones(
            num_embeddings, device=device
        )
    else:
        model._VectorQuantizer._embedding.weight.data = codebook.to(device)
    print(f"Codebook initialized with {num_embeddings} vectors")




def initialize_codebook_kmeans_plus_plus(model, embeddings_tensor, device=None):
    """Initialize the codebook using K-means++ algorithm for better vector distribution.
    
    Args:
        model: The Model instance with a VectorQuantizer component
        embeddings_tensor (torch.Tensor): Tensor of embeddings to cluster
        device (torch.device, optional): Device to initialize codebook on
    """
    device = device or next(model.parameters()).device
    embeddings_tensor = embeddings_tensor.to(device)
    print("Initializing codebook vectors using K-means++...")
    num_embeddings = model._VectorQuantizer._num_embeddings
    embedding_dim = model._VectorQuantizer._embedding_dim
    # Reshape to (-1, embedding_dim) to get all vectors
    flat_embeddings = embeddings_tensor.view(-1, embedding_dim)
    # Remove zero vectors (padding)
    mask = torch.any(flat_embeddings != 0, dim=1)
    flat_embeddings = flat_embeddings[mask]
    total_vectors = flat_embeddings.size(0)
    print(f"Total input vectors for initialization: {total_vectors}")
    # Convert to numpy for sklearn
    embeddings_np = flat_embeddings.cpu().numpy()
    # Apply K-means++ initialization and clustering
    kmeans = KMeans(
        n_clusters=num_embeddings,
        init='k-means++',
        n_init=5,  # Number of times to run k-means with different centroid seeds
        max_iter=100,
        random_state=42
    )
    kmeans.fit(embeddings_np)
    # Get the centroids as codebook vectors
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(device)
    # Update embeddings based on quantizer type
    if hasattr(model._VectorQuantizer, '_ema_w'):  # Check if it's EMA quantizer
        model._VectorQuantizer._embedding.data = centroids
        model._VectorQuantizer._ema_w.data = centroids.clone()
        model._VectorQuantizer._ema_cluster_size.data = torch.ones(
            num_embeddings, device=device
        )
    else:
        model._VectorQuantizer._embedding.weight.data = centroids
    print(f"Codebook initialized with {num_embeddings} vectors using K-means++")
    


def initialize_spherical_kmeans_plus_plus(model, embeddings_tensor, device=None):
    """Initialize the codebook using Spherical K-means++ with cluster-specific magnitude scaling.
    
    This optimizes for directional similarity while preserving magnitude information
    by scaling each centroid by the average magnitude of data points in its cluster.
    
    Args:
        model: The Model instance with a VectorQuantizer component
        embeddings_tensor (torch.Tensor): Tensor of embeddings to cluster
        device (torch.device, optional): Device to initialize codebook on
    """
    device = device or next(model.parameters()).device
    embeddings_tensor = embeddings_tensor.to(device)
    print("Initializing codebook vectors using Spherical K-means++ with cluster-specific scaling...")
    num_embeddings = model._VectorQuantizer._num_embeddings
    embedding_dim = model._VectorQuantizer._embedding_dim
    # Reshape to (-1, embedding_dim) to get all vectors
    flat_embeddings = embeddings_tensor.view(-1, embedding_dim)
    # Remove zero vectors (padding)
    mask = torch.any(flat_embeddings != 0, dim=1)
    flat_embeddings = flat_embeddings[mask]
    total_vectors = flat_embeddings.size(0)
    print(f"Total input vectors for initialization: {total_vectors}")
    # Calculate and store original norms before normalization
    norms = torch.norm(flat_embeddings, p=2, dim=1, keepdim=True)
    avg_norm = norms.mean().item()
    # Normalize embeddings to unit length (sphere)
    normalized_embeddings = flat_embeddings / (norms + 1e-8)  # Avoid division by zero
    # Convert to numpy for sklearn
    normalized_np = normalized_embeddings.cpu().numpy()
    original_np = flat_embeddings.cpu().numpy()
    # Apply K-means++ initialization and clustering on the normalized vectors
    kmeans = KMeans(
        n_clusters=num_embeddings,
        init='k-means++',
        n_init=5,
        max_iter=300,
        random_state=42
    )
    kmeans.fit(normalized_np)
    # Get the centroids and cluster assignments
    centroids = kmeans.cluster_centers_
    cluster_assignments = kmeans.labels_
    # Re-normalize centroids to get pure directional vectors
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    normalized_centroids = centroids / (centroid_norms + 1e-8)
    # Apply cluster-specific scaling
    scaled_centroids = np.zeros_like(normalized_centroids)
    # For diagnostics
    cluster_sizes = []
    cluster_avg_norms = []
    # Scale each centroid by average magnitude of points in its cluster
    for i in range(num_embeddings):
        # Find points in this cluster
        cluster_indices = np.where(cluster_assignments == i)[0]
        cluster_size = len(cluster_indices)
        cluster_sizes.append(cluster_size)
        if cluster_size > 0:
            # Get original embeddings for this cluster
            cluster_points = torch.tensor(original_np[cluster_indices])
            # Calculate average magnitude in this cluster
            avg_cluster_norm = torch.norm(cluster_points, p=2, dim=1).mean().item()
            cluster_avg_norms.append(avg_cluster_norm)
            # Scale the normalized centroid by this cluster's average magnitude
            scaled_centroids[i] = normalized_centroids[i] * avg_cluster_norm
        else:
            # For empty clusters (shouldn't happen with k-means++), use global average
            scaled_centroids[i] = normalized_centroids[i] * avg_norm
            cluster_avg_norms.append(avg_norm)
            print(f"Warning: Empty cluster {i} found, using global average magnitude")
    
    # Convert back to PyTorch tensors
    final_centroids = torch.tensor(scaled_centroids, dtype=torch.float).to(device)
    # Update embeddings based on quantizer type
    if hasattr(model._VectorQuantizer, '_ema_w'):  # Check if it's EMA quantizer
        model._VectorQuantizer._embedding.data = final_centroids
        model._VectorQuantizer._ema_w.data = final_centroids.clone()
        model._VectorQuantizer._ema_cluster_size.data = torch.ones(
            num_embeddings, device=device
        )
    else:
        model._VectorQuantizer._embedding.weight.data = final_centroids
    
    # Print statistics about the clustering
    print(f"Codebook initialized with {num_embeddings} vectors using Spherical K-means++")
    print(f"Cluster size statistics: min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
        f"average={sum(cluster_sizes)/len(cluster_sizes):.1f}")
    print(f"Magnitude statistics: min={min(cluster_avg_norms):.2f}, max={max(cluster_avg_norms):.2f}, "
        f"average={sum(cluster_avg_norms)/len(cluster_avg_norms):.2f}")
    # Calculate angular diversity
    with torch.no_grad():
        # Get normalized version of final centroids for angular calculations
        final_normalized = F.normalize(final_centroids, p=2, dim=1)
        cosine_sim = torch.mm(final_normalized, final_normalized.t())
        # Mask out self-similarities (diagonals)
        mask = ~torch.eye(num_embeddings, dtype=bool, device=device)
        mean_cosine = cosine_sim[mask].mean().item()
        print(f"Mean cosine similarity between codebook vectors: {mean_cosine:.4f}")



def initialize_codebook_from_type(model, embeddings_tensor, initialization_type='random', device=None):
    """Initialize the codebook based on the specified initialization type.
    
    Args:
        model: The Model instance with a VectorQuantizer component
        embeddings_tensor (torch.Tensor): Tensor of embeddings to use for initialization
        initialization_type (str): Type of initialization ('random', 'kmean++', 'spherical')
        device (torch.device, optional): Device to initialize codebook on
        
    Raises:
        ValueError: If an unknown initialization type is provided
    """
    if initialization_type == 'random':
        initialize_codebook(model, embeddings_tensor, device)
    elif initialization_type == 'kmean++':
        initialize_codebook_kmeans_plus_plus(model, embeddings_tensor, device)
    elif initialization_type == 'spherical':
        initialize_spherical_kmeans_plus_plus(model, embeddings_tensor, device)
    else:
        raise ValueError(f"Unknown initialization type: {initialization_type}. "
                         "Supported types: 'random', 'kmean++', 'spherical'")