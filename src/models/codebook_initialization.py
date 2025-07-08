import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import os
import random

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
    print("Initializing codebook vectors using Spherical K-means++ with cluster-specific scaling...")
    num_embeddings = model._VectorQuantizer._num_embeddings
    embedding_dim = model._VectorQuantizer._embedding_dim
    
    # Make sure embeddings are on CPU for scikit-learn processing
    embeddings_tensor_cpu = embeddings_tensor.cpu()
    
    # Reshape to (-1, embedding_dim) to get all vectors
    flat_embeddings = embeddings_tensor_cpu.view(-1, embedding_dim)
    
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
    
    # Convert to numpy for sklearn (data already on CPU)
    normalized_np = normalized_embeddings.numpy()
    original_np = flat_embeddings.numpy()
    
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
            cluster_points = original_np[cluster_indices]
            # Calculate average magnitude in this cluster using numpy
            avg_cluster_norm = np.linalg.norm(cluster_points, axis=1).mean()
            cluster_avg_norms.append(avg_cluster_norm)
            # Scale the normalized centroid by this cluster's average magnitude
            scaled_centroids[i] = normalized_centroids[i] * avg_cluster_norm
        else:
            # For empty clusters (shouldn't happen with k-means++), use global average
            scaled_centroids[i] = normalized_centroids[i] * avg_norm
            cluster_avg_norms.append(avg_norm)
            print(f"Warning: Empty cluster {i} found, using global average magnitude")
    
    # Only at the end, convert the final centroids to GPU
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
    
    # Calculate angular diversity - move this calculation to the GPU as the codebook should be small
    with torch.no_grad():
        # Get normalized version of final centroids for angular calculations 
        final_normalized = F.normalize(final_centroids, p=2, dim=1)
        cosine_sim = torch.mm(final_normalized, final_normalized.t())
        # Mask out self-similarities (diagonals)
        mask = ~torch.eye(num_embeddings, dtype=bool, device=device)
        mean_cosine = cosine_sim[mask].mean().item()
        print(f"Mean cosine similarity between codebook vectors: {mean_cosine:.4f}")


def initialize_codebook_kmeans_plus_plus(model, embeddings_tensor, device=None):
    """Initialize the codebook using K-means++ algorithm for better vector distribution.
    
    Args:
        model: The Model instance with a VectorQuantizer component
        embeddings_tensor (torch.Tensor): Tensor of embeddings to cluster
        device (torch.device, optional): Device to initialize codebook on
    """
    device = device or next(model.parameters()).device
    print("Initializing codebook vectors using K-means++...")
    num_embeddings = model._VectorQuantizer._num_embeddings
    embedding_dim = model._VectorQuantizer._embedding_dim
    
    # Make sure embeddings are on CPU for scikit-learn processing
    embeddings_tensor_cpu = embeddings_tensor.cpu()
    
    # Reshape to (-1, embedding_dim) to get all vectors
    flat_embeddings = embeddings_tensor_cpu.view(-1, embedding_dim)
    
    # Remove zero vectors (padding)
    mask = torch.any(flat_embeddings != 0, dim=1)
    flat_embeddings = flat_embeddings[mask]
    total_vectors = flat_embeddings.size(0)
    print(f"Total input vectors for initialization: {total_vectors}")
    
    # Convert to numpy for sklearn (already on CPU)
    embeddings_np = flat_embeddings.numpy()
    
    # Apply K-means++ initialization and clustering
    kmeans = KMeans(
        n_clusters=num_embeddings,
        init='k-means++',
        n_init=5,  # Number of times to run k-means with different centroid seeds
        max_iter=100,
        random_state=42
    )
    kmeans.fit(embeddings_np)
    
    # Get the centroids as codebook vectors and move to device
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


def initialize_codebook(model, embeddings_tensor, seed, device=None):
    """Initialize the codebook by randomly sampling from input embeddings.
    
    This method initializes the codebook vectors by randomly sampling from the input
    embeddings, excluding zero vectors (padding). This can help start training from
    a better initialization than random.
    
    Args:
        model: The Model instance with a VectorQuantizer component
        embeddings_tensor (torch.Tensor): Tensor of embeddings to sample from
        seed (int): Seed for random sampling
        device (torch.device, optional): Device to initialize codebook on
    """
    # Set seed for random vector selection
    gen = torch.Generator()
    gen.manual_seed(seed)

    device = device or next(model.parameters()).device
    
    # Keep embeddings on CPU for processing
    embeddings_tensor_cpu = embeddings_tensor.cpu()
    
    print("Initializing codebook vectors...")
    num_embeddings = model._VectorQuantizer._num_embeddings
    embedding_dim = model._VectorQuantizer._embedding_dim
    
    # Reshape to (-1, embedding_dim) to get all vectors
    flat_embeddings = embeddings_tensor_cpu.view(-1, embedding_dim)
    
    # Remove zero vectors (padding)
    mask = torch.any(flat_embeddings != 0, dim=1)
    flat_embeddings = flat_embeddings[mask]
    total_vectors = flat_embeddings.size(0)
    print(f"Total input vectors for initialization: {total_vectors}")
    
    # Randomly sample num_embeddings vectors from input features
    indices = torch.randperm(total_vectors, generator=gen)[:num_embeddings]
    codebook = flat_embeddings[indices].to(device)
    
    # Update embeddings based on quantizer type
    if hasattr(model._VectorQuantizer, '_ema_w'):  # Check if it's EMA quantizer
        model._VectorQuantizer._embedding.data = codebook
        model._VectorQuantizer._ema_w.data = codebook.clone()
        model._VectorQuantizer._ema_cluster_size.data = torch.ones(
            num_embeddings, device=device
        )
    else:
        model._VectorQuantizer._embedding.weight.data = codebook
    
    print(f"Codebook initialized with {num_embeddings} vectors")


def initialize_codebook_kmedoids_plus_plus(model, embeddings_tensor, samples_size=1000, n_samples=5, device=None):
    """Initialize the codebook using K-medoids++ algorithm with CLARA for large datasets.
    
    This implementation uses CLARA (Clustering LARge Applications) for scalability with
    large datasets while maintaining the benefits of K-medoids (actual data points as centers).
    
    Args:
        model: The Model instance with a VectorQuantizer component
        embeddings_tensor (torch.Tensor): Tensor of embeddings to cluster
        samples_size (int): Size of each sample for CLARA
        n_samples (int): Number of samples to take
        device (torch.device, optional): Device to initialize codebook on
    """
    device = device or next(model.parameters()).device
    print(f"Initializing codebook vectors using K-medoids++ with CLARA (samples={n_samples}, size={samples_size})...")
    num_embeddings = model._VectorQuantizer._num_embeddings
    embedding_dim = model._VectorQuantizer._embedding_dim
    
    # Process on CPU
    embeddings_tensor_cpu = embeddings_tensor.cpu()
    flat_embeddings = embeddings_tensor_cpu.view(-1, embedding_dim)
    
    # Remove zero vectors (padding)
    mask = torch.any(flat_embeddings != 0, dim=1)
    flat_embeddings = flat_embeddings[mask]
    total_vectors = flat_embeddings.size(0)
    print(f"Total input vectors for initialization: {total_vectors}")
    
    # Convert to numpy for sklearn
    embeddings_np = flat_embeddings.numpy()
    
    # CLARA algorithm for K-medoids
    best_medoids = None
    best_cost = float('inf')
    best_labels = None
    
    for i in range(n_samples):
        print(f"Processing sample {i+1}/{n_samples}...")
        
        # Take a random sample
        if total_vectors > samples_size:
            sample_indices = np.random.choice(total_vectors, samples_size, replace=False)
            sample = embeddings_np[sample_indices]
        else:
            sample = embeddings_np
            sample_indices = np.arange(total_vectors)
            
        # Apply K-medoids to the sample
        kmedoids = KMedoids(
            n_clusters=num_embeddings,
            metric='euclidean',
            method='pam',  # PAM algorithm (Partitioning Around Medoids)
            init='k-medoids++',
            max_iter=100,
            random_state=42
        )
        kmedoids.fit(sample)
        
        # Get medoids for this sample
        sample_medoids = sample[kmedoids.medoid_indices_]
        
        # Assign all points in the full dataset to these medoids
        # and calculate the cost
        all_distances = np.zeros((total_vectors, num_embeddings))
        for j, medoid in enumerate(sample_medoids):
            # Calculate distance from each point to this medoid
            all_distances[:, j] = np.linalg.norm(embeddings_np - medoid, axis=1)
        
        # Find closest medoid for each point
        labels = np.argmin(all_distances, axis=1)
        cost = np.sum(np.min(all_distances, axis=1))
        
        # If this is the best solution so far, keep it
        if cost < best_cost:
            best_cost = cost
            best_medoids = sample_medoids
            best_labels = labels
    
    # Convert best medoids to tensor and move to device
    centroids = torch.tensor(best_medoids, dtype=torch.float).to(device)
    
    # Update embeddings based on quantizer type
    if hasattr(model._VectorQuantizer, '_ema_w'):  # Check if it's EMA quantizer
        model._VectorQuantizer._embedding.data = centroids
        model._VectorQuantizer._ema_w.data = centroids.clone()
        model._VectorQuantizer._ema_cluster_size.data = torch.ones(
            num_embeddings, device=device
        )
    else:
        model._VectorQuantizer._embedding.weight.data = centroids
    
    # Calculate cluster sizes for diagnostic info if we have labels
    if best_labels is not None:
        cluster_sizes = []
        for i in range(num_embeddings):
            cluster_size = np.sum(best_labels == i)
            cluster_sizes.append(cluster_size)
        
        print(f"Codebook initialized with {num_embeddings} vectors using K-medoids++ with CLARA")
        print(f"Cluster size statistics: min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
            f"average={sum(cluster_sizes)/len(cluster_sizes):.1f}")
    else:
        print(f"Codebook initialized with {num_embeddings} vectors using K-medoids++ with CLARA")


def initialize_spherical_kmedoids_plus_plus(model, embeddings_tensor, samples_size=1000, n_samples=5, device=None):
    """Initialize the codebook using Spherical K-medoids++ with CLARA for large datasets.
    
    This optimizes for directional similarity while preserving the original
    magnitudes of the selected medoids, using CLARA for scalability.
    
    Args:
        model: The Model instance with a VectorQuantizer component
        embeddings_tensor (torch.Tensor): Tensor of embeddings to cluster
        samples_size (int): Size of each sample for CLARA
        n_samples (int): Number of samples to take
        device (torch.device, optional): Device to initialize codebook on
    """
    device = device or next(model.parameters()).device
    print(f"Initializing codebook vectors using Spherical K-medoids++ with CLARA (samples={n_samples}, size={samples_size})...")
    num_embeddings = model._VectorQuantizer._num_embeddings
    embedding_dim = model._VectorQuantizer._embedding_dim
    
    # Process on CPU
    embeddings_tensor_cpu = embeddings_tensor.cpu()
    flat_embeddings = embeddings_tensor_cpu.view(-1, embedding_dim)
    
    # Remove zero vectors (padding)
    mask = torch.any(flat_embeddings != 0, dim=1)
    flat_embeddings = flat_embeddings[mask]
    total_vectors = flat_embeddings.size(0)
    print(f"Total input vectors for initialization: {total_vectors}")
    
    # Calculate and store original norms
    norms = torch.norm(flat_embeddings, p=2, dim=1, keepdim=True)
    
    # Normalize embeddings to unit length (sphere)
    normalized_embeddings = flat_embeddings / (norms + 1e-8)  # Avoid division by zero
    
    # Convert to numpy for sklearn (data already on CPU)
    normalized_np = normalized_embeddings.numpy()
    original_np = flat_embeddings.numpy()
    
    # CLARA algorithm for Spherical K-medoids
    best_medoids = None
    best_cost = float('inf')
    best_medoid_indices = None
    
    for i in range(n_samples):
        print(f"Processing sample {i+1}/{n_samples}...")
        
        # Take a random sample
        if total_vectors > samples_size:
            sample_indices = np.random.choice(total_vectors, samples_size, replace=False)
            normalized_sample = normalized_np[sample_indices]
            original_sample_indices = sample_indices  # Keep track of these for later
        else:
            normalized_sample = normalized_np
            original_sample_indices = np.arange(total_vectors)
            
        # Apply K-medoids to the normalized sample
        kmedoids = KMedoids(
            n_clusters=num_embeddings,
            metric='euclidean',
            method='pam',
            init='k-medoids++',
            max_iter=300,
            random_state=42
        )
        kmedoids.fit(normalized_sample)
        
        # Get indices of medoids within the sample
        sample_medoid_indices = kmedoids.medoid_indices_
        
        # Convert to indices in the original dataset
        original_medoid_indices = original_sample_indices[sample_medoid_indices]
        
        # Get the normalized medoids
        sample_medoids = normalized_np[original_medoid_indices]
        
        # Assign all points in the full normalized dataset to these medoids
        # and calculate the cost
        all_distances = np.zeros((total_vectors, num_embeddings))
        for j, medoid in enumerate(sample_medoids):
            # Calculate distance from each normalized point to this medoid
            # Using Euclidean distance on the unit sphere (equivalent to angular distance)
            all_distances[:, j] = np.linalg.norm(normalized_np - medoid, axis=1)
        
        # Find closest medoid for each point
        cost = np.sum(np.min(all_distances, axis=1))
        
        # If this is the best solution so far, keep it
        if cost < best_cost:
            best_cost = cost
            best_medoid_indices = original_medoid_indices
    
    if best_medoid_indices is not None:
        # Use the original (non-normalized) vectors as the codebook
        # This preserves both direction and magnitude
        best_medoids = original_np[best_medoid_indices]
        
        # Convert medoids to tensor and move to device
        centroids = torch.tensor(best_medoids, dtype=torch.float).to(device)
        
        # Update embeddings based on quantizer type
        if hasattr(model._VectorQuantizer, '_ema_w'):  # Check if it's EMA quantizer
            model._VectorQuantizer._embedding.data = centroids
            model._VectorQuantizer._ema_w.data = centroids.clone()
            model._VectorQuantizer._ema_cluster_size.data = torch.ones(
                num_embeddings, device=device
            )
        else:
            model._VectorQuantizer._embedding.weight.data = centroids
        
        # Calculate cluster assignments for the full dataset
        all_distances = np.zeros((total_vectors, num_embeddings))
        for j, medoid in enumerate(best_medoids):
            all_distances[:, j] = np.linalg.norm(original_np - medoid, axis=1)
        cluster_assignments = np.argmin(all_distances, axis=1)
        
        # Calculate cluster sizes
        cluster_sizes = []
        for j in range(num_embeddings):
            cluster_size = np.sum(cluster_assignments == j)
            cluster_sizes.append(cluster_size)
        
        # Calculate angular diversity - move this calculation to the GPU
        with torch.no_grad():
            # Get normalized version of centroids for angular calculations 
            normalized_centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)
            cosine_sim = torch.mm(normalized_centroids, normalized_centroids.t())
            # Mask out self-similarities (diagonals)
            mask = ~torch.eye(num_embeddings, dtype=bool, device=device)
            mean_cosine = cosine_sim[mask].mean().item()
        
        print(f"Codebook initialized with {num_embeddings} vectors using Spherical K-medoids++ with CLARA")
        print(f"Cluster size statistics: min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
              f"average={sum(cluster_sizes)/len(cluster_sizes):.1f}")
        print(f"Mean cosine similarity between codebook vectors: {mean_cosine:.4f}")
    else:
        print("Failed to find valid medoids. Falling back to random initialization.")
        initialize_codebook(model, embeddings_tensor, 42, device)


def initialize_codebook_from_type(model, embeddings_tensor, initialization_type='random', 
                                random_vector_seed=42, input_layer=8, device=None, codebook_dir=None,
                                clara_samples_size=1000, clara_n_samples=5):
    """Initialize the codebook based on the specified initialization type.
    
    Args:
        model: The Model instance with a VectorQuantizer component
        embeddings_tensor (torch.Tensor): Tensor of embeddings to use for initialization
        initialization_type (str): Type of initialization ('random', 'kmean++', 'spherical',
                                  'kmedoid++', 'spherical_kmedoid++')
        random_vector_seed (int): Seed for random vector initialization
        input_layer (int): Layer number for codebook path construction
        device (torch.device, optional): Device to initialize codebook on
        codebook_dir (str, optional): Directory to save/load codebook from
        clara_samples_size (int): Size of each sample for K-medoids methods (which use CLARA)
        clara_n_samples (int): Number of samples for K-medoids methods (which use CLARA)
        
    Raises:
        ValueError: If an unknown initialization type is provided
    """
    # If codebook_dir is provided, check if we can load a pre-initialized codebook
    if codebook_dir is not None:
        os.makedirs(codebook_dir, exist_ok=True)
        codebook_path = os.path.join(
            codebook_dir, 
            f"codebook_input-layer_{input_layer}_init_{initialization_type}_K{model._VectorQuantizer._num_embeddings}_seed{random_vector_seed}.pt"
        )
        
        # If a saved codebook exists, load it
        if os.path.exists(codebook_path):
            print(f"Loading pre-initialized codebook from {codebook_path}")
            
            codebook_checkpoint = torch.load(codebook_path)
            codebook = codebook_checkpoint['codebook'].to(device)
            
            # Set the codebook based on quantizer type
            if hasattr(model._VectorQuantizer, '_ema_w'):  # EMA quantizer
                model._VectorQuantizer._embedding.data = codebook
                model._VectorQuantizer._ema_w.data = codebook.clone()
                model._VectorQuantizer._ema_cluster_size.data = torch.ones(
                    model._VectorQuantizer._num_embeddings, device=device
                )
            else:  # Regular quantizer
                model._VectorQuantizer._embedding.weight.data = codebook
                
            print("Codebook loaded successfully.")
            return
    
    # If no codebook was loaded, initialize a new one
    if initialization_type == 'random':
        initialize_codebook(model, embeddings_tensor, random_vector_seed, device)
    elif initialization_type == 'kmean++':
        initialize_codebook_kmeans_plus_plus(model, embeddings_tensor, device)
    elif initialization_type == 'spherical':
        initialize_spherical_kmeans_plus_plus(model, embeddings_tensor, device)
    elif initialization_type == 'kmedoid++':
        initialize_codebook_kmedoids_plus_plus(model, embeddings_tensor, 
                                              samples_size=clara_samples_size,
                                              n_samples=clara_n_samples, 
                                              device=device)
    elif initialization_type == 'spherical_kmedoid++':
        initialize_spherical_kmedoids_plus_plus(model, embeddings_tensor,
                                               samples_size=clara_samples_size,
                                               n_samples=clara_n_samples,
                                               device=device)
    else:
        raise ValueError(f"Unknown initialization type: {initialization_type}. "
                         "Supported types: 'random', 'kmean++', 'spherical', 'kmedoid++', 'spherical_kmedoid++'")
    
    # If codebook_dir is provided, save the newly initialized codebook
    if codebook_dir is not None:
        codebook_path = os.path.join(
            codebook_dir, 
            f"codebook_input-layer_{input_layer}_init_{initialization_type}_K{model._VectorQuantizer._num_embeddings}_seed{random_vector_seed}.pt"
        )
        
        # Extract codebook vectors based on quantizer type
        if hasattr(model._VectorQuantizer, '_ema_w'):  # EMA quantizer
            codebook = model._VectorQuantizer._embedding.data.cpu()
            use_ema = True
        else:  # Regular quantizer
            codebook = model._VectorQuantizer._embedding.weight.data.cpu()
            use_ema = False
        
        # Save the codebook
        torch.save({
            'codebook': codebook,
            'num_embeddings': model._VectorQuantizer._num_embeddings,
            'embedding_dim': model._VectorQuantizer._embedding_dim,
            'initialization': initialization_type,
            'random_vector_seed': random_vector_seed,
            'use_ema': use_ema
        }, codebook_path)
        
        print(f"Initialized codebook saved to {codebook_path}")