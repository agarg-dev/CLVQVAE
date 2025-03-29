import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class VectorQuantizer(nn.Module):
    """
    Vector Quantization module with sampling-based quantization.
    
    This implementation allows sampling from top-k nearest codebook vectors
    rather than always selecting the nearest vector, adding exploration.
    
    Args:
        num_embeddings (int): Size of the codebook (number of discrete codes)
        embedding_dim (int): Dimension of each embedding vector
        perplexity_weight (float): Weight for the perplexity loss
        use_sampling (bool): Whether to use sampling instead of deterministic selection
        top_k (int): Number of top candidates to consider for sampling
        temperature (float): Temperature parameter for sampling (higher = more exploration)
    """
    def __init__(self, num_embeddings, embedding_dim, perplexity_weight=0.01, 
                use_sampling=True, top_k=10, temperature=1.0):
        super(VectorQuantizer, self).__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._perplexity_weight = perplexity_weight
        # Sampling parameters
        self._use_sampling = use_sampling
        self._top_k = min(top_k, num_embeddings)  # Ensure top_k doesn't exceed codebook size
        self._temperature = temperature
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        # Add counter for usage statistics
        self.register_buffer('_usage_count', torch.zeros(num_embeddings))
    


    def get_usage_stats(self):
        """
        Get statistics about codebook usage.
        
        Returns:
            dict: A dictionary containing usage statistics
        """
        usage_count = self._usage_count.detach().cpu()
        total_usage = usage_count.sum().item()
        if total_usage == 0:
            usage_fraction = torch.zeros_like(usage_count)
        else:
            usage_fraction = usage_count / total_usage
        active_codes = torch.sum(usage_count > 0).item()
        return {
            "usage_count": usage_count,
            "usage_fraction": usage_fraction,
            "active_codes": active_codes,
            "total_codes": self._num_embeddings,
            "code_utilization": active_codes / self._num_embeddings,
        }



    def compute_similarity_metrics(self):
        """
        Compute cosine similarity metrics between codebook vectors.
        
        Returns:
            dict: A dictionary containing similarity statistics
        """
        # Extract embeddings
        embeddings = self._embedding.weight.data.detach().cpu().numpy()
        # Get active vectors (used at least once)
        active_indices = torch.where(self._usage_count > 0)[0].cpu().numpy()
        if len(active_indices) < 2:
            return {
                "cosine_mean_similarity": 0.0,
                "cosine_min_similarity": 0.0,
                "cosine_max_similarity": 0.0,
                "euclidean_mean_distance": 0.0,
                "euclidean_min_distance": 0.0,
                "euclidean_max_distance": 0.0,
            }
        active_embeddings = embeddings[active_indices]
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(active_embeddings)
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        cosine_non_diagonal = similarity_matrix[mask]
        # Compute Euclidean distance matrix
        distance_matrix = euclidean_distances(active_embeddings)
        euclidean_non_diagonal = distance_matrix[mask]
        
        return {
            "cosine_mean_similarity": float(np.mean(cosine_non_diagonal)),
            "cosine_min_similarity": float(np.min(cosine_non_diagonal)),
            "cosine_max_similarity": float(np.max(cosine_non_diagonal)),
            "euclidean_mean_distance": float(np.mean(euclidean_non_diagonal)),
            "euclidean_min_distance": float(np.min(euclidean_non_diagonal)),
            "euclidean_max_distance": float(np.max(euclidean_non_diagonal)),
        }



    
    def reset_usage_stats(self):
        """Reset the usage statistics."""
        self._usage_count.zero_()
    



    def sample_from_distances(self, distances):
        """
        Sample from top-k nearest vectors based on distance.
        
        Args:
            distances (torch.Tensor): Distance matrix of shape [batch_size, num_embeddings]
            
        Returns:
            Tuple: (sampled_indices, min_distances)
        """
        batch_size = distances.shape[0]
        # Get top-k indices with smallest distances
        top_k_distances, top_k_indices = torch.topk(
            distances, k=self._top_k, dim=1, largest=False
        )
        # Convert distances to probabilities via softmax
        logits = -top_k_distances / self._temperature
        probs = F.softmax(logits, dim=1)
        # Sample from the probability distribution
        sampled_indices = torch.zeros(batch_size, dtype=torch.long, device=distances.device)
        min_distances = torch.zeros(batch_size, device=distances.device)
        for i in range(batch_size):
            sample_idx = torch.multinomial(probs[i], 1).item()
            sampled_indices[i] = top_k_indices[i, sample_idx]
            min_distances[i] = top_k_distances[i, sample_idx]
        
        return sampled_indices, min_distances
    



    def forward(self, inputs):
        """
        Forward pass of the Vector Quantizer, excluding padding tokens.
        With optional sampling from top-k nearest codebook vectors.
        """

        inputs = inputs.to(self._embedding.weight.device)
        inputs = inputs.contiguous()
        input_shape = inputs.shape
        
        # Flatten input except for the last dimension
        flat_input = inputs.view(-1, self._embedding_dim)
        # Create a padding mask - use vector norm to identify padding vectors
        valid_mask = torch.norm(flat_input, dim=1) > 1e-6
        valid_flat_input = flat_input[valid_mask]
        # Calculate distances between valid inputs and embedding vectors
        distances = (
            torch.sum(valid_flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(valid_flat_input, self._embedding.weight.t())
        )
        # Choose between sampling and deterministic selection
        if self._use_sampling and self.training:
            encoding_indices_valid, min_distances_valid = self.sample_from_distances(distances)
        else:
            min_distances_valid, encoding_indices_valid = torch.min(distances, dim=1) 
        encodings_valid = F.one_hot(encoding_indices_valid, self._num_embeddings).float()
        # Update usage statistics only for valid tokens
        usage_count = torch.sum(encodings_valid, dim=0)
        self._usage_count += usage_count
        # Quantize the valid inputs
        quantized_valid = self._embedding(encoding_indices_valid)
        # Initialize full-sized outputs with zeros
        quantized_flat = torch.zeros_like(flat_input)
        indices_flat = torch.zeros(flat_input.shape[0], dtype=torch.long, device=flat_input.device)
        min_distances_flat = torch.zeros(flat_input.shape[0], device=flat_input.device)
        # Place valid outputs in the correct positions
        quantized_flat[valid_mask] = quantized_valid
        indices_flat[valid_mask] = encoding_indices_valid
        min_distances_flat[valid_mask] = min_distances_valid
        # Reshape to original dimensions
        quantized = quantized_flat.view(input_shape)
        indices = indices_flat.view(input_shape[:-1])
        min_distances = min_distances_flat.view(input_shape[:-1])
        # Calculate losses only for non-padding tokens
        loss_vq = F.mse_loss(quantized_valid, valid_flat_input.detach(), reduction="mean")
        loss_commit = F.mse_loss(valid_flat_input, quantized_valid.detach(), reduction="mean")
        loss = loss_commit * 0 + loss_vq
        # Straight-through estimator
        quantized_st = inputs + (quantized - inputs).detach()
        # Calculate perplexity of the code distribution for non-padding tokens
        avg_probs = torch.mean(encodings_valid, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        # Target perplexity is the maximum possible value (num_embeddings)
        perplexity_loss = -torch.log(perplexity + 1e-10)
        total_loss = loss + self._perplexity_weight * perplexity_loss

        return {
            "quantized": quantized_st,
            "loss": total_loss,
            "encoding_indices": indices_flat,
            "indices": indices,
            "min_distances": min_distances,
            "loss_commit": loss_commit,
            "perplexity_loss": perplexity_loss,
            "loss_theta": 0,
            "perplexity": perplexity,
            "similarity_metric": self.compute_similarity_metrics()
        }




class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average (EMA) updates and sampling-based quantization.
    
    This implementation uses EMA updates instead of gradient descent to update the codebook,
    and allows sampling from top-k nearest codebook vectors for exploration.
    
    Args:
        num_embeddings (int): Size of the codebook (number of discrete codes)
        embedding_dim (int): Dimension of each embedding vector
        perplexity_weight (float): Weight for the perplexity loss
        commitment_cost (float): Coefficient for commitment loss
        decay (float): Decay rate for EMA updates
        epsilon (float): Small constant for numerical stability in EMA updates
        use_sampling (bool): Whether to use sampling instead of deterministic selection
        top_k (int): Number of top candidates to consider for sampling
        temperature (float): Temperature parameter for sampling (higher = more exploration)
    """
    def __init__(self, num_embeddings, embedding_dim, perplexity_weight=0.1, commitment_cost=0.0, 
                decay=0.99, epsilon=1e-5, use_sampling=True, top_k=10, temperature=1.0):
        super(VectorQuantizerEMA, self).__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost
        self._perplexity_weight = perplexity_weight
        # Sampling parameters
        self._use_sampling = use_sampling
        self._top_k = min(top_k, num_embeddings)  # Ensure top_k doesn't exceed codebook size
        self._temperature = temperature
        # EMA related variables
        self.register_buffer('_decay', torch.tensor(decay))
        self.register_buffer('_epsilon', torch.tensor(epsilon))
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        # Embedding initialization
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self.register_buffer('_ema_w', self._embedding.weight.data.clone())
        # Add counter for usage statistics
        self.register_buffer('_usage_count', torch.zeros(num_embeddings))
    


    def get_usage_stats(self):
        """
        Get statistics about codebook usage.
        
        Returns:
            dict: A dictionary containing usage statistics
        """
        usage_count = self._usage_count.detach().cpu()
        total_usage = usage_count.sum().item()
        
        # Avoid division by zero
        if total_usage == 0:
            usage_fraction = torch.zeros_like(usage_count)
        else:
            usage_fraction = usage_count / total_usage    
        active_codes = torch.sum(usage_count > 0).item()
        return {
            "usage_count": usage_count,
            "usage_fraction": usage_fraction,
            "active_codes": active_codes,
            "total_codes": self._num_embeddings,
            "code_utilization": active_codes / self._num_embeddings,
        }




    def compute_similarity_metrics(self):
        """
        Compute cosine similarity metrics between codebook vectors.
        
        Returns:
            dict: A dictionary containing similarity statistics
        """
        # Extract embeddings
        embeddings = self._embedding.weight.data.detach().cpu().numpy()
        # Get active vectors (used at least once)
        active_indices = torch.where(self._usage_count > 0)[0].cpu().numpy()
        if len(active_indices) < 2:
            return {
                "cosine_mean_similarity": 0.0,
                "cosine_min_similarity": 0.0,
                "cosine_max_similarity": 0.0,
                "euclidean_mean_distance": 0.0,
                "euclidean_min_distance": 0.0,
                "euclidean_max_distance": 0.0,
            }
        active_embeddings = embeddings[active_indices]
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(active_embeddings)
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        cosine_non_diagonal = similarity_matrix[mask]
        # Compute Euclidean distance matrix
        distance_matrix = euclidean_distances(active_embeddings)
        euclidean_non_diagonal = distance_matrix[mask]
        return {
            "cosine_mean_similarity": float(np.mean(cosine_non_diagonal)),
            "cosine_min_similarity": float(np.min(cosine_non_diagonal)),
            "cosine_max_similarity": float(np.max(cosine_non_diagonal)),
            "euclidean_mean_distance": float(np.mean(euclidean_non_diagonal)),
            "euclidean_min_distance": float(np.min(euclidean_non_diagonal)),
            "euclidean_max_distance": float(np.max(euclidean_non_diagonal)),
        }


    def reset_usage_stats(self):
        """Reset the usage statistics."""
        self._usage_count.zero_()


    
    def sample_from_distances(self, distances):
        """
        Sample from top-k nearest vectors based on distance.
        
        Args:
            distances (torch.Tensor): Distance matrix of shape [batch_size, num_embeddings]
            
        Returns:
            Tuple: (sampled_indices, min_distances)
        """
        batch_size = distances.shape[0]
        # Get top-k indices with smallest distances
        top_k_distances, top_k_indices = torch.topk(
            distances, k=self._top_k, dim=1, largest=False
        )
        # Convert distances to probabilities via softmax
        logits = -top_k_distances / self._temperature
        probs = F.softmax(logits, dim=1)
        # Sample from the probability distribution
        sampled_indices = torch.zeros(batch_size, dtype=torch.long, device=distances.device)
        min_distances = torch.zeros(batch_size, device=distances.device)
        for i in range(batch_size):
            sample_idx = torch.multinomial(probs[i], 1).item()
            sampled_indices[i] = top_k_indices[i, sample_idx]
            min_distances[i] = top_k_distances[i, sample_idx]
        
        return sampled_indices, min_distances



    def forward(self, inputs):
        """
        Forward pass of the Vector Quantizer with EMA updates, excluding padding tokens.
        With optional sampling from top-k nearest codebook vectors.
        """
        inputs = inputs.to(self._embedding.weight.device)
        inputs = inputs.contiguous()
        input_shape = inputs.shape
        # Flatten input except for the last dimension
        flat_input = inputs.view(-1, self._embedding_dim)
        # Create a padding mask - use vector norm to identify padding vectors
        valid_mask = torch.norm(flat_input, dim=1) > 1e-6
        valid_flat_input = flat_input[valid_mask]
        # Calculate distances between valid inputs and embedding vectors
        distances = (
            torch.sum(valid_flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(valid_flat_input, self._embedding.weight.t())
        )
        # Choose between sampling and deterministic selection
        if self._use_sampling and self.training:
            # Sample from top-k during training
            encoding_indices_valid, min_distances_valid = self.sample_from_distances(distances)
        else:
            # Default behavior: deterministic selection of nearest vector
            min_distances_valid, encoding_indices_valid = torch.min(distances, dim=1)
        encodings_valid = F.one_hot(encoding_indices_valid, self._num_embeddings).float()
        # Update usage statistics only for valid tokens
        usage_count = torch.sum(encodings_valid, dim=0)
        self._usage_count += usage_count
        # Update the embedding vectors using EMA during training (only for valid tokens)
        if self.training:
            # Update cluster size EMA
            cluster_size = torch.sum(encodings_valid, dim=0)
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * cluster_size
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon) /
                (n + self._num_embeddings * self._epsilon) * n
            )
            # Update embedding EMA
            dw = torch.matmul(encodings_valid.t(), valid_flat_input)
            self._ema_w = self._ema_w * self._decay + (1 - self._decay) * dw
            # Normalize embedding weights by cluster size
            self._embedding.weight.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)
        # Quantize the valid inputs
        quantized_valid = self._embedding(encoding_indices_valid)
        # Initialize full-sized outputs with zeros
        quantized_flat = torch.zeros_like(flat_input)
        indices_flat = torch.zeros(flat_input.shape[0], dtype=torch.long, device=flat_input.device)
        min_distances_flat = torch.zeros(flat_input.shape[0], device=flat_input.device)
        # Place valid outputs in the correct positions
        quantized_flat[valid_mask] = quantized_valid
        indices_flat[valid_mask] = encoding_indices_valid
        min_distances_flat[valid_mask] = min_distances_valid
        # Reshape to original dimensions
        quantized = quantized_flat.view(input_shape)
        indices = indices_flat.view(input_shape[:-1])
        min_distances = min_distances_flat.view(input_shape[:-1])
        # Calculate commitment loss only for non-padding tokens
        loss_commit = F.mse_loss(valid_flat_input, quantized_valid.detach(), reduction="mean")
        loss = self._commitment_cost * loss_commit
        # Straight-through estimator
        quantized_st = inputs + (quantized - inputs).detach()
        # Calculate perplexity of the code distribution for non-padding tokens
        avg_probs = torch.mean(encodings_valid, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        perplexity_loss = -torch.log(perplexity + 1e-10)
        total_loss = loss + self._perplexity_weight * perplexity_loss
        
        return {
            "quantized": quantized_st,
            "loss": total_loss,
            "encoding_indices": indices_flat,
            "indices": indices,
            "min_distances": min_distances,
            "loss_commit": loss_commit,
            "perplexity_loss": perplexity_loss,
            "loss_theta": 0,
            "perplexity": perplexity,
            "similarity_metric": self.compute_similarity_metrics()
        }


