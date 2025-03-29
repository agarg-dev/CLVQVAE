import torch
import torch.nn as nn
import torch.nn.functional as F
from .vector_quantizer import VectorQuantizer, VectorQuantizerEMA
import os


class AdaptiveResidualEncoder(nn.Module):
    """Encoder that acts as an adaptive identity function with learnable residual.
    
    Preserves input distribution properties while allowing gradient flow.
    Includes layer normalization on the transformation path for stability.
    
    Args:
        embedding_dim (int): Dimension of the embeddings
    """
    def __init__(self, embedding_dim):
        super(AdaptiveResidualEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.a = nn.Parameter(torch.tensor(0.2))
        # Initialize as identity transform
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, x, padding_mask=None):
        alpha = torch.sigmoid(self.a)
        # Apply transformation with normalization
        transformed = self.layer_norm(self.linear(x))
        result = (1 - alpha) * x + alpha * transformed
        if padding_mask is not None:
            result = result.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return result



class ContinuousEmbeddings(nn.Module):
    """A simple layer that passes through continuous embeddings.
    
    Args:
        embedding_dim (int): Dimension of the embeddings (default: 768)
    """
    def __init__(self, embedding_dim=768):
        super(ContinuousEmbeddings, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, continuousEmbedding):
        return continuousEmbedding



class Decoder(nn.Module):
    """Transformer decoder that reconstructs the target embeddings from quantized vectors.
    
    Args:
        d_model (int): The dimension of the model's internal representation
        output_dim (int): The dimension of the output embeddings
        nhead (int): Number of attention heads
        num_layers (int): Number of decoder layers
        dim_feedforward (int): Dimension of feedforward network (default: 2048)
        dropout (float): Dropout probability (default: 0.1)
    """
    def __init__(self, d_model, output_dim, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(Decoder, self).__init__()
        
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer,
            num_layers=num_layers
        )
        # Output projection now projects to the output dimension
        self.output_projection = nn.Linear(d_model, output_dim)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, 
                memory_key_padding_mask=None):
        decoder_output = self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        reconstructed = self.output_projection(decoder_output)
        return reconstructed



def generate_square_subsequent_mask(sz, device=None):
    """Generate a square mask for the sequence. True = masked positions (don't attend).
    
    Args:
        sz (int): Size of square mask
        device (torch.device): Device to create mask on
        
    Returns:
        torch.Tensor: Boolean mask tensor of shape [sz, sz]
    """
    mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)
    mask = mask.transpose(0, 1)
    return mask

class Model(nn.Module):
    """Vector Quantized VAE model with sampling-based quantization
    
    This model takes continuous embeddings as input, quantizes them using a learned codebook,
    and reconstructs the target embeddings using a Transformer decoder. It supports
    sampling from top-k nearest codebook vectors for exploration.
    
    Args:
        num_embeddings (int): Size of the codebook (number of discrete codes)
        embedding_dim (int): Dimension of each input embedding vector
        output_dim (int, optional): Dimension of the output embedding vectors.
                                   If not provided, uses embedding_dim
        device (torch.device): Device to run the model on
        use_ema (bool): Whether to use EMA updates for codebook (default: True)
        perplexity_weight (float): Weight for perplexity loss
        use_sampling (bool): Whether to use sampling instead of deterministic selection
        top_k (int): Number of top candidates to consider for sampling
        temperature (float): Temperature parameter for sampling (higher = more exploration)
        use_adaptive_encoder (bool)
    """
    def __init__(self, num_embeddings, embedding_dim, output_dim=None, device=None, 
                use_ema=True, perplexity_weight=0.01, use_sampling=True, top_k=10, temperature=1.0,
                use_adaptive_encoder=False): 

        super(Model, self).__init__()
        # Set output_dim to embedding_dim if not provided
        if output_dim is None:
            output_dim = embedding_dim
        if use_adaptive_encoder:
            self._ContinuousEmbedding = AdaptiveResidualEncoder(embedding_dim=embedding_dim)
        else:
            # Use pass-through layer
            self._ContinuousEmbedding = ContinuousEmbeddings(embedding_dim)

        self.use_adaptive_encoder = use_adaptive_encoder
        # Sampling parameters
        self._use_sampling = use_sampling
        self._top_k = top_k
        self._temperature = temperature
        if use_ema:
            self._VectorQuantizer = VectorQuantizerEMA(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=0.0,
                decay=0.99,
                epsilon=1e-5,
                perplexity_weight=perplexity_weight,
                use_sampling=use_sampling,
                top_k=top_k,
                temperature=temperature
            )
        else:
            self._VectorQuantizer = VectorQuantizer(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                perplexity_weight=perplexity_weight,
                use_sampling=use_sampling,
                top_k=top_k,
                temperature=temperature
            )
            
        self._decoder = Decoder(
            d_model=embedding_dim,
            output_dim=output_dim,
            nhead=8,
            num_layers=6
        )
       

    def forward(self, continuousEmbedding, target_embedding=None, device=None):
        """Forward pass of the model.
        
        Args:
            continuousEmbedding (torch.Tensor): Input continuous embeddings 
                of shape [batch_size, seq_len, embedding_dim]
            target_embedding (torch.Tensor, optional): Target embeddings for reconstruction
                of shape [batch_size, seq_len, output_dim]. Not used directly in the model, 
                but provided for convenience in loss calculation.
            device (torch.device, optional): Device to run forward pass on
                
        Returns:
            dict: Dictionary containing model outputs and losses
        """

        inputs = continuousEmbedding.contiguous()
        padding_mask = torch.norm(inputs, dim=2) <= 1e-6
        if self.use_adaptive_encoder:
            z_e = self._ContinuousEmbedding(inputs, padding_mask=padding_mask)
        else:
            z_e = self._ContinuousEmbedding(inputs)
        # Apply padding mask
        z_e = z_e.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        # Vector quantization
        vq_output = self._VectorQuantizer(z_e)
        quantized = vq_output["quantized"]
        # Convert to [seq_len, batch_size, embedding_dim] for decoder
        z_e = z_e.transpose(0, 1)
        quantized = quantized.transpose(0, 1)
        # Generate mask
        tgt_len = quantized.size(0)
        tgt_mask = generate_square_subsequent_mask(tgt_len, device)
        quantized = quantized.to(device)
        z_e = z_e.to(device)
        tgt_mask = tgt_mask.to(device)
        key_padding_mask = padding_mask.to(device if device else continuousEmbedding.device)
        # Reconstruction
        reconstructed = self._decoder(
            tgt=quantized,
            memory=z_e,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=key_padding_mask,  # For target
            memory_key_padding_mask=key_padding_mask  # For encoder memory
        )
        # back to [batch_size, seq_len, embedding_dim]
        reconstructed = reconstructed.transpose(0, 1)
        z_e = z_e.transpose(0, 1)
        
        return {
            "loss": vq_output["loss"],
            "z_e": z_e,
            "reconstructed": reconstructed,
            "quantized": quantized.transpose(0, 1),
            "encoding_indices": vq_output["encoding_indices"],
            "indices": vq_output["indices"],
            "min_distances": vq_output["min_distances"],
            "perplexity": vq_output.get("perplexity", None),
            "similarity_metric": vq_output.get("similarity_metric"),
            "perplexity_loss": vq_output.get("perplexity_loss")
        }

    def get_codebook_usage(self):
        """
        Get codebook usage statistics.
        
        Returns:
            dict: Dictionary with codebook usage statistics from the VectorQuantizer
        """
        return self._VectorQuantizer.get_usage_stats()
    
    def reset_codebook_usage(self):
        """
        Reset codebook usage counters.
        """
        self._VectorQuantizer.reset_usage_stats()
    
    def analyze_codebook(self):
        """
        Analyze the current state of the codebook.
        
        Returns:
            dict: Dictionary containing basic codebook analysis metrics
        """
        usage_stats = self.get_codebook_usage()
        active_codes = usage_stats["active_codes"]
        total_codes = usage_stats["total_codes"]
        code_utilization = usage_stats["code_utilization"]
        
        # Get top used codes
        top_indices, top_counts = [], []
        if "usage_count" in usage_stats:
            usage_counts = usage_stats["usage_count"]
            # Get top 10 indices by usage count
            if len(usage_counts) > 0:
                _, indices = torch.topk(usage_counts, k=min(10, len(usage_counts)))
                top_indices = indices.tolist()
                top_counts = usage_counts[indices].tolist()
        
        # Get unused code count
        unused_count = total_codes - active_codes
        
        return {
            "active_codes": active_codes,
            "total_codes": total_codes,
            "utilization_percentage": code_utilization * 100,
            "unused_codes": unused_count,
            "top_used_codes": dict(zip(top_indices, top_counts)),
        }