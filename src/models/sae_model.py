import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SparseAutoencoder(nn.Module):
    """
    Basic Sparse Autoencoder for cross-layer concept discovery.
    
    Args:
        input_dim (int): Dimension of input embeddings
        hidden_dim (int): Dimension of hidden sparse layer (typically larger than input_dim)
        output_dim (int): Dimension of output embeddings  
        sparsity_weight (float): Weight for L1 sparsity loss
        tie_weights (bool): Whether to tie encoder and decoder weights (decoder = encoder.T)
    """
    def __init__(self, input_dim, hidden_dim, output_dim=None, sparsity_weight=0.01):
        super(SparseAutoencoder, self).__init__()
        
        if output_dim is None:
            output_dim = input_dim
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sparsity_weight = sparsity_weight
        
        # Separate encoder and decoder layers
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, output_dim, bias=True)
        
        # Track activation statistics for analysis
        self.register_buffer('activation_count', torch.zeros(hidden_dim))
        self.register_buffer('total_samples', torch.tensor(0.0))
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.constant_(self.encoder.bias, 0.1) 
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
    
    def encode(self, x):
        """Encode input to sparse hidden representation"""
        # Linear transformation followed by ReLU for sparsity
        hidden = F.relu(self.encoder(x))
        return hidden
    
    def decode(self, hidden):
        """Decode sparse hidden representation to output using separate decoder"""
        return self.decoder(hidden)
    
    def forward(self, x, update_stats=True):
        """
        Forward pass of the Sparse Autoencoder
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim]
            update_stats (bool): Whether to update activation statistics
            
        Returns:
            dict: Dictionary containing outputs and sparsity loss only
        """
        input_shape = x.shape
        
        # Flatten input for processing: [batch_size * seq_len, input_dim]
        x_flat = x.view(-1, self.input_dim)
        
        # Create padding mask to exclude padding tokens
        padding_mask = torch.norm(x_flat, dim=1) <= 1e-6
        valid_mask = ~padding_mask
        
        # Process only valid (non-padding) tokens
        if valid_mask.sum() == 0:
            # Handle case where all tokens are padding
            hidden_flat = torch.zeros(x_flat.size(0), self.hidden_dim, device=x.device)
            reconstructed_flat = torch.zeros(x_flat.size(0), self.output_dim, device=x.device)
            sparsity_loss = torch.tensor(0.0, device=x.device)
        else:
            x_valid = x_flat[valid_mask]
            
            # Encode valid tokens
            hidden_valid = self.encode(x_valid)
            
            # Update activation statistics
            if update_stats:
                # Count how many times each neuron is active (> threshold)
                active_neurons = (hidden_valid > 1e-4).float()
                self.activation_count += active_neurons.sum(dim=0)
                self.total_samples += x_valid.size(0)
            
            # Decode valid tokens
            reconstructed_valid = self.decode(hidden_valid)
            
            # Create full tensors and fill in valid positions
            hidden_flat = torch.zeros(x_flat.size(0), self.hidden_dim, device=x.device)
            reconstructed_flat = torch.zeros(x_flat.size(0), self.output_dim, device=x.device)
            
            hidden_flat[valid_mask] = hidden_valid
            reconstructed_flat[valid_mask] = reconstructed_valid
            
            # Calculate L1 sparsity loss on valid tokens only
            sparsity_loss = torch.mean(torch.sum(torch.abs(hidden_valid), dim=1))
        
        # Reshape back to original dimensions
        hidden = hidden_flat.view(input_shape[0], input_shape[1], self.hidden_dim)
        reconstructed = reconstructed_flat.view(input_shape[0], input_shape[1], self.output_dim)
        
        return {
            'hidden': hidden,
            'reconstructed': reconstructed,
            'sparsity_loss': sparsity_loss,
            'padding_mask': padding_mask.view(input_shape[:-1])
        }
    
    def get_activation_stats(self):
        """Get statistics about neuron activation"""
        if self.total_samples > 0:
            activation_frequency = self.activation_count / self.total_samples
            active_neurons = (self.activation_count > 0).sum().item()
        else:
            activation_frequency = torch.zeros_like(self.activation_count)
            active_neurons = 0
            
        return {
            'activation_count': self.activation_count.cpu(),
            'activation_frequency': activation_frequency.cpu(),
            'active_neurons': active_neurons,
            'total_neurons': self.hidden_dim,
            'utilization_rate': active_neurons / self.hidden_dim,
            'total_samples': self.total_samples.item()
        }
    
    def reset_stats(self):
        """Reset activation statistics"""
        self.activation_count.zero_()
        self.total_samples.zero_()
    
    def get_decoder_vectors(self):
        """Get decoder vectors (concept vectors) - shape [hidden_dim, output_dim]
        Each row corresponds to one hidden neuron's decoder vector"""
        return self.decoder.weight.t()  # [output_dim, hidden_dim] -> [hidden_dim, output_dim]
    
    def get_most_active_neuron(self, x):
        """
        Get the most active neuron for each token
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: Most active neuron indices [batch_size, seq_len]
        """
        with torch.no_grad():
            output = self.forward(x, update_stats=True)
            hidden = output['hidden']  # [batch_size, seq_len, hidden_dim]
            
            # Get most active neuron for each token
            most_active = torch.argmax(hidden, dim=-1)  # [batch_size, seq_len]
            
            return most_active, hidden


# class SAEModel(nn.Module):
#     """
#     Simple SAE model for cross-layer concept discovery
    
#     Args:
#         input_dim (int): Dimension of input embeddings
#         hidden_dim (int): Dimension of SAE hidden layer
#         output_dim (int): Dimension of output embeddings
#         sparsity_weight (float): Weight for sparsity loss
#     """
#     def __init__(self, input_dim, hidden_dim, output_dim=None, sparsity_weight=0.01):
#         super(SAEModel, self).__init__()
        
#         if output_dim is None:
#             output_dim = input_dim
            
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
        
#         # Sparse Autoencoder with tied weights
#         self._sae = SparseAutoencoder(
#             input_dim=input_dim,
#             hidden_dim=hidden_dim, 
#             output_dim=output_dim,
#             sparsity_weight=sparsity_weight
#         )
    
#     def forward(self, input_embedding, target_embedding=None, device=None):
#         """
#         Forward pass of the SAE model
        
#         Args:
#             input_embedding (torch.Tensor): Input embeddings [batch_size, seq_len, input_dim]
#             target_embedding (torch.Tensor): Target embeddings for loss calculation
#             device (torch.device): Device for computation
            
#         Returns:
#             dict: Dictionary containing model outputs and losses
#         """
#         inputs = input_embedding.contiguous()
        
#         # SAE forward pass
#         sae_output = self._sae(inputs)
        
#         return {
#             'z_e': inputs,
#             'reconstructed': sae_output['reconstructed'],
#             'hidden': sae_output['hidden'],
#             'sparsity_loss': sae_output['sparsity_loss'],
#             'padding_mask': sae_output['padding_mask']
#             # Remove: 'reconstruction_loss' and 'total_loss'
#         }
    
#     def get_activation_stats(self):
#         """Get SAE activation statistics"""
#         return self._sae.get_activation_stats()
    
#     def reset_stats(self):
#         """Reset SAE statistics"""
#         self._sae.reset_stats()
    
#     def get_concept_vectors_for_tokens(self, input_embedding):
#         """
#         Get concept vectors for the most active SAE neurons for each token
        
#         Args:
#             input_embedding (torch.Tensor): Input embeddings [batch_size, seq_len, input_dim]
            
#         Returns:
#             tuple: (most_active_neurons, concept_vectors, activations)
#         """
#         with torch.no_grad():
#             # Get SAE activations and most active neurons
#             most_active_neurons, activations = self._sae.get_most_active_neuron(input_embedding)
            
#             # Get decoder vectors (concept vectors)
#             decoder_vectors = self._sae.get_decoder_vectors()  # [hidden_dim, output_dim]
            
#             # Get concept vectors for most active neurons
#             batch_size, seq_len = most_active_neurons.shape
#             concept_vectors = torch.zeros(batch_size, seq_len, self.output_dim, device=input_embedding.device)
            
#             # Create padding mask
#             padding_mask = torch.norm(input_embedding, dim=2) <= 1e-6
            
#             for b in range(batch_size):
#                 for s in range(seq_len):
#                     if not padding_mask[b, s]:  # Skip padding tokens
#                         neuron_idx = most_active_neurons[b, s]
#                         concept_vectors[b, s] = decoder_vectors[neuron_idx]
            
#             return most_active_neurons, concept_vectors, activations
    
#     def analyze_sae(self):
#         """Analyze the current state of the SAE"""
#         stats = self.get_activation_stats()
        
#         return {
#             'active_neurons': stats['active_neurons'],
#             'total_neurons': stats['total_neurons'], 
#             'utilization_rate': stats['utilization_rate'],
#             'total_samples': stats['total_samples']
#         }