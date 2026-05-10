"""
Heterogeneous Graph Neural Network Module
Implements the HGNN with tier-specific projection heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class HeterogeneousGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network with type-specific message passing.
    Used in Phase 2 for distilling LLM-generated memory dynamics.
    """
    
    def __init__(self, 
                 input_dim: int = 256,
                 hidden_dim: int = 256,
                 output_dim: int = 256,
                 num_layers: int = 2,
                 num_edge_types: int = 4,
                 dropout: float = 0.2):
        """
        Initialize HGNN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of message passing layers
            num_edge_types: Number of different edge types
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Initial projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Type-specific transformation matrices for each layer
        self.edge_transforms = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) 
                for _ in range(num_edge_types)
            ])
            for _ in range(num_layers)
        ])
        
        # Self-transformation for each layer
        self.self_transforms = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Tier-specific projection heads
        self.proj_intrinsic = nn.Linear(hidden_dim, output_dim)
        self.proj_collaborative = nn.Linear(hidden_dim, output_dim)
        self.proj_interaction = nn.Linear(hidden_dim, output_dim)
        
        # Attention for neighbor aggregation
        self.attention = nn.ModuleList([
            nn.Linear(hidden_dim * 2, 1)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, 
                node_features: torch.Tensor,
                adjacency_lists: List[Dict[int, List[int]]],
                edge_weights: Optional[List[Dict[int, List[float]]]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of HGNN.
        
        Args:
            node_features: Tensor of shape (num_nodes, input_dim)
            adjacency_lists: List of adjacency dicts per edge type
                           Each dict maps source_idx -> list of target indices
            edge_weights: Optional edge weights per edge type
        
        Returns:
            Tuple of tensors:
            - h_intrinsic: (num_nodes, output_dim)
            - h_collaborative: (num_nodes, output_dim) 
            - h_interaction: (num_nodes, output_dim)
        """
        # Initial projection
        h = self.input_projection(node_features)
        h = F.relu(h)
        
        # Multi-layer message passing
        for layer in range(self.num_layers):
            h_new = self.self_transforms[layer](h)
            
            # Aggregate messages from each edge type
            for edge_type_idx, adj_dict in enumerate(adjacency_lists):
                edge_messages = self._aggregate_neighbors(
                    h, adj_dict, edge_type_idx, layer,
                    edge_weights[edge_type_idx] if edge_weights else None
                )
                h_new = h_new + edge_messages
            
            # Normalize and activate
            h_new = self.layer_norms[layer](h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            
            h = h_new
        
        # Tier-specific projections
        h_intrinsic = self.proj_intrinsic(h)
        h_collaborative = self.proj_collaborative(h)
        h_interaction = self.proj_interaction(h)
        
        return h_intrinsic, h_collaborative, h_interaction
    
    def _aggregate_neighbors(self,
                             node_features: torch.Tensor,
                             adj_dict: Dict[int, List[int]],
                             edge_type_idx: int,
                             layer_idx: int,
                             edge_weights: Optional[Dict[int, List[float]]] = None
                             ) -> torch.Tensor:
        """
        Aggregate messages from neighbors of a specific edge type.
        
        Args:
            node_features: Current node features
            adj_dict: Adjacency for this edge type
            edge_type_idx: Index of edge type
            layer_idx: Current layer index
            edge_weights: Optional weights for edges
        
        Returns:
            Aggregated messages tensor of shape (num_nodes, hidden_dim)
        """
        num_nodes = node_features.shape[0]
        device = node_features.device
        messages = torch.zeros(num_nodes, self.hidden_dim, device=device)
        count = torch.zeros(num_nodes, device=device)
        
        transform = self.edge_transforms[layer_idx][edge_type_idx]
        attention = self.attention[layer_idx]
        
        for src_idx, tgt_indices in adj_dict.items():
            if not tgt_indices:
                continue
            
            # Get source node features
            src_feat = node_features[src_idx]  # (hidden_dim,)
            
            # Get target node features
            tgt_feats = node_features[tgt_indices]  # (num_neighbors, hidden_dim)
            
            # Transform target features
            tgt_transformed = transform(tgt_feats)  # (num_neighbors, hidden_dim)
            
            # Attention weights
            src_expanded = src_feat.unsqueeze(0).expand(len(tgt_indices), -1)
            attn_input = torch.cat([src_expanded, tgt_feats], dim=-1)
            attn_weights = F.softmax(attention(attn_input).squeeze(-1), dim=0)
            
            # Apply edge weights if provided
            if edge_weights and src_idx in edge_weights:
                e_weights = torch.tensor(
                    edge_weights[src_idx], device=device
                )[:len(tgt_indices)]
                attn_weights = attn_weights * e_weights
                attn_weights = attn_weights / (attn_weights.sum() + 1e-8)
            
            # Weighted aggregation
            aggregated = (tgt_transformed * attn_weights.unsqueeze(-1)).sum(dim=0)
            
            # Add to messages
            messages[src_idx] += aggregated
            count[src_idx] += 1
        
        # Average
        count = count.clamp(min=1)
        messages = messages / count.unsqueeze(-1)
        
        return messages
    
    def predict_tier_embeddings(self, 
                                 graph_features: torch.Tensor
                                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict tier-specific embeddings from graph structure.
        
        Args:
            graph_features: Node features from graph
        
        Returns:
            Tuple of (intrinsic, collaborative, interaction) embeddings
        """
        return (
            self.proj_intrinsic(graph_features),
            self.proj_collaborative(graph_features),
            self.proj_interaction(graph_features)
        )
    
    def get_attention_weights(self, 
                               node_features: torch.Tensor,
                               adj_dict: Dict[int, List[int]],
                               edge_type_idx: int,
                               layer_idx: int = 0
                               ) -> Dict[int, np.ndarray]:
        """
        Extract attention weights for interpretability.
        Used for path importance distillation.
        
        Args:
            node_features: Node features
            adj_dict: Adjacency for this edge type
            edge_type_idx: Index of edge type
            layer_idx: Which layer to extract from
        
        Returns:
            Dictionary mapping node_idx -> attention weights array
        """
        attention_weights = {}
        attention = self.attention[layer_idx]
        
        for src_idx, tgt_indices in adj_dict.items():
            if not tgt_indices:
                continue
            
            src_feat = node_features[src_idx]
            tgt_feats = node_features[tgt_indices]
            src_expanded = src_feat.unsqueeze(0).expand(len(tgt_indices), -1)
            attn_input = torch.cat([src_expanded, tgt_feats], dim=-1)
            
            with torch.no_grad():
                weights = F.softmax(attention(attn_input).squeeze(-1), dim=0)
                attention_weights[src_idx] = weights.cpu().numpy()
        
        return attention_weights


class LightDecoder(nn.Module):
    """
    Lightweight decoder for regenerating text from tier embeddings.
    Used for on-demand text reconstruction in Phase 3.
    """
    
    def __init__(self,
                 embedding_dim: int = 256,
                 hidden_dim: int = 512,
                 vocab_size: int = 50000,
                 max_length: int = 100):
        """
        Initialize LightDecoder.
        
        Args:
            embedding_dim: Dimension of input tier embeddings
            hidden_dim: LSTM hidden dimension
            vocab_size: Size of output vocabulary
            max_length: Maximum generation length
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        # Tier fusion layer
        self.tier_fusion = nn.Linear(embedding_dim * 3, hidden_dim)
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Attention over tier embeddings
        self.tier_attention = nn.Linear(hidden_dim, 3)
    
    def forward(self, 
                h_intrinsic: torch.Tensor,
                h_collaborative: torch.Tensor,
                h_interaction: torch.Tensor,
                target_tokens: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Forward pass for training or inference.
        
        Args:
            h_intrinsic: Intrinsic tier embedding (batch_size, embedding_dim)
            h_collaborative: Collaborative tier embedding
            h_interaction: Interaction tier embedding
            target_tokens: Optional target tokens for teacher forcing
        
        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        batch_size = h_intrinsic.shape[0]
        device = h_intrinsic.device
        
        # Fuse tier embeddings
        tier_concat = torch.cat([h_intrinsic, h_collaborative, h_interaction], dim=-1)
        fused = F.relu(self.tier_fusion(tier_concat))  # (batch_size, hidden_dim)
        
        # Expand for sequence generation
        hidden = fused.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Initialize LSTM states
        h0 = fused.unsqueeze(0).repeat(2, 1, 1)  # (2, batch_size, hidden_dim)
        c0 = torch.zeros_like(h0)
        
        if target_tokens is not None:
            # Teacher forcing during training
            lstm_out, _ = self.lstm(target_tokens, (h0, c0))
            logits = self.output_projection(lstm_out)
        else:
            # Autoregressive generation
            logits = self._generate(hidden, h0, c0, device)
        
        return logits
    
    def _generate(self, 
                  initial_input: torch.Tensor,
                  h0: torch.Tensor,
                  c0: torch.Tensor,
                  device: torch.device
                  ) -> torch.Tensor:
        """
        Autoregressive text generation.
        
        Args:
            initial_input: Initial hidden state
            h0, c0: LSTM initial states
            device: Device to use
        
        Returns:
            Generated token logits
        """
        batch_size = initial_input.shape[0]
        generated = []
        hidden = (h0, c0)
        current_input = initial_input
        
        for _ in range(self.max_length):
            lstm_out, hidden = self.lstm(current_input, hidden)
            token_logits = self.output_projection(lstm_out)
            generated.append(token_logits)
            
            # Use argmax for next input (simplified, no embedding layer shown)
            next_token_emb = lstm_out
        
        return torch.cat(generated, dim=1)