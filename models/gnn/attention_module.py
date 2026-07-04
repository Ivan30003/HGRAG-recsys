"""
Attention Module for H-GRAGrecsys

This module implements various attention mechanisms for heterogeneous graph
neural networks. It provides:
- Multi-head self-attention for node representations
- Graph attention for neighborhood aggregation
- Path attention for metapath-based reasoning
- Relation-aware attention for heterogeneous graphs
- Adaptive attention with learnable temperature scaling

The attention mechanisms support both node-level and graph-level attention
with configurable parameters and multiple attention heads.
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Union, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax, degree
from torch_geometric.nn import GATConv, GatedGraphConv
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from utils
from utils.config_loader import ConfigLoader
from utils.logger import Logger
from utils.seed_manager import SeedManager


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism with optional scaling and dropout.
    
    This class implements scaled dot-product attention with multiple heads,
    supporting both self-attention and cross-attention modes.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = True
    ):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Input embedding dimension.
            num_heads: Number of attention heads.
            dropout: Dropout rate for attention weights.
            bias: Whether to use bias in projections.
            add_bias_kv: Whether to add bias to key and value projections.
            add_zero_attn: Whether to add zero attention for padding.
            kdim: Key dimension (if different from embed_dim).
            vdim: Value dimension (if different from embed_dim).
            batch_first: Whether batch dimension is first.
        
        Raises:
            ValueError: If embed_dim is not divisible by num_heads.
        """
        super(MultiHeadAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        
        # Validate dimensions
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        
        self.head_dim = embed_dim // num_heads
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        
        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Optional bias for key/value
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.register_parameter('bias_k', None)
            self.register_parameter('bias_v', None)
        
        self.add_zero_attn = add_zero_attn
        
        self.logger = Logger(
            log_dir='./logs',
            name=f'attention_head_{num_heads}'
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for multi-head attention.
        
        Args:
            query: Query tensor (batch_size, seq_len, embed_dim) if batch_first=True.
            key: Key tensor (batch_size, seq_len, embed_dim) if batch_first=True.
            value: Value tensor (batch_size, seq_len, embed_dim) if batch_first=True.
            attn_mask: Attention mask for specific positions.
            key_padding_mask: Mask for padding positions.
            return_attn_weights: Whether to return attention weights.
        
        Returns:
            If return_attn_weights=False: Output tensor.
            If return_attn_weights=True: Tuple of (output, attention_weights).
        
        Raises:
            ValueError: If input dimensions are invalid.
        """
        # Check dimensions
        if query.size(-1) != self.embed_dim:
            raise ValueError(
                f"Query dimension {query.size(-1)} does not match embed_dim {self.embed_dim}"
            )
        
        # Project queries, keys, and values
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Add bias for key and value if provided
        if self.bias_k is not None:
            k = torch.cat([k, self.bias_k.repeat(k.size(0), 1, 1)], dim=1)
        if self.bias_v is not None:
            v = torch.cat([v, self.bias_v.repeat(v.size(0), 1, 1)], dim=1)
        
        # Reshape for multi-head attention
        batch_size = q.size(0)
        seq_len_q = q.size(1)
        seq_len_k = k.size(1)
        
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        # Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), -1e9
            )
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply dropout
        attn_weights = self.dropout_layer(attn_weights)
        
        # Compute weighted sum of values
        # Shape: (batch_size, num_heads, seq_len_q, head_dim)
        output = torch.matmul(attn_weights, v)
        
        # Reshape output
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len_q, self.embed_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        if return_attn_weights:
            return output, attn_weights
        
        return output
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        self.out_proj.reset_parameters()


class GraphAttention(nn.Module):
    """
    Graph attention mechanism for heterogeneous graphs.
    
    This class implements graph attention with support for multiple relation
    types and edge weights.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        num_heads: int = 4,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        concat_heads: bool = True,
        edge_dim: Optional[int] = None
    ):
        """
        Initialize graph attention layer.
        
        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension. If None, uses in_dim.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            negative_slope: Negative slope for LeakyReLU.
            concat_heads: Whether to concatenate or average heads.
            edge_dim: Edge feature dimension (if available).
        """
        super(GraphAttention, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.concat_heads = concat_heads
        self.edge_dim = edge_dim
        
        # Head output dimension
        self.head_out_dim = self.out_dim // num_heads if concat_heads else self.out_dim
        
        # Linear transformations for queries and keys
        self.q_proj = nn.Linear(in_dim, self.num_heads * self.head_out_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, self.num_heads * self.head_out_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, self.num_heads * self.head_out_dim, bias=False)
        
        # Attention parameters
        self.attn_param = nn.Parameter(
            torch.zeros(1, num_heads, 2 * self.head_out_dim)
        )
        
        # Edge feature projection if provided
        if edge_dim is not None:
            self.edge_proj = nn.Linear(edge_dim, num_heads, bias=False)
        else:
            self.edge_proj = None
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.attn_param)
        
        self.logger = Logger(
            log_dir='./logs',
            name='graph_attention'
        )
    
    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for graph attention.
        
        Args:
            features: Node features (N x in_dim).
            edge_index: Edge indices (2 x E).
            edge_weights: Optional edge weights (E,).
            edge_features: Optional edge features (E x edge_dim).
            return_attention: Whether to return attention coefficients.
        
        Returns:
            If return_attention=False: Updated node features.
            If return_attention=True: Tuple of (updated_features, attention_coeffs).
        
        Raises:
            ValueError: If dimensions are invalid.
        """
        N, D = features.shape
        E = edge_index.shape[1]
        
        # Project features
        q = self.q_proj(features).view(N, self.num_heads, self.head_out_dim)
        k = self.k_proj(features).view(N, self.num_heads, self.head_out_dim)
        v = self.v_proj(features).view(N, self.num_heads, self.head_out_dim)
        
        # Compute attention scores
        # For each edge, compute attention coefficient
        src, dst = edge_index
        
        # Get source and target features
        q_src = q[src]  # (E, num_heads, head_out_dim)
        k_dst = k[dst]  # (E, num_heads, head_out_dim)
        
        # Concatenate source and target features
        attn_input = torch.cat([q_src, k_dst], dim=-1)  # (E, num_heads, 2*head_out_dim)
        
        # Compute attention scores
        attn_scores = torch.einsum('ehd,hd->eh', attn_input, self.attn_param.squeeze(0))
        
        # Apply LeakyReLU
        attn_scores = F.leaky_relu(attn_scores, self.negative_slope)
        
        # Add edge weights if provided
        if edge_weights is not None:
            attn_scores = attn_scores * edge_weights.unsqueeze(-1)
        
        # Add edge features if provided
        if edge_features is not None and self.edge_proj is not None:
            edge_attn = self.edge_proj(edge_features)  # (E, num_heads)
            attn_scores = attn_scores + edge_attn
        
        # Normalize attention coefficients
        attn_coeffs = softmax(attn_scores, src, num_nodes=N)
        attn_coeffs = self.dropout_layer(attn_coeffs)
        
        # Apply attention to values
        # For each edge, we need to propagate values
        v_src = v[src]  # (E, num_heads, head_out_dim)
        
        # Weighted sum of values
        # We need to aggregate messages per node
        output = torch.zeros(N, self.num_heads, self.head_out_dim, device=features.device)
        
        # Aggregate messages
        for i in range(E):
            s, t = src[i], dst[i]
            output[t] += attn_coeffs[i].unsqueeze(-1) * v_src[i]
        
        # Reshape output
        if self.concat_heads:
            output = output.view(N, self.num_heads * self.head_out_dim)
        else:
            output = output.mean(dim=1)
        
        if return_attention:
            return output, attn_coeffs
        
        return output
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        nn.init.xavier_uniform_(self.attn_param)
        
        if self.edge_proj is not None:
            self.edge_proj.reset_parameters()


class PathAttention(nn.Module):
    """
    Attention mechanism for metapath-based reasoning.
    
    This class computes attention scores over metapath instances to aggregate
    path-level information.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_paths: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize path attention.
        
        Args:
            embed_dim: Embedding dimension.
            num_paths: Number of path types/instances.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super(PathAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_paths = num_paths
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.head_dim = embed_dim // num_heads
        
        # Attention parameters
        self.attn_weights = nn.Parameter(
            torch.zeros(num_paths, embed_dim)
        )
        
        # Projection for path embeddings
        self.path_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize
        nn.init.xavier_uniform_(self.attn_weights)
        
        self.logger = Logger(
            log_dir='./logs',
            name='path_attention'
        )
    
    def forward(
        self,
        path_embeddings: torch.Tensor,
        return_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute attention-weighted path aggregation.
        
        Args:
            path_embeddings: Path embeddings (batch_size, num_paths, embed_dim).
            return_weights: Whether to return attention weights.
        
        Returns:
            If return_weights=False: Aggregated path representation.
            If return_weights=True: Tuple of (aggregated, attention_weights).
        
        Raises:
            ValueError: If input dimensions are invalid.
        """
        batch_size, num_paths, embed_dim = path_embeddings.shape
        
        if num_paths != self.num_paths:
            raise ValueError(
                f"Number of paths {num_paths} does not match {self.num_paths}"
            )
        
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Embedding dimension {embed_dim} does not match {self.embed_dim}"
            )
        
        # Project path embeddings
        projected = self.path_proj(path_embeddings)  # (batch, num_paths, embed_dim)
        
        # Compute attention scores
        # Shape: (batch_size, num_paths)
        attn_scores = torch.matmul(
            projected,
            self.attn_weights.unsqueeze(-1)
        ).squeeze(-1)
        
        # Scale
        attn_scores = attn_scores / math.sqrt(self.embed_dim)
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Weighted aggregation
        aggregated = torch.matmul(attn_weights.unsqueeze(1), path_embeddings).squeeze(1)
        
        if return_weights:
            return aggregated, attn_weights
        
        return aggregated
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        nn.init.xavier_uniform_(self.attn_weights)
        self.path_proj.reset_parameters()


class RelationAwareAttention(nn.Module):
    """
    Relation-aware attention for heterogeneous graphs.
    
    This class implements attention that incorporates relation-specific
    transformations and biases.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_relations: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize relation-aware attention.
        
        Args:
            embed_dim: Embedding dimension.
            num_relations: Number of relation types.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super(RelationAwareAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_relations = num_relations
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.head_dim = embed_dim // num_heads
        
        # Relation-specific transformations
        self.relation_q_proj = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim, bias=False)
            for _ in range(num_relations)
        ])
        
        self.relation_k_proj = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim, bias=False)
            for _ in range(num_relations)
        ])
        
        self.relation_v_proj = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim, bias=False)
            for _ in range(num_relations)
        ])
        
        # Relation biases
        self.relation_bias = nn.Parameter(
            torch.zeros(num_relations, embed_dim)
        )
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        self.logger = Logger(
            log_dir='./logs',
            name='relation_aware_attention'
        )
    
    def forward(
        self,
        features: torch.Tensor,
        relation_indices: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for relation-aware attention.
        
        Args:
            features: Input features (batch_size, seq_len, embed_dim).
            relation_indices: Relation indices for each position (batch_size, seq_len).
            return_attention: Whether to return attention weights.
        
        Returns:
            If return_attention=False: Updated features.
            If return_attention=True: Tuple of (updated_features, attention_weights).
        
        Raises:
            ValueError: If input dimensions are invalid.
        """
        batch_size, seq_len, embed_dim = features.shape
        
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Feature dimension {embed_dim} does not match {self.embed_dim}"
            )
        
        # Process each relation type
        all_attn_outputs = []
        all_attn_weights = []
        
        for r in range(self.num_relations):
            # Get positions for this relation
            rel_mask = (relation_indices == r).unsqueeze(-1)  # (batch, seq_len, 1)
            
            if not rel_mask.any():
                continue
            
            # Apply relation-specific projections
            q = self.relation_q_proj[r](features)  # (batch, seq_len, embed_dim)
            k = self.relation_k_proj[r](features)
            v = self.relation_v_proj[r](features)
            
            # Add relation bias
            q = q + self.relation_bias[r].unsqueeze(0).unsqueeze(0)
            
            # Reshape for multi-head attention
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Compute attention
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Apply mask for this relation
            rel_mask_expanded = rel_mask.transpose(-2, -1).unsqueeze(1)  # (batch, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(~rel_mask_expanded, -1e9)
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout_layer(attn_weights)
            
            # Compute output
            output = torch.matmul(attn_weights, v)
            output = output.transpose(1, 2).contiguous()
            output = output.view(batch_size, seq_len, embed_dim)
            
            all_attn_outputs.append(output * rel_mask.float())
            all_attn_weights.append(attn_weights)
        
        # Aggregate outputs from all relations
        if all_attn_outputs:
            output = torch.stack(all_attn_outputs, dim=0).sum(dim=0)
            output = self.out_proj(output)
        else:
            output = features
        
        if return_attention:
            return output, torch.cat(all_attn_weights, dim=1)
        
        return output
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        for proj in self.relation_q_proj:
            proj.reset_parameters()
        for proj in self.relation_k_proj:
            proj.reset_parameters()
        for proj in self.relation_v_proj:
            proj.reset_parameters()
        
        nn.init.zeros_(self.relation_bias)
        self.out_proj.reset_parameters()


class AdaptiveAttention(nn.Module):
    """
    Adaptive attention with learnable temperature and gating.
    
    This class implements attention with adaptive scaling and gating
    mechanisms for dynamic attention adjustment.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        learnable_temp: bool = True
    ):
        """
        Initialize adaptive attention.
        
        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            learnable_temp: Whether to use learnable temperature.
        """
        super(AdaptiveAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.head_dim = embed_dim // num_heads
        
        # Base multi-head attention
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Temperature scaling
        if learnable_temp:
            self.temperature = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('temperature', torch.ones(1))
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.logger = Logger(
            log_dir='./logs',
            name='adaptive_attention'
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for adaptive attention.
        
        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            attn_mask: Optional attention mask.
            return_attention: Whether to return attention weights.
        
        Returns:
            If return_attention=False: Output tensor.
            If return_attention=True: Tuple of (output, attention_weights).
        """
        # Apply attention with temperature scaling
        output, attn_weights = self.attention.forward(
            query, key, value, attn_mask, return_attn_weights=True
        )
        
        # Apply temperature scaling
        attn_weights = attn_weights / torch.clamp(self.temperature, min=0.1)
        
        # Compute gate
        gate_value = self.gate(query.mean(dim=1, keepdim=True))
        
        # Apply gating
        output = output * gate_value.unsqueeze(1)
        
        if return_attention:
            return output, attn_weights
        
        return output
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        self.attention.reset_parameters()
        
        for layer in self.gate:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        if hasattr(self, 'temperature') and self.temperature.requires_grad:
            nn.init.ones_(self.temperature)


class AttentionModule(nn.Module):
    """
    Main attention module orchestrating all attention mechanisms.
    
    This class combines multiple attention mechanisms and provides
    a unified interface for the H-GRAGrecsys system.
    """
    
    def __init__(self, config: Optional[Union[str, Dict, ConfigLoader]] = None):
        """
        Initialize the attention module.
        
        Args:
            config: Configuration object or path to config file.
                   Can be a string path, dict, or ConfigLoader instance.
        
        Raises:
            ValueError: If config is invalid or missing required fields.
        """
        super(AttentionModule, self).__init__()
        
        # Load configuration
        if config is None:
            self.config = {
                'model': {
                    'gnn': {
                        'num_heads': 4,
                        'dropout': 0.1,
                        'embed_dim': 256,
                        'attention_type': 'multi_head'
                    }
                }
            }
        elif isinstance(config, str):
            self.config_loader = ConfigLoader(config)
            self.config = self.config_loader.load_config()
        elif isinstance(config, dict):
            self.config = config
            self.config_loader = None
        elif isinstance(config, ConfigLoader):
            self.config_loader = config
            self.config = config.load_config()
        else:
            raise ValueError(f"Invalid config type: {type(config)}")
        
        # Setup logger
        self.logger = Logger(
            log_dir=self.config.get('logging', {}).get('log_dir', './logs'),
            name='attention_module'
        )
        
        # Extract configuration
        gnn_config = self.config.get('model', {}).get('gnn', {})
        
        self.embed_dim = gnn_config.get('embed_dim', 256)
        self.num_heads = gnn_config.get('num_heads', 4)
        self.dropout = gnn_config.get('dropout', 0.1)
        self.attention_type = gnn_config.get('attention_type', 'multi_head')
        
        # Number of relations from graph config
        graph_config = self.config.get('model', {}).get('graph', {})
        self.num_relations = len(graph_config.get('relation_types', 
            ['interact', 'similar_pref', 'co_inter', 'content_sim']
        ))
        
        # Initialize attention components
        self._initialize_components()
        
        self.logger.log_info(
            f"AttentionModule initialized: embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, type={self.attention_type}"
        )
    
    def _initialize_components(self):
        """Initialize all attention components based on configuration."""
        # Multi-head attention
        self.multi_head_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        # Graph attention
        self.graph_attn = GraphAttention(
            in_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        # Path attention
        self.path_attn = PathAttention(
            embed_dim=self.embed_dim,
            num_paths=self.num_relations,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        # Relation-aware attention
        self.relation_attn = RelationAwareAttention(
            embed_dim=self.embed_dim,
            num_relations=self.num_relations,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        # Adaptive attention
        self.adaptive_attn = AdaptiveAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            learnable_temp=True
        )
    
    def compute_attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_type: str = 'multi_head',
        attn_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute attention using specified mechanism.
        
        Args:
            queries: Query tensor.
            keys: Key tensor.
            values: Value tensor.
            attention_type: Type of attention to use.
            attn_mask: Optional attention mask.
            return_weights: Whether to return attention weights.
        
        Returns:
            Attention output and optionally weights.
        
        Raises:
            ValueError: If attention_type is invalid.
        """
        valid_types = ['multi_head', 'graph', 'path', 'relation', 'adaptive']
        if attention_type not in valid_types:
            raise ValueError(
                f"Invalid attention_type: {attention_type}. "
                f"Must be one of: {valid_types}"
            )
        
        if attention_type == 'multi_head':
            return self.multi_head_attn.forward(
                queries, keys, values, attn_mask, return_weights
            )
        elif attention_type == 'graph':
            # For graph attention, we need edge_index
            # This is a simplified version
            if len(queries.shape) == 3:
                # Reshape for graph attention
                N = queries.shape[0] * queries.shape[1]
                queries_flat = queries.reshape(N, -1)
            else:
                queries_flat = queries
            
            # Create simple complete graph for demonstration
            # In practice, this should use actual graph structure
            edge_index = torch.stack([
                torch.arange(N, device=queries.device).repeat_interleave(N),
                torch.arange(N, device=queries.device).repeat(N)
            ])
            
            return self.graph_attn.forward(
                queries_flat, edge_index, return_attention=return_weights
            )
        elif attention_type == 'path':
            # Assume queries are path embeddings
            return self.path_attn.forward(queries, return_weights)
        elif attention_type == 'relation':
            # For relation-aware attention
            relation_indices = torch.randint(
                0, self.num_relations, 
                queries.shape[:-1], 
                device=queries.device
            )
            return self.relation_attn.forward(
                queries, relation_indices, return_weights
            )
        elif attention_type == 'adaptive':
            return self.adaptive_attn.forward(
                queries, keys, values, attn_mask, return_weights
            )
    
    def multi_head_attention(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convenience method for multi-head attention.
        
        Args:
            query: Query tensor.
            keys: Key tensor.
            values: Value tensor.
            attn_mask: Optional attention mask.
        
        Returns:
            Attention output.
        """
        return self.multi_head_attn.forward(query, keys, values, attn_mask)
    
    def graph_attention(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convenience method for graph attention.
        
        Args:
            features: Node features.
            edge_index: Edge indices.
            edge_weights: Optional edge weights.
        
        Returns:
            Attended node features.
        """
        return self.graph_attn.forward(features, edge_index, edge_weights)
    
    def path_attention(
        self,
        path_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Convenience method for path attention.
        
        Args:
            path_embeddings: Path embeddings.
        
        Returns:
            Aggregated path representation.
        """
        return self.path_attn.forward(path_embeddings)
    
    def relation_attention(
        self,
        features: torch.Tensor,
        relation_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Convenience method for relation-aware attention.
        
        Args:
            features: Input features.
            relation_indices: Relation indices.
        
        Returns:
            Attended features.
        """
        return self.relation_attn.forward(features, relation_indices)
    
    def get_attention_weights(
        self,
        graph: Any,
        node: int,
        neighbors: List[int]
    ) -> torch.Tensor:
        """
        Get attention weights for a node and its neighbors.
        
        Args:
            graph: Graph object.
            node: Node index.
            neighbors: List of neighbor indices.
        
        Returns:
            Attention weights for neighbors.
        """
        # This is a placeholder implementation
        # In practice, this should compute actual attention weights
        num_neighbors = len(neighbors)
        weights = torch.ones(num_neighbors) / num_neighbors
        
        return weights
    
    def reset_parameters(self):
        """Reset all attention parameters."""
        self.multi_head_attn.reset_parameters()
        self.graph_attn.reset_parameters()
        self.path_attn.reset_parameters()
        self.relation_attn.reset_parameters()
        self.adaptive_attn.reset_parameters()
    
    def get_parameters(self) -> Dict[str, int]:
        """
        Get parameter statistics.
        
        Returns:
            Dict with parameter counts for each attention component.
        """
        return {
            'multi_head_attn': sum(p.numel() for p in self.multi_head_attn.parameters()),
            'graph_attn': sum(p.numel() for p in self.graph_attn.parameters()),
            'path_attn': sum(p.numel() for p in self.path_attn.parameters()),
            'relation_attn': sum(p.numel() for p in self.relation_attn.parameters()),
            'adaptive_attn': sum(p.numel() for p in self.adaptive_attn.parameters()),
            'total': sum(p.numel() for p in self.parameters())
        }
    
    def to_device(self, device: torch.device) -> 'AttentionModule':
        """
        Move all components to specified device.
        
        Args:
            device: PyTorch device.
        
        Returns:
            Self with components moved to device.
        """
        self.to(device)
        return self


# Module level variables and exports
__all__ = [
    'MultiHeadAttention',
    'GraphAttention',
    'PathAttention',
    'RelationAwareAttention',
    'AdaptiveAttention',
    'AttentionModule',
    '__doc__'
]

# Version information
__version__ = '1.0.0'
__author__ = 'H-GRAGrecsys Team'


def create_attention_module(
    config_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> AttentionModule:
    """
    Factory function to create an AttentionModule instance.
    
    Args:
        config_path: Optional path to configuration file.
        device: Optional device to move model to. Defaults to CUDA if available.
    
    Returns:
        Initialized AttentionModule instance.
    
    Example:
        >>> attention = create_attention_module('config/default_config.yaml')
        >>> attention.to_device(torch.device('cuda'))
    """
    attention = AttentionModule(config_path)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return attention.to_device(device)