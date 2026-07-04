"""
Heterogeneous GNN Implementation for H-GRAGrecsys

This module implements heterogeneous graph neural network architectures for
learning representations from graphs with multiple node and edge types.
It supports various aggregation strategies, attention mechanisms, and
relation-specific message passing.

The implementation is compatible with both PyTorch Geometric and DGL backends,
providing flexible integration with the H-GRAGrecsys system.
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    HeteroConv, GATConv, SAGEConv, GINConv, Linear,
    global_mean_pool, global_max_pool, global_add_pool
)
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax
import dgl
import dgl.function as fn
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from utils
from utils.config_loader import ConfigLoader
from utils.logger import Logger
from utils.seed_manager import SeedManager

# Import from graph module
from models.graph.heterogeneous_graph import HeterogeneousGraph
from models.graph.relation_types import RelationType


class HGNNLayer(nn.Module):
    """
    Single layer of Heterogeneous Graph Neural Network.
    
    This layer applies relation-specific transformations and message passing
    for each relation type in the heterogeneous graph.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        aggregation: str = 'sum',
        dropout: float = 0.1,
        use_attention: bool = True,
        num_heads: int = 4
    ):
        """
        Initialize a heterogeneous GNN layer.
        
        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension.
            num_relations: Number of relation types.
            aggregation: Aggregation method ('sum', 'mean', 'max').
            dropout: Dropout rate.
            use_attention: Whether to use attention mechanism.
            num_heads: Number of attention heads (if use_attention=True).
        
        Raises:
            ValueError: If aggregation method is invalid.
        """
        super(HGNNLayer, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.aggregation = aggregation
        self.dropout = dropout
        self.use_attention = use_attention
        self.num_heads = num_heads
        
        # Validate aggregation method
        valid_aggregations = ['sum', 'mean', 'max']
        if aggregation not in valid_aggregations:
            raise ValueError(
                f"Invalid aggregation: {aggregation}. "
                f"Must be one of: {valid_aggregations}"
            )
        
        # Relation-specific linear transformations
        self.relation_linear = nn.ModuleList([
            nn.Linear(in_dim, out_dim)
            for _ in range(num_relations)
        ])
        
        # Relation-specific attention (if enabled)
        if use_attention:
            self.attention = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=out_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
                for _ in range(num_relations)
            ])
        
        # Self-transformation for residual connection
        if in_dim != out_dim:
            self.self_transform = nn.Linear(in_dim, out_dim)
        else:
            self.self_transform = nn.Identity()
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Activation
        self.activation = nn.ReLU()
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_dim)
        
    def forward(
        self,
        adj_matrix: torch.Tensor,
        features: torch.Tensor,
        relation_type: int,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for a single relation type.
        
        Args:
            adj_matrix: Adjacency matrix for the relation (N x N).
            features: Node features (N x in_dim).
            relation_type: Index of the relation type.
            edge_weights: Optional edge weights for weighted aggregation.
        
        Returns:
            Updated node features (N x out_dim).
        
        Raises:
            ValueError: If adj_matrix shape is invalid.
        """
        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError(f"Adjacency matrix must be square, got {adj_matrix.shape}")
        
        if adj_matrix.shape[0] != features.shape[0]:
            raise ValueError(
                f"Adjacency matrix size ({adj_matrix.shape[0]}) "
                f"does not match features size ({features.shape[0]})"
            )
        
        # Apply relation-specific transformation
        transformed_features = self.relation_linear[relation_type](features)
        
        # Message passing with aggregation
        if self.aggregation == 'sum':
            if edge_weights is not None:
                # Weighted sum aggregation
                neighbor_features = torch.mm(adj_matrix * edge_weights, transformed_features)
            else:
                neighbor_features = torch.mm(adj_matrix, transformed_features)
        elif self.aggregation == 'mean':
            degrees = adj_matrix.sum(dim=1, keepdim=True)
            degrees = torch.clamp(degrees, min=1.0)
            if edge_weights is not None:
                neighbor_features = torch.mm(adj_matrix * edge_weights, transformed_features)
                neighbor_features = neighbor_features / degrees
            else:
                neighbor_features = torch.mm(adj_matrix, transformed_features)
                neighbor_features = neighbor_features / degrees
        elif self.aggregation == 'max':
            # Max aggregation
            neighbor_indices = adj_matrix.nonzero(as_tuple=True)
            if len(neighbor_indices[0]) > 0:
                neighbor_features = transformed_features[neighbor_indices[1]]
                # Max pooling over neighbors
                neighbor_features = neighbor_features.view(
                    adj_matrix.shape[0], -1, self.out_dim
                ).max(dim=1)[0]
            else:
                neighbor_features = torch.zeros(
                    adj_matrix.shape[0], self.out_dim,
                    device=features.device
                )
        
        # Apply attention if enabled
        if self.use_attention:
            # Multi-head attention
            attn_output, attn_weights = self.attention[relation_type](
                query=transformed_features.unsqueeze(1),
                key=neighbor_features.unsqueeze(1),
                value=neighbor_features.unsqueeze(1)
            )
            neighbor_features = attn_output.squeeze(1)
        
        # Residual connection
        residual = self.self_transform(features)
        output = neighbor_features + residual
        
        # Apply activation, dropout, and layer norm
        output = self.activation(output)
        output = self.dropout_layer(output)
        output = self.layer_norm(output)
        
        return output
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        for linear in self.relation_linear:
            linear.reset_parameters()
        
        if self.use_attention:
            for attn in self.attention:
                attn._reset_parameters()
        
        if hasattr(self.self_transform, 'reset_parameters'):
            self.self_transform.reset_parameters()
        
        self.layer_norm.reset_parameters()


class HeterogeneousGNN(nn.Module):
    """
    Multi-layer Heterogeneous Graph Neural Network.
    
    This model implements a multi-layer GNN for heterogeneous graphs with
    support for multiple relation types, attention mechanisms, and various
    aggregation strategies.
    """
    
    def __init__(self, config: Union[str, Dict, ConfigLoader]):
        """
        Initialize the Heterogeneous GNN model.
        
        Args:
            config: Configuration object or path to config file.
                    Can be a string path, dict, or ConfigLoader instance.
        
        Raises:
            ValueError: If config is invalid or missing required fields.
        """
        super(HeterogeneousGNN, self).__init__()
        
        # Load configuration
        if isinstance(config, str):
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
            name='heterogeneous_gnn'
        )
        
        # Extract GNN configuration
        gnn_config = self.config.get('model', {}).get('gnn', {})
        graph_config = self.config.get('model', {}).get('graph', {})
        
        # Model parameters
        self.input_dim = gnn_config.get('input_dim', 768)
        self.hidden_dim = gnn_config.get('hidden_dim', 256)
        self.output_dim = gnn_config.get('output_dim', 128)
        self.num_layers = gnn_config.get('num_layers', 3)
        self.num_heads = gnn_config.get('num_heads', 4)
        self.dropout = gnn_config.get('dropout', 0.1)
        self.aggregation = gnn_config.get('aggregation', 'sum')
        self.use_attention = gnn_config.get('use_attention', True)
        
        # Relation types from graph config
        self.relation_types = graph_config.get('relation_types', [
            'interact', 'similar_pref', 'co_inter', 'content_sim'
        ])
        self.num_relations = len(self.relation_types)
        
        # Set random seed
        if 'seed' in gnn_config:
            SeedManager.set_seed(gnn_config['seed'])
        
        # Initialize model layers
        self._build_layers()
        
        # Initialize output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(self.output_dim)
        )
        
        self.logger.log_info(
            f"HeterogeneousGNN initialized: "
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"output_dim={self.output_dim}, layers={self.num_layers}, "
            f"relations={self.num_relations}"
        )
    
    def _build_layers(self):
        """Build the GNN layers."""
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(
            HGNNLayer(
                in_dim=self.input_dim,
                out_dim=self.hidden_dim,
                num_relations=self.num_relations,
                aggregation=self.aggregation,
                dropout=self.dropout,
                use_attention=self.use_attention,
                num_heads=self.num_heads
            )
        )
        
        # Hidden layers
        for i in range(1, self.num_layers):
            self.layers.append(
                HGNNLayer(
                    in_dim=self.hidden_dim,
                    out_dim=self.hidden_dim,
                    num_relations=self.num_relations,
                    aggregation=self.aggregation,
                    dropout=self.dropout,
                    use_attention=self.use_attention,
                    num_heads=self.num_heads
                )
            )
    
    def forward(
        self,
        graph: Union[HeteroData, HeterogeneousGraph, dgl.DGLGraph],
        node_features: Optional[Dict[str, torch.Tensor]] = None,
        return_all_layers: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the heterogeneous GNN.
        
        Args:
            graph: Graph object (PyG HeteroData, H-GRAG HeterogeneousGraph, or DGL).
            node_features: Optional dict of node features by type.
                          If None, uses features from graph.
            return_all_layers: Whether to return intermediate layer outputs.
        
        Returns:
            Dict mapping node types to output embeddings.
            If return_all_layers=True, returns dict with 'final' and 'layers'.
        
        Raises:
            ValueError: If graph type is unsupported or features missing.
        """
        # Convert graph to PyG format if needed
        if isinstance(graph, HeterogeneousGraph):
            graph = graph.to_pytorch_geometric()
        elif isinstance(graph, dgl.DGLGraph):
            graph = self._dgl_to_pyg(graph, node_features)
        
        if not isinstance(graph, HeteroData):
            raise ValueError(
                f"Unsupported graph type: {type(graph)}. "
                "Must be HeteroData, HeterogeneousGraph, or DGLGraph."
            )
        
        # Extract node features
        if node_features is None:
            node_features = {}
            for node_type in graph.node_types:
                if hasattr(graph[node_type], 'x'):
                    node_features[node_type] = graph[node_type].x
                else:
                    raise ValueError(f"No features found for node type: {node_type}")
        
        # Initialize output dict
        layer_outputs = []
        current_features = node_features.copy()
        
        # Pass through each layer
        for layer_idx, layer in enumerate(self.layers):
            next_features = {}
            
            # Process each node type
            for node_type, features in current_features.items():
                # Get adjacency matrices for all relation types
                relation_features = []
                for rel_idx, rel_type in enumerate(self.relation_types):
                    # Get adjacency matrix for this relation and node type
                    adj_matrix = self._get_adjacency_matrix(
                        graph, node_type, rel_type, features.device
                    )
                    
                    # Get edge weights if available
                    edge_weights = self._get_edge_weights(
                        graph, node_type, rel_type, features.device
                    )
                    
                    # Apply layer
                    if adj_matrix is not None:
                        rel_output = layer(
                            adj_matrix,
                            features,
                            rel_idx,
                            edge_weights
                        )
                        relation_features.append(rel_output)
                
                # Aggregate relation-specific outputs
                if relation_features:
                    # Stack and aggregate
                    stacked_features = torch.stack(relation_features, dim=0)
                    
                    if self.aggregation == 'sum':
                        aggregated = stacked_features.sum(dim=0)
                    elif self.aggregation == 'mean':
                        aggregated = stacked_features.mean(dim=0)
                    elif self.aggregation == 'max':
                        aggregated = stacked_features.max(dim=0)[0]
                    else:
                        raise ValueError(f"Unsupported aggregation: {self.aggregation}")
                    
                    next_features[node_type] = aggregated
                else:
                    # No relations for this node type, keep features
                    next_features[node_type] = features
            
            # Store layer output
            layer_outputs.append(next_features.copy())
            current_features = next_features
        
        # Apply output projection
        final_outputs = {}
        for node_type, features in current_features.items():
            final_outputs[node_type] = self.output_projection(features)
        
        if return_all_layers:
            return {
                'final': final_outputs,
                'layers': layer_outputs
            }
        
        return final_outputs
    
    def _get_adjacency_matrix(
        self,
        graph: HeteroData,
        node_type: str,
        relation_type: str,
        device: torch.device
    ) -> Optional[torch.Tensor]:
        """
        Extract adjacency matrix for a specific relation and node type.
        
        Args:
            graph: PyG HeteroData object.
            node_type: Target node type.
            relation_type: Relation type string.
            device: Target device.
        
        Returns:
            Adjacency matrix or None if relation doesn't exist for node type.
        """
        # Check if relation exists
        edge_type = (node_type, relation_type, node_type)
        if edge_type not in graph.edge_types:
            # Try with different head/tail types
            for etype in graph.edge_types:
                if etype[1] == relation_type and etype[0] == node_type:
                    edge_type = etype
                    break
                elif etype[1] == relation_type and etype[2] == node_type:
                    edge_type = etype
                    break
            else:
                return None
        
        # Get edge indices
        edge_index = graph[edge_type].edge_index
        
        if edge_index.shape[1] == 0:
            # No edges
            num_nodes = graph[node_type].x.shape[0]
            return torch.zeros(num_nodes, num_nodes, device=device)
        
        # Convert to dense adjacency matrix
        num_nodes = graph[node_type].x.shape[0]
        adj_matrix = torch.zeros(num_nodes, num_nodes, device=device)
        adj_matrix[edge_index[0], edge_index[1]] = 1.0
        
        return adj_matrix
    
    def _get_edge_weights(
        self,
        graph: HeteroData,
        node_type: str,
        relation_type: str,
        device: torch.device
    ) -> Optional[torch.Tensor]:
        """
        Extract edge weights for a specific relation and node type.
        
        Args:
            graph: PyG HeteroData object.
            node_type: Target node type.
            relation_type: Relation type string.
            device: Target device.
        
        Returns:
            Edge weights tensor or None if no weights.
        """
        # Check if relation exists
        edge_type = (node_type, relation_type, node_type)
        if edge_type not in graph.edge_types:
            for etype in graph.edge_types:
                if etype[1] == relation_type and etype[0] == node_type:
                    edge_type = etype
                    break
                elif etype[1] == relation_type and etype[2] == node_type:
                    edge_type = etype
                    break
            else:
                return None
        
        # Get edge weights if available
        if hasattr(graph[edge_type], 'edge_weight'):
            return graph[edge_type].edge_weight.to(device)
        else:
            # Default to ones
            num_edges = graph[edge_type].edge_index.shape[1]
            return torch.ones(num_edges, device=device)
    
    def _dgl_to_pyg(
        self,
        dgl_graph: dgl.DGLGraph,
        node_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> HeteroData:
        """
        Convert DGL graph to PyG HeteroData format.
        
        Args:
            dgl_graph: DGL graph object.
            node_features: Optional node features by type.
        
        Returns:
            PyG HeteroData object.
        
        Raises:
            ValueError: If conversion fails.
        """
        pyg_graph = HeteroData()
        
        # Handle node types
        for node_type in dgl_graph.ntypes:
            if node_features is not None and node_type in node_features:
                pyg_graph[node_type].x = node_features[node_type]
            elif dgl_graph.nodes[node_type].data:
                # Extract features from DGL
                features = []
                for key in dgl_graph.nodes[node_type].data.keys():
                    if isinstance(dgl_graph.nodes[node_type].data[key], torch.Tensor):
                        features.append(dgl_graph.nodes[node_type].data[key])
                if features:
                    pyg_graph[node_type].x = torch.cat(features, dim=-1)
                else:
                    # Default features
                    pyg_graph[node_type].x = torch.ones(
                        dgl_graph.num_nodes(node_type), 1
                    )
        
        # Handle edge types
        for edge_type in dgl_graph.etypes:
            src_type, rel_type, dst_type = dgl_graph.to_canonical_etype(edge_type)
            
            # Get edge indices
            src, dst = dgl_graph.edges(etype=edge_type)
            edge_index = torch.stack([src, dst], dim=0)
            
            # Store in PyG format
            pyg_graph[(src_type, rel_type, dst_type)].edge_index = edge_index
            
            # Handle edge features if available
            if dgl_graph.edges[edge_type].data:
                edge_features = []
                for key in dgl_graph.edges[edge_type].data.keys():
                    if isinstance(dgl_graph.edges[edge_type].data[key], torch.Tensor):
                        edge_features.append(dgl_graph.edges[edge_type].data[key])
                if edge_features:
                    pyg_graph[(src_type, rel_type, dst_type)].edge_attr = torch.cat(
                        edge_features, dim=-1
                    )
        
        return pyg_graph
    
    def propagate_message(
        self,
        source_nodes: torch.Tensor,
        target_nodes: torch.Tensor,
        relation_type: str,
        features: Dict[str, torch.Tensor],
        graph: HeteroData
    ) -> Dict[str, torch.Tensor]:
        """
        Propagate messages from source to target nodes for a specific relation.
        
        Args:
            source_nodes: Indices of source nodes.
            target_nodes: Indices of target nodes.
            relation_type: Type of relation.
            features: Node features by type.
            graph: PyG HeteroData object.
        
        Returns:
            Dict with propagated features for target nodes.
        """
        # Get the appropriate layer
        layer_idx = self.num_layers - 1
        layer = self.layers[layer_idx]
        
        # Get relation index
        rel_idx = self.relation_types.index(relation_type)
        
        # Get adjacency matrix for the specific relation
        adj_matrix = self._get_adjacency_matrix(
            graph, 'user', relation_type, features['user'].device
        )
        
        if adj_matrix is None:
            return features
        
        # Propagate for each node type
        propagated_features = {}
        for node_type, feat in features.items():
            if node_type == 'user' or node_type == 'item':
                # Apply propagation
                propagated = layer(
                    adj_matrix[target_nodes][:, source_nodes],
                    feat[source_nodes],
                    rel_idx
                )
                propagated_features[node_type] = propagated
            else:
                propagated_features[node_type] = feat
        
        return propagated_features
    
    def aggregate_features(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        aggregation: Optional[str] = None
    ) -> torch.Tensor:
        """
        Aggregate features using specified aggregation method.
        
        Args:
            features: Node features (N x D).
            adjacency: Adjacency matrix (N x N).
            aggregation: Aggregation method. If None, uses self.aggregation.
        
        Returns:
            Aggregated features.
        """
        agg_method = aggregation or self.aggregation
        
        if agg_method == 'sum':
            return torch.mm(adjacency, features)
        elif agg_method == 'mean':
            degrees = adjacency.sum(dim=1, keepdim=True)
            degrees = torch.clamp(degrees, min=1.0)
            return torch.mm(adjacency, features) / degrees
        elif agg_method == 'max':
            neighbor_indices = adjacency.nonzero(as_tuple=True)
            if len(neighbor_indices[0]) > 0:
                neighbor_features = features[neighbor_indices[1]]
                return neighbor_features.view(
                    adjacency.shape[0], -1, features.shape[-1]
                ).max(dim=1)[0]
            else:
                return torch.zeros(
                    adjacency.shape[0], features.shape[-1],
                    device=features.device
                )
        else:
            raise ValueError(f"Unsupported aggregation: {agg_method}")
    
    def apply_activation(
        self,
        x: torch.Tensor,
        activation_type: str = 'relu'
    ) -> torch.Tensor:
        """
        Apply activation function to tensor.
        
        Args:
            x: Input tensor.
            activation_type: Type of activation function.
        
        Returns:
            Activated tensor.
        
        Raises:
            ValueError: If activation type is unsupported.
        """
        if activation_type == 'relu':
            return F.relu(x)
        elif activation_type == 'leaky_relu':
            return F.leaky_relu(x, 0.2)
        elif activation_type == 'elu':
            return F.elu(x)
        elif activation_type == 'gelu':
            return F.gelu(x)
        elif activation_type == 'tanh':
            return torch.tanh(x)
        elif activation_type == 'sigmoid':
            return torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation: {activation_type}")
    
    def get_node_embeddings(
        self,
        graph: Union[HeteroData, HeterogeneousGraph, dgl.DGLGraph],
        nodes: Optional[Dict[str, List[int]]] = None,
        node_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get embeddings for specific nodes.
        
        Args:
            graph: Graph object.
            nodes: Optional dict mapping node types to node indices.
                  If None, returns all nodes.
            node_features: Optional node features.
        
        Returns:
            Dict mapping node types to node embeddings.
        """
        # Forward pass
        embeddings = self.forward(graph, node_features)
        
        # Extract specific nodes if requested
        if nodes is not None:
            extracted = {}
            for node_type, indices in nodes.items():
                if node_type in embeddings:
                    extracted[node_type] = embeddings[node_type][indices]
                else:
                    raise ValueError(f"Node type '{node_type}' not found in graph")
            return extracted
        
        return embeddings
    
    def get_graph_embedding(
        self,
        graph: Union[HeteroData, HeterogeneousGraph, dgl.DGLGraph],
        node_features: Optional[Dict[str, torch.Tensor]] = None,
        pooling: str = 'mean'
    ) -> torch.Tensor:
        """
        Get graph-level embedding by pooling node embeddings.
        
        Args:
            graph: Graph object.
            node_features: Optional node features.
            pooling: Pooling method ('mean', 'max', 'add').
        
        Returns:
            Graph-level embedding tensor.
        
        Raises:
            ValueError: If pooling method is invalid.
        """
        # Get node embeddings
        embeddings = self.forward(graph, node_features)
        
        # Concatenate all node embeddings
        all_embeddings = []
        for node_type, emb in embeddings.items():
            all_embeddings.append(emb)
        
        if not all_embeddings:
            raise ValueError("No node embeddings found")
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Apply pooling
        if pooling == 'mean':
            graph_emb = all_embeddings.mean(dim=0)
        elif pooling == 'max':
            graph_emb = all_embeddings.max(dim=0)[0]
        elif pooling == 'add':
            graph_emb = all_embeddings.sum(dim=0)
        else:
            raise ValueError(f"Invalid pooling method: {pooling}")
        
        return graph_emb
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        for layer in self.layers:
            layer.reset_parameters()
        
        for module in self.output_projection:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
    
    def get_parameters(self) -> Dict[str, int]:
        """
        Get parameter statistics.
        
        Returns:
            Dict with parameter counts.
        """
        total_params = sum(p.numel() for p in self.parameters())
        
        layer_params = []
        for idx, layer in enumerate(self.layers):
            layer_params.append(sum(p.numel() for p in layer.parameters()))
        
        return {
            'total': total_params,
            'layers': layer_params,
            'output_projection': sum(p.numel() for p in self.output_projection.parameters())
        }
    
    def to_device(self, device: torch.device) -> 'HeterogeneousGNN':
        """
        Move model to specified device.
        
        Args:
            device: PyTorch device.
        
        Returns:
            Self with model moved to device.
        """
        self.to(device)
        return self


# Module level variables and exports
__all__ = [
    'HGNNLayer',
    'HeterogeneousGNN',
    '__doc__'
]

# Version information
__version__ = '1.0.0'
__author__ = 'H-GRAGrecsys Team'


def create_heterogeneous_gnn(
    config_path: str,
    device: Optional[torch.device] = None
) -> HeterogeneousGNN:
    """
    Factory function to create a HeterogeneousGNN instance.
    
    Args:
        config_path: Path to configuration file.
        device: Optional device to move model to. Defaults to CUDA if available.
    
    Returns:
        Initialized HeterogeneousGNN instance.
    
    Example:
        >>> model = create_heterogeneous_gnn('config/default_config.yaml')
        >>> model.to_device(torch.device('cuda'))
    """
    model = HeterogeneousGNN(config_path)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return model.to_device(device)