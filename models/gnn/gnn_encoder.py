"""
GNN Encoder Module for H-GRAGrecsys

This module implements the GNN encoder that combines heterogeneous GNN models
with projection heads to produce disentangled representations for nodes in
the heterogeneous graph. The encoder supports:
- Graph-level and node-level encoding
- Component-specific representation extraction
- Neighborhood aggregation with attention
- Integration with projection heads for disentanglement
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_dense_adj, subgraph
import dgl
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from sibling modules
from models.gnn.heterogeneous_gnn import HeterogeneousGNN, HGNNLayer
from models.gnn.projection_heads import ComponentProjectionHeads, ProjectionHead

# Import from graph module
from models.graph.heterogeneous_graph import HeterogeneousGraph
from models.graph.relation_types import RelationType

# Import from utils
from utils.config_loader import ConfigLoader
from utils.logger import Logger
from utils.seed_manager import SeedManager


class GNNEncoder(nn.Module):
    """
    GNN Encoder for heterogeneous graph representation learning.
    
    This class combines a heterogeneous GNN with projection heads to encode
    nodes into disentangled component representations. It supports various
    encoding strategies and aggregation methods.
    """
    
    def __init__(
        self,
        gnn_model: Optional[HeterogeneousGNN] = None,
        projection_heads: Optional[ComponentProjectionHeads] = None,
        config: Optional[Union[str, Dict, ConfigLoader]] = None
    ):
        """
        Initialize the GNN encoder.
        
        Args:
            gnn_model: Heterogeneous GNN model. If None, creates from config.
            projection_heads: Component projection heads. If None, creates from config.
            config: Configuration object or path to config file.
                   Can be a string path, dict, or ConfigLoader instance.
        
        Raises:
            ValueError: If config is invalid or missing required fields.
        """
        super(GNNEncoder, self).__init__()
        
        # Load configuration
        if config is None:
            # Default configuration
            self.config = {
                'model': {
                    'gnn': {
                        'input_dim': 768,
                        'hidden_dim': 256,
                        'output_dim': 128,
                        'num_layers': 3,
                        'num_heads': 4,
                        'dropout': 0.1,
                        'projection_dim': 128,
                        'aggregation': 'sum'
                    },
                    'graph': {
                        'relation_types': ['interact', 'similar_pref', 'co_inter', 'content_sim']
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
            name='gnn_encoder'
        )
        
        # Extract configuration
        gnn_config = self.config.get('model', {}).get('gnn', {})
        graph_config = self.config.get('model', {}).get('graph', {})
        
        self.input_dim = gnn_config.get('input_dim', 768)
        self.hidden_dim = gnn_config.get('hidden_dim', 256)
        self.output_dim = gnn_config.get('output_dim', 128)
        self.projection_dim = gnn_config.get('projection_dim', 128)
        self.num_layers = gnn_config.get('num_layers', 3)
        self.num_heads = gnn_config.get('num_heads', 4)
        self.dropout = gnn_config.get('dropout', 0.1)
        self.aggregation = gnn_config.get('aggregation', 'sum')
        
        # Relation types
        self.relation_types = graph_config.get('relation_types', [
            'interact', 'similar_pref', 'co_inter', 'content_sim'
        ])
        
        # Initialize components
        if gnn_model is not None:
            self.gnn_model = gnn_model
        else:
            self.gnn_model = self._create_gnn_model()
        
        if projection_heads is not None:
            self.projection_heads = projection_heads
        else:
            self.projection_heads = self._create_projection_heads()
        
        # Cache for node embeddings
        self.embedding_cache = {}
        self.cache_enabled = True
        
        self.logger.log_info(
            f"GNNEncoder initialized: input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, output_dim={self.output_dim}, "
            f"projection_dim={self.projection_dim}, layers={self.num_layers}"
        )
    
    def _create_gnn_model(self) -> HeterogeneousGNN:
        """
        Create a heterogeneous GNN model from configuration.
        
        Returns:
            Initialized HeterogeneousGNN model.
        """
        gnn_config = {
            'model': {
                'gnn': {
                    'input_dim': self.input_dim,
                    'hidden_dim': self.hidden_dim,
                    'output_dim': self.output_dim,
                    'num_layers': self.num_layers,
                    'num_heads': self.num_heads,
                    'dropout': self.dropout,
                    'aggregation': self.aggregation
                },
                'graph': {
                    'relation_types': self.relation_types
                }
            }
        }
        
        return HeterogeneousGNN(gnn_config)
    
    def _create_projection_heads(self) -> ComponentProjectionHeads:
        """
        Create component projection heads from configuration.
        
        Returns:
            Initialized ComponentProjectionHeads.
        """
        return ComponentProjectionHeads(
            input_dim=self.output_dim,
            config=self.config
        )
    
    def forward(
        self,
        graph: Union[HeteroData, HeterogeneousGraph, dgl.DGLGraph],
        node_features: Optional[Dict[str, torch.Tensor]] = None,
        return_components: bool = False
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """
        Encode graph nodes into embeddings.
        
        Args:
            graph: Graph object (PyG HeteroData, H-GRAG HeterogeneousGraph, or DGL).
            node_features: Optional dict of node features by type.
                          If None, uses features from graph.
            return_components: Whether to return disentangled component representations.
        
        Returns:
            If return_components=False: Dict mapping node types to node embeddings.
            If return_components=True: Tuple of (node_embeddings, component_embeddings).
        
        Raises:
            ValueError: If graph type is unsupported.
        """
        # Convert graph if needed
        if isinstance(graph, HeterogeneousGraph):
            graph = graph.to_pytorch_geometric()
        elif isinstance(graph, dgl.DGLGraph):
            graph = self._dgl_to_pyg(graph, node_features)
        
        if not isinstance(graph, HeteroData):
            raise ValueError(
                f"Unsupported graph type: {type(graph)}. "
                "Must be HeteroData, HeterogeneousGraph, or DGLGraph."
            )
        
        # Extract node features if not provided
        if node_features is None:
            node_features = {}
            for node_type in graph.node_types:
                if hasattr(graph[node_type], 'x'):
                    node_features[node_type] = graph[node_type].x
                else:
                    raise ValueError(f"No features found for node type: {node_type}")
        
        # Check cache
        cache_key = self._get_cache_key(graph, node_features)
        if self.cache_enabled and cache_key in self.embedding_cache:
            self.logger.log_info("Using cached embeddings")
            cached_data = self.embedding_cache[cache_key]
            if return_components:
                return cached_data['embeddings'], cached_data['components']
            return cached_data['embeddings']
        
        # Encode graph through GNN
        gnn_outputs = self.gnn_model.forward(graph, node_features)
        
        # Apply projection heads to get component representations
        component_outputs = {}
        for node_type, embeddings in gnn_outputs.items():
            # Project to component spaces
            component_outputs[node_type] = self.projection_heads.project_all(embeddings)
        
        # Cache if enabled
        if self.cache_enabled:
            self.embedding_cache[cache_key] = {
                'embeddings': gnn_outputs,
                'components': component_outputs
            }
        
        if return_components:
            return gnn_outputs, component_outputs
        
        return gnn_outputs
    
    def encode_graph(
        self,
        graph: Union[HeteroData, HeterogeneousGraph, dgl.DGLGraph],
        node_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode the entire graph into node embeddings.
        
        Args:
            graph: Graph object.
            node_features: Optional node features.
        
        Returns:
            Dict mapping node types to node embeddings.
        """
        return self.forward(graph, node_features, return_components=False)
    
    def get_node_representations(
        self,
        graph: Union[HeteroData, HeterogeneousGraph, dgl.DGLGraph],
        node_ids: Dict[str, List[int]],
        node_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get representations for specific nodes.
        
        Args:
            graph: Graph object.
            node_ids: Dict mapping node types to list of node indices.
            node_features: Optional node features.
        
        Returns:
            Dict mapping node types to node representations.
        
        Raises:
            ValueError: If requested node types not found in graph.
        """
        if not node_ids:
            raise ValueError("node_ids must not be empty")
        
        # Encode graph
        all_encodings = self.encode_graph(graph, node_features)
        
        # Extract specific node representations
        result = {}
        for node_type, ids in node_ids.items():
            if node_type in all_encodings:
                result[node_type] = all_encodings[node_type][ids]
            else:
                raise ValueError(f"Node type '{node_type}' not found in graph")
        
        return result
    
    def get_component_representation(
        self,
        graph: Union[HeteroData, HeterogeneousGraph, dgl.DGLGraph],
        node_id: int,
        node_type: str,
        component_type: str,
        node_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Get specific component representation for a node.
        
        Args:
            graph: Graph object.
            node_id: Index of the node.
            node_type: Type of the node.
            component_type: Type of component ('intrinsic', 'collaborative', 'interaction').
            node_features: Optional node features.
        
        Returns:
            Tensor containing the component representation.
        
        Raises:
            ValueError: If component_type is invalid or node not found.
        """
        valid_components = ['intrinsic', 'collaborative', 'interaction']
        if component_type not in valid_components:
            raise ValueError(
                f"Invalid component_type: {component_type}. "
                f"Must be one of: {valid_components}"
            )
        
        # Encode graph with components
        embeddings, components = self.forward(graph, node_features, return_components=True)
        
        if node_type not in components:
            raise ValueError(f"Node type '{node_type}' not found")
        
        return components[node_type][component_type][node_id]
    
    def get_all_component_representations(
        self,
        graph: Union[HeteroData, HeterogeneousGraph, dgl.DGLGraph],
        node_id: int,
        node_type: str,
        node_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get all component representations for a node.
        
        Args:
            graph: Graph object.
            node_id: Index of the node.
            node_type: Type of the node.
            node_features: Optional node features.
        
        Returns:
            Dict mapping component types to tensors.
        """
        _, components = self.forward(graph, node_features, return_components=True)
        
        if node_type not in components:
            raise ValueError(f"Node type '{node_type}' not found")
        
        result = {}
        for comp_type in ['intrinsic', 'collaborative', 'interaction']:
            result[comp_type] = components[node_type][comp_type][node_id]
        
        return result
    
    def aggregate_neighborhood(
        self,
        graph: Union[HeteroData, HeterogeneousGraph, dgl.DGLGraph],
        node_id: int,
        node_type: str,
        relation_type: Optional[str] = None,
        hop_count: int = 1,
        node_features: Optional[Dict[str, torch.Tensor]] = None,
        aggregation: str = 'mean'
    ) -> torch.Tensor:
        """
        Aggregate neighborhood information for a specific node.
        
        Args:
            graph: Graph object.
            node_id: Index of the target node.
            node_type: Type of the target node.
            relation_type: Optional specific relation type to aggregate over.
                          If None, aggregates over all relations.
            hop_count: Number of hops to consider for aggregation.
            node_features: Optional node features.
            aggregation: Aggregation method ('mean', 'sum', 'max').
        
        Returns:
            Tensor containing aggregated neighborhood representation.
        
        Raises:
            ValueError: If hop_count < 1 or node not found.
        """
        if hop_count < 1:
            raise ValueError("hop_count must be at least 1")
        
        # Encode graph
        embeddings = self.encode_graph(graph, node_features)
        
        if node_type not in embeddings:
            raise ValueError(f"Node type '{node_type}' not found")
        
        # Get node representation
        node_rep = embeddings[node_type][node_id]
        
        # Get neighborhood based on relation type
        if relation_type is not None:
            # Get neighbors for specific relation
            neighbors = self._get_neighbors(graph, node_id, node_type, relation_type)
        else:
            # Get neighbors for all relations
            all_neighbors = []
            for rel_type in self.relation_types:
                neighbors = self._get_neighbors(graph, node_id, node_type, rel_type)
                if neighbors:
                    all_neighbors.extend(neighbors)
            neighbors = list(set(all_neighbors))  # Remove duplicates
        
        if not neighbors:
            # No neighbors, return node representation
            return node_rep
        
        # Aggregate neighbor representations
        neighbor_reps = embeddings[node_type][neighbors]
        
        if aggregation == 'mean':
            aggregated = neighbor_reps.mean(dim=0)
        elif aggregation == 'sum':
            aggregated = neighbor_reps.sum(dim=0)
        elif aggregation == 'max':
            aggregated = neighbor_reps.max(dim=0)[0]
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        
        return aggregated
    
    def _get_neighbors(
        self,
        graph: Union[HeteroData, HeterogeneousGraph, dgl.DGLGraph],
        node_id: int,
        node_type: str,
        relation_type: str
    ) -> List[int]:
        """
        Get neighbors for a specific node and relation type.
        
        Args:
            graph: Graph object.
            node_id: Node index.
            node_type: Node type.
            relation_type: Relation type.
        
        Returns:
            List of neighbor indices.
        """
        if isinstance(graph, HeteroData):
            # PyG format
            edge_type = (node_type, relation_type, node_type)
            if edge_type in graph.edge_types:
                edge_index = graph[edge_type].edge_index
                neighbors = edge_index[1][edge_index[0] == node_id].tolist()
                return neighbors
            return []
        elif isinstance(graph, HeterogeneousGraph):
            # Custom graph
            neighbors = graph.get_neighbors(node_id, relation_type)
            return neighbors if neighbors else []
        elif isinstance(graph, dgl.DGLGraph):
            # DGL graph
            try:
                src, dst = graph.edges(etype=relation_type)
                neighbors = dst[src == node_id].tolist()
                return neighbors
            except:
                return []
        else:
            return []
    
    def compute_neighborhood_attention(
        self,
        graph: Union[HeteroData, HeterogeneousGraph, dgl.DGLGraph],
        node_id: int,
        node_type: str,
        neighbor_ids: List[int],
        node_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute attention weights between a node and its neighbors.
        
        Args:
            graph: Graph object.
            node_id: Index of the query node.
            node_type: Type of the query node.
            neighbor_ids: List of neighbor node indices.
            node_features: Optional node features.
        
        Returns:
            Tensor of attention weights for each neighbor.
        
        Raises:
            ValueError: If neighbor_ids is empty or node not found.
        """
        if not neighbor_ids:
            raise ValueError("neighbor_ids must not be empty")
        
        # Encode graph
        embeddings = self.encode_graph(graph, node_features)
        
        if node_type not in embeddings:
            raise ValueError(f"Node type '{node_type}' not found")
        
        # Get query node representation
        query_rep = embeddings[node_type][node_id]
        
        # Get neighbor representations
        neighbor_reps = embeddings[node_type][neighbor_ids]
        
        # Compute attention scores
        scores = torch.matmul(query_rep.unsqueeze(0), neighbor_reps.T)
        scores = scores.squeeze(0)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=0)
        
        return attention_weights
    
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
        """
        # Get node embeddings
        embeddings = self.encode_graph(graph, node_features)
        
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
                        dgl_graph.num_nodes(node_type), self.input_dim
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
    
    def _get_cache_key(
        self,
        graph: HeteroData,
        node_features: Dict[str, torch.Tensor]
    ) -> str:
        """
        Generate cache key for graph and features.
        
        Args:
            graph: PyG HeteroData object.
            node_features: Node features.
        
        Returns:
            Cache key string.
        """
        # Simple key based on node counts and feature shapes
        key_parts = []
        
        for node_type in graph.node_types:
            if hasattr(graph[node_type], 'x'):
                key_parts.append(f"{node_type}:{graph[node_type].x.shape}")
        
        for node_type, features in node_features.items():
            key_parts.append(f"feat_{node_type}:{features.shape}")
        
        return "_".join(key_parts)
    
    def enable_cache(self):
        """Enable embedding caching."""
        self.cache_enabled = True
        self.logger.log_info("Cache enabled")
    
    def disable_cache(self):
        """Disable embedding caching."""
        self.cache_enabled = False
        self.logger.log_info("Cache disabled")
    
    def clear_cache(self):
        """Clear embedding cache."""
        self.embedding_cache = {}
        self.logger.log_info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache statistics.
        """
        return {
            'cache_size': len(self.embedding_cache),
            'cache_enabled': self.cache_enabled
        }
    
    def get_embedding_dimensions(self) -> Dict[str, int]:
        """
        Get embedding dimensions for all components.
        
        Returns:
            Dict with dimension information.
        """
        return {
            'gnn_output': self.output_dim,
            'projection_dim': self.projection_dim,
            'component_dims': {
                'intrinsic': self.projection_dim,
                'collaborative': self.projection_dim,
                'interaction': self.projection_dim
            }
        }
    
    def to_device(self, device: torch.device) -> 'GNNEncoder':
        """
        Move all components to specified device.
        
        Args:
            device: PyTorch device.
        
        Returns:
            Self with components moved to device.
        """
        self.gnn_model.to(device)
        self.projection_heads.to(device)
        self.to(device)
        return self
    
    def get_parameters(self) -> Dict[str, int]:
        """
        Get parameter statistics.
        
        Returns:
            Dict with parameter counts.
        """
        gnn_params = self.gnn_model.get_parameters()
        projection_params = self.projection_heads.get_parameters()
        
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'gnn_model': gnn_params['total'],
            'projection_heads': projection_params['total'],
            'total': total_params
        }


# Module level variables and exports
__all__ = [
    'GNNEncoder',
    '__doc__'
]

# Version information
__version__ = '1.0.0'
__author__ = 'H-GRAGrecsys Team'


def create_gnn_encoder(
    config_path: Optional[str] = None,
    gnn_model: Optional[HeterogeneousGNN] = None,
    projection_heads: Optional[ComponentProjectionHeads] = None,
    device: Optional[torch.device] = None
) -> GNNEncoder:
    """
    Factory function to create a GNNEncoder instance.
    
    Args:
        config_path: Optional path to configuration file.
        gnn_model: Optional pre-initialized GNN model.
        projection_heads: Optional pre-initialized projection heads.
        device: Optional device to move model to. Defaults to CUDA if available.
    
    Returns:
        Initialized GNNEncoder instance.
    
    Example:
        >>> encoder = create_gnn_encoder('config/default_config.yaml')
        >>> encoder.to_device(torch.device('cuda'))
    """
    encoder = GNNEncoder(
        gnn_model=gnn_model,
        projection_heads=projection_heads,
        config=config_path
    )
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return encoder.to_device(device)