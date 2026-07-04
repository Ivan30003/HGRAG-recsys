"""
GNN Module for H-GRAGrecsys

This module implements the Graph Neural Network components for the heterogeneous
graph representation learning in H-GRAGrecsys. It provides heterogeneous GNN layers,
projection heads for disentangled representations, attention mechanisms, and
graph encoding functionality.

The module supports:
- Heterogeneous message passing with multiple relation types
- Multi-head attention for graph and path representations
- Component-wise projection for intrinsic, collaborative, and interaction features
- Graph-level and node-level representation learning
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, GINConv, HeteroConv, Linear
from torch_geometric.data import HeteroData
import dgl
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from sibling modules
from models.gnn.heterogeneous_gnn import HeterogeneousGNN, HGNNLayer
from models.gnn.projection_heads import ProjectionHead, ComponentProjectionHeads
from models.gnn.gnn_encoder import GNNEncoder
from models.gnn.attention_module import AttentionModule

# Import from utils
from utils.config_loader import ConfigLoader
from utils.logger import Logger
from utils.seed_manager import SeedManager

# Import from graph module
from models.graph.heterogeneous_graph import HeterogeneousGraph
from models.graph.relation_types import RelationType


class GNNModule:
    """
    Main GNN module that orchestrates all GNN components for H-GRAGrecsys.
    
    This class serves as the entry point for GNN operations, managing the
    heterogeneous GNN model, projection heads, attention mechanisms, and
    providing unified interfaces for graph encoding and representation learning.
    """
    
    def __init__(self, config: Union[str, Dict, ConfigLoader]):
        """
        Initialize the GNN module with configuration.
        
        Args:
            config: Configuration object or path to config file.
                    Can be a string path, dict, or ConfigLoader instance.
        
        Raises:
            ValueError: If config is invalid or missing required fields.
        """
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
            name='gnn_module'
        )
        
        # Get GNN-specific config
        self.gnn_config = self.config.get('model', {}).get('gnn', {})
        self.graph_config = self.config.get('model', {}).get('graph', {})
        
        # Set random seed
        if 'seed' in self.gnn_config:
            SeedManager.set_seed(self.gnn_config['seed'])
        
        # Initialize model components
        self._initialize_components()
        
        self.logger.log_info("GNN Module initialized successfully")
    
    def _initialize_components(self):
        """Initialize all GNN sub-components with configuration."""
        # Extract dimensions
        self.input_dim = self.gnn_config.get('input_dim', 768)
        self.hidden_dim = self.gnn_config.get('hidden_dim', 256)
        self.output_dim = self.gnn_config.get('output_dim', 128)
        self.num_layers = self.gnn_config.get('num_layers', 3)
        self.num_heads = self.gnn_config.get('num_heads', 4)
        self.dropout = self.gnn_config.get('dropout', 0.1)
        
        # Relation types from graph config
        self.relation_types = self.graph_config.get('relation_types', [
            'interact', 'similar_pref', 'co_inter', 'content_sim'
        ])
        
        # Initialize attention module
        self.attention_module = AttentionModule({
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'hidden_dim': self.hidden_dim,
            'attention_type': self.gnn_config.get('attention_type', 'multi_head')
        })
        
        # Initialize projection heads
        self.projection_heads = ComponentProjectionHeads(
            input_dim=self.hidden_dim,
            config={
                'output_dim': self.output_dim,
                'dropout': self.dropout,
                'activation': self.gnn_config.get('activation', 'relu')
            }
        )
        
        # Initialize heterogeneous GNN
        self.gnn_model = HeterogeneousGNN({
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'relation_types': self.relation_types,
            'aggregation': self.gnn_config.get('aggregation', 'sum')
        })
        
        # Initialize GNN encoder
        self.gnn_encoder = GNNEncoder(
            gnn_model=self.gnn_model,
            projection_heads=self.projection_heads
        )
        
        self.logger.log_info(
            f"GNN components initialized: input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, output_dim={self.output_dim}, "
            f"layers={self.num_layers}, heads={self.num_heads}"
        )
    
    def encode_graph(
        self, 
        graph: HeteroData,
        node_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode the heterogeneous graph to obtain node representations.
        
        Args:
            graph: PyTorch Geometric HeteroData object containing the graph structure.
            node_features: Optional dict mapping node types to feature tensors.
                         If None, uses features from graph object.
        
        Returns:
            Dict mapping node types to encoded node embeddings.
        
        Raises:
            ValueError: If graph is empty or invalid.
        """
        if graph is None or not hasattr(graph, 'node_types'):
            raise ValueError("Invalid graph object provided")
        
        self.logger.log_info(f"Encoding graph with node types: {graph.node_types}")
        
        # Use features from graph if not provided
        if node_features is None:
            node_features = {}
            for node_type in graph.node_types:
                if hasattr(graph[node_type], 'x'):
                    node_features[node_type] = graph[node_type].x
                else:
                    raise ValueError(f"No features found for node type: {node_type}")
        
        # Encode graph using GNN encoder
        encoded_nodes = self.gnn_encoder.encode_graph(graph, node_features)
        
        self.logger.log_info(f"Graph encoding complete. Encoded {len(encoded_nodes)} node types")
        
        return encoded_nodes
    
    def get_node_representations(
        self,
        graph: HeteroData,
        node_ids: Dict[str, List[int]],
        node_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get representations for specific nodes in the graph.
        
        Args:
            graph: PyTorch Geometric HeteroData object.
            node_ids: Dict mapping node types to list of node indices.
            node_features: Optional node features. If None, uses graph features.
        
        Returns:
            Dict mapping node types to tensors of node representations.
        
        Raises:
            ValueError: If requested node types not found in graph.
        """
        if not node_ids:
            raise ValueError("node_ids must not be empty")
        
        # Ensure all requested node types exist
        for node_type in node_ids.keys():
            if node_type not in graph.node_types:
                raise ValueError(f"Node type '{node_type}' not found in graph")
        
        # Encode graph first
        all_encodings = self.encode_graph(graph, node_features)
        
        # Extract specific node representations
        result = {}
        for node_type, ids in node_ids.items():
            if node_type in all_encodings:
                result[node_type] = all_encodings[node_type][ids]
            else:
                raise ValueError(f"No encodings found for node type: {node_type}")
        
        return result
    
    def get_component_representation(
        self,
        graph: HeteroData,
        node_id: int,
        node_type: str,
        component_type: str,
        node_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Get specific component representation (intrinsic, collaborative, interaction).
        
        Args:
            graph: PyTorch Geometric HeteroData object.
            node_id: Index of the node.
            node_type: Type of the node (e.g., 'user', 'item').
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
        
        # Encode graph
        encodings = self.encode_graph(graph, node_features)
        
        if node_type not in encodings:
            raise ValueError(f"Node type '{node_type}' not found in encodings")
        
        # Get node representation
        node_rep = encodings[node_type][node_id]
        
        # Project to component space
        component_rep = self.projection_heads.get_component_projection(
            node_rep.unsqueeze(0),
            component_type
        )
        
        return component_rep.squeeze(0)
    
    def aggregate_neighborhood(
        self,
        graph: HeteroData,
        node_id: int,
        node_type: str,
        relation_type: Optional[str] = None,
        hop_count: int = 1,
        node_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Aggregate neighborhood information for a specific node.
        
        Args:
            graph: PyTorch Geometric HeteroData object.
            node_id: Index of the target node.
            node_type: Type of the target node.
            relation_type: Optional specific relation type to aggregate over.
                          If None, aggregates over all relations.
            hop_count: Number of hops to consider for aggregation.
            node_features: Optional node features.
        
        Returns:
            Tensor containing aggregated neighborhood representation.
        
        Raises:
            ValueError: If node or relation type not found.
        """
        if hop_count < 1:
            raise ValueError("hop_count must be at least 1")
        
        # Encode graph
        encodings = self.encode_graph(graph, node_features)
        
        if node_type not in encodings:
            raise ValueError(f"Node type '{node_type}' not found")
        
        # Get node representation
        node_rep = encodings[node_type][node_id]
        
        # Aggregate neighborhood based on relation type
        if relation_type is not None:
            # Single relation aggregation
            if relation_type not in self.relation_types:
                raise ValueError(f"Invalid relation_type: {relation_type}")
            neighbor_reps = self.gnn_encoder.aggregate_neighborhood(
                node_id, node_type, relation_type, hop_count
            )
        else:
            # Aggregate over all relation types
            neighbor_reps = {}
            for rel_type in self.relation_types:
                neighbor_reps[rel_type] = self.gnn_encoder.aggregate_neighborhood(
                    node_id, node_type, rel_type, hop_count
                )
            # Combine all relation aggregations
            neighbor_reps = torch.cat(list(neighbor_reps.values()), dim=0).mean(0)
        
        return neighbor_reps
    
    def compute_attention_weights(
        self,
        graph: HeteroData,
        node_id: int,
        node_type: str,
        neighbor_ids: List[int],
        node_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute attention weights between a node and its neighbors.
        
        Args:
            graph: PyTorch Geometric HeteroData object.
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
        encodings = self.encode_graph(graph, node_features)
        
        if node_type not in encodings:
            raise ValueError(f"Node type '{node_type}' not found")
        
        # Get query node representation
        query_rep = encodings[node_type][node_id]
        
        # Get neighbor representations
        neighbor_reps = encodings[node_type][neighbor_ids]
        
        # Compute attention weights
        attention_weights = self.attention_module.compute_attention(
            queries=query_rep.unsqueeze(0).repeat(len(neighbor_ids), 1),
            keys=neighbor_reps,
            values=neighbor_reps
        )
        
        return attention_weights
    
    def get_graph_embedding(self, graph: HeteroData) -> torch.Tensor:
        """
        Get a global graph-level embedding.
        
        Args:
            graph: PyTorch Geometric HeteroData object.
        
        Returns:
            Tensor containing graph-level embedding.
        
        Raises:
            ValueError: If graph is empty.
        """
        if graph is None or len(graph.node_types) == 0:
            raise ValueError("Graph is empty or None")
        
        # Encode graph
        encodings = self.encode_graph(graph)
        
        # Pool node embeddings to get graph embedding
        all_embeddings = []
        for node_type in encodings.keys():
            all_embeddings.append(encodings[node_type])
        
        # Concatenate and pool
        all_embeddings = torch.cat(all_embeddings, dim=0)
        graph_embedding = all_embeddings.mean(0)
        
        return graph_embedding
    
    def train_step(
        self,
        graph: HeteroData,
        target_nodes: Dict[str, torch.Tensor],
        node_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            graph: PyTorch Geometric HeteroData object.
            target_nodes: Dict mapping node types to target labels/tensors.
            node_features: Optional node features.
        
        Returns:
            Dict containing loss and metrics for the step.
        
        Raises:
            ValueError: If training components not properly initialized.
        """
        if self.gnn_model is None:
            raise ValueError("GNN model not initialized")
        
        # Set model to training mode
        self.gnn_model.train()
        
        # Encode graph
        encodings = self.encode_graph(graph, node_features)
        
        # Calculate losses for each node type
        total_loss = 0.0
        metrics = {}
        
        for node_type, target in target_nodes.items():
            if node_type in encodings:
                # Get predictions (simplified - actual loss depends on task)
                predictions = encodings[node_type]
                
                # Example: MSE loss for reconstruction
                if isinstance(target, torch.Tensor) and target.shape == predictions.shape:
                    loss = F.mse_loss(predictions, target)
                    total_loss += loss
                    metrics[f'{node_type}_loss'] = loss.item()
        
        # Return training metrics
        metrics['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return metrics
    
    def save_model(self, save_path: str) -> None:
        """
        Save the GNN model and all components to disk.
        
        Args:
            save_path: Path to save the model checkpoint.
        
        Raises:
            IOError: If unable to save the model.
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Prepare state dict
            state_dict = {
                'gnn_model': self.gnn_model.state_dict(),
                'projection_heads': self.projection_heads.state_dict(),
                'attention_module': self.attention_module.state_dict(),
                'gnn_encoder': self.gnn_encoder.state_dict(),
                'config': self.config,
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'relation_types': self.relation_types
            }
            
            torch.save(state_dict, save_path)
            self.logger.log_info(f"Model saved to {save_path}")
            
        except Exception as e:
            self.logger.log_error(f"Failed to save model: {e}")
            raise IOError(f"Unable to save model to {save_path}: {e}")
    
    def load_model(self, load_path: str) -> None:
        """
        Load the GNN model and all components from disk.
        
        Args:
            load_path: Path to the model checkpoint.
        
        Raises:
            FileNotFoundError: If checkpoint not found.
            RuntimeError: If unable to load the model.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(load_path, map_location='cpu')
            
            # Load model weights
            self.gnn_model.load_state_dict(checkpoint['gnn_model'])
            self.projection_heads.load_state_dict(checkpoint['projection_heads'])
            self.attention_module.load_state_dict(checkpoint['attention_module'])
            self.gnn_encoder.load_state_dict(checkpoint['gnn_encoder'])
            
            # Update config if different
            if 'config' in checkpoint:
                self.config = checkpoint['config']
            
            self.logger.log_info(f"Model loaded from {load_path}")
            
        except Exception as e:
            self.logger.log_error(f"Failed to load model: {e}")
            raise RuntimeError(f"Unable to load model from {load_path}: {e}")
    
    def get_model_parameters(self) -> Dict[str, int]:
        """
        Get statistics about model parameters.
        
        Returns:
            Dict containing parameter counts for each component.
        """
        params = {}
        
        if self.gnn_model is not None:
            params['gnn_model'] = sum(p.numel() for p in self.gnn_model.parameters())
        
        if self.projection_heads is not None:
            params['projection_heads'] = sum(p.numel() for p in self.projection_heads.parameters())
        
        if self.attention_module is not None:
            params['attention_module'] = sum(p.numel() for p in self.attention_module.parameters())
        
        params['total'] = sum(params.values())
        
        return params
    
    def to_device(self, device: torch.device) -> 'GNNModule':
        """
        Move all model components to the specified device.
        
        Args:
            device: PyTorch device object.
        
        Returns:
            Self with components moved to device.
        """
        self.gnn_model.to(device)
        self.projection_heads.to(device)
        self.attention_module.to(device)
        self.gnn_encoder.to(device)
        
        self.logger.log_info(f"Model moved to device: {device}")
        
        return self
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        Get training configuration for GNN components.
        
        Returns:
            Dict containing training configuration parameters.
        """
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'learning_rate': self.gnn_config.get('learning_rate', 1e-4),
            'batch_size': self.gnn_config.get('batch_size', 64),
            'num_epochs': self.gnn_config.get('num_epochs', 50)
        }


# Module level variables and exports
__all__ = [
    # Classes
    'GNNModule',
    'HeterogeneousGNN',
    'HGNNLayer',
    'ProjectionHead',
    'ComponentProjectionHeads',
    'GNNEncoder',
    'AttentionModule',
    
    # Module docstring
    '__doc__'
]

# Version information
__version__ = '1.0.0'
__author__ = 'H-GRAGrecsys Team'


def create_gnn_module(
    config_path: str,
    device: Optional[torch.device] = None
) -> GNNModule:
    """
    Factory function to create a GNN module instance.
    
    Args:
        config_path: Path to configuration file.
        device: Optional device to move model to. Defaults to CUDA if available.
    
    Returns:
        Initialized GNNModule instance.
    
    Example:
        >>> module = create_gnn_module('config/default_config.yaml')
        >>> module.to_device(torch.device('cuda'))
    """
    module = GNNModule(config_path)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return module.to_device(device)


# Initialize module-level logger
def _init_module_logger():
    """Initialize module-level logging."""
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger


module_logger = _init_module_logger()