"""
Heterogeneous Graph Module for H-GRAGrecsys

This module implements the core heterogeneous graph structure that supports
multiple node types (users, items) and relation types (interactions, similarities).
The graph maintains feature embeddings and supports dynamic updates.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
from collections import defaultdict
import networkx as nx
from dataclasses import dataclass, field

try:
    import dgl
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False

try:
    from torch_geometric.data import Data
    from torch_geometric.utils import to_undirected
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.graph.relation_types import RelationType, EdgeWeightFunctions
from utils.logger import Logger
from utils.config_loader import ConfigLoader


@dataclass
class GraphNode:
    """
    Represents a node in the heterogeneous graph.
    
    Attributes:
        node_id (str): Unique identifier for the node
        node_type (str): Type of node (e.g., 'user', 'item')
        features (Dict[str, Any]): Feature dictionary for the node
        embedding (Optional[torch.Tensor]): Embedding vector for the node
        metadata (Dict[str, Any]): Additional metadata
        created_at (float): Timestamp of node creation
        updated_at (float): Timestamp of last update
    """
    node_id: str
    node_type: str
    features: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: torch.tensor(0.0).item())
    updated_at: float = field(default_factory=lambda: torch.tensor(0.0).item())
    
    def get_feature(self, feature_name: str) -> Optional[Any]:
        """
        Get a specific feature value.
        
        Args:
            feature_name: Name of the feature to retrieve
            
        Returns:
            Feature value or None if not found
        """
        return self.features.get(feature_name)
    
    def update_feature(self, feature_name: str, value: Any) -> None:
        """
        Update or add a feature value.
        
        Args:
            feature_name: Name of the feature to update
            value: New feature value
        """
        self.features[feature_name] = value
        self.updated_at = torch.tensor(torch.tensor(0.0).item()).item()
    
    def update_embedding(self, embedding: torch.Tensor) -> None:
        """
        Update the node embedding.
        
        Args:
            embedding: New embedding tensor
        """
        self.embedding = embedding
        self.updated_at = torch.tensor(torch.tensor(0.0).item()).item()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'features': self.features,
            'embedding': self.embedding.numpy().tolist() if self.embedding is not None else None,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphNode':
        """Create a GraphNode from dictionary data."""
        node = cls(
            node_id=data['node_id'],
            node_type=data['node_type'],
            features=data.get('features', {}),
            metadata=data.get('metadata', {})
        )
        if data.get('embedding') is not None:
            node.embedding = torch.tensor(data['embedding'])
        node.created_at = data.get('created_at', torch.tensor(0.0).item())
        node.updated_at = data.get('updated_at', torch.tensor(0.0).item())
        return node


class HeterogeneousGraph:
    """
    Main heterogeneous graph class supporting multiple node and edge types.
    
    This graph maintains:
    - Nodes: Different types (user, item) with features and embeddings
    - Edges: Different relation types (interact, similar_pref, co_inter, content_sim)
    - Dynamic updates: Edge weights can be updated over time
    - Graph statistics: Track density, degree distributions, etc.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the heterogeneous graph.
        
        Args:
            config: Configuration dictionary containing graph parameters
        """
        self.config = config
        self.logger = Logger.get_instance(log_dir='logs', name='heterogeneous_graph')
        
        # Graph structures
        self.nodes: Dict[str, GraphNode] = {}
        self.node_id_to_type: Dict[str, str] = {}
        self.type_to_nodes: Dict[str, List[str]] = defaultdict(list)
        
        # Edge storage: (source, target, relation_type) -> weight
        self.edges: Dict[Tuple[str, str, str], float] = {}
        self.adjacency_lists: Dict[str, Dict[str, List[Tuple[str, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )  # relation_type -> {node_id: [(neighbor_id, weight)]}
        
        # Reverse adjacency for efficient lookups
        self.reverse_adjacency: Dict[str, Dict[str, List[Tuple[str, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Edge metadata
        self.edge_metadata: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        
        # Statistics tracking
        self.graph_stats: Dict[str, Any] = {
            'total_nodes': 0,
            'total_edges': 0,
            'edge_distribution': defaultdict(int),
            'node_distribution': defaultdict(int),
            'density': 0.0,
            'avg_degree': 0.0
        }
        
        # Configuration parameters
        self.embedding_dim = config.get('model', {}).get('gnn', {}).get('hidden_dim', 256)
        self.max_nodes = config.get('model', {}).get('graph', {}).get('max_nodes', 100000)
        self.default_edge_weight = 1.0
        
        self.logger.log_info(f"Initialized HeterogeneousGraph with embedding_dim={self.embedding_dim}")
    
    def add_node(self, node_id: str, node_type: str, 
                 features: Optional[Dict[str, Any]] = None,
                 embedding: Optional[torch.Tensor] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> GraphNode:
        """
        Add a new node to the graph.
        
        Args:
            node_id: Unique identifier for the node
            node_type: Type of node ('user' or 'item')
            features: Feature dictionary for the node
            embedding: Initial embedding tensor for the node
            metadata: Additional metadata
            
        Returns:
            GraphNode: The created or existing node
            
        Raises:
            ValueError: If node_id already exists
            ValueError: If graph exceeds max_nodes limit
        """
        if node_id in self.nodes:
            self.logger.log_warning(f"Node {node_id} already exists, returning existing node")
            return self.nodes[node_id]
        
        if len(self.nodes) >= self.max_nodes:
            raise ValueError(f"Cannot add node {node_id}. Graph exceeds maximum nodes {self.max_nodes}")
        
        # Create node with timestamp
        current_time = torch.tensor(torch.tensor(0.0).item()).item()
        node = GraphNode(
            node_id=node_id,
            node_type=node_type,
            features=features or {},
            embedding=embedding,
            metadata=metadata or {},
            created_at=current_time,
            updated_at=current_time
        )
        
        # Store node
        self.nodes[node_id] = node
        self.node_id_to_type[node_id] = node_type
        self.type_to_nodes[node_type].append(node_id)
        
        # Update statistics
        self.graph_stats['total_nodes'] = len(self.nodes)
        self.graph_stats['node_distribution'][node_type] += 1
        
        self.logger.log_info(f"Added node {node_id} of type {node_type}")
        return node
    
    def add_edge(self, source: str, target: str, relation_type: Union[str, RelationType],
                weight: Optional[float] = None,
                metadata: Optional[Dict[str, Any]] = None) -> float:
        """
        Add an edge between two nodes in the graph.
        
        Args:
            source: Source node ID
            target: Target node ID
            relation_type: Type of relation (as string or RelationType enum)
            weight: Edge weight (calculated automatically if None)
            metadata: Additional edge metadata
            
        Returns:
            float: The actual weight of the edge
            
        Raises:
            ValueError: If source or target nodes don't exist
            ValueError: If source and target are the same node
        """
        # Validate nodes exist
        if source not in self.nodes:
            raise ValueError(f"Source node {source} does not exist in graph")
        if target not in self.nodes:
            raise ValueError(f"Target node {target} does not exist in graph")
        if source == target:
            self.logger.log_warning(f"Skipping self-loop edge for node {source}")
            return 0.0
        
        # Convert relation type to string
        if isinstance(relation_type, RelationType):
            relation_type = relation_type.value
        elif not isinstance(relation_type, str):
            raise ValueError(f"relation_type must be str or RelationType, got {type(relation_type)}")
        
        # Calculate edge weight if not provided
        if weight is None:
            weight = self._calculate_edge_weight(source, target, relation_type)
        
        # Store edge
        edge_key = (source, target, relation_type)
        self.edges[edge_key] = weight
        
        # Update adjacency lists
        self.adjacency_lists[relation_type][source].append((target, weight))
        self.reverse_adjacency[relation_type][target].append((source, weight))
        
        # Store metadata
        if metadata:
            self.edge_metadata[edge_key] = metadata
        
        # Update statistics
        self.graph_stats['total_edges'] = len(self.edges)
        self.graph_stats['edge_distribution'][relation_type] += 1
        self._update_density_statistics()
        
        self.logger.log_info(f"Added edge {source} -> {target} (type: {relation_type}, weight: {weight:.4f})")
        return weight
    
    def _calculate_edge_weight(self, source: str, target: str, relation_type: str) -> float:
        """
        Calculate edge weight based on relation type and node features.
        
        Args:
            source: Source node ID
            target: Target node ID
            relation_type: Type of relation
            
        Returns:
            float: Calculated edge weight between 0 and 1
        """
        source_node = self.nodes[source]
        target_node = self.nodes[target]
        
        if relation_type == RelationType.INTERACT.value:
            # For interaction edges, default weight is 1.0
            return 1.0
        
        elif relation_type == RelationType.SIMILAR_PREF.value:
            # Calculate user similarity using cosine similarity of embeddings
            if source_node.embedding is not None and target_node.embedding is not None:
                return EdgeWeightFunctions.cosine_similarity(
                    source_node.embedding, target_node.embedding
                )
            return 0.0
        
        elif relation_type == RelationType.CO_INTER.value:
            # Co-interaction similarity based on Jaccard of item sets
            # Check for interaction history in features
            source_items = set(source_node.features.get('interacted_items', []))
            target_items = set(target_node.features.get('interacted_items', []))
            return EdgeWeightFunctions.jaccard_similarity(source_items, target_items)
        
        elif relation_type == RelationType.CONTENT_SIM.value:
            # Content similarity for items
            if source_node.embedding is not None and target_node.embedding is not None:
                return EdgeWeightFunctions.cosine_similarity(
                    source_node.embedding, target_node.embedding
                )
            return 0.0
        
        else:
            # Default weight for unknown relation types
            self.logger.log_warning(f"Unknown relation type {relation_type}, using default weight")
            return self.default_edge_weight
    
    def get_neighbors(self, node_id: str, relation_type: Optional[Union[str, RelationType]] = None,
                     max_neighbors: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Get neighbors of a node for a specific relation type.
        
        Args:
            node_id: Query node ID
            relation_type: Type of relation (if None, return all neighbors)
            max_neighbors: Maximum number of neighbors to return
            
        Returns:
            List of (neighbor_id, weight) tuples
        """
        if node_id not in self.nodes:
            self.logger.log_warning(f"Node {node_id} does not exist")
            return []
        
        # Convert relation type
        if isinstance(relation_type, RelationType):
            relation_type = relation_type.value
        
        if relation_type is not None:
            neighbors = self.adjacency_lists.get(relation_type, {}).get(node_id, [])
        else:
            # Aggregate all relation types
            all_neighbors = []
            for rel_type, adj_dict in self.adjacency_lists.items():
                all_neighbors.extend(adj_dict.get(node_id, []))
            neighbors = all_neighbors
        
        # Sort by weight (higher weight first)
        neighbors.sort(key=lambda x: x[1], reverse=True)
        
        # Apply limit if specified
        if max_neighbors is not None:
            neighbors = neighbors[:max_neighbors]
        
        return neighbors
    
    def get_edge_weight(self, source: str, target: str, relation_type: Union[str, RelationType]) -> Optional[float]:
        """
        Get the weight of a specific edge.
        
        Args:
            source: Source node ID
            target: Target node ID
            relation_type: Type of relation
            
        Returns:
            float or None: Edge weight if exists, None otherwise
        """
        if isinstance(relation_type, RelationType):
            relation_type = relation_type.value
        
        edge_key = (source, target, relation_type)
        return self.edges.get(edge_key)
    
    def update_edge_weight(self, source: str, target: str, relation_type: Union[str, RelationType],
                          new_weight: float) -> bool:
        """
        Update the weight of an existing edge.
        
        Args:
            source: Source node ID
            target: Target node ID
            relation_type: Type of relation
            new_weight: New edge weight
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        if isinstance(relation_type, RelationType):
            relation_type = relation_type.value
        
        edge_key = (source, target, relation_type)
        
        if edge_key not in self.edges:
            self.logger.log_warning(f"Edge {edge_key} does not exist")
            return False
        
        # Update weight
        self.edges[edge_key] = new_weight
        
        # Update adjacency lists
        for i, (neighbor, _) in enumerate(self.adjacency_lists[relation_type][source]):
            if neighbor == target:
                self.adjacency_lists[relation_type][source][i] = (neighbor, new_weight)
                break
        
        for i, (neighbor, _) in enumerate(self.reverse_adjacency[relation_type][target]):
            if neighbor == source:
                self.reverse_adjacency[relation_type][target][i] = (neighbor, new_weight)
                break
        
        self.logger.log_info(f"Updated edge {source} -> {target} (type: {relation_type}) to {new_weight:.4f}")
        return True
    
    def remove_edge(self, source: str, target: str, relation_type: Union[str, RelationType]) -> bool:
        """
        Remove an edge from the graph.
        
        Args:
            source: Source node ID
            target: Target node ID
            relation_type: Type of relation
            
        Returns:
            bool: True if removal was successful, False otherwise
        """
        if isinstance(relation_type, RelationType):
            relation_type = relation_type.value
        
        edge_key = (source, target, relation_type)
        
        if edge_key not in self.edges:
            self.logger.log_warning(f"Edge {edge_key} does not exist")
            return False
        
        # Remove from edge storage
        del self.edges[edge_key]
        
        # Remove from adjacency lists
        if source in self.adjacency_lists[relation_type]:
            self.adjacency_lists[relation_type][source] = [
                (n, w) for n, w in self.adjacency_lists[relation_type][source] if n != target
            ]
            if not self.adjacency_lists[relation_type][source]:
                del self.adjacency_lists[relation_type][source]
        
        # Remove from reverse adjacency
        if target in self.reverse_adjacency[relation_type]:
            self.reverse_adjacency[relation_type][target] = [
                (n, w) for n, w in self.reverse_adjacency[relation_type][target] if n != source
            ]
            if not self.reverse_adjacency[relation_type][target]:
                del self.reverse_adjacency[relation_type][target]
        
        # Update statistics
        self.graph_stats['total_edges'] = len(self.edges)
        self.graph_stats['edge_distribution'][relation_type] -= 1
        if self.graph_stats['edge_distribution'][relation_type] <= 0:
            del self.graph_stats['edge_distribution'][relation_type]
        
        self._update_density_statistics()
        
        self.logger.log_info(f"Removed edge {source} -> {target} (type: {relation_type})")
        return True
    
    def get_node_features(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get features of a node.
        
        Args:
            node_id: Node ID
            
        Returns:
            Dict[str, Any] or None: Node features if exists, None otherwise
        """
        if node_id not in self.nodes:
            self.logger.log_warning(f"Node {node_id} does not exist")
            return None
        return self.nodes[node_id].features
    
    def update_node_features(self, node_id: str, features: Dict[str, Any]) -> bool:
        """
        Update features of an existing node.
        
        Args:
            node_id: Node ID
            features: New features to merge/update
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        if node_id not in self.nodes:
            self.logger.log_warning(f"Node {node_id} does not exist")
            return False
        
        self.nodes[node_id].features.update(features)
        self.nodes[node_id].updated_at = torch.tensor(torch.tensor(0.0).item()).item()
        self.logger.log_info(f"Updated features for node {node_id}")
        return True
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive graph statistics.
        
        Returns:
            Dict[str, Any]: Dictionary containing graph statistics
        """
        # Update statistics before returning
        self._update_density_statistics()
        self._compute_degree_statistics()
        
        # Add additional statistics
        stats = self.graph_stats.copy()
        stats.update({
            'num_node_types': len(self.type_to_nodes),
            'num_relation_types': len(self.adjacency_lists),
            'node_types': {k: len(v) for k, v in self.type_to_nodes.items()},
            'relation_types': {k: len(v) for k, v in self.adjacency_lists.items()},
            'embedding_dim': self.embedding_dim,
            'nodes_with_embeddings': sum(1 for n in self.nodes.values() if n.embedding is not None)
        })
        
        return stats
    
    def _update_density_statistics(self):
        """Update graph density statistics."""
        total_possible = len(self.nodes) * (len(self.nodes) - 1)
        if total_possible > 0:
            self.graph_stats['density'] = (2 * self.graph_stats['total_edges']) / total_possible
        else:
            self.graph_stats['density'] = 0.0
    
    def _compute_degree_statistics(self):
        """Compute degree statistics for all nodes."""
        degrees = []
        for node_id in self.nodes:
            degree = sum(len(adj.get(node_id, [])) for adj in self.adjacency_lists.values())
            degrees.append(degree)
        
        if degrees:
            self.graph_stats['avg_degree'] = np.mean(degrees)
            self.graph_stats['max_degree'] = np.max(degrees)
            self.graph_stats['min_degree'] = np.min(degrees)
            self.graph_stats['std_degree'] = np.std(degrees)
    
    def to_dgl(self) -> 'dgl.DGLGraph':
        """
        Convert the graph to DGL format.
        
        Returns:
            dgl.DGLGraph: DGL graph representation
            
        Raises:
            ImportError: If DGL is not installed
        """
        if not DGL_AVAILABLE:
            raise ImportError("DGL is not installed. Install with: pip install dgl")
        
        # Build node mapping
        node_ids = list(self.nodes.keys())
        node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        
        # Prepare edge lists
        src_indices = []
        dst_indices = []
        edge_types = []
        edge_weights = []
        
        for (src, dst, rel_type), weight in self.edges.items():
            src_idx = node_id_to_idx[src]
            dst_idx = node_id_to_idx[dst]
            src_indices.append(src_idx)
            dst_indices.append(dst_idx)
            edge_types.append(rel_type)
            edge_weights.append(weight)
        
        # Create DGL graph
        src_tensor = torch.tensor(src_indices, dtype=torch.long)
        dst_tensor = torch.tensor(dst_indices, dtype=torch.long)
        g = dgl.graph((src_tensor, dst_tensor))
        
        # Add node features
        node_types = [self.node_id_to_type[nid] for nid in node_ids]
        g.ndata['node_type'] = torch.tensor([hash(t) % 1000 for t in node_types], dtype=torch.long)
        g.ndata['node_id'] = torch.tensor([hash(nid) % 1000 for nid in node_ids], dtype=torch.long)
        
        # Add edge features
        g.edata['weight'] = torch.tensor(edge_weights, dtype=torch.float)
        g.edata['relation_type'] = torch.tensor([hash(rt) % 1000 for rt in edge_types], dtype=torch.long)
        
        # Add node embeddings if available
        embeddings = []
        for nid in node_ids:
            node = self.nodes[nid]
            if node.embedding is not None:
                embeddings.append(node.embedding)
            else:
                embeddings.append(torch.zeros(self.embedding_dim))
        g.ndata['embedding'] = torch.stack(embeddings)
        
        self.logger.log_info(f"Converted graph to DGL format with {g.num_nodes()} nodes and {g.num_edges()} edges")
        return g
    
    def to_pytorch_geometric(self) -> 'Data':
        """
        Convert the graph to PyTorch Geometric format.
        
        Returns:
            torch_geometric.data.Data: PyG data object
            
        Raises:
            ImportError: If PyTorch Geometric is not installed
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric is not installed. Install with: pip install torch-geometric")
        
        # Build node mapping
        node_ids = list(self.nodes.keys())
        node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        
        # Prepare edge lists
        src_indices = []
        dst_indices = []
        edge_weights = []
        edge_attr_types = []
        
        for (src, dst, rel_type), weight in self.edges.items():
            src_idx = node_id_to_idx[src]
            dst_idx = node_id_to_idx[dst]
            src_indices.append(src_idx)
            dst_indices.append(dst_idx)
            edge_weights.append(weight)
            edge_attr_types.append(hash(rel_type) % 1000)
        
        # Create PyG data object
        edge_index = torch.tensor([src_indices + dst_indices, dst_indices + src_indices], dtype=torch.long)
        edge_weights_tensor = torch.tensor(edge_weights + edge_weights, dtype=torch.float)
        
        # Get node features
        node_features = []
        for nid in node_ids:
            node = self.nodes[nid]
            if node.embedding is not None:
                node_features.append(node.embedding)
            else:
                node_features.append(torch.zeros(self.embedding_dim))
        
        x = torch.stack(node_features)
        
        # Create data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weights_tensor,
            num_nodes=len(node_ids)
        )
        
        # Add node types as additional features
        node_types = [hash(self.node_id_to_type[nid]) % 1000 for nid in node_ids]
        data.node_type = torch.tensor(node_types, dtype=torch.long)
        
        self.logger.log_info(f"Converted graph to PyTorch Geometric format with {data.num_nodes} nodes")
        return data
    
    def to_networkx(self) -> nx.MultiDiGraph:
        """
        Convert the graph to NetworkX MultiDiGraph format.
        
        Returns:
            nx.MultiDiGraph: NetworkX graph representation
        """
        G = nx.MultiDiGraph()
        
        # Add nodes
        for node_id, node in self.nodes.items():
            G.add_node(node_id, 
                      node_type=node.node_type,
                      features=node.features,
                      embedding=node.embedding.numpy().tolist() if node.embedding is not None else None)
        
        # Add edges
        for (src, dst, rel_type), weight in self.edges.items():
            G.add_edge(src, dst, key=rel_type, weight=weight, relation_type=rel_type)
        
        self.logger.log_info(f"Converted graph to NetworkX format with {G.number_of_nodes()} nodes")
        return G
    
    def get_connection_count(self, node_id: str) -> int:
        """
        Get the total number of connections for a node.
        
        Args:
            node_id: Node ID
            
        Returns:
            int: Number of connections
        """
        if node_id not in self.nodes:
            self.logger.log_warning(f"Node {node_id} does not exist")
            return 0
        
        total = 0
        for adj_dict in self.adjacency_lists.values():
            total += len(adj_dict.get(node_id, []))
        return total
    
    def get_edge_types_between(self, source: str, target: str) -> List[str]:
        """
        Get all relation types between two nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            List[str]: List of relation types between the nodes
        """
        edge_types = []
        for rel_type in self.adjacency_lists.keys():
            if (source, target, rel_type) in self.edges:
                edge_types.append(rel_type)
        return edge_types
    
    def get_subgraph(self, node_ids: List[str]) -> 'HeterogeneousGraph':
        """
        Extract a subgraph containing only specified nodes and their interconnections.
        
        Args:
            node_ids: List of node IDs to include in the subgraph
            
        Returns:
            HeterogeneousGraph: Subgraph containing only the specified nodes
        """
        # Create new graph with same config
        subgraph = HeterogeneousGraph(self.config)
        
        # Add nodes
        for node_id in node_ids:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                subgraph.add_node(
                    node_id=node_id,
                    node_type=node.node_type,
                    features=node.features.copy(),
                    embedding=node.embedding.clone() if node.embedding is not None else None,
                    metadata=node.metadata.copy()
                )
        
        # Add edges between nodes in the subset
        node_set = set(node_ids)
        for (src, dst, rel_type), weight in self.edges.items():
            if src in node_set and dst in node_set:
                subgraph.add_edge(src, dst, rel_type, weight)
        
        self.logger.log_info(f"Extracted subgraph with {subgraph.graph_stats['total_nodes']} nodes")
        return subgraph
    
    def clear(self) -> None:
        """Clear all graph data."""
        self.nodes.clear()
        self.node_id_to_type.clear()
        self.type_to_nodes.clear()
        self.edges.clear()
        self.adjacency_lists.clear()
        self.reverse_adjacency.clear()
        self.edge_metadata.clear()
        self.graph_stats = {
            'total_nodes': 0,
            'total_edges': 0,
            'edge_distribution': defaultdict(int),
            'node_distribution': defaultdict(int),
            'density': 0.0,
            'avg_degree': 0.0
        }
        self.logger.log_info("Cleared all graph data")
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save the entire graph to a file.
        
        Args:
            filepath: Path to save the graph
        """
        import pickle
        save_data = {
            'nodes': {nid: node.to_dict() for nid, node in self.nodes.items()},
            'edges': self.edges,
            'adjacency_lists': dict(self.adjacency_lists),
            'reverse_adjacency': dict(self.reverse_adjacency),
            'edge_metadata': self.edge_metadata,
            'graph_stats': self.graph_stats,
            'config': self.config,
            'node_id_to_type': self.node_id_to_type,
            'type_to_nodes': dict(self.type_to_nodes)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        self.logger.log_info(f"Saved graph to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str, config: Optional[Dict[str, Any]] = None) -> 'HeterogeneousGraph':
        """
        Load a graph from a file.
        
        Args:
            filepath: Path to load the graph from
            config: Optional configuration to override saved config
            
        Returns:
            HeterogeneousGraph: Loaded graph instance
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Use provided config or saved config
        config = config or data.get('config', {})
        graph = cls(config)
        
        # Restore nodes
        for node_id, node_data in data['nodes'].items():
            node = GraphNode.from_dict(node_data)
            graph.nodes[node_id] = node
            graph.node_id_to_type[node_id] = node.node_type
            graph.type_to_nodes[node.node_type].append(node_id)
        
        # Restore edges
        graph.edges = data['edges']
        graph.adjacency_lists = defaultdict(lambda: defaultdict(list), data['adjacency_lists'])
        graph.reverse_adjacency = defaultdict(lambda: defaultdict(list), data['reverse_adjacency'])
        graph.edge_metadata = data.get('edge_metadata', {})
        graph.graph_stats = data.get('graph_stats', graph.graph_stats)
        
        # Update statistics
        graph.graph_stats['total_nodes'] = len(graph.nodes)
        graph.graph_stats['total_edges'] = len(graph.edges)
        
        graph.logger.log_info(f"Loaded graph from {filepath} with {graph.graph_stats['total_nodes']} nodes")
        return graph
    
    def __len__(self) -> int:
        """Return number of nodes in the graph."""
        return len(self.nodes)
    
    def __str__(self) -> str:
        """String representation of the graph."""
        return f"HeterogeneousGraph(nodes={len(self.nodes)}, edges={len(self.edges)}, types={len(self.type_to_nodes)})"


# Example usage and testing
if __name__ == "__main__":
    # Create configuration
    config = {
        'model': {
            'gnn': {
                'hidden_dim': 256
            },
            'graph': {
                'max_nodes': 10000
            }
        }
    }
    
    # Initialize graph
    graph = HeterogeneousGraph(config)
    
    # Add some nodes
    user1 = graph.add_node('user_1', 'user', features={'name': 'Alice', 'age': 25})
    user2 = graph.add_node('user_2', 'user', features={'name': 'Bob', 'age': 30})
    item1 = graph.add_node('item_1', 'item', features={'title': 'Product A', 'category': 'electronics'})
    item2 = graph.add_node('item_2', 'item', features={'title': 'Product B', 'category': 'books'})
    
    # Add some edges
    graph.add_edge('user_1', 'item_1', RelationType.INTERACT, weight=1.0)
    graph.add_edge('user_2', 'item_2', RelationType.INTERACT, weight=0.8)
    graph.add_edge('user_1', 'user_2', RelationType.SIMILAR_PREF, weight=0.6)
    
    # Get statistics
    stats = graph.get_graph_statistics()
    print(f"Graph stats: {stats}")
    
    # Get neighbors
    neighbors = graph.get_neighbors('user_1', RelationType.INTERACT)
    print(f"Neighbors of user_1: {neighbors}")
    
    # Convert to different formats (if available)
    try:
        dgl_graph = graph.to_dgl()
        print(f"DGL graph: {dgl_graph}")
    except ImportError:
        print("DGL not available")
    
    try:
        pyg_data = graph.to_pytorch_geometric()
        print(f"PyG data: {pyg_data}")
    except ImportError:
        print("PyTorch Geometric not available")
    
    # Save and load
    graph.save_to_file('test_graph.pkl')
    loaded_graph = HeterogeneousGraph.load_from_file('test_graph.pkl')
    print(f"Loaded graph: {loaded_graph}")
    
    # Clean up
    os.remove('test_graph.pkl')