"""
Heterogeneous Graph Module
Implements the dynamic user-item interaction graph.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict


@dataclass
class GraphEdge:
    """Edge in the heterogeneous graph."""
    source: str
    target: str
    edge_type: str  # 'interact', 'similar_pref', 'co_interact', 'content_sim'
    weight: float
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'target': self.target,
            'edge_type': self.edge_type,
            'weight': self.weight,
            'metadata': self.metadata
        }


class HeterogeneousGraph:
    """
    Dynamic heterogeneous graph for agent interactions.
    Supports multiple edge types and dynamic weight updates.
    """
    
    def __init__(self):
        # Node storage
        self.user_nodes: Set[str] = set()
        self.item_nodes: Set[str] = set()
        
        # Edge storage by type
        self.edges: Dict[str, Dict[Tuple[str, str], GraphEdge]] = {
            'interact': {},
            'similar_pref': {},
            'co_interact': {},
            'content_sim': {}
        }
        
        # Adjacency lists for fast neighbor lookup
        self.adjacency: Dict[str, Dict[str, Set[str]]] = {
            'interact': defaultdict(set),
            'similar_pref': defaultdict(set),
            'co_interact': defaultdict(set),
            'content_sim': defaultdict(set)
        }
        
        # Node features (embeddings)
        self.node_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        
        # Statistics
        self.edge_count = 0
        self.update_count = 0
    
    def add_user(self, user_id: str):
        """Add a user node to the graph."""
        self.user_nodes.add(user_id)
    
    def add_item(self, item_id: str):
        """Add an item node to the graph."""
        self.item_nodes.add(item_id)
    
    def add_edge(self, source: str, target: str, edge_type: str, 
                 weight: float = 1.0, metadata: Optional[Dict] = None):
        """
        Add or update an edge in the graph.
        
        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Type of edge
            weight: Edge weight [0, 1]
            metadata: Optional edge metadata
        """
        if edge_type not in self.edges:
            raise ValueError(f"Invalid edge type: {edge_type}")
        
        # Ensure canonical ordering for undirected edges
        if edge_type in ['similar_pref', 'co_interact', 'content_sim']:
            source, target = sorted([source, target])
        
        edge_key = (source, target)
        edge = GraphEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            weight=weight,
            metadata=metadata or {}
        )
        
        self.edges[edge_type][edge_key] = edge
        self.adjacency[edge_type][source].add(target)
        self.adjacency[edge_type][target].add(source)
        self.edge_count += 1
    
    def update_edge_weight(self, source: str, target: str, edge_type: str,
                           delta: float, prune_threshold: float = 0.05):
        """
        Update edge weight with delta and optionally prune weak edges.
        
        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Type of edge
            delta: Weight change (+ for strengthen, - for weaken)
            prune_threshold: Minimum weight before pruning
        """
        if edge_type in ['similar_pref', 'co_interact', 'content_sim']:
            source, target = sorted([source, target])
        
        edge_key = (source, target)
        
        if edge_key in self.edges[edge_type]:
            edge = self.edges[edge_type][edge_key]
            edge.weight = max(0.0, min(1.0, edge.weight + delta))
            
            # Prune if weight too low
            if edge.weight < prune_threshold:
                self._remove_edge(source, target, edge_type)
            
            self.update_count += 1
    
    def _remove_edge(self, source: str, target: str, edge_type: str):
        """Remove an edge from the graph."""
        if edge_type in ['similar_pref', 'co_interact', 'content_sim']:
            source, target = sorted([source, target])
        
        edge_key = (source, target)
        
        if edge_key in self.edges[edge_type]:
            del self.edges[edge_type][edge_key]
            self.adjacency[edge_type][source].discard(target)
            self.adjacency[edge_type][target].discard(source)
            self.edge_count -= 1
    
    def get_neighbors(self, node_id: str, edge_types: Optional[List[str]] = None,
                      max_hops: int = 1) -> Dict[int, Set[str]]:
        """
        Get neighbors of a node up to max_hops away.
        
        Args:
            node_id: Center node ID
            edge_types: Types of edges to traverse (None = all)
            max_hops: Maximum number of hops
        
        Returns:
            Dictionary mapping hop number to set of neighbor IDs
        """
        if edge_types is None:
            edge_types = list(self.edges.keys())
        
        neighbors = {0: {node_id}}
        visited = {node_id}
        
        for hop in range(1, max_hops + 1):
            current_neighbors = set()
            
            for node in neighbors[hop - 1]:
                for edge_type in edge_types:
                    for neighbor in self.adjacency[edge_type].get(node, set()):
                        if neighbor not in visited:
                            current_neighbors.add(neighbor)
                            visited.add(neighbor)
            
            neighbors[hop] = current_neighbors
        
        return neighbors
    
    def get_edge_weight(self, source: str, target: str, edge_type: str) -> float:
        """Get weight of a specific edge."""
        if edge_type in ['similar_pref', 'co_interact', 'content_sim']:
            source, target = sorted([source, target])
        
        edge_key = (source, target)
        edge = self.edges[edge_type].get(edge_key)
        return edge.weight if edge else 0.0
    
    def set_node_embedding(self, node_id: str, tier: str, embedding: np.ndarray):
        """Store embedding for a node."""
        if node_id not in self.node_embeddings:
            self.node_embeddings[node_id] = {}
        self.node_embeddings[node_id][tier] = embedding
    
    def get_node_embedding(self, node_id: str, tier: str) -> Optional[np.ndarray]:
        """Retrieve embedding for a node."""
        return self.node_embeddings.get(node_id, {}).get(tier)
    
    def get_graph_statistics(self) -> Dict:
        """Get statistics about the graph."""
        return {
            'num_users': len(self.user_nodes),
            'num_items': len(self.item_nodes),
            'num_edges': self.edge_count,
            'edges_by_type': {
                edge_type: len(edges) 
                for edge_type, edges in self.edges.items()
            },
            'avg_degree': self.edge_count / max(1, len(self.user_nodes) + len(self.item_nodes)),
            'num_updates': self.update_count
        }