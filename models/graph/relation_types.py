"""
Relation Types Module for H-GRAGrecsys

This module defines all relation types used in the heterogeneous graph,
along with edge weight calculation functions and utility methods for
handling different types of relations between nodes.
"""

from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Any, Set, Union
import torch
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger import Logger
from utils.config_loader import ConfigLoader


class RelationType(Enum):
    """
    Enumeration of all relation types in the heterogeneous graph.
    
    These relations connect different types of nodes:
    - INTERACT: User-Item interaction
    - SIMILAR_PREF: User-User similarity in preferences
    - CO_INTER: User-User co-interaction (shared items)
    - CONTENT_SIM: Item-Item content similarity
    """
    
    INTERACT = "interact"
    SIMILAR_PREF = "similar_pref"
    CO_INTER = "co_inter"
    CONTENT_SIM = "content_sim"
    
    @classmethod
    def get_all_types(cls) -> List[str]:
        """Get all relation type names as strings."""
        return [e.value for e in cls]
    
    @classmethod
    def get_node_type_pairs(cls) -> Dict[str, Tuple[str, str]]:
        """
        Get the source and target node types for each relation.
        
        Returns:
            Dict mapping relation type to (source_type, target_type) tuple
        """
        return {
            cls.INTERACT.value: ('user', 'item'),
            cls.SIMILAR_PREF.value: ('user', 'user'),
            cls.CO_INTER.value: ('user', 'user'),
            cls.CONTENT_SIM.value: ('item', 'item')
        }
    
    @classmethod
    def get_relations_between(cls, source_type: str, target_type: str) -> List[str]:
        """
        Get all relation types between two node types.
        
        Args:
            source_type: Type of source node ('user' or 'item')
            target_type: Type of target node ('user' or 'item')
            
        Returns:
            List of relation type names
        """
        relations = []
        for rel_type, (src, tgt) in cls.get_node_type_pairs().items():
            if src == source_type and tgt == target_type:
                relations.append(rel_type)
        return relations
    
    @classmethod
    def is_valid_relation(cls, relation_type: str, source_type: str, target_type: str) -> bool:
        """
        Check if a relation is valid between given node types.
        
        Args:
            relation_type: Type of relation
            source_type: Type of source node
            target_type: Type of target node
            
        Returns:
            bool: True if relation is valid
        """
        valid_pairs = cls.get_node_type_pairs()
        return relation_type in valid_pairs and valid_pairs[relation_type] == (source_type, target_type)
    
    @classmethod
    def get_opposite_relation(cls, relation_type: str) -> Optional[str]:
        """
        Get the opposite direction of a relation (if applicable).
        
        Args:
            relation_type: Type of relation
            
        Returns:
            Optional[str]: Opposite relation type or None if symmetric
        """
        symmetric_relations = [cls.SIMILAR_PREF.value, cls.CO_INTER.value, cls.CONTENT_SIM.value]
        if relation_type in symmetric_relations:
            return relation_type  # These are symmetric
        elif relation_type == cls.INTERACT.value:
            # Reverse of interact is still interact (but direction matters)
            return cls.INTERACT.value
        return None


@dataclass
class RelationMetadata:
    """
    Metadata container for relation-specific information.
    
    Attributes:
        relation_type: Type of relation
        source_type: Type of source node
        target_type: Type of target node
        is_directed: Whether the relation is directed
        is_symmetric: Whether the relation is symmetric
        weight_range: Tuple of (min_weight, max_weight)
        default_weight: Default weight for the relation
        description: Human-readable description
    """
    relation_type: str
    source_type: str
    target_type: str
    is_directed: bool = True
    is_symmetric: bool = False
    weight_range: Tuple[float, float] = (0.0, 1.0)
    default_weight: float = 1.0
    description: str = ""
    
    def __post_init__(self):
        """Validate and set default description if not provided."""
        if not self.description:
            self.description = f"{self.source_type}->{self.target_type} ({self.relation_type})"


class EdgeWeightFunctions:
    """
    Static utility class for calculating edge weights between nodes.
    
    This class provides various similarity and weight calculation methods
    for different relation types.
    """
    
    @staticmethod
    def cosine_similarity(emb_a: Union[torch.Tensor, np.ndarray], 
                         emb_b: Union[torch.Tensor, np.ndarray]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            emb_a: First embedding vector
            emb_b: Second embedding vector
            
        Returns:
            float: Cosine similarity in [0, 1] range (clipped)
            
        Raises:
            ValueError: If embeddings have different dimensions
        """
        # Convert to numpy if tensors
        if torch.is_tensor(emb_a):
            emb_a = emb_a.detach().cpu().numpy()
        if torch.is_tensor(emb_b):
            emb_b = emb_b.detach().cpu().numpy()
        
        # Ensure 1D arrays
        if emb_a.ndim > 1:
            emb_a = emb_a.flatten()
        if emb_b.ndim > 1:
            emb_b = emb_b.flatten()
        
        if len(emb_a) != len(emb_b):
            raise ValueError(f"Embedding dimensions mismatch: {len(emb_a)} vs {len(emb_b)}")
        
        # Compute cosine similarity
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        similarity = np.dot(emb_a, emb_b) / (norm_a * norm_b)
        
        # Clip to [0, 1] range
        return max(0.0, min(1.0, similarity))
    
    @staticmethod
    def jaccard_similarity(set_a: Set[Any], set_b: Set[Any]) -> float:
        """
        Calculate Jaccard similarity between two sets.
        
        Args:
            set_a: First set
            set_b: Second set
            
        Returns:
            float: Jaccard similarity in [0, 1] range
        """
        if not set_a and not set_b:
            return 0.0
        
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def ppr_similarity(graph: Any, node_a: str, node_b: str, 
                      restart_prob: float = 0.15,
                      max_iter: int = 100,
                      tol: float = 1e-6) -> float:
        """
        Calculate Personalized PageRank similarity between two nodes.
        
        Args:
            graph: HeterogeneousGraph instance
            node_a: First node ID
            node_b: Second node ID
            restart_prob: Probability of restarting from node_a
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            
        Returns:
            float: PPR similarity in [0, 1] range
        """
        if node_a not in graph.nodes or node_b not in graph.nodes:
            return 0.0
        
        if node_a == node_b:
            return 1.0
        
        # Get adjacency matrix (simplified)
        # For a full implementation, we would use the graph's adjacency
        # Here we use a heuristic based on common neighbors
        neighbors_a = set()
        neighbors_b = set()
        
        # Get all neighbors
        for rel_type in graph.adjacency_lists:
            neighbors_a.update([n for n, _ in graph.get_neighbors(node_a, rel_type)])
            neighbors_b.update([n for n, _ in graph.get_neighbors(node_b, rel_type)])
        
        # Compute similarity based on common neighbors
        common = len(neighbors_a & neighbors_b)
        union = len(neighbors_a | neighbors_b)
        
        return common / union if union > 0 else 0.0
    
    @staticmethod
    def rating_normalization(rating: float, min_rating: float = 1.0, 
                            max_rating: float = 5.0) -> float:
        """
        Normalize a rating to [0, 1] range.
        
        Args:
            rating: Raw rating value
            min_rating: Minimum possible rating
            max_rating: Maximum possible rating
            
        Returns:
            float: Normalized rating
        """
        if max_rating == min_rating:
            return 1.0
        
        normalized = (rating - min_rating) / (max_rating - min_rating)
        return max(0.0, min(1.0, normalized))
    
    @staticmethod
    def time_decay_weight(timestamp: float, 
                         current_time: Optional[float] = None,
                         half_life: float = 86400.0) -> float:
        """
        Calculate time decay weight based on interaction recency.
        
        Args:
            timestamp: Timestamp of interaction
            current_time: Current timestamp (defaults to now)
            half_life: Half-life in seconds
            
        Returns:
            float: Decay weight in [0, 1] range
        """
        if current_time is None:
            import time
            current_time = time.time()
        
        age = current_time - timestamp
        if age <= 0:
            return 1.0
        
        # Exponential decay
        decay = np.exp(-np.log(2) * age / half_life)
        return max(0.0, min(1.0, decay))
    
    @staticmethod
    def combine_weights(weights: List[float], method: str = 'average') -> float:
        """
        Combine multiple weights into a single weight.
        
        Args:
            weights: List of weights to combine
            method: Combination method ('average', 'max', 'min', 'product')
            
        Returns:
            float: Combined weight
        """
        if not weights:
            return 0.0
        
        if method == 'average':
            return np.mean(weights)
        elif method == 'max':
            return max(weights)
        elif method == 'min':
            return min(weights)
        elif method == 'product':
            return np.prod(weights)
        else:
            raise ValueError(f"Unknown combination method: {method}")
    
    @staticmethod
    def similarity_to_weight(similarity: float, 
                            threshold: float = 0.0,
                            scaling: float = 1.0) -> float:
        """
        Convert similarity score to edge weight with optional threshold.
        
        Args:
            similarity: Raw similarity score
            threshold: Minimum similarity to consider
            scaling: Scaling factor
            
        Returns:
            float: Edge weight in [0, 1] range
        """
        if similarity < threshold:
            return 0.0
        
        weight = similarity * scaling
        return max(0.0, min(1.0, weight))


class RelationTypeRegistry:
    """
    Registry for managing relation types and their metadata.
    
    This class provides a central registry for all relation types,
    their properties, and utility functions.
    """
    
    _instance = None
    
    def __new__(cls, config: Optional[Dict[str, Any]] = None):
        """
        Singleton pattern to ensure single registry instance.
        
        Args:
            config: Configuration dictionary
        """
        if cls._instance is None:
            cls._instance = super(RelationTypeRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the relation type registry.
        
        Args:
            config: Configuration dictionary
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.config = config or {}
        self.logger = Logger.get_instance(log_dir='logs', name='relation_registry')
        
        # Initialize relation metadata
        self.relation_metadata: Dict[str, RelationMetadata] = {}
        self._register_default_relations()
        
        # Custom relation types from config
        self._register_custom_relations()
        
        self._initialized = True
        self.logger.log_info(f"Initialized RelationTypeRegistry with {len(self.relation_metadata)} relations")
    
    def _register_default_relations(self):
        """Register default relation types with their metadata."""
        default_relations = [
            RelationMetadata(
                relation_type=RelationType.INTERACT.value,
                source_type='user',
                target_type='item',
                is_directed=True,
                is_symmetric=False,
                weight_range=(0.0, 1.0),
                default_weight=1.0,
                description="User-item interaction edge"
            ),
            RelationMetadata(
                relation_type=RelationType.SIMILAR_PREF.value,
                source_type='user',
                target_type='user',
                is_directed=False,
                is_symmetric=True,
                weight_range=(0.0, 1.0),
                default_weight=0.5,
                description="User-user preference similarity"
            ),
            RelationMetadata(
                relation_type=RelationType.CO_INTER.value,
                source_type='user',
                target_type='user',
                is_directed=False,
                is_symmetric=True,
                weight_range=(0.0, 1.0),
                default_weight=0.3,
                description="User-user co-interaction (shared items)"
            ),
            RelationMetadata(
                relation_type=RelationType.CONTENT_SIM.value,
                source_type='item',
                target_type='item',
                is_directed=False,
                is_symmetric=True,
                weight_range=(0.0, 1.0),
                default_weight=0.5,
                description="Item-item content similarity"
            )
        ]
        
        for metadata in default_relations:
            self.relation_metadata[metadata.relation_type] = metadata
    
    def _register_custom_relations(self):
        """Register custom relation types from configuration."""
        custom_relations = self.config.get('model', {}).get('graph', {}).get('custom_relations', [])
        for rel_config in custom_relations:
            metadata = RelationMetadata(
                relation_type=rel_config.get('name'),
                source_type=rel_config.get('source_type'),
                target_type=rel_config.get('target_type'),
                is_directed=rel_config.get('is_directed', True),
                is_symmetric=rel_config.get('is_symmetric', False),
                weight_range=(rel_config.get('min_weight', 0.0), rel_config.get('max_weight', 1.0)),
                default_weight=rel_config.get('default_weight', 1.0),
                description=rel_config.get('description', 'Custom relation')
            )
            self.relation_metadata[metadata.relation_type] = metadata
    
    def get_metadata(self, relation_type: str) -> Optional[RelationMetadata]:
        """
        Get metadata for a relation type.
        
        Args:
            relation_type: Type of relation
            
        Returns:
            Optional[RelationMetadata]: Relation metadata or None
        """
        return self.relation_metadata.get(relation_type)
    
    def get_all_relations(self) -> List[str]:
        """
        Get all registered relation types.
        
        Returns:
            List[str]: All relation type names
        """
        return list(self.relation_metadata.keys())
    
    def get_relations_for_nodes(self, source_type: str, target_type: str) -> List[str]:
        """
        Get all relation types valid for given node types.
        
        Args:
            source_type: Type of source node
            target_type: Type of target node
            
        Returns:
            List[str]: Valid relation types
        """
        relations = []
        for rel_type, metadata in self.relation_metadata.items():
            if metadata.source_type == source_type and metadata.target_type == target_type:
                relations.append(rel_type)
        return relations
    
    def get_relations_by_source(self, source_type: str) -> List[str]:
        """
        Get all relation types from a given source node type.
        
        Args:
            source_type: Type of source node
            
        Returns:
            List[str]: Relation types
        """
        relations = []
        for rel_type, metadata in self.relation_metadata.items():
            if metadata.source_type == source_type:
                relations.append(rel_type)
        return relations
    
    def get_relations_by_target(self, target_type: str) -> List[str]:
        """
        Get all relation types to a given target node type.
        
        Args:
            target_type: Type of target node
            
        Returns:
            List[str]: Relation types
        """
        relations = []
        for rel_type, metadata in self.relation_metadata.items():
            if metadata.target_type == target_type:
                relations.append(rel_type)
        return relations
    
    def get_symmetric_relations(self) -> List[str]:
        """
        Get all symmetric relation types.
        
        Returns:
            List[str]: Symmetric relation types
        """
        relations = []
        for rel_type, metadata in self.relation_metadata.items():
            if metadata.is_symmetric:
                relations.append(rel_type)
        return relations
    
    def get_directed_relations(self) -> List[str]:
        """
        Get all directed relation types.
        
        Returns:
            List[str]: Directed relation types
        """
        relations = []
        for rel_type, metadata in self.relation_metadata.items():
            if metadata.is_directed:
                relations.append(rel_type)
        return relations
    
    def is_valid_relation(self, relation_type: str, 
                         source_type: Optional[str] = None,
                         target_type: Optional[str] = None) -> bool:
        """
        Check if a relation type is valid.
        
        Args:
            relation_type: Type of relation to check
            source_type: Optional source node type
            target_type: Optional target node type
            
        Returns:
            bool: True if relation is valid
        """
        if relation_type not in self.relation_metadata:
            return False
        
        if source_type and target_type:
            metadata = self.relation_metadata[relation_type]
            return metadata.source_type == source_type and metadata.target_type == target_type
        
        return True
    
    def get_weight_function(self, relation_type: str) -> Optional[str]:
        """
        Get the recommended weight calculation function for a relation.
        
        Args:
            relation_type: Type of relation
            
        Returns:
            Optional[str]: Name of weight function or None
        """
        weight_functions = {
            RelationType.INTERACT.value: 'rating_normalization',
            RelationType.SIMILAR_PREF.value: 'cosine_similarity',
            RelationType.CO_INTER.value: 'jaccard_similarity',
            RelationType.CONTENT_SIM.value: 'cosine_similarity'
        }
        return weight_functions.get(relation_type)
    
    def get_relation_stats(self, graph: Any) -> Dict[str, Any]:
        """
        Get statistics for each relation type in the graph.
        
        Args:
            graph: HeterogeneousGraph instance
            
        Returns:
            Dict[str, Any]: Statistics per relation type
        """
        stats = {}
        
        for rel_type in self.get_all_relations():
            # Count edges of this type
            edge_count = 0
            weights = []
            
            for (src, dst, rtype), weight in graph.edges.items():
                if rtype == rel_type:
                    edge_count += 1
                    weights.append(weight)
            
            metadata = self.get_metadata(rel_type)
            stats[rel_type] = {
                'count': edge_count,
                'metadata': {
                    'source_type': metadata.source_type if metadata else None,
                    'target_type': metadata.target_type if metadata else None,
                    'is_symmetric': metadata.is_symmetric if metadata else False,
                    'is_directed': metadata.is_directed if metadata else True
                }
            }
            
            if weights:
                stats[rel_type]['weight_stats'] = {
                    'mean': np.mean(weights),
                    'std': np.std(weights),
                    'min': np.min(weights),
                    'max': np.max(weights),
                    'median': np.median(weights)
                }
        
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert registry to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Registry data
        """
        return {
            'relations': {
                rel_type: {
                    'source_type': metadata.source_type,
                    'target_type': metadata.target_type,
                    'is_directed': metadata.is_directed,
                    'is_symmetric': metadata.is_symmetric,
                    'weight_range': metadata.weight_range,
                    'default_weight': metadata.default_weight,
                    'description': metadata.description
                }
                for rel_type, metadata in self.relation_metadata.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> 'RelationTypeRegistry':
        """
        Create registry from dictionary data.
        
        Args:
            data: Registry data dictionary
            config: Configuration dictionary
            
        Returns:
            RelationTypeRegistry: New registry instance
        """
        registry = cls(config)
        
        # Update metadata from data
        for rel_type, rel_data in data.get('relations', {}).items():
            metadata = RelationMetadata(
                relation_type=rel_type,
                source_type=rel_data['source_type'],
                target_type=rel_data['target_type'],
                is_directed=rel_data.get('is_directed', True),
                is_symmetric=rel_data.get('is_symmetric', False),
                weight_range=tuple(rel_data.get('weight_range', (0.0, 1.0))),
                default_weight=rel_data.get('default_weight', 1.0),
                description=rel_data.get('description', '')
            )
            registry.relation_metadata[rel_type] = metadata
        
        return registry


# Relation type-specific utility functions
class RelationUtils:
    """
    Utility functions for working with relation types.
    """
    
    @staticmethod
    def get_relation_priority(relation_type: str) -> int:
        """
        Get priority of a relation type for edge ranking.
        
        Args:
            relation_type: Type of relation
            
        Returns:
            int: Priority (higher = more important)
        """
        priorities = {
            RelationType.INTERACT.value: 10,
            RelationType.SIMILAR_PREF.value: 7,
            RelationType.CO_INTER.value: 5,
            RelationType.CONTENT_SIM.value: 6
        }
        return priorities.get(relation_type, 3)
    
    @staticmethod
    def get_relation_color(relation_type: str) -> str:
        """
        Get a color code for a relation type for visualization.
        
        Args:
            relation_type: Type of relation
            
        Returns:
            str: Hex color code
        """
        colors = {
            RelationType.INTERACT.value: '#4CAF50',      # Green
            RelationType.SIMILAR_PREF.value: '#2196F3',   # Blue
            RelationType.CO_INTER.value: '#FF9800',       # Orange
            RelationType.CONTENT_SIM.value: '#9C27B0'     # Purple
        }
        return colors.get(relation_type, '#607D8B')  # Gray default
    
    @staticmethod
    def get_relation_line_style(relation_type: str) -> str:
        """
        Get line style for a relation type for visualization.
        
        Args:
            relation_type: Type of relation
            
        Returns:
            str: Line style ('solid', 'dashed', 'dotted')
        """
        styles = {
            RelationType.INTERACT.value: 'solid',
            RelationType.SIMILAR_PREF.value: 'dashed',
            RelationType.CO_INTER.value: 'dotted',
            RelationType.CONTENT_SIM.value: 'dashdot'
        }
        return styles.get(relation_type, 'solid')
    
    @staticmethod
    def get_relation_weight_bounds(relation_type: str) -> Tuple[float, float]:
        """
        Get typical weight bounds for a relation type.
        
        Args:
            relation_type: Type of relation
            
        Returns:
            Tuple[float, float]: (min_weight, max_weight)
        """
        bounds = {
            RelationType.INTERACT.value: (0.0, 1.0),
            RelationType.SIMILAR_PREF.value: (0.0, 1.0),
            RelationType.CO_INTER.value: (0.0, 1.0),
            RelationType.CONTENT_SIM.value: (0.0, 1.0)
        }
        return bounds.get(relation_type, (0.0, 1.0))


# Example usage
if __name__ == "__main__":
    # Load configuration
    config_path = "config/default_config.yaml"
    config_loader = ConfigLoader(config_path)
    config = config_loader.load_config()
    
    # Initialize registry
    registry = RelationTypeRegistry(config)
    
    # Test enumeration
    print("All relation types:", RelationType.get_all_types())
    print("Node type pairs:", RelationType.get_node_type_pairs())
    print("Relations between user and item:", 
          RelationType.get_relations_between('user', 'item'))
    
    # Test weight functions
    emb_a = torch.randn(10)
    emb_b = torch.randn(10)
    similarity = EdgeWeightFunctions.cosine_similarity(emb_a, emb_b)
    print(f"Cosine similarity: {similarity:.4f}")
    
    set_a = {1, 2, 3, 4}
    set_b = {3, 4, 5, 6}
    jaccard = EdgeWeightFunctions.jaccard_similarity(set_a, set_b)
    print(f"Jaccard similarity: {jaccard:.4f}")
    
    # Test registry
    print("\nRegistry relations:", registry.get_all_relations())
    print("Symmetric relations:", registry.get_symmetric_relations())
    
    # Test relation utils
    priority = RelationUtils.get_relation_priority(RelationType.INTERACT.value)
    print(f"Interaction priority: {priority}")
    
    color = RelationUtils.get_relation_color(RelationType.INTERACT.value)
    print(f"Interaction color: {color}")
    
    # Test metadata
    metadata = registry.get_metadata(RelationType.INTERACT.value)
    if metadata:
        print(f"Interaction metadata: {metadata}")