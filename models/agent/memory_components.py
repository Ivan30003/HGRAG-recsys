"""
memory_components.py - Memory components for agents in H-GRAGrecsys

This module provides the individual memory components that make up
the hierarchical memory system, including intrinsic, collaborative,
and interaction memories.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from datetime import datetime
from collections import defaultdict, deque
import numpy as np
from abc import ABC, abstractmethod

# Configure logging
logger = logging.getLogger(__name__)


class MemoryComponent(ABC):
    """
    Abstract base class for memory components.
    
    This class defines the interface for all memory components,
    including storage, retrieval, and embedding generation.
    """
    
    def __init__(self, 
                 component_id: str,
                 agent_id: str,
                 agent_type: str,
                 config: Dict[str, Any]):
        """
        Initialize MemoryComponent.
        
        Args:
            component_id: Unique identifier for the component
            agent_id: Agent identifier
            agent_type: Type of agent ('user' or 'item')
            config: Configuration dictionary
        """
        self.component_id = component_id
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        
        # Component metadata
        self.component_type = self._get_component_type()
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self.version = 1
        
        # Storage
        self._data: Any = None
        self._embedding: Optional[np.ndarray] = None
        
        # Component-specific settings
        self.immutable = config.get('immutable', False)
        self.max_size = config.get('max_size', 100)
        self.embedding_dim = config.get('embedding_dim', 128)
        
        logger.info(f"Initialized {self.component_type} component: {component_id}")
    
    @abstractmethod
    def _get_component_type(self) -> str:
        """
        Get the type of this component.
        
        Returns:
            Component type string
        """
        pass
    
    @abstractmethod
    def _generate_embedding(self, data: Any) -> np.ndarray:
        """
        Generate embedding from data.
        
        Args:
            data: Data to embed
        
        Returns:
            Embedding vector
        """
        pass
    
    def update(self, data: Any) -> bool:
        """
        Update the component with new data.
        
        Args:
            data: New data to store
        
        Returns:
            True if update successful
        """
        if self.immutable and self._data is not None:
            logger.warning(f"Component {self.component_id} is immutable")
            return False
        
        # Validate data
        if not self._validate_data(data):
            logger.warning(f"Invalid data for component {self.component_id}")
            return False
        
        # Store data
        self._data = data
        
        # Update embedding
        self._embedding = self._generate_embedding(data)
        
        # Update metadata
        self.updated_at = datetime.now().isoformat()
        self.version += 1
        
        return True
    
    def get_data(self) -> Any:
        """
        Get the stored data.
        
        Returns:
            Stored data
        """
        return self._data
    
    def get_embedding(self) -> Optional[np.ndarray]:
        """
        Get the component embedding.
        
        Returns:
            Embedding vector or None
        """
        return self._embedding
    
    def get_size(self) -> int:
        """
        Get the size of the component.
        
        Returns:
            Size in bytes (approximate)
        """
        if self._data is None:
            return 0
        
        # Approximate size based on data type
        if isinstance(self._data, dict):
            return len(json.dumps(self._data))
        elif isinstance(self._data, list):
            return len(self._data)
        elif isinstance(self._data, str):
            return len(self._data)
        else:
            return 1
    
    def clear(self) -> bool:
        """
        Clear the component data.
        
        Returns:
            True if clear successful
        """
        if self.immutable and self._data is not None:
            logger.warning(f"Component {self.component_id} is immutable")
            return False
        
        self._data = None
        self._embedding = None
        self.updated_at = datetime.now().isoformat()
        self.version += 1
        
        return True
    
    def _validate_data(self, data: Any) -> bool:
        """
        Validate data before storing.
        
        Args:
            data: Data to validate
        
        Returns:
            True if valid
        """
        # Basic validation - can be overridden
        return data is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert component to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'component_id': self.component_id,
            'component_type': self.component_type,
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'version': self.version,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'immutable': self.immutable,
            'max_size': self.max_size,
            'embedding_dim': self.embedding_dim,
            'data': self._data,
            'has_embedding': self._embedding is not None
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Load component from dictionary.
        
        Args:
            data: Dictionary representation
        """
        self.version = data.get('version', 1)
        self.created_at = data.get('created_at', datetime.now().isoformat())
        self.updated_at = data.get('updated_at', datetime.now().isoformat())
        self.immutable = data.get('immutable', False)
        self.max_size = data.get('max_size', 100)
        self.embedding_dim = data.get('embedding_dim', 128)
        
        self._data = data.get('data')
        
        # Regenerate embedding if data exists
        if self._data is not None:
            self._embedding = self._generate_embedding(self._data)
        else:
            self._embedding = None
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"{self.__class__.__name__}(id={self.component_id}, "
                f"type={self.component_type}, version={self.version})")


class IntrinsicMemory(MemoryComponent):
    """
    Intrinsic memory component for agent identity and core attributes.
    
    This memory stores immutable or slowly-changing information about
    the agent, such as identity, core preferences, and demographic data.
    """
    
    def __init__(self, 
                 component_id: str,
                 agent_id: str,
                 agent_type: str,
                 config: Dict[str, Any]):
        """
        Initialize IntrinsicMemory.
        
        Args:
            component_id: Unique identifier for the component
            agent_id: Agent identifier
            agent_type: Type of agent ('user' or 'item')
            config: Configuration dictionary
        """
        super().__init__(component_id, agent_id, agent_type, config)
        
        # Intrinsic-specific attributes
        self.is_immutable = config.get('immutable', True)
        self.identity_data: Dict[str, Any] = {}
        self.core_attributes: Dict[str, Any] = {}
        
        logger.info(f"Initialized IntrinsicMemory: {component_id}")
    
    def _get_component_type(self) -> str:
        """Get component type."""
        return 'intrinsic'
    
    def _generate_embedding(self, data: Any) -> np.ndarray:
        """
        Generate embedding from intrinsic data.
        
        Args:
            data: Intrinsic data
        
        Returns:
            Embedding vector
        """
        embedding = np.zeros(self.embedding_dim)
        
        if isinstance(data, dict):
            # Convert dictionary to embedding
            for key, value in data.items():
                if isinstance(value, str):
                    # Simple hash-based embedding
                    for char in value[:100]:
                        hash_val = hash(f"{key}_{char}") % self.embedding_dim
                        embedding[hash_val] += 1
                elif isinstance(value, (int, float)):
                    # Numeric values
                    idx = hash(key) % self.embedding_dim
                    embedding[idx] = value / 10.0  # Normalize
                elif isinstance(value, list):
                    # List of values
                    for item in value[:10]:
                        if isinstance(item, str):
                            hash_val = hash(f"{key}_{item}") % self.embedding_dim
                            embedding[hash_val] += 1
        
        elif isinstance(data, str):
            # Text embedding
            for char in data[:500]:
                hash_val = hash(char) % self.embedding_dim
                embedding[hash_val] += 1
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def update(self, data: Any) -> bool:
        """
        Update intrinsic memory.
        
        Args:
            data: New intrinsic data
        
        Returns:
            True if update successful
        """
        if self.is_immutable and self._data is not None:
            logger.warning(f"Intrinsic memory {self.component_id} is immutable")
            return False
        
        # Store identity data
        if isinstance(data, dict):
            self.identity_data = data
            self.core_attributes = data.get('core_attributes', {})
        
        return super().update(data)
    
    def get_identity(self) -> Dict[str, Any]:
        """
        Get agent identity.
        
        Returns:
            Identity dictionary
        """
        return self.identity_data
    
    def get_core_attribute(self, key: str, default: Any = None) -> Any:
        """
        Get a core attribute.
        
        Args:
            key: Attribute key
            default: Default value if not found
        
        Returns:
            Attribute value
        """
        return self.core_attributes.get(key, default)
    
    def set_core_attribute(self, key: str, value: Any) -> bool:
        """
        Set a core attribute.
        
        Args:
            key: Attribute key
            value: Attribute value
        
        Returns:
            True if successful
        """
        if self.is_immutable:
            logger.warning(f"Intrinsic memory {self.component_id} is immutable")
            return False
        
        self.core_attributes[key] = value
        self.identity_data['core_attributes'] = self.core_attributes
        
        # Update data
        return self.update(self.identity_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update({
            'identity_data': self.identity_data,
            'core_attributes': self.core_attributes,
            'is_immutable': self.is_immutable
        })
        return data
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load from dictionary."""
        super().from_dict(data)
        self.identity_data = data.get('identity_data', {})
        self.core_attributes = data.get('core_attributes', {})
        self.is_immutable = data.get('is_immutable', True)


class CollaborativeMemory(MemoryComponent):
    """
    Collaborative memory component for social/relational patterns.
    
    This memory stores collaborative information derived from interactions
    with other agents, including preferences, patterns, and relationships.
    """
    
    def __init__(self,
                 component_id: str,
                 agent_id: str,
                 agent_type: str,
                 config: Dict[str, Any]):
        """
        Initialize CollaborativeMemory.
        
        Args:
            component_id: Unique identifier for the component
            agent_id: Agent identifier
            agent_type: Type of agent ('user' or 'item')
            config: Configuration dictionary
        """
        super().__init__(component_id, agent_id, agent_type, config)
        
        # Collaborative-specific attributes
        self.propagation_threshold = config.get('propagation_threshold', 0.3)
        self.collaborative_patterns: Dict[str, Any] = {}
        self.relationship_scores: Dict[str, float] = {}
        self.neighbor_history: List[str] = []
        self.max_neighbors = config.get('max_neighbors', 50)
        
        # Aggregation settings
        self.aggregation_method = config.get('aggregation_method', 'weighted_average')
        
        logger.info(f"Initialized CollaborativeMemory: {component_id}")
    
    def _get_component_type(self) -> str:
        """Get component type."""
        return 'collaborative'
    
    def _generate_embedding(self, data: Any) -> np.ndarray:
        """
        Generate embedding from collaborative data.
        
        Args:
            data: Collaborative data
        
        Returns:
            Embedding vector
        """
        embedding = np.zeros(self.embedding_dim)
        
        if isinstance(data, dict):
            # Extract patterns and relationships
            patterns = data.get('patterns', {})
            relationships = data.get('relationships', {})
            
            # Encode patterns
            for pattern_key, pattern_value in patterns.items():
                idx = hash(pattern_key) % self.embedding_dim
                embedding[idx] = pattern_value if isinstance(pattern_value, (int, float)) else 0.5
            
            # Encode relationships
            for neighbor, score in relationships.items():
                idx = hash(neighbor) % self.embedding_dim
                embedding[idx] = score
        
        elif isinstance(data, list):
            # List of collaborative interactions
            for item in data[:20]:
                if isinstance(item, dict):
                    # Encode interaction patterns
                    for key, value in item.items():
                        idx = hash(f"{key}_{item.get('item_id', '')}") % self.embedding_dim
                        embedding[idx] += 0.1
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def update(self, data: Any) -> bool:
        """
        Update collaborative memory.
        
        Args:
            data: New collaborative data
        
        Returns:
            True if update successful
        """
        if isinstance(data, dict):
            # Update patterns
            if 'patterns' in data:
                self.collaborative_patterns.update(data['patterns'])
            
            # Update relationships
            if 'relationships' in data:
                self.relationship_scores.update(data['relationships'])
                
                # Update neighbor history
                for neighbor in data['relationships'].keys():
                    if neighbor not in self.neighbor_history:
                        self.neighbor_history.append(neighbor)
                    
                    # Limit history
                    if len(self.neighbor_history) > self.max_neighbors:
                        self.neighbor_history = self.neighbor_history[-self.max_neighbors:]
        
        return super().update(data)
    
    def propagate(self, data: Any) -> bool:
        """
        Propagate collaborative information.
        
        Args:
            data: Data to propagate
        
        Returns:
            True if propagation successful
        """
        if not isinstance(data, dict):
            return False
        
        # Extract propagation data
        rating = data.get('rating', 0)
        item_id = data.get('item_id')
        user_id = data.get('user_id')
        
        # Check if significant enough to propagate
        if abs(rating - 3.0) >= 1.0:  # Significant rating (high or low)
            # Update collaborative patterns
            pattern_key = f"{user_id}_{item_id}"
            self.collaborative_patterns[pattern_key] = rating
            
            # Update relationships
            self.relationship_scores[user_id] = rating / 5.0
            
            # Update data
            return self.update({
                'patterns': {pattern_key: rating},
                'relationships': {user_id: rating / 5.0}
            })
        
        return False
    
    def get_neighbors(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top collaborative neighbors.
        
        Args:
            top_k: Number of neighbors to return
        
        Returns:
            List of (neighbor_id, score) tuples
        """
        sorted_neighbors = sorted(
            self.relationship_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_neighbors[:top_k]
    
    def get_aggregated_patterns(self) -> Dict[str, float]:
        """
        Get aggregated collaborative patterns.
        
        Returns:
            Dictionary of aggregated patterns
        """
        if self.aggregation_method == 'weighted_average':
            return self._weighted_average_aggregation()
        elif self.aggregation_method == 'sum':
            return self._sum_aggregation()
        else:
            return self.collaborative_patterns
    
    def _weighted_average_aggregation(self) -> Dict[str, float]:
        """
        Aggregate patterns using weighted average.
        
        Returns:
            Aggregated patterns
        """
        aggregated = {}
        
        for pattern, value in self.collaborative_patterns.items():
            # Weight by relationship scores if available
            if isinstance(value, (int, float)):
                aggregated[pattern] = value
        
        return aggregated
    
    def _sum_aggregation(self) -> Dict[str, float]:
        """Aggregate patterns using sum."""
        return self.collaborative_patterns.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update({
            'collaborative_patterns': self.collaborative_patterns,
            'relationship_scores': self.relationship_scores,
            'neighbor_history': self.neighbor_history,
            'propagation_threshold': self.propagation_threshold,
            'max_neighbors': self.max_neighbors,
            'aggregation_method': self.aggregation_method
        })
        return data
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load from dictionary."""
        super().from_dict(data)
        self.collaborative_patterns = data.get('collaborative_patterns', {})
        self.relationship_scores = data.get('relationship_scores', {})
        self.neighbor_history = data.get('neighbor_history', [])
        self.propagation_threshold = data.get('propagation_threshold', 0.3)
        self.max_neighbors = data.get('max_neighbors', 50)
        self.aggregation_method = data.get('aggregation_method', 'weighted_average')


class InteractionMemory(MemoryComponent):
    """
    Interaction memory component for temporal interaction history.
    
    This memory stores recent interactions with timestamps,
    maintaining a buffer of the most recent interactions.
    """
    
    def __init__(self,
                 component_id: str,
                 agent_id: str,
                 agent_type: str,
                 config: Dict[str, Any]):
        """
        Initialize InteractionMemory.
        
        Args:
            component_id: Unique identifier for the component
            agent_id: Agent identifier
            agent_type: Type of agent ('user' or 'item')
            config: Configuration dictionary
        """
        super().__init__(component_id, agent_id, agent_type, config)
        
        # Interaction-specific attributes
        self.buffer_size = config.get('buffer_size', 10)
        self.interaction_buffer: deque = deque(maxlen=self.buffer_size)
        self.interaction_history: List[Dict] = []
        self.interaction_stats: Dict[str, Any] = {}
        
        # Temporal tracking
        self.first_interaction_time: Optional[str] = None
        self.last_interaction_time: Optional[str] = None
        self.total_interactions = 0
        
        logger.info(f"Initialized InteractionMemory: {component_id}")
    
    def _get_component_type(self) -> str:
        """Get component type."""
        return 'interaction'
    
    def _generate_embedding(self, data: Any) -> np.ndarray:
        """
        Generate embedding from interaction data.
        
        Args:
            data: Interaction data
        
        Returns:
            Embedding vector
        """
        embedding = np.zeros(self.embedding_dim)
        
        if isinstance(data, dict):
            # Encode interaction features
            interaction_id = data.get('interaction_id')
            rating = data.get('rating', 0)
            timestamp = data.get('timestamp', '')
            
            idx = hash(f"rating_{rating}") % self.embedding_dim
            embedding[idx] = rating / 5.0
            
            if interaction_id:
                idx = hash(interaction_id) % self.embedding_dim
                embedding[idx] = 1.0
        
        elif isinstance(data, list):
            # Encode sequence of interactions
            for i, interaction in enumerate(data[:10]):
                if isinstance(interaction, dict):
                    rating = interaction.get('rating', 0)
                    idx = (hash(f"interaction_{i}") % self.embedding_dim)
                    embedding[idx] = rating / 5.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def add_interaction(self, interaction_data: Dict[str, Any]) -> bool:
        """
        Add an interaction to memory.
        
        Args:
            interaction_data: Interaction data
        
        Returns:
            True if successful
        """
        # Add timestamp if not present
        if 'timestamp' not in interaction_data:
            interaction_data['timestamp'] = datetime.now().isoformat()
        
        # Add interaction to buffer
        self.interaction_buffer.append(interaction_data)
        
        # Add to history
        self.interaction_history.append(interaction_data)
        
        # Update statistics
        self.total_interactions += 1
        
        if self.first_interaction_time is None:
            self.first_interaction_time = interaction_data['timestamp']
        
        self.last_interaction_time = interaction_data['timestamp']
        
        # Update data (embedding will be regenerated)
        return self.update(list(self.interaction_buffer))
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get interaction history.
        
        Args:
            limit: Maximum number of interactions to return
        
        Returns:
            List of interactions
        """
        if limit is None or limit >= len(self.interaction_history):
            return self.interaction_history.copy()
        else:
            return self.interaction_history[-limit:]
    
    def get_recent_interactions(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get recent interactions from buffer.
        
        Args:
            limit: Maximum number of interactions to return
        
        Returns:
            List of recent interactions
        """
        if limit is None or limit >= len(self.interaction_buffer):
            return list(self.interaction_buffer)
        else:
            return list(self.interaction_buffer)[-limit:]
    
    def get_interaction_stats(self) -> Dict[str, Any]:
        """
        Get interaction statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self.interaction_history:
            return {
                'total_interactions': 0,
                'avg_rating': 0,
                'min_rating': 0,
                'max_rating': 0,
                'std_rating': 0
            }
        
        ratings = [i.get('rating', 0) for i in self.interaction_history]
        
        return {
            'total_interactions': self.total_interactions,
            'avg_rating': np.mean(ratings) if ratings else 0,
            'min_rating': min(ratings) if ratings else 0,
            'max_rating': max(ratings) if ratings else 0,
            'std_rating': np.std(ratings) if ratings else 0,
            'unique_items': len(set(i.get('item_id', '') for i in self.interaction_history)),
            'first_interaction': self.first_interaction_time,
            'last_interaction': self.last_interaction_time
        }
    
    def propagate(self, data: Any) -> bool:
        """
        Propagate interaction data.
        
        Args:
            data: Data to propagate
        
        Returns:
            True if propagation successful
        """
        if isinstance(data, dict):
            # Add as interaction
            return self.add_interaction(data)
        
        return False
    
    def clear_buffer(self) -> None:
        """Clear the interaction buffer."""
        self.interaction_buffer.clear()
        self.update([])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update({
            'buffer_size': self.buffer_size,
            'interaction_buffer': list(self.interaction_buffer),
            'interaction_history': self.interaction_history,
            'interaction_stats': self.interaction_stats,
            'first_interaction_time': self.first_interaction_time,
            'last_interaction_time': self.last_interaction_time,
            'total_interactions': self.total_interactions
        })
        return data
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load from dictionary."""
        super().from_dict(data)
        self.buffer_size = data.get('buffer_size', 10)
        self.interaction_buffer = deque(
            data.get('interaction_buffer', []),
            maxlen=self.buffer_size
        )
        self.interaction_history = data.get('interaction_history', [])
        self.interaction_stats = data.get('interaction_stats', {})
        self.first_interaction_time = data.get('first_interaction_time')
        self.last_interaction_time = data.get('last_interaction_time')
        self.total_interactions = data.get('total_interactions', len(self.interaction_history))


class MemoryComponentFactory:
    """
    Factory class for creating memory components.
    """
    
    _component_registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, component_type: str, component_class: type) -> None:
        """
        Register a component class.
        
        Args:
            component_type: Type identifier
            component_class: Component class
        """
        cls._component_registry[component_type] = component_class
    
    @classmethod
    def create_component(cls,
                        component_type: str,
                        component_id: str,
                        agent_id: str,
                        agent_type: str,
                        config: Dict[str, Any]) -> MemoryComponent:
        """
        Create a memory component.
        
        Args:
            component_type: Type of component ('intrinsic', 'collaborative', 'interaction')
            component_id: Component identifier
            agent_id: Agent identifier
            agent_type: Type of agent
            config: Configuration dictionary
        
        Returns:
            MemoryComponent instance
        
        Raises:
            ValueError: If component_type is not registered
        """
        if component_type not in cls._component_registry:
            raise ValueError(f"Unknown component type: {component_type}")
        
        component_class = cls._component_registry[component_type]
        return component_class(component_id, agent_id, agent_type, config)
    
    @classmethod
    def get_component_types(cls) -> List[str]:
        """
        Get registered component types.
        
        Returns:
            List of component type names
        """
        return list(cls._component_registry.keys())


# Register default components
MemoryComponentFactory.register('intrinsic', IntrinsicMemory)
MemoryComponentFactory.register('collaborative', CollaborativeMemory)
MemoryComponentFactory.register('interaction', InteractionMemory)


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'immutable': False,
        'max_size': 50,
        'embedding_dim': 128,
        'buffer_size': 10,
        'propagation_threshold': 0.3,
        'max_neighbors': 50,
        'aggregation_method': 'weighted_average'
    }
    
    # Create intrinsic memory
    intrinsic = IntrinsicMemory('mem_001_intrinsic', 'agent_001', 'user', config)
    intrinsic.update({
        'name': 'Test User',
        'preferences': ['books', 'movies', 'music'],
        'age': 25,
        'location': 'NYC'
    })
    print(f"Intrinsic: {intrinsic}")
    print(f"Identity: {intrinsic.get_identity()}")
    
    # Create collaborative memory
    collaborative = CollaborativeMemory('mem_001_collab', 'agent_001', 'user', config)
    collaborative.update({
        'patterns': {'user_002_item_001': 5, 'user_003_item_002': 4},
        'relationships': {'user_002': 0.8, 'user_003': 0.6}
    })
    print(f"Collaborative: {collaborative}")
    print(f"Neighbors: {collaborative.get_neighbors(2)}")
    
    # Create interaction memory
    interaction = InteractionMemory('mem_001_inter', 'agent_001', 'user', config)
    interaction.add_interaction({
        'item_id': 'item_001',
        'rating': 5,
        'timestamp': datetime.now().isoformat()
    })
    interaction.add_interaction({
        'item_id': 'item_002',
        'rating': 3,
        'timestamp': datetime.now().isoformat()
    })
    print(f"Interaction: {interaction}")
    print(f"Stats: {interaction.get_interaction_stats()}")
    
    # Test factory
    factory = MemoryComponentFactory()
    print(f"Registered types: {factory.get_component_types()}")
    
    # Test propagation
    interaction.propagate({
        'item_id': 'item_003',
        'rating': 4,
        'user_id': 'user_004'
    })
    print(f"After propagation: {len(interaction.get_history())} interactions")