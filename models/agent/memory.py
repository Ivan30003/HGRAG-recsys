"""
memory.py - Memory management for agents in H-GRAGrecsys

This module provides the hierarchical memory system for agents,
including the main memory container and memory component management.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from models.agent.memory_components import (
    MemoryComponent,
    IntrinsicMemory,
    CollaborativeMemory,
    InteractionMemory,
    MemoryComponentFactory
)

# Configure logging
logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Enumeration of memory types."""
    INTRINSIC = 'intrinsic'
    COLLABORATIVE = 'collaborative'
    INTERACTION = 'interaction'


@dataclass
class MemoryUpdateResult:
    """Container for memory update results."""
    success: bool
    component_type: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'component_type': self.component_type,
            'message': self.message,
            'timestamp': self.timestamp,
            'data': self.data
        }


class AgentMemory:
    """
    Base memory management class for agents.
    
    This class provides the foundational memory management capabilities,
    including storing, retrieving, and updating memory components.
    """
    
    def __init__(self, agent_id: str, agent_type: str, config: Dict[str, Any]):
        """
        Initialize AgentMemory.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent ('user' or 'item')
            config: Configuration dictionary
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        
        # Memory components
        self.components: Dict[str, MemoryComponent] = {}
        
        # Memory metadata
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        
        # Version tracking
        self.version = 1
        
        # Initialize memory components
        self._initialize_components()
        
        logger.info(f"Initialized memory for {agent_type} agent: {agent_id}")
    
    def _initialize_components(self) -> None:
        """Initialize memory components based on configuration."""
        # Get memory configuration
        memory_config = self.config.get('memory', {})
        
        # Initialize intrinsic memory
        intrinsic_config = memory_config.get('intrinsic', {})
        intrinsic = IntrinsicMemory(
            component_id=f"{self.agent_id}_intrinsic",
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            config=intrinsic_config
        )
        self.components[MemoryType.INTRINSIC.value] = intrinsic
        
        # Initialize collaborative memory
        collaborative_config = memory_config.get('collaborative', {})
        collaborative = CollaborativeMemory(
            component_id=f"{self.agent_id}_collaborative",
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            config=collaborative_config
        )
        self.components[MemoryType.COLLABORATIVE.value] = collaborative
        
        # Initialize interaction memory
        interaction_config = memory_config.get('interaction', {})
        interaction = InteractionMemory(
            component_id=f"{self.agent_id}_interaction",
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            config=interaction_config
        )
        self.components[MemoryType.INTERACTION.value] = interaction
        
        logger.debug(f"Initialized all memory components for {self.agent_id}")
    
    def get_component(self, component_type: str) -> Optional[MemoryComponent]:
        """
        Get a memory component.
        
        Args:
            component_type: Type of memory component
        
        Returns:
            MemoryComponent or None
        """
        return self.components.get(component_type)
    
    def update_component(self, component_type: str, data: Any) -> bool:
        """
        Update a memory component.
        
        Args:
            component_type: Type of memory component
            data: Data to store
        
        Returns:
            True if update successful
        """
        component = self.get_component(component_type)
        if not component:
            logger.warning(f"Component {component_type} not found for {self.agent_id}")
            return False
        
        success = component.update(data)
        if success:
            self.updated_at = datetime.now().isoformat()
            self.version += 1
        
        return success
    
    def get_component_data(self, component_type: str) -> Optional[Any]:
        """
        Get data from a memory component.
        
        Args:
            component_type: Type of memory component
        
        Returns:
            Component data or None
        """
        component = self.get_component(component_type)
        if not component:
            return None
        
        return component.get_data()
    
    def get_component_embedding(self, component_type: str) -> Optional[np.ndarray]:
        """
        Get embedding from a memory component.
        
        Args:
            component_type: Type of memory component
        
        Returns:
            Embedding vector or None
        """
        component = self.get_component(component_type)
        if not component:
            return None
        
        if hasattr(component, 'get_embedding'):
            return component.get_embedding()
        
        return None
    
    def add_interaction(self, interaction_data: Dict[str, Any]) -> bool:
        """
        Add an interaction to the interaction memory.
        
        Args:
            interaction_data: Interaction data
        
        Returns:
            True if successful
        """
        interaction_component = self.get_component(MemoryType.INTERACTION.value)
        if not interaction_component:
            return False
        
        success = interaction_component.add_interaction(interaction_data)
        if success:
            self.updated_at = datetime.now().isoformat()
            self.version += 1
        
        return success
    
    def get_interaction_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get interaction history.
        
        Args:
            limit: Maximum number of interactions to return
        
        Returns:
            List of interactions
        """
        interaction_component = self.get_component(MemoryType.INTERACTION.value)
        if not interaction_component:
            return []
        
        if hasattr(interaction_component, 'get_history'):
            return interaction_component.get_history(limit)
        
        return []
    
    def get_intrinsic_data(self) -> Optional[Any]:
        """Get intrinsic memory data."""
        return self.get_component_data(MemoryType.INTRINSIC.value)
    
    def get_collaborative_data(self) -> Optional[Any]:
        """Get collaborative memory data."""
        return self.get_component_data(MemoryType.COLLABORATIVE.value)
    
    def get_interaction_data(self) -> Optional[Any]:
        """Get interaction memory data."""
        return self.get_component_data(MemoryType.INTERACTION.value)
    
    def get_size(self) -> int:
        """
        Get total memory size.
        
        Returns:
            Total size of all memory components
        """
        total_size = 0
        for component in self.components.values():
            total_size += component.get_size()
        return total_size
    
    def get_component_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all memory components.
        
        Returns:
            Dictionary of component statistics
        """
        stats = {}
        for component_type, component in self.components.items():
            stats[component_type] = {
                'size': component.get_size(),
                'type': component.component_type,
                'created_at': component.created_at,
                'updated_at': component.updated_at,
                'version': component.version
            }
        return stats
    
    def clear_component(self, component_type: str) -> bool:
        """
        Clear a memory component.
        
        Args:
            component_type: Type of memory component
        
        Returns:
            True if successful
        """
        component = self.get_component(component_type)
        if not component:
            return False
        
        success = component.clear()
        if success:
            self.updated_at = datetime.now().isoformat()
            self.version += 1
        
        return success
    
    def clear_all(self) -> None:
        """Clear all memory components."""
        for component_type in list(self.components.keys()):
            self.clear_component(component_type)
        
        self.updated_at = datetime.now().isoformat()
        self.version += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert memory to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'version': self.version,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'components': {
                component_type: component.to_dict()
                for component_type, component in self.components.items()
            },
            'config': self.config
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Load memory from dictionary.
        
        Args:
            data: Dictionary representation
        """
        self.version = data.get('version', 1)
        self.created_at = data.get('created_at', datetime.now().isoformat())
        self.updated_at = data.get('updated_at', datetime.now().isoformat())
        
        # Restore components
        components_data = data.get('components', {})
        for component_type, component_data in components_data.items():
            component = self.get_component(component_type)
            if component:
                component.from_dict(component_data)
        
        logger.info(f"Loaded memory for {self.agent_id} from dictionary")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"AgentMemory(agent_id={self.agent_id}, components={len(self.components)}, version={self.version})"


class HierarchicalMemory(AgentMemory):
    """
    Hierarchical memory management for agents.
    
    This class extends AgentMemory with hierarchical structure and
    specialized memory operations for intrinsic, collaborative, and
    interaction memories.
    """
    
    def __init__(self, agent_id: str, agent_type: str, config: Dict[str, Any]):
        """
        Initialize HierarchicalMemory.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent ('user' or 'item')
            config: Configuration dictionary
        """
        super().__init__(agent_id, agent_type, config)
        
        # Hierarchical-specific attributes
        self.memory_hierarchy = {
            MemoryType.INTRINSIC.value: 1,      # Level 1: Base/Immutable
            MemoryType.COLLABORATIVE.value: 2,   # Level 2: Collaborative
            MemoryType.INTERACTION.value: 3      # Level 3: Dynamic/Interactive
        }
        
        # Propagation settings
        self.propagation_enabled = config.get('propagation_enabled', True)
        self.propagation_threshold = config.get('propagation_threshold', 0.3)
        
        # Memory consistency tracking
        self.consistency_scores: Dict[str, float] = {}
        
        logger.info(f"Initialized HierarchicalMemory for {agent_type} agent: {agent_id}")
    
    def update_component(self, component_type: str, data: Any) -> bool:
        """
        Update a memory component with hierarchical propagation.
        
        Args:
            component_type: Type of memory component
            data: Data to store
        
        Returns:
            True if update successful
        """
        # Update the component
        success = super().update_component(component_type, data)
        
        if success and self.propagation_enabled:
            # Propagate updates to related components
            self._propagate_update(component_type, data)
        
        return success
    
    def _propagate_update(self, source_component: str, data: Any) -> None:
        """
        Propagate updates to related memory components.
        
        Args:
            source_component: Component that was updated
            data: Update data
        """
        # Get the hierarchy level of the source
        source_level = self.memory_hierarchy.get(source_component, 0)
        
        # Determine which components to propagate to
        if source_component == MemoryType.INTRINSIC.value:
            # Intrinsic updates may affect collaborative memory
            self._propagate_to_collaborative(data)
        elif source_component == MemoryType.COLLABORATIVE.value:
            # Collaborative updates may affect interaction memory
            self._propagate_to_interaction(data)
        elif source_component == MemoryType.INTERACTION.value:
            # Interaction updates may affect collaborative memory
            if self._should_propagate_interaction(data):
                self._propagate_to_collaborative(data)
    
    def _propagate_to_collaborative(self, data: Any) -> None:
        """
        Propagate data to collaborative memory.
        
        Args:
            data: Data to propagate
        """
        collaborative = self.get_component(MemoryType.COLLABORATIVE.value)
        if collaborative and hasattr(collaborative, 'propagate'):
            collaborative.propagate(data)
    
    def _propagate_to_interaction(self, data: Any) -> None:
        """
        Propagate data to interaction memory.
        
        Args:
            data: Data to propagate
        """
        interaction = self.get_component(MemoryType.INTERACTION.value)
        if interaction and hasattr(interaction, 'propagate'):
            interaction.propagate(data)
    
    def _should_propagate_interaction(self, data: Any) -> bool:
        """
        Determine if an interaction should be propagated.
        
        Args:
            data: Interaction data
        
        Returns:
            True if should propagate
        """
        # Check if interaction is significant enough
        if isinstance(data, dict):
            rating = data.get('rating', 0)
            if rating >= 4.0:  # High ratings are more significant
                return True
            if rating <= 2.0:  # Low ratings are also significant
                return True
        
        # Check confidence score
        if hasattr(data, 'confidence'):
            if data.confidence >= self.propagation_threshold:
                return True
        
        return False
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the hierarchical memory.
        
        Returns:
            Memory summary dictionary
        """
        summary = {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'total_size': self.get_size(),
            'components': {},
            'hierarchy': self.memory_hierarchy
        }
        
        for component_type, component in self.components.items():
            summary['components'][component_type] = {
                'size': component.get_size(),
                'level': self.memory_hierarchy.get(component_type, 0),
                'type': component.component_type,
                'created_at': component.created_at,
                'updated_at': component.updated_at
            }
        
        return summary
    
    def get_memory_consistency(self) -> float:
        """
        Get overall memory consistency score.
        
        Returns:
            Consistency score (0-1)
        """
        if not self.consistency_scores:
            return 1.0
        
        return np.mean(list(self.consistency_scores.values()))
    
    def check_consistency(self) -> Dict[str, float]:
        """
        Check consistency between memory components.
        
        Returns:
            Dictionary of consistency scores
        """
        consistency_scores = {}
        
        # Check intrinsic vs collaborative consistency
        intrinsic = self.get_component_data(MemoryType.INTRINSIC.value)
        collaborative = self.get_component_data(MemoryType.COLLABORATIVE.value)
        
        if intrinsic and collaborative:
            intrinsic_embedding = self.get_component_embedding(MemoryType.INTRINSIC.value)
            collaborative_embedding = self.get_component_embedding(MemoryType.COLLABORATIVE.value)
            
            if intrinsic_embedding is not None and collaborative_embedding is not None:
                # Calculate similarity
                norm1 = np.linalg.norm(intrinsic_embedding)
                norm2 = np.linalg.norm(collaborative_embedding)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = np.dot(intrinsic_embedding, collaborative_embedding) / (norm1 * norm2)
                    consistency_scores['intrinsic_collaborative'] = float(max(0.0, min(1.0, similarity)))
        
        # Check collaborative vs interaction consistency
        interaction = self.get_component_data(MemoryType.INTERACTION.value)
        
        if collaborative and interaction:
            collaborative_embedding = self.get_component_embedding(MemoryType.COLLABORATIVE.value)
            interaction_embedding = self.get_component_embedding(MemoryType.INTERACTION.value)
            
            if collaborative_embedding is not None and interaction_embedding is not None:
                norm1 = np.linalg.norm(collaborative_embedding)
                norm2 = np.linalg.norm(interaction_embedding)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = np.dot(collaborative_embedding, interaction_embedding) / (norm1 * norm2)
                    consistency_scores['collaborative_interaction'] = float(max(0.0, min(1.0, similarity)))
        
        # Store consistency scores
        self.consistency_scores.update(consistency_scores)
        
        return consistency_scores
    
    def maintain_consistency(self, threshold: float = 0.5) -> bool:
        """
        Maintain memory consistency by aligning components.
        
        Args:
            threshold: Minimum consistency threshold
        
        Returns:
            True if consistency was maintained
        """
        consistency_scores = self.check_consistency()
        
        for component_pair, score in consistency_scores.items():
            if score < threshold:
                logger.warning(f"Low consistency ({score:.2f}) for {component_pair} in {self.agent_id}")
                # Attempt to align components
                self._align_components(component_pair)
        
        return True
    
    def _align_components(self, component_pair: str) -> None:
        """
        Align two memory components.
        
        Args:
            component_pair: Pair of components to align
        """
        if component_pair == 'intrinsic_collaborative':
            # Align intrinsic and collaborative memories
            intrinsic = self.get_component(MemoryType.INTRINSIC.value)
            collaborative = self.get_component(MemoryType.COLLABORATIVE.value)
            
            if intrinsic and collaborative:
                # Get data from intrinsic to update collaborative
                intrinsic_data = intrinsic.get_data()
                if intrinsic_data:
                    collaborative.update(intrinsic_data)
        
        elif component_pair == 'collaborative_interaction':
            # Align collaborative and interaction memories
            collaborative = self.get_component(MemoryType.COLLABORATIVE.value)
            interaction = self.get_component(MemoryType.INTERACTION.value)
            
            if collaborative and interaction:
                # Get data from collaborative to update interaction
                collaborative_data = collaborative.get_data()
                if collaborative_data:
                    interaction.update(collaborative_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert hierarchical memory to dictionary.
        
        Returns:
            Dictionary representation
        """
        data = super().to_dict()
        data.update({
            'memory_hierarchy': self.memory_hierarchy,
            'propagation_enabled': self.propagation_enabled,
            'propagation_threshold': self.propagation_threshold,
            'consistency_scores': self.consistency_scores
        })
        return data
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Load hierarchical memory from dictionary.
        
        Args:
            data: Dictionary representation
        """
        super().from_dict(data)
        
        self.memory_hierarchy = data.get('memory_hierarchy', {
            MemoryType.INTRINSIC.value: 1,
            MemoryType.COLLABORATIVE.value: 2,
            MemoryType.INTERACTION.value: 3
        })
        self.propagation_enabled = data.get('propagation_enabled', True)
        self.propagation_threshold = data.get('propagation_threshold', 0.3)
        self.consistency_scores = data.get('consistency_scores', {})
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"HierarchicalMemory(agent_id={self.agent_id}, "
                f"components={len(self.components)}, "
                f"version={self.version}, "
                f"consistency={self.get_memory_consistency():.3f})")


# Factory function for creating memory instances
def create_memory(agent_id: str, 
                 agent_type: str, 
                 config: Dict[str, Any],
                 hierarchical: bool = True) -> AgentMemory:
    """
    Create a memory instance for an agent.
    
    Args:
        agent_id: Agent identifier
        agent_type: Type of agent ('user' or 'item')
        config: Configuration dictionary
        hierarchical: Whether to use hierarchical memory
    
    Returns:
        Memory instance
    """
    if hierarchical:
        return HierarchicalMemory(agent_id, agent_type, config)
    else:
        return AgentMemory(agent_id, agent_type, config)


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'memory': {
            'intrinsic': {
                'immutable': True,
                'max_size': 5,
                'embedding_dim': 128
            },
            'collaborative': {
                'immutable': False,
                'max_size': 20,
                'propagation_threshold': 0.3,
                'embedding_dim': 128
            },
            'interaction': {
                'immutable': False,
                'max_size': 15,
                'buffer_size': 10,
                'embedding_dim': 128
            }
        },
        'propagation_enabled': True,
        'propagation_threshold': 0.3
    }
    
    # Create hierarchical memory
    memory = HierarchicalMemory('agent_001', 'user', config)
    print(f"Created: {memory}")
    
    # Test memory operations
    print(f"Memory size: {memory.get_size()}")
    
    # Update intrinsic memory
    intrinsic_data = {
        'name': 'Test User',
        'preferences': ['books', 'movies'],
        'age': 25
    }
    memory.update_component(MemoryType.INTRINSIC.value, intrinsic_data)
    
    # Add interaction
    memory.add_interaction({
        'item_id': 'item_001',
        'rating': 5,
        'timestamp': datetime.now().isoformat()
    })
    
    # Update collaborative memory
    collaborative_data = {
        'neighbors': ['user_002', 'user_003'],
        'similarity_scores': {'user_002': 0.8, 'user_003': 0.6}
    }
    memory.update_component(MemoryType.COLLABORATIVE.value, collaborative_data)
    
    # Check consistency
    print(f"Memory consistency: {memory.get_memory_consistency():.3f}")
    print(f"Consistency scores: {memory.check_consistency()}")
    
    # Get summary
    print(f"Memory summary: {memory.get_memory_summary()}")