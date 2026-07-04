"""
base_agent.py - Base agent implementation for H-GRAGrecsys

This module provides the foundational agent classes and interfaces for
both user and item agents in the recommendation system.
"""

import abc
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import torch
import logging
from pathlib import Path

from models.agent.memory import HierarchicalMemory, MemoryType
from models.agent.memory_components import IntrinsicMemory, CollaborativeMemory, InteractionMemory

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Container for agent state information."""
    agent_id: str
    agent_type: str  # 'user' or 'item'
    created_at: str
    updated_at: str
    interaction_count: int
    memory_size: int
    version: int = 1
    
    # Optional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'interaction_count': self.interaction_count,
            'memory_size': self.memory_size,
            'version': self.version,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentState':
        """Create from dictionary."""
        return cls(
            agent_id=data['agent_id'],
            agent_type=data['agent_type'],
            created_at=data['created_at'],
            updated_at=data['updated_at'],
            interaction_count=data['interaction_count'],
            memory_size=data['memory_size'],
            version=data.get('version', 1),
            metadata=data.get('metadata', {})
        )


class BaseAgent(abc.ABC):
    """
    Abstract base class for all agents.
    
    This class defines the core interface for agents in the system,
    including memory management, state tracking, and basic operations.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initialize BaseAgent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Configuration dictionary
        """
        self.agent_id = agent_id
        self.config = config
        
        # Determine agent type from subclass
        self.agent_type = self._determine_agent_type()
        
        # Initialize memory
        self.memory = self._initialize_memory()
        
        # State tracking
        self.state = AgentState(
            agent_id=agent_id,
            agent_type=self.agent_type,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            interaction_count=0,
            memory_size=self.memory.get_size() if self.memory else 0
        )
        
        # Version tracking
        self.version = 1
        
        # Embedding cache
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        # Metadata storage
        self.metadata: Dict[str, Any] = {}
        
        logger.info(f"Initialized {self.agent_type} agent: {agent_id}")
    
    def _determine_agent_type(self) -> str:
        """
        Determine agent type from class name.
        
        Returns:
            Agent type string
        """
        class_name = self.__class__.__name__.lower()
        if 'user' in class_name:
            return 'user'
        elif 'item' in class_name:
            return 'item'
        else:
            return 'base'
    
    def _initialize_memory(self) -> HierarchicalMemory:
        """
        Initialize hierarchical memory for the agent.
        
        Returns:
            HierarchicalMemory instance
        """
        memory_config = self.config.get('memory', {})
        return HierarchicalMemory(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            config=memory_config
        )
    
    @abc.abstractmethod
    def get_embedding(self, component_type: Optional[str] = None) -> np.ndarray:
        """
        Get embedding representation of the agent.
        
        Args:
            component_type: Optional specific memory component
        
        Returns:
            Embedding vector
        """
        pass
    
    @abc.abstractmethod
    def get_text_representation(self) -> str:
        """
        Get text representation of the agent.
        
        Returns:
            Text representation
        """
        pass
    
    def update_memory(self, component_type: str, data: Any) -> bool:
        """
        Update a memory component.
        
        Args:
            component_type: Type of memory component
            data: Data to store
        
        Returns:
            True if update successful
        """
        if not self.memory:
            logger.warning(f"Memory not initialized for agent {self.agent_id}")
            return False
        
        success = self.memory.update_component(component_type, data)
        if success:
            self.state.updated_at = datetime.now().isoformat()
            self.state.memory_size = self.memory.get_size()
            self.version += 1
        
        return success
    
    def get_memory_component(self, component_type: str) -> Optional[Any]:
        """
        Get a memory component.
        
        Args:
            component_type: Type of memory component
        
        Returns:
            Memory component data
        """
        if not self.memory:
            return None
        
        return self.memory.get_component(component_type)
    
    def get_intrinsic_memory(self) -> Optional[IntrinsicMemory]:
        """Get intrinsic memory component."""
        return self.get_memory_component(MemoryType.INTRINSIC.value)
    
    def get_collaborative_memory(self) -> Optional[CollaborativeMemory]:
        """Get collaborative memory component."""
        return self.get_memory_component(MemoryType.COLLABORATIVE.value)
    
    def get_interaction_memory(self) -> Optional[InteractionMemory]:
        """Get interaction memory component."""
        return self.get_memory_component(MemoryType.INTERACTION.value)
    
    def add_interaction(self, interaction_data: Dict[str, Any]) -> bool:
        """
        Add an interaction to the agent's memory.
        
        Args:
            interaction_data: Interaction data
        
        Returns:
            True if successful
        """
        if not self.memory:
            return False
        
        success = self.memory.add_interaction(interaction_data)
        if success:
            self.state.interaction_count += 1
            self.state.updated_at = datetime.now().isoformat()
            self.version += 1
        
        return success
    
    def get_interaction_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get interaction history.
        
        Args:
            limit: Maximum number of interactions to return
        
        Returns:
            List of interaction data
        """
        if not self.memory:
            return []
        
        return self.memory.get_interaction_history(limit)
    
    def calculate_similarity(self, other_agent: 'BaseAgent') -> float:
        """
        Calculate similarity to another agent.
        
        Args:
            other_agent: Other agent instance
        
        Returns:
            Similarity score (0-1)
        """
        emb1 = self.get_embedding()
        emb2 = other_agent.get_embedding()
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Calculate cosine similarity
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
        
        # Ensure in [0, 1] range
        return float(max(0.0, min(1.0, similarity)))
    
    def get_similarity_to_component(self, 
                                   other_agent: 'BaseAgent',
                                   component_type: str) -> float:
        """
        Calculate similarity to another agent's specific memory component.
        
        Args:
            other_agent: Other agent instance
            component_type: Memory component type
        
        Returns:
            Similarity score (0-1)
        """
        emb1 = self.get_embedding(component_type)
        emb2 = other_agent.get_embedding(component_type)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
        return float(max(0.0, min(1.0, similarity)))
    
    def get_state(self) -> AgentState:
        """
        Get current agent state.
        
        Returns:
            AgentState instance
        """
        return self.state
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set metadata for the agent.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
        self.state.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata for the agent.
        
        Args:
            key: Metadata key
            default: Default value if key not found
        
        Returns:
            Metadata value
        """
        return self.metadata.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert agent to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'version': self.version,
            'state': self.state.to_dict(),
            'memory': self.memory.to_dict() if self.memory else {},
            'metadata': self.metadata,
            'config': self.config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseAgent':
        """
        Create agent from dictionary.
        
        Args:
            data: Dictionary representation
        
        Returns:
            Agent instance
        """
        agent_id = data['agent_id']
        config = data.get('config', {})
        
        # Determine class based on type
        agent_type = data.get('agent_type', 'base')
        if agent_type == 'user':
            from models.agent.user_agent import UserAgent
            agent = UserAgent(agent_id, config)
        elif agent_type == 'item':
            from models.agent.item_agent import ItemAgent
            agent = ItemAgent(agent_id, config)
        else:
            agent = cls(agent_id, config)
        
        # Restore state
        if 'state' in data:
            agent.state = AgentState.from_dict(data['state'])
        
        # Restore memory
        if 'memory' in data and agent.memory:
            agent.memory.from_dict(data['memory'])
        
        # Restore version
        agent.version = data.get('version', 1)
        
        # Restore metadata
        if 'metadata' in data:
            agent.metadata = data['metadata']
        
        return agent
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save agent to file.
        
        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        # Save as JSON for readability
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=self._json_serializer)
        
        logger.info(f"Saved agent {self.agent_id} to {filepath}")
    
    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer for numpy arrays."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    def load(self, filepath: Union[str, Path]) -> None:
        """
        Load agent from file.
        
        Args:
            filepath: Path to load file
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct agent from data
        reconstructed = self.from_dict(data)
        
        # Copy attributes
        self.agent_id = reconstructed.agent_id
        self.agent_type = reconstructed.agent_type
        self.version = reconstructed.version
        self.state = reconstructed.state
        self.memory = reconstructed.memory
        self.metadata = reconstructed.metadata
        
        logger.info(f"Loaded agent {self.agent_id} from {filepath}")
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(id={self.agent_id}, type={self.agent_type}, version={self.version})"
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"Agent {self.agent_id} ({self.agent_type}) - v{self.version}"


class AgentFactory:
    """
    Factory class for creating agents.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AgentFactory.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._agent_registry: Dict[str, type] = {}
        self._register_defaults()
    
    def _register_defaults(self) -> None:
        """Register default agent types."""
        from models.agent.user_agent import UserAgent
        from models.agent.item_agent import ItemAgent
        
        self.register('user', UserAgent)
        self.register('item', ItemAgent)
    
    def register(self, agent_type: str, agent_class: type) -> None:
        """
        Register an agent type.
        
        Args:
            agent_type: Type identifier
            agent_class: Agent class
        """
        self._agent_registry[agent_type] = agent_class
        logger.info(f"Registered agent type: {agent_type}")
    
    def create_agent(self, agent_id: str, agent_type: str) -> BaseAgent:
        """
        Create an agent instance.
        
        Args:
            agent_id: Agent identifier
            agent_type: Type of agent ('user' or 'item')
        
        Returns:
            Agent instance
        
        Raises:
            ValueError: If agent_type is not registered
        """
        if agent_type not in self._agent_registry:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_class = self._agent_registry[agent_type]
        return agent_class(agent_id, self.config)
    
    def create_batch(self, agent_ids: List[str], agent_type: str) -> List[BaseAgent]:
        """
        Create multiple agents of the same type.
        
        Args:
            agent_ids: List of agent identifiers
            agent_type: Type of agent
        
        Returns:
            List of agent instances
        """
        return [self.create_agent(agent_id, agent_type) for agent_id in agent_ids]
    
    def get_agent_types(self) -> List[str]:
        """
        Get list of registered agent types.
        
        Returns:
            List of agent type names
        """
        return list(self._agent_registry.keys())


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'memory': {
            'intrinsic': {'immutable': True, 'max_size': 5},
            'collaborative': {'immutable': False, 'max_size': 20},
            'interaction': {'immutable': False, 'max_size': 15}
        }
    }
    
    # Test base agent (should use UserAgent or ItemAgent in practice)
    class TestAgent(BaseAgent):
        def get_embedding(self, component_type=None):
            return np.zeros(128)
        
        def get_text_representation(self):
            return f"Test agent {self.agent_id}"
    
    # Create agent
    agent = TestAgent('test_001', config)
    print(agent)
    print(f"Agent state: {agent.get_state()}")
    
    # Test memory operations
    agent.add_interaction({'item': 'item_001', 'rating': 5})
    print(f"Interaction count: {agent.state.interaction_count}")
    
    # Test save/load
    agent.save('test_agent.json')
    loaded_agent = TestAgent('test_001', config)
    loaded_agent.load('test_agent.json')
    print(f"Loaded agent: {loaded_agent}")