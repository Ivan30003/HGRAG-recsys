"""
__init__.py - Agent models package initialization for H-GRAGrecsys

This module exports all agent-related classes and functions, providing
a unified interface for agent creation, memory management, and interactions.
"""

import logging
from typing import Dict, Any, Optional, List, Union

# Core agent classes
from models.agent.base_agent import BaseAgent, AgentFactory
from models.agent.user_agent import UserAgent
from models.agent.item_agent import ItemAgent

# Memory components
from models.agent.memory import (
    AgentMemory,
    HierarchicalMemory,
    MemoryType,
    MemoryUpdateResult
)

from models.agent.memory_components import (
    MemoryComponent,
    IntrinsicMemory,
    CollaborativeMemory,
    InteractionMemory,
    MemoryComponentFactory
)

# Package metadata
__version__ = '1.0.0'
__author__ = 'H-GRAGrecsys Team'

# Define what gets imported with "from models.agent import *"
__all__ = [
    # Core agent classes
    'BaseAgent',
    'AgentFactory',
    'UserAgent',
    'ItemAgent',
    
    # Memory classes
    'AgentMemory',
    'HierarchicalMemory',
    'MemoryType',
    'MemoryUpdateResult',
    
    # Memory components
    'MemoryComponent',
    'IntrinsicMemory',
    'CollaborativeMemory',
    'InteractionMemory',
    'MemoryComponentFactory',
]

# Module-level logger
logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Registry for agent types and configurations.
    
    This class provides a centralized registry for all agent types,
    enabling easy creation and retrieval of agent instances.
    """
    
    _agent_types = {}
    _default_configs = {}
    
    @classmethod
    def register(cls, agent_type: str, agent_class: type, 
                 default_config: Optional[Dict] = None):
        """
        Register an agent type.
        
        Args:
            agent_type: Type of agent ('user' or 'item')
            agent_class: Agent class
            default_config: Default configuration for this agent type
        """
        cls._agent_types[agent_type] = agent_class
        if default_config:
            cls._default_configs[agent_type] = default_config
        logger.info(f"Registered agent type: {agent_type}")
    
    @classmethod
    def get_agent_class(cls, agent_type: str) -> Optional[type]:
        """
        Get agent class for a type.
        
        Args:
            agent_type: Type of agent
        
        Returns:
            Agent class or None
        """
        return cls._agent_types.get(agent_type)
    
    @classprperty
    def get_default_config(cls, agent_type: str) -> Optional[Dict]:
        """
        Get default configuration for an agent type.
        
        Args:
            agent_type: Type of agent
        
        Returns:
            Configuration dictionary or None
        """
        return cls._default_configs.get(agent_type)
    
    @classmethod
    def list_agent_types(cls) -> List[str]:
        """
        List all registered agent types.
        
        Returns:
            List of agent type names
        """
        return list(cls._agent_types.keys())
    
    @classmethod
    def create_agent(cls, agent_id: str, agent_type: str, 
                    config: Optional[Dict] = None) -> BaseAgent:
        """
        Create an agent instance.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent ('user' or 'item')
            config: Configuration dictionary (optional)
        
        Returns:
            Agent instance
        
        Raises:
            ValueError: If agent_type is not registered
        """
        agent_class = cls.get_agent_class(agent_type)
        if agent_class is None:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Merge with default config
        if config is None:
            config = cls.get_default_config(agent_type) or {}
        
        # Create and return agent
        return agent_class(agent_id, config)


def create_user_agent(user_id: str, config: Dict[str, Any]) -> UserAgent:
    """
    Convenience function to create a user agent.
    
    Args:
        user_id: User identifier
        config: Configuration dictionary
    
    Returns:
        UserAgent instance
    """
    factory = AgentFactory(config)
    return factory.create_agent(user_id, 'user')


def create_item_agent(item_id: str, config: Dict[str, Any]) -> ItemAgent:
    """
    Convenience function to create an item agent.
    
    Args:
        item_id: Item identifier
        config: Configuration dictionary
    
    Returns:
        ItemAgent instance
    """
    factory = AgentFactory(config)
    return factory.create_agent(item_id, 'item')


def create_hierarchical_memory(agent_id: str, 
                              agent_type: str,
                              config: Dict[str, Any]) -> HierarchicalMemory:
    """
    Convenience function to create hierarchical memory.
    
    Args:
        agent_id: Agent identifier
        agent_type: Type of agent ('user' or 'item')
        config: Configuration dictionary
    
    Returns:
        HierarchicalMemory instance
    """
    return HierarchicalMemory(agent_id, agent_type, config)


def create_memory_component(component_type: str,
                           agent_id: str,
                           agent_type: str,
                           config: Dict[str, Any]) -> MemoryComponent:
    """
    Convenience function to create a memory component.
    
    Args:
        component_type: 'intrinsic', 'collaborative', or 'interaction'
        agent_id: Agent identifier
        agent_type: Type of agent ('user' or 'item')
        config: Configuration dictionary
    
    Returns:
        MemoryComponent instance
    
    Raises:
        ValueError: If component_type is unknown
    """
    factory = MemoryComponentFactory(config)
    return factory.create_component(component_type, agent_id, agent_type)


def get_agent_statistics(agent: BaseAgent) -> Dict[str, Any]:
    """
    Get statistics for an agent.
    
    Args:
        agent: Agent instance
    
    Returns:
        Dictionary of statistics
    """
    stats = {
        'agent_id': agent.agent_id,
        'agent_type': agent.agent_type,
        'memory_size': agent.get_memory_size(),
        'interaction_count': len(agent.get_interaction_memory()),
        'has_intrinsic': agent.has_intrinsic_memory(),
        'has_collaborative': agent.has_collaborative_memory()
    }
    
    # Add memory component statistics
    if hasattr(agent, 'memory') and agent.memory:
        stats['memory_components'] = agent.memory.get_component_stats()
    
    return stats


def compare_agents(agent1: BaseAgent, agent2: BaseAgent) -> Dict[str, Any]:
    """
    Compare two agents.
    
    Args:
        agent1: First agent
        agent2: Second agent
    
    Returns:
        Dictionary of comparison results
    """
    return {
        'same_type': agent1.agent_type == agent2.agent_type,
        'similarity_score': agent1.calculate_similarity(agent2) 
                           if hasattr(agent1, 'calculate_similarity') else None,
        'memory_overlap': agent1.get_memory_overlap(agent2) 
                         if hasattr(agent1, 'get_memory_overlap') else None
    }


# Register default agent types
def initialize_agent_registry():
    """Initialize the agent registry with default agent types."""
    AgentRegistry.register('user', UserAgent, {
        'memory_buffer_size': 10,
        'embedding_dim': 1536,
        'consistency_threshold': 0.15
    })
    AgentRegistry.register('item', ItemAgent, {
        'memory_buffer_size': 10,
        'embedding_dim': 1536,
        'consistency_threshold': 0.15
    })
    logger.info("Agent registry initialized with default types")


# Default agent configuration
DEFAULT_AGENT_CONFIG = {
    'memory_buffer_size': 10,
    'embedding_dim': 1536,
    'consistency_threshold': 0.15,
    'memory': {
        'intrinsic': {
            'immutable': True,
            'max_size': 5
        },
        'collaborative': {
            'immutable': False,
            'max_size': 20,
            'propagation_threshold': 0.3
        },
        'interaction': {
            'immutable': False,
            'max_size': 15,
            'buffer_size': 10
        }
    },
    'reflection': {
        'enabled': True,
        'max_reflections': 5,
        'update_threshold': 0.1
    }
}


class AgentConfig:
    """
    Configuration manager for agents.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AgentConfig.
        
        Args:
            config: Configuration dictionary
        """
        self.config = DEFAULT_AGENT_CONFIG.copy()
        if config:
            self.update(config)
    
    def update(self, config: Dict[str, Any]) -> None:
        """
        Update configuration.
        
        Args:
            config: Configuration dictionary
        """
        for key, value in config.items():
            if key in self.config and isinstance(value, dict):
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory configuration."""
        return self.get('memory', {})
    
    def get_reflection_config(self) -> Dict[str, Any]:
        """Get reflection configuration."""
        return self.get('reflection', {})
    
    def get_intrinsic_config(self) -> Dict[str, Any]:
        """Get intrinsic memory configuration."""
        return self.get_memory_config().get('intrinsic', {})
    
    def get_collaborative_config(self) -> Dict[str, Any]:
        """Get collaborative memory configuration."""
        return self.get_memory_config().get('collaborative', {})
    
    def get_interaction_config(self) -> Dict[str, Any]:
        """Get interaction memory configuration."""
        return self.get_memory_config().get('interaction', {})


# Module initialization
logger.info(f"Agent models package initialized (version {__version__})")

# Initialize agent registry
initialize_agent_registry()

# Check dependencies
def check_agent_dependencies() -> Dict[str, bool]:
    """
    Check if all agent dependencies are available.
    
    Returns:
        Dictionary with dependency status
    """
    dependencies = {
        'numpy': False,
        'torch': False,
        'transformers': False
    }
    
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        pass
    
    try:
        import torch
        dependencies['torch'] = True
    except ImportError:
        pass
    
    try:
        import transformers
        dependencies['transformers'] = True
    except ImportError:
        pass
    
    return dependencies


deps = check_agent_dependencies()
missing = [dep for dep, available in deps.items() if not available]
if missing:
    logger.warning(f"Missing optional agent dependencies: {', '.join(missing)}")
    logger.warning("Some agent features may not be available")
else:
    logger.info("All agent dependencies available")


def demo():
    """
    Demonstrate usage of agent package.
    
    This function shows how to use the exported classes and functions.
    """
    print("=" * 60)
    print("Agent Package Demo")
    print("=" * 60)
    
    # 1. Show available agent types
    print("\n1. Available Agent Types:")
    agent_types = AgentRegistry.list_agent_types()
    for agent_type in agent_types:
        print(f"  - {agent_type}")
    
    # 2. Create a user agent
    print("\n2. Creating User Agent...")
    config = {
        'memory_buffer_size': 5,
        'embedding_dim': 128,
        'consistency_threshold': 0.15
    }
    
    user_agent = create_user_agent('user_001', config)
    print(f"  Created: {user_agent}")
    print(f"  ID: {user_agent.agent_id}")
    print(f"  Type: {user_agent.agent_type}")
    
    # 3. Create an item agent
    print("\n3. Creating Item Agent...")
    item_agent = create_item_agent('item_001', config)
    print(f"  Created: {item_agent}")
    print(f"  ID: {item_agent.agent_id}")
    print(f"  Type: {item_agent.agent_type}")
    
    # 4. Test memory creation
    print("\n4. Creating Hierarchical Memory...")
    memory = create_hierarchical_memory('agent_001', 'user', config)
    print(f"  Created memory: {memory}")
    
    # 5. Test memory components
    print("\n5. Creating Memory Components...")
    intrinsic = create_memory_component('intrinsic', 'agent_001', 'user', config)
    collaborative = create_memory_component('collaborative', 'agent_001', 'user', config)
    interaction = create_memory_component('interaction', 'agent_001', 'user', config)
    print(f"  Intrinsic: {intrinsic}")
    print(f"  Collaborative: {collaborative}")
    print(f"  Interaction: {interaction}")
    
    # 6. Test agent statistics
    print("\n6. Agent Statistics:")
    stats = get_agent_statistics(user_agent)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 7. Test AgentConfig
    print("\n7. Testing AgentConfig...")
    agent_config = AgentConfig()
    print(f"  Memory config: {agent_config.get_memory_config()}")
    print(f"  Reflection config: {agent_config.get_reflection_config()}")
    
    print("\n" + "=" * 60)
    print("Demo complete")
    print("=" * 60)


if __name__ == "__main__":
    demo()