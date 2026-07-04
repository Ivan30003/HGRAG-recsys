"""
__init__.py - Models package initialization for H-GRAGrecsys

This module exports all model classes and functions from the models package,
providing a unified interface for accessing agent, graph, LLM, GNN, and hybrid models.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union
import torch
import torch.nn as nn

# Agent Models
from models.agent.base_agent import BaseAgent, AgentFactory
from models.agent.user_agent import UserAgent
from models.agent.item_agent import ItemAgent
from models.agent.memory import AgentMemory, HierarchicalMemory
from models.agent.memory_components import (
    MemoryComponent,
    IntrinsicMemory,
    CollaborativeMemory,
    InteractionMemory
)

# Graph Models
from models.graph.heterogeneous_graph import HeterogeneousGraph, GraphNode
from models.graph.graph_builder import GraphBuilder
from models.graph.edge_updater import EdgeUpdater
from models.graph.relation_types import RelationType, EdgeWeightFunctions
from models.graph.dynamic_graph import DynamicGraph

# Graph RAG Models
from models.graph_rag.retriever import GraphRAGRetriever
from models.graph_rag.metapath_extractor import MetapathExtractor
from models.graph_rag.context_constructor import ContextConstructor
from models.graph_rag.ppr_sampler import PPRSampler

# LLM Models
from models.llm.llm_interface import LLMInterface, LLMFactory
from models.llm.prompt_templates import PromptTemplates
from models.llm.reflection_engine import ReflectionEngine
from models.llm.fusion_engine import FusionEngine
from models.llm.text_encoder import TextEncoder

# GNN Models
from models.gnn.heterogeneous_gnn import HeterogeneousGNN, HGNNLayer
from models.gnn.projection_heads import ProjectionHead, ComponentProjectionHeads
from models.gnn.gnn_encoder import GNNEncoder
from models.gnn.attention_module import AttentionModule

# Hybrid Models
from models.hybrid.adaptive_gate import AdaptiveGate, GatingFeatures
from models.hybrid.router import Router
from models.hybrid.inference_engine import HybridInferenceEngine

# Package metadata
__version__ = '1.0.0'
__author__ = 'H-GRAGrecsys Team'

# Define what gets imported with "from models import *"
__all__ = [
    # Agent models
    'BaseAgent',
    'AgentFactory',
    'UserAgent',
    'ItemAgent',
    'AgentMemory',
    'HierarchicalMemory',
    'MemoryComponent',
    'IntrinsicMemory',
    'CollaborativeMemory',
    'InteractionMemory',
    
    # Graph models
    'HeterogeneousGraph',
    'GraphNode',
    'GraphBuilder',
    'EdgeUpdater',
    'RelationType',
    'EdgeWeightFunctions',
    'DynamicGraph',
    
    # Graph RAG models
    'GraphRAGRetriever',
    'MetapathExtractor',
    'ContextConstructor',
    'PPRSampler',
    
    # LLM models
    'LLMInterface',
    'LLMFactory',
    'PromptTemplates',
    'ReflectionEngine',
    'FusionEngine',
    'TextEncoder',
    
    # GNN models
    'HeterogeneousGNN',
    'HGNNLayer',
    'ProjectionHead',
    'ComponentProjectionHeads',
    'GNNEncoder',
    'AttentionModule',
    
    # Hybrid models
    'AdaptiveGate',
    'GatingFeatures',
    'Router',
    'HybridInferenceEngine',
]

# Module-level logger
logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for model classes and configurations.
    
    This class provides a centralized registry for all models,
    enabling easy creation and retrieval of model instances.
    """
    
    _models = {}
    _configs = {}
    
    @classmethod
    def register(cls, name: str, model_class: type, config: Optional[Dict] = None):
        """
        Register a model class.
        
        Args:
            name: Model name
            model_class: Model class
            config: Default configuration
        """
        cls._models[name] = model_class
        if config:
            cls._configs[name] = config
        logger.info(f"Registered model: {name}")
    
    @classmethod
    def get_model(cls, name: str) -> Optional[type]:
        """
        Get a registered model class.
        
        Args:
            name: Model name
        
        Returns:
            Model class or None
        """
        return cls._models.get(name)
    
    @classmethod
    def get_config(cls, name: str) -> Optional[Dict]:
        """
        Get default configuration for a model.
        
        Args:
            name: Model name
        
        Returns:
            Configuration dictionary or None
        """
        return cls._configs.get(name)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """
        List all registered models.
        
        Returns:
            List of model names
        """
        return list(cls._models.keys())
    
    @classmethod
    def create_model(cls, name: str, **kwargs) -> Any:
        """
        Create a model instance.
        
        Args:
            name: Model name
            **kwargs: Model parameters
        
        Returns:
            Model instance
        """
        model_class = cls.get_model(name)
        if model_class is None:
            raise ValueError(f"Model '{name}' not found in registry")
        
        # Merge with default config
        if name in cls._configs:
            config = cls._configs[name].copy()
            config.update(kwargs)
            return model_class(**config)
        else:
            return model_class(**kwargs)


def get_models_package() -> Dict[str, Any]:
    """
    Get all model classes from the models package.
    
    Returns:
        Dictionary mapping model names to classes
    """
    return {
        # Agent models
        'BaseAgent': BaseAgent,
        'UserAgent': UserAgent,
        'ItemAgent': ItemAgent,
        'HierarchicalMemory': HierarchicalMemory,
        
        # Graph models
        'HeterogeneousGraph': HeterogeneousGraph,
        'DynamicGraph': DynamicGraph,
        'GraphBuilder': GraphBuilder,
        
        # Graph RAG models
        'GraphRAGRetriever': GraphRAGRetriever,
        'MetapathExtractor': MetapathExtractor,
        'ContextConstructor': ContextConstructor,
        
        # LLM models
        'LLMInterface': LLMInterface,
        'ReflectionEngine': ReflectionEngine,
        'FusionEngine': FusionEngine,
        'TextEncoder': TextEncoder,
        
        # GNN models
        'HeterogeneousGNN': HeterogeneousGNN,
        'GNNEncoder': GNNEncoder,
        'AttentionModule': AttentionModule,
        
        # Hybrid models
        'AdaptiveGate': AdaptiveGate,
        'Router': Router,
        'HybridInferenceEngine': HybridInferenceEngine,
    }


def create_agent(agent_type: str, agent_id: str, config: Dict[str, Any]) -> BaseAgent:
    """
    Factory function for creating agents.
    
    Args:
        agent_type: 'user' or 'item'
        agent_id: Unique identifier
        config: Configuration dictionary
    
    Returns:
        Agent instance
    """
    factory = AgentFactory(config)
    return factory.create_agent(agent_id, agent_type)


def create_graph(config: Dict[str, Any]) -> HeterogeneousGraph:
    """
    Factory function for creating graph.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        HeterogeneousGraph instance
    """
    return HeterogeneousGraph(config)


def create_graph_builder(config: Dict[str, Any]) -> GraphBuilder:
    """
    Factory function for creating graph builder.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        GraphBuilder instance
    """
    return GraphBuilder(config)


def create_llm_interface(model_name: str, config: Dict[str, Any]) -> LLMInterface:
    """
    Factory function for creating LLM interface.
    
    Args:
        model_name: Name of LLM model
        config: Configuration dictionary
    
    Returns:
        LLMInterface instance
    """
    factory = LLMFactory(config)
    return factory.create_llm(model_name)


def create_gnn_encoder(graph: HeterogeneousGraph, config: Dict[str, Any]) -> GNNEncoder:
    """
    Factory function for creating GNN encoder.
    
    Args:
        graph: HeterogeneousGraph instance
        config: Configuration dictionary
    
    Returns:
        GNNEncoder instance
    """
    gnn_model = HeterogeneousGNN(config)
    projection_heads = ComponentProjectionHeads(
        input_dim=gnn_model.hidden_dim,
        config=config
    )
    return GNNEncoder(gnn_model, projection_heads)


def create_hybrid_engine(gnn_encoder: GNNEncoder,
                        llm_interface: LLMInterface,
                        config: Dict[str, Any]) -> HybridInferenceEngine:
    """
    Factory function for creating hybrid inference engine.
    
    Args:
        gnn_encoder: GNNEncoder instance
        llm_interface: LLMInterface instance
        config: Configuration dictionary
    
    Returns:
        HybridInferenceEngine instance
    """
    gate = AdaptiveGate(config)
    router = Router(gate, config.get('gate_threshold', 0.3))
    return HybridInferenceEngine(gnn_encoder, llm_interface, router, config)


# Model Registry Initialization
def initialize_model_registry():
    """Initialize the model registry with default models."""
    ModelRegistry.register('user_agent', UserAgent)
    ModelRegistry.register('item_agent', ItemAgent)
    ModelRegistry.register('hierarchical_memory', HierarchicalMemory)
    ModelRegistry.register('heterogeneous_graph', HeterogeneousGraph)
    ModelRegistry.register('dynamic_graph', DynamicGraph)
    ModelRegistry.register('graph_rag_retriever', GraphRAGRetriever)
    ModelRegistry.register('llm_interface', LLMInterface)
    ModelRegistry.register('gnn_encoder', GNNEncoder)
    ModelRegistry.register('adaptive_gate', AdaptiveGate)
    ModelRegistry.register('hybrid_engine', HybridInferenceEngine)
    
    logger.info("Model registry initialized")


def check_model_dependencies() -> Dict[str, bool]:
    """
    Check if all model dependencies are available.
    
    Returns:
        Dictionary with dependency status
    """
    dependencies = {
        'torch': False,
        'torch_geometric': False,
        'dgl': False,
        'transformers': False,
        'sentence_transformers': False
    }
    
    try:
        import torch
        dependencies['torch'] = True
    except ImportError:
        pass
    
    try:
        import torch_geometric
        dependencies['torch_geometric'] = True
    except ImportError:
        pass
    
    try:
        import dgl
        dependencies['dgl'] = True
    except ImportError:
        pass
    
    try:
        import transformers
        dependencies['transformers'] = True
    except ImportError:
        pass
    
    try:
        from sentence_transformers import SentenceTransformer
        dependencies['sentence_transformers'] = True
    except ImportError:
        pass
    
    return dependencies


class ModelConfig:
    """
    Configuration manager for models.
    """
    
    DEFAULT_CONFIG = {
        'agent': {
            'memory_buffer_size': 10,
            'embedding_dim': 1536,
            'consistency_threshold': 0.15
        },
        'graph': {
            'edge_update_rate': 0.1,
            'pruning_threshold': 0.05,
            'user_similarity_threshold': 0.7,
            'item_similarity_threshold': 0.6,
            'co_interaction_threshold': 3
        },
        'gnn': {
            'hidden_dim': 256,
            'num_layers': 3,
            'num_heads': 4,
            'dropout': 0.1
        },
        'llm': {
            'model_name': 'gpt-3.5-turbo',
            'temperature': 0.7,
            'max_tokens': 512
        },
        'hybrid': {
            'gate_threshold': 0.3,
            'staleness_lambda': 0.1,
            'uniform_llm_rate': 0.15
        }
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ModelConfig.
        
        Args:
            config: Configuration dictionary
        """
        self.config = self.DEFAULT_CONFIG.copy()
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
    
    def get(self, section: str, key: Optional[str] = None) -> Any:
        """
        Get configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key (optional)
        
        Returns:
            Configuration value
        """
        if section not in self.config:
            return None
        
        if key is None:
            return self.config[section]
        else:
            return self.config[section].get(key)
    
    def get_agent_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return self.get('agent', {})
    
    def get_graph_config(self) -> Dict[str, Any]:
        """Get graph configuration."""
        return self.get('graph', {})
    
    def get_gnn_config(self) -> Dict[str, Any]:
        """Get GNN configuration."""
        return self.get('gnn', {})
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self.get('llm', {})
    
    def get_hybrid_config(self) -> Dict[str, Any]:
        """Get hybrid configuration."""
        return self.get('hybrid', {})


# Module initialization
logger.info(f"Models package initialized (version {__version__})")

# Initialize model registry
initialize_model_registry()

# Check dependencies
deps = check_model_dependencies()
missing = [dep for dep, available in deps.items() if not available]
if missing:
    logger.warning(f"Missing optional model dependencies: {', '.join(missing)}")
    logger.warning("Some model features may not be available")
else:
    logger.info("All model dependencies available")


def demo():
    """
    Demonstrate usage of models package.
    
    This function shows how to use the exported classes and functions.
    """
    print("=" * 60)
    print("Models Package Demo")
    print("=" * 60)
    
    # 1. Show available models
    print("\n1. Available Models:")
    models = get_models_package()
    for model_name in models:
        print(f"  - {model_name}")
    
    # 2. Show registered models
    print("\n2. Registered Models:")
    registered = ModelRegistry.list_models()
    for model_name in registered:
        print(f"  - {model_name}")
    
    # 3. Test ModelConfig
    print("\n3. Testing ModelConfig...")
    config = ModelConfig()
    print(f"  Agent config: {config.get_agent_config()}")
    print(f"  Graph config: {config.get_graph_config()}")
    print(f"  GNN config: {config.get_gnn_config()}")
    
    # 4. Test dependency check
    print("\n4. Checking dependencies...")
    deps = check_model_dependencies()
    for dep, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}")
    
    # 5. Test model creation
    print("\n5. Testing model creation...")
    try:
        # Create an agent
        sample_config = {
            'agent': {
                'memory_buffer_size': 5,
                'embedding_dim': 128
            }
        }
        agent = create_agent('user', 'test_user', sample_config)
        print(f"  Created UserAgent: {agent}")
    except Exception as e:
        print(f"  Failed to create agent: {e}")
    
    print("\n" + "=" * 60)
    print("Demo complete")
    print("=" * 60)


if __name__ == "__main__":
    demo()