"""
Phase 1: Bootstrap Training Module for H-GRAGrecsys

This module implements the initial bootstrap phase where agents are initialized
with collaborative reflections on interactions. The bootstrap phase creates
agent memories through LLM-based reflection on user-item interactions and
establishes the initial collaborative signals in the graph.

Key Responsibilities:
- Initialize user and item agents with their respective memories
- Run collaborative reflection on interaction data
- Collect and store reflection traces
- Build initial heterogeneous graph with agent nodes and edges
- Save agent states for subsequent phases
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle

# Add project root to path if needed
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Core imports
import torch

# Data imports
from data.dataset import BaseDataset, AmazonDataset
from data.data_loader import DataLoader, InteractionDataLoader
from data.data_preprocessor import DataPreprocessor

# Agent imports
from models.agent.user_agent import UserAgent
from models.agent.item_agent import ItemAgent
from models.agent.memory import AgentMemory, HierarchicalMemory
from models.agent.memory_components import (
    MemoryComponent,
    IntrinsicMemory,
    CollaborativeMemory,
    InteractionMemory
)
from models.agent.base_agent import AgentFactory

# Graph imports
from models.graph.heterogeneous_graph import HeterogeneousGraph
from models.graph.graph_builder import GraphBuilder
from models.graph.relation_types import RelationType, EdgeWeightFunctions

# LLM imports
from models.llm.llm_interface import LLMInterface, LLMFactory
from models.llm.prompt_templates import PromptTemplates
from models.llm.reflection_engine import ReflectionEngine
from models.llm.text_encoder import TextEncoder

# Utils imports
from utils.logger import Logger
from utils.config_loader import ConfigLoader
from utils.seed_manager import SeedManager
from utils.timer import Timer

# Training imports
from .trainer_base import BaseTrainer
from .checkpoint_manager import CheckpointManager


class Phase1Bootstrap(BaseTrainer):
    """
    Phase 1 Bootstrap Trainer
    
    Initializes and bootstraps the agent-based system through collaborative
    reflection on user-item interactions. This phase creates the foundation
    for subsequent distillation and hybrid training phases.
    """
    
    def __init__(
        self,
        dataset: BaseDataset,
        llm: LLMInterface,
        config: Dict[str, Any],
        graph_builder: Optional[GraphBuilder] = None,
        reflection_engine: Optional[ReflectionEngine] = None,
        text_encoder: Optional[TextEncoder] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        logger: Optional[Logger] = None
    ):
        """
        Initialize Phase 1 Bootstrap Trainer
        
        Args:
            dataset: BaseDataset instance containing interaction data
            llm: LLMInterface instance for reflection generation
            config: Configuration dictionary containing phase1 settings
            graph_builder: Optional GraphBuilder instance (created if None)
            reflection_engine: Optional ReflectionEngine instance (created if None)
            text_encoder: Optional TextEncoder instance (created if None)
            checkpoint_manager: Optional CheckpointManager instance (created if None)
            logger: Optional Logger instance (created if None)
            
        Raises:
            ValueError: If dataset is empty or configuration is invalid
        """
        # Initialize base trainer
        super().__init__(config, model=None, data_loader=None)
        
        if dataset is None:
            raise ValueError("Dataset cannot be None for Phase 1 Bootstrap")
        
        if llm is None:
            raise ValueError("LLM interface cannot be None for Phase 1 Bootstrap")
        
        # Store core components
        self.dataset = dataset
        self.llm = llm
        self.config = config
        
        # Extract phase-specific configuration
        self.phase_config = config.get('phase1', {})
        if not self.phase_config:
            from . import PHASE1_CONFIG
            self.phase_config = PHASE1_CONFIG.copy()
        
        # Initialize components
        self.logger = logger or Logger(
            log_dir=config.get('common', {}).get('log_dir', './logs'),
            name='phase1_bootstrap'
        )
        
        self.text_encoder = text_encoder or TextEncoder(
            model_name=config.get('llm', {}).get('encoder_model', 'sentence-transformers/all-MiniLM-L6-v2'),
            config=config
        )
        
        self.graph_builder = graph_builder or GraphBuilder(
            config=config.get('graph', {})
        )
        
        self.reflection_engine = reflection_engine or ReflectionEngine(
            llm=self.llm,
            config=config
        )
        
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            save_dir=config.get('common', {}).get('checkpoint_dir', './checkpoints'),
            max_checkpoints=config.get('common', {}).get('max_checkpoints', 5)
        )
        
        # Initialize state
        self.users: Dict[str, UserAgent] = {}
        self.items: Dict[str, ItemAgent] = {}
        self.graph: Optional[HeterogeneousGraph] = None
        self.reflection_traces: List[Dict[str, Any]] = []
        self.interaction_history: List[Dict[str, Any]] = []
        
        # Metrics tracking
        self.metrics = {
            'num_users': 0,
            'num_items': 0,
            'num_interactions': 0,
            'num_reflections': 0,
            'reflection_success_rate': 0.0,
            'avg_reflection_time': 0.0,
            'memory_consistency_scores': [],
            'graph_statistics': {}
        }
        
        # Set device
        self.device = config.get('common', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.logger.log_info(f"Phase1Bootstrap initialized on device: {self.device}")
        self.logger.log_info(f"Phase config: {self.phase_config}")
    
    def bootstrap_agents(self) -> Dict[str, Any]:
        """
        Initialize and bootstrap all agents from dataset
        
        This method creates UserAgent and ItemAgent instances for all users
        and items in the dataset, initializing their intrinsic memories with
        available metadata and features.
        
        Returns:
            Dict[str, Any]: Statistics about initialized agents
            {
                'num_users': int,
                'num_items': int,
                'user_features_available': bool,
                'item_features_available': bool
            }
            
        Raises:
            RuntimeError: If agent initialization fails
        """
        self.logger.log_info("Starting agent bootstrap process...")
        
        timer = Timer()
        timer.start()
        
        try:
            # Get user and item data
            user_items_dict = self.dataset.get_user_items()
            item_features = self.dataset.get_item_features()
            interactions = self.dataset.get_interactions()
            
            # Initialize user agents
            self.logger.log_info(f"Initializing {len(user_items_dict)} user agents...")
            
            # Create user agents
            agent_factory = AgentFactory()
            
            for user_id, items in tqdm(user_items_dict.items(), desc="Initializing User Agents"):
                try:
                    # Create user agent
                    user_agent = UserAgent(
                        user_id=user_id,
                        config=self.config.get('agent', {})
                    )
                    
                    # Initialize intrinsic memory with user features
                    user_profile = self._create_user_profile(user_id, items, interactions)
                    user_agent.get_intrinsic_memory().set_immutable(user_profile)
                    
                    # Initialize interaction memory with user interactions
                    user_interactions = self._get_user_interactions(user_id, interactions)
                    for interaction in user_interactions[:self.phase_config.get('memory_buffer_size', 10)]:
                        user_agent.get_interaction_memory().add_trace(
                            interaction=interaction,
                            explanation="Initial interaction trace"
                        )
                    
                    self.users[user_id] = user_agent
                    
                except Exception as e:
                    self.logger.log_warning(f"Failed to initialize user agent {user_id}: {e}")
                    continue
            
            # Initialize item agents
            self.logger.log_info(f"Initializing {len(item_features)} item agents...")
            
            for item_id, features in tqdm(item_features.items(), desc="Initializing Item Agents"):
                try:
                    # Create item agent
                    item_agent = ItemAgent(
                        item_id=item_id,
                        config=self.config.get('agent', {})
                    )
                    
                    # Initialize intrinsic memory with item metadata
                    item_agent.get_intrinsic_memory().set_immutable(features)
                    
                    # Initialize collaborative memory with interaction patterns
                    interactions_for_item = self._get_item_interactions(item_id, interactions)
                    item_agent.update_collaborative_pattern(
                        users=[inter['user_id'] for inter in interactions_for_item],
                        interactions=interactions_for_item
                    )
                    
                    self.items[item_id] = item_agent
                    
                except Exception as e:
                    self.logger.log_warning(f"Failed to initialize item agent {item_id}: {e}")
                    continue
            
            # Build initial graph
            self.logger.log_info("Building initial heterogeneous graph...")
            self.graph = self.graph_builder.build_graph(
                agents=list(self.users.values()) + list(self.items.values()),
                interactions=interactions
            )
            
            # Update metrics
            self.metrics['num_users'] = len(self.users)
            self.metrics['num_items'] = len(self.items)
            self.metrics['num_interactions'] = len(interactions)
            
            timer.stop()
            self.logger.log_info(f"Agent bootstrap completed in {timer.get_elapsed_time():.2f} seconds")
            
            return {
                'num_users': len(self.users),
                'num_items': len(self.items),
                'user_features_available': len(self.users) > 0,
                'item_features_available': len(self.items) > 0,
                'elapsed_time': timer.get_elapsed_time()
            }
            
        except Exception as e:
            self.logger.log_error(f"Agent bootstrap failed: {e}")
            raise RuntimeError(f"Agent bootstrap failed: {e}")
    
    def initialize_agent_memories(self) -> Dict[str, Any]:
        """
        Initialize hierarchical memories for all agents
        
        This method sets up the three-component hierarchical memory system:
        - Intrinsic memory (immutable core features)
        - Collaborative memory (neighbor-based signals)
        - Interaction memory (temporal interaction traces)
        
        Returns:
            Dict[str, Any]: Memory initialization statistics
            {
                'total_agents': int,
                'intrinsic_memory_initialized': int,
                'collaborative_memory_initialized': int,
                'interaction_memory_initialized': int
            }
        """
        self.logger.log_info("Initializing agent hierarchical memories...")
        
        timer = Timer()
        timer.start()
        
        stats = {
            'total_agents': 0,
            'intrinsic_memory_initialized': 0,
            'collaborative_memory_initialized': 0,
            'interaction_memory_initialized': 0
        }
        
        try:
            # Initialize memories for user agents
            for user_id, user_agent in tqdm(self.users.items(), desc="Initializing User Memories"):
                # Intrinsic memory already set in bootstrap_agents
                stats['intrinsic_memory_initialized'] += 1
                
                # Initialize collaborative memory through graph propagation
                if self.graph is not None:
                    neighbors = self.graph.get_neighbors(user_id, RelationType.SIMILAR_PREF.value)
                    if neighbors:
                        collaborative_memory = user_agent.get_collaborative_memory()
                        collaborative_memory.aggregate(neighbors)
                        stats['collaborative_memory_initialized'] += 1
                
                stats['total_agents'] += 1
            
            # Initialize memories for item agents
            for item_id, item_agent in tqdm(self.items.items(), desc="Initializing Item Memories"):
                # Intrinsic memory already set in bootstrap_agents
                stats['intrinsic_memory_initialized'] += 1
                
                # Initialize collaborative memory through graph propagation
                if self.graph is not None:
                    neighbors = self.graph.get_neighbors(item_id, RelationType.CONTENT_SIM.value)
                    if neighbors:
                        collaborative_memory = item_agent.get_collaborative_memory()
                        collaborative_memory.aggregate(neighbors)
                        stats['collaborative_memory_initialized'] += 1
                
                stats['total_agents'] += 1
            
            timer.stop()
            self.logger.log_info(f"Memory initialization completed in {timer.get_elapsed_time():.2f} seconds")
            
            return stats
            
        except Exception as e:
            self.logger.log_error(f"Memory initialization failed: {e}")
            raise RuntimeError(f"Memory initialization failed: {e}")
    
    def run_collaborative_reflection(
        self,
        interactions: Optional[List[Dict[str, Any]]] = None,
        num_reflections: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run collaborative reflection on interactions
        
        This method processes interactions through the reflection engine to
        generate insights and update agent memories. It uses LLM-based reflection
        on user-item interactions to capture preferences, patterns, and relationships.
        
        Args:
            interactions: Optional list of interactions to reflect on.
                         If None, uses all dataset interactions.
            num_reflections: Optional number of interactions to process.
                            If None, uses all available.
        
        Returns:
            Dict[str, Any]: Reflection statistics
            {
                'total_processed': int,
                'successful_reflections': int,
                'failed_reflections': int,
                'avg_time_per_reflection': float,
                'memory_updates': int
            }
        """
        self.logger.log_info("Starting collaborative reflection process...")
        
        timer = Timer()
        timer.start()
        
        # Get interactions to process
        if interactions is None:
            interactions = self.dataset.get_interactions()
        
        if num_reflections is not None and num_reflections < len(interactions):
            interactions = interactions[:num_reflections]
        
        stats = {
            'total_processed': 0,
            'successful_reflections': 0,
            'failed_reflections': 0,
            'avg_time_per_reflection': 0.0,
            'memory_updates': 0,
            'reflection_quality_scores': []
        }
        
        reflection_times = []
        
        try:
            # Process interactions in batches
            batch_size = self.phase_config.get('reflection_batch_size', 16)
            
            for i in tqdm(range(0, len(interactions), batch_size), desc="Processing Reflections"):
                batch = interactions[i:i + batch_size]
                
                for interaction in batch:
                    try:
                        # Get user and item agents
                        user_id = interaction.get('user_id')
                        item_id = interaction.get('item_id')
                        
                        if user_id not in self.users or item_id not in self.items:
                            self.logger.log_warning(f"Missing agent for interaction: user={user_id}, item={item_id}")
                            stats['failed_reflections'] += 1
                            continue
                        
                        user_agent = self.users[user_id]
                        item_agent = self.items[item_id]
                        
                        # Prepare context for reflection
                        context = self._prepare_reflection_context(
                            user_agent=user_agent,
                            item_agent=item_agent,
                            interaction=interaction
                        )
                        
                        # Run reflection
                        reflection_start = Timer()
                        reflection_start.start()
                        
                        reflection_result = self.reflection_engine.reflect(
                            user_agent=user_agent,
                            item_agent=item_agent,
                            outcome=interaction.get('rating', 1.0),
                            context=context
                        )
                        
                        reflection_start.stop()
                        reflection_time = reflection_start.get_elapsed_time()
                        reflection_times.append(reflection_time)
                        
                        # Update agent memories with reflection
                        if reflection_result:
                            # Update user agent memory
                            user_agent.update_memory(
                                memory_type='interaction',
                                content={
                                    'reflection': reflection_result,
                                    'item': item_id,
                                    'timestamp': Timer.get_current_timestamp()
                                }
                            )
                            
                            # Update item agent memory
                            item_agent.update_memory(
                                memory_type='collaborative',
                                content={
                                    'reflection': reflection_result,
                                    'user': user_id,
                                    'timestamp': Timer.get_current_timestamp()
                                }
                            )
                            
                            # Store reflection trace
                            trace = {
                                'user_id': user_id,
                                'item_id': item_id,
                                'interaction': interaction,
                                'reflection': reflection_result,
                                'timestamp': Timer.get_current_timestamp()
                            }
                            self.reflection_traces.append(trace)
                            
                            stats['successful_reflections'] += 1
                            stats['memory_updates'] += 2  # One for user, one for item
                            
                            # Evaluate reflection quality
                            quality_score = self._evaluate_reflection_quality(
                                reflection_result,
                                user_agent,
                                item_agent
                            )
                            stats['reflection_quality_scores'].append(quality_score)
                        else:
                            stats['failed_reflections'] += 1
                            
                    except Exception as e:
                        self.logger.log_warning(f"Reflection failed for interaction {interaction}: {e}")
                        stats['failed_reflections'] += 1
                        continue
                    
                    stats['total_processed'] += 1
                
                # Update graph after each batch
                self._update_graph_with_reflections(batch)
                
                # Save intermediate checkpoint
                if i % (batch_size * 10) == 0 and i > 0:
                    self.save_agent_states(
                        checkpoint_name=f"phase1_intermediate_{i}"
                    )
            
            # Update metrics
            if reflection_times:
                stats['avg_time_per_reflection'] = np.mean(reflection_times)
            
            self.metrics['num_reflections'] = stats['successful_reflections']
            self.metrics['reflection_success_rate'] = (
                stats['successful_reflections'] / stats['total_processed']
                if stats['total_processed'] > 0 else 0.0
            )
            self.metrics['avg_reflection_time'] = stats['avg_time_per_reflection']
            
            timer.stop()
            self.logger.log_info(f"Collaborative reflection completed in {timer.get_elapsed_time():.2f} seconds")
            self.logger.log_info(f"Reflection stats: {stats}")
            
            return stats
            
        except Exception as e:
            self.logger.log_error(f"Collaborative reflection failed: {e}")
            raise RuntimeError(f"Collaborative reflection failed: {e}")
    
    def collect_reflection_traces(self) -> List[Dict[str, Any]]:
        """
        Collect and organize all reflection traces
        
        Returns:
            List[Dict[str, Any]]: List of reflection traces with metadata
            
        Raises:
            RuntimeError: If no reflection traces are available
        """
        if not self.reflection_traces:
            self.logger.log_warning("No reflection traces collected yet")
            return []
        
        self.logger.log_info(f"Collecting {len(self.reflection_traces)} reflection traces...")
        
        # Organize traces by user and item
        organized_traces = []
        
        for trace in self.reflection_traces:
            organized_trace = {
                'trace_id': hash(f"{trace['user_id']}_{trace['item_id']}_{trace['timestamp']}"),
                'user_id': trace['user_id'],
                'item_id': trace['item_id'],
                'reflection': trace['reflection'],
                'interaction': trace['interaction'],
                'timestamp': trace['timestamp'],
                'user_embedding': self._get_agent_embedding(trace['user_id'], 'user'),
                'item_embedding': self._get_agent_embedding(trace['item_id'], 'item')
            }
            
            organized_traces.append(organized_trace)
        
        return organized_traces
    
    def save_agent_states(
        self,
        checkpoint_name: Optional[str] = None,
        save_dir: Optional[str] = None
    ) -> str:
        """
        Save all agent states, graph, and reflection traces to disk
        
        Args:
            checkpoint_name: Optional name for the checkpoint.
                           If None, uses timestamp-based name.
            save_dir: Optional directory to save checkpoint.
                     If None, uses checkpoint_manager's directory.
        
        Returns:
            str: Path to saved checkpoint
        
        Raises:
            RuntimeError: If saving fails
        """
        self.logger.log_info("Saving agent states...")
        
        timer = Timer()
        timer.start()
        
        try:
            # Prepare state dictionary
            state = {
                'users': {
                    user_id: agent.to_dict()
                    for user_id, agent in self.users.items()
                },
                'items': {
                    item_id: agent.to_dict()
                    for item_id, agent in self.items.items()
                },
                'graph': self.graph.to_dgl() if self.graph else None,
                'reflection_traces': self.reflection_traces,
                'metrics': self.metrics,
                'config': self.config,
                'timestamp': Timer.get_current_timestamp(),
                'phase': 'phase1_bootstrap'
            }
            
            # Save using checkpoint manager
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                state=state,
                epoch=self._get_current_epoch(),
                step=self._get_current_step(),
                name=checkpoint_name
            )
            
            timer.stop()
            self.logger.log_info(f"Agent states saved to {checkpoint_path} in {timer.get_elapsed_time():.2f} seconds")
            
            return checkpoint_path
            
        except Exception as e:
            self.logger.log_error(f"Failed to save agent states: {e}")
            raise RuntimeError(f"Failed to save agent states: {e}")
    
    def load_agent_states(
        self,
        checkpoint_path: Optional[str] = None,
        checkpoint_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load agent states from checkpoint
        
        Args:
            checkpoint_path: Optional direct path to checkpoint file
            checkpoint_name: Optional name of checkpoint to load
                           (used if checkpoint_path is None)
        
        Returns:
            Dict[str, Any]: Loaded state dictionary
        
        Raises:
            FileNotFoundError: If checkpoint not found
            RuntimeError: If loading fails
        """
        self.logger.log_info("Loading agent states...")
        
        timer = Timer()
        timer.start()
        
        try:
            # Load checkpoint
            if checkpoint_path:
                checkpoint = self.checkpoint_manager.load_checkpoint(
                    checkpoint_name=checkpoint_path
                )
            elif checkpoint_name:
                checkpoint = self.checkpoint_manager.load_checkpoint(
                    checkpoint_name=checkpoint_name
                )
            else:
                # Load latest checkpoint
                checkpoint = self.checkpoint_manager.load_checkpoint(
                    checkpoint_name=self.checkpoint_manager.get_latest_checkpoint()
                )
            
            if not checkpoint:
                raise FileNotFoundError("No checkpoint found to load")
            
            # Restore agents
            if 'users' in checkpoint:
                for user_id, user_data in checkpoint['users'].items():
                    user_agent = UserAgent.from_dict(user_data)
                    self.users[user_id] = user_agent
            
            if 'items' in checkpoint:
                for item_id, item_data in checkpoint['items'].items():
                    item_agent = ItemAgent.from_dict(item_data)
                    self.items[item_id] = item_agent
            
            # Restore graph
            if 'graph' in checkpoint and checkpoint['graph'] is not None:
                self.graph = HeterogeneousGraph.from_dgl(checkpoint['graph'])
            
            # Restore reflection traces
            if 'reflection_traces' in checkpoint:
                self.reflection_traces = checkpoint['reflection_traces']
            
            # Restore metrics
            if 'metrics' in checkpoint:
                self.metrics.update(checkpoint['metrics'])
            
            timer.stop()
            self.logger.log_info(f"Agent states loaded in {timer.get_elapsed_time():.2f} seconds")
            
            return checkpoint
            
        except Exception as e:
            self.logger.log_error(f"Failed to load agent states: {e}")
            raise RuntimeError(f"Failed to load agent states: {e}")
    
    def train(self) -> Dict[str, Any]:
        """
        Execute the complete Phase 1 bootstrap training process
        
        This method orchestrates the entire bootstrap process:
        1. Bootstrap agents
        2. Initialize agent memories
        3. Run collaborative reflection
        4. Collect reflection traces
        5. Save final agent states
        
        Returns:
            Dict[str, Any]: Complete training metrics and statistics
            
        Raises:
            RuntimeError: If training fails at any stage
        """
        self.logger.log_info("=" * 50)
        self.logger.log_info("Starting Phase 1 Bootstrap Training")
        self.logger.log_info("=" * 50)
        
        total_timer = Timer()
        total_timer.start()
        
        try:
            # Step 1: Bootstrap agents
            self.logger.log_info("Step 1: Bootstrapping agents...")
            bootstrap_stats = self.bootstrap_agents()
            self.logger.log_info(f"Bootstrap stats: {bootstrap_stats}")
            
            # Step 2: Initialize agent memories
            self.logger.log_info("Step 2: Initializing agent memories...")
            memory_stats = self.initialize_agent_memories()
            self.logger.log_info(f"Memory stats: {memory_stats}")
            
            # Step 3: Run collaborative reflection
            self.logger.log_info("Step 3: Running collaborative reflection...")
            reflection_stats = self.run_collaborative_reflection()
            self.logger.log_info(f"Reflection stats: {reflection_stats}")
            
            # Step 4: Collect reflection traces
            self.logger.log_info("Step 4: Collecting reflection traces...")
            traces = self.collect_reflection_traces()
            self.logger.log_info(f"Collected {len(traces)} reflection traces")
            
            # Step 5: Save final agent states
            self.logger.log_info("Step 5: Saving final agent states...")
            checkpoint_path = self.save_agent_states(
                checkpoint_name="phase1_complete"
            )
            self.logger.log_info(f"Final checkpoint saved to: {checkpoint_path}")
            
            # Prepare final metrics
            final_metrics = {
                'bootstrap_stats': bootstrap_stats,
                'memory_stats': memory_stats,
                'reflection_stats': reflection_stats,
                'total_traces': len(traces),
                'checkpoint_path': checkpoint_path,
                'total_time': total_timer.get_elapsed_time(),
                'metrics': self.metrics
            }
            
            total_timer.stop()
            self.logger.log_info("=" * 50)
            self.logger.log_info(f"Phase 1 Bootstrap completed in {total_timer.get_elapsed_time():.2f} seconds")
            self.logger.log_info("=" * 50)
            
            return final_metrics
            
        except Exception as e:
            self.logger.log_error(f"Phase 1 Bootstrap failed: {e}")
            raise RuntimeError(f"Phase 1 Bootstrap failed: {e}")
    
    # Private helper methods
    
    def _create_user_profile(
        self,
        user_id: str,
        items: List[str],
        interactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create user profile from interactions and items
        
        Args:
            user_id: User identifier
            items: List of item IDs interacted with
            interactions: All interaction data
            
        Returns:
            Dict[str, Any]: User profile features
        """
        user_interactions = self._get_user_interactions(user_id, interactions)
        
        # Extract user features
        profile = {
            'user_id': user_id,
            'num_interactions': len(user_interactions),
            'interacted_items': items,
            'avg_rating': np.mean([i.get('rating', 1.0) for i in user_interactions]) if user_interactions else 0.0,
            'interaction_history': user_interactions[:self.phase_config.get('memory_buffer_size', 10)],
            'preference_categories': self._extract_preference_categories(user_interactions)
        }
        
        return profile
    
    def _get_user_interactions(
        self,
        user_id: str,
        interactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get all interactions for a specific user"""
        return [i for i in interactions if i.get('user_id') == user_id]
    
    def _get_item_interactions(
        self,
        item_id: str,
        interactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get all interactions for a specific item"""
        return [i for i in interactions if i.get('item_id') == item_id]
    
    def _extract_preference_categories(
        self,
        interactions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Extract preference categories from interactions"""
        categories = defaultdict(float)
        total = len(interactions)
        
        for interaction in interactions:
            category = interaction.get('category', 'unknown')
            rating = interaction.get('rating', 1.0)
            categories[category] += rating
        
        # Normalize
        for category in categories:
            categories[category] /= total
        
        return dict(categories)
    
    def _prepare_reflection_context(
        self,
        user_agent: UserAgent,
        item_agent: ItemAgent,
        interaction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare context for reflection"""
        context = {
            'user_id': user_agent.agent_id,
            'item_id': item_agent.agent_id,
            'interaction': interaction,
            'user_preferences': user_agent.get_preference_memory(),
            'item_metadata': item_agent.get_item_metadata(),
            'user_history': user_agent.get_recommendation_history()[:5],
            'item_similarity': item_agent.calculate_content_similarity(item_agent)
        }
        
        return context
    
    def _evaluate_reflection_quality(
        self,
        reflection: Dict[str, Any],
        user_agent: UserAgent,
        item_agent: ItemAgent
    ) -> float:
        """Evaluate the quality of a reflection"""
        quality_score = 0.0
        
        # Check if reflection contains meaningful content
        if reflection.get('preference_insight'):
            quality_score += 0.4
        
        if reflection.get('collaborative_signal'):
            quality_score += 0.3
        
        if reflection.get('explanation'):
            quality_score += 0.3
        
        return quality_score
    
    def _update_graph_with_reflections(
        self,
        batch: List[Dict[str, Any]]
    ) -> None:
        """Update graph with new reflection information"""
        if self.graph is None:
            return
        
        for interaction in batch:
            user_id = interaction.get('user_id')
            item_id = interaction.get('item_id')
            
            # Update edge weights based on reflection
            self.graph.update_edge_weight(
                source=user_id,
                target=item_id,
                new_weight=interaction.get('reflection_weight', 1.0)
            )
    
    def _get_agent_embedding(
        self,
        agent_id: str,
        agent_type: str
    ) -> np.ndarray:
        """Get embedding for an agent"""
        if agent_type == 'user' and agent_id in self.users:
            return self.users[agent_id].get_embedding()
        elif agent_type == 'item' and agent_id in self.items:
            return self.items[agent_id].get_content_embedding()
        else:
            return np.zeros(self.text_encoder.get_embedding_dimension())
    
    def _get_current_epoch(self) -> int:
        """Get current epoch for checkpoint"""
        return self.metrics.get('epoch', 0)
    
    def _get_current_step(self) -> int:
        """Get current step for checkpoint"""
        return self.metrics.get('step', 0)
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate the bootstrap process
        
        Returns:
            Dict[str, Any]: Validation metrics
        """
        validation_metrics = {
            'agents_initialized': len(self.users) + len(self.items),
            'graph_edges': self.graph.get_graph_statistics().get('num_edges', 0) if self.graph else 0,
            'reflection_traces': len(self.reflection_traces),
            'memory_consistency': np.mean(self.metrics.get('memory_consistency_scores', [0.0]))
        }
        
        self.logger.log_info(f"Validation metrics: {validation_metrics}")
        return validation_metrics
    
    def test(self) -> Dict[str, Any]:
        """
        Test the bootstrap results
        
        Returns:
            Dict[str, Any]: Test metrics
        """
        test_metrics = {
            'user_agent_count': len(self.users),
            'item_agent_count': len(self.items),
            'reflection_success_rate': self.metrics['reflection_success_rate'],
            'avg_reflection_time': self.metrics['avg_reflection_time'],
            'total_reflections': self.metrics['num_reflections']
        }
        
        self.logger.log_info(f"Test metrics: {test_metrics}")
        return test_metrics


# Command-line interface for running Phase 1 independently
def main(config_path: str) -> None:
    """
    Main entry point for running Phase 1 independently
    
    Args:
        config_path: Path to configuration file
        
    Raises:
        FileNotFoundError: If config file not found
        RuntimeError: If execution fails
    """
    # Load configuration
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config_loader = ConfigLoader(config_path)
    config = config_loader.load_config()
    
    # Set up logging
    logger = Logger(
        log_dir=config.get('common', {}).get('log_dir', './logs'),
        name='phase1_main'
    )
    
    logger.log_info("Starting Phase 1 Bootstrap main execution...")
    
    try:
        # Initialize dataset
        dataset_config = config.get('data', {})
        dataset = AmazonDataset(
            dataset_name=dataset_config.get('name', 'amazon_reviews'),
            config=dataset_config
        )
        dataset.load_data()
        
        # Initialize LLM
        llm_config = config.get('llm', {})
        llm = LLMFactory.create_llm(
            model_type=llm_config.get('model_type', 'openai'),
            config=llm_config
        )
        
        # Initialize Phase 1 trainer
        trainer = Phase1Bootstrap(
            dataset=dataset,
            llm=llm,
            config=config
        )
        
        # Run training
        results = trainer.train()
        
        # Save results
        results_path = os.path.join(
            config.get('common', {}).get('log_dir', './logs'),
            'phase1_results.json'
        )
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.log_info(f"Results saved to {results_path}")
        logger.log_info("Phase 1 Bootstrap completed successfully")
        
    except Exception as e:
        logger.log_error(f"Phase 1 Bootstrap failed: {e}")
        raise


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase1_bootstrap.py <config_path>")
        sys.exit(1)
    
    main(sys.argv[1])