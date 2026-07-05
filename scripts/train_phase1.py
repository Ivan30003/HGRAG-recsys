"""
scripts/train_phase1.py

Phase 1 Training Script for H-GRAGrecsys - Bootstrap Agents with LLM Reflections

This script implements the first phase of the H-GRAGrecsys training pipeline:
1. Load and preprocess data
2. Initialize user and item agents
3. Initialize agent memories (intrinsic, collaborative, interaction)
4. Run collaborative reflection using LLM
5. Collect reflection traces
6. Save agent states and checkpoints

Features:
- Configurable training parameters
- Checkpoint saving and loading
- Progress logging and visualization
- GPU support
- Distributed training support
- Experiment tracking
"""

import os
import sys
import json
import yaml
import argparse
import pickle
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple
from datetime import datetime
import traceback
import shutil

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.logger import get_logger
from utils.config_loader import ConfigLoader, load_config
from utils.seed_manager import create_seed_manager
from utils.timer import Timer, global_timer
from utils.visualizer import create_visualizer

# Import training module
from training.phase1_bootstrap import Phase1Bootstrap

# Import data modules
from data.amazon_dataset import AmazonDataset
from data.data_loader import DataLoader
from data.data_preprocessor import DataPreprocessor

# Import model components
from models.agent.user_agent import UserAgent
from models.agent.item_agent import ItemAgent
from models.agent.memory import AgentMemory, HierarchicalMemory
from models.llm.llm_interface import LLMInterface
from models.graph.graph_builder import GraphBuilder
from models.graph.heterogeneous_graph import HeterogeneousGraph

# Import evaluation
from evaluation.evaluator import Evaluator

# Try to import torch for GPU support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class Phase1Trainer:
    """
    Phase 1 Trainer for H-GRAGrecsys - Bootstrap Agents with LLM Reflections.
    
    Features:
    - Agent initialization with hierarchical memory
    - Collaborative reflection using LLM
    - Reflection trace collection
    - Checkpoint saving and loading
    - Progress monitoring and visualization
    - GPU support
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        resume_from: Optional[Union[str, Path]] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        logger: Optional['Logger'] = None,
        verbose: bool = True
    ):
        """
        Initialize the Phase 1 Trainer.
        
        Args:
            config_path (str, Path, optional): Path to configuration file
            checkpoint_dir (str, Path, optional): Directory to save checkpoints
            output_dir (str, Path, optional): Directory for outputs
            resume_from (str, Path, optional): Checkpoint to resume from
            seed (int, optional): Random seed for reproducibility
            device (str, optional): Device to use ('cpu', 'cuda')
            logger (Logger, optional): Logger instance
            verbose (bool): Whether to enable verbose output
        
        Example:
            trainer = Phase1Trainer(
                config_path='config/default_config.yaml',
                output_dir='experiments/phase1'
            )
            trainer.train()
        """
        # Setup paths
        self.config_path = Path(config_path) if config_path else None
        self.output_dir = Path(output_dir) if output_dir else Path("experiments/phase1")
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        if logger is None:
            self.logger = get_logger(
                log_dir=self.log_dir,
                name="phase1_trainer",
                verbose=verbose
            )
        else:
            self.logger = logger
        
        # Load configuration
        if self.config_path and self.config_path.exists():
            self.config_loader = ConfigLoader(
                config_path=self.config_path,
                logger=self.logger
            )
            self.config = self.config_loader.config
        else:
            self.config_loader = ConfigLoader(load_defaults=True)
            self.config = self.config_loader.config
        
        # Save config to output directory
        self.config_loader.save_config(path=self.output_dir / "config.yaml")
        
        # Setup seed manager
        self.seed = seed or self.config.get('seed', 42)
        self.seed_manager = create_seed_manager(
            seed=self.seed,
            config_path=self.config_path
        )
        self.seed_manager.set_all_seeds()
        
        # Setup timer
        self.timer = Timer(
            name="phase1_training",
            logger=self.logger,
            track_memory=True,
            track_gpu=True,
            save_report=True,
            report_dir=self.output_dir / "timing"
        )
        
        # Setup visualizer
        self.visualizer = create_visualizer(
            config_path=self.config_path,
            output_dir=self.output_dir / "plots",
            interactive=False
        )
        
        # Set device
        self.device = device or self._get_default_device()
        self.logger.log_info(f"Using device: {self.device}")
        
        # Resume from checkpoint
        self.resume_from = Path(resume_from) if resume_from else None
        self.start_epoch = 0
        
        # Initialize training components
        self.phase1 = None
        self.dataset = None
        self.data_loader = None
        self.llm_interface = None
        self.graph_builder = None
        self.graph = None
        
        # Training state
        self.state = {
            'epoch': 0,
            'best_metrics': {},
            'training_completed': False
        }
        
        self.logger.log_info("Phase1Trainer initialized")
        self.logger.log_info(f"Output directory: {self.output_dir}")
        self.logger.log_info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def _get_default_device(self) -> str:
        """
        Get the default device (GPU if available, else CPU).
        
        Returns:
            str: Device name
        """
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def train(self) -> Dict[str, Any]:
        """
        Run Phase 1 training.
        
        Returns:
            Dict[str, Any]: Training results and metrics
        
        Example:
            results = trainer.train()
            print(f"Training completed: {results['epochs_completed']} epochs")
        """
        self.logger.log_info("=" * 80)
        self.logger.log_info("PHASE 1: Bootstrap Agents with LLM Reflections")
        self.logger.log_info("=" * 80)
        
        with self.timer.measure("phase1_training"):
            # Step 1: Load data
            self._load_data()
            
            # Step 2: Initialize LLM interface
            self._initialize_llm()
            
            # Step 3: Initialize agents
            self._initialize_agents()
            
            # Step 4: Initialize graph
            self._initialize_graph()
            
            # Step 5: Initialize Phase 1
            self._initialize_phase1()
            
            # Step 6: Resume if requested
            if self.resume_from:
                self._resume_training()
            
            # Step 7: Run training
            results = self._run_training()
            
            # Step 8: Save final model
            self._save_final_model()
            
            # Step 9: Generate summary
            self._generate_summary(results)
        
        self.logger.log_info("=" * 80)
        self.logger.log_info("Phase 1 Training Completed")
        self.logger.log_info("=" * 80)
        
        return results
    
    def _load_data(self) -> None:
        """
        Load and prepare dataset.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("LOADING DATA")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("load_data"):
            # Get dataset name from config
            dataset_name = self.config.get('data', {}).get('dataset_name', 'Amazon_Books')
            
            # Load dataset
            self.dataset = AmazonDataset(dataset_name, self.config)
            self.dataset.load_data()
            
            # Get statistics
            stats = self.dataset.get_statistics()
            self.logger.log_info(f"Dataset: {dataset_name}")
            self.logger.log_info(f"Users: {stats.get('num_users', 0):,}")
            self.logger.log_info(f"Items: {stats.get('num_items', 0):,}")
            self.logger.log_info(f"Interactions: {stats.get('num_interactions', 0):,}")
            self.logger.log_info(f"Sparsity: {stats.get('sparsity', 1.0):.4f}")
            
            # Create data loader
            batch_size = self.config.get('training', {}).get('phase1', {}).get('batch_size', 32)
            self.data_loader = DataLoader(
                dataset=self.dataset,
                batch_size=batch_size,
                shuffle=True
            )
            
            self.logger.log_info(f"Batch size: {batch_size}")
            self.logger.log_info(f"Number of batches: {len(self.data_loader.get_train_batches())}")
    
    def _initialize_llm(self) -> None:
        """
        Initialize LLM interface for reflections.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("INITIALIZING LLM INTERFACE")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("initialize_llm"):
            model_name = self.config.get('llm', {}).get('model_name', 'gpt-3.5-turbo')
            self.llm_interface = LLMInterface(
                model_name=model_name,
                config=self.config
            )
            
            self.logger.log_info(f"LLM model: {model_name}")
            self.logger.log_info(f"LLM interface initialized successfully")
    
    def _initialize_agents(self) -> None:
        """
        Initialize user and item agents with hierarchical memory.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("INITIALIZING AGENTS")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("initialize_agents"):
            # Get users and items
            users = list(self.dataset.get_user_items().keys())
            items = list(self.dataset.get_item_features().keys())
            
            self.logger.log_info(f"Initializing {len(users)} user agents...")
            self.logger.log_info(f"Initializing {len(items)} item agents...")
            
            # Initialize user agents
            self.user_agents = {}
            for user_id in tqdm(users, desc="Creating user agents", 
                               disable=not TQDM_AVAILABLE):
                self.user_agents[user_id] = UserAgent(
                    user_id=user_id,
                    config=self.config
                )
                
                # Initialize intrinsic memory with user profile
                user_items = self.dataset.get_user_items().get(user_id, [])
                user_profile = self._create_user_profile(user_id, user_items)
                self.user_agents[user_id].intrinsic_memory.set_immutable(user_profile)
            
            # Initialize item agents
            self.item_agents = {}
            for item_id in tqdm(items, desc="Creating item agents",
                               disable=not TQDM_AVAILABLE):
                self.item_agents[item_id] = ItemAgent(
                    item_id=item_id,
                    config=self.config
                )
                
                # Initialize intrinsic memory with item metadata
                item_metadata = self.dataset.get_item_features().get(item_id, {})
                self.item_agents[item_id].intrinsic_memory.set_immutable(item_metadata)
            
            self.logger.log_info(f"Initialized {len(self.user_agents)} user agents")
            self.logger.log_info(f"Initialized {len(self.item_agents)} item agents")
    
    def _create_user_profile(self, user_id: str, items: List[str]) -> Dict[str, Any]:
        """
        Create user profile from interactions.
        
        Args:
            user_id (str): User ID
            items (List[str]): List of item IDs
            
        Returns:
            Dict[str, Any]: User profile
        """
        # Get item features
        item_features = []
        for item_id in items:
            features = self.dataset.get_item_features().get(item_id, {})
            if features:
                item_features.append(features)
        
        # Aggregate features
        profile = {
            'user_id': user_id,
            'num_interactions': len(items),
            'item_ids': items,
            'preferences': self._aggregate_preferences(item_features)
        }
        
        return profile
    
    def _aggregate_preferences(self, item_features: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregate preferences from item features.
        
        Args:
            item_features (List[Dict[str, Any]]): Item features
            
        Returns:
            Dict[str, float]: Aggregated preferences
        """
        if not item_features:
            return {}
        
        # Aggregate categorical features
        categories = []
        for features in item_features:
            if 'categories' in features:
                categories.extend(features['categories'])
        
        # Count categories
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Normalize
        total = len(categories) if categories else 1
        preferences = {cat: count / total for cat, count in category_counts.items()}
        
        return preferences
    
    def _initialize_graph(self) -> None:
        """
        Initialize heterogeneous graph with agents and interactions.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("INITIALIZING GRAPH")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("initialize_graph"):
            # Initialize graph builder
            self.graph_builder = GraphBuilder(self.config)
            
            # Build graph
            all_agents = list(self.user_agents.values()) + list(self.item_agents.values())
            interactions = self.dataset.get_interactions()
            
            self.graph = self.graph_builder.build_graph(
                agents=all_agents,
                interactions=interactions
            )
            
            self.logger.log_info(f"Graph built successfully")
            self.logger.log_info(f"Nodes: {self.graph.get_graph_statistics().get('num_nodes', 0)}")
            self.logger.log_info(f"Edges: {self.graph.get_graph_statistics().get('num_edges', 0)}")
    
    def _initialize_phase1(self) -> None:
        """
        Initialize Phase 1 training module.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("INITIALIZING PHASE 1 TRAINING")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("initialize_phase1"):
            self.phase1 = Phase1Bootstrap(
                dataset=self.dataset,
                llm=self.llm_interface,
                config=self.config,
                logger=self.logger,
                seed_manager=self.seed_manager
            )
            
            # Bootstrap agents
            self.phase1.bootstrap_agents()
            
            self.logger.log_info("Phase 1 training module initialized")
    
    def _run_training(self) -> Dict[str, Any]:
        """
        Run the Phase 1 training loop.
        
        Returns:
            Dict[str, Any]: Training results
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("RUNNING TRAINING")
        self.logger.log_info("-" * 50)
        
        # Get training parameters
        train_config = self.config.get('training', {}).get('phase1', {})
        num_epochs = train_config.get('num_epochs', 50)
        learning_rate = train_config.get('learning_rate', 1e-4)
        easy_threshold = train_config.get('easy_threshold', 10)
        
        self.logger.log_info(f"Number of epochs: {num_epochs}")
        self.logger.log_info(f"Learning rate: {learning_rate}")
        self.logger.log_info(f"Easy threshold: {easy_threshold}")
        
        # Training loop
        training_results = {
            'epochs_completed': 0,
            'metrics_history': [],
            'best_metrics': {},
            'reflection_traces': []
        }
        
        for epoch in range(self.start_epoch, num_epochs):
            self.logger.log_info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            with self.timer.measure(f"epoch_{epoch}"):
                # Run one epoch
                epoch_metrics = self._train_epoch(epoch)
                
                # Store metrics
                training_results['metrics_history'].append(epoch_metrics)
                training_results['epochs_completed'] = epoch + 1
                
                # Update best metrics
                if not training_results['best_metrics'] or epoch_metrics.get('ndcg@10', 0) > training_results['best_metrics'].get('ndcg@10', 0):
                    training_results['best_metrics'] = epoch_metrics
                    self._save_checkpoint(epoch, epoch_metrics, is_best=True)
                
                # Save checkpoint
                if (epoch + 1) % train_config.get('save_interval', 5) == 0:
                    self._save_checkpoint(epoch, epoch_metrics, is_best=False)
                
                # Log progress
                self._log_epoch_metrics(epoch, epoch_metrics)
                
                # Early stopping
                if self._check_early_stopping(training_results['metrics_history']):
                    self.logger.log_info("Early stopping triggered")
                    break
        
        # Collect reflection traces
        training_results['reflection_traces'] = self.phase1.collect_reflection_traces()
        
        self.logger.log_info(f"\nTraining completed: {training_results['epochs_completed']} epochs")
        self.logger.log_info(f"Best NDCG@10: {training_results['best_metrics'].get('ndcg@10', 0):.4f}")
        
        return training_results
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            Dict[str, float]: Epoch metrics
        """
        # Get training batches
        batches = self.data_loader.get_train_batches()
        
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(batches):
            # Run collaborative reflection on batch
            reflections = self.phase1.run_collaborative_reflection(batch)
            
            # Update agent memories
            for reflection in reflections:
                self._update_agent_memory(reflection)
            
            # Update graph
            self._update_graph(batch)
            
            batch_count += 1
            
            # Log batch progress
            if (batch_idx + 1) % 10 == 0:
                self.logger.log_debug(f"  Batch {batch_idx + 1}/{len(batches)} processed")
        
        # Evaluate after epoch
        eval_metrics = self._evaluate()
        
        return eval_metrics
    
    def _update_agent_memory(self, reflection: Dict[str, Any]) -> None:
        """
        Update agent memory with reflection results.
        
        Args:
            reflection (Dict[str, Any]): Reflection data
        """
        user_id = reflection.get('user_id')
        item_id = reflection.get('item_id')
        outcome = reflection.get('outcome')
        explanation = reflection.get('explanation')
        
        # Update user agent
        if user_id in self.user_agents:
            self.user_agents[user_id].reflect_on_interaction(
                item=item_id,
                outcome=outcome,
                context={'explanation': explanation}
            )
        
        # Update item agent
        if item_id in self.item_agents:
            self.item_agents[item_id].update_collaborative_pattern(
                users=[user_id],
                interactions=[{'outcome': outcome}]
            )
    
    def _update_graph(self, batch: Dict[str, Any]) -> None:
        """
        Update graph with new interactions.
        
        Args:
            batch (Dict[str, Any]): Batch data
        """
        # Extract interactions from batch
        interactions = batch.get('interactions', [])
        
        # Update graph edges
        for interaction in interactions:
            user = interaction.get('user')
            item = interaction.get('item')
            rating = interaction.get('rating', 1.0)
            
            if user and item:
                # Update edge weight
                self.graph.update_edge_weight(
                    source=user,
                    target=item,
                    new_weight=rating
                )
    
    def _evaluate(self) -> Dict[str, float]:
        """
        Evaluate current model.
        
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Create evaluator
        evaluator = Evaluator(
            model=self._get_model(),
            dataset=self.dataset,
            config=self.config,
            logger=self.logger
        )
        
        # Run evaluation
        results = evaluator.evaluate()
        
        # Extract metrics
        metrics = {}
        if 'metrics' in results:
            for key, value in results['metrics'].items():
                if isinstance(value, (int, float)):
                    metrics[key] = value
        
        return metrics
    
    def _get_model(self):
        """
        Get current model state.
        
        Returns:
            HybridInferenceEngine: Current model
        """
        # Create inference engine with current state
        from models.hybrid.inference_engine import HybridInferenceEngine
        from models.gnn.gnn_encoder import GNNEncoder
        
        gnn_encoder = GNNEncoder(self.config)
        
        return HybridInferenceEngine(
            gnn_encoder=gnn_encoder,
            llm_interface=self.llm_interface,
            gate=None,
            config=self.config
        )
    
    def _check_early_stopping(self, history: List[Dict[str, float]]) -> bool:
        """
        Check if early stopping criteria are met.
        
        Args:
            history (List[Dict[str, float]]): Training history
            
        Returns:
            bool: Whether to stop training
        """
        if len(history) < 10:
            return False
        
        # Check if NDCG@10 has plateaued
        ndcg_values = [h.get('ndcg@10', 0) for h in history[-10:]]
        
        if len(ndcg_values) < 10:
            return False
        
        # Check if improvement is less than threshold
        improvement = ndcg_values[-1] - ndcg_values[0]
        if improvement < 0.001:
            self.logger.log_info("No significant improvement in last 10 epochs")
            return True
        
        return False
    
    def _log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Log epoch metrics.
        
        Args:
            epoch (int): Epoch number
            metrics (Dict[str, float]): Epoch metrics
        """
        self.logger.log_info(f"Epoch {epoch + 1} Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.logger.log_info(f"  {key}: {value:.4f}")
        
        # Log to visualizer
        self.visualizer.plot_training_metrics(
            {'epoch': list(range(1, epoch + 2))},
            title="Phase 1 Training Progress",
            save_name="phase1_training_progress",
            show=False
        )
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        """
        Save training checkpoint.
        
        Args:
            epoch (int): Current epoch
            metrics (Dict[str, float]): Evaluation metrics
            is_best (bool): Whether this is the best model
        """
        checkpoint = {
            'epoch': epoch,
            'epochs_completed': epoch + 1,
            'metrics': metrics,
            'state': {
                'user_agents': self.user_agents,
                'item_agents': self.item_agents,
                'graph': self.graph,
                'phase1_state': self.phase1.get_state() if self.phase1 else {}
            },
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save checkpoint
        if is_best:
            filename = "phase1_best.pt"
        else:
            filename = f"phase1_epoch_{epoch+1:03d}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        self.logger.log_debug(f"Checkpoint saved: {checkpoint_path}")
        
        # Also save as JSON for inspection
        json_path = checkpoint_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump({
                'epoch': epoch,
                'epochs_completed': epoch + 1,
                'metrics': metrics,
                'timestamp': checkpoint['timestamp']
            }, f, indent=2, default=str)
    
    def _resume_training(self) -> None:
        """
        Resume training from checkpoint.
        """
        self.logger.log_info(f"\nResuming from checkpoint: {self.resume_from}")
        
        if not self.resume_from.exists():
            self.logger.log_warning(f"Checkpoint not found: {self.resume_from}")
            return
        
        with open(self.resume_from, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Restore state
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.state = checkpoint.get('state', {})
        
        # Restore agents
        if 'user_agents' in self.state:
            self.user_agents = self.state['user_agents']
        if 'item_agents' in self.state:
            self.item_agents = self.state['item_agents']
        if 'graph' in self.state:
            self.graph = self.state['graph']
        
        self.logger.log_info(f"Resuming from epoch {self.start_epoch}")
        self.logger.log_info(f"Best metrics: {checkpoint.get('metrics', {})}")
    
    def _save_final_model(self) -> None:
        """
        Save final model and agent states.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("SAVING FINAL MODEL")
        self.logger.log_info("-" * 50)
        
        # Save agent states
        agent_states_path = self.checkpoint_dir / "agent_states.pt"
        with open(agent_states_path, 'wb') as f:
            pickle.dump({
                'user_agents': self.user_agents,
                'item_agents': self.item_agents,
                'graph': self.graph,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }, f)
        
        self.logger.log_info(f"Agent states saved to: {agent_states_path}")
        
        # Save reflection traces
        if self.phase1:
            traces_path = self.checkpoint_dir / "reflection_traces.pt"
            traces = self.phase1.collect_reflection_traces()
            with open(traces_path, 'wb') as f:
                pickle.dump(traces, f)
            self.logger.log_info(f"Reflection traces saved to: {traces_path}")
    
    def _generate_summary(self, results: Dict[str, Any]) -> None:
        """
        Generate training summary.
        
        Args:
            results (Dict[str, Any]): Training results
        """
        summary_path = self.output_dir / "training_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PHASE 1 TRAINING SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write(f"Seed: {self.seed}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Epochs completed: {results.get('epochs_completed', 0)}\n")
            f.write("\n")
            
            f.write("BEST METRICS\n")
            f.write("-" * 40 + "\n")
            best_metrics = results.get('best_metrics', {})
            for key, value in best_metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"  {key}: {value:.4f}\n")
            f.write("\n")
            
            f.write("FINAL METRICS\n")
            f.write("-" * 40 + "\n")
            if results.get('metrics_history'):
                final_metrics = results['metrics_history'][-1]
                for key, value in final_metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {key}: {value:.4f}\n")
            f.write("\n")
            
            f.write("REFLECTION TRACES\n")
            f.write("-" * 40 + "\n")
            traces = results.get('reflection_traces', [])
            f.write(f"Total reflections: {len(traces)}\n")
            if traces:
                f.write(f"Sample reflection:\n")
                sample = traces[0]
                f.write(f"  User: {sample.get('user_id', 'N/A')}\n")
                f.write(f"  Item: {sample.get('item_id', 'N/A')}\n")
                f.write(f"  Outcome: {sample.get('outcome', 'N/A')}\n")
                f.write(f"  Explanation: {sample.get('explanation', 'N/A')[:100]}...\n")
            f.write("\n")
            
            f.write("OUTPUTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Output directory: {self.output_dir}\n")
            f.write(f"  Checkpoint directory: {self.checkpoint_dir}\n")
            f.write(f"  Log directory: {self.log_dir}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("End of Summary\n")
            f.write("=" * 80 + "\n")
        
        self.logger.log_info(f"Training summary saved to: {summary_path}")


def main():
    """
    Main entry point for Phase 1 training.
    """
    parser = argparse.ArgumentParser(description="H-GRAGrecsys Phase 1 Training Script")
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory to save checkpoints'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for training artifacts'
    )
    
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Checkpoint to resume from'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--no-verbose',
        action='store_true',
        help='Disable verbose output'
    )
    
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only run evaluation, skip training'
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = Phase1Trainer(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        seed=args.seed,
        device=args.device,
        verbose=not args.no_verbose
    )
    
    # Run training or evaluation
    if args.eval_only:
        # Load checkpoint and evaluate
        if args.resume_from:
            trainer._resume_training()
            results = trainer._evaluate()
            print(f"Evaluation results: {results}")
        else:
            print("Error: --resume-from required for evaluation-only mode")
            sys.exit(1)
    else:
        results = trainer.train()
        
        # Print summary
        print("\n" + "=" * 40)
        print("Training completed!")
        print(f"Best NDCG@10: {results['best_metrics'].get('ndcg@10', 0):.4f}")
        print(f"Best Hit Rate: {results['best_metrics'].get('hit_rate', 0):.4f}")
        print(f"Checkpoints saved in: {trainer.checkpoint_dir}")
        print("=" * 40 + "\n")
    
    return results


if __name__ == "__main__":
    main()