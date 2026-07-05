"""
scripts/train_phase2.py

Phase 2 Training Script for H-GRAGrecsys - Distillation from LLM to GNN

This script implements the second phase of the H-GRAGrecsys training pipeline:
1. Load Phase 1 trained agents and reflection traces
2. Initialize teacher (LLM) and student (GNN) models
3. Perform knowledge distillation with component-wise losses
4. Align representations between teacher and student
5. Transfer memory dynamics
6. Save distilled model and checkpoints

Features:
- Component-wise distillation loss
- Path importance loss
- Contrastive learning
- Memory dynamics transfer
- Configurable distillation parameters
- Checkpoint saving and loading
- Progress logging and visualization
- GPU support
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
import numpy as np

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.logger import get_logger
from utils.config_loader import ConfigLoader, load_config
from utils.seed_manager import create_seed_manager
from utils.timer import Timer, global_timer
from utils.visualizer import create_visualizer

# Import training module
from training.phase2_distillation import Phase2Distillation

# Import distillation modules
from distillation.distillation_trainer import DistillationTrainer
from distillation.loss_functions import DistillationLoss
from distillation.knowledge_distiller import KnowledgeDistiller
from distillation.component_disentangler import ComponentDisentangler

# Import data modules
from data.amazon_dataset import AmazonDataset
from data.data_loader import DataLoader

# Import model components
from models.agent.user_agent import UserAgent
from models.agent.item_agent import ItemAgent
from models.gnn.gnn_encoder import GNNEncoder
from models.gnn.heterogeneous_gnn import HeterogeneousGNN
from models.gnn.projection_heads import ProjectionHead, ComponentProjectionHeads
from models.llm.llm_interface import LLMInterface
from models.graph.graph_builder import GraphBuilder
from models.graph.heterogeneous_graph import HeterogeneousGraph

# Import evaluation
from evaluation.evaluator import Evaluator

# Try to import torch for GPU support
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader as TorchDataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class Phase2Trainer:
    """
    Phase 2 Trainer for H-GRAGrecsys - Distillation from LLM to GNN.
    
    Features:
    - Teacher-student distillation
    - Component-wise loss computation
    - Representation alignment
    - Memory dynamics transfer
    - Path importance distillation
    - Contrastive learning
    - Checkpoint saving and loading
    - Progress monitoring and visualization
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        teacher_path: Optional[Union[str, Path]] = None,
        student_path: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        resume_from: Optional[Union[str, Path]] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        logger: Optional['Logger'] = None,
        verbose: bool = True
    ):
        """
        Initialize the Phase 2 Trainer.
        
        Args:
            config_path (str, Path, optional): Path to configuration file
            teacher_path (str, Path, optional): Path to Phase 1 teacher model
            student_path (str, Path, optional): Path to student model (GNN)
            checkpoint_dir (str, Path, optional): Directory to save checkpoints
            output_dir (str, Path, optional): Directory for outputs
            resume_from (str, Path, optional): Checkpoint to resume from
            seed (int, optional): Random seed for reproducibility
            device (str, optional): Device to use ('cpu', 'cuda')
            logger (Logger, optional): Logger instance
            verbose (bool): Whether to enable verbose output
        
        Example:
            trainer = Phase2Trainer(
                config_path='config/default_config.yaml',
                teacher_path='experiments/phase1/checkpoints/phase1_best.pt',
                output_dir='experiments/phase2'
            )
            trainer.train()
        """
        # Setup paths
        self.config_path = Path(config_path) if config_path else None
        self.teacher_path = Path(teacher_path) if teacher_path else None
        self.student_path = Path(student_path) if student_path else None
        self.output_dir = Path(output_dir) if output_dir else Path("experiments/phase2")
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
                name="phase2_trainer",
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
            name="phase2_training",
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
        self.phase2 = None
        self.dataset = None
        self.data_loader = None
        self.teacher_model = None
        self.student_model = None
        self.distillation_trainer = None
        self.knowledge_distiller = None
        self.distillation_loss = None
        self.disentangler = None
        
        # Training state
        self.state = {
            'epoch': 0,
            'best_metrics': {},
            'training_completed': False,
            'distillation_stats': {}
        }
        
        self.logger.log_info("Phase2Trainer initialized")
        self.logger.log_info(f"Output directory: {self.output_dir}")
        self.logger.log_info(f"Checkpoint directory: {self.checkpoint_dir}")
        self.logger.log_info(f"Teacher path: {self.teacher_path}")
        self.logger.log_info(f"Student path: {self.student_path}")
    
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
        Run Phase 2 distillation training.
        
        Returns:
            Dict[str, Any]: Training results and metrics
        
        Example:
            results = trainer.train()
            print(f"Distillation completed: {results['epochs_completed']} epochs")
        """
        self.logger.log_info("=" * 80)
        self.logger.log_info("PHASE 2: Distillation from LLM to GNN")
        self.logger.log_info("=" * 80)
        
        with self.timer.measure("phase2_training"):
            # Step 1: Load data
            self._load_data()
            
            # Step 2: Load teacher model (Phase 1)
            self._load_teacher_model()
            
            # Step 3: Initialize student model (GNN)
            self._initialize_student_model()
            
            # Step 4: Initialize distillation components
            self._initialize_distillation()
            
            # Step 5: Resume if requested
            if self.resume_from:
                self._resume_training()
            
            # Step 6: Run training
            results = self._run_training()
            
            # Step 7: Save final model
            self._save_final_model()
            
            # Step 8: Generate summary
            self._generate_summary(results)
        
        self.logger.log_info("=" * 80)
        self.logger.log_info("Phase 2 Distillation Completed")
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
            
            # Create data loader
            batch_size = self.config.get('training', {}).get('phase2', {}).get('batch_size', 64)
            self.data_loader = DataLoader(
                dataset=self.dataset,
                batch_size=batch_size,
                shuffle=True
            )
            
            self.logger.log_info(f"Batch size: {batch_size}")
            self.logger.log_info(f"Number of batches: {len(self.data_loader.get_train_batches())}")
    
    def _load_teacher_model(self) -> None:
        """
        Load teacher model from Phase 1 checkpoint.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("LOADING TEACHER MODEL")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("load_teacher"):
            if not self.teacher_path or not self.teacher_path.exists():
                self.logger.log_warning("Teacher model not found. Creating default teacher...")
                self._create_default_teacher()
                return
            
            try:
                # Load checkpoint
                with open(self.teacher_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                
                # Extract teacher components
                self.teacher_agents = checkpoint.get('state', {}).get('user_agents', {})
                self.teacher_item_agents = checkpoint.get('state', {}).get('item_agents', {})
                self.teacher_graph = checkpoint.get('state', {}).get('graph', None)
                self.teacher_reflections = checkpoint.get('reflection_traces', [])
                
                # Create LLM interface as teacher
                model_name = self.config.get('llm', {}).get('model_name', 'gpt-3.5-turbo')
                self.teacher_model = LLMInterface(
                    model_name=model_name,
                    config=self.config
                )
                
                self.logger.log_info(f"Teacher model loaded from: {self.teacher_path}")
                self.logger.log_info(f"Teacher has {len(self.teacher_agents)} user agents")
                self.logger.log_info(f"Teacher has {len(self.teacher_reflections)} reflections")
                
            except Exception as e:
                self.logger.log_error(f"Failed to load teacher model: {e}")
                self.logger.log_info("Creating default teacher...")
                self._create_default_teacher()
    
    def _create_default_teacher(self) -> None:
        """
        Create a default teacher model.
        """
        # Create LLM interface as teacher
        model_name = self.config.get('llm', {}).get('model_name', 'gpt-3.5-turbo')
        self.teacher_model = LLMInterface(
            model_name=model_name,
            config=self.config
        )
        
        # Create default agents
        self.teacher_agents = {}
        self.teacher_item_agents = {}
        
        # Get users and items
        users = list(self.dataset.get_user_items().keys())
        items = list(self.dataset.get_item_features().keys())
        
        # Create user agents
        for user_id in users[:100]:  # Limit for default
            self.teacher_agents[user_id] = UserAgent(user_id, self.config)
        
        # Create item agents
        for item_id in items[:100]:  # Limit for default
            self.teacher_item_agents[item_id] = ItemAgent(item_id, self.config)
        
        self.teacher_reflections = []
        
        self.logger.log_info("Default teacher model created")
    
    def _initialize_student_model(self) -> None:
        """
        Initialize student model (GNN).
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("INITIALIZING STUDENT MODEL")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("initialize_student"):
            # Get GNN config
            gnn_config = self.config.get('model', {}).get('gnn', {})
            hidden_dim = gnn_config.get('hidden_dim', 256)
            num_layers = gnn_config.get('num_layers', 3)
            num_heads = gnn_config.get('num_heads', 4)
            dropout = gnn_config.get('dropout', 0.1)
            
            # Initialize GNN encoder
            self.student_model = GNNEncoder(self.config)
            
            # Initialize heterogeneous GNN
            self.student_gnn = HeterogeneousGNN(self.config)
            
            # Initialize projection heads
            self.projection_heads = ComponentProjectionHeads(
                input_dim=hidden_dim,
                config=self.config
            )
            
            # Load student checkpoint if provided
            if self.student_path and self.student_path.exists():
                self.logger.log_info(f"Loading student model from: {self.student_path}")
                with open(self.student_path, 'rb') as f:
                    student_checkpoint = pickle.load(f)
                    # Load student state
                    if 'model_state' in student_checkpoint:
                        self.student_model.load_state_dict(student_checkpoint['model_state'])
            
            self.logger.log_info(f"Student GNN initialized: hidden_dim={hidden_dim}, layers={num_layers}")
            self.logger.log_info(f"Student has {self._count_parameters(self.student_model):,} parameters")
    
    def _count_parameters(self, model) -> int:
        """
        Count trainable parameters in a model.
        
        Args:
            model: PyTorch model
            
        Returns:
            int: Number of parameters
        """
        if TORCH_AVAILABLE and hasattr(model, 'parameters'):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return 0
    
    def _initialize_distillation(self) -> None:
        """
        Initialize distillation components.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("INITIALIZING DISTILLATION COMPONENTS")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("initialize_distillation"):
            # Get distillation config
            distill_config = self.config.get('model', {}).get('distillation', {})
            alpha = distill_config.get('alpha', 0.5)
            beta = distill_config.get('beta', 0.3)
            gamma = distill_config.get('gamma', 0.2)
            temperature = distill_config.get('temperature', 0.07)
            component_weights = distill_config.get('component_weights', [1.0, 1.0, 1.0])
            
            # Initialize distillation loss
            self.distillation_loss = DistillationLoss(self.config)
            
            # Initialize knowledge distiller
            self.knowledge_distiller = KnowledgeDistiller(
                teacher_llm=self.teacher_model,
                student_gnn=self.student_model,
                config=self.config
            )
            
            # Initialize component disentangler
            self.disentangler = ComponentDisentangler(self.config)
            
            # Initialize Phase 2
            self.phase2 = Phase2Distillation(
                teachers=[self.teacher_model],
                student_graph=self._get_student_graph(),
                config=self.config,
                logger=self.logger,
                seed_manager=self.seed_manager
            )
            
            # Initialize distillation trainer
            self.distillation_trainer = DistillationTrainer(
                teacher=self.teacher_model,
                student=self.student_model,
                config=self.config
            )
            
            self.logger.log_info(f"Distillation alpha: {alpha}, beta: {beta}, gamma: {gamma}")
            self.logger.log_info(f"Temperature: {temperature}")
            self.logger.log_info("Distillation components initialized")
    
    def _get_student_graph(self) -> HeterogeneousGraph:
        """
        Get student graph for distillation.
        
        Returns:
            HeterogeneousGraph: Student graph
        """
        # Build graph from dataset
        graph_builder = GraphBuilder(self.config)
        
        # Get agents
        agents = []
        if hasattr(self, 'teacher_agents'):
            agents.extend(list(self.teacher_agents.values()))
        if hasattr(self, 'teacher_item_agents'):
            agents.extend(list(self.teacher_item_agents.values()))
        
        # Build graph
        graph = graph_builder.build_graph(
            agents=agents if agents else self._create_sample_agents(),
            interactions=self.dataset.get_interactions()
        )
        
        return graph
    
    def _create_sample_agents(self):
        """Create sample agents for graph building."""
        agents = []
        users = list(self.dataset.get_user_items().keys())[:50]
        items = list(self.dataset.get_item_features().keys())[:50]
        
        for user_id in users:
            agents.append(UserAgent(user_id, self.config))
        
        for item_id in items:
            agents.append(ItemAgent(item_id, self.config))
        
        return agents
    
    def _run_training(self) -> Dict[str, Any]:
        """
        Run the Phase 2 distillation training loop.
        
        Returns:
            Dict[str, Any]: Training results
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("RUNNING DISTILLATION TRAINING")
        self.logger.log_info("-" * 50)
        
        # Get training parameters
        train_config = self.config.get('training', {}).get('phase2', {})
        num_epochs = train_config.get('num_epochs', 30)
        learning_rate = train_config.get('learning_rate', 1e-4)
        
        self.logger.log_info(f"Number of epochs: {num_epochs}")
        self.logger.log_info(f"Learning rate: {learning_rate}")
        
        # Prepare training data from reflection traces
        training_data = self.phase2.prepare_training_data(
            self.teacher_reflections if hasattr(self, 'teacher_reflections') else []
        )
        
        self.logger.log_info(f"Training data prepared: {len(training_data)} samples")
        
        # Training loop
        training_results = {
            'epochs_completed': 0,
            'metrics_history': [],
            'loss_history': [],
            'best_metrics': {},
            'distillation_stats': {}
        }
        
        for epoch in range(self.start_epoch, num_epochs):
            self.logger.log_info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            with self.timer.measure(f"epoch_{epoch}"):
                # Train one epoch
                epoch_loss, epoch_metrics = self._train_epoch(epoch, training_data)
                
                # Store metrics
                training_results['metrics_history'].append(epoch_metrics)
                training_results['loss_history'].append(epoch_loss)
                training_results['epochs_completed'] = epoch + 1
                
                # Update best metrics
                if not training_results['best_metrics'] or epoch_metrics.get('ndcg@10', 0) > training_results['best_metrics'].get('ndcg@10', 0):
                    training_results['best_metrics'] = epoch_metrics
                    self._save_checkpoint(epoch, epoch_metrics, epoch_loss, is_best=True)
                
                # Save checkpoint
                if (epoch + 1) % train_config.get('save_interval', 5) == 0:
                    self._save_checkpoint(epoch, epoch_metrics, epoch_loss, is_best=False)
                
                # Log progress
                self._log_epoch_metrics(epoch, epoch_loss, epoch_metrics)
                
                # Early stopping
                if self._check_early_stopping(training_results['metrics_history']):
                    self.logger.log_info("Early stopping triggered")
                    break
        
        # Get distillation statistics
        if self.knowledge_distiller:
            training_results['distillation_stats'] = self.knowledge_distiller.get_distillation_stats()
        
        self.logger.log_info(f"\nDistillation completed: {training_results['epochs_completed']} epochs")
        self.logger.log_info(f"Best NDCG@10: {training_results['best_metrics'].get('ndcg@10', 0):.4f}")
        
        return training_results
    
    def _train_epoch(self, epoch: int, training_data: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            epoch (int): Current epoch number
            training_data (List[Dict[str, Any]]): Training data
            
        Returns:
            Tuple[float, Dict[str, float]]: (Loss, Metrics)
        """
        # Shuffle training data
        shuffled_data = self.seed_manager.shuffle(
            training_data,
            operation_name=f'epoch_{epoch}_shuffle'
        )
        
        total_loss = 0.0
        batch_count = 0
        batch_losses = []
        
        # Get batch size
        batch_size = self.config.get('training', {}).get('phase2', {}).get('batch_size', 64)
        
        # Process batches
        for i in range(0, len(shuffled_data), batch_size):
            batch = shuffled_data[i:i + batch_size]
            
            # Run distillation step
            loss, stats = self.knowledge_distiller.distill_knowledge(batch)
            
            total_loss += loss
            batch_losses.append(loss)
            batch_count += 1
            
            # Log batch progress
            if (i + 1) % (batch_size * 10) == 0:
                self.logger.log_debug(f"  Batch {batch_count}/{len(shuffled_data)//batch_size + 1}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        
        # Evaluate after epoch
        eval_metrics = self._evaluate()
        
        # Log distillation loss components
        if hasattr(self.distillation_loss, 'get_last_loss_components'):
            loss_components = self.distillation_loss.get_last_loss_components()
            self.logger.log_debug(f"Loss components: {loss_components}")
        
        return avg_loss, eval_metrics
    
    def _evaluate(self) -> Dict[str, float]:
        """
        Evaluate current distilled model.
        
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
        Get current distilled model.
        
        Returns:
            HybridInferenceEngine: Current model
        """
        from models.hybrid.inference_engine import HybridInferenceEngine
        from models.hybrid.adaptive_gate import AdaptiveGate
        
        gate = AdaptiveGate(self.config)
        
        return HybridInferenceEngine(
            gnn_encoder=self.student_model,
            llm_interface=self.teacher_model,
            gate=gate,
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
    
    def _log_epoch_metrics(self, epoch: int, loss: float, metrics: Dict[str, float]) -> None:
        """
        Log epoch metrics.
        
        Args:
            epoch (int): Epoch number
            loss (float): Epoch loss
            metrics (Dict[str, float]): Epoch metrics
        """
        self.logger.log_info(f"Epoch {epoch + 1} Results:")
        self.logger.log_info(f"  Loss: {loss:.4f}")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.logger.log_info(f"  {key}: {value:.4f}")
        
        # Plot training progress
        if hasattr(self, 'visualizer'):
            # This would be updated with actual plotting
            pass
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], loss: float, is_best: bool = False) -> None:
        """
        Save distillation checkpoint.
        
        Args:
            epoch (int): Current epoch
            metrics (Dict[str, float]): Evaluation metrics
            loss (float): Training loss
            is_best (bool): Whether this is the best model
        """
        checkpoint = {
            'epoch': epoch,
            'epochs_completed': epoch + 1,
            'loss': loss,
            'metrics': metrics,
            'model_state': self.student_model.state_dict() if TORCH_AVAILABLE else None,
            'gnn_state': self.student_gnn.state_dict() if TORCH_AVAILABLE else None,
            'projection_heads': self.projection_heads.state_dict() if TORCH_AVAILABLE else None,
            'config': self.config,
            'distillation_stats': self.knowledge_distiller.get_distillation_stats() if self.knowledge_distiller else {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Save checkpoint
        if is_best:
            filename = "phase2_best.pt"
        else:
            filename = f"phase2_epoch_{epoch+1:03d}.pt"
        
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
                'loss': loss,
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
        
        # Restore model states
        if TORCH_AVAILABLE:
            if 'model_state' in checkpoint and self.student_model:
                self.student_model.load_state_dict(checkpoint['model_state'])
            if 'gnn_state' in checkpoint and self.student_gnn:
                self.student_gnn.load_state_dict(checkpoint['gnn_state'])
            if 'projection_heads' in checkpoint and self.projection_heads:
                self.projection_heads.load_state_dict(checkpoint['projection_heads'])
        
        self.logger.log_info(f"Resuming from epoch {self.start_epoch}")
        self.logger.log_info(f"Best metrics: {checkpoint.get('metrics', {})}")
    
    def _save_final_model(self) -> None:
        """
        Save final distilled model.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("SAVING FINAL DISTILLED MODEL")
        self.logger.log_info("-" * 50)
        
        # Save distilled model
        model_path = self.checkpoint_dir / "distilled_model.pt"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'student_model': self.student_model,
                'student_gnn': self.student_gnn,
                'projection_heads': self.projection_heads,
                'config': self.config,
                'distillation_stats': self.knowledge_distiller.get_distillation_stats() if self.knowledge_distiller else {},
                'timestamp': datetime.now().isoformat()
            }, f)
        
        self.logger.log_info(f"Distilled model saved to: {model_path}")
        
        # Save as PyTorch model if available
        if TORCH_AVAILABLE:
            torch_path = self.checkpoint_dir / "distilled_model.pt"
            torch.save({
                'model_state_dict': self.student_model.state_dict(),
                'gnn_state_dict': self.student_gnn.state_dict(),
                'projection_heads_state_dict': self.projection_heads.state_dict(),
                'config': self.config
            }, torch_path)
            self.logger.log_info(f"PyTorch model saved to: {torch_path}")
    
    def _generate_summary(self, results: Dict[str, Any]) -> None:
        """
        Generate training summary.
        
        Args:
            results (Dict[str, Any]): Training results
        """
        summary_path = self.output_dir / "training_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PHASE 2 DISTILLATION SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write(f"Seed: {self.seed}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Epochs completed: {results.get('epochs_completed', 0)}\n")
            f.write("\n")
            
            f.write("DISTILLATION STATISTICS\n")
            f.write("-" * 40 + "\n")
            stats = results.get('distillation_stats', {})
            for key, value in stats.items():
                f.write(f"  {key}: {value}\n")
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
            
            f.write("LOSS HISTORY\n")
            f.write("-" * 40 + "\n")
            loss_history = results.get('loss_history', [])
            if loss_history:
                f.write(f"  Initial loss: {loss_history[0]:.4f}\n")
                f.write(f"  Final loss: {loss_history[-1]:.4f}\n")
                if len(loss_history) > 1:
                    improvement = (loss_history[0] - loss_history[-1]) / loss_history[0] * 100
                    f.write(f"  Improvement: {improvement:.1f}%\n")
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
    
    def evaluate_distillation(self) -> Dict[str, Any]:
        """
        Evaluate the distillation quality.
        
        Returns:
            Dict[str, Any]: Evaluation results
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("EVALUATING DISTILLATION")
        self.logger.log_info("-" * 50)
        
        if not self.phase2:
            self.logger.log_error("Phase 2 not initialized")
            return {}
        
        results = self.phase2.evaluate_distillation()
        
        self.logger.log_info("Distillation evaluation completed")
        
        return results


def main():
    """
    Main entry point for Phase 2 training.
    """
    parser = argparse.ArgumentParser(description="H-GRAGrecsys Phase 2 Distillation Script")
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--teacher-path',
        type=str,
        default=None,
        help='Path to Phase 1 teacher model checkpoint'
    )
    
    parser.add_argument(
        '--student-path',
        type=str,
        default=None,
        help='Path to student model checkpoint'
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
    
    parser.add_argument(
        '--distillation-params',
        type=str,
        default=None,
        help='JSON string of distillation parameters to override'
    )
    
    args = parser.parse_args()
    
    # Override distillation parameters if provided
    if args.distillation_params:
        import json
        overrides = json.loads(args.distillation_params)
        # Apply overrides to config
        # This would be implemented in the trainer
    
    # Create trainer
    trainer = Phase2Trainer(
        config_path=args.config,
        teacher_path=args.teacher_path,
        student_path=args.student_path,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        seed=args.seed,
        device=args.device,
        verbose=not args.no_verbose
    )
    
    # Run training or evaluation
    if args.eval_only:
        results = trainer.evaluate_distillation()
        print(f"Distillation evaluation results: {results}")
    else:
        results = trainer.train()
        
        # Print summary
        print("\n" + "=" * 40)
        print("Distillation completed!")
        print(f"Best NDCG@10: {results['best_metrics'].get('ndcg@10', 0):.4f}")
        print(f"Best Hit Rate: {results['best_metrics'].get('hit_rate', 0):.4f}")
        print(f"Checkpoints saved in: {trainer.checkpoint_dir}")
        print("=" * 40 + "\n")
    
    return results


if __name__ == "__main__":
    main()