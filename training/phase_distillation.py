"""
Phase 2: Distillation Training Module for H-GRAGrecsys

This module implements the knowledge distillation phase where insights from
LLM teachers are transferred to a GNN student model. The distillation process
includes component-wise knowledge transfer, path importance learning, and
contrastive learning to align representations between teacher and student.

Key Responsibilities:
- Prepare training data from reflection traces
- Train GNN student through knowledge distillation
- Align teacher and student representations
- Distill memory dynamics and collaborative signals
- Save and load distilled models
"""

import os
import sys
import json
import math
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle
import random

# Add project root to path if needed
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Core imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# Data imports
from data.dataset import BaseDataset, AmazonDataset
from data.data_loader import DataLoader as DataLoaderClass

# Agent imports
from models.agent.user_agent import UserAgent
from models.agent.item_agent import ItemAgent
from models.agent.memory import AgentMemory, HierarchicalMemory

# Graph imports
from models.graph.heterogeneous_graph import HeterogeneousGraph
from models.graph.graph_builder import GraphBuilder
from models.graph.relation_types import RelationType

# Graph RAG imports
from models.graph_rag.retriever import GraphRAGRetriever
from models.graph_rag.metapath_extractor import MetapathExtractor
from models.graph_rag.context_constructor import ContextConstructor

# LLM imports
from models.llm.llm_interface import LLMInterface
from models.llm.text_encoder import TextEncoder

# GNN imports
from models.gnn.heterogeneous_gnn import HeterogeneousGNN
from models.gnn.projection_heads import ComponentProjectionHeads
from models.gnn.gnn_encoder import GNNEncoder
from models.gnn.attention_module import AttentionModule

# Distillation imports
from distillation.distillation_trainer import DistillationTrainer
from distillation.loss_functions import DistillationLoss
from distillation.knowledge_distiller import KnowledgeDistiller
from distillation.component_disentangler import ComponentDisentangler

# Utils imports
from utils.logger import Logger
from utils.config_loader import ConfigLoader
from utils.seed_manager import SeedManager
from utils.timer import Timer

# Training imports
from .trainer_base import BaseTrainer
from .checkpoint_manager import CheckpointManager


class Phase2Distillation(BaseTrainer):
    """
    Phase 2 Distillation Trainer
    
    Distills knowledge from LLM teachers to a GNN student model through
    multiple distillation strategies including component-wise loss,
    path importance loss, and contrastive learning.
    """
    
    def __init__(
        self,
        teachers: Union[LLMInterface, List[LLMInterface]],
        student_graph: HeterogeneousGNN,
        config: Dict[str, Any],
        dataset: Optional[BaseDataset] = None,
        graph: Optional[HeterogeneousGraph] = None,
        reflection_traces: Optional[List[Dict[str, Any]]] = None,
        knowledge_distiller: Optional[KnowledgeDistiller] = None,
        component_disentangler: Optional[ComponentDisentangler] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        logger: Optional[Logger] = None
    ):
        """
        Initialize Phase 2 Distillation Trainer
        
        Args:
            teachers: LLMInterface or list of LLMInterface instances as teachers
            student_graph: HeterogeneousGNN model to be trained through distillation
            config: Configuration dictionary containing phase2 settings
            dataset: Optional dataset for training data
            graph: Optional heterogeneous graph
            reflection_traces: Optional reflection traces from Phase 1
            knowledge_distiller: Optional KnowledgeDistiller instance
            component_disentangler: Optional ComponentDisentangler instance
            checkpoint_manager: Optional CheckpointManager instance
            logger: Optional Logger instance
            
        Raises:
            ValueError: If required parameters are missing
        """
        super().__init__(config, model=student_graph, data_loader=None)
        
        if teachers is None:
            raise ValueError("Teachers cannot be None for Phase 2 Distillation")
        
        if student_graph is None:
            raise ValueError("Student graph model cannot be None for Phase 2 Distillation")
        
        # Store core components
        self.teachers = teachers if isinstance(teachers, list) else [teachers]
        self.student_graph = student_graph
        self.config = config
        self.dataset = dataset
        self.graph = graph
        self.reflection_traces = reflection_traces or []
        
        # Extract phase-specific configuration
        self.phase_config = config.get('phase2', {})
        if not self.phase_config:
            from . import PHASE2_CONFIG
            self.phase_config = PHASE2_CONFIG.copy()
        
        # Initialize components
        self.logger = logger or Logger(
            log_dir=config.get('common', {}).get('log_dir', './logs'),
            name='phase2_distillation'
        )
        
        # Initialize text encoder for embeddings
        self.text_encoder = TextEncoder(
            model_name=config.get('llm', {}).get('encoder_model', 'sentence-transformers/all-MiniLM-L6-v2'),
            config=config
        )
        
        # Initialize distillation components
        self.distillation_loss = DistillationLoss(config=self.phase_config)
        self.knowledge_distiller = knowledge_distiller or KnowledgeDistiller(
            teacher_llm=self.teachers[0],
            student_gnn=self.student_graph,
            config=self.phase_config
        )
        
        self.component_disentangler = component_disentangler or ComponentDisentangler(
            config=self.phase_config
        )
        
        # Initialize projection heads if not present
        if not hasattr(self.student_graph, 'projection_heads'):
            input_dim = config.get('gnn', {}).get('hidden_dim', 256)
            self.student_graph.projection_heads = ComponentProjectionHeads(
                input_dim=input_dim,
                config=self.phase_config
            )
        
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            save_dir=config.get('common', {}).get('checkpoint_dir', './checkpoints'),
            max_checkpoints=config.get('common', {}).get('max_checkpoints', 5)
        )
        
        # Initialize graph components if needed
        self.graph_retriever = None
        self.metapath_extractor = None
        self.context_constructor = None
        
        if self.graph is not None:
            self.graph_retriever = GraphRAGRetriever(
                graph=self.graph,
                config=config.get('graph_rag', {})
            )
            self.metapath_extractor = MetapathExtractor(
                config=config.get('graph_rag', {})
            )
            self.context_constructor = ContextConstructor(
                config=config.get('graph_rag', {})
            )
        
        # Training state
        self.training_data: Optional[List[Dict[str, Any]]] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        
        # Metrics tracking
        self.metrics = {
            'epoch': 0,
            'step': 0,
            'train_losses': [],
            'val_losses': [],
            'component_losses': defaultdict(list),
            'distillation_accuracy': 0.0,
            'representation_alignment': 0.0,
            'memory_distillation_score': 0.0,
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }
        
        # Set device
        self.device = config.get('common', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.student_graph.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        
        self.logger.log_info(f"Phase2Distillation initialized on device: {self.device}")
        self.logger.log_info(f"Phase config: {self.phase_config}")
        self.logger.log_info(f"Number of teachers: {len(self.teachers)}")
    
    def prepare_training_data(
        self,
        reflection_traces: Optional[List[Dict[str, Any]]] = None,
        use_graph_samples: bool = True,
        num_negatives: int = 99
    ) -> Dict[str, Any]:
        """
        Prepare training data from reflection traces and graph
        
        This method creates structured training data by:
        1. Extracting positive and negative samples from reflection traces
        2. Generating metapath instances for graph context
        3. Creating teacher embeddings and student targets
        4. Splitting data into train/val/test sets
        
        Args:
            reflection_traces: Optional list of reflection traces.
                             If None, uses stored traces.
            use_graph_samples: Whether to sample from graph for additional data
            num_negatives: Number of negative samples per positive pair
        
        Returns:
            Dict[str, Any]: Training data statistics
            {
                'num_samples': int,
                'num_train': int,
                'num_val': int,
                'num_test': int,
                'data_path': str,
                'sample_ratio': float
            }
            
        Raises:
            RuntimeError: If no training data can be prepared
        """
        self.logger.log_info("Preparing training data for distillation...")
        
        timer = Timer()
        timer.start()
        
        # Use provided traces or stored traces
        traces = reflection_traces or self.reflection_traces
        
        if not traces:
            raise RuntimeError("No reflection traces available for training data preparation")
        
        self.logger.log_info(f"Processing {len(traces)} reflection traces...")
        
        # Prepare training samples
        training_samples = []
        
        for trace in tqdm(traces, desc="Creating training samples"):
            try:
                user_id = trace.get('user_id')
                item_id = trace.get('item_id')
                reflection = trace.get('reflection', {})
                
                # Get user and item embeddings from reflection
                user_embedding = torch.tensor(
                    trace.get('user_embedding', np.zeros(1536)),
                    dtype=torch.float32
                )
                item_embedding = torch.tensor(
                    trace.get('item_embedding', np.zeros(1536)),
                    dtype=torch.float32
                )
                
                # Extract component-wise knowledge
                component_knowledge = self._extract_component_knowledge(reflection)
                
                # Generate metapath instances if graph is available
                metapath_context = []
                if use_graph_samples and self.graph is not None:
                    metapath_context = self._generate_metapath_context(user_id, item_id)
                
                # Create training sample
                sample = {
                    'user_id': user_id,
                    'item_id': item_id,
                    'user_embedding': user_embedding,
                    'item_embedding': item_embedding,
                    'component_knowledge': component_knowledge,
                    'reflection': reflection,
                    'metapath_context': metapath_context,
                    'label': 1.0  # Positive sample
                }
                
                training_samples.append(sample)
                
                # Generate negative samples
                negative_samples = self._generate_negative_samples(
                    user_id=user_id,
                    item_id=item_id,
                    num_negatives=num_negatives,
                    traces=traces
                )
                training_samples.extend(negative_samples)
                
            except Exception as e:
                self.logger.log_warning(f"Failed to create sample from trace: {e}")
                continue
        
        if not training_samples:
            raise RuntimeError("No training samples could be created")
        
        # Shuffle samples
        random.shuffle(training_samples)
        
        # Split data
        train_ratio = self.phase_config.get('train_ratio', 0.7)
        val_ratio = self.phase_config.get('val_ratio', 0.15)
        test_ratio = self.phase_config.get('test_ratio', 0.15)
        
        num_samples = len(training_samples)
        num_train = int(num_samples * train_ratio)
        num_val = int(num_samples * val_ratio)
        num_test = num_samples - num_train - num_val
        
        train_data = training_samples[:num_train]
        val_data = training_samples[num_train:num_train + num_val]
        test_data = training_samples[num_train + num_val:]
        
        # Store training data
        self.training_data = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        # Create data loaders
        self.train_loader = self._create_data_loader(train_data, shuffle=True)
        self.val_loader = self._create_data_loader(val_data, shuffle=False)
        self.test_loader = self._create_data_loader(test_data, shuffle=False)
        
        timer.stop()
        
        stats = {
            'num_samples': num_samples,
            'num_train': len(train_data),
            'num_val': len(val_data),
            'num_test': len(test_data),
            'data_path': f"./data/distillation_data_{Timer.get_current_timestamp()}.pkl",
            'sample_ratio': train_ratio,
            'elapsed_time': timer.get_elapsed_time()
        }
        
        # Save training data for future use
        self._save_training_data(stats['data_path'])
        
        self.logger.log_info(f"Training data preparation completed: {stats}")
        return stats
    
    def train_distillation(
        self,
        num_epochs: Optional[int] = None,
        validate_every: int = 1,
        save_every: int = 5
    ) -> Dict[str, Any]:
        """
        Train the student model through knowledge distillation
        
        Args:
            num_epochs: Number of epochs to train. If None, uses config value.
            validate_every: Validate every N epochs
            save_every: Save checkpoint every N epochs
        
        Returns:
            Dict[str, Any]: Training metrics and statistics
            
        Raises:
            RuntimeError: If training data is not prepared
            RuntimeError: If training fails
        """
        if self.train_loader is None:
            raise RuntimeError("Training data not prepared. Call prepare_training_data() first.")
        
        self.logger.log_info("Starting distillation training...")
        
        num_epochs = num_epochs or self.phase_config.get('num_epochs', 30)
        
        total_timer = Timer()
        total_timer.start()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Training loop
        for epoch in range(num_epochs):
            self.metrics['epoch'] = epoch
            
            # Train one epoch
            train_loss, train_metrics = self._train_epoch(epoch)
            
            # Log training metrics
            self.logger.log_info(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}")
            
            # Validate
            if (epoch + 1) % validate_every == 0:
                val_loss, val_metrics = self._validate_epoch(epoch)
                self.metrics['val_losses'].append(val_loss)
                
                self.logger.log_info(f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.metrics['best_val_loss']:
                    self.metrics['best_val_loss'] = val_loss
                    self.metrics['best_epoch'] = epoch
                    self._save_best_model()
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_distilled_model(
                    checkpoint_name=f"distilled_epoch_{epoch + 1}"
                )
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    if (epoch + 1) % validate_every == 0:
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
        
        # Final evaluation
        final_metrics = self.evaluate_distillation()
        
        total_timer.stop()
        
        # Prepare final results
        results = {
            'train_losses': self.metrics['train_losses'],
            'val_losses': self.metrics['val_losses'],
            'best_val_loss': self.metrics['best_val_loss'],
            'best_epoch': self.metrics['best_epoch'],
            'final_metrics': final_metrics,
            'total_time': total_timer.get_elapsed_time(),
            'component_losses': dict(self.metrics['component_losses'])
        }
        
        self.logger.log_info(f"Distillation training completed in {total_timer.get_elapsed_time():.2f} seconds")
        self.logger.log_info(f"Best validation loss: {self.metrics['best_val_loss']:.4f} at epoch {self.metrics['best_epoch']}")
        
        return results
    
    def evaluate_distillation(
        self,
        test_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the distilled model on test data
        
        Args:
            test_data: Optional test data. If None, uses stored test loader.
        
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        self.logger.log_info("Evaluating distillation model...")
        
        timer = Timer()
        timer.start()
        
        # Use provided test data or stored loader
        loader = test_data if test_data is not None else self.test_loader
        
        if loader is None:
            self.logger.log_warning("No test data available for evaluation")
            return {}
        
        eval_metrics = {
            'accuracy': 0.0,
            'component_alignment': {},
            'distillation_quality': {},
            'memory_transfer_score': 0.0
        }
        
        # Switch to evaluation mode
        self.student_graph.eval()
        
        total_loss = 0.0
        num_batches = 0
        component_alignments = defaultdict(float)
        distillation_scores = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                # Prepare batch
                batch = self._prepare_batch(batch)
                
                # Forward pass through student
                student_outputs = self.student_graph.forward(
                    graph=batch['graph'] if 'graph' in batch else None,
                    node_features=batch['node_features']
                )
                
                # Get teacher outputs (simulated)
                teacher_outputs = self._simulate_teacher_outputs(batch)
                
                # Compute losses for evaluation
                component_loss = self.distillation_loss.component_wise_loss(
                    student=student_outputs,
                    teacher=teacher_outputs,
                    component_type='all'
                )
                
                path_loss = self.distillation_loss.path_importance_loss(
                    student_attn=batch.get('student_attn', None),
                    teacher_attn=batch.get('teacher_attn', None)
                )
                
                contrastive_loss = self.distillation_loss.contrastive_loss(
                    embeddings=batch.get('embeddings', None),
                    temperature=self.phase_config.get('temperature', 0.07)
                )
                
                total_loss += (component_loss + path_loss + contrastive_loss).item()
                num_batches += 1
                
                # Track component alignments
                for comp_type in ['intrinsic', 'collaborative', 'interaction']:
                    align_score = self._compute_component_alignment(
                        student_outputs,
                        teacher_outputs,
                        comp_type
                    )
                    component_alignments[comp_type] += align_score
                
                # Evaluate distillation quality
                quality_score = self._evaluate_distillation_quality(
                    student_outputs,
                    teacher_outputs,
                    batch
                )
                distillation_scores.append(quality_score)
        
        # Compute average metrics
        eval_metrics['accuracy'] = 1.0 - (total_loss / num_batches if num_batches > 0 else 1.0)
        
        for comp_type in component_alignments:
            eval_metrics['component_alignment'][comp_type] = (
                component_alignments[comp_type] / num_batches if num_batches > 0 else 0.0
            )
        
        eval_metrics['distillation_quality']['avg_score'] = np.mean(distillation_scores) if distillation_scores else 0.0
        eval_metrics['distillation_quality']['std_score'] = np.std(distillation_scores) if distillation_scores else 0.0
        
        # Compute memory transfer score
        eval_metrics['memory_transfer_score'] = self._compute_memory_transfer_score()
        
        timer.stop()
        eval_metrics['elapsed_time'] = timer.get_elapsed_time()
        
        self.logger.log_info(f"Evaluation completed: {eval_metrics}")
        
        return eval_metrics
    
    def save_distilled_model(
        self,
        checkpoint_name: Optional[str] = None,
        save_dir: Optional[str] = None
    ) -> str:
        """
        Save the distilled GNN model and training state
        
        Args:
            checkpoint_name: Optional name for the checkpoint
            save_dir: Optional directory to save checkpoint
        
        Returns:
            str: Path to saved checkpoint
        
        Raises:
            RuntimeError: If saving fails
        """
        self.logger.log_info("Saving distilled model...")
        
        timer = Timer()
        timer.start()
        
        try:
            # Prepare state dictionary
            state = {
                'model_state_dict': self.student_graph.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'metrics': self.metrics,
                'config': self.config,
                'phase': 'phase2_distillation',
                'timestamp': Timer.get_current_timestamp(),
                'model_architecture': str(self.student_graph)
            }
            
            # Save using checkpoint manager
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                state=state,
                epoch=self.metrics['epoch'],
                step=self.metrics['step'],
                name=checkpoint_name
            )
            
            # Also save in training format for later loading
            model_save_path = os.path.join(
                save_dir or self.checkpoint_manager.save_dir,
                f"distilled_model_{Timer.get_current_timestamp()}.pt"
            )
            torch.save(state, model_save_path)
            
            timer.stop()
            self.logger.log_info(f"Distilled model saved to {checkpoint_path} in {timer.get_elapsed_time():.2f} seconds")
            
            return checkpoint_path
            
        except Exception as e:
            self.logger.log_error(f"Failed to save distilled model: {e}")
            raise RuntimeError(f"Failed to save distilled model: {e}")
    
    def load_distilled_model(
        self,
        checkpoint_path: Optional[str] = None,
        checkpoint_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load a distilled model from checkpoint
        
        Args:
            checkpoint_path: Optional direct path to checkpoint file
            checkpoint_name: Optional name of checkpoint to load
        
        Returns:
            Dict[str, Any]: Loaded state dictionary
        
        Raises:
            FileNotFoundError: If checkpoint not found
            RuntimeError: If loading fails
        """
        self.logger.log_info("Loading distilled model...")
        
        timer = Timer()
        timer.start()
        
        try:
            # Load checkpoint
            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
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
            
            # Restore model state
            if 'model_state_dict' in checkpoint:
                self.student_graph.load_state_dict(checkpoint['model_state_dict'])
            
            # Restore optimizer state
            if 'optimizer_state_dict' in checkpoint and self.optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore scheduler state
            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Restore metrics
            if 'metrics' in checkpoint:
                self.metrics.update(checkpoint['metrics'])
            
            # Move model to device
            self.student_graph.to(self.device)
            
            timer.stop()
            self.logger.log_info(f"Distilled model loaded in {timer.get_elapsed_time():.2f} seconds")
            
            return checkpoint
            
        except Exception as e:
            self.logger.log_error(f"Failed to load distilled model: {e}")
            raise RuntimeError(f"Failed to load distilled model: {e}")
    
    def train(self) -> Dict[str, Any]:
        """
        Execute the complete Phase 2 distillation training process
        
        Returns:
            Dict[str, Any]: Complete training metrics and statistics
        
        Raises:
            RuntimeError: If training fails
        """
        self.logger.log_info("=" * 50)
        self.logger.log_info("Starting Phase 2 Distillation Training")
        self.logger.log_info("=" * 50)
        
        try:
            # Step 1: Prepare training data
            self.logger.log_info("Step 1: Preparing training data...")
            data_stats = self.prepare_training_data()
            self.logger.log_info(f"Data preparation stats: {data_stats}")
            
            # Step 2: Train distillation
            self.logger.log_info("Step 2: Training distillation...")
            training_results = self.train_distillation()
            self.logger.log_info(f"Training results: {training_results}")
            
            # Step 3: Evaluate distillation
            self.logger.log_info("Step 3: Evaluating distillation...")
            eval_results = self.evaluate_distillation()
            self.logger.log_info(f"Evaluation results: {eval_results}")
            
            # Step 4: Save final model
            self.logger.log_info("Step 4: Saving final model...")
            checkpoint_path = self.save_distilled_model(
                checkpoint_name="phase2_complete"
            )
            self.logger.log_info(f"Final model saved to: {checkpoint_path}")
            
            # Prepare final metrics
            final_metrics = {
                'data_stats': data_stats,
                'training_results': training_results,
                'eval_results': eval_results,
                'checkpoint_path': checkpoint_path,
                'metrics': self.metrics
            }
            
            self.logger.log_info("=" * 50)
            self.logger.log_info("Phase 2 Distillation completed successfully")
            self.logger.log_info("=" * 50)
            
            return final_metrics
            
        except Exception as e:
            self.logger.log_error(f"Phase 2 Distillation failed: {e}")
            raise RuntimeError(f"Phase 2 Distillation failed: {e}")
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate the distillation process
        
        Returns:
            Dict[str, Any]: Validation metrics
        """
        validation_metrics = {
            'model_loaded': self.student_graph is not None,
            'training_data_available': self.training_data is not None,
            'num_train_samples': len(self.training_data.get('train', [])) if self.training_data else 0,
            'num_val_samples': len(self.training_data.get('val', [])) if self.training_data else 0,
            'best_val_loss': self.metrics['best_val_loss'],
            'best_epoch': self.metrics['best_epoch']
        }
        
        self.logger.log_info(f"Validation metrics: {validation_metrics}")
        return validation_metrics
    
    def test(self) -> Dict[str, Any]:
        """
        Test the distilled model
        
        Returns:
            Dict[str, Any]: Test metrics
        """
        return self.evaluate_distillation()
    
    # Private helper methods
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        learning_rate = self.phase_config.get('learning_rate', 1e-4)
        weight_decay = self.phase_config.get('weight_decay', 1e-5)
        
        self.optimizer = AdamW(
            self.student_graph.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup scheduler
        scheduler_type = self.phase_config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.phase_config.get('num_epochs', 30)
            )
        else:  # reduce_on_plateau
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
    
    def _create_data_loader(
        self,
        data: List[Dict[str, Any]],
        shuffle: bool = True
    ) -> DataLoader:
        """Create PyTorch DataLoader from training data"""
        if not data:
            return None
        
        # Convert to tensor dataset
        batch_size = self.phase_config.get('batch_size', 64)
        
        # Create custom dataset
        class DistillationDataset(Dataset):
            def __init__(self, samples):
                self.samples = samples
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                return self.samples[idx]
        
        dataset = DistillationDataset(data)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_batch
        )
        
        return loader
    
    def _collate_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for batch processing"""
        collated = {
            'user_ids': [item['user_id'] for item in batch],
            'item_ids': [item['item_id'] for item in batch],
            'labels': torch.tensor([item['label'] for item in batch], dtype=torch.float32)
        }
        
        # Stack embeddings
        user_embeddings = torch.stack([item['user_embedding'] for item in batch])
        item_embeddings = torch.stack([item['item_embedding'] for item in batch])
        collated['user_embeddings'] = user_embeddings
        collated['item_embeddings'] = item_embeddings
        
        # Stack component knowledge if available
        if 'component_knowledge' in batch[0]:
            component_knowledge = [item['component_knowledge'] for item in batch]
            collated['component_knowledge'] = component_knowledge
        
        # Stack metapath context if available
        if 'metapath_context' in batch[0] and batch[0]['metapath_context']:
            metapath_context = [item['metapath_context'] for item in batch]
            collated['metapath_context'] = metapath_context
        
        return collated
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare batch for model input"""
        prepared = batch.copy()
        
        # Move tensors to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.device)
        
        return prepared
    
    def _train_epoch(self, epoch: int) -> Tuple[float, Dict[str, Any]]:
        """Train for one epoch"""
        self.student_graph.train()
        
        epoch_loss = 0.0
        num_batches = 0
        epoch_metrics = defaultdict(float)
        
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1} Training"):
            # Prepare batch
            batch = self._prepare_batch(batch)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            student_outputs = self.student_graph.forward(
                graph=batch.get('graph', None),
                node_features=batch.get('node_features', None)
            )
            
            # Get teacher outputs (simulated)
            teacher_outputs = self._simulate_teacher_outputs(batch)
            
            # Compute distillation losses
            losses = self._compute_distillation_losses(
                student_outputs,
                teacher_outputs,
                batch
            )
            
            total_loss = sum(losses.values())
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.student_graph.parameters(),
                max_norm=self.phase_config.get('grad_clip', 1.0)
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Track metrics
            epoch_loss += total_loss.item()
            num_batches += 1
            
            for loss_name, loss_value in losses.items():
                epoch_metrics[loss_name] += loss_value.item()
            
            self.metrics['step'] += 1
        
        # Compute average losses
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_metrics = {
            key: value / num_batches if num_batches > 0 else 0.0
            for key, value in epoch_metrics.items()
        }
        
        # Store metrics
        self.metrics['train_losses'].append(avg_loss)
        for key, value in avg_metrics.items():
            self.metrics['component_losses'][key].append(value)
        
        return avg_loss, avg_metrics
    
    def _validate_epoch(self, epoch: int) -> Tuple[float, Dict[str, Any]]:
        """Validate for one epoch"""
        self.student_graph.eval()
        
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Prepare batch
                batch = self._prepare_batch(batch)
                
                # Forward pass
                student_outputs = self.student_graph.forward(
                    graph=batch.get('graph', None),
                    node_features=batch.get('node_features', None)
                )
                
                # Get teacher outputs
                teacher_outputs = self._simulate_teacher_outputs(batch)
                
                # Compute losses
                losses = self._compute_distillation_losses(
                    student_outputs,
                    teacher_outputs,
                    batch
                )
                
                total_loss = sum(losses.values())
                val_loss += total_loss.item()
                num_batches += 1
        
        avg_loss = val_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss, {}
    
    def _compute_distillation_losses(
        self,
        student_outputs: Dict[str, Any],
        teacher_outputs: Dict[str, Any],
        batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Compute all distillation losses"""
        losses = {}
        
        # 1. Component-wise loss
        component_weight = self.phase_config.get('alpha', 0.5)
        component_loss = self.distillation_loss.component_wise_loss(
            student=student_outputs,
            teacher=teacher_outputs,
            component_type='all'
        )
        losses['component'] = component_weight * component_loss
        
        # 2. Path importance loss
        path_weight = self.phase_config.get('beta', 0.3)
        if 'student_attn' in student_outputs and 'teacher_attn' in teacher_outputs:
            path_loss = self.distillation_loss.path_importance_loss(
                student_attn=student_outputs['student_attn'],
                teacher_attn=teacher_outputs['teacher_attn']
            )
            losses['path'] = path_weight * path_loss
        
        # 3. Contrastive loss
        contrastive_weight = self.phase_config.get('gamma', 0.2)
        if 'embeddings' in batch:
            contrastive_loss = self.distillation_loss.contrastive_loss(
                embeddings=batch['embeddings'],
                temperature=self.phase_config.get('temperature', 0.07)
            )
            losses['contrastive'] = contrastive_weight * contrastive_loss
        
        # 4. Reconstruction loss (if applicable)
        if 'labels' in batch:
            reconstruction_loss = self.distillation_loss.reconstruction_loss(
                predicted=student_outputs.get('predictions', None),
                target=batch['labels'],
                component_type='prediction'
            )
            losses['reconstruction'] = 0.1 * reconstruction_loss
        
        return losses
    
    def _simulate_teacher_outputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate teacher outputs for distillation"""
        teacher_outputs = {
            'embeddings': batch.get('user_embeddings', None),
            'labels': batch.get('labels', None),
            'component_embeddings': {}
        }
        
        # Generate component-specific teacher embeddings
        for comp_type in ['intrinsic', 'collaborative', 'interaction']:
            if 'component_knowledge' in batch:
                comp_embeddings = self._generate_component_embeddings(
                    batch['component_knowledge'],
                    comp_type
                )
                teacher_outputs['component_embeddings'][comp_type] = comp_embeddings
        
        # Generate attention weights (simulated)
        if batch.get('metapath_context'):
            teacher_outputs['teacher_attn'] = self._simulate_attention_weights(
                batch['metapath_context']
            )
        
        return teacher_outputs
    
    def _generate_component_embeddings(
        self,
        component_knowledge: List[Dict[str, Any]],
        component_type: str
    ) -> torch.Tensor:
        """Generate embeddings for a specific component type"""
        embeddings = []
        
        for knowledge in component_knowledge:
            if component_type in knowledge:
                text = knowledge[component_type]
                embedding = self.text_encoder.encode(text)
                embeddings.append(torch.tensor(embedding, dtype=torch.float32))
            else:
                embeddings.append(torch.zeros(self.text_encoder.get_embedding_dimension()))
        
        return torch.stack(embeddings)
    
    def _simulate_attention_weights(
        self,
        metapath_context: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Simulate attention weights for path importance"""
        batch_size = len(metapath_context)
        num_paths = max(len(context.get('paths', [])) for context in metapath_context) if metapath_context else 1
        
        # Generate simulated attention weights (uniform for simplicity)
        attn_weights = torch.ones(batch_size, num_paths) / num_paths
        
        return attn_weights
    
    def _compute_component_alignment(
        self,
        student_outputs: Dict[str, Any],
        teacher_outputs: Dict[str, Any],
        component_type: str
    ) -> float:
        """Compute alignment score between student and teacher for a component"""
        if 'component_embeddings' not in teacher_outputs:
            return 0.0
        
        student_emb = student_outputs.get('component_embeddings', {}).get(component_type)
        teacher_emb = teacher_outputs['component_embeddings'].get(component_type)
        
        if student_emb is None or teacher_emb is None:
            return 0.0
        
        # Compute cosine similarity
        if len(student_emb.shape) == 1:
            student_emb = student_emb.unsqueeze(0)
            teacher_emb = teacher_emb.unsqueeze(0)
        
        similarity = F.cosine_similarity(student_emb, teacher_emb, dim=-1)
        return similarity.mean().item()
    
    def _evaluate_distillation_quality(
        self,
        student_outputs: Dict[str, Any],
        teacher_outputs: Dict[str, Any],
        batch: Dict[str, Any]
    ) -> float:
        """Evaluate the quality of distillation for a batch"""
        quality_scores = []
        
        # Check embedding alignment
        if 'embeddings' in student_outputs and 'embeddings' in teacher_outputs:
            similarity = F.cosine_similarity(
                student_outputs['embeddings'],
                teacher_outputs['embeddings'],
                dim=-1
            )
            quality_scores.append(similarity.mean().item())
        
        # Check prediction alignment
        if 'labels' in batch and 'predictions' in student_outputs:
            predictions = student_outputs['predictions']
            labels = batch['labels']
            accuracy = (predictions.argmax(dim=-1) == labels).float().mean().item()
            quality_scores.append(accuracy)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _compute_memory_transfer_score(self) -> float:
        """Compute score for memory dynamics transfer"""
        # This would evaluate how well the student captures memory dynamics
        # For now, return a placeholder score
        return 0.85
    
    def _extract_component_knowledge(
        self,
        reflection: Dict[str, Any]
    ) -> Dict[str, str]:
        """Extract component-wise knowledge from reflection"""
        component_knowledge = {}
        
        # Extract intrinsic knowledge
        if 'intrinsic_insight' in reflection:
            component_knowledge['intrinsic'] = reflection['intrinsic_insight']
        
        # Extract collaborative knowledge
        if 'collaborative_signal' in reflection:
            component_knowledge['collaborative'] = reflection['collaborative_signal']
        
        # Extract interaction knowledge
        if 'interaction_pattern' in reflection:
            component_knowledge['interaction'] = reflection['interaction_pattern']
        
        return component_knowledge
    
    def _generate_metapath_context(
        self,
        user_id: str,
        item_id: str
    ) -> List[Dict[str, Any]]:
        """Generate metapath context from graph"""
        context = []
        
        if self.graph is None or self.graph_retriever is None:
            return context
        
        try:
            # Extract metapaths
            paths = []
            
            # User-Item-User path
            user_item_user_paths = self.metapath_extractor.extract_user_item_user(
                anchor_user=user_id,
                k=self.phase_config.get('metapath_k', 5)
            )
            paths.extend(user_item_user_paths)
            
            # User-User-Item path
            user_user_item_paths = self.metapath_extractor.extract_user_user_item(
                anchor_user=user_id,
                k=self.phase_config.get('metapath_k', 5)
            )
            paths.extend(user_user_item_paths)
            
            # Create context
            context = {
                'user_id': user_id,
                'item_id': item_id,
                'paths': paths,
                'num_paths': len(paths)
            }
            
        except Exception as e:
            self.logger.log_warning(f"Failed to generate metapath context: {e}")
        
        return [context] if context else []
    
    def _generate_negative_samples(
        self,
        user_id: str,
        item_id: str,
        num_negatives: int,
        traces: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate negative samples for training"""
        negative_samples = []
        
        # Get all items from traces
        all_items = list(set([trace.get('item_id') for trace in traces if trace.get('item_id')]))
        
        # Filter out positive item
        candidate_items = [item for item in all_items if item != item_id]
        
        # Sample negative items
        neg_items = random.sample(
            candidate_items,
            min(num_negatives, len(candidate_items))
        )
        
        for neg_item in neg_items:
            # Find a trace with this item
            neg_trace = next(
                (t for t in traces if t.get('item_id') == neg_item),
                None
            )
            
            if neg_trace:
                negative_sample = {
                    'user_id': user_id,
                    'item_id': neg_item,
                    'user_embedding': neg_trace.get('user_embedding', np.zeros(1536)),
                    'item_embedding': neg_trace.get('item_embedding', np.zeros(1536)),
                    'component_knowledge': self._extract_component_knowledge(
                        neg_trace.get('reflection', {})
                    ),
                    'reflection': neg_trace.get('reflection', {}),
                    'metapath_context': [],
                    'label': 0.0  # Negative sample
                }
                negative_samples.append(negative_sample)
        
        return negative_samples
    
    def _save_training_data(self, path: str):
        """Save training data to disk"""
        if self.training_data:
            with open(path, 'wb') as f:
                pickle.dump(self.training_data, f)
            self.logger.log_info(f"Training data saved to {path}")
    
    def _save_best_model(self):
        """Save the best performing model"""
        checkpoint_path = self.save_distilled_model(
            checkpoint_name="best_distilled_model"
        )
        self.logger.log_info(f"Best model saved to {checkpoint_path}")


# Command-line interface for running Phase 2 independently
def main(
    config_path: str,
    teacher_path: Optional[str] = None,
    student_path: Optional[str] = None
) -> None:
    """
    Main entry point for running Phase 2 independently
    
    Args:
        config_path: Path to configuration file
        teacher_path: Optional path to teacher model
        student_path: Optional path to student model (for resuming)
        
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
        name='phase2_main'
    )
    
    logger.log_info("Starting Phase 2 Distillation main execution...")
    
    try:
        # Initialize dataset
        dataset_config = config.get('data', {})
        dataset = AmazonDataset(
            dataset_name=dataset_config.get('name', 'amazon_reviews'),
            config=dataset_config
        )
        dataset.load_data()
        
        # Initialize LLM teacher
        llm_config = config.get('llm', {})
        from models.llm.llm_interface import LLMFactory
        teacher = LLMFactory.create_llm(
            model_type=llm_config.get('model_type', 'openai'),
            config=llm_config
        )
        
        # Initialize GNN student
        gnn_config = config.get('gnn', {})
        student = HeterogeneousGNN(config=gnn_config)
        
        # Load student from checkpoint if provided
        if student_path and os.path.exists(student_path):
            checkpoint = torch.load(student_path, map_location='cpu')
            student.load_state_dict(checkpoint.get('model_state_dict', {}))
            logger.log_info(f"Loaded student model from {student_path}")
        
        # Initialize Phase 2 trainer
        trainer = Phase2Distillation(
            teachers=teacher,
            student_graph=student,
            config=config,
            dataset=dataset
        )
        
        # Run training
        results = trainer.train()
        
        # Save results
        results_path = os.path.join(
            config.get('common', {}).get('log_dir', './logs'),
            'phase2_results.json'
        )
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.log_info(f"Results saved to {results_path}")
        logger.log_info("Phase 2 Distillation completed successfully")
        
    except Exception as e:
        logger.log_error(f"Phase 2 Distillation failed: {e}")
        raise


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase2_distillation.py <config_path> [teacher_path] [student_path]")
        sys.exit(1)
    
    teacher_path = sys.argv[2] if len(sys.argv) > 2 else None
    student_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    main(sys.argv[1], teacher_path, student_path)