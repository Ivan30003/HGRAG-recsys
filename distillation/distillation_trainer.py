"""
Distillation Trainer for H-GRAGrecsys

This module implements the knowledge distillation training framework for
transferring knowledge from LLM teacher to GNN student. It supports:
- Component-wise distillation for disentangled representations
- Path importance distillation for metapath-based reasoning
- Contrastive learning for representation alignment
- Memory dynamics distillation for agent memories
- Multi-phase training with checkpoint management

The trainer enables efficient distillation of complex LLM reasoning into
lightweight GNN models for fast inference.
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import time
import math
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from sibling modules
from distillation.loss_functions import DistillationLoss
from distillation.knowledge_distiller import KnowledgeDistiller
from distillation.component_disentangler import ComponentDisentangler

# Import from GNN module
from models.gnn.gnn_encoder import GNNEncoder
from models.gnn.heterogeneous_gnn import HeterogeneousGNN

# Import from LLM module
from models.llm.llm_interface import LLMInterface

# Import from agent module
from models.agent.user_agent import UserAgent
from models.agent.item_agent import ItemAgent

# Import from training
from training.checkpoint_manager import CheckpointManager

# Import from utils
from utils.config_loader import ConfigLoader
from utils.logger import Logger
from utils.seed_manager import SeedManager
from utils.timer import Timer


@dataclass
class DistillationConfig:
    """
    Configuration for distillation training.
    
    Attributes:
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        component_weights: Weights for each component loss.
        temperature: Temperature for softmax in distillation.
        alpha: Weight for distillation loss.
        beta: Weight for contrastive loss.
        gamma: Weight for orthogonality loss.
        use_teacher_ensemble: Whether to use ensemble of teachers.
        distill_memory: Whether to distill memory dynamics.
    """
    num_epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-4
    component_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    temperature: float = 0.07
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.2
    use_teacher_ensemble: bool = False
    distill_memory: bool = True


class DistillationTrainer:
    """
    Knowledge distillation trainer for H-GRAGrecsys.
    
    This class orchestrates the distillation process from teacher LLM to
    student GNN, managing the training loop, loss computation, and
    checkpoint management.
    """
    
    def __init__(
        self,
        teacher: Optional[LLMInterface] = None,
        student: Optional[GNNEncoder] = None,
        config: Optional[Union[str, Dict, ConfigLoader]] = None,
        teacher_llm: Optional[LLMInterface] = None  # Alias for backward compatibility
    ):
        """
        Initialize the distillation trainer.
        
        Args:
            teacher: Optional teacher LLM interface. If None, creates from config.
            student: Optional student GNN encoder. If None, creates from config.
            config: Configuration object or path to config file.
            teacher_llm: Alias for teacher (backward compatibility).
        
        Raises:
            ValueError: If config is invalid or missing required fields.
        """
        super(DistillationTrainer, self).__init__()
        
        # Handle teacher alias
        if teacher_llm is not None and teacher is None:
            teacher = teacher_llm
        
        # Load configuration
        if config is None:
            self.config = {
                'model': {
                    'distillation': {
                        'num_epochs': 30,
                        'batch_size': 64,
                        'learning_rate': 1e-4,
                        'component_weights': [1.0, 1.0, 1.0],
                        'temperature': 0.07,
                        'alpha': 0.5,
                        'beta': 0.3,
                        'gamma': 0.2,
                        'use_teacher_ensemble': False,
                        'distill_memory': True
                    }
                },
                'training': {
                    'checkpoint_dir': './checkpoints',
                    'log_dir': './logs'
                }
            }
        elif isinstance(config, str):
            self.config_loader = ConfigLoader(config)
            self.config = self.config_loader.load_config()
        elif isinstance(config, dict):
            self.config = config
            self.config_loader = None
        elif isinstance(config, ConfigLoader):
            self.config_loader = config
            self.config = config.load_config()
        else:
            raise ValueError(f"Invalid config type: {type(config)}")
        
        # Setup logger
        self.logger = Logger(
            log_dir=self.config.get('training', {}).get('log_dir', './logs'),
            name='distillation_trainer'
        )
        
        # Extract configuration
        self.distillation_config = self._parse_distillation_config()
        
        # Initialize components
        self.teacher = teacher if teacher is not None else self._create_teacher()
        self.student = student if student is not None else self._create_student()
        
        # Initialize knowledge distiller
        self.knowledge_distiller = KnowledgeDistiller(
            teacher_llm=self.teacher,
            student_gnn=self.student,
            config=self.config
        )
        
        # Initialize loss functions
        self.distillation_loss = DistillationLoss(self.config)
        
        # Initialize component disentangler
        self.disentangler = ComponentDisentangler(self.config)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.student.parameters(),
            lr=self.distillation_config.learning_rate
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.config.get('training', {}).get('checkpoint_dir', './checkpoints'),
            max_checkpoints=5
        )
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float('inf')
        self.best_metrics = {}
        
        # Training history
        self.train_history = {
            'loss': [],
            'distillation_loss': [],
            'contrastive_loss': [],
            'orthogonality_loss': [],
            'validation_loss': [],
            'validation_metrics': []
        }
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to_device(self.device)
        
        self.logger.log_info(
            f"DistillationTrainer initialized: epochs={self.distillation_config.num_epochs}, "
            f"batch_size={self.distillation_config.batch_size}, "
            f"temperature={self.distillation_config.temperature}"
        )
    
    def _parse_distillation_config(self) -> DistillationConfig:
        """Parse distillation configuration from config dict."""
        dist_config = self.config.get('model', {}).get('distillation', {})
        
        return DistillationConfig(
            num_epochs=dist_config.get('num_epochs', 30),
            batch_size=dist_config.get('batch_size', 64),
            learning_rate=dist_config.get('learning_rate', 1e-4),
            component_weights=dist_config.get('component_weights', [1.0, 1.0, 1.0]),
            temperature=dist_config.get('temperature', 0.07),
            alpha=dist_config.get('alpha', 0.5),
            beta=dist_config.get('beta', 0.3),
            gamma=dist_config.get('gamma', 0.2),
            use_teacher_ensemble=dist_config.get('use_teacher_ensemble', False),
            distill_memory=dist_config.get('distill_memory', True)
        )
    
    def _create_teacher(self) -> LLMInterface:
        """Create teacher LLM interface from configuration."""
        return LLMInterface(config=self.config)
    
    def _create_student(self) -> GNNEncoder:
        """Create student GNN encoder from configuration."""
        return GNNEncoder(config=self.config)
    
    def train_step(
        self,
        batch: Dict[str, Any],
        teacher_outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Training batch containing graph, nodes, and targets.
            teacher_outputs: Optional pre-computed teacher outputs.
        
        Returns:
            Dict containing loss values for the step.
        """
        self.student.train()
        
        # Move batch to device
        batch = self._move_batch_to_device(batch)
        
        # Forward pass through student
        graph = batch.get('graph')
        node_features = batch.get('node_features')
        
        if graph is None:
            raise ValueError("Batch must contain 'graph'")
        
        # Get student outputs
        student_embeddings = self.student.encode_graph(graph, node_features)
        
        # Get teacher outputs if not provided
        if teacher_outputs is None and self.teacher is not None:
            teacher_outputs = self._get_teacher_outputs(batch)
        
        # Compute distillation loss
        loss_components = {}
        
        # 1. Component-wise distillation loss
        if teacher_outputs is not None:
            dist_loss = self.distillation_loss.compute_total_loss(
                teacher_logits=teacher_outputs.get('logits'),
                student_logits=student_embeddings,
                labels=batch.get('labels')
            )
            loss_components['distillation'] = self.distillation_config.alpha * dist_loss
        
        # 2. Contrastive loss for representation alignment
        if teacher_outputs is not None:
            contrastive_loss = self.distillation_loss.contrastive_loss(
                embeddings=student_embeddings,
                temperature=self.distillation_config.temperature
            )
            loss_components['contrastive'] = self.distillation_config.beta * contrastive_loss
        
        # 3. Orthogonality loss for disentangled representations
        if hasattr(self.student, 'projection_heads'):
            # Get component projections
            components = {}
            for node_type, emb in student_embeddings.items():
                components[node_type] = self.student.projection_heads.project_all(emb)
            
            ortho_loss = self.distillation_loss.orthogonality_loss(components)
            loss_components['orthogonality'] = self.distillation_config.gamma * ortho_loss
        
        # 4. Path importance distillation
        if batch.get('metapaths') is not None:
            path_loss = self.distillation_loss.path_importance_loss(
                student_attn=batch.get('student_attention'),
                teacher_attn=batch.get('teacher_attention')
            )
            loss_components['path'] = 0.1 * path_loss
        
        # Total loss
        total_loss = sum(loss_components.values())
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update step counter
        self.current_step += 1
        
        # Return loss components
        loss_components['total'] = total_loss.item()
        
        return loss_components
    
    def _get_teacher_outputs(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Get teacher outputs for a batch.
        
        Args:
            batch: Training batch.
        
        Returns:
            Dict containing teacher outputs.
        """
        if self.teacher is None:
            return None
        
        teacher_outputs = {}
        
        # Get teacher predictions
        if 'prompts' in batch:
            # Generate teacher predictions
            prompts = batch['prompts']
            teacher_logits = []
            
            for prompt in prompts:
                output = self.teacher.generate(prompt)
                # Convert to logits (simplified)
                logits = torch.randn(10)  # Placeholder
                teacher_logits.append(logits)
            
            teacher_outputs['logits'] = torch.stack(teacher_logits)
        
        # Get teacher embeddings
        if 'texts' in batch:
            teacher_embeddings = []
            for text in batch['texts']:
                embedding = self.teacher.get_embedding(text)
                teacher_embeddings.append(embedding)
            teacher_outputs['embeddings'] = torch.stack(teacher_embeddings)
        
        return teacher_outputs
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move batch tensors to device.
        
        Args:
            batch: Batch dictionary.
        
        Returns:
            Batch with tensors moved to device.
        """
        moved_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                moved_batch[key] = self._move_batch_to_device(value)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                moved_batch[key] = [v.to(self.device) for v in value]
            else:
                moved_batch[key] = value
        
        return moved_batch
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.
        
        Returns:
            Dict containing average losses for the epoch.
        """
        self.logger.log_info(f"Starting epoch {epoch + 1}/{self.distillation_config.num_epochs}")
        
        epoch_losses = defaultdict(float)
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for batch in pbar:
            # Training step
            losses = self.train_step(batch)
            
            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key] += value
            
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'],
                'dist': losses.get('distillation', 0),
                'cont': losses.get('contrastive', 0)
            })
        
        # Compute average losses
        avg_losses = {key: value / num_batches for key, value in epoch_losses.items()}
        
        # Update learning rate
        self.scheduler.step(avg_losses['total'])
        
        # Log epoch results
        self.logger.log_info(
            f"Epoch {epoch + 1} completed: "
            f"total_loss={avg_losses['total']:.4f}, "
            f"dist_loss={avg_losses.get('distillation', 0):.4f}"
        )
        
        return avg_losses
    
    def validate(
        self,
        val_loader: DataLoader,
        metrics: Optional[List[Callable]] = None
    ) -> Dict[str, float]:
        """
        Validate the student model.
        
        Args:
            val_loader: Validation data loader.
            metrics: Optional list of metric functions.
        
        Returns:
            Dict containing validation metrics.
        """
        self.student.eval()
        
        val_losses = defaultdict(float)
        val_metrics = defaultdict(list)
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                graph = batch.get('graph')
                node_features = batch.get('node_features')
                
                if graph is None:
                    continue
                
                student_embeddings = self.student.encode_graph(graph, node_features)
                
                # Compute losses
                if 'labels' in batch:
                    # Compute loss
                    loss = F.mse_loss(student_embeddings, batch['labels'])
                    val_losses['loss'] += loss.item()
                
                # Compute metrics if provided
                if metrics:
                    for metric_fn in metrics:
                        metric_value = metric_fn(student_embeddings, batch.get('labels'))
                        val_metrics[metric_fn.__name__].append(metric_value)
                
                num_batches += 1
        
        # Compute averages
        avg_losses = {key: value / num_batches for key, value in val_losses.items()}
        avg_metrics = {key: np.mean(values) for key, values in val_metrics.items()}
        
        # Combine results
        results = {**avg_losses, **avg_metrics}
        
        self.logger.log_info(f"Validation: loss={results.get('loss', 0):.4f}")
        
        return results
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
        save_best: bool = True
    ) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            num_epochs: Optional number of epochs. If None, uses config.
            save_best: Whether to save best model checkpoint.
        
        Returns:
            Dict containing training history.
        """
        if num_epochs is None:
            num_epochs = self.distillation_config.num_epochs
        
        self.logger.log_info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(self.current_epoch, num_epochs):
            # Train epoch
            train_losses = self.train_epoch(train_loader, epoch)
            
            # Save to history
            self.train_history['loss'].append(train_losses['total'])
            self.train_history['distillation_loss'].append(train_losses.get('distillation', 0))
            self.train_history['contrastive_loss'].append(train_losses.get('contrastive', 0))
            self.train_history['orthogonality_loss'].append(train_losses.get('orthogonality', 0))
            
            # Validate
            if val_loader is not None:
                val_results = self.validate(val_loader)
                self.train_history['validation_loss'].append(val_results.get('loss', 0))
                self.train_history['validation_metrics'].append(val_results)
            
            # Save checkpoint
            self.current_epoch = epoch + 1
            
            if save_best and val_loader is not None:
                val_loss = val_results.get('loss', float('inf'))
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.best_metrics = val_results
                    self._save_checkpoint(f'best_model_epoch_{epoch + 1}.pt')
                    self.logger.log_info(f"Best model saved (loss: {self.best_loss:.4f})")
            
            # Periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
        
        self.logger.log_info("Training completed")
        
        return self.train_history
    
    def _save_checkpoint(self, filename: str):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename.
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'student_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_metrics': self.best_metrics,
            'config': self.config,
            'train_history': self.train_history
        }
        
        self.checkpoint_manager.save_checkpoint(
            state=checkpoint,
            epoch=self.current_epoch,
            step=self.current_step
        )
    
    def load_checkpoint(self, checkpoint_name: str) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint to load.
        
        Returns:
            Dict containing checkpoint data.
        """
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        
        if checkpoint is not None:
            self.student.load_state_dict(checkpoint['student_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.current_step = checkpoint['step']
            self.best_loss = checkpoint['best_loss']
            self.best_metrics = checkpoint['best_metrics']
            
            self.logger.log_info(f"Loaded checkpoint: {checkpoint_name}")
            return checkpoint
        
        self.logger.log_warning(f"Checkpoint not found: {checkpoint_name}")
        return None
    
    def save_model(self, save_path: str):
        """
        Save trained model.
        
        Args:
            save_path: Path to save the model.
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save student model
            torch.save({
                'student_state_dict': self.student.state_dict(),
                'config': self.config,
                'train_history': self.train_history,
                'best_loss': self.best_loss,
                'best_metrics': self.best_metrics
            }, save_path)
            
            self.logger.log_info(f"Model saved to {save_path}")
        
        except Exception as e:
            self.logger.log_error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, load_path: str):
        """
        Load trained model.
        
        Args:
            load_path: Path to load the model from.
        
        Raises:
            FileNotFoundError: If checkpoint not found.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model checkpoint not found: {load_path}")
        
        try:
            checkpoint = torch.load(load_path, map_location=self.device)
            
            self.student.load_state_dict(checkpoint['student_state_dict'])
            self.train_history = checkpoint.get('train_history', self.train_history)
            self.best_loss = checkpoint.get('best_loss', self.best_loss)
            self.best_metrics = checkpoint.get('best_metrics', self.best_metrics)
            
            self.logger.log_info(f"Model loaded from {load_path}")
        
        except Exception as e:
            self.logger.log_error(f"Failed to load model: {e}")
            raise
    
    def evaluate_distillation(
        self,
        test_loader: DataLoader,
        metrics: Optional[List[Callable]] = None
    ) -> Dict[str, float]:
        """
        Evaluate distillation quality.
        
        Args:
            test_loader: Test data loader.
            metrics: Optional list of metric functions.
        
        Returns:
            Dict containing evaluation metrics.
        """
        self.logger.log_info("Evaluating distillation quality")
        
        return self.validate(test_loader, metrics)
    
    def get_distillation_stats(self) -> Dict[str, Any]:
        """
        Get distillation statistics.
        
        Returns:
            Dict containing distillation statistics.
        """
        return {
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'best_loss': self.best_loss,
            'best_metrics': self.best_metrics,
            'num_parameters': sum(p.numel() for p in self.student.parameters()),
            'teacher_type': type(self.teacher).__name__ if self.teacher else None,
            'student_type': type(self.student).__name__ if self.student else None,
            'train_history_length': len(self.train_history['loss'])
        }
    
    def set_teacher(self, teacher: LLMInterface):
        """
        Set teacher model.
        
        Args:
            teacher: Teacher LLM interface.
        """
        self.teacher = teacher
        self.knowledge_distiller.teacher_llm = teacher
        self.logger.log_info("Teacher model updated")
    
    def set_student(self, student: GNNEncoder):
        """
        Set student model.
        
        Args:
            student: Student GNN encoder.
        """
        self.student = student
        self.knowledge_distiller.student_gnn = student
        
        # Recreate optimizer
        self.optimizer = torch.optim.Adam(
            self.student.parameters(),
            lr=self.distillation_config.learning_rate
        )
        
        self.logger.log_info("Student model updated")
    
    def set_learning_rate(self, learning_rate: float):
        """
        Update learning rate.
        
        Args:
            learning_rate: New learning rate.
        """
        self.distillation_config.learning_rate = learning_rate
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        self.logger.log_info(f"Learning rate updated to {learning_rate}")
    
    def to_device(self, device: torch.device) -> 'DistillationTrainer':
        """
        Move all components to specified device.
        
        Args:
            device: PyTorch device.
        
        Returns:
            Self with components moved to device.
        """
        self.device = device
        
        if self.student:
            self.student.to_device(device)
        if self.teacher and hasattr(self.teacher, 'to_device'):
            self.teacher.to_device(device)
        
        self.to(device)
        self.logger.log_info(f"Trainer moved to device: {device}")
        
        return self
    
    def reset(self):
        """Reset training state."""
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float('inf')
        self.best_metrics = {}
        self.train_history = {
            'loss': [],
            'distillation_loss': [],
            'contrastive_loss': [],
            'orthogonality_loss': [],
            'validation_loss': [],
            'validation_metrics': []
        }
        
        # Reinitialize optimizer
        self.optimizer = torch.optim.Adam(
            self.student.parameters(),
            lr=self.distillation_config.learning_rate
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        self.logger.log_info("Training state reset")


# Module level variables and exports
__all__ = [
    'DistillationConfig',
    'DistillationTrainer',
    '__doc__'
]

# Version information
__version__ = '1.0.0'
__author__ = 'H-GRAGrecsys Team'


def create_distillation_trainer(
    teacher: Optional[LLMInterface] = None,
    student: Optional[GNNEncoder] = None,
    config_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    teacher_llm: Optional[LLMInterface] = None  # Alias for backward compatibility
) -> DistillationTrainer:
    """
    Factory function to create a DistillationTrainer instance.
    
    Args:
        teacher: Optional teacher LLM interface.
        student: Optional student GNN encoder.
        config_path: Optional path to configuration file.
        device: Optional device to move trainer to.
        teacher_llm: Alias for teacher (backward compatibility).
    
    Returns:
        Initialized DistillationTrainer instance.
    
    Example:
        >>> trainer = create_distillation_trainer(
        ...     teacher=llm_teacher,
        ...     student=gnn_student,
        ...     config_path='config/default_config.yaml'
        ... )
        >>> trainer.train(train_loader, val_loader)
    """
    trainer = DistillationTrainer(
        teacher=teacher or teacher_llm,
        student=student,
        config=config_path
    )
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return trainer.to_device(device)


def create_distillation_config(
    num_epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    component_weights: List[float] = None,
    temperature: float = 0.07,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    use_teacher_ensemble: bool = False,
    distill_memory: bool = True
) -> DistillationConfig:
    """
    Factory function to create a DistillationConfig object.
    
    Args:
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        component_weights: Weights for each component loss.
        temperature: Temperature for softmax in distillation.
        alpha: Weight for distillation loss.
        beta: Weight for contrastive loss.
        gamma: Weight for orthogonality loss.
        use_teacher_ensemble: Whether to use ensemble of teachers.
        distill_memory: Whether to distill memory dynamics.
    
    Returns:
        DistillationConfig object.
    
    Example:
        >>> config = create_distillation_config(
        ...     num_epochs=50,
        ...     batch_size=128,
        ...     learning_rate=5e-5
        ... )
    """
    if component_weights is None:
        component_weights = [1.0, 1.0, 1.0]
    
    return DistillationConfig(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        component_weights=component_weights,
        temperature=temperature,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        use_teacher_ensemble=use_teacher_ensemble,
        distill_memory=distill_memory
    )