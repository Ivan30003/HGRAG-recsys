"""
Base Trainer Module for H-GRAGrecsys

This module provides the foundational trainer class that serves as the base
for all training phases (Phase 1: Bootstrap, Phase 2: Distillation, Phase 3: Hybrid).
It implements common functionality for model training, validation, testing,
checkpoint management, logging, and metrics tracking.

Key Responsibilities:
- Provide base training/validation/testing interface
- Handle checkpoint management
- Manage configuration and logging
- Track training metrics
- Support model serialization
"""

import os
import sys
import json
import abc
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from pathlib import Path
import numpy as np
from collections import defaultdict
from datetime import datetime
import pickle
import hashlib

# Add project root to path if needed
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Core imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# Data imports
from data.dataset import BaseDataset
from data.data_loader import DataLoader as DataLoaderClass

# Utils imports
from utils.logger import Logger
from utils.config_loader import ConfigLoader
from utils.seed_manager import SeedManager
from utils.timer import Timer
from utils.visualization import Visualizer

# Training imports
from .checkpoint_manager import CheckpointManager


class BaseTrainer(abc.ABC):
    """
    Abstract base class for all trainers in the H-GRAGrecsys system
    
    This class provides a unified interface and common functionality for
    all training phases. Subclasses must implement the abstract methods.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model: Optional[nn.Module] = None,
        data_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        logger: Optional[Logger] = None,
        visualizer: Optional[Visualizer] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the base trainer
        
        Args:
            config: Configuration dictionary for the trainer
            model: PyTorch model to be trained
            data_loader: Data loader for training data
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            checkpoint_manager: Manager for saving/loading checkpoints
            logger: Logger for tracking progress
            visualizer: Visualizer for plotting metrics
            device: Device to use ('cuda' or 'cpu')
        """
        self.config = config
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or self._get_default_device()
        
        # Initialize components
        self.logger = logger or self._create_default_logger()
        self.checkpoint_manager = checkpoint_manager or self._create_default_checkpoint_manager()
        self.visualizer = visualizer or Visualizer(
            config=config.get('visualization', {})
        )
        
        # Set random seed
        seed = config.get('common', {}).get('seed', 42)
        SeedManager.set_seed(seed)
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'test_accuracy': [],
            'epochs': 0,
            'best_val_loss': float('inf'),
            'best_val_accuracy': 0.0,
            'best_epoch': -1,
            'start_time': Timer.get_current_timestamp(),
            'end_time': None,
            'total_time': 0.0,
            'training_history': defaultdict(list)
        }
        
        # State tracking
        self.is_trained = False
        self.is_loaded = False
        self.current_epoch = 0
        self.current_step = 0
        self.best_model_state = None
        
        # Move model to device
        if self.model is not None:
            self.model.to(self.device)
            self.logger.log_info(f"Model moved to device: {self.device}")
        
        self.logger.log_info(f"BaseTrainer initialized on device: {self.device}")
        self.logger.log_info(f"Config keys: {list(config.keys())}")
    
    @abc.abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Execute the training process
        
        This method must be implemented by subclasses to define the specific
        training logic for each phase.
        
        Returns:
            Dict[str, Any]: Training results and metrics
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            "Subclasses must implement the train() method"
        )
    
    @abc.abstractmethod
    def validate(self) -> Dict[str, Any]:
        """
        Validate the model on validation data
        
        Returns:
            Dict[str, Any]: Validation metrics
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            "Subclasses must implement the validate() method"
        )
    
    @abc.abstractmethod
    def test(self) -> Dict[str, Any]:
        """
        Test the model on test data
        
        Returns:
            Dict[str, Any]: Test metrics
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            "Subclasses must implement the test() method"
        )
    
    def save_model(
        self,
        path: Optional[str] = None,
        include_optimizer: bool = True,
        include_scheduler: bool = True,
        include_metrics: bool = True,
        include_config: bool = True
    ) -> str:
        """
        Save the model and training state to disk
        
        Args:
            path: Path to save the model. If None, uses default path.
            include_optimizer: Whether to include optimizer state
            include_scheduler: Whether to include scheduler state
            include_metrics: Whether to include metrics
            include_config: Whether to include config
            
        Returns:
            str: Path where the model was saved
            
        Raises:
            RuntimeError: If model is None or saving fails
        """
        if self.model is None:
            raise RuntimeError("No model to save")
        
        self.logger.log_info("Saving model...")
        
        timer = Timer()
        timer.start()
        
        try:
            # Prepare state dictionary
            state = {
                'model_state_dict': self.model.state_dict(),
                'model_class': self.model.__class__.__name__,
                'model_architecture': str(self.model),
                'epoch': self.current_epoch,
                'step': self.current_step,
                'timestamp': Timer.get_current_timestamp(),
                'device': self.device
            }
            
            if include_optimizer and self.optimizer is not None:
                state['optimizer_state_dict'] = self.optimizer.state_dict()
            
            if include_scheduler and self.scheduler is not None:
                state['scheduler_state_dict'] = self.scheduler.state_dict()
            
            if include_metrics:
                state['metrics'] = self.metrics
            
            if include_config:
                state['config'] = self.config
            
            # Generate default path if not provided
            if path is None:
                timestamp = Timer.get_current_timestamp()
                model_name = self.model.__class__.__name__.lower()
                path = os.path.join(
                    self.checkpoint_manager.save_dir,
                    f"{model_name}_epoch{self.current_epoch}_{timestamp}.pt"
                )
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model
            torch.save(state, path)
            
            timer.stop()
            self.logger.log_info(f"Model saved to {path} in {timer.get_elapsed_time():.2f} seconds")
            
            return path
            
        except Exception as e:
            self.logger.log_error(f"Failed to save model: {e}")
            raise RuntimeError(f"Failed to save model: {e}")
    
    def load_model(
        self,
        path: str,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        load_metrics: bool = True,
        load_config: bool = False,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load a saved model from disk
        
        Args:
            path: Path to the saved model file
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            load_metrics: Whether to load metrics
            load_config: Whether to load config (use with caution)
            strict: Whether to strictly enforce state dict compatibility
            
        Returns:
            Dict[str, Any]: Loaded state dictionary
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            RuntimeError: If loading fails
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.logger.log_info(f"Loading model from {path}...")
        
        timer = Timer()
        timer.start()
        
        try:
            # Load checkpoint
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                if self.model is None:
                    raise RuntimeError("Model is None. Cannot load state dict.")
                
                self.model.load_state_dict(
                    checkpoint['model_state_dict'],
                    strict=strict
                )
                self.logger.log_info("Model state dict loaded successfully")
            else:
                self.logger.log_warning("No model_state_dict found in checkpoint")
            
            # Load optimizer state
            if load_optimizer and 'optimizer_state_dict' in checkpoint:
                if self.optimizer is not None:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.logger.log_info("Optimizer state loaded successfully")
                else:
                    self.logger.log_warning("Optimizer is None. Cannot load optimizer state.")
            
            # Load scheduler state
            if load_scheduler and 'scheduler_state_dict' in checkpoint:
                if self.scheduler is not None:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    self.logger.log_info("Scheduler state loaded successfully")
                else:
                    self.logger.log_warning("Scheduler is None. Cannot load scheduler state.")
            
            # Load metrics
            if load_metrics and 'metrics' in checkpoint:
                self.metrics.update(checkpoint['metrics'])
                self.logger.log_info("Metrics loaded successfully")
            
            # Load config (with caution)
            if load_config and 'config' in checkpoint:
                self.config.update(checkpoint['config'])
                self.logger.log_warning("Config loaded from checkpoint. This may overwrite current config.")
            
            # Update state
            self.current_epoch = checkpoint.get('epoch', 0)
            self.current_step = checkpoint.get('step', 0)
            self.is_loaded = True
            
            timer.stop()
            self.logger.log_info(f"Model loaded in {timer.get_elapsed_time():.2f} seconds")
            
            return checkpoint
            
        except Exception as e:
            self.logger.log_error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        """
        Log metrics to the logger
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number (epoch or iteration)
            prefix: Prefix for log messages
        """
        if step is None:
            step = self.current_step
        
        log_message = f"{prefix}Step {step} - "
        metric_strings = []
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_strings.append(f"{key}: {value:.4f}")
            elif isinstance(value, str):
                metric_strings.append(f"{key}: {value}")
            else:
                metric_strings.append(f"{key}: {value}")
        
        log_message += " - ".join(metric_strings)
        self.logger.log_info(log_message)
        
        # Also store in training history
        for key, value in metrics.items():
            if key not in self.metrics['training_history']:
                self.metrics['training_history'][key] = []
            self.metrics['training_history'][key].append(value)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all training metrics
        
        Returns:
            Dict[str, Any]: Dictionary of all metrics
        """
        return self.metrics.copy()
    
    def get_best_model_state(self) -> Optional[Dict[str, Any]]:
        """
        Get the state dictionary of the best performing model
        
        Returns:
            Optional[Dict[str, Any]]: Best model state or None if not available
        """
        return self.best_model_state
    
    def reset_metrics(self) -> None:
        """
        Reset all metrics to initial state
        """
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'test_accuracy': [],
            'epochs': 0,
            'best_val_loss': float('inf'),
            'best_val_accuracy': 0.0,
            'best_epoch': -1,
            'start_time': Timer.get_current_timestamp(),
            'end_time': None,
            'total_time': 0.0,
            'training_history': defaultdict(list)
        }
        self.logger.log_info("Metrics reset")
    
    def set_model(self, model: nn.Module) -> None:
        """
        Set the model and move it to the appropriate device
        
        Args:
            model: PyTorch model to set
        """
        self.model = model
        if self.model is not None:
            self.model.to(self.device)
            self.logger.log_info(f"Model set and moved to device: {self.device}")
        else:
            self.logger.log_warning("Model set to None")
    
    def set_optimizer(self, optimizer: Optimizer) -> None:
        """
        Set the optimizer
        
        Args:
            optimizer: Optimizer to set
        """
        self.optimizer = optimizer
        self.logger.log_info("Optimizer set")
    
    def set_scheduler(self, scheduler: _LRScheduler) -> None:
        """
        Set the learning rate scheduler
        
        Args:
            scheduler: Learning rate scheduler to set
        """
        self.scheduler = scheduler
        self.logger.log_info("Scheduler set")
    
    def set_data_loader(self, data_loader: DataLoader) -> None:
        """
        Set the data loader
        
        Args:
            data_loader: Data loader to set
        """
        self.data_loader = data_loader
        self.logger.log_info("Data loader set")
    
    def get_device(self) -> str:
        """
        Get the current device
        
        Returns:
            str: Device name
        """
        return self.device
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Move a tensor to the current device
        
        Args:
            tensor: Tensor to move
            
        Returns:
            torch.Tensor: Tensor on the current device
        """
        return tensor.to(self.device)
    
    def save_checkpoint(
        self,
        checkpoint_name: Optional[str] = None,
        include_model: bool = True,
        include_optimizer: bool = True,
        include_scheduler: bool = True,
        include_metrics: bool = True
    ) -> str:
        """
        Save a training checkpoint
        
        Args:
            checkpoint_name: Name of the checkpoint
            include_model: Whether to include model state
            include_optimizer: Whether to include optimizer state
            include_scheduler: Whether to include scheduler state
            include_metrics: Whether to include metrics
            
        Returns:
            str: Path where checkpoint was saved
        """
        # Prepare checkpoint state
        checkpoint_state = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'timestamp': Timer.get_current_timestamp(),
            'device': self.device
        }
        
        if include_model and self.model is not None:
            checkpoint_state['model_state_dict'] = self.model.state_dict()
        
        if include_optimizer and self.optimizer is not None:
            checkpoint_state['optimizer_state_dict'] = self.optimizer.state_dict()
        
        if include_scheduler and self.scheduler is not None:
            checkpoint_state['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if include_metrics:
            checkpoint_state['metrics'] = self.metrics
        
        checkpoint_state['config'] = self.config
        
        # Save using checkpoint manager
        return self.checkpoint_manager.save_checkpoint(
            state=checkpoint_state,
            epoch=self.current_epoch,
            step=self.current_step,
            name=checkpoint_name
        )
    
    def load_checkpoint(
        self,
        checkpoint_name: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        load_model: bool = True,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        load_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Load a training checkpoint
        
        Args:
            checkpoint_name: Name of the checkpoint to load
            checkpoint_path: Direct path to checkpoint file
            load_model: Whether to load model state
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            load_metrics: Whether to load metrics
            
        Returns:
            Dict[str, Any]: Loaded checkpoint state
            
        Raises:
            FileNotFoundError: If checkpoint not found
        """
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        elif checkpoint_name:
            checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        else:
            latest = self.checkpoint_manager.get_latest_checkpoint()
            if latest:
                checkpoint = self.checkpoint_manager.load_checkpoint(latest)
            else:
                raise FileNotFoundError("No checkpoint found to load")
        
        if not checkpoint:
            raise RuntimeError("Failed to load checkpoint")
        
        # Load model state
        if load_model and 'model_state_dict' in checkpoint and self.model is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.log_info("Model state loaded from checkpoint")
        
        # Load optimizer state
        if load_optimizer and 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.log_info("Optimizer state loaded from checkpoint")
        
        # Load scheduler state
        if load_scheduler and 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.logger.log_info("Scheduler state loaded from checkpoint")
        
        # Load metrics
        if load_metrics and 'metrics' in checkpoint:
            self.metrics.update(checkpoint['metrics'])
            self.logger.log_info("Metrics loaded from checkpoint")
        
        # Update state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.current_step = checkpoint.get('step', 0)
        self.is_loaded = True
        
        return checkpoint
    
    def update_best_model(self, val_metric: float, metric_name: str = 'loss') -> bool:
        """
        Update the best model if current validation metric is better
        
        Args:
            val_metric: Current validation metric value
            metric_name: Name of the metric ('loss' or 'accuracy')
            
        Returns:
            bool: True if model was updated, False otherwise
        """
        is_better = False
        
        if metric_name == 'loss':
            if val_metric < self.metrics['best_val_loss']:
                self.metrics['best_val_loss'] = val_metric
                self.metrics['best_epoch'] = self.current_epoch
                is_better = True
        elif metric_name == 'accuracy':
            if val_metric > self.metrics['best_val_accuracy']:
                self.metrics['best_val_accuracy'] = val_metric
                self.metrics['best_epoch'] = self.current_epoch
                is_better = True
        else:
            raise ValueError(f"Unknown metric_name: {metric_name}")
        
        if is_better:
            # Save best model state
            if self.model is not None:
                self.best_model_state = {
                    'state_dict': self.model.state_dict(),
                    'epoch': self.current_epoch,
                    'metric': val_metric
                }
                self.logger.log_info(f"New best model (epoch {self.current_epoch}) with {metric_name}: {val_metric:.4f}")
        
        return is_better
    
    def restore_best_model(self) -> None:
        """
        Restore the best model state
        """
        if self.best_model_state is not None and self.model is not None:
            self.model.load_state_dict(self.best_model_state['state_dict'])
            self.logger.log_info(
                f"Restored best model from epoch {self.best_model_state['epoch']} "
                f"with metric: {self.best_model_state['metric']:.4f}"
            )
        else:
            self.logger.log_warning("No best model state available to restore")
    
    def get_trainable_parameters(self) -> int:
        """
        Get the number of trainable parameters in the model
        
        Returns:
            int: Number of trainable parameters
        """
        if self.model is None:
            return 0
        
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_total_parameters(self) -> int:
        """
        Get the total number of parameters in the model
        
        Returns:
            int: Total number of parameters
        """
        if self.model is None:
            return 0
        
        return sum(p.numel() for p in self.model.parameters())
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model
        
        Returns:
            Dict[str, Any]: Model summary information
        """
        return {
            'model_class': self.model.__class__.__name__ if self.model else None,
            'total_parameters': self.get_total_parameters(),
            'trainable_parameters': self.get_trainable_parameters(),
            'device': self.device,
            'is_trained': self.is_trained,
            'is_loaded': self.is_loaded,
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'best_val_loss': self.metrics['best_val_loss'],
            'best_val_accuracy': self.metrics['best_val_accuracy'],
            'best_epoch': self.metrics['best_epoch']
        }
    
    def generate_report(self) -> str:
        """
        Generate a training report
        
        Returns:
            str: Formatted training report
        """
        report_lines = [
            "=" * 60,
            "TRAINING REPORT",
            "=" * 60,
            "",
            f"Model: {self.model.__class__.__name__ if self.model else 'None'}",
            f"Device: {self.device}",
            f"Total Parameters: {self.get_total_parameters():,}",
            f"Trainable Parameters: {self.get_trainable_parameters():,}",
            f"Trained: {self.is_trained}",
            f"Loaded: {self.is_loaded}",
            f"Current Epoch: {self.current_epoch}",
            f"Current Step: {self.current_step}",
            f"Start Time: {self.metrics['start_time']}",
            f"End Time: {self.metrics.get('end_time', 'N/A')}",
            f"Total Time: {self.metrics.get('total_time', 0.0):.2f}s",
            "",
            "BEST PERFORMANCE:",
            f"  Best Val Loss: {self.metrics['best_val_loss']:.4f} (Epoch {self.metrics['best_epoch']})",
            f"  Best Val Accuracy: {self.metrics['best_val_accuracy']:.4f} (Epoch {self.metrics['best_epoch']})",
            "",
            "TRAINING HISTORY:",
        ]
        
        for metric_name, values in self.metrics['training_history'].items():
            if values:
                latest_value = values[-1] if values else None
                if isinstance(latest_value, (int, float)):
                    report_lines.append(f"  {metric_name}: latest={latest_value:.4f}, avg={np.mean(values[-10:]):.4f}")
                else:
                    report_lines.append(f"  {metric_name}: {latest_value}")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def save_report(self, path: Optional[str] = None) -> str:
        """
        Save the training report to a file
        
        Args:
            path: Path to save the report. If None, uses default path.
            
        Returns:
            str: Path where the report was saved
        """
        if path is None:
            timestamp = Timer.get_current_timestamp()
            path = os.path.join(
                self.logger.log_dir,
                f"training_report_{timestamp}.txt"
            )
        
        report = self.generate_report()
        
        with open(path, 'w') as f:
            f.write(report)
        
        self.logger.log_info(f"Training report saved to {path}")
        return path
    
    # Private helper methods
    
    def _get_default_device(self) -> str:
        """Get the default device"""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _create_default_logger(self) -> Logger:
        """Create a default logger if none provided"""
        log_dir = self.config.get('common', {}).get('log_dir', './logs')
        name = self.__class__.__name__.lower()
        return Logger(log_dir=log_dir, name=name)
    
    def _create_default_checkpoint_manager(self) -> CheckpointManager:
        """Create a default checkpoint manager if none provided"""
        save_dir = self.config.get('common', {}).get('checkpoint_dir', './checkpoints')
        max_checkpoints = self.config.get('common', {}).get('max_checkpoints', 5)
        return CheckpointManager(save_dir=save_dir, max_checkpoints=max_checkpoints)
    
    def _validate_data_loader(self) -> bool:
        """Validate that the data loader is properly set up"""
        if self.data_loader is None:
            self.logger.log_warning("Data loader is None")
            return False
        
        if hasattr(self.data_loader, 'dataset') and self.data_loader.dataset is None:
            self.logger.log_warning("Data loader dataset is None")
            return False
        
        return True
    
    def _validate_model(self) -> bool:
        """Validate that the model is properly set up"""
        if self.model is None:
            self.logger.log_warning("Model is None")
            return False
        
        return True
    
    def _log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_acc: Optional[float] = None,
        val_acc: Optional[float] = None,
        learning_rate: Optional[float] = None
    ) -> None:
        """
        Log a summary of the current epoch
        
        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
            train_loss: Training loss
            val_loss: Validation loss (optional)
            train_acc: Training accuracy (optional)
            val_acc: Validation accuracy (optional)
            learning_rate: Current learning rate (optional)
        """
        summary = f"Epoch {epoch}/{total_epochs} - "
        summary += f"Train Loss: {train_loss:.4f}"
        
        if train_acc is not None:
            summary += f", Train Acc: {train_acc:.4f}"
        
        if val_loss is not None:
            summary += f", Val Loss: {val_loss:.4f}"
        
        if val_acc is not None:
            summary += f", Val Acc: {val_acc:.4f}"
        
        if learning_rate is not None:
            summary += f", LR: {learning_rate:.6f}"
        
        self.logger.log_info(summary)
    
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            metrics: List of metrics to compute ('accuracy', 'precision', 'recall', etc.)
            
        Returns:
            Dict[str, float]: Computed metrics
        """
        if metrics is None:
            metrics = ['accuracy']
        
        result = {}
        
        with torch.no_grad():
            if 'accuracy' in metrics:
                pred_labels = predictions.argmax(dim=-1) if predictions.dim() > 1 else predictions
                correct = (pred_labels == targets).float()
                result['accuracy'] = correct.mean().item()
            
            if 'precision' in metrics:
                # Binary precision
                pred_labels = predictions.argmax(dim=-1) if predictions.dim() > 1 else predictions
                true_pos = ((pred_labels == 1) & (targets == 1)).float().sum()
                predicted_pos = (pred_labels == 1).float().sum()
                result['precision'] = (true_pos / predicted_pos).item() if predicted_pos > 0 else 0.0
            
            if 'recall' in metrics:
                pred_labels = predictions.argmax(dim=-1) if predictions.dim() > 1 else predictions
                true_pos = ((pred_labels == 1) & (targets == 1)).float().sum()
                actual_pos = (targets == 1).float().sum()
                result['recall'] = (true_pos / actual_pos).item() if actual_pos > 0 else 0.0
            
            if 'f1' in metrics:
                precision = result.get('precision', 0.0)
                recall = result.get('recall', 0.0)
                result['f1'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return result
    
    def _get_config_hash(self) -> str:
        """Get a hash of the configuration for reproducibility"""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _get_model_hash(self) -> str:
        """Get a hash of the model state for reproducibility"""
        if self.model is None:
            return "no_model"
        
        state_dict = self.model.state_dict()
        state_str = str({k: v.shape for k, v in state_dict.items()})
        return hashlib.md5(state_str.encode()).hexdigest()[:8]
    
    def __str__(self) -> str:
        """String representation of the trainer"""
        return f"{self.__class__.__name__}(model={self.model.__class__.__name__ if self.model else None}, device={self.device})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the trainer"""
        return self.__str__()


class TrainerFactory:
    """
    Factory class for creating trainers based on configuration
    """
    
    @staticmethod
    def create_trainer(
        trainer_type: str,
        config: Dict[str, Any],
        **kwargs
    ) -> BaseTrainer:
        """
        Create a trainer instance based on type
        
        Args:
            trainer_type: Type of trainer to create ('phase1', 'phase2', 'phase3')
            config: Configuration dictionary
            **kwargs: Additional arguments for the specific trainer
            
        Returns:
            BaseTrainer: Trainer instance
            
        Raises:
            ValueError: If trainer_type is unknown
        """
        if trainer_type == 'phase1':
            from .phase1_bootstrap import Phase1Bootstrap
            return Phase1Bootstrap(**kwargs)
        elif trainer_type == 'phase2':
            from .phase2_distillation import Phase2Distillation
            return Phase2Distillation(**kwargs)
        elif trainer_type == 'phase3':
            from .phase3_hybrid import Phase3Hybrid
            return Phase3Hybrid(**kwargs)
        else:
            raise ValueError(f"Unknown trainer type: {trainer_type}")
    
    @staticmethod
    def get_available_trainers() -> List[str]:
        """
        Get list of available trainer types
        
        Returns:
            List[str]: List of trainer type names
        """
        return ['phase1', 'phase2', 'phase3']
    
    @staticmethod
    def get_trainer_description(trainer_type: str) -> str:
        """
        Get description of a trainer type
        
        Args:
            trainer_type: Trainer type
            
        Returns:
            str: Description of the trainer
            
        Raises:
            ValueError: If trainer_type is unknown
        """
        descriptions = {
            'phase1': 'Bootstrap trainer for agent-based collaborative reflection',
            'phase2': 'Distillation trainer for knowledge transfer from LLM to GNN',
            'phase3': 'Hybrid trainer for adaptive gating between GNN and LLM paths'
        }
        
        if trainer_type in descriptions:
            return descriptions[trainer_type]
        else:
            raise ValueError(f"Unknown trainer type: {trainer_type}")


# Export common utilities and classes
__all__ = [
    'BaseTrainer',
    'TrainerFactory'
]