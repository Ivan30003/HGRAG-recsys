"""
GNN Trainer Module
Handles training loop, optimization, and checkpointing for Phase 2 distillation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import time
import json
import numpy as np
from collections import defaultdict

from .hgnn import HeterogeneousGNN, LightDecoder, TierSpecificLoss

logger = logging.getLogger(__name__)


class GNNTrainer:
    """
    Trainer for Phase 2: Tier-Disentangled Knowledge Distillation.
    
    Handles:
    - Training loop with multiple loss components
    - Validation and early stopping
    - Learning rate scheduling
    - Gradient clipping and monitoring
    - Checkpointing and resumption
    """
    
    def __init__(self,
                 hgnn: HeterogeneousGNN,
                 decoder: Optional[LightDecoder] = None,
                 loss_module: Optional[TierSpecificLoss] = None,
                 config: Optional[Dict] = None):
        """
        Initialize trainer.
        
        Args:
            hgnn: HeterogeneousGNN model
            decoder: Optional LightDecoder for text reconstruction
            loss_module: Optional TierSpecificLoss
            config: Training configuration
        """
        self.hgnn = hgnn
        self.decoder = decoder
        self.loss_module = loss_module or TierSpecificLoss()
        self.config = config or {}
        
        # Training configuration
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.weight_decay = self.config.get('weight_decay', 0.0001)
        self.max_epochs = self.config.get('num_epochs', 200)
        self.gradient_clip = self.config.get('gradient_clip', 1.0)
        self.early_stopping_patience = self.config.get('early_stopping_patience', 20)
        
        # Device
        self.device = torch.device(
            self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Move models to device
        self.hgnn.to(self.device)
        if self.decoder:
            self.decoder.to(self.device)
        
        # Optimizer
        params = list(self.hgnn.parameters())
        if self.decoder:
            params.extend(list(self.decoder.parameters()))
        
        self.optimizer = AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )
        
        # Optional secondary scheduler
        self.plateau_scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Training history
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        
        # Timing
        self.start_time = None
        self.epoch_times = []
    
    def train_epoch(self,
                    dataloader: DataLoader,
                    epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Dictionary of average losses
        """
        self.hgnn.train()
        if self.decoder:
            self.decoder.train()
        
        epoch_losses = defaultdict(float)
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Extract batch components
            node_features = batch['node_features']
            adjacency_lists = batch.get('adjacency_lists', [{} for _ in range(4)])
            edge_weights = batch.get('edge_weights', None)
            
            # Target embeddings from LLM
            targets = (
                batch['h_intrinsic_target'],
                batch['h_collaborative_target'],
                batch['h_interaction_target']
            )
            
            # Forward pass through HGNN
            predictions = self.hgnn(node_features, adjacency_lists, edge_weights)
            
            # Path importance (if available)
            path_imp_pred = None
            path_imp_target = batch.get('path_importance')
            
            if path_imp_target is not None and hasattr(self.hgnn, 'get_attention_weights'):
                attn_weights = self.hgnn.get_attention_weights(
                    node_features, 
                    adjacency_lists[0] if adjacency_lists else {},
                    edge_type_idx=0
                )
                if attn_weights:
                    path_imp_pred = self._aggregate_attention(attn_weights)
            
            # Reconstruction (if decoder is available)
            recon_logits = None
            recon_targets = batch.get('text_tokens')
            
            if self.decoder and recon_targets is not None:
                h_int, h_col, h_intr = predictions
                recon_logits = self.decoder(h_int, h_col, h_intr, recon_targets)
            
            # Compute losses
            losses = self.loss_module(
                predictions, targets,
                path_imp_pred, path_imp_target,
                recon_logits, recon_targets
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['loss_total'].backward()
            
            # Gradient clipping
            params = list(self.hgnn.parameters())
            if self.decoder:
                params.extend(list(self.decoder.parameters()))
            torch.nn.utils.clip_grad_norm_(params, self.gradient_clip)
            
            self.optimizer.step()
            
            # Track losses
            for key, value in losses.items():
                epoch_losses[key] += value.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 50 == 0:
                logger.debug(f"Epoch {epoch}, Batch {batch_idx}: "
                           f"Loss={losses['loss_total'].item():.4f}")
        
        # Average losses
        avg_losses = {
            key: val / max(1, num_batches)
            for key, val in epoch_losses.items()
        }
        
        return avg_losses
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
        
        Returns:
            Dictionary of average validation losses
        """
        self.hgnn.eval()
        if self.decoder:
            self.decoder.eval()
        
        val_losses = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)
                
                node_features = batch['node_features']
                adjacency_lists = batch.get('adjacency_lists', [{} for _ in range(4)])
                edge_weights = batch.get('edge_weights', None)
                
                targets = (
                    batch['h_intrinsic_target'],
                    batch['h_collaborative_target'],
                    batch['h_interaction_target']
                )
                
                predictions = self.hgnn(node_features, adjacency_lists, edge_weights)
                
                losses = self.loss_module(predictions, targets)
                
                for key, value in losses.items():
                    val_losses[key] += value.item()
                num_batches += 1
        
        avg_losses = {
            key: val / max(1, num_batches)
            for key, val in val_losses.items()
        }
        
        return avg_losses
    
    def train(self,
              train_dataloader: DataLoader,
              val_dataloader: Optional[DataLoader] = None,
              checkpoint_dir: Optional[str] = None):
        """
        Run full training loop.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            checkpoint_dir: Directory for saving checkpoints
        """
        self.start_time = time.time()
        logger.info(f"Starting training for {self.max_epochs} epochs...")
        
        for epoch in range(self.current_epoch, self.max_epochs):
            epoch_start = time.time()
            
            # Training
            train_losses = self.train_epoch(train_dataloader, epoch)
            for key, val in train_losses.items():
                self.train_history[key].append(val)
            
            # Validation
            if val_dataloader:
                val_losses = self.validate(val_dataloader)
                for key, val in val_losses.items():
                    self.val_history[key].append(val)
                
                val_total = val_losses.get('loss_total', float('inf'))
            else:
                val_total = train_losses.get('loss_total', float('inf'))
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Track epoch time
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)
            
            # Logging
            log_msg = f"Epoch {epoch+1}/{self.max_epochs} "
            log_msg += f"| Train Loss: {train_losses.get('loss_total', 0):.4f}"
            if val_dataloader:
                log_msg += f" | Val Loss: {val_total:.4f}"
            log_msg += f" | Time: {epoch_time:.1f}s"
            logger.info(log_msg)
            
            # Checkpointing
            if checkpoint_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(checkpoint_dir, epoch, is_best=False)
            
            # Early stopping
            if val_total < self.best_val_loss:
                self.best_val_loss = val_total
                self.best_epoch = epoch
                self.patience_counter = 0
                
                if checkpoint_dir:
                    self.save_checkpoint(checkpoint_dir, epoch, is_best=True)
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            self.current_epoch = epoch + 1
        
        total_time = time.time() - self.start_time
        logger.info(f"Training complete in {total_time:.1f}s. "
                   f"Best val loss: {self.best_val_loss:.4f} at epoch {self.best_epoch+1}")
    
    def save_checkpoint(self, 
                         checkpoint_dir: str, 
                         epoch: int, 
                         is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            epoch: Current epoch
            is_best: Whether this is the best model
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'hgnn_state_dict': self.hgnn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history),
            'config': self.config
        }
        
        if self.decoder:
            checkpoint['decoder_state_dict'] = self.decoder.state_dict()
        
        # Save checkpoint
        filename = f"checkpoint_epoch_{epoch+1}.pt"
        filepath = Path(checkpoint_dir) / filename
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = Path(checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model checkpoint to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.hgnn.load_state_dict(checkpoint['hgnn_state_dict'])
        
        if self.decoder and 'decoder_state_dict' in checkpoint:
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if 'train_history' in checkpoint:
            self.train_history = defaultdict(list, checkpoint['train_history'])
        if 'val_history' in checkpoint:
            self.val_history = defaultdict(list, checkpoint['val_history'])
        
        logger.info(f"Resumed from epoch {self.current_epoch}")
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            elif isinstance(value, dict):
                moved[key] = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            elif isinstance(value, list):
                moved[key] = [
                    self._move_batch_to_device(v) if isinstance(v, dict) 
                    else v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for v in value
                ]
            else:
                moved[key] = value
        return moved
    
    def _aggregate_attention(self, attn_weights: Dict) -> torch.Tensor:
        """Aggregate attention weights for path importance."""
        if not attn_weights:
            return None
        
        all_weights = []
        for src_idx, weights in attn_weights.items():
            all_weights.append(np.mean(list(weights.values())))
        
        if not all_weights:
            return None
        
        return torch.tensor(np.mean(all_weights), device=self.device).unsqueeze(0)
    
    def get_training_summary(self) -> Dict:
        """Get training summary statistics."""
        return {
            'total_epochs': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'total_time': time.time() - self.start_time if self.start_time else 0,
            'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
            'final_train_loss': self.train_history.get('loss_total', [0])[-1] if self.train_history.get('loss_total') else 0,
            'final_val_loss': self.val_history.get('loss_total', [0])[-1] if self.val_history.get('loss_total') else 0
        }