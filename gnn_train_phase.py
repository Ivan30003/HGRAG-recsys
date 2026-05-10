"""
GNN Training Phase: Tier-Disentangled Knowledge Distillation
Phase 2 of the Hybrid-GraphRAG pipeline.

Distills LLM-generated memory dynamics from bootstrap traces
into efficient heterogeneous graph neural encoders.
"""

import argparse
import yaml
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from gnn_utils.hgnn import HeterogeneousGNN, LightDecoder
from graph_utils.heterogeneous_graph import HeterogeneousGraph
from llm_utils.embedding_client import EmbeddingClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReflectionTraceDataset(Dataset):
    """
    Dataset of reflection traces from Phase 1 bootstrap.
    Converts traces into training samples for HGNN distillation.
    """
    
    def __init__(self, 
                 traces: List[Dict],
                 embedding_client: EmbeddingClient,
                 graph: HeterogeneousGraph):
        """
        Initialize dataset from reflection traces.
        
        Args:
            traces: List of reflection trace dictionaries
            embedding_client: Client for text-to-embedding conversion
            graph: Heterogeneous graph with agent information
        """
        self.traces = traces
        self.embedding_client = embedding_client
        self.graph = graph
        
        # Pre-compute embeddings for all traces
        self.samples = []
        self._prepare_samples()
    
    def _prepare_samples(self):
        """Pre-compute training samples from traces."""
        logger.info(f"Preparing {len(self.traces)} training samples...")
        
        for trace in self.traces:
            try:
                sample = self._trace_to_sample(trace)
                if sample is not None:
                    self.samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to process trace: {e}")
                continue
        
        logger.info(f"Prepared {len(self.samples)} valid samples")
    
    def _trace_to_sample(self, trace: Dict) -> Optional[Dict]:
        """
        Convert a single trace to a training sample.
        
        Args:
            trace: Reflection trace dictionary
        
        Returns:
            Training sample dict or None if invalid
        """
        user_id = trace['user_id']
        pos_item_id = trace['positive_item_id']
        
        # Get agent memories before update
        user_memory_before = trace.get('user_memory_before', {})
        pos_item_memory_before = trace.get('pos_item_memory_before', {})
        
        # Get agent memories after update (from reflection result)
        reflection_result = trace.get('reflection_result')
        
        if not reflection_result:
            return None  # Skip correct predictions (no reflection)
        
        # Encode text memories to embeddings
        # User embeddings
        user_int_text = self._get_memory_text(user_memory_before, 'user', 'intrinsic')
        user_col_text_before = self._get_memory_text(user_memory_before, 'user', 'collaborative')
        user_intr_text = self._get_memory_text(user_memory_before, 'user', 'interaction')
        
        # Item embeddings
        pos_int_text = self._get_memory_text(pos_item_memory_before, 'item', 'intrinsic')
        pos_col_text_before = self._get_memory_text(pos_item_memory_before, 'item', 'collaborative')
        pos_intr_text = self._get_memory_text(pos_item_memory_before, 'item', 'interaction')
        
        # Encode to embeddings (target values)
        user_int_emb = self.embedding_client.encode(user_int_text)
        user_col_emb = self.embedding_client.encode(user_col_text_before)
        user_intr_emb = self.embedding_client.encode(user_intr_text)
        
        pos_int_emb = self.embedding_client.encode(pos_int_text)
        pos_col_emb = self.embedding_client.encode(pos_col_text_before)
        pos_intr_emb = self.embedding_client.encode(pos_intr_text)
        
        # Get graph context features
        graph_context = trace.get('graph_context', {})
        influential_paths = trace.get('influential_paths', [])
        
        # Build adjacency information
        adj_info = self._build_adjacency_info(user_id, pos_item_id, graph_context)
        
        # Path importance weights (for distillation loss)
        path_importance = self._compute_path_importance(influential_paths)
        
        return {
            'user_id': user_id,
            'pos_item_id': pos_item_id,
            'user_int_emb': torch.tensor(user_int_emb, dtype=torch.float32),
            'user_col_emb': torch.tensor(user_col_emb, dtype=torch.float32),
            'user_intr_emb': torch.tensor(user_intr_emb, dtype=torch.float32),
            'pos_int_emb': torch.tensor(pos_int_emb, dtype=torch.float32),
            'pos_col_emb': torch.tensor(pos_col_emb, dtype=torch.float32),
            'pos_intr_emb': torch.tensor(pos_intr_emb, dtype=torch.float32),
            'adj_info': adj_info,
            'path_importance': torch.tensor(path_importance, dtype=torch.float32),
            'is_correct': trace.get('is_correct', False)
        }
    
    def _get_memory_text(self, memory_dict: Dict, agent_type: str, tier: str) -> str:
        """Extract text from memory dictionary."""
        if tier == 'intrinsic':
            return memory_dict.get('intrinsic_memory', {}).get('summary_text', '')
        elif tier == 'collaborative':
            patterns = memory_dict.get('collaborative_memory', {}).get('preference_patterns', [])
            return ' '.join(patterns) if patterns else 'no patterns'
        elif tier == 'interaction':
            return memory_dict.get('interaction_memory', {}).get('recent_context', '')
        return ''
    
    def _build_adjacency_info(self, 
                               user_id: str, 
                               item_id: str,
                               graph_context: Dict) -> Dict:
        """
        Build adjacency information for HGNN input.
        
        Args:
            user_id: Center user ID
            item_id: Center item ID
            graph_context: Graph context from retrieval
        
        Returns:
            Dictionary with adjacency lists per edge type
        """
        adj_info = {
            'interact': defaultdict(list),
            'similar_pref': defaultdict(list),
            'co_interact': defaultdict(list),
            'content_sim': defaultdict(list)
        }
        
        # Get neighbors from graph context
        neighbors_1hop = graph_context.get('neighbors_1hop', {})
        neighbors_2hop = graph_context.get('neighbors_2hop', {})
        
        # Build adjacency from context (simplified indexing)
        node_to_idx = {user_id: 0, item_id: 1}
        current_idx = 2
        
        for edge_type in adj_info.keys():
            edges = graph_context.get(f'{edge_type}_edges', [])
            for edge in edges:
                src = edge.get('source', '')
                tgt = edge.get('target', '')
                
                if src not in node_to_idx:
                    node_to_idx[src] = current_idx
                    current_idx += 1
                if tgt not in node_to_idx:
                    node_to_idx[tgt] = current_idx
                    current_idx += 1
                
                adj_info[edge_type][node_to_idx[src]].append(node_to_idx[tgt])
        
        return {
            'adjacency': dict(adj_info),
            'node_to_idx': node_to_idx,
            'num_nodes': current_idx
        }
    
    def _compute_path_importance(self, influential_paths: List[Dict]) -> np.ndarray:
        """
        Compute importance weights for metapaths.
        
        Args:
            influential_paths: List of influential path dictionaries
        
        Returns:
            numpy array of importance weights
        """
        if not influential_paths:
            return np.ones(3) / 3  # Uniform if no info
        
        # Count by metapath type
        path_counts = defaultdict(int)
        for path in influential_paths:
            path_type = path.get('type', 'unknown')
            path_counts[path_type] += 1
        
        # Map to 3 metapath types
        importance = np.zeros(3)
        type_map = {'user-item-user': 0, 'user-user-item': 1, 'item-item-analogy': 2}
        
        for path_type, count in path_counts.items():
            idx = type_map.get(path_type, -1)
            if idx >= 0:
                importance[idx] = count
        
        # Normalize
        total = importance.sum()
        if total > 0:
            importance = importance / total
        else:
            importance = np.ones(3) / 3
        
        return importance
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


class DistillationTrainer:
    """
    Trainer for Phase 2: Tier-Disentangled Knowledge Distillation.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize distillation trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.distill_config = config['distillation']
        
        # Set device
        self.device = torch.device(
            config['experiment']['device'] 
            if torch.cuda.is_available() 
            else 'cpu'
        )
        
        # Initialize models
        hgnn_config = self.distill_config['hgnn']
        self.hgnn = HeterogeneousGNN(
            input_dim=hgnn_config['input_dim'],
            hidden_dim=hgnn_config['hidden_dim'],
            output_dim=hgnn_config['output_dim'],
            num_layers=hgnn_config['num_layers'],
            dropout=hgnn_config['dropout']
        ).to(self.device)
        
        self.decoder = LightDecoder(
            embedding_dim=hgnn_config['output_dim'],
            hidden_dim=512,
            vocab_size=50000,
            max_length=100
        ).to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            list(self.hgnn.parameters()) + list(self.decoder.parameters()),
            lr=self.distill_config['learning_rate'],
            weight_decay=self.distill_config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2
        )
        
        # Loss weights
        loss_weights = self.distill_config['loss_weights']
        self.lambda_tier = loss_weights['tier_regression']
        self.lambda_path = loss_weights['path_importance']
        self.lambda_contrast = loss_weights['contrastive']
        self.lambda_recon = loss_weights['reconstruction']
        
        # Tier-specific weights
        tier_weights = loss_weights['tier_weights']
        self.tier_weights = {
            'intrinsic': tier_weights['intrinsic'],
            'collaborative': tier_weights['collaborative'],
            'interaction': tier_weights['interaction']
        }
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def compute_tier_regression_loss(self, 
                                      predicted: Tuple[torch.Tensor, ...],
                                      targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Compute MSE loss for tier-specific embeddings.
        
        Args:
            predicted: (h_int, h_col, h_intr) from HGNN
            targets: (target_int, target_col, target_intr) from LLM
        
        Returns:
            Weighted MSE loss
        """
        h_int_pred, h_col_pred, h_intr_pred = predicted
        h_int_target, h_col_target, h_intr_target = targets
        
        # MSE for each tier
        loss_int = F.mse_loss(h_int_pred, h_int_target)
        loss_col = F.mse_loss(h_col_pred, h_col_target)
        loss_intr = F.mse_loss(h_intr_pred, h_intr_target)
        
        # Weighted sum
        total_loss = (
            self.tier_weights['intrinsic'] * loss_int +
            self.tier_weights['collaborative'] * loss_col +
            self.tier_weights['interaction'] * loss_intr
        )
        
        return total_loss
    
    def compute_path_importance_loss(self,
                                      attention_weights: Dict[int, np.ndarray],
                                      target_importance: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss for path importance alignment.
        
        Args:
            attention_weights: HGNN attention weights
            target_importance: Path importance from LLM traces
        
        Returns:
            KL divergence loss
        """
        # Aggregate attention weights across nodes
        all_attentions = []
        for node_idx, weights in attention_weights.items():
            if len(weights) > 0:
                all_attentions.append(np.mean(weights))
        
        if not all_attentions:
            return torch.tensor(0.0, device=self.device)
        
        # Convert to tensor
        pred_importance = torch.tensor(
            np.mean(all_attentions), device=self.device
        ).unsqueeze(0)
        
        # Ensure target has same shape
        if target_importance.dim() == 1:
            target_importance = target_importance.mean().unsqueeze(0)
        
        # KL divergence (log-space for stability)
        pred_log = F.log_softmax(pred_importance.unsqueeze(0), dim=-1)
        target_soft = F.softmax(target_importance.unsqueeze(0), dim=-1)
        
        loss = F.kl_div(pred_log, target_soft, reduction='batchmean')
        
        return loss
    
    def compute_contrastive_loss(self,
                                  h_int: torch.Tensor,
                                  h_col: torch.Tensor,
                                  h_intr: torch.Tensor,
                                  temperature: float = 0.1) -> torch.Tensor:
        """
        Compute contrastive loss to separate tier embeddings.
        
        Args:
            h_int: Intrinsic embeddings (batch_size, dim)
            h_col: Collaborative embeddings (batch_size, dim)
            h_intr: Interaction embeddings (batch_size, dim)
            temperature: Temperature for softmax
        
        Returns:
            Contrastive loss
        """
        batch_size = h_int.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Stack all embeddings
        all_embeddings = torch.cat([h_int, h_col, h_intr], dim=0)  # (3*batch_size, dim)
        
        # Normalize
        all_embeddings = F.normalize(all_embeddings, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(all_embeddings, all_embeddings.t()) / temperature
        
        # Create labels: embeddings from same sample should be similar
        labels = torch.arange(batch_size, device=self.device)
        labels = labels.repeat(3)  # Each sample appears 3 times
        
        # Mask out self-similarity
        mask = torch.eye(3 * batch_size, dtype=torch.bool, device=self.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        
        # Contrastive loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def compute_reconstruction_loss(self,
                                     reconstructed_logits: torch.Tensor,
                                     target_text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute text reconstruction loss.
        
        Args:
            reconstructed_logits: Output from LightDecoder
            target_text_embeddings: Target text embeddings
        
        Returns:
            Reconstruction loss
        """
        # Simplified: MSE between mean pooled reconstructions and targets
        recon_mean = reconstructed_logits.mean(dim=1)  # (batch_size, vocab_size)
        
        # Project target to same space (simplified)
        target_proj = F.adaptive_avg_pool1d(
            target_text_embeddings.unsqueeze(1), 
            recon_mean.shape[-1]
        ).squeeze(1)
        
        loss = F.mse_loss(recon_mean[:, :target_proj.shape[-1]], target_proj)
        
        return loss
    
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
        self.decoder.train()
        
        epoch_losses = defaultdict(float)
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            user_int = batch['user_int_emb'].to(self.device)
            user_col = batch['user_col_emb'].to(self.device)
            user_intr = batch['user_intr_emb'].to(self.device)
            pos_int = batch['pos_int_emb'].to(self.device)
            pos_col = batch['pos_col_emb'].to(self.device)
            pos_intr = batch['pos_intr_emb'].to(self.device)
            path_imp = batch['path_importance'].to(self.device)
            
            # Build node features tensor (simplified for batch size 1)
            # In practice, use batching with padding
            batch_size = user_int.shape[0]
            
            for i in range(batch_size):
                # Prepare node features for this sample
                node_features = torch.stack([
                    user_int[i],  # User node
                    pos_int[i],   # Item node
                ], dim=0)  # (2, dim)
                
                # Prepare adjacency lists
                adj_info = batch['adj_info'][i] if isinstance(batch['adj_info'], list) else batch['adj_info']
                
                # Convert adjacency to format expected by HGNN
                adj_lists = self._prepare_adj_lists(adj_info, node_features.device)
                
                # Forward pass through HGNN
                h_int_pred, h_col_pred, h_intr_pred = self.hgnn(
                    node_features, adj_lists
                )
                
                # Prepare targets
                targets = (
                    torch.stack([user_int[i], pos_int[i]], dim=0),
                    torch.stack([user_col[i], pos_col[i]], dim=0),
                    torch.stack([user_intr[i], pos_intr[i]], dim=0)
                )
                
                # Compute losses
                # 1. Tier regression loss
                loss_tier = self.compute_tier_regression_loss(
                    (h_int_pred, h_col_pred, h_intr_pred),
                    targets
                )
                
                # 2. Path importance loss
                attn_weights = self.hgnn.get_attention_weights(
                    node_features, 
                    adj_lists[0],  # First edge type
                    edge_type_idx=0
                )
                loss_path = self.compute_path_importance_loss(
                    attn_weights, 
                    path_imp[i]
                )
                
                # 3. Contrastive loss
                # Combine all node embeddings
                h_int_combined = h_int_pred
                h_col_combined = h_col_pred
                h_intr_combined = h_intr_pred
                
                loss_contrast = self.compute_contrastive_loss(
                    h_int_combined, h_col_combined, h_intr_combined
                )
                
                # 4. Reconstruction loss
                recon_logits = self.decoder(
                    h_int_pred.mean(dim=0, keepdim=True),
                    h_col_pred.mean(dim=0, keepdim=True),
                    h_intr_pred.mean(dim=0, keepdim=True)
                )
                loss_recon = torch.tensor(0.0, device=self.device)  # Simplified
                
                # Total loss
                total_loss = (
                    self.lambda_tier * loss_tier +
                    self.lambda_path * loss_path +
                    self.lambda_contrast * loss_contrast +
                    self.lambda_recon * loss_recon
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.hgnn.parameters()) + list(self.decoder.parameters()),
                    max_norm=1.0
                )
                
                self.optimizer.step()
                
                # Track losses
                epoch_losses['total'] += total_loss.item()
                epoch_losses['tier'] += loss_tier.item()
                epoch_losses['path'] += loss_path.item()
                epoch_losses['contrast'] += loss_contrast.item()
                epoch_losses['recon'] += loss_recon.item()
                num_batches += 1
        
        # Average losses
        avg_losses = {
            key: val / max(1, num_batches) 
            for key, val in epoch_losses.items()
        }
        
        return avg_losses
    
    def _prepare_adj_lists(self, 
                           adj_info: Dict,
                           device: torch.device) -> List[Dict[int, List[int]]]:
        """
        Prepare adjacency lists in format expected by HGNN.
        
        Args:
            adj_info: Adjacency information dictionary
            device: Torch device
        
        Returns:
            List of adjacency dicts per edge type
        """
        if isinstance(adj_info, dict) and 'adjacency' in adj_info:
            adj_dict = adj_info['adjacency']
        else:
            adj_dict = adj_info
        
        # Convert to list per edge type
        edge_types = ['interact', 'similar_pref', 'co_interact', 'content_sim']
        adj_lists = []
        
        for edge_type in edge_types:
            adj_lists.append(adj_dict.get(edge_type, {}))
        
        return adj_lists
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
        
        Returns:
            Dictionary of average validation losses
        """
        self.hgnn.eval()
        self.decoder.eval()
        
        val_losses = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Similar to train_epoch but without gradient computation
                user_int = batch['user_int_emb'].to(self.device)
                pos_int = batch['pos_int_emb'].to(self.device)
                
                # Simplified validation
                node_features = torch.stack([
                    user_int[0], pos_int[0]
                ], dim=0)
                
                adj_info = batch['adj_info'][0] if isinstance(batch['adj_info'], list) else batch['adj_info']
                adj_lists = self._prepare_adj_lists(adj_info, self.device)
                
                h_int_pred, h_col_pred, h_intr_pred = self.hgnn(
                    node_features, adj_lists
                )
                
                # Simple MSE loss
                targets = (
                    torch.stack([user_int[0], pos_int[0]], dim=0),
                    torch.stack([user_int[0], pos_int[0]], dim=0),
                    torch.stack([user_int[0], pos_int[0]], dim=0)
                )
                
                loss = self.compute_tier_regression_loss(
                    (h_int_pred, h_col_pred, h_intr_pred),
                    targets
                )
                
                val_losses['total'] += loss.item()
                num_batches += 1
        
        avg_losses = {
            key: val / max(1, num_batches) 
            for key, val in val_losses.items()
        }
        
        return avg_losses
    
    def train(self, 
              train_dataset: ReflectionTraceDataset,
              val_dataset: Optional[ReflectionTraceDataset] = None,
              num_epochs: Optional[int] = None):
        """
        Run full training loop.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            num_epochs: Number of epochs (default from config)
        """
        num_epochs = num_epochs or self.distill_config['num_epochs']
        batch_size = self.distill_config['batch_size']
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Single process for trace datasets
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
        
        logger.info(f"Starting distillation training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training
            train_losses = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_losses)
            
            # Validation
            if val_loader:
                val_losses = self.validate(val_loader)
                self.val_losses.append(val_losses)
                
                # Save best model
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.save_checkpoint('best_model.pt')
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                log_msg = f"Epoch {epoch+1}/{num_epochs}: "
                log_msg += f"Train Loss={train_losses['total']:.4f}, "
                if val_loader:
                    log_msg += f"Val Loss={val_losses['total']:.4f}"
                logger.info(log_msg)
        
        logger.info("Distillation training complete!")
    
    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.
        
        Args:
            filename: Output filename
        """
        output_dir = Path(self.config['output']['save_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'hgnn_state_dict': self.hgnn.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, output_dir / filename)
        logger.info(f"Checkpoint saved to {output_dir / filename}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.hgnn.load_state_dict(checkpoint['hgnn_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Checkpoint loaded from {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Phase 2: GNN Distillation Training')
    parser.add_argument('--config', type=str, default='experiment_launch_confg.yaml',
                       help='Path to configuration file')
    parser.add_argument('--traces_file', type=str, default='results/bootstrap_results.json',
                       help='Path to bootstrap traces file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for resuming training')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if distillation is enabled
    if not config.get('distillation', {}).get('enabled', True):
        logger.info("Distillation phase disabled in config. Skipping.")
        return
    
    # Load bootstrap traces
    traces_file = Path(args.traces_file)
    if not traces_file.exists():
        logger.error(f"Traces file not found: {traces_file}")
        logger.error("Run bootstrap_phase.py first to generate traces.")
        return
    
    with open(traces_file, 'r') as f:
        bootstrap_results = json.load(f)
    
    traces = bootstrap_results.get('traces', [])
    logger.info(f"Loaded {len(traces)} reflection traces")
    
    if len(traces) < 10:
        logger.warning("Very few traces available. Distillation may not be effective.")
    
    # Initialize components
    embedding_client = EmbeddingClient()
    
    # Reconstruct graph from bootstrap results
    graph = HeterogeneousGraph()
    graph_stats = bootstrap_results.get('graph_stats', {})
    logger.info(f"Graph from bootstrap: {graph_stats}")
    
    # Create dataset
    logger.info("Preparing training dataset...")
    dataset = ReflectionTraceDataset(traces, embedding_client, graph)
    
    # Split into train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    logger.info(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # Initialize trainer
    trainer = DistillationTrainer(config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train
    trainer.train(train_dataset, val_dataset)
    
    # Save final model
    trainer.save_checkpoint('final_model.pt')
    
    # Save training history
    output_dir = Path(config['output']['save_dir'])
    history = {
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'best_val_loss': trainer.best_val_loss
    }
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Phase 2 complete. Model saved to {output_dir}")


if __name__ == '__main__':
    main()