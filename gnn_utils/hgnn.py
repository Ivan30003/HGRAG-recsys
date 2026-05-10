"""
Heterogeneous Graph Neural Network Module
Implements the core HGNN architecture with tier-specific projection heads
and type-aware message passing for distilling agent memory dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TypeSpecificMessagePassing(nn.Module):
    """
    Type-specific message passing layer for heterogeneous graphs.
    Handles different edge types with separate transformation matrices.
    """
    
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 num_edge_types: int = 4,
                 dropout: float = 0.2,
                 use_attention: bool = True):
        """
        Initialize message passing layer.
        
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            num_edge_types: Number of different edge types
            dropout: Dropout rate
            use_attention: Whether to use attention-based aggregation
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_edge_types = num_edge_types
        self.use_attention = use_attention
        
        # Type-specific transformation matrices
        self.edge_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_edge_types)
        ])
        
        # Self-transformation
        self.self_transform = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism for neighbor importance
        if use_attention:
            self.attention = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(out_dim * 2, out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, 1)
                )
                for _ in range(num_edge_types)
            ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(out_dim * (num_edge_types + 1), out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                node_features: torch.Tensor,
                adjacency_lists: List[Dict[int, List[int]]],
                edge_weights: Optional[List[Dict[int, List[float]]]] = None
                ) -> Tuple[torch.Tensor, Dict[int, Dict[int, float]]]:
        """
        Forward pass with type-specific message passing.
        
        Args:
            node_features: (num_nodes, in_dim) tensor
            adjacency_lists: List of adjacency dicts per edge type
            edge_weights: Optional edge weight dicts per edge type
        
        Returns:
            Tuple of:
            - Updated node features (num_nodes, out_dim)
            - Attention weights for interpretability
        """
        num_nodes = node_features.shape[0]
        device = node_features.device
        
        # Self-transformation
        h_self = self.self_transform(node_features)  # (num_nodes, out_dim)
        
        # Type-specific aggregations
        type_messages = []
        all_attention_weights = []
        
        for edge_type_idx in range(self.num_edge_types):
            adj_dict = adjacency_lists[edge_type_idx] if edge_type_idx < len(adjacency_lists) else {}
            weights_dict = edge_weights[edge_type_idx] if edge_weights and edge_type_idx < len(edge_weights) else None
            
            # Aggregate messages from this edge type
            h_type, attn_weights = self._aggregate_type(
                node_features, adj_dict, weights_dict, edge_type_idx, device
            )
            
            type_messages.append(h_type)
            all_attention_weights.append(attn_weights)
        
        # Concatenate all type messages with self
        combined = torch.cat([h_self] + type_messages, dim=-1)  # (num_nodes, out_dim * (num_types + 1))
        
        # Final projection
        output = self.output_projection(combined)
        output = self.dropout(output)
        
        # Add residual connection if dimensions match
        if self.in_dim == self.out_dim:
            output = output + node_features
        
        return output, all_attention_weights
    
    def _aggregate_type(self,
                         node_features: torch.Tensor,
                         adj_dict: Dict[int, List[int]],
                         edge_weights: Optional[Dict[int, List[float]]],
                         edge_type_idx: int,
                         device: torch.device
                         ) -> Tuple[torch.Tensor, Dict[int, Dict[int, float]]]:
        """
        Aggregate messages from neighbors of a specific edge type.
        
        Args:
            node_features: Node features tensor
            adj_dict: Adjacency for this edge type
            edge_weights: Optional edge weights
            edge_type_idx: Index of this edge type
            device: Torch device
        
        Returns:
            Tuple of:
            - Aggregated messages (num_nodes, out_dim)
            - Attention weights dict
        """
        num_nodes = node_features.shape[0]
        messages = torch.zeros(num_nodes, self.out_dim, device=device)
        attention_weights = {}
        
        if not adj_dict:
            return messages, attention_weights
        
        transform = self.edge_transforms[edge_type_idx]
        
        for src_idx, tgt_indices in adj_dict.items():
            if not tgt_indices or src_idx >= num_nodes:
                continue
            
            # Get source and target features
            src_feat = node_features[src_idx]  # (in_dim,)
            
            valid_indices = [idx for idx in tgt_indices if idx < num_nodes]
            if not valid_indices:
                continue
            
            tgt_feats = node_features[valid_indices]  # (num_neighbors, in_dim)
            
            # Transform target features
            tgt_transformed = transform(tgt_feats)  # (num_neighbors, out_dim)
            
            # Compute attention weights
            if self.use_attention:
                src_expanded = src_feat.unsqueeze(0).expand(len(valid_indices), -1)
                attn_input = torch.cat([src_expanded, tgt_feats], dim=-1)
                attn_scores = self.attention[edge_type_idx](attn_input).squeeze(-1)
                
                # Apply edge weights if provided
                if edge_weights and src_idx in edge_weights:
                    e_weights = torch.tensor(
                        edge_weights[src_idx][:len(valid_indices)], 
                        device=device
                    )
                    attn_scores = attn_scores * e_weights
                
                attn_weights = F.softmax(attn_scores, dim=0)
                
                # Store for interpretability
                attention_weights[src_idx] = {
                    valid_indices[i]: attn_weights[i].item() 
                    for i in range(len(valid_indices))
                }
                
                # Weighted aggregation
                aggregated = (tgt_transformed * attn_weights.unsqueeze(-1)).sum(dim=0)
            else:
                # Mean aggregation
                if edge_weights and src_idx in edge_weights:
                    e_weights = torch.tensor(
                        edge_weights[src_idx][:len(valid_indices)], 
                        device=device
                    )
                    e_weights = e_weights / (e_weights.sum() + 1e-8)
                    aggregated = (tgt_transformed * e_weights.unsqueeze(-1)).sum(dim=0)
                else:
                    aggregated = tgt_transformed.mean(dim=0)
            
            messages[src_idx] = aggregated
        
        return messages, attention_weights


class HeterogeneousGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network with tier-specific projection heads.
    
    Used in Phase 2 for distilling LLM-generated memory dynamics
    into efficient graph neural encoders.
    
    Architecture:
    - Multiple type-specific message passing layers
    - Tier-disentangled projection heads (intrinsic, collaborative, interaction)
    - Attention-based interpretability
    - Residual connections and layer normalization
    """
    
    def __init__(self,
                 input_dim: int = 256,
                 hidden_dim: int = 256,
                 output_dim: int = 256,
                 num_layers: int = 2,
                 num_edge_types: int = 4,
                 dropout: float = 0.2,
                 use_attention: bool = True,
                 use_batch_norm: bool = False):
        """
        Initialize HGNN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of message passing layers
            num_edge_types: Number of different edge types
            dropout: Dropout rate
            use_attention: Whether to use attention-based aggregation
            use_batch_norm: Whether to use batch norm (vs layer norm)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_edge_types = num_edge_types
        self.use_attention = use_attention
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Message passing layers
        self.message_layers = nn.ModuleList([
            TypeSpecificMessagePassing(
                in_dim=hidden_dim if i == 0 else hidden_dim,
                out_dim=hidden_dim,
                num_edge_types=num_edge_types,
                dropout=dropout,
                use_attention=use_attention
            )
            for i in range(num_layers)
        ])
        
        # Layer normalization
        if use_batch_norm:
            self.norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim)
                for _ in range(num_layers)
            ])
        else:
            self.norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim)
                for _ in range(num_layers)
            ])
        
        # Tier-specific projection heads
        self.proj_intrinsic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.proj_collaborative = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.proj_interaction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Global pooling for graph-level tasks
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self,
                node_features: torch.Tensor,
                adjacency_lists: List[Dict[int, List[int]]],
                edge_weights: Optional[List[Dict[int, List[float]]]] = None,
                return_attention: bool = False
                ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                          Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], List[Dict]]]:
        """
        Forward pass through HGNN.
        
        Args:
            node_features: (num_nodes, input_dim) tensor
            adjacency_lists: List of adjacency dicts per edge type
            edge_weights: Optional list of edge weight dicts per edge type
            return_attention: Whether to return attention weights
        
        Returns:
            If return_attention=False:
                Tuple of (h_intrinsic, h_collaborative, h_interaction)
            If return_attention=True:
                Tuple of ((h_intrinsic, h_collaborative, h_interaction), attention_weights)
        """
        # Input projection
        h = self.input_projection(node_features)
        
        # Store all attention weights
        all_attention_weights = []
        
        # Message passing layers
        for layer_idx in range(self.num_layers):
            h, attn_weights = self.message_layers[layer_idx](
                h, adjacency_lists, edge_weights
            )
            
            # Apply normalization
            if h.shape[0] > 1 or len(h.shape) == 2:
                h = self.norms[layer_idx](h)
            
            h = F.relu(h)
            h = self.dropout(h)
            
            if return_attention:
                all_attention_weights.append(attn_weights)
        
        # Tier-specific projections
        h_intrinsic = self.proj_intrinsic(h)
        h_collaborative = self.proj_collaborative(h)
        h_interaction = self.proj_interaction(h)
        
        if return_attention:
            return (h_intrinsic, h_collaborative, h_interaction), all_attention_weights
        
        return h_intrinsic, h_collaborative, h_interaction
    
    def predict_tier_embeddings(self,
                                 graph_features: torch.Tensor,
                                 tier: str = 'collaborative'
                                 ) -> torch.Tensor:
        """
        Predict embeddings for a specific tier from graph features.
        
        Args:
            graph_features: Node features tensor
            tier: Which tier to predict ('intrinsic', 'collaborative', 'interaction')
        
        Returns:
            Predicted embeddings tensor
        """
        if tier == 'intrinsic':
            return self.proj_intrinsic(graph_features)
        elif tier == 'collaborative':
            return self.proj_collaborative(graph_features)
        elif tier == 'interaction':
            return self.proj_interaction(graph_features)
        else:
            raise ValueError(f"Unknown tier: {tier}")
    
    def get_attention_weights(self,
                               node_features: torch.Tensor,
                               adj_dict: Dict[int, List[int]],
                               edge_type_idx: int = 0,
                               layer_idx: int = 0
                               ) -> Dict[int, np.ndarray]:
        """
        Extract attention weights for interpretability.
        Used for path importance distillation.
        
        Args:
            node_features: Node features tensor
            adj_dict: Adjacency for specific edge type
            edge_type_idx: Which edge type
            layer_idx: Which layer
        
        Returns:
            Dictionary mapping node_idx -> attention weights array
        """
        if not self.use_attention:
            return {}
        
        # Get attention from specific layer and edge type
        message_layer = self.message_layers[layer_idx]
        
        if edge_type_idx >= len(message_layer.attention):
            return {}
        
        attention_module = message_layer.attention[edge_type_idx]
        transform = message_layer.edge_transforms[edge_type_idx]
        
        attention_weights = {}
        
        with torch.no_grad():
            for src_idx, tgt_indices in adj_dict.items():
                if not tgt_indices or src_idx >= node_features.shape[0]:
                    continue
                
                valid_indices = [idx for idx in tgt_indices if idx < node_features.shape[0]]
                if not valid_indices:
                    continue
                
                src_feat = node_features[src_idx]
                tgt_feats = node_features[valid_indices]
                
                src_expanded = src_feat.unsqueeze(0).expand(len(valid_indices), -1)
                attn_input = torch.cat([src_expanded, tgt_feats], dim=-1)
                attn_scores = attention_module(attn_input).squeeze(-1)
                
                weights = F.softmax(attn_scores, dim=0)
                attention_weights[src_idx] = weights.cpu().numpy()
        
        return attention_weights
    
    def get_node_embeddings(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Get final node embeddings before tier projection.
        
        Args:
            node_features: Input node features
        
        Returns:
            Node embeddings tensor
        """
        h = self.input_projection(node_features)
        
        for layer_idx in range(self.num_layers):
            # Use empty adjacency for self-loop only pass
            empty_adj = [{} for _ in range(self.num_edge_types)]
            h, _ = self.message_layers[layer_idx](h, empty_adj)
            h = self.norms[layer_idx](h)
            h = F.relu(h)
        
        return h
    
    def global_pooling(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Global pooling over all nodes.
        
        Args:
            node_features: Node features tensor
        
        Returns:
            Global graph embedding
        """
        return self.global_pool(node_features.mean(dim=0))


class LightDecoder(nn.Module):
    """
    Lightweight decoder for regenerating text from tier embeddings.
    
    Used for on-demand text reconstruction in Phase 3 when
    the GNN path needs to produce human-readable output.
    
    Architecture:
    - Tier fusion layer (combines 3 tier embeddings)
    - LSTM decoder with attention
    - Output projection to vocabulary
    - Optional teacher forcing during training
    """
    
    def __init__(self,
                 embedding_dim: int = 256,
                 hidden_dim: int = 512,
                 vocab_size: int = 50000,
                 num_lstm_layers: int = 2,
                 max_length: int = 100,
                 dropout: float = 0.2,
                 tie_weights: bool = False):
        """
        Initialize LightDecoder.
        
        Args:
            embedding_dim: Dimension of input tier embeddings
            hidden_dim: LSTM hidden dimension
            vocab_size: Size of output vocabulary
            num_lstm_layers: Number of LSTM layers
            max_length: Maximum generation length
            dropout: Dropout rate
            tie_weights: Whether to tie embedding and output weights
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Tier fusion layer
        self.tier_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Tier attention for dynamic weighting
        self.tier_attention = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        # Token embedding (for autoregressive generation)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            bidirectional=False
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Layer normalization
        self.output_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Optional weight tying
        if tie_weights:
            self.output_projection.weight = self.token_embedding.weight
        
        # Start and end tokens
        self.start_token_id = 0
        self.end_token_id = 1
        self.pad_token_id = 2
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize decoder weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self,
                h_intrinsic: torch.Tensor,
                h_collaborative: torch.Tensor,
                h_interaction: torch.Tensor,
                target_tokens: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.5
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for training or inference.
        
        Args:
            h_intrinsic: (batch_size, embedding_dim)
            h_collaborative: (batch_size, embedding_dim)
            h_interaction: (batch_size, embedding_dim)
            target_tokens: Optional (batch_size, seq_len) for teacher forcing
            teacher_forcing_ratio: Probability of teacher forcing during training
        
        Returns:
            If target_tokens provided:
                Logits tensor (batch_size, seq_len, vocab_size)
            If target_tokens is None:
                Tuple of (generated_token_ids, generation_logits)
        """
        batch_size = h_intrinsic.shape[0]
        device = h_intrinsic.device
        
        # Fuse tier embeddings
        tier_concat = torch.cat([h_intrinsic, h_collaborative, h_interaction], dim=-1)
        fused = self.tier_fusion(tier_concat)  # (batch_size, hidden_dim)
        
        # Compute tier attention weights
        tier_weights = self.tier_attention(fused)  # (batch_size, 3)
        
        # Initialize decoder hidden state
        h0 = fused.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        hidden_state = (h0, c0)
        
        if target_tokens is not None and self.training:
            # Training with teacher forcing
            return self._train_forward(fused, target_tokens, hidden_state, 
                                       teacher_forcing_ratio, device)
        else:
            # Inference: autoregressive generation
            return self._generate(fused, hidden_state, device)
    
    def _train_forward(self,
                        fused: torch.Tensor,
                        target_tokens: torch.Tensor,
                        hidden_state: Tuple[torch.Tensor, torch.Tensor],
                        teacher_forcing_ratio: float,
                        device: torch.device
                        ) -> torch.Tensor:
        """
        Training forward pass with teacher forcing.
        
        Args:
            fused: Fused tier embedding
            target_tokens: Target token sequence
            hidden_state: Initial LSTM state
            teacher_forcing_ratio: Teacher forcing probability
            device: Torch device
        
        Returns:
            Logits tensor
        """
        batch_size, seq_len = target_tokens.shape
        
        # Embed target tokens
        token_embeddings = self.token_embedding(target_tokens)  # (batch_size, seq_len, hidden_dim)
        
        # Prepare decoder input
        decoder_input = torch.cat([
            fused.unsqueeze(1),  # Start with fused embedding
            token_embeddings[:, :-1, :]  # Shift right
        ], dim=1)  # (batch_size, seq_len, hidden_dim)
        
        # LSTM forward
        lstm_out, _ = self.lstm(decoder_input, hidden_state)
        
        # Apply teacher forcing mask
        if teacher_forcing_ratio < 1.0:
            # Randomly replace some positions with previous predictions
            teacher_mask = torch.rand(batch_size, seq_len, device=device) < teacher_forcing_ratio
            # (Simplified - full implementation would use scheduled sampling)
        
        # Project to vocabulary
        lstm_out = self.output_norm(lstm_out)
        logits = self.output_projection(lstm_out)
        logits = self.dropout(logits)
        
        return logits
    
    def _generate(self,
                   fused: torch.Tensor,
                   hidden_state: Tuple[torch.Tensor, torch.Tensor],
                   device: torch.device,
                   temperature: float = 0.7,
                   top_k: int = 50,
                   top_p: float = 0.9
                   ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Autoregressive text generation.
        
        Args:
            fused: Fused tier embedding
            hidden_state: Initial LSTM state
            device: Torch device
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
        
        Returns:
            Tuple of (generated_token_ids, generation_logits)
        """
        batch_size = fused.shape[0]
        
        # Start with fused embedding
        current_input = fused.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        generated_tokens = []
        generation_logits = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        current_hidden = hidden_state
        
        for step in range(self.max_length):
            # LSTM step
            lstm_out, current_hidden = self.lstm(current_input, current_hidden)
            
            # Project to vocabulary
            lstm_out = self.output_norm(lstm_out)
            logits = self.output_projection(lstm_out.squeeze(1))  # (batch_size, vocab_size)
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                min_top_k = top_k_values[:, -1].unsqueeze(-1)
                logits[logits < min_top_k] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # Check for end token
            finished = finished | (next_token == self.end_token_id)
            
            generated_tokens.append(next_token)
            generation_logits.append(logits)
            
            if finished.all():
                break
            
            # Prepare next input
            current_input = self.token_embedding(next_token).unsqueeze(1)
        
        # Stack generated tokens
        generated = torch.stack(generated_tokens, dim=1)  # (batch_size, seq_len)
        
        return generated, generation_logits
    
    def generate_text(self,
                       h_intrinsic: torch.Tensor,
                       h_collaborative: torch.Tensor,
                       h_interaction: torch.Tensor,
                       tokenizer=None,
                       temperature: float = 0.7
                       ) -> List[str]:
        """
        Generate human-readable text from tier embeddings.
        
        Args:
            h_intrinsic: Intrinsic tier embedding
            h_collaborative: Collaborative tier embedding
            h_interaction: Interaction tier embedding
            tokenizer: Optional tokenizer for decoding
            temperature: Sampling temperature
        
        Returns:
            List of generated text strings
        """
        self.eval()
        
        with torch.no_grad():
            token_ids, _ = self.forward(
                h_intrinsic, h_collaborative, h_interaction,
                target_tokens=None
            )
        
        if tokenizer:
            texts = [
                tokenizer.decode(ids, skip_special_tokens=True)
                for ids in token_ids
            ]
        else:
            texts = [f"Generated text for sample {i}" for i in range(len(token_ids))]
        
        return texts
    
    def compute_reconstruction_loss(self,
                                      logits: torch.Tensor,
                                      target_tokens: torch.Tensor,
                                      ignore_index: int = -100
                                      ) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            logits: (batch_size, seq_len, vocab_size)
            target_tokens: (batch_size, seq_len)
            ignore_index: Index to ignore in loss
        
        Returns:
            Scalar loss value
        """
        # Reshape for cross-entropy
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = target_tokens.reshape(-1)
        
        # Cross-entropy loss
        loss = F.cross_entropy(
            logits_flat, targets_flat,
            ignore_index=ignore_index,
            reduction='mean'
        )
        
        return loss


class TierSpecificLoss(nn.Module):
    """
    Tier-specific loss module for distillation training.
    Combines MSE regression, contrastive, and path importance losses.
    """
    
    def __init__(self,
                 tier_weights: Dict[str, float] = None,
                 contrastive_temperature: float = 0.1,
                 path_importance_weight: float = 0.3,
                 reconstruction_weight: float = 0.05):
        """
        Initialize tier-specific loss.
        
        Args:
            tier_weights: Weights for each tier
            contrastive_temperature: Temperature for contrastive loss
            path_importance_weight: Weight for path importance loss
            reconstruction_weight: Weight for reconstruction loss
        """
        super().__init__()
        
        self.tier_weights = tier_weights or {
            'intrinsic': 0.3,
            'collaborative': 1.0,
            'interaction': 0.2
        }
        
        self.contrastive_temperature = contrastive_temperature
        self.path_importance_weight = path_importance_weight
        self.reconstruction_weight = reconstruction_weight
    
    def forward(self,
                predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                path_importance_pred: Optional[torch.Tensor] = None,
                path_importance_target: Optional[torch.Tensor] = None,
                recon_logits: Optional[torch.Tensor] = None,
                recon_targets: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            predictions: (h_int_pred, h_col_pred, h_intr_pred)
            targets: (h_int_target, h_col_target, h_intr_target)
            path_importance_pred: Predicted path importance
            path_importance_target: Target path importance
            recon_logits: Reconstruction logits
            recon_targets: Reconstruction targets
        
        Returns:
            Dictionary of loss components
        """
        h_int_pred, h_col_pred, h_intr_pred = predictions
        h_int_target, h_col_target, h_intr_target = targets
        
        losses = {}
        
        # 1. Tier-specific MSE regression loss
        loss_int = F.mse_loss(h_int_pred, h_int_target)
        loss_col = F.mse_loss(h_col_pred, h_col_target)
        loss_intr = F.mse_loss(h_intr_pred, h_intr_target)
        
        losses['loss_intrinsic'] = loss_int
        losses['loss_collaborative'] = loss_col
        losses['loss_interaction'] = loss_intr
        
        # Weighted tier regression
        losses['loss_tier'] = (
            self.tier_weights['intrinsic'] * loss_int +
            self.tier_weights['collaborative'] * loss_col +
            self.tier_weights['interaction'] * loss_intr
        )
        
        # 2. Contrastive tier separation loss
        losses['loss_contrastive'] = self._contrastive_loss(
            h_int_pred, h_col_pred, h_intr_pred
        )
        
        # 3. Path importance distillation loss
        if path_importance_pred is not None and path_importance_target is not None:
            losses['loss_path'] = self.path_importance_weight * F.kl_div(
                F.log_softmax(path_importance_pred, dim=-1),
                F.softmax(path_importance_target, dim=-1),
                reduction='batchmean'
            )
        
        # 4. Reconstruction loss
        if recon_logits is not None and recon_targets is not None:
            losses['loss_recon'] = self.reconstruction_weight * F.cross_entropy(
                recon_logits.reshape(-1, recon_logits.size(-1)),
                recon_targets.reshape(-1),
                ignore_index=-100
            )
        
        # Total loss
        losses['loss_total'] = losses['loss_tier']
        if 'loss_path' in losses:
            losses['loss_total'] = losses['loss_total'] + losses['loss_path']
        losses['loss_total'] = losses['loss_total'] + losses['loss_contrastive']
        if 'loss_recon' in losses:
            losses['loss_total'] = losses['loss_total'] + losses['loss_recon']
        
        return losses
    
    def _contrastive_loss(self,
                           h_int: torch.Tensor,
                           h_col: torch.Tensor,
                           h_intr: torch.Tensor
                           ) -> torch.Tensor:
        """
        Compute contrastive loss to separate tier embeddings.
        
        Args:
            h_int: Intrinsic embeddings
            h_col: Collaborative embeddings
            h_intr: Interaction embeddings
        
        Returns:
            Contrastive loss value
        """
        batch_size = h_int.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=h_int.device)
        
        # Normalize embeddings
        h_int = F.normalize(h_int, dim=-1)
        h_col = F.normalize(h_col, dim=-1)
        h_intr = F.normalize(h_intr, dim=-1)
        
        # Stack all embeddings
        all_embeddings = torch.cat([h_int, h_col, h_intr], dim=0)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(all_embeddings, all_embeddings.t()) / self.contrastive_temperature
        
        # Labels: embeddings from same sample should be similar
        labels = torch.arange(batch_size, device=h_int.device).repeat(3)
        
        # Mask self-similarity
        mask = torch.eye(3 * batch_size, dtype=torch.bool, device=h_int.device)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
        
        # Contrastive loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss