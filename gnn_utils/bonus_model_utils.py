"""
Model Utilities Module
Helper functions for GNN model management and inference.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {'trainable': trainable, 'total': total, 'non_trainable': total - trainable}


def freeze_model(model: nn.Module):
    """Freeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: nn.Module):
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in megabytes."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def prepare_adjacency_from_edges(edge_list: List[Tuple[int, int, int, float]],
                                  num_nodes: int,
                                  num_edge_types: int = 4
                                  ) -> Tuple[List[Dict[int, List[int]]], List[Dict[int, List[float]]]]:
    """
    Prepare adjacency lists and edge weights from edge list.
    
    Args:
        edge_list: List of (src, tgt, edge_type, weight) tuples
        num_nodes: Total number of nodes
        num_edge_types: Number of edge types
    
    Returns:
        Tuple of (adjacency_lists, edge_weights)
    """
    adjacency_lists = [
        {i: [] for i in range(num_nodes)}
        for _ in range(num_edge_types)
    ]
    edge_weights = [
        {i: [] for i in range(num_nodes)}
        for _ in range(num_edge_types)
    ]
    
    for src, tgt, edge_type, weight in edge_list:
        if edge_type < num_edge_types:
            adjacency_lists[edge_type][src].append(tgt)
            edge_weights[edge_type][src].append(weight)
    
    return adjacency_lists, edge_weights


def compute_embedding_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix."""
    embeddings_norm = F.normalize(embeddings, dim=-1)
    return torch.mm(embeddings_norm, embeddings_norm.t())


def compute_ndcg(scores: torch.Tensor, relevance: torch.Tensor, k: int) -> float:
    """
    Compute NDCG@k.
    
    Args:
        scores: Predicted scores
        relevance: Binary relevance labels
        k: Cutoff
    
    Returns:
        NDCG@k value
    """
    _, indices = torch.sort(scores, descending=True)
    top_k_relevance = relevance[indices[:k]]
    
    # DCG
    positions = torch.arange(1, k + 1, dtype=torch.float32)
    dcg = (top_k_relevance / torch.log2(positions + 1)).sum()
    
    # IDCG
    ideal_relevance, _ = torch.sort(relevance, descending=True)
    ideal_relevance = ideal_relevance[:k]
    idcg = (ideal_relevance / torch.log2(positions + 1)).sum()
    
    if idcg == 0:
        return 0.0
    
    return (dcg / idcg).item()


def compute_hit_rate(scores: torch.Tensor, positive_idx: int, k: int) -> float:
    """Compute Hit Rate @ k."""
    _, top_k_indices = torch.topk(scores, k)
    return 1.0 if positive_idx in top_k_indices else 0.0


def compute_mrr(scores: torch.Tensor, positive_idx: int) -> float:
    """Compute Mean Reciprocal Rank."""
    _, sorted_indices = torch.sort(scores, descending=True)
    rank = (sorted_indices == positive_idx).nonzero(as_tuple=True)[0].item() + 1
    return 1.0 / rank