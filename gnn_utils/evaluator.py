"""
GNN Evaluator Module
Handles evaluation of trained GNN models on recommendation and embedding quality metrics.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import ndcg_score
from collections import defaultdict
import logging

from .hgnn import HeterogeneousGNN, LightDecoder

logger = logging.getLogger(__name__)


class GNNEvaluator:
    """
    Evaluator for trained GNN models.
    
    Evaluates:
    - Embedding quality (alignment with LLM targets)
    - Recommendation performance (NDCG, HR, MRR)
    - Cold-start performance
    - Memory consistency
    """
    
    def __init__(self,
                 hgnn: HeterogeneousGNN,
                 decoder: Optional[LightDecoder] = None,
                 device: str = 'cuda'):
        """
        Initialize evaluator.
        
        Args:
            hgnn: Trained HGNN model
            decoder: Optional decoder for text generation
            device: Computing device
        """
        self.hgnn = hgnn
        self.decoder = decoder
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.hgnn.to(self.device)
        self.hgnn.eval()
        
        if self.decoder:
            self.decoder.to(self.device)
            self.decoder.eval()
    
    def evaluate_embedding_quality(self,
                                    node_features: torch.Tensor,
                                    adjacency_lists: List[Dict],
                                    targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                                    ) -> Dict[str, float]:
        """
        Evaluate quality of predicted embeddings vs LLM targets.
        
        Args:
            node_features: Input node features
            adjacency_lists: Graph adjacency lists
            targets: Target embeddings from LLM
        
        Returns:
            Dictionary of quality metrics
        """
        with torch.no_grad():
            predictions = self.hgnn(node_features.to(self.device), adjacency_lists)
        
        h_int_pred, h_col_pred, h_intr_pred = predictions
        h_int_target, h_col_target, h_intr_target = targets
        
        metrics = {}
        
        # MSE per tier
        metrics['mse_intrinsic'] = F.mse_loss(
            h_int_pred.cpu(), h_int_target.cpu()
        ).item()
        metrics['mse_collaborative'] = F.mse_loss(
            h_col_pred.cpu(), h_col_target.cpu()
        ).item()
        metrics['mse_interaction'] = F.mse_loss(
            h_intr_pred.cpu(), h_intr_target.cpu()
        ).item()
        
        # Cosine similarity per tier
        metrics['cos_sim_intrinsic'] = float(F.cosine_similarity(
            h_int_pred.cpu().mean(0), h_int_target.cpu().mean(0), dim=0
        ))
        metrics['cos_sim_collaborative'] = float(F.cosine_similarity(
            h_col_pred.cpu().mean(0), h_col_target.cpu().mean(0), dim=0
        ))
        metrics['cos_sim_interaction'] = float(F.cosine_similarity(
            h_intr_pred.cpu().mean(0), h_intr_target.cpu().mean(0), dim=0
        ))
        
        # Overall quality
        metrics['overall_quality'] = np.mean([
            1.0 - min(1.0, metrics['mse_collaborative']),
            metrics['cos_sim_collaborative']
        ])
        
        return metrics
    
    def evaluate_recommendation(self,
                                 user_embeddings: torch.Tensor,
                                 item_embeddings: torch.Tensor,
                                 test_pairs: List[Tuple[int, int]],
                                 all_item_ids: List[int],
                                 k_values: List[int] = [1, 5, 10],
                                 num_negatives: int = 99
                                 ) -> Dict[str, float]:
        """
        Evaluate recommendation performance.
        
        Args:
            user_embeddings: User node embeddings
            item_embeddings: Item node embeddings
            test_pairs: List of (user_idx, positive_item_idx) pairs
            all_item_ids: List of all item indices
            k_values: K values for NDCG and HR
            num_negatives: Number of negative samples
        
        Returns:
            Dictionary of recommendation metrics
        """
        metrics = defaultdict(list)
        
        for user_idx, pos_item_idx in test_pairs:
            # Get user embedding
            user_emb = user_embeddings[user_idx:user_idx+1]  # (1, dim)
            
            # Sample negative items
            neg_indices = self._sample_negatives(
                pos_item_idx, all_item_ids, num_negatives
            )
            
            # Combine positive and negatives
            candidate_indices = [pos_item_idx] + neg_indices
            candidate_embs = item_embeddings[candidate_indices]  # (1+neg, dim)
            
            # Compute scores (dot product)
            scores = torch.mm(user_emb, candidate_embs.t()).squeeze(0)  # (1+neg,)
            
            # Rank by score (descending)
            _, ranking = torch.sort(scores, descending=True)
            
            # Find position of positive item
            pos_rank = (ranking == 0).nonzero(as_tuple=True)[0].item()  # 0-indexed
            
            # Compute metrics
            for k in k_values:
                # Hit Rate
                hit = 1.0 if pos_rank < k else 0.0
                metrics[f'HR@{k}'].append(hit)
                
                # NDCG
                relevance = np.zeros(len(candidate_indices))
                relevance[0] = 1  # Positive is at index 0
                
                # Get top-k scores
                top_k_scores = scores.cpu().numpy()[:k]
                top_k_relevance = relevance[:k]
                
                if np.sum(top_k_relevance) > 0:
                    ndcg = ndcg_score([top_k_relevance], [list(range(k, 0, -1))])
                    metrics[f'NDCG@{k}'].append(ndcg)
                else:
                    metrics[f'NDCG@{k}'].append(0.0)
            
            # MRR
            metrics['MRR'].append(1.0 / (pos_rank + 1))
        
        # Average metrics
        results = {}
        for key, values in metrics.items():
            results[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values)
            }
        
        return results
    
    def evaluate_cold_start(self,
                             cold_start_items: List[int],
                             warm_item_embeddings: torch.Tensor,
                             item_intrinsic_embeddings: torch.Tensor,
                             user_embeddings: torch.Tensor,
                             test_pairs: List[Tuple[int, int]],
                             k: int = 10
                             ) -> Dict[str, float]:
        """
        Evaluate cold-start recommendation performance.
        
        Args:
            cold_start_items: Indices of cold-start items
            warm_item_embeddings: Embeddings of warm items
            item_intrinsic_embeddings: Intrinsic embeddings of all items
            user_embeddings: User embeddings
            test_pairs: Test pairs for cold-start items
            k: K for evaluation
        
        Returns:
            Dictionary of cold-start metrics
        """
        # Predict cold-start embeddings from intrinsic features
        cold_embs = self.hgnn.predict_tier_embeddings(
            item_intrinsic_embeddings[cold_start_items].to(self.device),
            tier='collaborative'
        )
        
        # Combine with warm embeddings
        all_item_embs = warm_item_embeddings.clone()
        all_item_embs[cold_start_items] = cold_embs.cpu()
        
        # Evaluate
        cold_test_pairs = [
            (u, i) for u, i in test_pairs if i in cold_start_items
        ]
        
        if not cold_test_pairs:
            return {'HR@10': {'mean': 0.0}, 'NDCG@10': {'mean': 0.0}}
        
        return self.evaluate_recommendation(
            user_embeddings, all_item_embs,
            cold_test_pairs, list(range(len(all_item_embs))),
            k_values=[k]
        )
    
    def evaluate_memory_consistency(self,
                                     initial_embeddings: torch.Tensor,
                                     updated_embeddings: torch.Tensor,
                                     tier: str = 'intrinsic'
                                     ) -> Dict[str, float]:
        """
        Evaluate memory consistency over optimization steps.
        
        Args:
            initial_embeddings: Initial tier embeddings
            updated_embeddings: Updated tier embeddings
            tier: Which tier to evaluate
        
        Returns:
            Dictionary of consistency metrics
        """
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(
            initial_embeddings.cpu(), updated_embeddings.cpu(), dim=-1
        )
        
        # Compute L2 distance
        l2_dist = torch.norm(
            initial_embeddings.cpu() - updated_embeddings.cpu(), dim=-1
        )
        
        metrics = {
            f'{tier}_mean_cos_sim': float(cos_sim.mean()),
            f'{tier}_min_cos_sim': float(cos_sim.min()),
            f'{tier}_mean_l2_dist': float(l2_dist.mean()),
            f'{tier}_max_l2_dist': float(l2_dist.max()),
            f'{tier}_drift_ratio': float((cos_sim < 0.5).float().mean())
        }
        
        return metrics
    
    def evaluate_path_importance_accuracy(self,
                                           predicted_importance: torch.Tensor,
                                           true_importance: torch.Tensor
                                           ) -> Dict[str, float]:
        """
        Evaluate accuracy of path importance predictions.
        
        Args:
            predicted_importance: Predicted importance scores
            true_importance: Ground truth importance from LLM
        
        Returns:
            Dictionary of accuracy metrics
        """
        # Correlation
        pred_np = predicted_importance.cpu().numpy()
        true_np = true_importance.cpu().numpy()
        
        correlation = np.corrcoef(pred_np.flatten(), true_np.flatten())[0, 1]
        
        # Top-k overlap
        k = min(3, len(pred_np))
        pred_topk = set(np.argsort(pred_np)[-k:])
        true_topk = set(np.argsort(true_np)[-k:])
        topk_overlap = len(pred_topk & true_topk) / k
        
        # KL divergence
        kl_div = F.kl_div(
            F.log_softmax(predicted_importance, dim=-1),
            F.softmax(true_importance, dim=-1),
            reduction='batchmean'
        ).item()
        
        return {
            'correlation': float(correlation),
            'topk_overlap': topk_overlap,
            'kl_divergence': kl_div
        }
    
    def _sample_negatives(self,
                           positive_idx: int,
                           all_indices: List[int],
                           num_negatives: int
                           ) -> List[int]:
        """Sample negative items excluding the positive."""
        available = [i for i in all_indices if i != positive_idx]
        
        if len(available) < num_negatives:
            return available
        
        return list(np.random.choice(available, num_negatives, replace=False))
    
    def comprehensive_evaluation(self,
                                  node_features: torch.Tensor,
                                  adjacency_lists: List[Dict],
                                  targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                  test_pairs: List[Tuple[int, int]],
                                  all_item_ids: List[int],
                                  cold_start_items: Optional[List[int]] = None,
                                  initial_embeddings: Optional[torch.Tensor] = None
                                  ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation across all metrics.
        
        Args:
            node_features: Input node features
            adjacency_lists: Graph adjacency lists
            targets: Target embeddings
            test_pairs: Test user-item pairs
            all_item_ids: All item indices
            cold_start_items: Cold-start item indices
            initial_embeddings: Initial embeddings for consistency check
        
        Returns:
            Dictionary of all evaluation metrics
        """
        logger.info("Running comprehensive evaluation...")
        
        results = {}
        
        # 1. Embedding quality
        logger.info("Evaluating embedding quality...")
        results['embedding_quality'] = self.evaluate_embedding_quality(
            node_features, adjacency_lists, targets
        )
        
        # 2. Recommendation performance
        logger.info("Evaluating recommendation performance...")
        with torch.no_grad():
            predictions = self.hgnn(
                node_features.to(self.device), adjacency_lists
            )
        
        h_int, h_col, h_intr = predictions
        
        # Assume first half are users, second half are items
        # (In practice, use proper node type mapping)
        num_users = len(node_features) // 2
        user_embs = h_col[:num_users].cpu()
        item_embs = h_col[num_users:].cpu()
        
        results['recommendation'] = self.evaluate_recommendation(
            user_embs, item_embs, test_pairs, all_item_ids
        )
        
        # 3. Cold-start performance
        if cold_start_items:
            logger.info("Evaluating cold-start performance...")
            results['cold_start'] = self.evaluate_cold_start(
                cold_start_items, item_embs,
                h_int[num_users:].cpu(), user_embs,
                test_pairs
            )
        
        # 4. Memory consistency
        if initial_embeddings is not None:
            logger.info("Evaluating memory consistency...")
            results['consistency_intrinsic'] = self.evaluate_memory_consistency(
                initial_embeddings, h_int.cpu(), 'intrinsic'
            )
            results['consistency_collaborative'] = self.evaluate_memory_consistency(
                initial_embeddings, h_col.cpu(), 'collaborative'
            )
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*50)
        
        if 'recommendation' in results:
            rec = results['recommendation']
            for k in [1, 5, 10]:
                if f'NDCG@{k}' in rec:
                    logger.info(f"NDCG@{k}: {rec[f'NDCG@{k}']['mean']:.4f} "
                              f"± {rec[f'NDCG@{k}']['std']:.4f}")
                if f'HR@{k}' in rec:
                    logger.info(f"HR@{k}: {rec[f'HR@{k}']['mean']:.4f}")
        
        if 'embedding_quality' in results:
            eq = results['embedding_quality']
            logger.info(f"Embedding Quality: {eq.get('overall_quality', 0):.4f}")
        
        return results