"""
PPR Sampler Module for H-GRAGrecsys

This module implements Personalized PageRank (PPR) based sampling for
negative sampling, hard negative mining, and importance sampling in the
heterogeneous graph.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from collections import defaultdict, Counter
import heapq
from tqdm import tqdm
import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.graph.dynamic_graph import DynamicGraph
from models.graph.relation_types import RelationType, EdgeWeightFunctions
from utils.logger import Logger
from utils.config_loader import ConfigLoader


class PPRSampler:
    """
    Personalized PageRank based sampler for graph operations.
    
    This class handles:
    - Computing PPR scores for nodes
    - Sampling negative examples using PPR
    - Easy and hard negative sampling
    - Adaptive sampling distributions
    - Batch sampling for training
    """
    
    def __init__(self, graph: DynamicGraph, config: Dict[str, Any]):
        """
        Initialize the PPR sampler.
        
        Args:
            graph: DynamicGraph instance
            config: Configuration dictionary
        """
        self.graph = graph
        self.config = config
        self.logger = Logger.get_instance(log_dir='logs', name='ppr_sampler')
        
        # Extract configuration
        graph_rag_config = config.get('model', {}).get('graph_rag', {})
        self.default_restart_prob = graph_rag_config.get('ppr_restart_prob', 0.15)
        self.default_max_iterations = graph_rag_config.get('ppr_max_iterations', 100)
        self.default_tolerance = graph_rag_config.get('ppr_tolerance', 1e-6)
        self.sampling_batch_size = graph_rag_config.get('sampling_batch_size', 32)
        self.hard_negative_ratio = graph_rag_config.get('hard_negative_ratio', 0.5)
        self.easy_negative_ratio = graph_rag_config.get('easy_negative_ratio', 0.3)
        self.min_ppr_threshold = graph_rag_config.get('min_ppr_threshold', 0.001)
        
        # PPR cache
        self.ppr_cache: Dict[Tuple[str, float], Dict[str, float]] = {}
        self.cache_size = graph_rag_config.get('ppr_cache_size', 100)
        
        # Sampling statistics
        self.sampling_stats = {
            'total_samples': 0,
            'easy_negatives': 0,
            'hard_negatives': 0,
            'ppr_computations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_sampling_time': 0.0,
            'sampling_times': []
        }
        
        # Precomputed node degrees for efficiency
        self.node_degrees = self._compute_node_degrees()
        
        self.logger.log_info(f"Initialized PPRSampler with restart_prob={self.default_restart_prob}")
    
    def _compute_node_degrees(self) -> Dict[str, int]:
        """
        Compute degrees for all nodes in the graph.
        
        Returns:
            Dict[str, int]: Node degrees
        """
        degrees = {}
        for node_id in self.graph.nodes:
            degrees[node_id] = self.graph.get_connection_count(node_id)
        return degrees
    
    def compute_ppr(self, start_node: str, 
                   restart_prob: Optional[float] = None,
                   max_iterations: Optional[int] = None,
                   tolerance: Optional[float] = None,
                   use_cache: bool = True) -> Dict[str, float]:
        """
        Compute PPR scores from a start node.
        
        Args:
            start_node: Starting node ID
            restart_prob: Probability of restarting
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            use_cache: Whether to use cached results
            
        Returns:
            Dict[str, float]: PPR scores for all nodes
        """
        restart_prob = restart_prob or self.default_restart_prob
        max_iterations = max_iterations or self.default_max_iterations
        tolerance = tolerance or self.default_tolerance
        
        # Check cache
        cache_key = (start_node, restart_prob)
        if use_cache and cache_key in self.ppr_cache:
            self.logger.log_info(f"PPR cache hit for node {start_node}")
            self.sampling_stats['cache_hits'] += 1
            return self.ppr_cache[cache_key].copy()
        
        self.sampling_stats['cache_misses'] += 1
        self.sampling_stats['ppr_computations'] += 1
        
        # Delegate to graph's PPR computation
        ppr_scores = self.graph.compute_ppr_scores(
            start_node, 
            restart_prob, 
            max_iterations, 
            tolerance
        )
        
        # Cache results
        if use_cache:
            if len(self.ppr_cache) >= self.cache_size:
                # Remove oldest entry
                self.ppr_cache.pop(next(iter(self.ppr_cache)))
            self.ppr_cache[cache_key] = ppr_scores.copy()
        
        return ppr_scores
    
    def sample_negative(self, start_node: str, 
                       positive_items: List[str],
                       num_negatives: int = 10,
                       sampling_strategy: str = 'hybrid',
                       restart_prob: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Sample negative examples for a given node.
        
        Args:
            start_node: Starting node ID
            positive_items: List of positive items (to avoid)
            num_negatives: Number of negatives to sample
            sampling_strategy: Strategy to use ('easy', 'hard', 'hybrid', 'uniform')
            restart_prob: PPR restart probability
            
        Returns:
            List[Tuple[str, float]]: List of (negative_item, score) pairs
        """
        self.logger.log_info(f"Sampling {num_negatives} negatives for {start_node} using {sampling_strategy}")
        
        if sampling_strategy == 'easy':
            return self.easy_negative_sampling(start_node, positive_items, num_negatives)
        elif sampling_strategy == 'hard':
            return self.hard_negative_sampling(start_node, positive_items, num_negatives, restart_prob)
        elif sampling_strategy == 'hybrid':
            return self.hybrid_negative_sampling(start_node, positive_items, num_negatives, restart_prob)
        elif sampling_strategy == 'uniform':
            return self.uniform_negative_sampling(start_node, positive_items, num_negatives)
        else:
            self.logger.log_warning(f"Unknown sampling strategy {sampling_strategy}, using hybrid")
            return self.hybrid_negative_sampling(start_node, positive_items, num_negatives, restart_prob)
    
    def easy_negative_sampling(self, start_node: str, 
                              positive_items: List[str],
                              num_negatives: int = 10) -> List[Tuple[str, float]]:
        """
        Sample easy negatives (random nodes not connected to start node).
        
        Args:
            start_node: Starting node ID
            positive_items: List of positive items
            num_negatives: Number of negatives to sample
            
        Returns:
            List[Tuple[str, float]]: List of (negative_item, score) pairs
        """
        # Get all items in graph
        all_items = [nid for nid, node in self.graph.nodes.items() 
                    if node.node_type == 'item']
        
        # Get items connected to start node
        connected_items = set()
        for item_id in positive_items:
            connected_items.add(item_id)
        
        # Also get items connected through interactions
        for rel_type in RelationType.get_all_types():
            neighbors = self.graph.get_neighbors(start_node, rel_type)
            for neighbor, weight in neighbors:
                if self.graph.node_id_to_type.get(neighbor) == 'item':
                    connected_items.add(neighbor)
        
        # Candidate negatives
        candidates = [item for item in all_items if item not in connected_items]
        
        # If not enough candidates, add some random items
        if len(candidates) < num_negatives:
            extra_needed = num_negatives - len(candidates)
            extra_items = [item for item in all_items if item not in candidates]
            candidates.extend(random.sample(extra_items, min(extra_needed, len(extra_items))))
        
        # Sample negatives
        sampled = random.sample(candidates, min(num_negatives, len(candidates)))
        
        # Assign scores (uniform weight)
        result = [(item, 1.0 / len(sampled)) for item in sampled]
        
        self.sampling_stats['easy_negatives'] += len(sampled)
        self.sampling_stats['total_samples'] += len(sampled)
        
        return result
    
    def hard_negative_sampling(self, start_node: str, 
                              positive_items: List[str],
                              num_negatives: int = 10,
                              restart_prob: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Sample hard negatives (nodes with high PPR but not positive).
        
        Args:
            start_node: Starting node ID
            positive_items: List of positive items
            num_negatives: Number of negatives to sample
            restart_prob: PPR restart probability
            
        Returns:
            List[Tuple[str, float]]: List of (negative_item, score) pairs
        """
        restart_prob = restart_prob or self.default_restart_prob
        
        # Get PPR scores
        ppr_scores = self.compute_ppr(start_node, restart_prob)
        
        # Filter out positive items
        positive_set = set(positive_items)
        candidate_scores = {
            node_id: score for node_id, score in ppr_scores.items()
            if node_id not in positive_set and 
            self.graph.node_id_to_type.get(node_id) == 'item' and
            node_id != start_node
        }
        
        # Sort by PPR score (descending)
        sorted_candidates = sorted(
            candidate_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Take top candidates as hard negatives
        num_to_sample = min(num_negatives, len(sorted_candidates))
        hard_negatives = sorted_candidates[:num_to_sample]
        
        # If not enough, pad with easy negatives
        if len(hard_negatives) < num_negatives:
            remaining = num_negatives - len(hard_negatives)
            easy_negatives = self.easy_negative_sampling(
                start_node, 
                positive_items + [nid for nid, _ in hard_negatives],
                remaining
            )
            hard_negatives.extend([(nid, score) for nid, score in easy_negatives])
        
        self.sampling_stats['hard_negatives'] += len(hard_negatives)
        self.sampling_stats['total_samples'] += len(hard_negatives)
        
        return hard_negatives
    
    def hybrid_negative_sampling(self, start_node: str, 
                                positive_items: List[str],
                                num_negatives: int = 10,
                                restart_prob: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Sample both easy and hard negatives.
        
        Args:
            start_node: Starting node ID
            positive_items: List of positive items
            num_negatives: Number of negatives to sample
            restart_prob: PPR restart probability
            
        Returns:
            List[Tuple[str, float]]: List of (negative_item, score) pairs
        """
        num_hard = int(num_negatives * self.hard_negative_ratio)
        num_easy = num_negatives - num_hard
        
        negatives = []
        
        # Sample hard negatives
        if num_hard > 0:
            hard_negatives = self.hard_negative_sampling(
                start_node, 
                positive_items, 
                num_hard, 
                restart_prob
            )
            negatives.extend(hard_negatives)
        
        # Sample easy negatives
        if num_easy > 0:
            positive_set = set(positive_items + [nid for nid, _ in negatives])
            easy_negatives = self.easy_negative_sampling(
                start_node, 
                list(positive_set),
                num_easy
            )
            negatives.extend(easy_negatives)
        
        return negatives
    
    def uniform_negative_sampling(self, start_node: str, 
                                 positive_items: List[str],
                                 num_negatives: int = 10) -> List[Tuple[str, float]]:
        """
        Uniform random negative sampling.
        
        Args:
            start_node: Starting node ID
            positive_items: List of positive items
            num_negatives: Number of negatives to sample
            
        Returns:
            List[Tuple[str, float]]: List of (negative_item, score) pairs
        """
        all_items = [nid for nid, node in self.graph.nodes.items() 
                    if node.node_type == 'item']
        
        positive_set = set(positive_items)
        candidates = [item for item in all_items if item not in positive_set]
        
        if len(candidates) < num_negatives:
            # Not enough candidates, sample with replacement
            sampled = random.choices(candidates, k=num_negatives)
        else:
            sampled = random.sample(candidates, num_negatives)
        
        result = [(item, 1.0 / len(sampled)) for item in sampled]
        
        self.sampling_stats['total_samples'] += len(sampled)
        
        return result
    
    def get_sampling_distribution(self, step: int, 
                                  start_node: str, 
                                  positive_items: List[str],
                                  strategy: str = 'adaptive') -> Dict[str, float]:
        """
        Get sampling distribution based on training step.
        
        Args:
            step: Current training step
            start_node: Starting node ID
            positive_items: List of positive items
            strategy: Sampling strategy ('adaptive', 'fixed', 'annealing')
            
        Returns:
            Dict[str, float]: Sampling distribution
        """
        if strategy == 'adaptive':
            return self._adaptive_sampling_distribution(step, start_node, positive_items)
        elif strategy == 'annealing':
            return self._annealing_sampling_distribution(step, start_node, positive_items)
        else:  # fixed
            return self._fixed_sampling_distribution(start_node, positive_items)
    
    def _adaptive_sampling_distribution(self, step: int, 
                                       start_node: str,
                                       positive_items: List[str]) -> Dict[str, float]:
        """
        Adaptive sampling distribution based on step.
        
        Args:
            step: Current training step
            start_node: Starting node ID
            positive_items: List of positive items
            
        Returns:
            Dict[str, float]: Sampling distribution
        """
        # Adjust ratio of hard negatives based on step
        hard_ratio = min(0.7, 0.1 + 0.01 * (step // 100))
        easy_ratio = 1.0 - hard_ratio
        
        # Get PPR scores for hard negatives
        ppr_scores = self.compute_ppr(start_node)
        
        # Get candidates
        all_items = [nid for nid, node in self.graph.nodes.items() 
                    if node.node_type == 'item']
        positive_set = set(positive_items)
        
        # Build distribution
        distribution = {}
        
        # Add hard negatives (weighted by PPR)
        ppr_items = [(nid, score) for nid, score in ppr_scores.items() 
                    if nid not in positive_set and nid in all_items]
        ppr_items.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 50% as hard candidates
        hard_candidates = ppr_items[:len(ppr_items)//2]
        
        for node_id, score in hard_candidates:
            distribution[node_id] = score * hard_ratio
        
        # Add easy negatives (uniform)
        easy_candidates = [nid for nid in all_items 
                          if nid not in positive_set and nid not in distribution]
        
        if easy_candidates:
            uniform_weight = easy_ratio / len(easy_candidates)
            for node_id in easy_candidates:
                distribution[node_id] = uniform_weight
        
        # Normalize
        total = sum(distribution.values())
        if total > 0:
            distribution = {k: v/total for k, v in distribution.items()}
        
        return distribution
    
    def _annealing_sampling_distribution(self, step: int,
                                        start_node: str,
                                        positive_items: List[str]) -> Dict[str, float]:
        """
        Annealing sampling distribution (temperature-based).
        
        Args:
            step: Current training step
            start_node: Starting node ID
            positive_items: List of positive items
            
        Returns:
            Dict[str, float]: Sampling distribution
        """
        # Temperature decreases over time
        temperature = max(0.1, 1.0 - 0.01 * (step // 50))
        
        # Get PPR scores
        ppr_scores = self.compute_ppr(start_node)
        
        # Get candidates
        all_items = [nid for nid, node in self.graph.nodes.items() 
                    if node.node_type == 'item']
        positive_set = set(positive_items)
        
        # Build distribution with temperature
        distribution = {}
        for node_id in all_items:
            if node_id in positive_set:
                continue
            
            # Base score: PPR score with temperature
            base_score = ppr_scores.get(node_id, 0.0)
            if base_score > 0:
                # Apply temperature
                if temperature < 0.5:
                    # Amplify high scores, suppress low scores
                    distribution[node_id] = base_score ** (1.0 / temperature)
                else:
                    distribution[node_id] = base_score
        
        # Normalize
        total = sum(distribution.values())
        if total > 0:
            distribution = {k: v/total for k, v in distribution.items()}
        else:
            # Fallback to uniform
            uniform_weight = 1.0 / len(all_items)
            distribution = {nid: uniform_weight for nid in all_items if nid not in positive_set}
        
        return distribution
    
    def _fixed_sampling_distribution(self, start_node: str,
                                    positive_items: List[str]) -> Dict[str, float]:
        """
        Fixed sampling distribution (uniform).
        
        Args:
            start_node: Starting node ID
            positive_items: List of positive items
            
        Returns:
            Dict[str, float]: Sampling distribution
        """
        all_items = [nid for nid, node in self.graph.nodes.items() 
                    if node.node_type == 'item']
        positive_set = set(positive_items)
        
        candidates = [nid for nid in all_items if nid not in positive_set]
        
        if not candidates:
            return {}
        
        uniform_weight = 1.0 / len(candidates)
        return {nid: uniform_weight for nid in candidates}
    
    def sample_from_distribution(self, distribution: Dict[str, float], 
                                num_samples: int) -> List[str]:
        """
        Sample nodes from a distribution.
        
        Args:
            distribution: Sampling distribution
            num_samples: Number of samples
            
        Returns:
            List[str]: Sampled node IDs
        """
        if not distribution:
            return []
        
        # Convert to lists
        items = list(distribution.keys())
        weights = [distribution[k] for k in items]
        
        # Sample with replacement
        samples = np.random.choice(items, size=num_samples, p=weights, replace=True)
        return samples.tolist()
    
    def batch_sample_negatives(self, start_nodes: List[str],
                              positive_dict: Dict[str, List[str]],
                              num_negatives_per_node: int = 10,
                              sampling_strategy: str = 'hybrid') -> Dict[str, List[Tuple[str, float]]]:
        """
        Sample negatives for multiple nodes in batch.
        
        Args:
            start_nodes: List of node IDs
            positive_dict: Dictionary mapping node to list of positive items
            num_negatives_per_node: Number of negatives per node
            sampling_strategy: Sampling strategy
            
        Returns:
            Dict[str, List[Tuple[str, float]]]: Mapping node to negative samples
        """
        result = {}
        
        for node_id in tqdm(start_nodes, desc="Batch sampling negatives"):
            positive_items = positive_dict.get(node_id, [])
            negatives = self.sample_negative(
                node_id,
                positive_items,
                num_negatives_per_node,
                sampling_strategy
            )
            result[node_id] = negatives
        
        return result
    
    def get_ppr_neighbors(self, start_node: str, 
                         k: int = 10,
                         restart_prob: Optional[float] = None,
                         min_score: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Get top k neighbors by PPR score.
        
        Args:
            start_node: Starting node ID
            k: Number of neighbors to return
            restart_prob: PPR restart probability
            min_score: Minimum PPR score threshold
            
        Returns:
            List[Tuple[str, float]]: Top k (neighbor, score) pairs
        """
        ppr_scores = self.compute_ppr(start_node, restart_prob)
        
        # Filter by minimum score
        if min_score is not None:
            filtered = [(nid, score) for nid, score in ppr_scores.items() 
                       if nid != start_node and score >= min_score]
        else:
            filtered = [(nid, score) for nid, score in ppr_scores.items() 
                       if nid != start_node]
        
        # Sort by score
        filtered.sort(key=lambda x: x[1], reverse=True)
        
        return filtered[:k]
    
    def get_sampling_stats(self) -> Dict[str, Any]:
        """
        Get sampling statistics.
        
        Returns:
            Dict[str, Any]: Sampling statistics
        """
        stats = self.sampling_stats.copy()
        stats['ppr_cache_size'] = len(self.ppr_cache)
        stats['cache_hit_rate'] = (
            stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
            if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0.0
        )
        stats['avg_sampling_time'] = np.mean(stats['sampling_times']) if stats['sampling_times'] else 0.0
        stats['recent_sampling_times'] = stats['sampling_times'][-10:]
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the PPR cache."""
        self.ppr_cache.clear()
        self.logger.log_info("Cleared PPR cache")
    
    def precompute_ppr_for_nodes(self, node_ids: List[str], 
                                restart_prob: Optional[float] = None) -> None:
        """
        Precompute PPR scores for a list of nodes.
        
        Args:
            node_ids: List of node IDs
            restart_prob: PPR restart probability
        """
        restart_prob = restart_prob or self.default_restart_prob
        
        self.logger.log_info(f"Precomputing PPR for {len(node_ids)} nodes")
        
        for node_id in tqdm(node_ids, desc="Precomputing PPR"):
            self.compute_ppr(node_id, restart_prob, use_cache=True)
    
    def calculate_ppr_similarity(self, node_a: str, node_b: str,
                                restart_prob: Optional[float] = None) -> float:
        """
        Calculate PPR similarity between two nodes.
        
        Args:
            node_a: First node ID
            node_b: Second node ID
            restart_prob: PPR restart probability
            
        Returns:
            float: PPR similarity score
        """
        restart_prob = restart_prob or self.default_restart_prob
        
        # Get PPR scores from node_a
        ppr_scores_a = self.compute_ppr(node_a, restart_prob)
        
        # Get PPR scores from node_b
        ppr_scores_b = self.compute_ppr(node_b, restart_prob)
        
        # Calculate similarity (average of two directions)
        score_ab = ppr_scores_a.get(node_b, 0.0)
        score_ba = ppr_scores_b.get(node_a, 0.0)
        
        return (score_ab + score_ba) / 2.0
    
    def get_importance_weight(self, node_id: str, start_node: str,
                             restart_prob: Optional[float] = None) -> float:
        """
        Get importance weight of a node relative to start node.
        
        Args:
            node_id: Node ID to get weight for
            start_node: Starting node ID
            restart_prob: PPR restart probability
            
        Returns:
            float: Importance weight
        """
        ppr_scores = self.compute_ppr(start_node, restart_prob)
        return ppr_scores.get(node_id, 0.0)
    
    def adaptive_hard_negative_ratio(self, step: int, 
                                    total_steps: int) -> float:
        """
        Calculate adaptive hard negative ratio based on training progress.
        
        Args:
            step: Current training step
            total_steps: Total training steps
            
        Returns:
            float: Hard negative ratio (0-1)
        """
        progress = step / total_steps if total_steps > 0 else 0.0
        
        # Start low, increase gradually
        min_ratio = 0.1
        max_ratio = 0.7
        
        # Sigmoid-like schedule
        ratio = min_ratio + (max_ratio - min_ratio) * (1 / (1 + np.exp(-10 * (progress - 0.5))))
        
        return ratio
    
    def sample_with_replacement(self, candidates: List[str], 
                               weights: Optional[List[float]] = None,
                               num_samples: int = 10) -> List[str]:
        """
        Sample with replacement from candidates.
        
        Args:
            candidates: List of candidate nodes
            weights: Optional weights for sampling
            num_samples: Number of samples
            
        Returns:
            List[str]: Sampled nodes
        """
        if not candidates:
            return []
        
        if weights is None:
            weights = [1.0 / len(candidates)] * len(candidates)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Sample
        indices = np.random.choice(len(candidates), size=num_samples, p=weights, replace=True)
        
        return [candidates[i] for i in indices]
    
    def weighted_neighbor_sampling(self, node_id: str, 
                                  relation_types: Optional[List[str]] = None,
                                  num_samples: int = 10) -> List[Tuple[str, float]]:
        """
        Sample neighbors weighted by edge weights.
        
        Args:
            node_id: Node ID
            relation_types: Types of relations to include
            num_samples: Number of neighbors to sample
            
        Returns:
            List[Tuple[str, float]]: Sampled (neighbor, weight) pairs
        """
        if relation_types is None:
            relation_types = RelationType.get_all_types()
        
        # Get all neighbors with weights
        all_neighbors = []
        for rel_type in relation_types:
            neighbors = self.graph.get_neighbors(node_id, rel_type)
            all_neighbors.extend([(n, w) for n, w in neighbors if w > 0])
        
        if not all_neighbors:
            return []
        
        # Sort by weight
        all_neighbors.sort(key=lambda x: x[1], reverse=True)
        
        # Take top candidates
        top_candidates = all_neighbors[:min(len(all_neighbors), num_samples * 2)]
        
        # Sample with weights
        if len(top_candidates) <= num_samples:
            return top_candidates
        
        # Weighted sampling
        weights = [w for _, w in top_candidates]
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        indices = np.random.choice(len(top_candidates), size=num_samples, p=weights, replace=False)
        
        return [top_candidates[i] for i in indices]
    
    def reset_statistics(self) -> None:
        """Reset sampling statistics."""
        self.sampling_stats = {
            'total_samples': 0,
            'easy_negatives': 0,
            'hard_negatives': 0,
            'ppr_computations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_sampling_time': 0.0,
            'sampling_times': []
        }
        self.logger.log_info("Reset sampling statistics")


# Example usage
if __name__ == "__main__":
    # Load configuration
    config_path = "config/default_config.yaml"
    config_loader = ConfigLoader(config_path)
    config = config_loader.load_config()
    
    # Create dynamic graph
    from models.graph.dynamic_graph import DynamicGraph
    graph = DynamicGraph(config)
    
    # Add some nodes and edges (sample data)
    graph.add_node('user_1', 'user', features={'name': 'Alice'})
    graph.add_node('user_2', 'user', features={'name': 'Bob'})
    graph.add_node('user_3', 'user', features={'name': 'Charlie'})
    graph.add_node('item_1', 'item', features={'title': 'Product A'})
    graph.add_node('item_2', 'item', features={'title': 'Product B'})
    graph.add_node('item_3', 'item', features={'title': 'Product C'})
    graph.add_node('item_4', 'item', features={'title': 'Product D'})
    graph.add_node('item_5', 'item', features={'title': 'Product E'})
    
    graph.add_edge('user_1', 'item_1', RelationType.INTERACT, weight=0.9)
    graph.add_edge('user_1', 'item_2', RelationType.INTERACT, weight=0.7)
    graph.add_edge('user_1', 'item_3', RelationType.INTERACT, weight=0.3)
    graph.add_edge('user_2', 'item_2', RelationType.INTERACT, weight=0.8)
    graph.add_edge('user_2', 'item_3', RelationType.INTERACT, weight=0.6)
    graph.add_edge('user_2', 'item_4', RelationType.INTERACT, weight=0.4)
    graph.add_edge('user_3', 'item_1', RelationType.INTERACT, weight=0.5)
    graph.add_edge('user_3', 'item_5', RelationType.INTERACT, weight=0.7)
    graph.add_edge('user_1', 'user_2', RelationType.SIMILAR_PREF, weight=0.8)
    graph.add_edge('user_1', 'user_3', RelationType.SIMILAR_PREF, weight=0.6)
    graph.add_edge('item_1', 'item_2', RelationType.CONTENT_SIM, weight=0.6)
    graph.add_edge('item_2', 'item_3', RelationType.CONTENT_SIM, weight=0.4)
    
    # Create PPR sampler
    sampler = PPRSampler(graph, config)
    
    # Compute PPR scores
    ppr_scores = sampler.compute_ppr('user_1')
    print(f"PPR scores for user_1: {dict(list(ppr_scores.items())[:5])}")
    
    # Sample negatives
    positive_items = ['item_1', 'item_2']
    easy_negatives = sampler.easy_negative_sampling('user_1', positive_items, num_negatives=3)
    print(f"Easy negatives: {easy_negatives}")
    
    hard_negatives = sampler.hard_negative_sampling('user_1', positive_items, num_negatives=3)
    print(f"Hard negatives: {hard_negatives}")
    
    hybrid_negatives = sampler.hybrid_negative_sampling('user_1', positive_items, num_negatives=5)
    print(f"Hybrid negatives: {hybrid_negatives}")
    
    # Get PPR neighbors
    ppr_neighbors = sampler.get_ppr_neighbors('user_1', k=5)
    print(f"PPR neighbors: {ppr_neighbors}")
    
    # Calculate PPR similarity
    similarity = sampler.calculate_ppr_similarity('item_1', 'item_2')
    print(f"PPR similarity between item_1 and item_2: {similarity:.4f}")
    
    # Get sampling distribution
    distribution = sampler.get_sampling_distribution(100, 'user_1', positive_items)
    print(f"Sampling distribution: {dict(list(distribution.items())[:3])}")
    
    # Get statistics
    stats = sampler.get_sampling_stats()
    print(f"Sampling statistics: {stats}")
    
    # Batch sample negatives
    batch_sampling = sampler.batch_sample_negatives(
        ['user_1', 'user_2'],
        {'user_1': ['item_1', 'item_2'], 'user_2': ['item_2', 'item_3']},
        num_negatives_per_node=3
    )
    print(f"Batch sampling results: {batch_sampling}")