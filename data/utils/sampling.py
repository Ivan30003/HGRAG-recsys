"""
sampling.py - Data sampling utilities for H-GRAGrecsys

This module provides various sampling strategies for creating dataset subsets,
negative sampling for training, and balanced sampling for evaluation.
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set, Union, Iterator
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
from tqdm import tqdm
import torch
from torch.utils.data import Sampler, WeightedRandomSampler

from data.dataset import BaseDataset
from data.data_preprocessor import TextProcessor

# Configure logging
logger = logging.getLogger(__name__)


class SubsetSampler:
    """Sampling strategies for creating dataset subsets."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SubsetSampler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.random_seed = config['evaluation'].get('seed', 42)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Sampling parameters
        self.min_user_interactions = config.get('min_user_interactions', 5)
        self.min_item_interactions = config.get('min_item_interactions', 5)
        
        logger.info("SubsetSampler initialized")
    
    def sample_users_uniform(self, 
                            dataset: BaseDataset,
                            n_users: int,
                            strategy: str = 'random') -> List[str]:
        """
        Sample users uniformly from dataset.
        
        Args:
            dataset: BaseDataset instance
            n_users: Number of users to sample
            strategy: 'random', 'first', or 'stratified'
        
        Returns:
            List of user IDs
        """
        user_ids = list(dataset.users.keys())
        
        if len(user_ids) <= n_users:
            return user_ids
        
        if strategy == 'random':
            sampled = random.sample(user_ids, n_users)
        elif strategy == 'first':
            sampled = user_ids[:n_users]
        elif strategy == 'stratified':
            sampled = self._sample_users_stratified(dataset, n_users)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        
        logger.info(f"Sampled {len(sampled)} users using {strategy} strategy")
        return sampled
    
    def _sample_users_stratified(self, 
                                dataset: BaseDataset,
                                n_users: int) -> List[str]:
        """
        Sample users stratified by interaction count.
        
        Args:
            dataset: BaseDataset instance
            n_users: Number of users to sample
        
        Returns:
            List of user IDs
        """
        # Get user interaction counts
        user_counts = [(uid, len(items)) for uid, items in dataset.user_items.items()]
        user_counts.sort(key=lambda x: x[1])
        
        # Create strata based on interaction counts
        counts = [count for _, count in user_counts]
        percentiles = [20, 40, 60, 80, 100]
        strata = []
        
        for i in range(len(percentiles)):
            if i == 0:
                lower = 0
            else:
                lower = percentiles[i-1]
            upper = percentiles[i]
            
            lower_val = np.percentile(counts, lower)
            upper_val = np.percentile(counts, upper)
            
            stratum = [
                uid for uid, count in user_counts
                if lower_val <= count < upper_val
            ]
            strata.append(stratum)
        
        # Sample from each stratum
        sampled = []
        per_stratum = n_users // len(strata)
        
        for stratum in strata:
            if len(stratum) > per_stratum:
                sampled.extend(random.sample(stratum, per_stratum))
            else:
                sampled.extend(stratum)
        
        # If still need more users, sample from remaining
        if len(sampled) < n_users:
            remaining = list(set([uid for uid, _ in user_counts]) - set(sampled))
            if remaining:
                to_sample = min(n_users - len(sampled), len(remaining))
                sampled.extend(random.sample(remaining, to_sample))
        
        return sampled[:n_users]
    
    def sample_items_uniform(self,
                            dataset: BaseDataset,
                            n_items: int,
                            strategy: str = 'random') -> List[str]:
        """
        Sample items uniformly from dataset.
        
        Args:
            dataset: BaseDataset instance
            n_items: Number of items to sample
            strategy: 'random', 'first', or 'popularity'
        
        Returns:
            List of item IDs
        """
        item_ids = list(dataset.items.keys())
        
        if len(item_ids) <= n_items:
            return item_ids
        
        if strategy == 'random':
            sampled = random.sample(item_ids, n_items)
        elif strategy == 'first':
            sampled = item_ids[:n_items]
        elif strategy == 'popularity':
            sampled = self._sample_items_by_popularity(dataset, n_items)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        
        logger.info(f"Sampled {len(sampled)} items using {strategy} strategy")
        return sampled
    
    def _sample_items_by_popularity(self,
                                   dataset: BaseDataset,
                                   n_items: int) -> List[str]:
        """
        Sample items by popularity (weighted sampling).
        
        Args:
            dataset: BaseDataset instance
            n_items: Number of items to sample
        
        Returns:
            List of item IDs
        """
        # Get item interaction counts
        item_counts = [(iid, len(users)) for iid, users in dataset.item_users.items()]
        
        # Create weights (proportional to popularity)
        weights = [count for _, count in item_counts]
        weights = np.array(weights) / sum(weights)
        
        # Sample weighted
        items = [iid for iid, _ in item_counts]
        sampled = np.random.choice(
            items,
            size=min(n_items, len(items)),
            replace=False,
            p=weights
        ).tolist()
        
        return sampled
    
    def create_dense_subset(self,
                           dataset: BaseDataset,
                           n_users: int,
                           min_items_per_user: Optional[int] = None) -> BaseDataset:
        """
        Create a dense subset (users with most interactions).
        
        Args:
            dataset: BaseDataset instance
            n_users: Number of users to include
            min_items_per_user: Minimum items per user
        
        Returns:
            Subset dataset
        """
        if min_items_per_user is None:
            min_items_per_user = self.min_user_interactions
        
        # Get users with at least min_items_per_user interactions
        eligible_users = [
            uid for uid, items in dataset.user_items.items()
            if len(items) >= min_items_per_user
        ]
        
        if len(eligible_users) <= n_users:
            selected_users = eligible_users
        else:
            # Sort by interaction count (descending)
            eligible_users.sort(
                key=lambda uid: len(dataset.user_items[uid]),
                reverse=True
            )
            selected_users = eligible_users[:n_users]
        
        logger.info(f"Creating dense subset with {len(selected_users)} users")
        return self._create_subset_from_users(dataset, selected_users)
    
    def create_sparse_subset(self,
                            dataset: BaseDataset,
                            n_users: int,
                            max_items_per_user: Optional[int] = None) -> BaseDataset:
        """
        Create a sparse subset (users with fewest interactions).
        
        Args:
            dataset: BaseDataset instance
            n_users: Number of users to include
            max_items_per_user: Maximum items per user
        
        Returns:
            Subset dataset
        """
        if max_items_per_user is None:
            max_items_per_user = self.min_user_interactions * 2
        
        # Get users with at most max_items_per_user interactions
        eligible_users = [
            uid for uid, items in dataset.user_items.items()
            if len(items) <= max_items_per_user
        ]
        
        if len(eligible_users) <= n_users:
            selected_users = eligible_users
        else:
            # Sort by interaction count (ascending)
            eligible_users.sort(
                key=lambda uid: len(dataset.user_items[uid])
            )
            selected_users = eligible_users[:n_users]
        
        logger.info(f"Creating sparse subset with {len(selected_users)} users")
        return self._create_subset_from_users(dataset, selected_users)
    
    def _create_subset_from_users(self,
                                 dataset: BaseDataset,
                                 user_ids: List[str]) -> BaseDataset:
        """
        Create subset dataset from selected users.
        
        Args:
            dataset: BaseDataset instance
            user_ids: List of selected user IDs
        
        Returns:
            Subset dataset
        """
        # Create subset configuration
        subset_config = self.config.copy()
        subset_config['data']['min_interactions'] = 1
        
        # Create subset dataset
        from data.amazon_dataset import AmazonDataset
        subset = AmazonDataset(dataset.dataset_name, subset_config)
        
        # Build subset data
        subset_users = {}
        subset_items = {}
        subset_interactions = []
        subset_user_items = defaultdict(set)
        subset_item_users = defaultdict(set)
        
        for user_id in user_ids:
            # Copy user
            if user_id in dataset.users:
                subset_users[user_id] = dataset.users[user_id].copy()
            
            # Get user's interactions
            user_ints = [i for i in dataset.interactions if i['user_id'] == user_id]
            subset_interactions.extend(user_ints)
            
            # Update mappings
            for interaction in user_ints:
                item_id = interaction['item_id']
                subset_user_items[user_id].add(item_id)
                subset_item_users[item_id].add(user_id)
                
                # Add item
                if item_id not in subset_items and item_id in dataset.items:
                    subset_items[item_id] = dataset.items[item_id].copy()
        
        # Update subset
        subset.users = subset_users
        subset.items = subset_items
        subset.interactions = subset_interactions
        subset.user_items = dict(subset_user_items)
        subset.item_users = dict(subset_item_users)
        
        # Compute stats
        subset.split_data()
        subset.get_statistics()
        
        return subset
    
    def sample_cold_start_items(self,
                               dataset: BaseDataset,
                               n_items: int,
                               max_interactions: int = 5) -> List[str]:
        """
        Sample cold-start items (items with few interactions).
        
        Args:
            dataset: BaseDataset instance
            n_items: Number of items to sample
            max_interactions: Maximum interactions for cold-start items
        
        Returns:
            List of cold-start item IDs
        """
        # Get items with few interactions
        cold_items = [
            iid for iid, users in dataset.item_users.items()
            if len(users) <= max_interactions
        ]
        
        if not cold_items:
            logger.warning("No cold-start items found")
            return []
        
        # Sample
        if len(cold_items) <= n_items:
            sampled = cold_items
        else:
            sampled = random.sample(cold_items, n_items)
        
        logger.info(f"Sampled {len(sampled)} cold-start items")
        return sampled


class NegativeSampler:
    """Negative sampling strategies for training and evaluation."""
    
    def __init__(self, 
                 dataset: BaseDataset,
                 config: Dict[str, Any]):
        """
        Initialize NegativeSampler.
        
        Args:
            dataset: BaseDataset instance
            config: Configuration dictionary
        """
        self.dataset = dataset
        self.config = config
        self.random_seed = config['evaluation'].get('seed', 42)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Build item popularity
        self.item_popularity = self._compute_item_popularity()
        
        # Cache for negative samples
        self.cache: Dict[str, List[str]] = {}
        
        logger.info("NegativeSampler initialized")
    
    def _compute_item_popularity(self) -> Dict[str, int]:
        """Compute popularity of each item."""
        popularity = defaultdict(int)
        for interaction in self.dataset.interactions:
            item_id = interaction['item_id']
            popularity[item_id] += 1
        return dict(popularity)
    
    def sample_negatives(self,
                        user_id: str,
                        positive_items: List[str],
                        num_negatives: int = 1,
                        strategy: str = 'uniform',
                        exclude_items: Optional[List[str]] = None) -> List[List[str]]:
        """
        Sample negative items for each positive item.
        
        Args:
            user_id: User ID
            positive_items: List of positive item IDs
            num_negatives: Number of negatives per positive
            strategy: 'uniform', 'popularity', 'hard', 'adaptive'
            exclude_items: Items to exclude from sampling
        
        Returns:
            List of negative item lists
        """
        # Get all items
        all_items = list(self.dataset.item_to_idx.keys()) if hasattr(self.dataset, 'item_to_idx') else list(self.dataset.items.keys())
        
        # Get user's positive items
        user_positives = self.dataset.user_items.get(user_id, set())
        
        # Exclude positive items and specified items
        exclude_set = set(positive_items).union(user_positives)
        if exclude_items:
            exclude_set.update(exclude_items)
        
        candidate_items = [item for item in all_items if item not in exclude_set]
        
        if not candidate_items:
            logger.warning(f"No candidate items for user {user_id}")
            return [[] for _ in positive_items]
        
        negatives = []
        for positive in positive_items:
            # Sample based on strategy
            if strategy == 'uniform':
                sampled = self._sample_uniform(candidate_items, num_negatives)
            elif strategy == 'popularity':
                sampled = self._sample_by_popularity(candidate_items, num_negatives)
            elif strategy == 'hard':
                sampled = self._sample_hard(candidate_items, num_negatives, positive)
            elif strategy == 'adaptive':
                sampled = self._sample_adaptive(user_id, candidate_items, num_negatives, positive)
            else:
                raise ValueError(f"Unknown sampling strategy: {strategy}")
            
            negatives.append(sampled)
        
        return negatives
    
    def _sample_uniform(self, candidates: List[str], k: int) -> List[str]:
        """Sample uniformly from candidates."""
        if len(candidates) <= k:
            return candidates[:]
        return random.sample(candidates, k)
    
    def _sample_by_popularity(self, candidates: List[str], k: int) -> List[str]:
        """Sample weighted by popularity."""
        if len(candidates) <= k:
            return candidates[:]
        
        # Get weights
        weights = [self.item_popularity.get(item, 0) + 1 for item in candidates]
        weights = np.array(weights) / sum(weights)
        
        # Sample
        sampled = np.random.choice(
            candidates,
            size=k,
            replace=False,
            p=weights
        ).tolist()
        
        return sampled
    
    def _sample_hard(self, candidates: List[str], k: int, positive: str) -> List[str]:
        """Sample hard negatives (similar to positive item)."""
        if len(candidates) <= k:
            return candidates[:]
        
        # Get positive item popularity
        pos_pop = self.item_popularity.get(positive, 0)
        
        # Calculate similarity to positive (based on popularity)
        weights = []
        for item in candidates:
            item_pop = self.item_popularity.get(item, 0)
            # Items with similar popularity are harder negatives
            similarity = 1.0 / (abs(pos_pop - item_pop) + 1)
            weights.append(similarity)
        
        weights = np.array(weights) / sum(weights)
        
        # Sample
        sampled = np.random.choice(
            candidates,
            size=k,
            replace=False,
            p=weights
        ).tolist()
        
        return sampled
    
    def _sample_adaptive(self, 
                        user_id: str,
                        candidates: List[str],
                        k: int,
                        positive: str) -> List[str]:
        """
        Adaptive sampling using user preferences.
        
        Args:
            user_id: User ID
            candidates: List of candidate items
            k: Number of negatives
            positive: Positive item
        
        Returns:
            Sampled negatives
        """
        if len(candidates) <= k:
            return candidates[:]
        
        # Get user's preferred categories
        user_prefs = self.dataset.users.get(user_id, {})
        pref_categories = user_prefs.get('preferences', {}).get('preferred_categories', {})
        
        if not pref_categories:
            # Fallback to hard sampling
            return self._sample_hard(candidates, k, positive)
        
        # Get top categories
        top_categories = sorted(pref_categories.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Score candidates
        scores = []
        for item in candidates:
            item_data = self.dataset.items.get(item, {})
            item_category = item_data.get('category', '')
            
            # Check if item belongs to preferred categories
            category_score = 0
            for cat, weight in top_categories:
                if cat in item_category:
                    category_score += weight
            
            # Combine with popularity
            pop_score = self.item_popularity.get(item, 0) / max(self.item_popularity.values())
            combined_score = 0.6 * category_score + 0.4 * pop_score
            
            scores.append(combined_score)
        
        # Normalize scores
        scores = np.array(scores)
        scores = scores / (scores.sum() + 1e-8)
        
        # Sample
        sampled = np.random.choice(
            candidates,
            size=k,
            replace=False,
            p=scores
        ).tolist()
        
        return sampled
    
    def sample_negatives_for_user(self,
                                 user_id: str,
                                 num_negatives: int = 99,
                                 strategy: str = 'uniform') -> List[str]:
        """
        Sample negative items for a user.
        
        Args:
            user_id: User ID
            num_negatives: Number of negatives to sample
            strategy: Sampling strategy
        
        Returns:
            List of negative item IDs
        """
        # Check cache
        cache_key = f"{user_id}_{num_negatives}_{strategy}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Get user's positive items
        positive_items = self.dataset.user_items.get(user_id, set())
        
        # Get all items
        all_items = list(self.dataset.items.keys())
        
        # Candidate items (not interacted with)
        candidate_items = [item for item in all_items if item not in positive_items]
        
        if not candidate_items:
            return []
        
        # Sample
        if strategy == 'uniform':
            sampled = random.sample(candidate_items, min(num_negatives, len(candidate_items)))
        elif strategy == 'popularity':
            sampled = self._sample_by_popularity(candidate_items, min(num_negatives, len(candidate_items)))
        elif strategy == 'hard':
            # For hard negatives, we need a positive item
            if positive_items:
                positive = random.choice(list(positive_items))
                sampled = self._sample_hard(candidate_items, min(num_negatives, len(candidate_items)), positive)
            else:
                sampled = self._sample_uniform(candidate_items, min(num_negatives, len(candidate_items)))
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        
        # Cache
        self.cache[cache_key] = sampled
        
        return sampled
    
    def sample_negatives_for_users(self,
                                   user_ids: List[str],
                                   num_negatives: int = 99,
                                   strategy: str = 'uniform') -> Dict[str, List[str]]:
        """
        Sample negative items for multiple users.
        
        Args:
            user_ids: List of user IDs
            num_negatives: Number of negatives per user
            strategy: Sampling strategy
        
        Returns:
            Dictionary mapping user_id to negative items
        """
        negatives = {}
        for user_id in tqdm(user_ids, desc="Sampling negatives"):
            negatives[user_id] = self.sample_negatives_for_user(
                user_id, num_negatives, strategy
            )
        
        return negatives
    
    def create_evaluation_negatives(self,
                                   dataset: BaseDataset,
                                   strategy: str = 'uniform',
                                   num_negatives: int = 99) -> Dict[str, List[str]]:
        """
        Create negative samples for evaluation.
        
        Args:
            dataset: BaseDataset instance
            strategy: Sampling strategy
            num_negatives: Number of negatives per user
        
        Returns:
            Dictionary mapping user_id to negative items
        """
        # Get test users
        test_users = []
        for idx in dataset.test_indices:
            interaction = dataset.interactions[idx]
            if interaction['user_id'] not in test_users:
                test_users.append(interaction['user_id'])
        
        return self.sample_negatives_for_users(test_users, num_negatives, strategy)
    
    def clear_cache(self) -> None:
        """Clear negative sample cache."""
        self.cache.clear()
        logger.info("Cleared negative sampler cache")


class BalancedSampler:
    """Balanced sampling strategies for training."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize BalancedSampler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.random_seed = config['evaluation'].get('seed', 42)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        logger.info("BalancedSampler initialized")
    
    def balance_by_interactions(self,
                               dataset: BaseDataset,
                               target_count: Optional[int] = None) -> List[str]:
        """
        Balance users by interaction count.
        
        Args:
            dataset: BaseDataset instance
            target_count: Target number of interactions per user
        
        Returns:
            List of balanced user IDs
        """
        if target_count is None:
            # Find median interaction count
            counts = [len(items) for items in dataset.user_items.values()]
            target_count = int(np.median(counts))
        
        balanced_users = []
        for user_id, items in dataset.user_items.items():
            if len(items) >= target_count:
                # Randomly sample target_count items
                sampled_items = random.sample(list(items), target_count)
                balanced_users.append(user_id)
        
        logger.info(f"Balanced {len(balanced_users)} users with target {target_count} interactions")
        return balanced_users
    
    def balance_by_popularity(self,
                             dataset: BaseDataset,
                             bins: int = 5) -> Dict[str, List[str]]:
        """
        Balance items by popularity bins.
        
        Args:
            dataset: BaseDataset instance
            bins: Number of popularity bins
        
        Returns:
            Dictionary mapping bin to item IDs
        """
        # Get item popularity
        item_popularity = [(iid, len(users)) for iid, users in dataset.item_users.items()]
        item_popularity.sort(key=lambda x: x[1])
        
        # Create bins
        bin_size = len(item_popularity) // bins
        bins_dict = {}
        
        for i in range(bins):
            start = i * bin_size
            end = start + bin_size if i < bins - 1 else len(item_popularity)
            bin_items = [iid for iid, _ in item_popularity[start:end]]
            bins_dict[f"bin_{i}_{item_popularity[start][1]}_{item_popularity[end-1][1]}"] = bin_items
        
        logger.info(f"Balanced items into {len(bins_dict)} popularity bins")
        return bins_dict
    
    def sample_balanced_batch(self,
                             dataset: BaseDataset,
                             batch_size: int,
                             user_weights: Optional[Dict[str, float]] = None) -> List[str]:
        """
        Sample balanced batch of users.
        
        Args:
            dataset: BaseDataset instance
            batch_size: Batch size
            user_weights: User weights (optional)
        
        Returns:
            List of user IDs
        """
        user_ids = list(dataset.users.keys())
        
        if user_weights is None:
            # Weight by interaction count inverse (to balance)
            counts = [len(dataset.user_items[uid]) for uid in user_ids]
            weights = [1.0 / (count + 1) for count in counts]
            weights = np.array(weights) / sum(weights)
        else:
            weights = np.array([user_weights.get(uid, 1.0) for uid in user_ids])
            weights = weights / sum(weights)
        
        sampled = np.random.choice(
            user_ids,
            size=min(batch_size, len(user_ids)),
            replace=False,
            p=weights
        ).tolist()
        
        return sampled


class HardNegativeMiner:
    """Hard negative mining for training."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize HardNegativeMiner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.random_seed = config['evaluation'].get('seed', 42)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Mining parameters
        self.initial_hard_ratio = config.get('initial_hard_ratio', 0.1)
        self.final_hard_ratio = config.get('final_hard_ratio', 0.5)
        self.warmup_steps = config.get('hard_mining_warmup', 1000)
        
        # Cache for mined negatives
        self.mined_cache: Dict[str, List[str]] = {}
        
        logger.info("HardNegativeMiner initialized")
    
    def mine_hard_negatives(self,
                           user_id: str,
                           positive_items: List[str],
                           candidate_items: List[str],
                           embeddings: Dict[str, np.ndarray],
                           num_negatives: int = 10,
                           step: int = 0) -> List[str]:
        """
        Mine hard negatives using embeddings.
        
        Args:
            user_id: User ID
            positive_items: List of positive item IDs
            candidate_items: List of candidate item IDs
            embeddings: Item embeddings
            num_negatives: Number of negatives to mine
            step: Current training step
        
        Returns:
            List of hard negative item IDs
        """
        if not candidate_items:
            return []
        
        # Get user embedding (average of positive items)
        positive_embeddings = []
        for item in positive_items:
            if item in embeddings:
                positive_embeddings.append(embeddings[item])
        
        if not positive_embeddings:
            return random.sample(candidate_items, min(num_negatives, len(candidate_items)))
        
        user_embedding = np.mean(positive_embeddings, axis=0)
        
        # Calculate similarity scores
        scores = []
        for item in candidate_items:
            if item in embeddings:
                sim = np.dot(user_embedding, embeddings[item])
                scores.append((item, sim))
        
        # Sort by similarity (highest = hardest)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Determine how many hard negatives to return
        hard_ratio = self._get_hard_ratio(step)
        num_hard = int(num_negatives * hard_ratio)
        num_easy = num_negatives - num_hard
        
        # Sample hard negatives (top scores)
        hard_negatives = [item for item, _ in scores[:num_hard]]
        
        # Sample easy negatives (random from remaining)
        remaining_items = [item for item, _ in scores[num_hard:]]
        easy_negatives = random.sample(
            remaining_items,
            min(num_easy, len(remaining_items))
        )
        
        all_negatives = hard_negatives + easy_negatives
        
        return all_negatives
    
    def _get_hard_ratio(self, step: int) -> float:
        """
        Get hard negative ratio based on training step.
        
        Args:
            step: Current training step
        
        Returns:
            Hard negative ratio
        """
        if step < self.warmup_steps:
            # Linear warmup
            progress = step / self.warmup_steps
            ratio = self.initial_hard_ratio + progress * (self.final_hard_ratio - self.initial_hard_ratio)
        else:
            ratio = self.final_hard_ratio
        
        return min(ratio, 1.0)
    
    def mine_batch_negatives(self,
                            user_ids: List[str],
                            positive_items: List[List[str]],
                            candidate_items: List[List[str]],
                            embeddings: Dict[str, np.ndarray],
                            num_negatives: int = 10,
                            step: int = 0) -> List[List[str]]:
        """
        Mine hard negatives for a batch.
        
        Args:
            user_ids: List of user IDs
            positive_items: List of positive item lists
            candidate_items: List of candidate item lists
            embeddings: Item embeddings
            num_negatives: Number of negatives per item
            step: Current training step
        
        Returns:
            List of negative item lists
        """
        mined = []
        for i, user_id in enumerate(user_ids):
            negatives = self.mine_hard_negatives(
                user_id,
                positive_items[i],
                candidate_items[i],
                embeddings,
                num_negatives,
                step
            )
            mined.append(negatives)
        
        return mined


class PyTorchSampler(Sampler):
    """Custom PyTorch sampler for balanced sampling."""
    
    def __init__(self,
                 weights: List[float],
                 num_samples: int,
                 replacement: bool = True,
                 random_seed: Optional[int] = None):
        """
        Initialize PyTorchSampler.
        
        Args:
            weights: Sampling weights
            num_samples: Number of samples
            replacement: Whether to sample with replacement
            random_seed: Random seed
        """
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.num_samples = num_samples
        self.replacement = replacement
        self.random_seed = random_seed
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
    
    def __iter__(self) -> Iterator[int]:
        """Generate indices."""
        return iter(torch.multinomial(
            self.weights,
            self.num_samples,
            self.replacement
        ).tolist())
    
    def __len__(self) -> int:
        """Return number of samples."""
        return self.num_samples


# Helper functions

def create_subsets_for_experiments(dataset: BaseDataset,
                                  config: Dict[str, Any]) -> Dict[str, BaseDataset]:
    """
    Create all subsets needed for experiments.
    
    Args:
        dataset: BaseDataset instance
        config: Configuration dictionary
    
    Returns:
        Dictionary of subsets
    """
    sampler = SubsetSampler(config)
    
    subsets = {}
    
    # Dense and sparse subsets for 100 and 500 users
    for n_users in [100, 500]:
        dense_name = f"dense_{n_users}"
        sparse_name = f"sparse_{n_users}"
        
        logger.info(f"Creating {dense_name}")
        subsets[dense_name] = sampler.create_dense_subset(dataset, n_users)
        
        logger.info(f"Creating {sparse_name}")
        subsets[sparse_name] = sampler.create_sparse_subset(dataset, n_users)
    
    return subsets


def create_negative_samples_for_evaluation(dataset: BaseDataset,
                                          config: Dict[str, Any],
                                          num_negatives: int = 99) -> Dict[str, List[str]]:
    """
    Create negative samples for evaluation.
    
    Args:
        dataset: BaseDataset instance
        config: Configuration dictionary
        num_negatives: Number of negatives per user
    
    Returns:
        Dictionary mapping user_id to negative items
    """
    sampler = NegativeSampler(dataset, config)
    negatives = sampler.create_evaluation_negatives(dataset, strategy='uniform', num_negatives=num_negatives)
    
    return negatives


def sample_cold_start_users(dataset: BaseDataset,
                           n_users: int,
                           min_interactions: int = 1,
                           max_interactions: int = 5) -> List[str]:
    """
    Sample cold-start users.
    
    Args:
        dataset: BaseDataset instance
        n_users: Number of users to sample
        min_interactions: Minimum interactions
        max_interactions: Maximum interactions
    
    Returns:
        List of cold-start user IDs
    """
    cold_users = []
    for user_id, items in dataset.user_items.items():
        if min_interactions <= len(items) <= max_interactions:
            cold_users.append(user_id)
    
    if len(cold_users) > n_users:
        return random.sample(cold_users, n_users)
    
    return cold_users


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'data': {
            'data_dir': './data/amazon',
            'processed_dir': './data/processed',
            'min_interactions': 5
        },
        'evaluation': {
            'seed': 42,
            'num_negatives': 99
        },
        'initial_hard_ratio': 0.1,
        'final_hard_ratio': 0.5,
        'hard_mining_warmup': 1000
    }
    
    # Load dataset
    from data.amazon_dataset import AmazonDataset
    
    dataset = AmazonDataset('CDs_and_Vinyl', config)
    dataset.load_data(limit_users=500)
    
    # Create subsets
    subsets = create_subsets_for_experiments(dataset, config)
    
    for name, subset in subsets.items():
        print(f"Subset {name}: {subset.statistics}")
    
    # Create negative samples
    negatives = create_negative_samples_for_evaluation(dataset, config)
    print(f"Created negatives for {len(negatives)} users")
    
    # Sample cold-start users
    cold_users = sample_cold_start_users(dataset, n_users=50)
    print(f"Sampled {len(cold_users)} cold-start users")