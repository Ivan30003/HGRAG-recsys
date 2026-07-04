"""
data_loader.py - Data loading and batching for H-GRAGrecsys

This module provides data loaders for training, validation, and testing
with support for negative sampling, batch generation, and interaction sampling.
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Iterator, Set
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import logging
from pathlib import Path

from data.dataset import AmazonDataset, BaseDataset

# Configure logging
logger = logging.getLogger(__name__)


class InteractionBatch:
    """Container for a batch of interactions."""
    
    def __init__(self, 
                 user_ids: List[str],
                 item_ids: List[str],
                 ratings: List[float],
                 timestamps: List[int],
                 user_indices: torch.Tensor,
                 item_indices: torch.Tensor,
                 labels: torch.Tensor):
        """
        Initialize InteractionBatch.
        
        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
            ratings: List of ratings
            timestamps: List of timestamps
            user_indices: Tensor of user indices
            item_indices: Tensor of item indices
            labels: Tensor of binary labels
        """
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        self.timestamps = timestamps
        self.user_indices = user_indices
        self.item_indices = item_indices
        self.labels = labels
        self.size = len(user_ids)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert batch to dictionary."""
        return {
            'user_ids': self.user_ids,
            'item_ids': self.item_ids,
            'ratings': self.ratings,
            'timestamps': self.timestamps,
            'user_indices': self.user_indices,
            'item_indices': self.item_indices,
            'labels': self.labels
        }
    
    def to_device(self, device: torch.device):
        """Move tensors to device."""
        self.user_indices = self.user_indices.to(device)
        self.item_indices = self.item_indices.to(device)
        self.labels = self.labels.to(device)
        return self


class RecommendationDataset(Dataset):
    """PyTorch Dataset for recommendation interactions."""
    
    def __init__(self, 
                 dataset: BaseDataset,
                 indices: List[int],
                 user_to_idx: Dict[str, int],
                 item_to_idx: Dict[str, int]):
        """
        Initialize RecommendationDataset.
        
        Args:
            dataset: BaseDataset instance
            indices: List of interaction indices to include
            user_to_idx: Mapping from user_id to index
            item_to_idx: Mapping from item_id to index
        """
        self.dataset = dataset
        self.indices = indices
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        
        # Pre-filter interactions
        self.interactions = []
        for idx in indices:
            interaction = dataset.interactions[idx]
            user_id = interaction['user_id']
            item_id = interaction['item_id']
            if user_id in user_to_idx and item_id in item_to_idx:
                self.interactions.append(interaction)
        
        logger.info(f"Dataset initialized with {len(self.interactions)} interactions")
    
    def __len__(self) -> int:
        return len(self.interactions)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single interaction.
        
        Args:
            idx: Index of interaction
        
        Returns:
            Dictionary containing interaction data
        """
        interaction = self.interactions[idx]
        user_id = interaction['user_id']
        item_id = interaction['item_id']
        
        return {
            'user_id': user_id,
            'item_id': item_id,
            'rating': interaction.get('rating', 1.0),
            'timestamp': interaction.get('timestamp', 0),
            'user_idx': self.user_to_idx[user_id],
            'item_idx': self.item_to_idx[item_id]
        }


class DataLoader:
    """Main data loader class for H-GRAGrecsys."""
    
    def __init__(self, 
                 dataset: BaseDataset,
                 config: Dict[str, Any],
                 batch_size: int = 32,
                 shuffle: bool = True):
        """
        Initialize DataLoader.
        
        Args:
            dataset: BaseDataset instance
            config: Configuration dictionary
            batch_size: Batch size for loading
            shuffle: Whether to shuffle data
        """
        self.dataset = dataset
        self.config = config
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Get config values
        self.seed = config['evaluation'].get('seed', 42)
        self.num_negatives = config['evaluation'].get('num_negatives', 99)
        
        # Create user/item mappings
        self.user_to_idx, self.item_to_idx = self._create_mappings()
        self.idx_to_user = {v: k for k, v in self.user_to_idx.items()}
        self.idx_to_item = {v: k for k, v in self.item_to_idx.items()}
        
        # Validate splits
        self._validate_splits()
        
        # Create datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Initialize loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Set random seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
    
    def _create_mappings(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Create mappings from IDs to indices."""
        # Get all user and item IDs
        user_ids = list(self.dataset.users.keys())
        item_ids = list(self.dataset.items.keys())
        
        # Create mappings
        user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
        
        logger.info(f"Created mappings: {len(user_to_idx)} users, {len(item_to_idx)} items")
        
        return user_to_idx, item_to_idx
    
    def _validate_splits(self) -> None:
        """Validate that splits exist in the dataset."""
        # Check if splits exist
        if not self.dataset.train_indices:
            logger.warning("Train indices not found. Generating splits...")
            self.dataset.split_data()
    
    def _create_dataloader(self, 
                          indices: List[int],
                          batch_size: Optional[int] = None,
                          shuffle: bool = True) -> TorchDataLoader:
        """
        Create PyTorch DataLoader.
        
        Args:
            indices: List of interaction indices
            batch_size: Batch size (uses default if None)
            shuffle: Whether to shuffle
        
        Returns:
            PyTorch DataLoader
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        dataset = RecommendationDataset(
            self.dataset,
            indices,
            self.user_to_idx,
            self.item_to_idx
        )
        
        # Create custom collate function
        def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
            """Collate function for batch."""
            user_ids = [item['user_id'] for item in batch]
            item_ids = [item['item_id'] for item in batch]
            ratings = [item['rating'] for item in batch]
            timestamps = [item['timestamp'] for item in batch]
            user_indices = torch.tensor([item['user_idx'] for item in batch], dtype=torch.long)
            item_indices = torch.tensor([item['item_idx'] for item in batch], dtype=torch.long)
            labels = torch.tensor([1.0] * len(batch), dtype=torch.float32)
            
            return {
                'user_ids': user_ids,
                'item_ids': item_ids,
                'ratings': ratings,
                'timestamps': timestamps,
                'user_indices': user_indices,
                'item_indices': item_indices,
                'labels': labels
            }
        
        loader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        return loader
    
    def prepare_loaders(self) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader]:
        """
        Prepare train, validation, and test loaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create train loader
        self.train_loader = self._create_dataloader(
            self.dataset.train_indices,
            shuffle=True
        )
        self.train_dataset = self.train_loader.dataset
        
        # Create validation loader
        self.val_loader = self._create_dataloader(
            self.dataset.val_indices,
            shuffle=False
        )
        self.val_dataset = self.val_loader.dataset
        
        # Create test loader
        self.test_loader = self._create_dataloader(
            self.dataset.test_indices,
            shuffle=False
        )
        self.test_dataset = self.test_loader.dataset
        
        logger.info(f"Loaders created: Train={len(self.train_loader.dataset)}, "
                   f"Val={len(self.val_loader.dataset)}, "
                   f"Test={len(self.test_loader.dataset)}")
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_batch(self, loader_type: str = 'train') -> Optional[Dict[str, Any]]:
        """
        Get a batch from a specific loader.
        
        Args:
            loader_type: 'train', 'val', or 'test'
        
        Returns:
            Batch dictionary or None if no more batches
        """
        if loader_type == 'train':
            loader = self.train_loader
        elif loader_type == 'val':
            loader = self.val_loader
        elif loader_type == 'test':
            loader = self.test_loader
        else:
            raise ValueError(f"Unknown loader type: {loader_type}")
        
        if loader is None:
            raise ValueError(f"Loader {loader_type} not initialized")
        
        # Get iterator
        iterator = iter(loader)
        
        try:
            batch = next(iterator)
            return batch
        except StopIteration:
            return None
    
    def get_full_batch(self, loader_type: str = 'train') -> Dict[str, Any]:
        """
        Get all batches from a specific loader.
        
        Args:
            loader_type: 'train', 'val', or 'test'
        
        Returns:
            Combined batch dictionary
        """
        if loader_type == 'train':
            loader = self.train_loader
        elif loader_type == 'val':
            loader = self.val_loader
        elif loader_type == 'test':
            loader = self.test_loader
        else:
            raise ValueError(f"Unknown loader type: {loader_type}")
        
        if loader is None:
            raise ValueError(f"Loader {loader_type} not initialized")
        
        all_user_ids = []
        all_item_ids = []
        all_ratings = []
        all_timestamps = []
        all_user_indices = []
        all_item_indices = []
        all_labels = []
        
        for batch in loader:
            all_user_ids.extend(batch['user_ids'])
            all_item_ids.extend(batch['item_ids'])
            all_ratings.extend(batch['ratings'])
            all_timestamps.extend(batch['timestamps'])
            all_user_indices.append(batch['user_indices'])
            all_item_indices.append(batch['item_indices'])
            all_labels.append(batch['labels'])
        
        if all_user_indices:
            return {
                'user_ids': all_user_ids,
                'item_ids': all_item_ids,
                'ratings': all_ratings,
                'timestamps': all_timestamps,
                'user_indices': torch.cat(all_user_indices, dim=0),
                'item_indices': torch.cat(all_item_indices, dim=0),
                'labels': torch.cat(all_labels, dim=0)
            }
        else:
            return {
                'user_ids': [],
                'item_ids': [],
                'ratings': [],
                'timestamps': [],
                'user_indices': torch.tensor([]),
                'item_indices': torch.tensor([]),
                'labels': torch.tensor([])
            }
    
    def sample_interaction_batch(self, 
                                batch_size: int) -> InteractionBatch:
        """
        Sample random interactions from the dataset.
        
        Args:
            batch_size: Number of interactions to sample
        
        Returns:
            InteractionBatch
        """
        # Sample random interactions
        sampled_interactions = random.sample(
            self.dataset.interactions,
            min(batch_size, len(self.dataset.interactions))
        )
        
        user_ids = [i['user_id'] for i in sampled_interactions]
        item_ids = [i['item_id'] for i in sampled_interactions]
        ratings = [i.get('rating', 1.0) for i in sampled_interactions]
        timestamps = [i.get('timestamp', 0) for i in sampled_interactions]
        
        # Convert to indices
        user_indices = torch.tensor(
            [self.user_to_idx[uid] for uid in user_ids if uid in self.user_to_idx],
            dtype=torch.long
        )
        item_indices = torch.tensor(
            [self.item_to_idx[iid] for iid in item_ids if iid in self.item_to_idx],
            dtype=torch.long
        )
        
        labels = torch.ones(len(user_indices), dtype=torch.float32)
        
        return InteractionBatch(
            user_ids=user_ids,
            item_ids=item_ids,
            ratings=ratings,
            timestamps=timestamps,
            user_indices=user_indices,
            item_indices=item_indices,
            labels=labels
        )
    
    def get_user_batch(self, 
                      user_ids: List[str]) -> Tuple[torch.Tensor, List[str]]:
        """
        Get batch of user data.
        
        Args:
            user_ids: List of user IDs
        
        Returns:
            Tuple of (user_indices, existing_user_ids)
        """
        existing_users = []
        user_indices = []
        
        for user_id in user_ids:
            if user_id in self.user_to_idx:
                existing_users.append(user_id)
                user_indices.append(self.user_to_idx[user_id])
        
        if user_indices:
            user_tensor = torch.tensor(user_indices, dtype=torch.long)
        else:
            user_tensor = torch.tensor([], dtype=torch.long)
        
        return user_tensor, existing_users
    
    def get_item_batch(self, 
                      item_ids: List[str]) -> Tuple[torch.Tensor, List[str]]:
        """
        Get batch of item data.
        
        Args:
            item_ids: List of item IDs
        
        Returns:
            Tuple of (item_indices, existing_item_ids)
        """
        existing_items = []
        item_indices = []
        
        for item_id in item_ids:
            if item_id in self.item_to_idx:
                existing_items.append(item_id)
                item_indices.append(self.item_to_idx[item_id])
        
        if item_indices:
            item_tensor = torch.tensor(item_indices, dtype=torch.long)
        else:
            item_tensor = torch.tensor([], dtype=torch.long)
        
        return item_tensor, existing_items


class NegativeSampler:
    """Negative sampling strategies for recommendation."""
    
    def __init__(self, 
                 dataset: BaseDataset,
                 user_to_idx: Dict[str, int],
                 item_to_idx: Dict[str, int],
                 config: Dict[str, Any]):
        """
        Initialize NegativeSampler.
        
        Args:
            dataset: BaseDataset instance
            user_to_idx: User to index mapping
            item_to_idx: Item to index mapping
            config: Configuration dictionary
        """
        self.dataset = dataset
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.config = config
        
        # Build item popularity (for popularity-based sampling)
        self.item_popularity = self._compute_item_popularity()
        
        # Set random seed
        self.seed = config['evaluation'].get('seed', 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
    
    def _compute_item_popularity(self) -> Dict[str, int]:
        """Compute popularity of each item."""
        popularity = defaultdict(int)
        for interaction in self.dataset.interactions:
            item_id = interaction['item_id']
            popularity[item_id] += 1
        return dict(popularity)
    
    def sample_negatives(self, 
                        positive_items: List[str],
                        num_negatives: int = 1,
                        strategy: str = 'uniform') -> List[List[str]]:
        """
        Sample negative items for each positive item.
        
        Args:
            positive_items: List of positive item IDs
            num_negatives: Number of negatives per positive
            strategy: Sampling strategy ('uniform', 'popularity', 'hard')
        
        Returns:
            List of negative item lists
        """
        # Get all item IDs
        all_items = list(self.item_to_idx.keys())
        
        # Remove positive items
        positive_set = set(positive_items)
        candidate_items = [item for item in all_items if item not in positive_set]
        
        negatives = []
        for positive in positive_items:
            # Sample negatives
            if strategy == 'uniform':
                sampled = random.sample(candidate_items, min(num_negatives, len(candidate_items)))
            elif strategy == 'popularity':
                # Sample based on popularity (more popular items more likely to be negative)
                weights = [self.item_popularity.get(item, 0) + 1 for item in candidate_items]
                weights = np.array(weights) / sum(weights)
                sampled = np.random.choice(
                    candidate_items,
                    size=min(num_negatives, len(candidate_items)),
                    replace=False,
                    p=weights
                ).tolist()
            elif strategy == 'hard':
                # For hard negatives, sample from items with similar popularity
                # This simulates hard negative mining
                pos_pop = self.item_popularity.get(positive, 0)
                candidate_weights = [1.0 / (abs(self.item_popularity.get(item, 0) - pos_pop) + 1) 
                                   for item in candidate_items]
                candidate_weights = np.array(candidate_weights) / sum(candidate_weights)
                sampled = np.random.choice(
                    candidate_items,
                    size=min(num_negatives, len(candidate_items)),
                    replace=False,
                    p=candidate_weights
                ).tolist()
            else:
                raise ValueError(f"Unknown sampling strategy: {strategy}")
            
            negatives.append(sampled)
        
        return negatives
    
    def sample_negatives_for_users(self,
                                  user_ids: List[str],
                                  num_negatives: int = 1,
                                  strategy: str = 'uniform') -> Dict[str, List[str]]:
        """
        Sample negative items for each user.
        
        Args:
            user_ids: List of user IDs
            num_negatives: Number of negatives per user
            strategy: Sampling strategy
        
        Returns:
            Dictionary mapping user_id to list of negative items
        """
        user_negatives = {}
        
        for user_id in user_ids:
            # Get user's positive items
            positive_items = self.dataset.get_user_items(user_id)
            
            # Get all items
            all_items = list(self.item_to_idx.keys())
            
            # Candidate items (not interacted with)
            candidate_items = [item for item in all_items if item not in positive_items]
            
            # Sample negatives
            if strategy == 'uniform':
                sampled = random.sample(candidate_items, min(num_negatives, len(candidate_items)))
            elif strategy == 'popularity':
                weights = [self.item_popularity.get(item, 0) + 1 for item in candidate_items]
                weights = np.array(weights) / sum(weights)
                sampled = np.random.choice(
                    candidate_items,
                    size=min(num_negatives, len(candidate_items)),
                    replace=False,
                    p=weights
                ).tolist()
            else:
                raise ValueError(f"Unknown sampling strategy: {strategy}")
            
            user_negatives[user_id] = sampled
        
        return user_negatives


class InteractionDataLoader(DataLoader):
    """Specialized data loader for interaction data."""
    
    def __init__(self, 
                 dataset: BaseDataset,
                 config: Dict[str, Any],
                 batch_size: int = 32,
                 shuffle: bool = True,
                 negative_strategy: str = 'uniform'):
        """
        Initialize InteractionDataLoader.
        
        Args:
            dataset: BaseDataset instance
            config: Configuration dictionary
            batch_size: Batch size
            shuffle: Whether to shuffle
            negative_strategy: Negative sampling strategy
        """
        super().__init__(dataset, config, batch_size, shuffle)
        
        self.negative_strategy = negative_strategy
        self.negative_sampler = NegativeSampler(
            dataset,
            self.user_to_idx,
            self.item_to_idx,
            config
        )
        
        # Pre-compute negative samples for validation/test
        self.val_negatives = None
        self.test_negatives = None
        self._prepare_negative_samples()
    
    def _prepare_negative_samples(self) -> None:
        """Pre-compute negative samples for validation and test sets."""
        # For validation
        val_users = []
        for idx in self.dataset.val_indices:
            interaction = self.dataset.interactions[idx]
            val_users.append(interaction['user_id'])
        
        if val_users:
            self.val_negatives = self.negative_sampler.sample_negatives_for_users(
                val_users,
                num_negatives=self.num_negatives,
                strategy=self.negative_strategy
            )
        
        # For test
        test_users = []
        for idx in self.dataset.test_indices:
            interaction = self.dataset.interactions[idx]
            test_users.append(interaction['user_id'])
        
        if test_users:
            self.test_negatives = self.negative_sampler.sample_negatives_for_users(
                test_users,
                num_negatives=self.num_negatives,
                strategy=self.negative_strategy
            )
    
    def get_negative_samples(self, 
                            user_id: str, 
                            positive_item: str,
                            num_negatives: int = 99) -> List[str]:
        """
        Get negative samples for a user-item pair.
        
        Args:
            user_id: User ID
            positive_item: Positive item ID
            num_negatives: Number of negative samples
        
        Returns:
            List of negative item IDs
        """
        # Get all items
        all_items = list(self.item_to_idx.keys())
        
        # Remove positive item
        candidates = [item for item in all_items if item != positive_item]
        
        # Sample negatives
        negatives = random.sample(candidates, min(num_negatives, len(candidates)))
        
        return negatives
    
    def get_negative_loader(self, 
                           loader_type: str = 'test') -> Optional[Dict[str, List[str]]]:
        """
        Get negative samples for a specific loader.
        
        Args:
            loader_type: 'val' or 'test'
        
        Returns:
            Dictionary mapping user_id to negative items
        """
        if loader_type == 'val':
            return self.val_negatives
        elif loader_type == 'test':
            return self.test_negatives
        else:
            raise ValueError(f"Unknown loader type: {loader_type}")
    
    def get_training_pairs(self, 
                          batch_size: int = 32,
                          num_negatives: int = 1) -> Iterator[Dict[str, Any]]:
        """
        Generate training pairs with negative sampling.
        
        Args:
            batch_size: Batch size
            num_negatives: Number of negatives per positive
        
        Yields:
            Dictionary with positive and negative pairs
        """
        # Get all positive interactions
        interactions = self.dataset.interactions
        random.shuffle(interactions)
        
        for i in range(0, len(interactions), batch_size):
            batch = interactions[i:i + batch_size]
            
            # Extract positive pairs
            positive_users = []
            positive_items = []
            for interaction in batch:
                user_id = interaction['user_id']
                item_id = interaction['item_id']
                if user_id in self.user_to_idx and item_id in self.item_to_idx:
                    positive_users.append(user_id)
                    positive_items.append(item_id)
            
            if not positive_users:
                continue
            
            # Sample negatives for each positive pair
            negative_items = self.negative_sampler.sample_negatives(
                positive_items,
                num_negatives=num_negatives,
                strategy=self.negative_strategy
            )
            
            # Create batch
            user_indices = torch.tensor(
                [self.user_to_idx[uid] for uid in positive_users],
                dtype=torch.long
            )
            item_indices = torch.tensor(
                [self.item_to_idx[iid] for iid in positive_items],
                dtype=torch.long
            )
            
            # Flatten negative items
            flat_negatives = []
            for neg_list in negative_items:
                flat_negatives.extend(neg_list[:num_negatives])
            
            neg_indices = torch.tensor(
                [self.item_to_idx[iid] for iid in flat_negatives if iid in self.item_to_idx],
                dtype=torch.long
            )
            
            # Create labels (1 for positive, 0 for negative)
            pos_labels = torch.ones(len(positive_users), dtype=torch.float32)
            neg_labels = torch.zeros(len(flat_negatives), dtype=torch.float32)
            
            yield {
                'positive_users': positive_users,
                'positive_items': positive_items,
                'negative_items': flat_negatives,
                'user_indices': user_indices,
                'item_indices': item_indices,
                'negative_indices': neg_indices,
                'pos_labels': pos_labels,
                'neg_labels': neg_labels
            }


# Helper functions

def create_data_loaders(dataset: BaseDataset, 
                       config: Dict[str, Any]) -> Dict[str, DataLoader]:
    """
    Create data loaders for all phases.
    
    Args:
        dataset: BaseDataset instance
        config: Configuration dictionary
    
    Returns:
        Dictionary of data loaders for different phases
    """
    batch_size = config['training'].get('batch_size', 32)
    
    # Create base data loader
    base_loader = DataLoader(
        dataset=dataset,
        config=config,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Prepare loaders
    base_loader.prepare_loaders()
    
    # Create interaction data loader
    interaction_loader = InteractionDataLoader(
        dataset=dataset,
        config=config,
        batch_size=batch_size,
        shuffle=True,
        negative_strategy=config.get('negative_strategy', 'uniform')
    )
    
    return {
        'base': base_loader,
        'interaction': interaction_loader
    }


def create_batch_from_indices(dataset: BaseDataset,
                             indices: List[int],
                             user_to_idx: Dict[str, int],
                             item_to_idx: Dict[str, int],
                             device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Create a batch from interaction indices.
    
    Args:
        dataset: BaseDataset instance
        indices: List of interaction indices
        user_to_idx: User to index mapping
        item_to_idx: Item to index mapping
        device: Device to place tensors on
    
    Returns:
        Batch dictionary
    """
    user_ids = []
    item_ids = []
    ratings = []
    timestamps = []
    user_indices = []
    item_indices = []
    
    for idx in indices:
        interaction = dataset.interactions[idx]
        user_id = interaction['user_id']
        item_id = interaction['item_id']
        
        if user_id in user_to_idx and item_id in item_to_idx:
            user_ids.append(user_id)
            item_ids.append(item_id)
            ratings.append(interaction.get('rating', 1.0))
            timestamps.append(interaction.get('timestamp', 0))
            user_indices.append(user_to_idx[user_id])
            item_indices.append(item_to_idx[item_id])
    
    # Convert to tensors
    user_tensor = torch.tensor(user_indices, dtype=torch.long)
    item_tensor = torch.tensor(item_indices, dtype=torch.long)
    rating_tensor = torch.tensor(ratings, dtype=torch.float32)
    timestamp_tensor = torch.tensor(timestamps, dtype=torch.long)
    labels = torch.ones(len(user_indices), dtype=torch.float32)
    
    if device is not None:
        user_tensor = user_tensor.to(device)
        item_tensor = item_tensor.to(device)
        rating_tensor = rating_tensor.to(device)
        timestamp_tensor = timestamp_tensor.to(device)
        labels = labels.to(device)
    
    return {
        'user_ids': user_ids,
        'item_ids': item_ids,
        'ratings': ratings,
        'timestamps': timestamps,
        'user_indices': user_tensor,
        'item_indices': item_tensor,
        'rating_tensor': rating_tensor,
        'timestamp_tensor': timestamp_tensor,
        'labels': labels
    }


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'data': {
            'data_dir': './data/amazon',
            'processed_dir': './data/processed',
            'max_text_length': 512,
            'min_interactions': 5
        },
        'training': {
            'batch_size': 32
        },
        'evaluation': {
            'seed': 42,
            'num_negatives': 99
        },
        'negative_strategy': 'uniform'
    }
    
    # Load dataset
    from data.dataset import AmazonDataset
    
    dataset = AmazonDataset('CDs_and_Vinyl', config)
    dataset.load_data(limit_users=500)
    
    # Create data loaders
    loaders = create_data_loaders(dataset, config)
    
    # Get training batch
    interaction_loader = loaders['interaction']
    for batch in interaction_loader.get_training_pairs(batch_size=32, num_negatives=1):
        print(f"Training batch: {len(batch['positive_users'])} positive pairs, "
              f"{len(batch['negative_items'])} negative items")
        break
    
    # Get test batch
    base_loader = loaders['base']
    test_batch = base_loader.get_batch('test')
    if test_batch:
        print(f"Test batch: {len(test_batch['user_ids'])} interactions")