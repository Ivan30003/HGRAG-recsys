"""
dataset.py - Dataset loading and preprocessing for H-GRAGrecsys

This module handles loading Amazon review datasets, preprocessing text features,
and creating sampled subsets for the recommendation experiments.
"""

import os
import json
import pickle
import random
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class DatasetStatistics:
    """Container for dataset statistics."""
    num_users: int
    num_items: int
    num_interactions: int
    sparsity: float
    avg_words: float
    user_interaction_stats: Dict[str, float]
    item_interaction_stats: Dict[str, float]
    dense_ratio: float

class BaseDataset:
    """Base dataset class for handling recommendation data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize BaseDataset.
        
        Args:
            config: Configuration dictionary containing data parameters
        """
        self.config = config
        self.data_dir = config['data']['data_dir']
        self.max_text_length = config['data'].get('max_text_length', 512)
        self.min_interactions = config['data'].get('min_interactions', 5)
        self.validation_ratio = config['data'].get('validation_ratio', 0.1)
        self.test_ratio = config['data'].get('test_ratio', 0.2)
        self.seed = config['evaluation']['seed']
        
        # Data containers
        self.users: Dict[str, Dict] = {}
        self.items: Dict[str, Dict] = {}
        self.interactions: List[Dict] = []
        self.user_items: Dict[str, Set[str]] = defaultdict(set)
        self.item_users: Dict[str, Set[str]] = defaultdict(set)
        
        # Split indices
        self.train_indices: List[int] = []
        self.val_indices: List[int] = []
        self.test_indices: List[int] = []
        
        # Statistics
        self.statistics: Optional[DatasetStatistics] = None
        
        # Set seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
    
    def load_data(self) -> None:
        """Load dataset from files. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement load_data()")
    
    def get_user_items(self, user_id: str) -> Set[str]:
        """Get items interacted with by a user."""
        return self.user_items.get(user_id, set())
    
    def get_item_users(self, item_id: str) -> Set[str]:
        """Get users who interacted with an item."""
        return self.item_users.get(item_id, set())
    
    def get_item_features(self, item_id: str) -> Dict:
        """Get features/attributes of an item."""
        return self.items.get(item_id, {})
    
    def get_user_features(self, user_id: str) -> Dict:
        """Get features/attributes of a user."""
        return self.users.get(user_id, {})
    
    def get_interactions(self) -> List[Dict]:
        """Get all interactions."""
        return self.interactions
    
    def split_data(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Split data into train, validation, test sets using leave-one-out protocol.
        
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        # Group interactions by user
        user_interactions = defaultdict(list)
        for idx, interaction in enumerate(self.interactions):
            user_id = interaction['user_id']
            user_interactions[user_id].append(idx)
        
        train_idx, val_idx, test_idx = [], [], []
        
        for user_id, indices in user_interactions.items():
            # Sort by timestamp if available
            if 'timestamp' in self.interactions[indices[0]]:
                indices.sort(key=lambda x: self.interactions[x]['timestamp'])
            
            # Ensure at least min_interactions + 1 for validation/test
            if len(indices) >= self.min_interactions + 1:
                # Keep most recent for test, second most recent for validation
                test_idx.append(indices[-1])
                if len(indices) >= self.min_interactions + 2:
                    val_idx.append(indices[-2])
                    train_idx.extend(indices[:-2])
                else:
                    val_idx.append(indices[-2])
                    train_idx.extend(indices[:-2])
            else:
                # If not enough interactions, put all in train
                train_idx.extend(indices)
        
        # Shuffle train indices
        random.shuffle(train_idx)
        
        self.train_indices = train_idx
        self.val_indices = val_idx
        self.test_indices = test_idx
        
        logger.info(f"Data split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        
        return train_idx, val_idx, test_idx
    
    def get_statistics(self) -> DatasetStatistics:
        """Compute and return dataset statistics."""
        if self.statistics is not None:
            return self.statistics
        
        num_users = len(self.users)
        num_items = len(self.items)
        num_interactions = len(self.interactions)
        sparsity = 1 - (num_interactions / (num_users * num_items))
        
        # Word count statistics
        avg_words = 0
        word_counts = []
        for item_id, item_data in self.items.items():
            if 'description' in item_data:
                words = len(item_data['description'].split())
                word_counts.append(words)
        avg_words = np.mean(word_counts) if word_counts else 0
        
        # User interaction statistics
        user_inter_counts = [len(interactions) for interactions in self.user_items.values()]
        item_inter_counts = [len(interactions) for interactions in self.item_users.values()]
        
        self.statistics = DatasetStatistics(
            num_users=num_users,
            num_items=num_items,
            num_interactions=num_interactions,
            sparsity=sparsity,
            avg_words=avg_words,
            user_interaction_stats={
                'mean': np.mean(user_inter_counts) if user_inter_counts else 0,
                'std': np.std(user_inter_counts) if user_inter_counts else 0,
                'min': np.min(user_inter_counts) if user_inter_counts else 0,
                'max': np.max(user_inter_counts) if user_inter_counts else 0,
            },
            item_interaction_stats={
                'mean': np.mean(item_inter_counts) if item_inter_counts else 0,
                'std': np.std(item_inter_counts) if item_inter_counts else 0,
                'min': np.min(item_inter_counts) if item_inter_counts else 0,
                'max': np.max(item_inter_counts) if item_inter_counts else 0,
            },
            dense_ratio=num_interactions / (num_users * num_items)
        )
        
        return self.statistics

class AmazonDataset(BaseDataset):
    """Amazon review dataset handler."""
    
    def __init__(self, dataset_name: str, config: Dict[str, Any]):
        """
        Initialize AmazonDataset.
        
        Args:
            dataset_name: Name of Amazon dataset ('CDs_and_Vinyl' or 'Office_Products')
            config: Configuration dictionary
        """
        super().__init__(config)
        self.dataset_name = dataset_name
        
        # Determine file paths
        self.reviews_path = Path(self.data_dir) / dataset_name / 'reviews.json'
        self.metadata_path = Path(self.data_dir) / dataset_name / 'metadata.json'
        
        # Check if files exist
        if not self.reviews_path.exists():
            raise FileNotFoundError(f"Reviews file not found: {self.reviews_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        logger.info(f"Initialized AmazonDataset for {dataset_name}")
    
    def load_data(self, limit_users: Optional[int] = None) -> None:
        """
        Load Amazon review and metadata data.
        
        Args:
            limit_users: Optional limit on number of users to load
        """
        logger.info(f"Loading Amazon dataset: {self.dataset_name}")
        
        # Load reviews
        reviews_data = self._load_reviews(limit_users)
        
        # Load metadata
        metadata_data = self._load_metadata()
        
        # Build dataset
        self._build_dataset(reviews_data, metadata_data)
        
        # Split data
        self.split_data()
        
        # Compute statistics
        self.get_statistics()
        
        logger.info(f"Loaded dataset: {self.statistics}")
    
    def _load_reviews(self, limit_users: Optional[int] = None) -> Dict[str, List[Dict]]:
        """
        Load review data from JSON file.
        
        Args:
            limit_users: Optional limit on number of users
        
        Returns:
            Dictionary mapping user_id to list of reviews
        """
        logger.info("Loading reviews...")
        
        user_reviews = defaultdict(list)
        user_count = 0
        
        with open(self.reviews_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading reviews"):
                try:
                    review = json.loads(line.strip())
                    user_id = review['reviewerID']
                    item_id = review['asin']
                    
                    # Skip if user limit reached
                    if limit_users and user_id not in user_reviews:
                        if user_count >= limit_users:
                            continue
                    
                    # Extract review data
                    review_data = {
                        'user_id': user_id,
                        'item_id': item_id,
                        'rating': float(review.get('overall', 0)),
                        'timestamp': int(review.get('unixReviewTime', 0)),
                        'review_text': review.get('reviewText', ''),
                        'summary': review.get('summary', '')
                    }
                    
                    user_reviews[user_id].append(review_data)
                    
                    if user_id not in user_reviews:
                        user_count += 1
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse review line: {e}")
                    continue
                except KeyError as e:
                    logger.warning(f"Missing key in review: {e}")
                    continue
        
        logger.info(f"Loaded {len(user_reviews)} users with reviews")
        return dict(user_reviews)
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """
        Load item metadata from JSON file.
        
        Returns:
            Dictionary mapping item_id to metadata
        """
        logger.info("Loading metadata...")
        
        metadata = {}
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading metadata"):
                try:
                    item_data = json.loads(line.strip())
                    item_id = item_data.get('asin')
                    
                    if not item_id:
                        continue
                    
                    # Extract metadata
                    metadata[item_id] = {
                        'title': item_data.get('title', ''),
                        'description': item_data.get('description', ''),
                        'category': item_data.get('category', ''),
                        'price': item_data.get('price', 0.0),
                        'brand': item_data.get('brand', ''),
                        'average_rating': float(item_data.get('averageRating', 0)),
                        'num_ratings': int(item_data.get('ratingNumber', 0))
                    }
                    
                    # Clean text fields
                    if 'description' in metadata[item_id] and isinstance(metadata[item_id]['description'], list):
                        metadata[item_id]['description'] = ' '.join(metadata[item_id]['description'])
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse metadata line: {e}")
                    continue
                except KeyError as e:
                    logger.warning(f"Missing key in metadata: {e}")
                    continue
        
        logger.info(f"Loaded metadata for {len(metadata)} items")
        return metadata
    
    def _build_dataset(self, reviews_data: Dict[str, List[Dict]], 
                      metadata_data: Dict[str, Dict]) -> None:
        """
        Build dataset from reviews and metadata.
        
        Args:
            reviews_data: Dictionary mapping user_id to list of reviews
            metadata_data: Dictionary mapping item_id to metadata
        """
        logger.info("Building dataset...")
        
        # Build user and item mappings
        user_items = defaultdict(set)
        item_users = defaultdict(set)
        interactions = []
        
        # Track user metadata
        for user_id, reviews in tqdm(reviews_data.items(), desc="Processing users"):
            # Create user
            user_data = {
                'user_id': user_id,
                'num_interactions': len(reviews),
                'average_rating': np.mean([r['rating'] for r in reviews]),
                'preferences': self._extract_user_preferences(reviews)
            }
            self.users[user_id] = user_data
            
            # Process each review
            for review in reviews:
                item_id = review['item_id']
                
                # Skip if item has no metadata
                if item_id not in metadata_data:
                    continue
                
                # Create interaction
                interaction = {
                    'user_id': user_id,
                    'item_id': item_id,
                    'rating': review['rating'],
                    'timestamp': review['timestamp'],
                    'review_text': review['review_text'],
                    'summary': review.get('summary', '')
                }
                interactions.append(interaction)
                
                # Update mappings
                user_items[user_id].add(item_id)
                item_users[item_id].add(user_id)
        
        # Build items
        for item_id in tqdm(metadata_data.keys(), desc="Processing items"):
            if item_id in item_users:  # Only keep items with interactions
                item_data = metadata_data[item_id]
                item_data['item_id'] = item_id
                item_data['num_interactions'] = len(item_users[item_id])
                self.items[item_id] = item_data
        
        # Update data containers
        self.interactions = interactions
        self.user_items = dict(user_items)
        self.item_users = dict(item_users)
        
        # Filter users with too few interactions
        if self.min_interactions > 0:
            self._filter_by_interactions()
        
        logger.info(f"Built dataset: {len(self.users)} users, {len(self.items)} items, "
                   f"{len(self.interactions)} interactions")
    
    def _extract_user_preferences(self, reviews: List[Dict]) -> Dict:
        """
        Extract user preferences from review history.
        
        Args:
            reviews: List of user reviews
        
        Returns:
            Dictionary of user preferences
        """
        if not reviews:
            return {}
        
        # Analyze rating patterns
        ratings = [r['rating'] for r in reviews]
        avg_rating = np.mean(ratings)
        
        # Extract preference categories based on high ratings
        preferred_items = [r for r in reviews if r['rating'] >= 4.0]
        
        # Get preferred categories (if metadata available)
        preferred_categories = []
        for review in preferred_items:
            item_id = review['item_id']
            if item_id in self.items:
                category = self.items[item_id].get('category', '')
                if category:
                    preferred_categories.append(category)
        
        # Count category preferences
        category_counts = defaultdict(int)
        for cat in preferred_categories:
            category_counts[cat] += 1
        
        return {
            'average_rating': avg_rating,
            'num_high_ratings': len(preferred_items),
            'preferred_categories': dict(category_counts),
            'high_rating_items': [r['item_id'] for r in preferred_items[:10]]
        }
    
    def _filter_by_interactions(self) -> None:
        """Filter users and items with insufficient interactions."""
        # Filter users
        users_to_keep = set()
        for user_id, items in self.user_items.items():
            if len(items) >= self.min_interactions:
                users_to_keep.add(user_id)
        
        # Filter items
        items_to_keep = set()
        for item_id, users in self.item_users.items():
            if len(users) >= self.min_interactions:
                items_to_keep.add(item_id)
        
        # Update data
        filtered_interactions = []
        for interaction in self.interactions:
            if interaction['user_id'] in users_to_keep and interaction['item_id'] in items_to_keep:
                filtered_interactions.append(interaction)
        
        # Update mappings
        user_items = defaultdict(set)
        item_users = defaultdict(set)
        for interaction in filtered_interactions:
            user_items[interaction['user_id']].add(interaction['item_id'])
            item_users[interaction['item_id']].add(interaction['user_id'])
        
        # Update containers
        self.users = {uid: self.users[uid] for uid in users_to_keep}
        self.items = {iid: self.items[iid] for iid in items_to_keep}
        self.interactions = filtered_interactions
        self.user_items = dict(user_items)
        self.item_users = dict(item_users)
        
        logger.info(f"Filtered dataset: {len(self.users)} users, {len(self.items)} items, "
                   f"{len(self.interactions)} interactions")
    
    def sample_subset(self, n_users: int, density: str = 'dense') -> 'AmazonDataset':
        """
        Create a sampled subset of the dataset.
        
        Args:
            n_users: Number of users to sample
            density: 'dense' or 'sparse' sampling strategy
        
        Returns:
            Sampled AmazonDataset
        """
        logger.info(f"Creating {density} subset with {n_users} users")
        
        # Sample users
        all_users = list(self.users.keys())
        
        if density == 'dense':
            # Select users with most interactions (dense)
            user_interaction_counts = [(uid, len(self.user_items[uid])) for uid in all_users]
            user_interaction_counts.sort(key=lambda x: x[1], reverse=True)
            selected_users = [uid for uid, _ in user_interaction_counts[:n_users]]
        else:  # sparse
            # Select users with fewest interactions (sparse)
            user_interaction_counts = [(uid, len(self.user_items[uid])) for uid in all_users]
            user_interaction_counts.sort(key=lambda x: x[1])
            selected_users = [uid for uid, _ in user_interaction_counts[:n_users]]
        
        # Create new dataset
        subset_config = self.config.copy()
        subset_config['data']['min_interactions'] = 1  # Lower threshold for subset
        
        # Initialize dataset (will be in-memory)
        subset = AmazonDataset(self.dataset_name, subset_config)
        
        # Build subset data
        subset_users = {}
        subset_items = {}
        subset_interactions = []
        subset_user_items = defaultdict(set)
        subset_item_users = defaultdict(set)
        
        for user_id in selected_users:
            # Copy user data
            subset_users[user_id] = self.users[user_id].copy()
            
            # Get user's interactions
            user_interactions = [i for i in self.interactions if i['user_id'] == user_id]
            subset_interactions.extend(user_interactions)
            
            # Update mappings
            for interaction in user_interactions:
                item_id = interaction['item_id']
                subset_user_items[user_id].add(item_id)
                subset_item_users[item_id].add(user_id)
                
                # Add item if not exists
                if item_id not in subset_items and item_id in self.items:
                    subset_items[item_id] = self.items[item_id].copy()
        
        # Update subset containers
        subset.users = subset_users
        subset.items = subset_items
        subset.interactions = subset_interactions
        subset.user_items = dict(subset_user_items)
        subset.item_users = dict(subset_item_users)
        
        # Split and compute statistics
        subset.split_data()
        subset.get_statistics()
        
        logger.info(f"Created {density} subset: {len(subset.users)} users, "
                   f"{len(subset.items)} items, {len(subset.interactions)} interactions")
        
        return subset
    
    def get_dense_subset(self, n_users: int = 100) -> 'AmazonDataset':
        """
        Get dense subset (users with most interactions).
        
        Args:
            n_users: Number of users to include
        
        Returns:
            Dense subset dataset
        """
        return self.sample_subset(n_users, 'dense')
    
    def get_sparse_subset(self, n_users: int = 100) -> 'AmazonDataset':
        """
        Get sparse subset (users with fewest interactions).
        
        Args:
            n_users: Number of users to include
        
        Returns:
            Sparse subset dataset
        """
        return self.sample_subset(n_users, 'sparse')
    
    def get_full_dataset(self) -> 'AmazonDataset':
        """Get full dataset without sampling."""
        return self.sample_subset(len(self.users), 'dense')
    
    def save_to_disk(self, output_dir: str) -> None:
        """
        Save dataset to disk in pickle format.
        
        Args:
            output_dir: Directory to save dataset files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create dataset object for saving
        save_data = {
            'dataset_name': self.dataset_name,
            'users': self.users,
            'items': self.items,
            'interactions': self.interactions,
            'user_items': self.user_items,
            'item_users': self.item_users,
            'train_indices': self.train_indices,
            'val_indices': self.val_indices,
            'test_indices': self.test_indices,
            'statistics': self.statistics,
            'config': self.config
        }
        
        # Save to pickle
        save_path = output_path / f"{self.dataset_name}_dataset.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        # Also save statistics as JSON
        stats_path = output_path / f"{self.dataset_name}_statistics.json"
        if self.statistics:
            stats_dict = {
                'num_users': self.statistics.num_users,
                'num_items': self.statistics.num_items,
                'num_interactions': self.statistics.num_interactions,
                'sparsity': self.statistics.sparsity,
                'avg_words': self.statistics.avg_words,
                'user_interaction_stats': self.statistics.user_interaction_stats,
                'item_interaction_stats': self.statistics.item_interaction_stats,
                'dense_ratio': self.statistics.dense_ratio
            }
            with open(stats_path, 'w') as f:
                json.dump(stats_dict, f, indent=2)
        
        logger.info(f"Dataset saved to {save_path}")
    
    @classmethod
    def load_from_disk(cls, dataset_name: str, load_path: str, config: Dict[str, Any]) -> 'AmazonDataset':
        """
        Load dataset from disk.
        
        Args:
            dataset_name: Name of the dataset
            load_path: Path to the saved dataset
            config: Configuration dictionary
        
        Returns:
            Loaded AmazonDataset instance
        """
        load_file = Path(load_path) / f"{dataset_name}_dataset.pkl"
        
        if not load_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {load_file}")
        
        with open(load_file, 'rb') as f:
            data = pickle.load(f)
        
        # Create dataset instance
        dataset = cls(dataset_name, config)
        
        # Restore data
        dataset.users = data['users']
        dataset.items = data['items']
        dataset.interactions = data['interactions']
        dataset.user_items = data['user_items']
        dataset.item_users = data['item_users']
        dataset.train_indices = data.get('train_indices', [])
        dataset.val_indices = data.get('val_indices', [])
        dataset.test_indices = data.get('test_indices', [])
        dataset.statistics = data.get('statistics')
        
        logger.info(f"Dataset loaded from {load_file}")
        return dataset

def create_datasets(dataset_names: List[str], config: Dict[str, Any], 
                   output_dir: str, force_reload: bool = False) -> Dict[str, AmazonDataset]:
    """
    Create Amazon datasets for specified dataset names.
    
    Args:
        dataset_names: List of dataset names to create
        config: Configuration dictionary
        output_dir: Directory to save datasets
        force_reload: Whether to force reload from source
    
    Returns:
        Dictionary mapping dataset names to AmazonDataset instances
    """
    datasets = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for dataset_name in dataset_names:
        logger.info(f"Processing dataset: {dataset_name}")
        
        # Check if dataset already exists
        load_file = output_path / f"{dataset_name}_dataset.pkl"
        if load_file.exists() and not force_reload:
            logger.info(f"Loading existing dataset: {dataset_name}")
            dataset = AmazonDataset.load_from_disk(dataset_name, output_dir, config)
        else:
            logger.info(f"Creating new dataset: {dataset_name}")
            dataset = AmazonDataset(dataset_name, config)
            dataset.load_data()
            dataset.save_to_disk(output_dir)
        
        datasets[dataset_name] = dataset
    
    return datasets

def create_experiment_datasets(config: Dict[str, Any]) -> Dict[str, Dict[str, AmazonDataset]]:
    """
    Create all datasets needed for experiments.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Dictionary containing all datasets for experiments
    """
    dataset_names = config['data'].get('datasets', ['CDs_and_Vinyl', 'Office_Products'])
    output_dir = config['data']['processed_dir']
    
    # Create full datasets
    logger.info("Creating full datasets...")
    full_datasets = create_datasets(dataset_names, config, output_dir)
    
    experiment_datasets = {}
    
    for name, full_dataset in full_datasets.items():
        logger.info(f"Creating subsets for {name}")
        
        # Create dense and sparse subsets
        dense_100 = full_dataset.get_dense_subset(100)
        sparse_100 = full_dataset.get_sparse_subset(100)
        dense_500 = full_dataset.get_dense_subset(500)
        sparse_500 = full_dataset.get_sparse_subset(500)
        
        # Save subsets
        dense_100.save_to_disk(output_dir)
        sparse_100.save_to_disk(output_dir)
        dense_500.save_to_disk(output_dir)
        sparse_500.save_to_disk(output_dir)
        
        experiment_datasets[name] = {
            'full': full_dataset,
            'dense_100': dense_100,
            'sparse_100': sparse_100,
            'dense_500': dense_500,
            'sparse_500': sparse_500
        }
    
    return experiment_datasets

def get_dataset_interaction_matrix(dataset: AmazonDataset) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Create interaction matrix from dataset.
    
    Args:
        dataset: AmazonDataset instance
    
    Returns:
        Tuple of (interaction_matrix, user_ids, item_ids)
    """
    user_ids = list(dataset.users.keys())
    item_ids = list(dataset.items.keys())
    
    # Create mapping
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
    
    # Initialize matrix
    matrix = np.zeros((len(user_ids), len(item_ids)), dtype=np.float32)
    
    # Fill matrix
    for interaction in dataset.interactions:
        user_idx = user_to_idx[interaction['user_id']]
        item_idx = item_to_idx[interaction['item_id']]
        matrix[user_idx, item_idx] = interaction['rating']
    
    return matrix, user_ids, item_ids

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
        'evaluation': {
            'seed': 42
        }
    }
    
    # Create datasets
    datasets = create_experiment_datasets(config)
    
    # Access individual datasets
    for name, subset_dict in datasets.items():
        print(f"\nDataset: {name}")
        for subset_name, dataset in subset_dict.items():
            if dataset.statistics:
                stats = dataset.statistics
                print(f"  {subset_name}: Users={stats.num_users}, Items={stats.num_items}, "
                      f"Interactions={stats.num_interactions}, Sparsity={stats.sparsity:.4f}")