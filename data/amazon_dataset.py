"""
amazon_dataset.py - Amazon dataset loader and manager for H-GRAGrecsys

This module provides specialized Amazon dataset handling with support for
multiple subsets, caching, and integration with the data preprocessing pipeline.
"""

import os
import json
import pickle
import gzip
import shutil
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import hashlib
from datetime import datetime
import requests
from urllib.parse import urlparse
import time

from data.dataset import BaseDataset, DatasetStatistics
from data.data_preprocessor import DataPreprocessor, PreprocessedItem, PreprocessedUser

# Configure logging
logger = logging.getLogger(__name__)


class AmazonDatasetManager:
    """Manager for downloading and caching Amazon datasets."""
    
    # Amazon review dataset URLs (from UCSD datasets)
    DATASET_URLS = {
        'CDs_and_Vinyl': {
            'reviews': 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/CDs_and_Vinyl_5.json.gz',
            'metadata': 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_CDs_and_Vinyl.json.gz'
        },
        'Office_Products': {
            'reviews': 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Office_Products_5.json.gz',
            'metadata': 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Office_Products.json.gz'
        },
        'Electronics': {
            'reviews': 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Electronics_5.json.gz',
            'metadata': 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Electronics.json.gz'
        },
        'Movies_and_TV': {
            'reviews': 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Movies_and_TV_5.json.gz',
            'metadata': 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Movies_and_TV.json.gz'
        }
    }
    
    def __init__(self, data_dir: str, config: Dict[str, Any]):
        """
        Initialize AmazonDatasetManager.
        
        Args:
            data_dir: Directory to store datasets
            config: Configuration dictionary
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        
        # Subdirectories
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.cache_dir = self.data_dir / 'cache'
        
        for d in [self.raw_dir, self.processed_dir, self.cache_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Check for existing datasets
        self.available_datasets = self._check_available_datasets()
        
        logger.info(f"AmazonDatasetManager initialized with {len(self.available_datasets)} datasets available")
    
    def _check_available_datasets(self) -> List[str]:
        """Check which datasets are already available."""
        available = []
        for dataset_name in self.DATASET_URLS.keys():
            dataset_dir = self.raw_dir / dataset_name
            if dataset_dir.exists():
                reviews_file = dataset_dir / 'reviews.json'
                metadata_file = dataset_dir / 'metadata.json'
                if reviews_file.exists() or (dataset_dir / 'reviews.json.gz').exists():
                    available.append(dataset_name)
        
        return available
    
    def download_dataset(self, dataset_name: str, force: bool = False) -> bool:
        """
        Download dataset files.
        
        Args:
            dataset_name: Name of dataset to download
            force: Force re-download even if exists
        
        Returns:
            True if download successful
        """
        if dataset_name not in self.DATASET_URLS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_dir = self.raw_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        urls = self.DATASET_URLS[dataset_name]
        
        for file_type, url in urls.items():
            filename = f"{file_type}.json.gz"
            filepath = dataset_dir / filename
            
            if filepath.exists() and not force:
                logger.info(f"File {filename} already exists, skipping")
                continue
            
            logger.info(f"Downloading {url} to {filepath}")
            
            try:
                # Download with retry
                self._download_file(url, filepath)
                
                # Extract gzip
                extracted_path = dataset_dir / f"{file_type}.json"
                if not extracted_path.exists() or force:
                    self._extract_gzip(filepath, extracted_path)
                
                logger.info(f"Successfully downloaded and extracted {file_type}")
                
            except Exception as e:
                logger.error(f"Failed to download {file_type}: {e}")
                return False
        
        # Update available datasets
        self.available_datasets = self._check_available_datasets()
        
        return True
    
    def _download_file(self, url: str, filepath: Path, max_retries: int = 3) -> None:
        """
        Download file with retry logic.
        
        Args:
            url: URL to download
            filepath: Local file path
            max_retries: Maximum number of retries
        """
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(filepath, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, 
                             desc=f"Downloading {filepath.name}") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                return
                
            except Exception as e:
                retries += 1
                logger.warning(f"Download failed (attempt {retries}/{max_retries}): {e}")
                
                if retries < max_retries:
                    wait_time = 2 ** retries  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    raise
        
        raise RuntimeError(f"Failed to download {url} after {max_retries} attempts")
    
    def _extract_gzip(self, gz_path: Path, output_path: Path) -> None:
        """
        Extract gzip file.
        
        Args:
            gz_path: Path to gzip file
            output_path: Path to extracted file
        """
        logger.info(f"Extracting {gz_path} to {output_path}")
        
        with gzip.open(gz_path, 'rt', encoding='utf-8') as f_in:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for line in tqdm(f_in, desc=f"Extracting {gz_path.name}"):
                    f_out.write(line)
        
        logger.info(f"Extraction complete: {output_path}")
    
    def get_dataset_paths(self, dataset_name: str) -> Dict[str, Path]:
        """
        Get file paths for a dataset.
        
        Args:
            dataset_name: Name of dataset
        
        Returns:
            Dictionary of file paths
        """
        dataset_dir = self.raw_dir / dataset_name
        
        reviews_json = dataset_dir / 'reviews.json'
        metadata_json = dataset_dir / 'metadata.json'
        
        # If .json files don't exist, check for .json.gz
        if not reviews_json.exists():
            reviews_gz = dataset_dir / 'reviews.json.gz'
            if reviews_gz.exists():
                self._extract_gzip(reviews_gz, reviews_json)
        
        if not metadata_json.exists():
            metadata_gz = dataset_dir / 'metadata.json.gz'
            if metadata_gz.exists():
                self._extract_gzip(metadata_gz, metadata_json)
        
        return {
            'reviews': reviews_json,
            'metadata': metadata_json
        }
    
    def delete_dataset(self, dataset_name: str) -> None:
        """
        Delete dataset files.
        
        Args:
            dataset_name: Name of dataset to delete
        """
        dataset_dir = self.raw_dir / dataset_name
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
            logger.info(f"Deleted dataset: {dataset_name}")
        
        # Update available datasets
        self.available_datasets = self._check_available_datasets()
    
    def list_available_datasets(self) -> List[str]:
        """List available datasets."""
        return self.available_datasets


class AmazonDataset(BaseDataset):
    """Amazon dataset loader with preprocessing capabilities."""
    
    def __init__(self, dataset_name: str, config: Dict[str, Any]):
        """
        Initialize AmazonDataset.
        
        Args:
            dataset_name: Name of Amazon dataset (e.g., 'CDs_and_Vinyl')
            config: Configuration dictionary
        """
        super().__init__(config)
        self.dataset_name = dataset_name
        self.manager = AmazonDatasetManager(config['data']['data_dir'], config)
        
        # Preprocessor
        self.preprocessor = None
        
        # Processed data cache
        self.processed_data = None
        
        # Data versions
        self.version = config.get('dataset_version', '5-core')
        
        # Statistics
        self.statistics = None
        
        # Load if available
        paths = self.manager.get_dataset_paths(dataset_name)
        self.reviews_path = paths['reviews']
        self.metadata_path = paths['metadata']
        
        # Check if data exists
        if not self.reviews_path.exists() or not self.metadata_path.exists():
            logger.warning(f"Dataset files not found for {dataset_name}. Use download_dataset() first.")
        
        logger.info(f"AmazonDataset initialized for {dataset_name}")
    
    def download_data(self, force: bool = False) -> bool:
        """
        Download dataset files.
        
        Args:
            force: Force re-download
        
        Returns:
            True if successful
        """
        return self.manager.download_dataset(self.dataset_name, force)
    
    def load_data(self, limit_users: Optional[int] = None, 
                  force_reload: bool = False) -> None:
        """
        Load data from files.
        
        Args:
            limit_users: Optional limit on number of users
            force_reload: Force reload even if cached
        """
        logger.info(f"Loading data for {self.dataset_name} (limit_users={limit_users})")
        
        # Check cache
        cache_key = f"{self.dataset_name}_{limit_users if limit_users else 'full'}"
        cache_file = self.manager.cache_dir / f"{cache_key}_data.pkl"
        
        if cache_file.exists() and not force_reload:
            logger.info(f"Loading from cache: {cache_file}")
            self._load_from_cache(cache_file)
            return
        
        # Check if files exist
        if not self.reviews_path.exists():
            raise FileNotFoundError(f"Reviews file not found: {self.reviews_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        # Load data
        reviews_data = self._load_reviews(limit_users)
        metadata_data = self._load_metadata()
        
        # Build dataset
        self._build_dataset(reviews_data, metadata_data)
        
        # Split data
        self.split_data()
        
        # Compute statistics
        self.get_statistics()
        
        # Cache processed data
        self._save_to_cache(cache_file)
        
        logger.info(f"Loaded {self.dataset_name}: {self.statistics}")
    
    def _load_reviews(self, limit_users: Optional[int] = None) -> Dict[str, List[Dict]]:
        """
        Load review data from file.
        
        Args:
            limit_users: Optional user limit
        
        Returns:
            Dictionary of user reviews
        """
        logger.info("Loading reviews...")
        
        user_reviews = defaultdict(list)
        user_count = 0
        
        # Try to read reviews
        try:
            with open(self.reviews_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Loading reviews"):
                    try:
                        review = json.loads(line.strip())
                        user_id = review['reviewerID']
                        item_id = review['asin']
                        
                        # Check user limit
                        if limit_users and user_id not in user_reviews:
                            if user_count >= limit_users:
                                continue
                            user_count += 1
                        
                        # Process review
                        review_data = {
                            'user_id': user_id,
                            'item_id': item_id,
                            'rating': float(review.get('overall', 0)),
                            'timestamp': int(review.get('unixReviewTime', 0)),
                            'review_text': review.get('reviewText', ''),
                            'summary': review.get('summary', '')
                        }
                        
                        # Validate rating
                        if review_data['rating'] < 0 or review_data['rating'] > 5:
                            continue
                        
                        # Keep only if enough content
                        if len(review_data['review_text']) < 10 and len(review_data['summary']) < 5:
                            continue
                        
                        user_reviews[user_id].append(review_data)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse review line: {e}")
                        continue
                    except KeyError as e:
                        logger.warning(f"Missing key in review: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error loading reviews: {e}")
            raise
        
        # Filter users with too few reviews
        min_reviews = self.min_interactions
        filtered_reviews = {
            uid: reviews for uid, reviews in user_reviews.items()
            if len(reviews) >= min_reviews
        }
        
        logger.info(f"Loaded {len(filtered_reviews)} users with reviews")
        return dict(filtered_reviews)
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """
        Load item metadata from file.
        
        Returns:
            Dictionary of item metadata
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
                        'description': self._parse_description(item_data.get('description', '')),
                        'category': self._parse_category(item_data.get('category', '')),
                        'brand': item_data.get('brand', ''),
                        'price': float(item_data.get('price', 0)) if item_data.get('price') else 0.0,
                        'average_rating': float(item_data.get('averageRating', 0)) if item_data.get('averageRating') else 0.0,
                        'num_ratings': int(item_data.get('ratingNumber', 0)) if item_data.get('ratingNumber') else 0,
                        'features': item_data.get('features', [])
                    }
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse metadata line: {e}")
                    continue
                except KeyError as e:
                    logger.warning(f"Missing key in metadata: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing metadata: {e}")
                    continue
        
        # Clean metadata
        cleaned_metadata = {}
        for item_id, meta in metadata.items():
            # Clean text fields
            meta['title'] = self._clean_text(meta['title'])
            meta['description'] = self._clean_text(meta['description'])
            meta['category'] = self._clean_text(meta['category'])
            
            # Skip items with no content
            if not meta['title'] and not meta['description']:
                continue
            
            cleaned_metadata[item_id] = meta
        
        logger.info(f"Loaded metadata for {len(cleaned_metadata)} items")
        return cleaned_metadata
    
    def _parse_description(self, description: Union[str, List[str]]) -> str:
        """Parse description field."""
        if not description:
            return ''
        if isinstance(description, list):
            return ' '.join(str(d) for d in description)
        return str(description)
    
    def _parse_category(self, category: Union[str, List[str]]) -> str:
        """Parse category field."""
        if not category:
            return ''
        if isinstance(category, list):
            return category[0] if category else ''
        return str(category)
    
    def _clean_text(self, text: str) -> str:
        """Clean text field."""
        if not text or not isinstance(text, str):
            return ''
        # Remove extra whitespace and special characters
        import re
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def _build_dataset(self, reviews_data: Dict[str, List[Dict]], 
                      metadata_data: Dict[str, Dict]) -> None:
        """
        Build dataset from reviews and metadata.
        
        Args:
            reviews_data: User reviews
            metadata_data: Item metadata
        """
        logger.info("Building dataset...")
        
        # Track items with interactions
        item_has_interaction = defaultdict(int)
        
        # Build interactions
        interactions = []
        user_items = defaultdict(set)
        item_users = defaultdict(set)
        
        for user_id, reviews in tqdm(reviews_data.items(), desc="Processing users"):
            # Create user
            user_data = {
                'user_id': user_id,
                'num_interactions': len(reviews),
                'average_rating': np.mean([r['rating'] for r in reviews]) if reviews else 0,
                'preferences': self._extract_user_preferences(reviews)
            }
            self.users[user_id] = user_data
            
            # Process each review
            for review in reviews:
                item_id = review['item_id']
                
                # Skip if item not in metadata
                if item_id not in metadata_data:
                    continue
                
                # Track item
                item_has_interaction[item_id] += 1
                
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
        for item_id, meta in tqdm(metadata_data.items(), desc="Processing items"):
            if item_id in item_users:  # Only keep items with interactions
                item_data = meta.copy()
                item_data['item_id'] = item_id
                item_data['num_interactions'] = len(item_users[item_id])
                self.items[item_id] = item_data
        
        # Update data containers
        self.interactions = interactions
        self.user_items = dict(user_items)
        self.item_users = dict(item_users)
        
        # Filter by min interactions
        if self.min_interactions > 0:
            self._filter_by_interactions()
        
        logger.info(f"Built dataset: {len(self.users)} users, {len(self.items)} items, "
                   f"{len(self.interactions)} interactions")
    
    def _extract_user_preferences(self, reviews: List[Dict]) -> Dict:
        """
        Extract user preferences from reviews.
        
        Args:
            reviews: List of user reviews
        
        Returns:
            User preferences dictionary
        """
        if not reviews:
            return {}
        
        ratings = [r['rating'] for r in reviews]
        
        # Category preferences (from items)
        categories = defaultdict(int)
        high_rated_items = []
        
        for review in reviews:
            if review['rating'] >= 4.0:
                high_rated_items.append(review['item_id'])
            
            # Try to get category from item
            item_id = review['item_id']
            if item_id in self.items:
                category = self.items[item_id].get('category', '')
                if category:
                    categories[category] += 1
        
        return {
            'average_rating': np.mean(ratings) if ratings else 0,
            'num_high_ratings': len(high_rated_items),
            'preferred_categories': dict(categories),
            'high_rating_items': high_rated_items[:20],
            'rating_distribution': {
                k: ratings.count(k) for k in [1, 2, 3, 4, 5]
            }
        }
    
    def _filter_by_interactions(self) -> None:
        """Filter users and items with insufficient interactions."""
        # Filter users
        users_to_keep = {
            uid for uid, items in self.user_items.items()
            if len(items) >= self.min_interactions
        }
        
        # Filter items
        items_to_keep = {
            iid for iid, users in self.item_users.items()
            if len(users) >= self.min_interactions
        }
        
        # Filter interactions
        filtered_interactions = [
            i for i in self.interactions
            if i['user_id'] in users_to_keep and i['item_id'] in items_to_keep
        ]
        
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
    
    def _save_to_cache(self, cache_file: Path) -> None:
        """
        Save data to cache.
        
        Args:
            cache_file: Path to cache file
        """
        cache_data = {
            'users': self.users,
            'items': self.items,
            'interactions': self.interactions,
            'user_items': self.user_items,
            'item_users': self.item_users,
            'train_indices': self.train_indices,
            'val_indices': self.val_indices,
            'test_indices': self.test_indices,
            'statistics': self.statistics,
            'config': self.config,
            'dataset_name': self.dataset_name
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Saved to cache: {cache_file}")
    
    def _load_from_cache(self, cache_file: Path) -> None:
        """
        Load data from cache.
        
        Args:
            cache_file: Path to cache file
        """
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.users = cache_data['users']
        self.items = cache_data['items']
        self.interactions = cache_data['interactions']
        self.user_items = cache_data['user_items']
        self.item_users = cache_data['item_users']
        self.train_indices = cache_data.get('train_indices', [])
        self.val_indices = cache_data.get('val_indices', [])
        self.test_indices = cache_data.get('test_indices', [])
        self.statistics = cache_data.get('statistics')
        
        logger.info(f"Loaded from cache: {cache_file}")
    
    def get_statistics(self) -> DatasetStatistics:
        """Get dataset statistics."""
        if self.statistics is not None:
            return self.statistics
        
        num_users = len(self.users)
        num_items = len(self.items)
        num_interactions = len(self.interactions)
        sparsity = 1 - (num_interactions / (num_users * num_items)) if num_users * num_items > 0 else 1.0
        
        # Calculate average words
        word_counts = []
        for item in self.items.values():
            description = item.get('description', '')
            if description:
                word_counts.append(len(description.split()))
        
        avg_words = np.mean(word_counts) if word_counts else 0
        
        # User/item statistics
        user_inter_counts = [len(items) for items in self.user_items.values()]
        item_inter_counts = [len(users) for users in self.item_users.values()]
        
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
                'max': np.max(user_inter_counts) if user_inter_counts else 0
            },
            item_interaction_stats={
                'mean': np.mean(item_inter_counts) if item_inter_counts else 0,
                'std': np.std(item_inter_counts) if item_inter_counts else 0,
                'min': np.min(item_inter_counts) if item_inter_counts else 0,
                'max': np.max(item_inter_counts) if item_inter_counts else 0
            },
            dense_ratio=num_interactions / (num_users * num_items) if num_users * num_items > 0 else 0
        )
        
        return self.statistics
    
    def create_subsets(self, n_users_list: List[int] = [100, 500]) -> Dict[str, 'AmazonDataset']:
        """
        Create dense and sparse subsets.
        
        Args:
            n_users_list: List of user counts for subsets
        
        Returns:
            Dictionary of subset datasets
        """
        subsets = {}
        
        for n_users in n_users_list:
            # Dense subset
            dense_name = f"dense_{n_users}"
            dense_subset = self._create_subset(n_users, 'dense')
            subsets[dense_name] = dense_subset
            
            # Sparse subset
            sparse_name = f"sparse_{n_users}"
            sparse_subset = self._create_subset(n_users, 'sparse')
            subsets[sparse_name] = sparse_subset
        
        return subsets
    
    def _create_subset(self, n_users: int, strategy: str) -> 'AmazonDataset':
        """
        Create a subset of the dataset.
        
        Args:
            n_users: Number of users to include
            strategy: 'dense' or 'sparse'
        
        Returns:
            Subset dataset
        """
        logger.info(f"Creating {strategy} subset with {n_users} users")
        
        # Select users based on strategy
        user_interaction_counts = [
            (uid, len(items)) for uid, items in self.user_items.items()
        ]
        
        if strategy == 'dense':
            user_interaction_counts.sort(key=lambda x: x[1], reverse=True)
        else:  # sparse
            user_interaction_counts.sort(key=lambda x: x[1])
        
        selected_users = [uid for uid, _ in user_interaction_counts[:n_users]]
        
        # Create subset data
        subset_config = self.config.copy()
        subset_config['data']['min_interactions'] = 1
        
        subset = AmazonDataset(self.dataset_name, subset_config)
        
        # Build subset
        subset_users = {}
        subset_items = {}
        subset_interactions = []
        subset_user_items = defaultdict(set)
        subset_item_users = defaultdict(set)
        
        for user_id in selected_users:
            # Copy user
            subset_users[user_id] = self.users[user_id].copy()
            
            # Get user's interactions
            user_ints = [i for i in self.interactions if i['user_id'] == user_id]
            subset_interactions.extend(user_ints)
            
            # Update mappings
            for interaction in user_ints:
                item_id = interaction['item_id']
                subset_user_items[user_id].add(item_id)
                subset_item_users[item_id].add(user_id)
                
                # Add item
                if item_id not in subset_items and item_id in self.items:
                    subset_items[item_id] = self.items[item_id].copy()
        
        # Update subset
        subset.users = subset_users
        subset.items = subset_items
        subset.interactions = subset_interactions
        subset.user_items = dict(subset_user_items)
        subset.item_users = dict(subset_item_users)
        
        # Compute stats
        subset.split_data()
        subset.get_statistics()
        
        # Save subset
        subset_name = f"{self.dataset_name}_{strategy}_{n_users}"
        subset.save_subset(subset_name)
        
        logger.info(f"Created {strategy} subset: {len(subset.users)} users, "
                   f"{len(subset.items)} items, {len(subset.interactions)} interactions")
        
        return subset
    
    def save_subset(self, subset_name: str) -> None:
        """
        Save subset to disk.
        
        Args:
            subset_name: Name of the subset
        """
        subset_dir = self.manager.processed_dir / self.dataset_name / 'subsets'
        subset_dir.mkdir(parents=True, exist_ok=True)
        
        subset_file = subset_dir / f"{subset_name}.pkl"
        
        # Convert numpy arrays to lists for serialization
        save_data = {
            'users': self.users,
            'items': self.items,
            'interactions': self.interactions,
            'user_items': dict(self.user_items),
            'item_users': dict(self.item_users),
            'train_indices': self.train_indices,
            'val_indices': self.val_indices,
            'test_indices': self.test_indices,
            'statistics': self.statistics,
            'config': self.config,
            'dataset_name': self.dataset_name,
            'subset_name': subset_name
        }
        
        with open(subset_file, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Saved subset to {subset_file}")
    
    @classmethod
    def load_subset(cls, dataset_name: str, subset_name: str, config: Dict[str, Any]) -> 'AmazonDataset':
        """
        Load subset from disk.
        
        Args:
            dataset_name: Name of dataset
            subset_name: Name of subset
            config: Configuration dictionary
        
        Returns:
            Loaded AmazonDataset
        """
        manager = AmazonDatasetManager(config['data']['data_dir'], config)
        subset_file = manager.processed_dir / dataset_name / 'subsets' / f"{subset_name}.pkl"
        
        if not subset_file.exists():
            raise FileNotFoundError(f"Subset file not found: {subset_file}")
        
        with open(subset_file, 'rb') as f:
            data = pickle.load(f)
        
        # Create dataset
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
        
        logger.info(f"Loaded subset {subset_name} from {subset_file}")
        return dataset
    
    def preprocess(self, save: bool = True) -> Dict[str, Any]:
        """
        Preprocess dataset.
        
        Args:
            save: Whether to save preprocessed data
        
        Returns:
            Preprocessed data dictionary
        """
        logger.info(f"Preprocessing dataset: {self.dataset_name}")
        
        if not self.preprocessor:
            self.preprocessor = DataPreprocessor(self.config)
        
        processed_data = self.preprocessor.process_dataset(self, save)
        
        self.processed_data = processed_data
        return processed_data
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert dataset to pandas DataFrame.
        
        Returns:
            DataFrame of interactions
        """
        df = pd.DataFrame(self.interactions)
        return df
    
    def get_item_text(self, item_id: str) -> str:
        """
        Get text representation of item.
        
        Args:
            item_id: Item ID
        
        Returns:
            Text representation
        """
        item = self.items.get(item_id)
        if not item:
            return ""
        
        parts = []
        if item.get('title'):
            parts.append(item['title'])
        if item.get('description'):
            parts.append(item['description'])
        if item.get('category'):
            parts.append(item['category'])
        
        return " ".join(parts)
    
    def get_user_text(self, user_id: str) -> str:
        """
        Get text representation of user.
        
        Args:
            user_id: User ID
        
        Returns:
            Text representation
        """
        user = self.users.get(user_id)
        if not user:
            return ""
        
        preferences = user.get('preferences', {})
        parts = [f"User has {user.get('num_interactions', 0)} interactions"]
        
        if preferences.get('preferred_categories'):
            categories = preferences['preferred_categories']
            top_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_cats:
                parts.append(f"Prefers categories: {', '.join([cat for cat, _ in top_cats])}")
        
        if preferences.get('average_rating'):
            parts.append(f"Average rating: {preferences['average_rating']:.1f}")
        
        return " ".join(parts)


def create_amazon_datasets(config: Dict[str, Any], 
                          dataset_names: Optional[List[str]] = None,
                          download: bool = True,
                          force_download: bool = False,
                          process: bool = True) -> Dict[str, AmazonDataset]:
    """
    Create and process Amazon datasets.
    
    Args:
        config: Configuration dictionary
        dataset_names: List of dataset names (default: all)
        download: Whether to download missing datasets
        force_download: Force re-download
        process: Whether to preprocess datasets
    
    Returns:
        Dictionary of AmazonDataset instances
    """
    if dataset_names is None:
        dataset_names = ['CDs_and_Vinyl', 'Office_Products']
    
    manager = AmazonDatasetManager(config['data']['data_dir'], config)
    datasets = {}
    
    for dataset_name in dataset_names:
        logger.info(f"Creating dataset: {dataset_name}")
        
        # Download if needed
        if download and dataset_name not in manager.available_datasets:
            logger.info(f"Downloading dataset: {dataset_name}")
            if not manager.download_dataset(dataset_name, force_download):
                logger.error(f"Failed to download {dataset_name}")
                continue
        elif force_download:
            logger.info(f"Re-downloading dataset: {dataset_name}")
            manager.download_dataset(dataset_name, force=True)
        
        # Create dataset instance
        dataset = AmazonDataset(dataset_name, config)
        
        # Load data
        try:
            dataset.load_data()
        except FileNotFoundError as e:
            logger.warning(f"Dataset files not found for {dataset_name}: {e}")
            continue
        
        # Process if requested
        if process:
            dataset.preprocess()
        
        datasets[dataset_name] = dataset
        
        # Log statistics
        if dataset.statistics:
            logger.info(f"Dataset {dataset_name}: {dataset.statistics}")
    
    return datasets


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
        'dataset_version': '5-core'
    }
    
    # Create datasets
    datasets = create_amazon_datasets