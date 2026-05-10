"""
Amazon Data Loader Module
Handles loading and parsing of Amazon review datasets.
Supports multiple product categories and review metadata.
"""

import os
import json
import gzip
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator, Any
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AmazonDataLoader:
    """
    Loader for Amazon review datasets.
    
    Supports:
    - 5-core and full datasets
    - Multiple product categories
    - Review text, metadata, and ratings
    - Streaming for large files
    """
    
    # Amazon product categories
    CATEGORIES = [
        'CDs_and_Vinyl',
        'Office_Products',
        'Books',
        'Electronics',
        'Movies_and_TV',
        'Clothing_Shoes_and_Jewelry',
        'Home_and_Kitchen',
        'Sports_and_Outdoors',
        'Toys_and_Games',
        'Video_Games'
    ]
    
    def __init__(self, 
                 data_dir: str = './data/amazon',
                 category: str = 'CDs_and_Vinyl',
                 version: str = '5core'):
        """
        Initialize Amazon data loader.
        
        Args:
            data_dir: Root directory for Amazon datasets
            category: Product category to load
            version: '5core' or 'full'
        """
        self.data_dir = Path(data_dir)
        self.category = category
        self.version = version
        
        # File paths
        self.reviews_file = self.data_dir / f'{category}_{version}.json.gz'
        self.metadata_file = self.data_dir / f'meta_{category}.json.gz'
        
        # Cached data
        self._reviews_cache: Optional[pd.DataFrame] = None
        self._metadata_cache: Optional[Dict] = None
        self._user_map: Dict[str, int] = {}
        self._item_map: Dict[str, int] = {}
    
    def load_reviews(self, 
                     max_reviews: Optional[int] = None,
                     min_rating: float = 0.0,
                     max_rating: float = 5.0,
                     years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load review data with optional filtering.
        
        Args:
            max_reviews: Maximum number of reviews to load
            min_rating: Minimum rating filter
            max_rating: Maximum rating filter
            years: Filter by review years
        
        Returns:
            DataFrame with columns:
            - reviewerID: User identifier
            - asin: Item identifier
            - overall: Rating (1.0-5.0)
            - reviewText: Review text
            - summary: Review summary
            - unixReviewTime: Timestamp
            - verified: Whether purchase was verified
            - vote: Helpful votes
        """
        logger.info(f"Loading reviews from {self.reviews_file}")
        
        if self._reviews_cache is not None:
            df = self._reviews_cache.copy()
        else:
            df = self._parse_reviews_file()
            self._reviews_cache = df.copy()
        
        # Apply filters
        original_len = len(df)
        
        if min_rating > 0:
            df = df[df['overall'] >= min_rating]
        
        if max_rating < 5.0:
            df = df[df['overall'] <= max_rating]
        
        if years:
            df['year'] = pd.to_datetime(df['unixReviewTime'], unit='s').dt.year
            df = df[df['year'].isin(years)]
        
        if max_reviews:
            df = df.head(max_reviews)
        
        logger.info(f"Loaded {len(df)} reviews (filtered from {original_len})")
        
        return df
    
    def _parse_reviews_file(self) -> pd.DataFrame:
        """
        Parse the gzipped JSON reviews file.
        
        Returns:
            DataFrame with review data
        """
        if not self.reviews_file.exists():
            raise FileNotFoundError(f"Reviews file not found: {self.reviews_file}")
        
        records = []
        
        with gzip.open(self.reviews_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    records.append({
                        'reviewerID': record.get('reviewerID', ''),
                        'asin': record.get('asin', ''),
                        'overall': float(record.get('overall', 0)),
                        'reviewText': record.get('reviewText', ''),
                        'summary': record.get('summary', ''),
                        'unixReviewTime': int(record.get('unixReviewTime', 0)),
                        'verified': record.get('verified', False),
                        'vote': int(record.get('vote', 0) if record.get('vote') else 0),
                        'reviewerName': record.get('reviewerName', ''),
                        'style': record.get('style', {})
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping malformed record: {e}")
                    continue
        
        df = pd.DataFrame(records)
        
        # Sort by time
        if 'unixReviewTime' in df.columns:
            df = df.sort_values('unixReviewTime')
        
        return df
    
    def load_metadata(self, 
                      item_ids: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Load item metadata.
        
        Args:
            item_ids: Optional list of specific item IDs to load
        
        Returns:
            Dictionary mapping asin -> metadata dict with keys:
            - title: Item title
            - description: Item description
            - category: Categories list
            - brand: Brand name
            - price: Price if available
            - salesRank: Sales rank information
            - also_bought: Related items
            - also_viewed: Related items
        """
        logger.info(f"Loading metadata from {self.metadata_file}")
        
        if self._metadata_cache is not None:
            metadata = self._metadata_cache
        else:
            metadata = self._parse_metadata_file()
            self._metadata_cache = metadata
        
        if item_ids:
            metadata = {k: v for k, v in metadata.items() if k in item_ids}
        
        logger.info(f"Loaded metadata for {len(metadata)} items")
        
        return metadata
    
    def _parse_metadata_file(self) -> Dict[str, Dict]:
        """
        Parse the gzipped JSON metadata file.
        
        Returns:
            Dictionary of item metadata
        """
        if not self.metadata_file.exists():
            logger.warning(f"Metadata file not found: {self.metadata_file}")
            return {}
        
        metadata = {}
        
        with gzip.open(self.metadata_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    asin = record.get('asin', '')
                    
                    if not asin:
                        continue
                    
                    # Extract relevant fields
                    categories = self._extract_categories(record.get('categories', []))
                    
                    metadata[asin] = {
                        'asin': asin,
                        'title': record.get('title', ''),
                        'description': self._extract_description(record),
                        'categories': categories,
                        'main_category': categories[0] if categories else 'Unknown',
                        'brand': record.get('brand', ''),
                        'price': record.get('price', ''),
                        'salesRank': record.get('salesRank', {}),
                        'also_bought': record.get('also_bought', []),
                        'also_viewed': record.get('also_viewed', []),
                        'bought_together': record.get('bought_together', []),
                        'image_url': record.get('imUrl', ''),
                        'tech_details': record.get('tech1', ''),
                        'fit': record.get('fit', ''),
                        'date': record.get('date', '')
                    }
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping malformed metadata record: {e}")
                    continue
        
        return metadata
    
    def _extract_categories(self, categories: List) -> List[str]:
        """
        Extract flat category list from nested Amazon category structure.
        
        Args:
            categories: Nested category list
        
        Returns:
            Flat list of category strings
        """
        if not categories:
            return []
        
        flat_categories = []
        
        for category in categories:
            if isinstance(category, list):
                # Amazon stores categories as nested lists
                # e.g., [['Electronics', 'Accessories', 'Headphones']]
                for sub in category:
                    if isinstance(sub, list):
                        flat_categories.append(' > '.join(sub))
                    else:
                        flat_categories.append(str(sub))
            else:
                flat_categories.append(str(category))
        
        return flat_categories
    
    def _extract_description(self, record: Dict) -> str:
        """
        Extract description from various possible fields.
        
        Args:
            record: Raw metadata record
        
        Returns:
            Best available description string
        """
        # Try different description fields
        description = record.get('description', '')
        
        if not description:
            description = record.get('title', '')
        
        if not description:
            # Try to combine available info
            parts = []
            if record.get('feature'):
                parts.append('; '.join(record['feature'][:3]))
            if record.get('brand'):
                parts.append(f"Brand: {record['brand']}")
            description = '. '.join(parts)
        
        # Clean description
        if isinstance(description, list):
            description = ' '.join(description)
        elif not isinstance(description, str):
            description = str(description)
        
        return description
    
    def build_interaction_dataset(self,
                                   min_user_interactions: int = 5,
                                   min_item_interactions: int = 5,
                                   max_users: Optional[int] = None,
                                   max_items: Optional[int] = None) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Build a clean interaction dataset with user/item filtering.
        
        Args:
            min_user_interactions: Minimum interactions per user
            min_item_interactions: Minimum interactions per item
            max_users: Maximum number of users to include
            max_items: Maximum number of items to include
        
        Returns:
            Tuple of:
            - Filtered reviews DataFrame
            - User statistics dict
            - Item statistics dict
        """
        logger.info("Building interaction dataset...")
        
        # Load reviews
        reviews = self.load_reviews()
        
        # Filter users by interaction count
        user_counts = reviews['reviewerID'].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        
        if max_users:
            valid_users = valid_users[:max_users]
        
        reviews = reviews[reviews['reviewerID'].isin(valid_users)]
        
        # Filter items by interaction count
        item_counts = reviews['asin'].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        
        if max_items:
            valid_items = valid_items[:max_items]
        
        reviews = reviews[reviews['asin'].isin(valid_items)]
        
        # Re-filter users after item filtering
        user_counts = reviews['reviewerID'].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        reviews = reviews[reviews['reviewerID'].isin(valid_users)]
        
        # Create mappings
        self._user_map = {uid: i for i, uid in enumerate(valid_users)}
        self._item_map = {iid: i for i, iid in enumerate(valid_items)}
        
        # Compute statistics
        user_stats = {
            'num_users': len(valid_users),
            'avg_interactions': float(user_counts.mean()),
            'median_interactions': float(user_counts.median()),
            'min_interactions': int(user_counts.min()),
            'max_interactions': int(user_counts.max())
        }
        
        item_stats = {
            'num_items': len(valid_items),
            'avg_interactions': float(item_counts.mean()),
            'median_interactions': float(item_counts.median()),
            'min_interactions': int(item_counts.min()),
            'max_interactions': int(item_counts.max())
        }
        
        logger.info(f"Built dataset: {len(valid_users)} users, "
                   f"{len(valid_items)} items, {len(reviews)} interactions")
        
        return reviews, user_stats, item_stats
    
    def create_user_sequences(self,
                               reviews: pd.DataFrame,
                               min_sequence_length: int = 5,
                               max_sequence_length: Optional[int] = None) -> Dict[str, List[str]]:
        """
        Create chronological item sequences per user.
        
        Args:
            reviews: DataFrame with review data
            min_sequence_length: Minimum sequence length
            max_sequence_length: Maximum sequence length (truncate)
        
        Returns:
            Dictionary mapping user_id -> list of item_ids in chronological order
        """
        sequences = {}
        
        for user_id, group in reviews.groupby('reviewerID'):
            # Sort by time
            group = group.sort_values('unixReviewTime')
            
            # Get item sequence
            items = group['asin'].tolist()
            
            if len(items) >= min_sequence_length:
                if max_sequence_length:
                    items = items[:max_sequence_length]
                sequences[user_id] = items
        
        logger.info(f"Created {len(sequences)} user sequences")
        
        return sequences
    
    def create_train_test_split(self,
                                 sequences: Dict[str, List[str]],
                                 train_ratio: float = 0.8,
                                 val_ratio: float = 0.1,
                                 leave_last_out: bool = True) -> Dict[str, Dict[str, List[str]]]:
        """
        Split sequences into train/validation/test sets.
        
        Args:
            sequences: User-item sequences
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            leave_last_out: If True, use last item as test
        
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        splits = {'train': {}, 'val': {}, 'test': {}}
        
        for user_id, items in sequences.items():
            if leave_last_out:
                # Leave-one-out: last item for test, second-to-last for validation
                if len(items) >= 3:
                    splits['test'][user_id] = [items[-1]]
                    splits['val'][user_id] = [items[-2]]
                    splits['train'][user_id] = items[:-2]
                elif len(items) >= 2:
                    splits['test'][user_id] = [items[-1]]
                    splits['train'][user_id] = items[:-1]
                else:
                    splits['train'][user_id] = items
            else:
                # Ratio-based split
                n_train = max(1, int(len(items) * train_ratio))
                n_val = max(0, int(len(items) * val_ratio))
                
                splits['train'][user_id] = items[:n_train]
                splits['val'][user_id] = items[n_train:n_train + n_val] if n_val > 0 else []
                splits['test'][user_id] = items[n_train + n_val:] if n_train + n_val < len(items) else []
        
        # Log split sizes
        for split_name, split_data in splits.items():
            total_items = sum(len(v) for v in split_data.values())
            logger.info(f"{split_name}: {len(split_data)} users, {total_items} items")
        
        return splits
    
    def load_subset(self,
                    num_users: int = 100,
                    sparsity: str = 'dense',
                    random_seed: int = 42) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Load a specific subset of the dataset for experiments.
        
        Args:
            num_users: Number of users to sample
            sparsity: 'dense' or 'sparse'
            random_seed: Random seed for reproducibility
        
        Returns:
            Tuple of (reviews, user_stats, item_stats)
        """
        np.random.seed(random_seed)
        
        # Load full interactions
        reviews, user_stats, item_stats = self.build_interaction_dataset(
            min_user_interactions=5,
            min_item_interactions=3
        )
        
        # Get users
        users = list(reviews['reviewerID'].unique())
        
        if sparsity == 'dense':
            # Select users with most interactions
            user_counts = reviews.groupby('reviewerID').size()
            selected_users = user_counts.nlargest(num_users).index.tolist()
        elif sparsity == 'sparse':
            # Select users near minimum interactions
            user_counts = reviews.groupby('reviewerID').size()
            median_count = user_counts.median()
            eligible = user_counts[
                (user_counts >= 5) & (user_counts <= median_count)
            ]
            if len(eligible) >= num_users:
                selected_users = np.random.choice(eligible.index, num_users, replace=False).tolist()
            else:
                selected_users = eligible.index.tolist()[:num_users]
        else:
            # Random sample
            if len(users) >= num_users:
                selected_users = np.random.choice(users, num_users, replace=False).tolist()
            else:
                selected_users = users
        
        # Filter reviews
        subset_reviews = reviews[reviews['reviewerID'].isin(selected_users)]
        
        # Filter items with enough interactions
        item_counts = subset_reviews['asin'].value_counts()
        valid_items = item_counts[item_counts >= 2].index
        subset_reviews = subset_reviews[subset_reviews['asin'].isin(valid_items)]
        
        # Update stats
        subset_user_stats = {
            'num_users': len(selected_users),
            'num_items': len(valid_items),
            'num_interactions': len(subset_reviews),
            'sparsity': f"{len(subset_reviews) / (len(selected_users) * len(valid_items)):.4%}",
            'avg_user_interactions': len(subset_reviews) / len(selected_users)
        }
        
        logger.info(f"Loaded {sparsity} subset: {subset_user_stats}")
        
        return subset_reviews, subset_user_stats, self.load_metadata(
            item_ids=list(valid_items)
        )
    
    def get_item_text(self, asin: str) -> Dict[str, str]:
        """
        Get text features for a specific item.
        
        Args:
            asin: Item identifier
        
        Returns:
            Dictionary with title, description, category
        """
        metadata = self.load_metadata(item_ids=[asin])
        
        if asin in metadata:
            meta = metadata[asin]
            return {
                'title': meta.get('title', ''),
                'description': meta.get('description', ''),
                'category': meta.get('main_category', ''),
                'brand': meta.get('brand', '')
            }
        
        return {
            'title': f'Item {asin}',
            'description': '',
            'category': 'Unknown',
            'brand': ''
        }
    
    def get_user_profile(self, 
                          reviewer_id: str,
                          reviews: pd.DataFrame) -> Dict[str, Any]:
        """
        Build a user profile from their reviews.
        
        Args:
            reviewer_id: User identifier
            reviews: Full reviews DataFrame
        
        Returns:
            User profile dictionary
        """
        user_reviews = reviews[reviews['reviewerID'] == reviewer_id]
        
        if len(user_reviews) == 0:
            return {
                'reviewerID': reviewer_id,
                'num_reviews': 0,
                'avg_rating': 0.0,
                'categories': []
            }
        
        # Aggregate statistics
        categories = set()
        for asin in user_reviews['asin'].unique():
            item_text = self.get_item_text(asin)
            if item_text['category']:
                categories.add(item_text['category'])
        
        return {
            'reviewerID': reviewer_id,
            'num_reviews': len(user_reviews),
            'avg_rating': float(user_reviews['overall'].mean()),
            'rating_std': float(user_reviews['overall'].std()) if len(user_reviews) > 1 else 0.0,
            'verified_ratio': float(user_reviews['verified'].mean()),
            'categories': list(categories),
            'avg_review_length': float(user_reviews['reviewText'].str.len().mean()),
            'review_period_days': (
                (user_reviews['unixReviewTime'].max() - user_reviews['unixReviewTime'].min()) 
                / 86400 if len(user_reviews) > 1 else 0
            )
        }
    
    def get_dataset_statistics(self, reviews: pd.DataFrame) -> Dict:
        """
        Compute comprehensive dataset statistics.
        
        Args:
            reviews: Reviews DataFrame
        
        Returns:
            Statistics dictionary
        """
        stats = {}
        
        # Basic counts
        stats['num_reviews'] = len(reviews)
        stats['num_users'] = reviews['reviewerID'].nunique()
        stats['num_items'] = reviews['asin'].nunique()
        
        # Sparsity
        stats['sparsity'] = 1 - stats['num_reviews'] / (stats['num_users'] * stats['num_items'])
        
        # Ratings
        stats['avg_rating'] = float(reviews['overall'].mean())
        stats['rating_std'] = float(reviews['overall'].std())
        stats['rating_distribution'] = reviews['overall'].value_counts().to_dict()
        
        # User statistics
        user_counts = reviews.groupby('reviewerID').size()
        stats['avg_user_interactions'] = float(user_counts.mean())
        stats['median_user_interactions'] = float(user_counts.median())
        stats['max_user_interactions'] = int(user_counts.max())
        
        # Item statistics
        item_counts = reviews.groupby('asin').size()
        stats['avg_item_interactions'] = float(item_counts.mean())
        stats['median_item_interactions'] = float(item_counts.median())
        
        # Review text
        if 'reviewText' in reviews.columns:
            review_lengths = reviews['reviewText'].str.len()
            stats['avg_review_length'] = float(review_lengths.mean())
            stats['empty_reviews'] = int((review_lengths == 0).sum())
        
        # Temporal
        if 'unixReviewTime' in reviews.columns:
            timestamps = pd.to_datetime(reviews['unixReviewTime'], unit='s')
            stats['date_range'] = {
                'start': timestamps.min().strftime('%Y-%m-%d'),
                'end': timestamps.max().strftime('%Y-%m-%d')
            }
            stats['years_covered'] = timestamps.dt.year.nunique()
        
        return stats
    
    def export_to_json(self, 
                        reviews: pd.DataFrame,
                        output_path: str,
                        include_metadata: bool = True):
        """
        Export processed dataset to JSON format.
        
        Args:
            reviews: Reviews DataFrame
            output_path: Output file path
            include_metadata: Whether to include item metadata
        """
        output = {
            'reviews': reviews.to_dict(orient='records'),
            'statistics': self.get_dataset_statistics(reviews)
        }
        
        if include_metadata:
            item_ids = reviews['asin'].unique().tolist()
            output['metadata'] = {
                asin: self.get_item_text(asin) 
                for asin in item_ids
            }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Exported dataset to {output_path}")