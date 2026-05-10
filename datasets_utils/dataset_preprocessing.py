"""
Data Preprocessing Module
Handles text cleaning, feature extraction, and dataset preparation
for the Hybrid-GraphRAG framework.
"""

import re
import html
import string
import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import Counter, defaultdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses raw Amazon review data for agent-based recommendation.
    
    Handles:
    - Text cleaning and normalization
    - Feature extraction
    - Item description preparation
    - User preference extraction
    """
    
    def __init__(self,
                 max_description_length: int = 100,
                 min_word_count: int = 3,
                 remove_html: bool = True,
                 normalize_whitespace: bool = True,
                 lowercase: bool = True):
        """
        Initialize preprocessor.
        
        Args:
            max_description_length: Maximum tokens in cleaned description
            min_word_count: Minimum word count for text filtering
            remove_html: Whether to strip HTML tags
            normalize_whitespace: Whether to normalize whitespace
            lowercase: Whether to convert to lowercase
        """
        self.max_description_length = max_description_length
        self.min_word_count = min_word_count
        self.remove_html = remove_html
        self.normalize_whitespace = normalize_whitespace
        self.lowercase = lowercase
        
        # Statistics
        self.stats = defaultdict(int)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw input text
        
        Returns:
            Cleaned text string
        """
        if not text or not isinstance(text, str):
            return ''
        
        self.stats['total_texts_processed'] += 1
        
        # Remove HTML
        if self.remove_html:
            text = html.unescape(text)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\'\"()-]', ' ', text)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove very short texts
        words = text.split()
        if len(words) < self.min_word_count:
            self.stats['short_texts_filtered'] += 1
            return ''
        
        # Truncate to max length
        if len(words) > self.max_description_length:
            text = ' '.join(words[:self.max_description_length])
            self.stats['texts_truncated'] += 1
        
        return text
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean a batch of texts.
        
        Args:
            texts: List of raw text strings
        
        Returns:
            List of cleaned text strings
        """
        return [self.clean_text(t) for t in texts]
    
    def prepare_item_description(self, 
                                  item_data: Dict[str, Any],
                                  include_brand: bool = True,
                                  include_category: bool = True) -> str:
        """
        Prepare a formatted item description for agent memory.
        
        Args:
            item_data: Dictionary with item metadata
            include_brand: Whether to include brand
            include_category: Whether to include category
        
        Returns:
            Formatted description string
        """
        parts = []
        
        # Title (always included)
        title = self.clean_text(item_data.get('title', ''))
        if title:
            parts.append(title)
        
        # Category
        if include_category:
            category = item_data.get('main_category', item_data.get('category', ''))
            if category:
                parts.append(f"Category: {category}")
        
        # Brand
        if include_brand:
            brand = self.clean_text(item_data.get('brand', ''))
            if brand:
                parts.append(f"Brand: {brand}")
        
        # Description
        description = self.clean_text(item_data.get('description', ''))
        if description:
            parts.append(description)
        
        result = '. '.join(parts)
        
        # Truncate if needed
        words = result.split()
        if len(words) > self.max_description_length:
            result = ' '.join(words[:self.max_description_length])
        
        return result
    
    def extract_keywords(self, 
                          text: str, 
                          top_k: int = 10,
                          min_length: int = 3) -> List[str]:
        """
        Extract keywords from text using TF-IDF.
        
        Args:
            text: Input text
            top_k: Number of top keywords
            min_length: Minimum keyword length
        
        Returns:
            List of keywords
        """
        if not text:
            return []
        
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        words = text.split()
        
        # Remove short words and stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                      'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were',
                      'this', 'that', 'it', 'its', 'i', 'my', 'me', 'you', 'your'}
        
        words = [w for w in words if len(w) >= min_length and w not in stop_words]
        
        # Count frequencies
        word_counts = Counter(words)
        
        # Return top-k
        return [word for word, _ in word_counts.most_common(top_k)]
    
    def extract_user_preferences(self,
                                  user_reviews: pd.DataFrame,
                                  metadata: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Extract user preference profile from review history.
        
        Args:
            user_reviews: DataFrame of user's reviews
            metadata: Item metadata dictionary
        
        Returns:
            User preference dictionary
        """
        preferences = {
            'preferred_categories': Counter(),
            'preferred_brands': Counter(),
            'avg_rating_given': 0.0,
            'rating_behavior': 'neutral',
            'keywords_positive': [],
            'keywords_negative': [],
            'review_style': {
                'avg_length': 0,
                'uses_summary': False,
                'verified_ratio': 0.0
            }
        }
        
        if len(user_reviews) == 0:
            return preferences
        
        # Category and brand preferences from positive reviews
        positive_reviews = user_reviews[user_reviews['overall'] >= 4.0]
        negative_reviews = user_reviews[user_reviews['overall'] <= 2.0]
        
        # Positive preferences
        for _, review in positive_reviews.iterrows():
            asin = review['asin']
            if asin in metadata:
                meta = metadata[asin]
                if meta.get('main_category'):
                    preferences['preferred_categories'][meta['main_category']] += 1
                if meta.get('brand'):
                    preferences['preferred_brands'][meta['brand']] += 1
        
        # Extract keywords from positive reviews
        positive_texts = positive_reviews['reviewText'].fillna('').tolist()
        if positive_texts:
            combined = ' '.join(positive_texts)
            preferences['keywords_positive'] = self.extract_keywords(combined, top_k=15)
        
        # Extract keywords from negative reviews
        negative_texts = negative_reviews['reviewText'].fillna('').tolist()
        if negative_texts:
            combined = ' '.join(negative_texts)
            preferences['keywords_negative'] = self.extract_keywords(combined, top_k=15)
        
        # Rating behavior
        preferences['avg_rating_given'] = float(user_reviews['overall'].mean())
        
        if preferences['avg_rating_given'] > 4.0:
            preferences['rating_behavior'] = 'generous'
        elif preferences['avg_rating_given'] < 3.0:
            preferences['rating_behavior'] = 'critical'
        else:
            preferences['rating_behavior'] = 'neutral'
        
        # Review style
        preferences['review_style']['avg_length'] = float(
            user_reviews['reviewText'].fillna('').str.len().mean()
        )
        preferences['review_style']['uses_summary'] = bool(
            user_reviews['summary'].fillna('').str.len().sum() > 0
        )
        preferences['review_style']['verified_ratio'] = float(
            user_reviews['verified'].mean()
        )
        
        # Convert Counters to sorted lists
        preferences['preferred_categories'] = [
            cat for cat, _ in preferences['preferred_categories'].most_common(10)
        ]
        preferences['preferred_brands'] = [
            brand for brand, _ in preferences['preferred_brands'].most_common(10)
        ]
        
        return preferences
    
    def prepare_training_data(self,
                               reviews: pd.DataFrame,
                               sequences: Dict[str, List[str]],
                               metadata: Dict[str, Dict]) -> List[Dict]:
        """
        Prepare training samples for Phase 1 bootstrap.
        
        Args:
            reviews: Reviews DataFrame
            sequences: User-item interaction sequences
            metadata: Item metadata
        
        Returns:
            List of training sample dictionaries
        """
        samples = []
        
        for user_id, items in sequences.items():
            user_reviews = reviews[reviews['reviewerID'] == user_id]
            user_prefs = self.extract_user_preferences(user_reviews, metadata)
            
            for i, item_id in enumerate(items[:-1]):  # All but last for training
                if item_id not in metadata:
                    continue
                
                # Positive item
                pos_item = metadata[item_id]
                
                # Get next item as context
                next_item_id = items[i + 1] if i + 1 < len(items) else None
                
                sample = {
                    'user_id': user_id,
                    'item_id': item_id,
                    'user_preferences': user_prefs,
                    'item_data': {
                        'title': pos_item.get('title', ''),
                        'category': pos_item.get('main_category', ''),
                        'description': self.prepare_item_description(pos_item),
                        'brand': pos_item.get('brand', '')
                    },
                    'context': {
                        'next_item_id': next_item_id,
                        'position_in_sequence': i,
                        'sequence_length': len(items)
                    },
                    'rating': float(
                        user_reviews[user_reviews['asin'] == item_id]['overall'].iloc[0]
                        if len(user_reviews[user_reviews['asin'] == item_id]) > 0
                        else 0.0
                    )
                }
                
                samples.append(sample)
        
        logger.info(f"Prepared {len(samples)} training samples")
        
        return samples
    
    def get_preprocessing_stats(self) -> Dict:
        """Get preprocessing statistics."""
        return dict(self.stats)
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = defaultdict(int)


class DatasetSplitter:
    """
    Handles dataset splitting strategies for recommendation experiments.
    
    Supports:
    - Leave-one-out split
    - Temporal split
    - Random split
    - K-fold cross-validation
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize splitter.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def leave_one_out_split(self,
                             sequences: Dict[str, List[str]],
                             val_ratio: float = 0.0) -> Dict[str, Dict[str, List[str]]]:
        """
        Leave-one-out split: last item for test.
        
        Args:
            sequences: User-item sequences
            val_ratio: Ratio for validation (from remaining)
        
        Returns:
            Dictionary with train, val, test splits
        """
        splits = {'train': {}, 'val': {}, 'test': {}}
        
        for user_id, items in sequences.items():
            if len(items) < 2:
                splits['train'][user_id] = items
                continue
            
            # Test: last item
            splits['test'][user_id] = [items[-1]]
            
            # Validation: second-to-last (if requested)
            if val_ratio > 0 and len(items) >= 3:
                splits['val'][user_id] = [items[-2]]
                splits['train'][user_id] = items[:-2]
            else:
                splits['train'][user_id] = items[:-1]
        
        logger.info(f"Leave-one-out split: "
                   f"train={len(splits['train'])}, "
                   f"val={len(splits['val'])}, "
                   f"test={len(splits['test'])}")
        
        return splits
    
    def temporal_split(self,
                        sequences: Dict[str, List[str]],
                        timestamps: Dict[str, List[int]],
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15) -> Dict[str, Dict[str, List[str]]]:
        """
        Temporal split based on timestamps.
        
        Args:
            sequences: User-item sequences
            timestamps: User-item timestamps
            train_ratio: Training ratio
            val_ratio: Validation ratio
        
        Returns:
            Dictionary with train, val, test splits
        """
        splits = {'train': {}, 'val': {}, 'test': {}}
        
        for user_id, items in sequences.items():
            if user_id not in timestamps or len(items) < 3:
                splits['train'][user_id] = items
                continue
            
            # Get timestamps for this user
            user_times = timestamps[user_id]
            
            # Sort by time
            sorted_pairs = sorted(zip(user_times, items), key=lambda x: x[0])
            sorted_items = [item for _, item in sorted_pairs]
            
            # Split by ratio
            n = len(sorted_items)
            n_train = max(1, int(n * train_ratio))
            n_val = max(0, int(n * val_ratio))
            
            splits['train'][user_id] = sorted_items[:n_train]
            splits['val'][user_id] = sorted_items[n_train:n_train + n_val]
            splits['test'][user_id] = sorted_items[n_train + n_val:]
        
        logger.info(f"Temporal split: "
                   f"train={len(splits['train'])}, "
                   f"val={len(splits['val'])}, "
                   f"test={len(splits['test'])}")
        
        return splits
    
    def random_split(self,
                      sequences: Dict[str, List[str]],
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15) -> Dict[str, Dict[str, List[str]]]:
        """
        Random split of sequences.
        
        Args:
            sequences: User-item sequences
            train_ratio: Training ratio
            val_ratio: Validation ratio
        
        Returns:
            Dictionary with train, val, test splits
        """
        splits = {'train': {}, 'val': {}, 'test': {}}
        
        for user_id, items in sequences.items():
            if len(items) < 3:
                splits['train'][user_id] = items
                continue
            
            # Shuffle items
            items_shuffled = items.copy()
            np.random.shuffle(items_shuffled)
            
            # Split
            n = len(items_shuffled)
            n_train = max(1, int(n * train_ratio))
            n_val = max(0, int(n * val_ratio))
            
            splits['train'][user_id] = items_shuffled[:n_train]
            splits['val'][user_id] = items_shuffled[n_train:n_train + n_val]
            splits['test'][user_id] = items_shuffled[n_train + n_val:]
        
        logger.info(f"Random split: "
                   f"train={len(splits['train'])}, "
                   f"val={len(splits['val'])}, "
                   f"test={len(splits['test'])}")
        
        return splits
    
    def k_fold_split(self,
                      sequences: Dict[str, List[str]],
                      k: int = 5) -> List[Dict[str, Dict[str, List[str]]]]:
        """
        K-fold cross-validation split.
        
        Args:
            sequences: User-item sequences
            k: Number of folds
        
        Returns:
            List of k split dictionaries
        """
        folds = []
        
        for fold_idx in range(k):
            fold_splits = {'train': {}, 'val': {}, 'test': {}}
            
            for user_id, items in sequences.items():
                if len(items) < k:
                    fold_splits['train'][user_id] = items
                    continue
                
                # Determine fold boundaries
                fold_size = len(items) // k
                test_start = fold_idx * fold_size
                test_end = min((fold_idx + 1) * fold_size, len(items))
                
                # Test fold
                fold_splits['test'][user_id] = items[test_start:test_end]
                
                # Training: all other items
                fold_splits['train'][user_id] = (
                    items[:test_start] + items[test_end:]
                )
            
            folds.append(fold_splits)
        
        logger.info(f"Created {k}-fold split")
        
        return folds
    
    def stratified_split(self,
                          sequences: Dict[str, List[str]],
                          user_groups: Dict[str, str],
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.15) -> Dict[str, Dict[str, List[str]]]:
        """
        Stratified split based on user groups.
        Ensures proportional representation across splits.
        
        Args:
            sequences: User-item sequences
            user_groups: Mapping user_id -> group label
            train_ratio: Training ratio
            val_ratio: Validation ratio
        
        Returns:
            Dictionary with train, val, test splits
        """
        splits = {'train': {}, 'val': {}, 'test': {}}
        
        # Group users
        groups = defaultdict(list)
        for user_id, group in user_groups.items():
            groups[group].append(user_id)
        
        # Split each group proportionally
        for group, users in groups.items():
            np.random.shuffle(users)
            
            n = len(users)
            n_train = max(1, int(n * train_ratio))
            n_val = max(0, int(n * val_ratio))
            
            train_users = users[:n_train]
            val_users = users[n_train:n_train + n_val]
            test_users = users[n_train + n_val:]
            
            for user_id in train_users:
                splits['train'][user_id] = sequences.get(user_id, [])
            for user_id in val_users:
                splits['val'][user_id] = sequences.get(user_id, [])
            for user_id in test_users:
                splits['test'][user_id] = sequences.get(user_id, [])
        
        logger.info(f"Stratified split: "
                   f"train={len(splits['train'])}, "
                   f"val={len(splits['val'])}, "
                   f"test={len(splits['test'])}")
        
        return splits
    
    def get_split_statistics(self, 
                              splits: Dict[str, Dict[str, List[str]]]) -> Dict:
        """
        Compute statistics for dataset splits.
        
        Args:
            splits: Dictionary with train/val/test
        
        Returns:
            Statistics dictionary
        """
        stats = {}
        
        for split_name, split_data in splits.items():
            num_users = len(split_data)
            num_items = sum(len(items) for items in split_data.values())
            
            # Unique items
            all_items = set()
            for items in split_data.values():
                all_items.update(items)
            
            stats[split_name] = {
                'num_users': num_users,
                'num_items': num_items,
                'num_unique_items': len(all_items),
                'avg_items_per_user': num_items / max(1, num_users),
                'empty_users': sum(1 for items in split_data.values() if len(items) == 0)
            }
        
        return stats
    
    def validate_splits(self, 
                         splits: Dict[str, Dict[str, List[str]]]) -> bool:
        """
        Validate that splits are correct.
        
        Args:
            splits: Dictionary with train/val/test
        
        Returns:
            True if splits are valid
        """
        # Check for data leakage between splits
        for split_a_name, split_a in splits.items():
            for split_b_name, split_b in splits.items():
                if split_a_name >= split_b_name:
                    continue
                
                # Check user overlap
                common_users = set(split_a.keys()) & set(split_b.keys())
                if common_users:
                    # Users can appear in multiple splits (different items)
                    # Check item overlap for common users
                    for user_id in common_users:
                        items_a = set(split_a.get(user_id, []))
                        items_b = set(split_b.get(user_id, []))
                        
                        if items_a & items_b:
                            logger.error(f"Data leakage detected: "
                                       f"user {user_id} has overlapping items "
                                       f"between {split_a_name} and {split_b_name}")
                            return False
        
        logger.info("Splits validated successfully")
        return True