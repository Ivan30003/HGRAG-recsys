"""
scripts/preprocess_data.py

Comprehensive data preprocessing script for H-GRAGrecsys with support for:
- Amazon dataset preprocessing
- Text feature extraction and summarization
- User profile creation
- Vocabulary building
- Data sampling (dense/sparse subsets)
- Train/validation/test splitting
- Feature normalization
- Data statistics generation
- Multiple dataset support
- Parallel processing
"""

import os
import sys
import json
import yaml
import argparse
import pickle
import gzip
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple, Set
from datetime import datetime
from collections import defaultdict, Counter
import traceback
import multiprocessing as mp
from functools import partial
import re
import hashlib

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.logger import get_logger
from utils.config_loader import ConfigLoader, load_config
from utils.seed_manager import create_seed_manager
from utils.timer import Timer, global_timer

# Import data modules
from data.amazon_dataset import AmazonDataset
from data.data_preprocessor import DataPreprocessor
from data.utils.text_processor import TextProcessor
from data.utils.sampling import Sampler

# Try to import optional libraries
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class DataPreprocessingPipeline:
    """
    Comprehensive data preprocessing pipeline for H-GRAGrecsys.
    
    Features:
    - Multi-dataset support (Amazon, MovieLens, etc.)
    - Text feature extraction and summarization
    - User and item profile creation
    - Vocabulary building
    - Data sampling strategies
    - Train/val/test splitting
    - Feature normalization
    - Statistics generation
    - Caching and checkpointing
    """
    
    # Supported datasets
    SUPPORTED_DATASETS = [
        'Amazon_Books',
        'Amazon_Electronics',
        'Amazon_Clothing',
        'Amazon_Toys',
        'MovieLens_1M',
        'MovieLens_100K',
        'Yelp',
        'LastFM'
    ]
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        dataset_name: Optional[str] = None,
        data_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        seed: Optional[int] = None,
        use_cache: bool = True,
        parallel: bool = False,
        num_workers: int = 4,
        logger: Optional['Logger'] = None,
        verbose: bool = True
    ):
        """
        Initialize the DataPreprocessingPipeline.
        
        Args:
            config_path (str, Path, optional): Path to configuration file
            dataset_name (str, optional): Name of the dataset to preprocess
            data_dir (str, Path, optional): Directory containing raw data
            output_dir (str, Path, optional): Directory to save processed data
            seed (int, optional): Random seed for reproducibility
            use_cache (bool): Whether to use cached preprocessing
            parallel (bool): Whether to use parallel processing
            num_workers (int): Number of parallel workers
            logger (Logger, optional): Logger instance
            verbose (bool): Whether to enable verbose output
        
        Example:
            pipeline = DataPreprocessingPipeline(
                config_path='config/default_config.yaml',
                dataset_name='Amazon_Books',
                output_dir='data/processed'
            )
            pipeline.run()
        """
        # Setup paths
        self.config_path = Path(config_path) if config_path else None
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir) if data_dir else Path("data/raw")
        self.output_dir = Path(output_dir) if output_dir else Path("data/processed")
        self.cache_dir = self.output_dir / "cache"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        if logger is None:
            self.logger = get_logger(
                log_dir=self.output_dir / "logs",
                name="data_preprocessing",
                verbose=verbose
            )
        else:
            self.logger = logger
        
        # Load configuration
        if self.config_path and self.config_path.exists():
            self.config_loader = ConfigLoader(
                config_path=self.config_path,
                logger=self.logger
            )
            self.config = self.config_loader.config
        else:
            self.config_loader = ConfigLoader(load_defaults=True)
            self.config = self.config_loader.config
        
        # Get dataset config
        self.dataset_config = self.config.get('data', {})
        if self.dataset_name:
            self.dataset_config['dataset_name'] = self.dataset_name
        
        # Setup seed manager
        self.seed = seed or self.config.get('seed', 42)
        self.seed_manager = create_seed_manager(seed=self.seed)
        self.seed_manager.set_all_seeds()
        
        # Setup timer
        self.timer = Timer(
            name="data_preprocessing",
            logger=self.logger,
            track_memory=True,
            save_report=True,
            report_dir=self.output_dir / "timing"
        )
        
        # Processing parameters
        self.use_cache = use_cache
        self.parallel = parallel
        self.num_workers = num_workers
        
        # Initialize components
        self.text_processor = TextProcessor(self.config)
        self.preprocessor = DataPreprocessor(self.config)
        self.sampler = Sampler(self.config)
        
        # Statistics
        self.stats = {}
        
        self.logger.log_info(f"DataPreprocessingPipeline initialized")
        self.logger.log_info(f"Dataset: {self.dataset_name or 'Not specified'}")
        self.logger.log_info(f"Output directory: {self.output_dir}")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete preprocessing pipeline.
        
        Returns:
            Dict[str, Any]: Preprocessing results and statistics
        
        Example:
            results = pipeline.run()
            print(f"Processed {results['num_users']} users and {results['num_items']} items")
        """
        self.logger.log_info("=" * 80)
        self.logger.log_info("Starting Data Preprocessing Pipeline")
        self.logger.log_info("=" * 80)
        
        with self.timer.measure("preprocessing"):
            # Step 1: Load raw data
            raw_data = self._load_raw_data()
            
            # Step 2: Preprocess text
            processed_text = self._preprocess_text(raw_data)
            
            # Step 3: Build vocabulary
            vocabulary = self._build_vocabulary(processed_text)
            
            # Step 4: Create user profiles
            user_profiles = self._create_user_profiles(raw_data)
            
            # Step 5: Create item features
            item_features = self._create_item_features(raw_data, processed_text)
            
            # Step 6: Split data
            splits = self._split_data(raw_data)
            
            # Step 7: Apply sampling (if configured)
            sampled_data = self._apply_sampling(splits)
            
            # Step 8: Normalize features
            normalized_features = self._normalize_features(item_features)
            
            # Step 9: Generate statistics
            self.stats = self._generate_statistics(raw_data, sampled_data)
            
            # Step 10: Save processed data
            self._save_processed_data(
                raw_data, processed_text, vocabulary,
                user_profiles, normalized_features,
                sampled_data, self.stats
            )
        
        self.logger.log_info("=" * 80)
        self.logger.log_info("Data Preprocessing Pipeline Completed")
        self.logger.log_info("=" * 80)
        
        return {
            'stats': self.stats,
            'output_dir': str(self.output_dir),
            'cache_dir': str(self.cache_dir)
        }
    
    def _load_raw_data(self) -> Dict[str, Any]:
        """
        Load raw dataset.
        
        Returns:
            Dict[str, Any]: Raw data dictionary
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("LOADING RAW DATA")
        self.logger.log_info("-" * 50)
        
        # Check cache
        cache_path = self.cache_dir / "raw_data.pkl"
        if self.use_cache and cache_path.exists():
            self.logger.log_info(f"Loading raw data from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        if not self.dataset_name:
            self.logger.log_error("Dataset name not specified")
            raise ValueError("Dataset name must be specified")
        
        if self.dataset_name not in self.SUPPORTED_DATASETS:
            self.logger.log_warning(f"Dataset {self.dataset_name} not in supported list")
        
        with self.timer.measure("load_data"):
            try:
                # Initialize dataset
                dataset = AmazonDataset(self.dataset_name, self.config)
                
                # Load data
                raw_data = {
                    'dataset_name': self.dataset_name,
                    'reviews': dataset._load_reviews(),
                    'item_metadata': dataset._load_item_metadata(),
                    'interactions': dataset.get_interactions(),
                    'user_items': dataset.get_user_items(),
                    'item_features': dataset.get_item_features()
                }
                
                self.logger.log_info(f"Loaded {len(raw_data['reviews'])} reviews")
                self.logger.log_info(f"Loaded {len(raw_data['item_metadata'])} items")
                self.logger.log_info(f"Found {len(raw_data['user_items'])} users")
                
                # Cache raw data
                if self.use_cache:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(raw_data, f)
                    self.logger.log_info(f"Cached raw data to: {cache_path}")
                
                return raw_data
                
            except Exception as e:
                self.logger.log_error(f"Failed to load data: {e}")
                raise
    
    def _preprocess_text(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess text data.
        
        Args:
            raw_data (Dict[str, Any]): Raw data dictionary
            
        Returns:
            Dict[str, Any]: Preprocessed text data
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("PREPROCESSING TEXT")
        self.logger.log_info("-" * 50)
        
        # Check cache
        cache_path = self.cache_dir / "processed_text.pkl"
        if self.use_cache and cache_path.exists():
            self.logger.log_info(f"Loading processed text from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        with self.timer.measure("preprocess_text"):
            processed = {}
            
            # Process reviews
            reviews = raw_data.get('reviews', [])
            if reviews:
                self.logger.log_info(f"Processing {len(reviews)} reviews...")
                
                # Process in parallel if enabled
                if self.parallel and len(reviews) > 1000:
                    with mp.Pool(processes=self.num_workers) as pool:
                        processed_reviews = pool.map(
                            self._process_single_text,
                            [(review.get('text', ''), review.get('summary', '')) 
                             for review in reviews]
                        )
                else:
                    processed_reviews = []
                    for review in tqdm(reviews, desc="Processing reviews", 
                                      disable=not TQDM_AVAILABLE):
                        processed_reviews.append(
                            self._process_single_text(
                                (review.get('text', ''), review.get('summary', ''))
                            )
                        )
                
                processed['reviews'] = processed_reviews
                self.logger.log_info(f"Processed {len(processed_reviews)} reviews")
            
            # Process item metadata
            metadata = raw_data.get('item_metadata', {})
            if metadata:
                self.logger.log_info(f"Processing {len(metadata)} items...")
                
                processed_metadata = {}
                for item_id, meta in tqdm(metadata.items(), desc="Processing metadata",
                                         disable=not TQDM_AVAILABLE):
                    text = ' '.join([
                        meta.get('title', ''),
                        meta.get('description', ''),
                        ' '.join(meta.get('categories', []))
                    ])
                    processed_metadata[item_id] = self._process_single_text((text, ''))
                
                processed['metadata'] = processed_metadata
                self.logger.log_info(f"Processed {len(processed_metadata)} items")
            
            # Cache processed text
            if self.use_cache:
                with open(cache_path, 'wb') as f:
                    pickle.dump(processed, f)
                self.logger.log_info(f"Cached processed text to: {cache_path}")
            
            return processed
    
    def _process_single_text(self, text_tuple: Tuple[str, str]) -> Dict[str, Any]:
        """
        Process a single text item.
        
        Args:
            text_tuple (Tuple[str, str]): (text, summary) tuple
            
        Returns:
            Dict[str, Any]: Processed text
        """
        text, summary = text_tuple
        
        # Clean text
        text = self._clean_text(text)
        summary = self._clean_text(summary)
        
        # Summarize if too long
        if len(text) > self.dataset_config.get('max_text_length', 512):
            text = self.text_processor.summarize_text(
                text, self.dataset_config.get('max_text_length', 512)
            )
        
        return {
            'text': text,
            'summary': summary,
            'tokens': self.text_processor.tokenize(text),
            'length': len(text)
        }
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text data.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters (keep alphanumeric and basic punctuation)
        text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
        
        # Trim
        text = text.strip()
        
        return text
    
    def _build_vocabulary(self, processed_text: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build vocabulary from processed text.
        
        Args:
            processed_text (Dict[str, Any]): Processed text data
            
        Returns:
            Dict[str, Any]: Vocabulary
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("BUILDING VOCABULARY")
        self.logger.log_info("-" * 50)
        
        # Check cache
        cache_path = self.cache_dir / "vocabulary.pkl"
        if self.use_cache and cache_path.exists():
            self.logger.log_info(f"Loading vocabulary from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        with self.timer.measure("build_vocabulary"):
            # Collect all tokens
            all_tokens = []
            
            # From reviews
            if 'reviews' in processed_text:
                for review in processed_text['reviews']:
                    all_tokens.extend(review.get('tokens', []))
            
            # From metadata
            if 'metadata' in processed_text:
                for meta in processed_text['metadata'].values():
                    all_tokens.extend(meta.get('tokens', []))
            
            self.logger.log_info(f"Collected {len(all_tokens)} tokens")
            
            # Count tokens
            token_counts = Counter(all_tokens)
            self.logger.log_info(f"Found {len(token_counts)} unique tokens")
            
            # Build vocabulary
            min_freq = self.dataset_config.get('min_token_freq', 5)
            vocab_size = self.dataset_config.get('vocab_size', 50000)
            
            # Filter by frequency
            filtered_tokens = {
                token: count for token, count in token_counts.items()
                if count >= min_freq
            }
            self.logger.log_info(f"Filtered to {len(filtered_tokens)} tokens (min_freq={min_freq})")
            
            # Sort by frequency and limit size
            sorted_tokens = sorted(
                filtered_tokens.items(),
                key=lambda x: x[1],
                reverse=True
            )[:vocab_size]
            
            # Create vocabulary
            vocab = {
                '<PAD>': 0,
                '<UNK>': 1,
                '<BOS>': 2,
                '<EOS>': 3
            }
            
            for i, (token, count) in enumerate(sorted_tokens, start=len(vocab)):
                vocab[token] = i
            
            self.logger.log_info(f"Final vocabulary size: {len(vocab)}")
            
            # Create reverse mapping
            reverse_vocab = {idx: token for token, idx in vocab.items()}
            
            vocabulary = {
                'vocab': vocab,
                'reverse_vocab': reverse_vocab,
                'size': len(vocab),
                'stats': {
                    'total_tokens': len(all_tokens),
                    'unique_tokens': len(token_counts),
                    'filtered_tokens': len(filtered_tokens),
                    'min_freq': min_freq,
                    'max_size': vocab_size
                }
            }
            
            # Cache vocabulary
            if self.use_cache:
                with open(cache_path, 'wb') as f:
                    pickle.dump(vocabulary, f)
                self.logger.log_info(f"Cached vocabulary to: {cache_path}")
            
            return vocabulary
    
    def _create_user_profiles(self, raw_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Create user profiles from interactions.
        
        Args:
            raw_data (Dict[str, Any]): Raw data dictionary
            
        Returns:
            Dict[str, Dict[str, Any]]: User profiles
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("CREATING USER PROFILES")
        self.logger.log_info("-" * 50)
        
        # Check cache
        cache_path = self.cache_dir / "user_profiles.pkl"
        if self.use_cache and cache_path.exists():
            self.logger.log_info(f"Loading user profiles from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        with self.timer.measure("create_user_profiles"):
            user_profiles = {}
            
            user_items = raw_data.get('user_items', {})
            item_features = raw_data.get('item_features', {})
            
            self.logger.log_info(f"Creating profiles for {len(user_items)} users...")
            
            for user_id, items in tqdm(user_items.items(), desc="Creating user profiles",
                                      disable=not TQDM_AVAILABLE):
                # Collect user data
                user_profile = {
                    'user_id': user_id,
                    'items': items,
                    'num_interactions': len(items),
                    'item_features': [item_features.get(item, {}) for item in items],
                    'preference_vector': self._compute_preference_vector(items, item_features)
                }
                
                user_profiles[user_id] = user_profile
            
            self.logger.log_info(f"Created {len(user_profiles)} user profiles")
            
            # Cache user profiles
            if self.use_cache:
                with open(cache_path, 'wb') as f:
                    pickle.dump(user_profiles, f)
                self.logger.log_info(f"Cached user profiles to: {cache_path}")
            
            return user_profiles
    
    def _compute_preference_vector(
        self,
        items: List[str],
        item_features: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute user preference vector from items.
        
        Args:
            items (List[str]): Item IDs
            item_features (Dict[str, Dict[str, Any]]): Item features
            
        Returns:
            Dict[str, float]: Preference vector
        """
        # Aggregate item features
        preference = defaultdict(float)
        
        for item_id in items:
            features = item_features.get(item_id, {})
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    preference[key] += value
                elif isinstance(value, str):
                    # For categorical features, count occurrences
                    key = f"{key}_{value}"
                    preference[key] += 1
        
        # Normalize
        total = sum(preference.values())
        if total > 0:
            preference = {k: v / total for k, v in preference.items()}
        
        return dict(preference)
    
    def _create_item_features(
        self,
        raw_data: Dict[str, Any],
        processed_text: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create item features.
        
        Args:
            raw_data (Dict[str, Any]): Raw data dictionary
            processed_text (Dict[str, Any]): Processed text data
            
        Returns:
            Dict[str, Dict[str, Any]]: Item features
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("CREATING ITEM FEATURES")
        self.logger.log_info("-" * 50)
        
        # Check cache
        cache_path = self.cache_dir / "item_features.pkl"
        if self.use_cache and cache_path.exists():
            self.logger.log_info(f"Loading item features from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        with self.timer.measure("create_item_features"):
            item_features = {}
            
            item_metadata = raw_data.get('item_metadata', {})
            processed_metadata = processed_text.get('metadata', {})
            
            self.logger.log_info(f"Creating features for {len(item_metadata)} items...")
            
            for item_id, metadata in tqdm(item_metadata.items(), desc="Creating item features",
                                         disable=not TQDM_AVAILABLE):
                # Get processed text
                text_data = processed_metadata.get(item_id, {})
                
                # Extract features
                features = {
                    'item_id': item_id,
                    'title': metadata.get('title', ''),
                    'description': metadata.get('description', ''),
                    'categories': metadata.get('categories', []),
                    'brand': metadata.get('brand', ''),
                    'price': metadata.get('price', 0.0),
                    'rating': metadata.get('rating', 0.0),
                    'num_reviews': metadata.get('num_reviews', 0),
                    'text_processed': text_data.get('text', ''),
                    'tokens': text_data.get('tokens', []),
                    'text_length': text_data.get('length', 0)
                }
                
                # Compute popularity score
                features['popularity'] = features['num_reviews'] / (1 + max(1, features['num_reviews']))
                
                # Compute text embedding if available
                if TRANSFORMERS_AVAILABLE and len(features['text_processed']) > 0:
                    try:
                        features['embedding'] = self._get_text_embedding(features['text_processed'])
                    except Exception as e:
                        self.logger.log_warning(f"Failed to compute embedding for {item_id}: {e}")
                
                item_features[item_id] = features
            
            self.logger.log_info(f"Created features for {len(item_features)} items")
            
            # Cache item features
            if self.use_cache:
                with open(cache_path, 'wb') as f:
                    pickle.dump(item_features, f)
                self.logger.log_info(f"Cached item features to: {cache_path}")
            
            return item_features
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Get text embedding using transformer model.
        
        Args:
            text (str): Text to embed
            
        Returns:
            List[float]: Text embedding
        """
        # Simple placeholder - in real implementation, use proper embedding
        # This is a simplified version for demonstration
        try:
            # Use a simple token-based hash for demonstration
            # In production, use a proper embedding model like Sentence-BERT
            import hashlib
            hash_bytes = hashlib.md5(text.encode()).digest()
            embedding = [float(b) / 255.0 for b in hash_bytes[:128]]
            return embedding
        except:
            return [0.0] * 128
    
    def _split_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Split data into train/val/test sets.
        
        Args:
            raw_data (Dict[str, Any]): Raw data dictionary
            
        Returns:
            Dict[str, Any]: Split data
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("SPLITTING DATA")
        self.logger.log_info("-" * 50)
        
        # Check cache
        cache_path = self.cache_dir / "splits.pkl"
        if self.use_cache and cache_path.exists():
            self.logger.log_info(f"Loading splits from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        with self.timer.measure("split_data"):
            interactions = raw_data.get('interactions', [])
            
            # Get split ratios
            train_ratio = self.dataset_config.get('train_ratio', 0.7)
            val_ratio = self.dataset_config.get('validation_ratio', 0.1)
            test_ratio = self.dataset_config.get('test_ratio', 0.2)
            
            # Normalize ratios
            total = train_ratio + val_ratio + test_ratio
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total
            
            self.logger.log_info(f"Train: {train_ratio:.2f}, Val: {val_ratio:.2f}, Test: {test_ratio:.2f}")
            
            # Split interactions
            num_interactions = len(interactions)
            indices = list(range(num_interactions))
            
            # Shuffle
            shuffled_indices = self.seed_manager.shuffle(
                indices,
                operation_name='data_split_shuffle'
            )
            
            # Calculate split points
            train_end = int(train_ratio * num_interactions)
            val_end = train_end + int(val_ratio * num_interactions)
            
            # Create splits
            splits = {
                'train': [interactions[i] for i in shuffled_indices[:train_end]],
                'val': [interactions[i] for i in shuffled_indices[train_end:val_end]],
                'test': [interactions[i] for i in shuffled_indices[val_end:]]
            }
            
            self.logger.log_info(
                f"Train: {len(splits['train'])}, "
                f"Val: {len(splits['val'])}, "
                f"Test: {len(splits['test'])}"
            )
            
            # Cache splits
            if self.use_cache:
                with open(cache_path, 'wb') as f:
                    pickle.dump(splits, f)
                self.logger.log_info(f"Cached splits to: {cache_path}")
            
            return splits
    
    def _apply_sampling(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply sampling strategies to data.
        
        Args:
            data (Dict[str, Any]): Data dictionary
            
        Returns:
            Dict[str, Any]: Sampled data
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("APPLYING SAMPLING")
        self.logger.log_info("-" * 50)
        
        # Check if sampling is enabled
        sample_size = self.dataset_config.get('sample_size', 0)
        density = self.dataset_config.get('density', 1.0)
        
        if sample_size == 0 and density == 1.0:
            self.logger.log_info("No sampling applied")
            return data
        
        # Check cache
        cache_path = self.cache_dir / "sampled_data.pkl"
        if self.use_cache and cache_path.exists():
            self.logger.log_info(f"Loading sampled data from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        with self.timer.measure("apply_sampling"):
            sampled_data = {}
            
            for split_name, split_data in data.items():
                if sample_size > 0:
                    # Sample fixed number of items
                    sampled = self.seed_manager.random_choice(
                        split_data,
                        size=min(sample_size, len(split_data)),
                        operation_name=f'sample_{split_name}'
                    )
                    sampled_data[split_name] = sampled.tolist() if hasattr(sampled, 'tolist') else sampled
                else:
                    # No fixed size sampling
                    sampled_data[split_name] = split_data
            
            self.logger.log_info(f"Applied sampling: {len(sampled_data['train'])} train samples")
            
            # Cache sampled data
            if self.use_cache:
                with open(cache_path, 'wb') as f:
                    pickle.dump(sampled_data, f)
                self.logger.log_info(f"Cached sampled data to: {cache_path}")
            
            return sampled_data
    
    def _normalize_features(self, item_features: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Normalize item features.
        
        Args:
            item_features (Dict[str, Dict[str, Any]]): Item features
            
        Returns:
            Dict[str, Dict[str, Any]]: Normalized features
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("NORMALIZING FEATURES")
        self.logger.log_info("-" * 50)
        
        # Check cache
        cache_path = self.cache_dir / "normalized_features.pkl"
        if self.use_cache and cache_path.exists():
            self.logger.log_info(f"Loading normalized features from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        with self.timer.measure("normalize_features"):
            # Collect numeric features
            numeric_features = defaultdict(list)
            
            for features in item_features.values():
                for key, value in features.items():
                    if isinstance(value, (int, float)):
                        numeric_features[key].append(value)
            
            # Compute normalization parameters
            normalization_params = {}
            
            for key, values in numeric_features.items():
                if len(values) > 1:
                    min_val = np.min(values)
                    max_val = np.max(values)
                    mean_val = np.mean(values)
                    std_val = np.std(values) if np.std(values) > 0 else 1.0
                    
                    normalization_params[key] = {
                        'min': min_val,
                        'max': max_val,
                        'mean': mean_val,
                        'std': std_val,
                        'method': 'z_score'  # or 'min_max'
                    }
            
            # Normalize features
            normalized_features = {}
            
            for item_id, features in item_features.items():
                normalized = {}
                for key, value in features.items():
                    if key in normalization_params:
                        params = normalization_params[key]
                        if params['method'] == 'z_score':
                            normalized_value = (value - params['mean']) / params['std']
                        else:
                            normalized_value = (value - params['min']) / (params['max'] - params['min'] + 1e-8)
                        
                        normalized[key] = float(normalized_value)
                    else:
                        normalized[key] = value
                
                normalized_features[item_id] = normalized
            
            self.logger.log_info(f"Normalized {len(normalized_features)} items with {len(normalization_params)} features")
            
            # Cache normalized features
            if self.use_cache:
                with open(cache_path, 'wb') as f:
                    pickle.dump(normalized_features, f)
                self.logger.log_info(f"Cached normalized features to: {cache_path}")
            
            return normalized_features
    
    def _generate_statistics(
        self,
        raw_data: Dict[str, Any],
        sampled_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive statistics.
        
        Args:
            raw_data (Dict[str, Any]): Raw data
            sampled_data (Dict[str, Any]): Sampled data
            
        Returns:
            Dict[str, Any]: Statistics
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("GENERATING STATISTICS")
        self.logger.log_info("-" * 50)
        
        stats = {
            'dataset': {
                'name': self.dataset_name,
                'processed_at': datetime.now().isoformat()
            },
            'raw': {
                'num_users': len(raw_data.get('user_items', {})),
                'num_items': len(raw_data.get('item_features', {})),
                'num_interactions': len(raw_data.get('interactions', [])),
                'num_reviews': len(raw_data.get('reviews', []))
            },
            'splits': {
                'train_size': len(sampled_data.get('train', [])),
                'val_size': len(sampled_data.get('val', [])),
                'test_size': len(sampled_data.get('test', []))
            },
            'sparsity': self._compute_sparsity(raw_data),
            'interaction_stats': self._compute_interaction_stats(raw_data)
        }
        
        # Compute additional statistics
        if PANDAS_AVAILABLE and NUMPY_AVAILABLE:
            try:
                # Interaction distribution
                user_items = raw_data.get('user_items', {})
                interaction_counts = [len(items) for items in user_items.values()]
                stats['interaction_distribution'] = {
                    'mean': np.mean(interaction_counts),
                    'std': np.std(interaction_counts),
                    'min': np.min(interaction_counts),
                    'max': np.max(interaction_counts),
                    'median': np.median(interaction_counts),
                    'total': sum(interaction_counts)
                }
            except Exception as e:
                self.logger.log_warning(f"Failed to compute interaction distribution: {e}")
        
        self.logger.log_info("Statistics generation complete")
        
        return stats
    
    def _compute_sparsity(self, raw_data: Dict[str, Any]) -> float:
        """
        Compute data sparsity.
        
        Args:
            raw_data (Dict[str, Any]): Raw data
            
        Returns:
            float: Sparsity value
        """
        num_users = len(raw_data.get('user_items', {}))
        num_items = len(raw_data.get('item_features', {}))
        num_interactions = len(raw_data.get('interactions', []))
        
        if num_users == 0 or num_items == 0:
            return 1.0
        
        max_possible = num_users * num_items
        density = num_interactions / max_possible if max_possible > 0 else 0
        
        return 1.0 - density
    
    def _compute_interaction_stats(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute interaction statistics.
        
        Args:
            raw_data (Dict[str, Any]): Raw data
            
        Returns:
            Dict[str, Any]: Interaction statistics
        """
        interactions = raw_data.get('interactions', [])
        
        if not interactions:
            return {}
        
        # Count per user and item
        user_counts = defaultdict(int)
        item_counts = defaultdict(int)
        
        for interaction in interactions:
            if len(interaction) >= 2:
                user, item = interaction[0], interaction[1]
                user_counts[user] += 1
                item_counts[item] += 1
        
        return {
            'num_users': len(user_counts),
            'num_items': len(item_counts),
            'avg_interactions_per_user': sum(user_counts.values()) / len(user_counts) if user_counts else 0,
            'avg_interactions_per_item': sum(item_counts.values()) / len(item_counts) if item_counts else 0,
            'max_interactions_per_user': max(user_counts.values()) if user_counts else 0,
            'max_interactions_per_item': max(item_counts.values()) if item_counts else 0
        }
    
    def _save_processed_data(
        self,
        raw_data: Dict[str, Any],
        processed_text: Dict[str, Any],
        vocabulary: Dict[str, Any],
        user_profiles: Dict[str, Dict[str, Any]],
        item_features: Dict[str, Dict[str, Any]],
        splits: Dict[str, Any],
        stats: Dict[str, Any]
    ) -> None:
        """
        Save all processed data to files.
        
        Args:
            raw_data (Dict[str, Any]): Raw data
            processed_text (Dict[str, Any]): Processed text
            vocabulary (Dict[str, Any]): Vocabulary
            user_profiles (Dict[str, Dict[str, Any]]): User profiles
            item_features (Dict[str, Dict[str, Any]]): Item features
            splits (Dict[str, Any]): Data splits
            stats (Dict[str, Any]): Statistics
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("SAVING PROCESSED DATA")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("save_data"):
            # Create dataset-specific directory
            dataset_dir = self.output_dir / (self.dataset_name or "default")
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Save raw data
            with open(dataset_dir / "raw_data.pkl", 'wb') as f:
                pickle.dump(raw_data, f)
            
            # Save processed text
            with open(dataset_dir / "processed_text.pkl", 'wb') as f:
                pickle.dump(processed_text, f)
            
            # Save vocabulary
            with open(dataset_dir / "vocabulary.pkl", 'wb') as f:
                pickle.dump(vocabulary, f)
            
            # Save user profiles
            with open(dataset_dir / "user_profiles.pkl", 'wb') as f:
                pickle.dump(user_profiles, f)
            
            # Save item features
            with open(dataset_dir / "item_features.pkl", 'wb') as f:
                pickle.dump(item_features, f)
            
            # Save splits
            with open(dataset_dir / "splits.pkl", 'wb') as f:
                pickle.dump(splits, f)
            
            # Save statistics
            with open(dataset_dir / "statistics.json", 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            # Save config
            with open(dataset_dir / "config.yaml", 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            # Save in CSV format for easy inspection
            if PANDAS_AVAILABLE and splits.get('train'):
                self._save_csv_splits(dataset_dir, splits)
            
            self.logger.log_info(f"All data saved to: {dataset_dir}")
            
            # Save summary
            summary = self._generate_summary(stats)
            with open(dataset_dir / "SUMMARY.txt", 'w') as f:
                f.write(summary)
            
            self.logger.log_info(f"Summary saved to: {dataset_dir / 'SUMMARY.txt'}")
    
    def _save_csv_splits(self, dataset_dir: Path, splits: Dict[str, Any]) -> None:
        """
        Save splits in CSV format.
        
        Args:
            dataset_dir (Path): Directory to save to
            splits (Dict[str, Any]): Data splits
        """
        try:
            import pandas as pd
            
            for split_name, split_data in splits.items():
                if not split_data:
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(split_data, columns=['user', 'item', 'rating', 'timestamp'] 
                                 if len(split_data[0]) >= 4 else ['user', 'item'])
                df.to_csv(dataset_dir / f"{split_name}.csv", index=False)
            
            self.logger.log_info(f"CSV splits saved to: {dataset_dir}")
        except Exception as e:
            self.logger.log_warning(f"Failed to save CSV splits: {e}")
    
    def _generate_summary(self, stats: Dict[str, Any]) -> str:
        """
        Generate a text summary of preprocessing results.
        
        Args:
            stats (Dict[str, Any]): Statistics
            
        Returns:
            str: Text summary
        """
        lines = [
            "=" * 80,
            f"DATA PREPROCESSING SUMMARY: {self.dataset_name}",
            "=" * 80,
            f"Processed at: {stats.get('dataset', {}).get('processed_at', 'Unknown')}",
            "",
            "DATASET STATISTICS",
            "-" * 40,
        ]
        
        # Raw data stats
        raw_stats = stats.get('raw', {})
        lines.append(f"Number of Users: {raw_stats.get('num_users', 0):,}")
        lines.append(f"Number of Items: {raw_stats.get('num_items', 0):,}")
        lines.append(f"Number of Interactions: {raw_stats.get('num_interactions', 0):,}")
        lines.append(f"Number of Reviews: {raw_stats.get('num_reviews', 0):,}")
        
        # Sparsity
        sparsity = stats.get('sparsity', 1.0)
        lines.append(f"Sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
        
        # Interaction stats
        interaction_stats = stats.get('interaction_stats', {})
        if interaction_stats:
            lines.append("")
            lines.append("INTERACTION DISTRIBUTION")
            lines.append("-" * 40)
            lines.append(f"Avg interactions per user: {interaction_stats.get('avg_interactions_per_user', 0):.2f}")
            lines.append(f"Avg interactions per item: {interaction_stats.get('avg_interactions_per_item', 0):.2f}")
            lines.append(f"Max interactions per user: {interaction_stats.get('max_interactions_per_user', 0):,}")
            lines.append(f"Max interactions per item: {interaction_stats.get('max_interactions_per_item', 0):,}")
        
        # Split stats
        split_stats = stats.get('splits', {})
        if split_stats:
            lines.append("")
            lines.append("DATA SPLITS")
            lines.append("-" * 40)
            lines.append(f"Train: {split_stats.get('train_size', 0):,}")
            lines.append(f"Validation: {split_stats.get('val_size', 0):,}")
            lines.append(f"Test: {split_stats.get('test_size', 0):,}")
            total = sum(split_stats.values())
            if total > 0:
                lines.append(f"Total: {total:,}")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("End of Summary")
        lines.append("=" * 80)
        
        return "\n".join(lines)


def main():
    """
    Main entry point for preprocessing data from command line.
    """
    parser = argparse.ArgumentParser(description="H-GRAGrecsys Data Preprocessing Script")
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Name of the dataset to preprocess'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Directory containing raw data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save processed data'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel processing'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of parallel workers'
    )
    
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List supported datasets'
    )
    
    parser.add_argument(
        '--download-only',
        action='store_true',
        help='Only download the dataset, skip preprocessing'
    )
    
    args = parser.parse_args()
    
    # List datasets if requested
    if args.list_datasets:
        print("\nSupported Datasets:")
        print("-" * 40)
        for dataset in DataPreprocessingPipeline.SUPPORTED_DATASETS:
            print(f"  - {dataset}")
        return
    
    # Create preprocessing pipeline
    pipeline = DataPreprocessingPipeline(
        config_path=args.config,
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        use_cache=not args.no_cache,
        parallel=args.parallel,
        num_workers=args.num_workers
    )
    
    # Run preprocessing
    results = pipeline.run()
    
    # Print summary
    print("\n" + pipeline._generate_summary(results['stats']))
    
    return results


if __name__ == "__main__":
    main()