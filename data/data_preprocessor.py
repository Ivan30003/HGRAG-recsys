"""
data_preprocessor.py - Data preprocessing for H-GRAGrecsys

This module handles text preprocessing, feature extraction, summarization,
and encoding for Amazon review dataset.
"""

import re
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from dataclasses import dataclass, field
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import torch
from transformers import AutoTokenizer, AutoModel

# Download NLTK data if not available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class PreprocessedItem:
    """Container for preprocessed item data."""
    item_id: str
    title: str
    description: str
    category: str
    brand: str
    price: float
    average_rating: float
    num_ratings: int
    cleaned_title: str
    cleaned_description: str
    summary: str
    keywords: List[str]
    embedding: Optional[np.ndarray] = None
    tfidf_vector: Optional[np.ndarray] = None


@dataclass
class PreprocessedUser:
    """Container for preprocessed user data."""
    user_id: str
    num_interactions: int
    average_rating: float
    preference_summary: str
    preferred_categories: Dict[str, int]
    preferred_items: List[str]
    high_rating_items: List[str]
    embedding: Optional[np.ndarray] = None
    preference_vector: Optional[np.ndarray] = None


class TextPreprocessor:
    """Text preprocessing utilities for item descriptions and user preferences."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TextPreprocessor.
        
        Args:
            config: Configuration dictionary containing preprocessing parameters
        """
        self.config = config
        self.max_text_length = config['data'].get('max_text_length', 512)
        self.min_word_freq = config.get('min_word_freq', 2)
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Add custom stopwords
        custom_stopwords = config.get('custom_stopwords', [])
        self.stop_words.update(custom_stopwords)
        
        # Initialize stemmer
        self.stemmer = PorterStemmer()
        
        # Initialize spaCy for advanced NLP (if available)
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            logger.warning("spaCy model not found. Using NLTK only.")
            self.nlp = None
        
        # Initialize tokenizer for LLM (if needed)
        self.tokenizer = None
        tokenizer_name = config.get('tokenizer_name', 'bert-base-uncased')
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except:
            logger.warning(f"Tokenizer {tokenizer_name} not found. Using basic tokenizer.")
        
        # Initialize embedding model (if needed)
        self.embedding_model = None
        embedding_model_name = config.get('embedding_model', 'bert-base-uncased')
        try:
            self.embedding_model = AutoModel.from_pretrained(embedding_model_name)
        except:
            logger.warning(f"Embedding model {embedding_model_name} not found. Using TF-IDF.")
        
        # TF-IDF vectorizer for fallback
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=config.get('max_tfidf_features', 10000),
            stop_words='english',
            max_df=0.8,
            min_df=2
        )
        
        # PCA for dimensionality reduction (if needed)
        self.pca = PCA(
            n_components=config.get('pca_components', 256),
            random_state=42
        )
        
        logger.info("TextPreprocessor initialized")
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters, extra spaces, etc.
        
        Args:
            text: Raw text string
        
        Returns:
            Cleaned text string
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits (keep words)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into tokens.
        
        Args:
            text: Text string
        
        Returns:
            List of tokens
        """
        if self.nlp:
            doc = self.nlp(text)
            tokens = [token.text.lower() for token in doc if not token.is_stop and token.is_alpha]
        else:
            tokens = word_tokenize(text)
            tokens = [t.lower() for t in tokens if t.isalpha() and t not in self.stop_words]
        
        return tokens
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming to tokens.
        
        Args:
            tokens: List of tokens
        
        Returns:
            Stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Text string
            top_k: Number of keywords to extract
        
        Returns:
            List of top keywords
        """
        if not text:
            return []
        
        # Clean and tokenize
        cleaned = self.clean_text(text)
        tokens = self.tokenize_text(cleaned)
        
        # Calculate frequency
        freq = Counter(tokens)
        
        # Get top k
        keywords = [word for word, _ in freq.most_common(top_k)]
        
        return keywords
    
    def summarize_text(self, text: str, max_words: int = 50) -> str:
        """
        Generate a summary of text.
        
        Args:
            text: Text string
            max_words: Maximum number of words in summary
        
        Returns:
            Summary string
        """
        if not text:
            return ""
        
        # Clean text
        cleaned = self.clean_text(text)
        sentences = cleaned.split('. ')
        
        if len(sentences) <= 1:
            # If only one sentence, return truncated version
            words = cleaned.split()
            return ' '.join(words[:max_words])
        
        # Simple extractive summarization: take first sentence and additional sentences
        # with high TF-IDF scores
        summary_parts = []
        
        # Always include first sentence
        if sentences[0]:
            summary_parts.append(sentences[0].strip())
        
        # Calculate TF-IDF scores for sentences
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            scores = tfidf_matrix.sum(axis=1).A1
            
            # Get top sentences (excluding first)
            sentence_scores = [(i, scores[i]) for i in range(1, len(sentences))]
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Add top sentences until max_words reached
            current_words = sum(len(s.split()) for s in summary_parts)
            for idx, _ in sentence_scores[:3]:  # Limit to 3 additional sentences
                if current_words + len(sentences[idx].split()) <= max_words:
                    summary_parts.append(sentences[idx].strip())
                    current_words += len(sentences[idx].split())
                else:
                    break
        
        except:
            # Fallback: take first few sentences
            for sentence in sentences[1:]:
                if len(summary_parts) < 3 and sentence.strip():
                    summary_parts.append(sentence.strip())
                    break
        
        # Join and truncate
        summary = '. '.join(summary_parts)
        if len(summary.split()) > max_words:
            summary = ' '.join(summary.split()[:max_words]) + '...'
        
        return summary
    
    def get_text_embedding(self, text: str, model_type: str = 'bert') -> np.ndarray:
        """
        Get text embedding from pre-trained model.
        
        Args:
            text: Text string
            model_type: 'bert', 'tfidf', or 'basic'
        
        Returns:
            Embedding vector
        """
        if not text:
            return np.zeros(self.get_embedding_dimension())
        
        if model_type == 'bert' and self.embedding_model and self.tokenizer:
            # BERT embedding
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            return embedding
        
        elif model_type == 'tfidf':
            # TF-IDF embedding
            if not hasattr(self, 'tfidf_vectorizer'):
                self.tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
                self.tfidf_vectorizer.fit([text])  # Fit on single document
            embedding = self.tfidf_vectorizer.transform([text]).toarray().squeeze()
            return embedding
        
        else:
            # Basic embedding: use normalized token count
            tokens = self.tokenize_text(text)
            embedding = np.zeros(100)
            for token in tokens:
                # Simple hash-based embedding
                hash_val = int(hashlib.md5(token.encode()).hexdigest(), 16) % 100
                embedding[hash_val] += 1
            return normalize(embedding.reshape(1, -1)).flatten()
    
    def get_embedding_dimension(self) -> int:
        """Get dimension of embeddings."""
        if self.embedding_model:
            return self.embedding_model.config.hidden_size
        elif hasattr(self, 'tfidf_vectorizer'):
            return 10000
        else:
            return 100


class DataPreprocessor:
    """Main data preprocessing class."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataPreprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_dir = Path(config['data']['data_dir'])
        self.processed_dir = Path(config['data'].get('processed_dir', './data/processed'))
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize text preprocessor
        self.text_processor = TextPreprocessor(config)
        
        # Initialize encoders
        self.use_llm_encoder = config.get('use_llm_encoder', False)
        self.encoder_model = config.get('encoder_model', 'text-embedding-ada-002')
        
        # Feature extraction parameters
        self.max_features = config.get('max_features', 10000)
        self.min_feature_freq = config.get('min_feature_freq', 2)
        
        # Cache for processed data
        self.cache_dir = self.processed_dir / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("DataPreprocessor initialized")
    
    def extract_text_features(self, items: List[Dict]) -> Dict[str, Dict]:
        """
        Extract and preprocess text features from items.
        
        Args:
            items: List of item dictionaries
        
        Returns:
            Dictionary mapping item_id to processed features
        """
        logger.info(f"Extracting text features for {len(items)} items")
        
        processed_features = {}
        
        for item in tqdm(items, desc="Extracting text features"):
            item_id = item['item_id']
            
            # Extract and clean text fields
            title = item.get('title', '')
            description = item.get('description', '')
            category = item.get('category', '')
            brand = item.get('brand', '')
            
            # Clean text
            cleaned_title = self.text_processor.clean_text(title)
            cleaned_description = self.text_processor.clean_text(description)
            
            # Generate summary
            summary = self.text_processor.summarize_text(
                description if description else title,
                max_words=50
            )
            
            # Extract keywords
            keywords = self.text_processor.extract_keywords(
                description if description else title,
                top_k=10
            )
            
            # Create preprocessed item
            processed_item = PreprocessedItem(
                item_id=item_id,
                title=title,
                description=description,
                category=category,
                brand=brand,
                price=item.get('price', 0.0),
                average_rating=item.get('average_rating', 0.0),
                num_ratings=item.get('num_ratings', 0),
                cleaned_title=cleaned_title,
                cleaned_description=cleaned_description,
                summary=summary,
                keywords=keywords
            )
            
            processed_features[item_id] = processed_item
        
        logger.info(f"Extracted features for {len(processed_features)} items")
        return processed_features
    
    def extract_user_features(self, users: List[Dict], 
                            interactions: List[Dict]) -> Dict[str, Dict]:
        """
        Extract and preprocess user features from interactions.
        
        Args:
            users: List of user dictionaries
            interactions: List of interaction dictionaries
        
        Returns:
            Dictionary mapping user_id to processed features
        """
        logger.info(f"Extracting user features for {len(users)} users")
        
        # Group interactions by user
        user_interactions = defaultdict(list)
        for interaction in interactions:
            user_id = interaction['user_id']
            user_interactions[user_id].append(interaction)
        
        processed_features = {}
        
        for user in tqdm(users, desc="Extracting user features"):
            user_id = user['user_id']
            user_ints = user_interactions.get(user_id, [])
            
            if not user_ints:
                continue
            
            # Calculate statistics
            ratings = [i.get('rating', 0) for i in user_ints]
            avg_rating = np.mean(ratings) if ratings else 0
            
            # Extract preferred categories
            preferred_categories = defaultdict(int)
            for interaction in user_ints:
                if 'category' in interaction:
                    preferred_categories[interaction['category']] += 1
            
            # Get high-rating items (rating >= 4)
            high_rating_items = [
                i['item_id'] for i in user_ints 
                if i.get('rating', 0) >= 4
            ]
            
            # Generate preference summary
            preferred_cats = sorted(
                preferred_categories.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            preference_summary = f"User has {len(user_ints)} interactions. "
            if preferred_cats:
                preference_summary += f"Prefers categories: {', '.join([cat for cat, _ in preferred_cats])}. "
            if high_rating_items:
                preference_summary += f"Rated {len(high_rating_items)} items highly."
            
            # Create preprocessed user
            processed_user = PreprocessedUser(
                user_id=user_id,
                num_interactions=len(user_ints),
                average_rating=avg_rating,
                preference_summary=preference_summary,
                preferred_categories=dict(preferred_categories),
                preferred_items=[i['item_id'] for i in user_ints if i.get('rating', 0) >= 4],
                high_rating_items=high_rating_items[:20]  # Keep top 20
            )
            
            processed_features[user_id] = processed_user
        
        logger.info(f"Extracted features for {len(processed_features)} users")
        return processed_features
    
    def encode_text_features(self, 
                           items: Dict[str, PreprocessedItem],
                           method: str = 'tfidf') -> Dict[str, np.ndarray]:
        """
        Encode text features using specified method.
        
        Args:
            items: Dictionary of PreprocessedItem objects
            method: Encoding method ('tfidf', 'bert', 'hash')
        
        Returns:
            Dictionary mapping item_id to embedding vector
        """
        logger.info(f"Encoding text features using {method}")
        
        embeddings = {}
        
        if method == 'bert':
            # Use BERT embeddings
            for item_id, item in tqdm(items.items(), desc="BERT encoding"):
                text = f"{item.cleaned_title} {item.summary}"
                embedding = self.text_processor.get_text_embedding(text, model_type='bert')
                embeddings[item_id] = embedding
        
        elif method == 'tfidf':
            # Use TF-IDF with PCA
            texts = []
            item_ids = []
            
            for item_id, item in items.items():
                text = f"{item.cleaned_title} {item.cleaned_description}"
                texts.append(text)
                item_ids.append(item_id)
            
            # Fit TF-IDF
            tfidf_matrix = self.text_processor.tfidf_vectorizer.fit_transform(texts)
            
            # Apply PCA if needed
            if tfidf_matrix.shape[1] > 256:
                tfidf_matrix = self.text_processor.pca.fit_transform(tfidf_matrix.toarray())
                # Normalize
                tfidf_matrix = normalize(tfidf_matrix)
            
            # Store embeddings
            for idx, item_id in enumerate(item_ids):
                embeddings[item_id] = tfidf_matrix[idx]
        
        elif method == 'hash':
            # Simple hash-based encoding
            for item_id, item in items.items():
                text = f"{item.cleaned_title} {item.cleaned_description}"
                embedding = self.text_processor.get_text_embedding(text, model_type='basic')
                embeddings[item_id] = embedding
        
        else:
            raise ValueError(f"Unknown encoding method: {method}")
        
        logger.info(f"Encoded {len(embeddings)} items")
        return embeddings
    
    def encode_user_preferences(self,
                              users: Dict[str, PreprocessedUser],
                              items: Dict[str, PreprocessedItem],
                              method: str = 'tfidf') -> Dict[str, np.ndarray]:
        """
        Encode user preferences.
        
        Args:
            users: Dictionary of PreprocessedUser objects
            items: Dictionary of PreprocessedItem objects
            method: Encoding method
        
        Returns:
            Dictionary mapping user_id to preference embedding
        """
        logger.info(f"Encoding user preferences using {method}")
        
        embeddings = {}
        
        for user_id, user in tqdm(users.items(), desc="Encoding users"):
            # Get preference text
            pref_text = user.preference_summary
            
            # Add preferred items info
            if user.preferred_items:
                item_texts = []
                for item_id in user.preferred_items[:5]:  # Top 5 items
                    if item_id in items:
                        item = items[item_id]
                        item_text = f"{item.cleaned_title} {item.category} {item.summary}"
                        item_texts.append(item_text)
                
                if item_texts:
                    pref_text += " " + " ".join(item_texts)
            
            # Encode
            if method == 'bert':
                embedding = self.text_processor.get_text_embedding(pref_text, model_type='bert')
            elif method == 'tfidf':
                embedding = self.text_processor.get_text_embedding(pref_text, model_type='tfidf')
            else:
                embedding = self.text_processor.get_text_embedding(pref_text, model_type='basic')
            
            embeddings[user_id] = embedding
        
        logger.info(f"Encoded {len(embeddings)} users")
        return embeddings
    
    def create_feature_matrices(self,
                               users: Dict[str, PreprocessedUser],
                               items: Dict[str, PreprocessedItem],
                               user_encodings: Dict[str, np.ndarray],
                               item_encodings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Create feature matrices for training.
        
        Args:
            users: Dictionary of PreprocessedUser objects
            items: Dictionary of PreprocessedItem objects
            user_encodings: User embeddings
            item_encodings: Item embeddings
        
        Returns:
            Dictionary containing feature matrices
        """
        logger.info("Creating feature matrices")
        
        # Get item IDs and user IDs
        user_ids = list(users.keys())
        item_ids = list(items.keys())
        
        # Create mappings
        user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
        
        # Create feature matrices
        user_features = np.array([user_encodings[uid] for uid in user_ids])
        item_features = np.array([item_encodings[iid] for iid in item_ids])
        
        # Create interaction matrix (if interactions available)
        interaction_matrix = np.zeros((len(user_ids), len(item_ids)), dtype=np.float32)
        # Will be filled by data loader
        
        return {
            'user_ids': user_ids,
            'item_ids': item_ids,
            'user_to_idx': user_to_idx,
            'item_to_idx': item_to_idx,
            'user_features': user_features,
            'item_features': item_features,
            'interaction_matrix': interaction_matrix
        }
    
    def create_item_embeddings(self,
                              items: Dict[str, PreprocessedItem],
                              method: str = 'tfidf') -> Dict[str, np.ndarray]:
        """
        Create item embeddings for Graph RAG.
        
        Args:
            items: Dictionary of PreprocessedItem objects
            method: Encoding method
        
        Returns:
            Dictionary mapping item_id to embedding
        """
        return self.encode_text_features(items, method)
    
    def create_user_embeddings(self,
                              users: Dict[str, PreprocessedUser],
                              items: Dict[str, PreprocessedItem],
                              method: str = 'tfidf') -> Dict[str, np.ndarray]:
        """
        Create user embeddings for Graph RAG.
        
        Args:
            users: Dictionary of PreprocessedUser objects
            items: Dictionary of PreprocessedItem objects
            method: Encoding method
        
        Returns:
            Dictionary mapping user_id to embedding
        """
        return self.encode_user_preferences(users, items, method)
    
    def save_preprocessed_data(self,
                              data: Dict[str, Any],
                              filename: str) -> None:
        """
        Save preprocessed data to disk.
        
        Args:
            data: Data dictionary to save
            filename: Output filename
        """
        filepath = self.processed_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            elif isinstance(value, dict) and all(isinstance(v, np.ndarray) for v in value.values()):
                serializable_data[key] = {k: v.tolist() for k, v in value.items()}
            else:
                serializable_data[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        logger.info(f"Saved preprocessed data to {filepath}")
    
    def load_preprocessed_data(self, filename: str) -> Dict[str, Any]:
        """
        Load preprocessed data from disk.
        
        Args:
            filename: Input filename
        
        Returns:
            Data dictionary
        """
        filepath = self.processed_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        for key, value in data.items():
            if key.endswith('_matrix') and isinstance(value, list):
                data[key] = np.array(value, dtype=np.float32)
            elif isinstance(value, dict) and all(isinstance(v, list) for v in value.values()):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, list) and subvalue:
                        data[key][subkey] = np.array(subvalue, dtype=np.float32)
        
        logger.info(f"Loaded preprocessed data from {filepath}")
        return data
    
    def process_dataset(self, 
                       dataset: Any,
                       save: bool = True) -> Dict[str, Any]:
        """
        Process complete dataset.
        
        Args:
            dataset: Dataset object (AmazonDataset)
            save: Whether to save processed data
        
        Returns:
            Dictionary containing all processed data
        """
        logger.info(f"Processing dataset: {dataset.dataset_name}")
        
        # Extract features
        item_features = self.extract_text_features(list(dataset.items.values()))
        user_features = self.extract_user_features(
            list(dataset.users.values()),
            dataset.interactions
        )
        
        # Create embeddings
        item_embeddings = self.create_item_embeddings(item_features, method='tfidf')
        user_embeddings = self.create_user_embeddings(
            user_features,
            item_features,
            method='tfidf'
        )
        
        # Create feature matrices
        matrices = self.create_feature_matrices(
            user_features,
            item_features,
            user_embeddings,
            item_embeddings
        )
        
        # Prepare final data
        processed_data = {
            'dataset_name': dataset.dataset_name,
            'item_features': item_features,
            'user_features': user_features,
            'item_embeddings': item_embeddings,
            'user_embeddings': user_embeddings,
            'matrices': matrices,
            'statistics': {
                'num_users': len(user_features),
                'num_items': len(item_features),
                'num_interactions': len(dataset.interactions)
            }
        }
        
        if save:
            filename = f"{dataset.dataset_name}_processed.json"
            self.save_preprocessed_data(processed_data, filename)
        
        logger.info(f"Processed dataset: {dataset.dataset_name}")
        return processed_data


class FeatureExtractor:
    """Feature extraction utilities for recommendation."""
    
    @staticmethod
    def extract_interaction_features(interactions: List[Dict]) -> Dict[str, Any]:
        """
        Extract features from interactions.
        
        Args:
            interactions: List of interaction dictionaries
        
        Returns:
            Dictionary of interaction statistics
        """
        if not interactions:
            return {}
        
        ratings = [i.get('rating', 0) for i in interactions]
        timestamps = [i.get('timestamp', 0) for i in interactions]
        
        return {
            'num_interactions': len(interactions),
            'avg_rating': np.mean(ratings),
            'std_rating': np.std(ratings),
            'max_rating': max(ratings) if ratings else 0,
            'min_rating': min(ratings) if ratings else 0,
            'first_interaction': min(timestamps) if timestamps else 0,
            'last_interaction': max(timestamps) if timestamps else 0
        }
    
    @staticmethod
    def extract_category_features(items: List[Dict]) -> Dict[str, Any]:
        """
        Extract features from item categories.
        
        Args:
            items: List of item dictionaries
        
        Returns:
            Dictionary of category statistics
        """
        categories = [item.get('category', '') for item in items if item.get('category')]
        category_counts = Counter(categories)
        
        return {
            'num_categories': len(category_counts),
            'category_counts': dict(category_counts),
            'most_common_categories': category_counts.most_common(5)
        }
    
    @staticmethod
    def extract_temporal_features(interactions: List[Dict]) -> Dict[str, Any]:
        """
        Extract temporal features from interactions.
        
        Args:
            interactions: List of interaction dictionaries
        
        Returns:
            Dictionary of temporal features
        """
        if not interactions:
            return {}
        
        timestamps = [i.get('timestamp', 0) for i in interactions]
        timestamps = sorted(timestamps)
        
        if len(timestamps) < 2:
            return {'time_span': 0}
        
        return {
            'time_span': timestamps[-1] - timestamps[0],
            'avg_time_between': np.mean(np.diff(timestamps)),
            'std_time_between': np.std(np.diff(timestamps)),
            'interaction_frequency': len(timestamps) / (timestamps[-1] - timestamps[0])
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
        'min_word_freq': 2,
        'custom_stopwords': ['amazon', 'product', 'buy', 'purchase'],
        'max_tfidf_features': 10000,
        'pca_components': 256
    }
    
    # Load dataset
    from data.dataset import AmazonDataset
    
    dataset = AmazonDataset('CDs_and_Vinyl', config)
    dataset.load_data(limit_users=100)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Process dataset
    processed_data = preprocessor.process_dataset(dataset, save=True)
    
    print(f"Processed {processed_data['statistics']['num_users']} users")
    print(f"Processed {processed_data['statistics']['num_items']} items")
    print(f"Feature matrices shape: {processed_data['matrices']['user_features'].shape}")