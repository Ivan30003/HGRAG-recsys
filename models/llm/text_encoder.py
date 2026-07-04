"""
Text Encoder Module for H-GRAGrecsys

This module provides text encoding capabilities for converting text to embeddings,
supporting multiple embedding models, batch processing, and memory encoding.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import sys
import os
import json
from functools import lru_cache

# Import optional libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.agent.memory import AgentMemory
from models.agent.memory_components import MemoryComponent
from utils.logger import Logger
from utils.config_loader import ConfigLoader


@dataclass
class EncodedText:
    """
    Encoded text with metadata.
    
    Attributes:
        embedding: Embedding vector
        text: Original text
        tokens: Number of tokens
        model: Model used for encoding
        timestamp: Encoding timestamp
        metadata: Additional metadata
    """
    embedding: torch.Tensor
    text: str = ""
    tokens: int = 0
    model: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextEncoder:
    """
    Text encoder for converting text to embeddings.
    
    This class handles:
    - Encoding text with multiple models
    - Batch encoding for efficiency
    - Computing similarity between texts
    - Encoding agent memories
    - Caching embeddings for performance
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the text encoder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = Logger.get_instance(log_dir='logs', name='text_encoder')
        
        # Extract configuration
        llm_config = config.get('model', {}).get('llm', {})
        self.model_name = llm_config.get('embedding_model', 'all-MiniLM-L6-v2')
        self.model_type = llm_config.get('embedding_model_type', 'sentence_transformers')
        self.embedding_dim = llm_config.get('embedding_dim', 384)
        self.max_length = llm_config.get('max_embedding_length', 512)
        self.batch_size = llm_config.get('embedding_batch_size', 32)
        self.device = llm_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cache configuration
        self.use_cache = llm_config.get('use_embedding_cache', True)
        self.cache_size = llm_config.get('embedding_cache_size', 10000)
        self.embedding_cache: Dict[str, torch.Tensor] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        self._initialize_model()
        
        # Statistics
        self.encoder_stats = {
            'total_encodings': 0,
            'batch_encodings': 0,
            'total_tokens': 0,
            'avg_encoding_time': 0.0,
            'encoding_times': [],
            'cache_hit_rate': 0.0
        }
        
        self.logger.log_info(f"Initialized TextEncoder with model={self.model_name}, dim={self.embedding_dim}")
    
    def _initialize_model(self) -> None:
        """
        Initialize the embedding model based on configuration.
        """
        try:
            if self.model_type == 'sentence_transformers':
                if not SENTENCE_TRANSFORMERS_AVAILABLE:
                    raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")
                
                self.logger.log_info(f"Loading SentenceTransformer model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name, device=self.device)
                
                # Get embedding dimension
                sample_embedding = self.model.encode(["test"], convert_to_tensor=True)
                self.embedding_dim = sample_embedding.shape[1]
                self.logger.log_info(f"Loaded model with embedding dimension: {self.embedding_dim}")
            
            elif self.model_type == 'transformers':
                if not TRANSFORMERS_AVAILABLE:
                    raise ImportError("transformers not installed. Install with: pip install transformers")
                
                self.logger.log_info(f"Loading Transformers model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                
                # Get embedding dimension
                self.embedding_dim = self.model.config.hidden_size
                self.logger.log_info(f"Loaded model with embedding dimension: {self.embedding_dim}")
            
            elif self.model_type == 'openai':
                if not OPENAI_AVAILABLE:
                    raise ImportError("openai not installed. Install with: pip install openai")
                
                self.logger.log_info(f"Using OpenAI embedding model: {self.model_name}")
                # OpenAI embeddings will be fetched via API
                # Embedding dimension is known from model name
                if 'ada' in self.model_name:
                    self.embedding_dim = 1536
                elif 'text-embedding-3' in self.model_name:
                    self.embedding_dim = 3072
                else:
                    self.embedding_dim = 1536  # Default
            
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            self.logger.log_info(f"TextEncoder initialized successfully on {self.device}")
            
        except Exception as e:
            self.logger.log_error(f"Failed to initialize model: {str(e)}")
            raise
    
    def encode(self, text: Union[str, List[str]], 
              max_length: Optional[int] = None,
              use_cache: Optional[bool] = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Encode text(s) to embeddings.
        
        Args:
            text: Single text or list of texts
            max_length: Maximum text length
            use_cache: Whether to use cache
            
        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: Embedding(s)
        """
        use_cache = use_cache if use_cache is not None else self.use_cache
        max_length = max_length or self.max_length
        
        if isinstance(text, str):
            return self._encode_single(text, max_length, use_cache)
        else:
            return self._encode_batch(text, max_length, use_cache)
    
    def _encode_single(self, text: str, max_length: int, use_cache: bool) -> torch.Tensor:
        """
        Encode a single text.
        
        Args:
            text: Input text
            max_length: Maximum text length
            use_cache: Whether to use cache
            
        Returns:
            torch.Tensor: Embedding vector
        """
        # Check cache
        cache_key = self._get_cache_key(text)
        if use_cache and cache_key in self.embedding_cache:
            self.cache_hits += 1
            self.encoder_stats['cache_hit_rate'] = self.cache_hits / (self.cache_hits + self.cache_misses)
            return self.embedding_cache[cache_key]
        
        self.cache_misses += 1
        
        # Truncate text
        if len(text) > max_length * 4:  # Rough character to token ratio
            text = text[:max_length * 4]
        
        # Encode based on model type
        start_time = datetime.now().timestamp()
        
        if self.model_type == 'sentence_transformers':
            embedding = self._encode_sentence_transformers(text)
        elif self.model_type == 'transformers':
            embedding = self._encode_transformers(text, max_length)
        elif self.model_type == 'openai':
            embedding = self._encode_openai(text)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Update statistics
        encoding_time = datetime.now().timestamp() - start_time
        self.encoder_stats['total_encodings'] += 1
        self.encoder_stats['encoding_times'].append(encoding_time)
        self.encoder_stats['avg_encoding_time'] = np.mean(self.encoder_stats['encoding_times'])
        
        # Cache embedding
        if use_cache:
            if len(self.embedding_cache) >= self.cache_size:
                # Remove oldest entry
                self.embedding_cache.pop(next(iter(self.embedding_cache)))
            self.embedding_cache[cache_key] = embedding
        
        return embedding
    
    def _encode_batch(self, texts: List[str], max_length: int, use_cache: bool) -> List[torch.Tensor]:
        """
        Encode multiple texts in batch.
        
        Args:
            texts: List of input texts
            max_length: Maximum text length
            use_cache: Whether to use cache
            
        Returns:
            List[torch.Tensor]: List of embeddings
        """
        if not texts:
            return []
        
        # Check which texts are in cache
        cached_embeddings = {}
        texts_to_encode = []
        
        if use_cache:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self.embedding_cache:
                    cached_embeddings[i] = self.embedding_cache[cache_key]
                    self.cache_hits += 1
                else:
                    texts_to_encode.append((i, text))
                    self.cache_misses += 1
        else:
            texts_to_encode = [(i, text) for i, text in enumerate(texts)]
        
        # Update cache hit rate
        total_requests = self.cache_hits + self.cache_misses
        self.encoder_stats['cache_hit_rate'] = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        # If all texts are cached, return cached embeddings
        if not texts_to_encode:
            return [cached_embeddings[i] for i in range(len(texts))]
        
        # Prepare texts for encoding
        encode_texts = [text for _, text in texts_to_encode]
        
        # Process in batches
        start_time = datetime.now().timestamp()
        all_embeddings = []
        
        for i in range(0, len(encode_texts), self.batch_size):
            batch_texts = encode_texts[i:i + self.batch_size]
            
            # Encode based on model type
            if self.model_type == 'sentence_transformers':
                batch_embeddings = self._encode_batch_sentence_transformers(batch_texts)
            elif self.model_type == 'transformers':
                batch_embeddings = self._encode_batch_transformers(batch_texts, max_length)
            elif self.model_type == 'openai':
                batch_embeddings = self._encode_batch_openai(batch_texts)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            all_embeddings.extend(batch_embeddings)
        
        # Update statistics
        encoding_time = datetime.now().timestamp() - start_time
        self.encoder_stats['batch_encodings'] += 1
        self.encoder_stats['total_encodings'] += len(encode_texts)
        self.encoder_stats['encoding_times'].append(encoding_time)
        self.encoder_stats['avg_encoding_time'] = np.mean(self.encoder_stats['encoding_times'])
        
        # Create result mapping
        result = [None] * len(texts)
        
        # Add cached embeddings
        for idx, embedding in cached_embeddings.items():
            result[idx] = embedding
        
        # Add newly encoded embeddings
        for (idx, _), embedding in zip(texts_to_encode, all_embeddings):
            result[idx] = embedding
            
            # Cache if enabled
            if use_cache:
                cache_key = self._get_cache_key(texts[idx])
                if len(self.embedding_cache) < self.cache_size:
                    self.embedding_cache[cache_key] = embedding
        
        return result
    
    def _encode_sentence_transformers(self, text: str) -> torch.Tensor:
        """
        Encode using SentenceTransformer.
        
        Args:
            text: Input text
            
        Returns:
            torch.Tensor: Embedding vector
        """
        embedding = self.model.encode(text, convert_to_tensor=True)
        # Normalize
        embedding = embedding / torch.norm(embedding)
        return embedding
    
    def _encode_batch_sentence_transformers(self, texts: List[str]) -> List[torch.Tensor]:
        """
        Encode batch using SentenceTransformer.
        
        Args:
            texts: List of input texts
            
        Returns:
            List[torch.Tensor]: List of embeddings
        """
        embeddings = self.model.encode(texts, convert_to_tensor=True, batch_size=self.batch_size)
        # Normalize
        embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
        return [emb for emb in embeddings]
    
    def _encode_transformers(self, text: str, max_length: int) -> torch.Tensor:
        """
        Encode using Transformers model.
        
        Args:
            text: Input text
            max_length: Maximum text length
            
        Returns:
            torch.Tensor: Embedding vector
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        
        # Normalize
        embedding = embedding / torch.norm(embedding)
        return embedding
    
    def _encode_batch_transformers(self, texts: List[str], max_length: int) -> List[torch.Tensor]:
        """
        Encode batch using Transformers model.
        
        Args:
            texts: List of input texts
            max_length: Maximum text length
            
        Returns:
            List[torch.Tensor]: List of embeddings
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Normalize
        embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
        return [emb for emb in embeddings]
    
    def _encode_openai(self, text: str) -> torch.Tensor:
        """
        Encode using OpenAI API.
        
        Args:
            text: Input text
            
        Returns:
            torch.Tensor: Embedding vector
        """
        try:
            response = openai.Embedding.create(
                model=self.model_name,
                input=text
            )
            embedding = np.array(response.data[0].embedding)
            embedding = torch.tensor(embedding, dtype=torch.float32)
            # Normalize
            embedding = embedding / torch.norm(embedding)
            return embedding
        except Exception as e:
            self.logger.log_error(f"OpenAI encoding error: {str(e)}")
            raise
    
    def _encode_batch_openai(self, texts: List[str]) -> List[torch.Tensor]:
        """
        Encode batch using OpenAI API.
        
        Args:
            texts: List of input texts
            
        Returns:
            List[torch.Tensor]: List of embeddings
        """
        try:
            response = openai.Embedding.create(
                model=self.model_name,
                input=texts
            )
            embeddings = []
            for data in response.data:
                embedding = np.array(data.embedding)
                embedding = torch.tensor(embedding, dtype=torch.float32)
                embedding = embedding / torch.norm(embedding)
                embeddings.append(embedding)
            return embeddings
        except Exception as e:
            self.logger.log_error(f"OpenAI batch encoding error: {str(e)}")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for text.
        
        Args:
            text: Input text
            
        Returns:
            str: Cache key
        """
        # Use hash of text (truncated for long texts)
        if len(text) > 1000:
            # Use first 1000 chars + hash of rest
            text_hash = hash(text[1000:])
            key_text = text[:1000] + str(text_hash)
        else:
            key_text = text
        
        return str(hash(key_text))
    
    def batch_encode(self, texts: List[str], 
                    batch_size: Optional[int] = None,
                    max_length: Optional[int] = None,
                    use_cache: Optional[bool] = None) -> List[torch.Tensor]:
        """
        Encode multiple texts with explicit batch control.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            max_length: Maximum text length
            use_cache: Whether to use cache
            
        Returns:
            List[torch.Tensor]: List of embeddings
        """
        if not texts:
            return []
        
        batch_size = batch_size or self.batch_size
        self.batch_size = batch_size
        return self._encode_batch(texts, max_length or self.max_length, use_cache)
    
    def compute_similarity(self, text_a: Union[str, torch.Tensor], 
                          text_b: Union[str, torch.Tensor]) -> float:
        """
        Compute similarity between two texts or embeddings.
        
        Args:
            text_a: First text or embedding
            text_b: Second text or embedding
            
        Returns:
            float: Cosine similarity
        """
        # Convert to embeddings if text
        if isinstance(text_a, str):
            emb_a = self.encode(text_a)
        else:
            emb_a = text_a
        
        if isinstance(text_b, str):
            emb_b = self.encode(text_b)
        else:
            emb_b = text_b
        
        # Compute cosine similarity
        if torch.is_tensor(emb_a) and torch.is_tensor(emb_b):
            similarity = torch.cosine_similarity(
                emb_a.unsqueeze(0) if emb_a.dim() == 1 else emb_a,
                emb_b.unsqueeze(0) if emb_b.dim() == 1 else emb_b
            )
            return similarity.item() if torch.is_tensor(similarity) else similarity
        
        return 0.0
    
    def encode_memory(self, agent_memory: AgentMemory, 
                     component_type: str) -> torch.Tensor:
        """
        Encode a specific memory component from an agent.
        
        Args:
            agent_memory: AgentMemory instance
            component_type: Type of memory component ('intrinsic', 'collaborative', 'interaction')
            
        Returns:
            torch.Tensor: Encoded memory representation
        """
        self.logger.log_info(f"Encoding memory component: {component_type}")
        
        # Get memory component
        if component_type == 'intrinsic':
            memory = agent_memory.get_intrinsic_memory()
        elif component_type == 'collaborative':
            memory = agent_memory.get_collaborative_memory()
        elif component_type == 'interaction':
            memory = agent_memory.get_interaction_memory()
        else:
            raise ValueError(f"Unknown component type: {component_type}")
        
        # Extract text from memory
        text = self._extract_memory_text(memory, component_type)
        
        # Encode text
        embedding = self.encode(text)
        
        return embedding
    
    def _extract_memory_text(self, memory: Any, component_type: str) -> str:
        """
        Extract text from memory for encoding.
        
        Args:
            memory: Memory object
            component_type: Type of memory component
            
        Returns:
            str: Extracted text
        """
        if isinstance(memory, dict):
            # Try to extract text from dictionary
            if 'content' in memory:
                if isinstance(memory['content'], str):
                    return memory['content']
                elif isinstance(memory['content'], dict):
                    return json.dumps(memory['content'])
            elif 'summary' in memory:
                return memory['summary']
            else:
                return json.dumps(memory)
        
        elif isinstance(memory, str):
            return memory
        
        elif hasattr(memory, 'get_text_summary'):
            return memory.get_text_summary()
        
        elif hasattr(memory, 'to_dict'):
            return json.dumps(memory.to_dict())
        
        else:
            return str(memory)
    
    def encode_memory_components(self, agent_memory: AgentMemory) -> Dict[str, torch.Tensor]:
        """
        Encode all memory components of an agent.
        
        Args:
            agent_memory: AgentMemory instance
            
        Returns:
            Dict[str, torch.Tensor]: Encoded components
        """
        components = {}
        
        for comp_type in ['intrinsic', 'collaborative', 'interaction']:
            try:
                components[comp_type] = self.encode_memory(agent_memory, comp_type)
            except Exception as e:
                self.logger.log_warning(f"Failed to encode {comp_type}: {str(e)}")
                components[comp_type] = torch.zeros(self.embedding_dim)
        
        return components
    
    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension.
        
        Returns:
            int: Embedding dimension
        """
        return self.embedding_dim
    
    def get_encoder_statistics(self) -> Dict[str, Any]:
        """
        Get encoder statistics.
        
        Returns:
            Dict[str, Any]: Encoder statistics
        """
        stats = self.encoder_stats.copy()
        stats['cache_size'] = len(self.embedding_cache)
        stats['cache_hit_rate'] = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0
        stats['model_name'] = self.model_name
        stats['model_type'] = self.model_type
        stats['device'] = self.device
        stats['embedding_dim'] = self.embedding_dim
        stats['batch_size'] = self.batch_size
        stats['max_length'] = self.max_length
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger.log_info("Cleared embedding cache")
    
    def reset_statistics(self) -> None:
        """Reset encoder statistics."""
        self.encoder_stats = {
            'total_encodings': 0,
            'batch_encodings': 0,
            'total_tokens': 0,
            'avg_encoding_time': 0.0,
            'encoding_times': [],
            'cache_hit_rate': 0.0
        }
        self.logger.log_info("Reset encoder statistics")
    
    def to_device(self, device: str) -> None:
        """
        Move model to specified device.
        
        Args:
            device: Device to move to ('cuda' or 'cpu')
        """
        self.device = device
        if self.model is not None:
            self.model.to(device)
        self.logger.log_info(f"Moved model to {device}")
    
    def __str__(self) -> str:
        """String representation."""
        return (f"TextEncoder(model={self.model_name}, type={self.model_type}, "
                f"dim={self.embedding_dim}, cache={len(self.embedding_cache)})")


# Example usage
if __name__ == "__main__":
    # Load configuration
    config_path = "config/default_config.yaml"
    config_loader = ConfigLoader(config_path)
    config = config_loader.load_config()
    
    # Create text encoder
    try:
        encoder = TextEncoder(config)
    except Exception as e:
        print(f"Failed to initialize encoder: {e}")
        print("Using mock encoder for demonstration...")
        
        # Create a simple mock encoder
        class MockEncoder:
            def __init__(self):
                self.embedding_dim = 384
            def encode(self, text):
                if isinstance(text, str):
                    return torch.randn(384)
                else:
                    return [torch.randn(384) for _ in text]
            def compute_similarity(self, a, b):
                return 0.5
            def get_embedding_dimension(self):
                return 384
            def get_encoder_statistics(self):
                return {'embedding_dim': 384}
            def clear_cache(self):
                pass
        
        encoder = MockEncoder()
    
    # Test encoding
    test_texts = [
        "This is a sample text for encoding.",
        "Another text with different content.",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    # Single encoding
    print("Testing single encoding...")
    embedding = encoder.encode(test_texts[0])
    print(f"Single embedding shape: {embedding.shape}")
    
    # Batch encoding
    print("\nTesting batch encoding...")
    embeddings = encoder.batch_encode(test_texts)
    print(f"Batch embeddings: {len(embeddings)} embeddings")
    print(f"Each embedding shape: {embeddings[0].shape}")
    
    # Similarity computation
    print("\nTesting similarity computation...")
    similarity = encoder.compute_similarity(test_texts[0], test_texts[1])
    print(f"Similarity between text 0 and 1: {similarity:.4f}")
    
    similarity = encoder.compute_similarity(test_texts[0], test_texts[2])
    print(f"Similarity between text 0 and 2: {similarity:.4f}")
    
    # Get statistics
    print("\nEncoder Statistics:")
    stats = encoder.get_encoder_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Clear cache
    encoder.clear_cache()
    print("\nCache cleared!")