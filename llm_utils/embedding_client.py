"""
Embedding Client Module
Handles text-to-embedding conversion using LLM APIs.
"""

import time
from typing import List, Optional
import numpy as np


class EmbeddingClient:
    """
    Client for generating text embeddings.
    Wraps both API-based and local embedding methods.
    """
    
    def __init__(self, model: str = 'text-embedding-ada-002', dimension: int = 1536):
        """
        Initialize embedding client.
        
        Args:
            model: Embedding model name
            dimension: Output embedding dimension
        """
        self.model = model
        self.dimension = dimension
        self.call_count = 0
        self.total_texts_embedded = 0
    
    def encode(self, text: str) -> np.ndarray:
        """
        Convert text to embedding vector.
        
        Args:
            text: Input text
        
        Returns:
            numpy array of shape (dimension,)
        """
        # In production, this calls the OpenAI Embeddings API
        # For the implementation, we simulate with a deterministic hash-based embedding
        return self._simulate_embedding(text)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Convert multiple texts to embeddings.
        
        Args:
            texts: List of input texts
        
        Returns:
            numpy array of shape (len(texts), dimension)
        """
        embeddings = []
        for text in texts:
            embeddings.append(self.encode(text))
        
        self.call_count += 1
        self.total_texts_embedded += len(texts)
        
        return np.array(embeddings)
    
    def _simulate_embedding(self, text: str) -> np.ndarray:
        """
        Simulate embedding generation using hash.
        In production, replace with actual API call.
        
        Args:
            text: Input text
        
        Returns:
            Deterministic pseudo-embedding
        """
        # Use hash to generate a deterministic vector
        import hashlib
        hash_bytes = hashlib.sha256(text.encode()).digest()
        
        # Convert hash to float array
        embedding = np.zeros(self.dimension)
        for i in range(min(self.dimension, len(hash_bytes) * 8)):
            byte_idx = i // 8
            bit_idx = i % 8
            if byte_idx < len(hash_bytes):
                bit = (hash_bytes[byte_idx] >> bit_idx) & 1
                embedding[i] = bit * 2.0 - 1.0  # Map to [-1, 1]
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Cosine similarity in [0, 1]
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        return float(np.dot(emb1, emb2))
    
    def get_statistics(self) -> dict:
        """Get usage statistics."""
        return {
            'model': self.model,
            'dimension': self.dimension,
            'call_count': self.call_count,
            'texts_embedded': self.total_texts_embedded
        }