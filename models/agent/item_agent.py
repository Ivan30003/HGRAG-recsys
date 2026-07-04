"""
item_agent.py - Item agent implementation for H-GRAGrecsys

This module provides the ItemAgent class that represents items in the
recommendation system with hierarchical memory and collaborative capabilities.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np
import torch

from models.agent.base_agent import BaseAgent, AgentState
from models.agent.memory import HierarchicalMemory, MemoryType
from models.agent.memory_components import (
    IntrinsicMemory,
    CollaborativeMemory,
    InteractionMemory
)

# Configure logging
logger = logging.getLogger(__name__)


class ItemAgent(BaseAgent):
    """
    Item agent representing an item in the recommendation system.
    
    This class extends BaseAgent with item-specific functionality including
    content features, collaborative patterns, and interaction tracking.
    """
    
    def __init__(self, item_id: str, config: Dict[str, Any]):
        """
        Initialize ItemAgent.
        
        Args:
            item_id: Unique identifier for the item
            config: Configuration dictionary
        """
        super().__init__(item_id, config)
        
        # Item-specific attributes
        self.item_id = item_id
        self.title: str = ""
        self.description: str = ""
        self.category: str = ""
        self.brand: str = ""
        self.price: float = 0.0
        self.average_rating: float = 0.0
        self.num_ratings: int = 0
        
        # Content features
        self.content_features: Dict[str, Any] = {}
        self.keywords: List[str] = []
        self.summary: str = ""
        
        # Interaction tracking
        self.interaction_count = 0
        self.user_history: List[str] = []
        self.rating_history: List[Dict] = []
        self.last_interaction_time: Optional[datetime] = None
        
        # Collaborative filtering data
        self.collaborative_similar_items: List[str] = []
        self.similarity_scores: Dict[str, float] = {}
        
        # Embedding cache
        self._content_embedding: Optional[np.ndarray] = None
        self._collaborative_embedding_cache: Optional[np.ndarray] = None
        
        logger.info(f"Initialized ItemAgent: {item_id}")
    
    def get_embedding(self, component_type: Optional[str] = None) -> np.ndarray:
        """
        Get embedding representation of the item.
        
        Args:
            component_type: Optional specific memory component
        
        Returns:
            Embedding vector
        """
        if component_type:
            # Get specific component embedding
            if component_type == MemoryType.INTRINSIC.value:
                return self.get_intrinsic_embedding()
            elif component_type == MemoryType.COLLABORATIVE.value:
                return self.get_collaborative_embedding()
            elif component_type == MemoryType.INTERACTION.value:
                return self.get_interaction_embedding()
            elif component_type == 'content':
                return self.get_content_embedding()
            else:
                logger.warning(f"Unknown component type: {component_type}")
                return self.get_full_embedding()
        else:
            return self.get_full_embedding()
    
    def get_full_embedding(self) -> np.ndarray:
        """
        Get combined embedding of all memory components.
        
        Returns:
            Combined embedding vector
        """
        embeddings = []
        
        # Intrinsic embedding (content-based)
        intrinsic = self.get_intrinsic_embedding()
        if intrinsic is not None:
            embeddings.append(intrinsic)
        
        # Collaborative embedding
        collaborative = self.get_collaborative_embedding()
        if collaborative is not None:
            embeddings.append(collaborative)
        
        # Interaction embedding
        interaction = self.get_interaction_embedding()
        if interaction is not None:
            embeddings.append(interaction)
        
        if not embeddings:
            return np.zeros(128)
        
        # Combine embeddings (average)
        combined = np.mean(embeddings, axis=0)
        
        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        
        return combined
    
    def get_intrinsic_embedding(self) -> Optional[np.ndarray]:
        """
        Get embedding from intrinsic memory (content-based).
        
        Returns:
            Intrinsic embedding or None
        """
        intrinsic = self.get_intrinsic_memory()
        if intrinsic and hasattr(intrinsic, 'get_embedding'):
            return intrinsic.get_embedding()
        return None
    
    def get_collaborative_embedding(self) -> Optional[np.ndarray]:
        """
        Get embedding from collaborative memory.
        
        Returns:
            Collaborative embedding or None
        """
        if self._collaborative_embedding_cache is not None:
            return self._collaborative_embedding_cache
        
        collaborative = self.get_collaborative_memory()
        if collaborative and hasattr(collaborative, 'get_embedding'):
            embedding = collaborative.get_embedding()
            self._collaborative_embedding_cache = embedding
            return embedding
        
        return None
    
    def get_interaction_embedding(self) -> Optional[np.ndarray]:
        """
        Get embedding from interaction memory.
        
        Returns:
            Interaction embedding or None
        """
        interaction = self.get_interaction_memory()
        if interaction and hasattr(interaction, 'get_embedding'):
            return interaction.get_embedding()
        return None
    
    def get_content_embedding(self) -> np.ndarray:
        """
        Get embedding of item content features.
        
        Returns:
            Content embedding vector
        """
        if self._content_embedding is not None:
            return self._content_embedding
        
        # Build content embedding from text features
        text_parts = []
        
        if self.title:
            text_parts.append(self.title)
        if self.description:
            text_parts.append(self.description[:500])  # Limit length
        if self.category:
            text_parts.append(self.category)
        if self.keywords:
            text_parts.append(' '.join(self.keywords[:10]))
        
        if not text_parts:
            return np.zeros(128)
        
        # In practice, use a real embedding model here
        # For now, create a hash-based embedding from text
        combined_text = ' '.join(text_parts)
        
        # Simple hash-based embedding (should be replaced with proper text embedding)
        embedding = np.zeros(128)
        for char in combined_text[:1000]:
            hash_val = hash(char) % 128
            embedding[hash_val] += 1
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        self._content_embedding = embedding
        return embedding
    
    def get_text_representation(self) -> str:
        """
        Get text representation of the item.
        
        Returns:
            Text representation
        """
        parts = [f"Item {self.item_id}"]
        
        if self.title:
            parts.append(f"Title: {self.title[:50]}")
        
        if self.category:
            parts.append(f"Category: {self.category}")
        
        if self.brand:
            parts.append(f"Brand: {self.brand}")
        
        parts.append(f"Price: ${self.price:.2f}")
        
        if self.num_ratings > 0:
            parts.append(f"Rating: {self.average_rating:.1f} ({self.num_ratings} reviews)")
        
        parts.append(f"Interactions: {self.interaction_count}")
        
        return " | ".join(parts)
    
    def set_content(self, content_data: Dict[str, Any]) -> None:
        """
        Set item content features.
        
        Args:
            content_data: Content data dictionary
        """
        self.title = content_data.get('title', '')
        self.description = content_data.get('description', '')
        self.category = content_data.get('category', '')
        self.brand = content_data.get('brand', '')
        self.price = content_data.get('price', 0.0)
        self.average_rating = content_data.get('average_rating', 0.0)
        self.num_ratings = content_data.get('num_ratings', 0)
        self.keywords = content_data.get('keywords', [])
        self.summary = content_data.get('summary', '')
        self.content_features = content_data
        
        # Clear embedding cache
        self._content_embedding = None
        
        # Update intrinsic memory
        self.update_memory(MemoryType.INTRINSIC.value, content_data)
        
        logger.info(f"Updated content for item {self.item_id}")
    
    def add_interaction(self, interaction_data: Dict[str, Any]) -> bool:
        """
        Add an interaction to the item's history.
        
        Args:
            interaction_data: Interaction data (user_id, rating, timestamp, etc.)
        
        Returns:
            True if successful
        """
        # Extract data
        user_id = interaction_data.get('user_id')
        rating = interaction_data.get('rating', 0)
        timestamp = interaction_data.get('timestamp', datetime.now().isoformat())
        review_text = interaction_data.get('review_text', '')
        summary = interaction_data.get('summary', '')
        
        if not user_id:
            logger.warning(f"Missing user_id in interaction for item {self.item_id}")
            return False
        
        # Create interaction entry
        interaction = {
            'user_id': user_id,
            'rating': rating,
            'timestamp': timestamp,
            'review_text': review_text,
            'summary': summary
        }
        
        # Add to history
        self.rating_history.append(interaction)
        self.user_history.append(user_id)
        self.interaction_count += 1
        self.last_interaction_time = datetime.now()
        
        # Update interaction memory
        self.add_interaction_to_memory(interaction)
        
        # Update average rating
        self._update_average_rating()
        
        # Update state
        self.state.interaction_count = self.interaction_count
        self.state.updated_at = datetime.now().isoformat()
        
        # Clear collaborative embedding cache
        self._collaborative_embedding_cache = None
        
        logger.debug(f"Added interaction for item {self.item_id}: user {user_id}, rating {rating}")
        return True
    
    def add_interaction_to_memory(self, interaction: Dict[str, Any]) -> bool:
        """
        Add interaction to memory system.
        
        Args:
            interaction: Interaction data
        
        Returns:
            True if successful
        """
        return super().add_interaction(interaction)
    
    def _update_average_rating(self) -> None:
        """Update average rating based on rating history."""
        if not self.rating_history:
            self.average_rating = 0.0
            self.num_ratings = 0
            return
        
        ratings = [r.get('rating', 0) for r in self.rating_history]
        self.average_rating = np.mean(ratings)
        self.num_ratings = len(ratings)
    
    def get_rating_for_user(self, user_id: str) -> Optional[float]:
        """
        Get rating from a specific user.
        
        Args:
            user_id: User identifier
        
        Returns:
            Rating or None if not found
        """
        for interaction in reversed(self.rating_history):
            if interaction.get('user_id') == user_id:
                return interaction.get('rating')
        return None
    
    def get_interaction_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get interaction history.
        
        Args:
            limit: Maximum number of interactions
        
        Returns:
            List of interactions
        """
        if limit:
            return self.rating_history[-limit:]
        return self.rating_history.copy()
    
    def get_high_rating_users(self, threshold: float = 4.0) -> List[str]:
        """
        Get users who rated the item above threshold.
        
        Args:
            threshold: Rating threshold
        
        Returns:
            List of user IDs
        """
        return [
            interaction['user_id']
            for interaction in self.rating_history
            if interaction.get('rating', 0) >= threshold
        ]
    
    def get_low_rating_users(self, threshold: float = 2.0) -> List[str]:
        """
        Get users who rated the item below threshold.
        
        Args:
            threshold: Rating threshold
        
        Returns:
            List of user IDs
        """
        return [
            interaction['user_id']
            for interaction in self.rating_history
            if interaction.get('rating', 0) <= threshold
        ]
    
    def get_item_statistics(self) -> Dict[str, Any]:
        """
        Get item statistics.
        
        Returns:
            Dictionary of statistics
        """
        ratings = [r.get('rating', 0) for r in self.rating_history]
        rating_distribution = Counter(ratings)
        
        return {
            'item_id': self.item_id,
            'title': self.title,
            'category': self.category,
            'brand': self.brand,
            'price': self.price,
            'total_interactions': len(self.rating_history),
            'unique_users': len(set(self.user_history)),
            'avg_rating': self.average_rating,
            'num_ratings': self.num_ratings,
            'rating_distribution': dict(rating_distribution),
            'rating_std': np.std(ratings) if ratings else 0,
            'interaction_sparsity': 1 - (len(self.rating_history) / (len(set(self.user_history)) + 1)),
            'keywords': self.keywords[:10]
        }
    
    def calculate_item_similarity(self, other_item: 'ItemAgent') -> float:
        """
        Calculate similarity to another item.
        
        Args:
            other_item: Other ItemAgent instance
        
        Returns:
            Similarity score (0-1)
        """
        # Base similarity from embeddings
        base_similarity = self.calculate_similarity(other_item)
        
        # Enhance with content-based similarity
        content_similarity = self._calculate_content_similarity(other_item)
        
        # Combine similarities
        combined = 0.6 * base_similarity + 0.4 * content_similarity
        return min(1.0, combined)
    
    def _calculate_content_similarity(self, other_item: 'ItemAgent') -> float:
        """
        Calculate content-based similarity to another item.
        
        Args:
            other_item: Other ItemAgent instance
        
        Returns:
            Content similarity score (0-1)
        """
        # Category similarity
        category_sim = 1.0 if self.category == other_item.category else 0.0
        
        # Keyword overlap
        my_keywords = set(self.keywords)
        other_keywords = set(other_item.keywords)
        
        if my_keywords and other_keywords:
            intersection = len(my_keywords & other_keywords)
            union = len(my_keywords | other_keywords)
            keyword_sim = intersection / union if union > 0 else 0.0
        else:
            keyword_sim = 0.0
        
        # Brand similarity
        brand_sim = 1.0 if self.brand == other_item.brand else 0.0
        
        # Price similarity (normalized)
        if self.price > 0 and other_item.price > 0:
            price_diff = abs(self.price - other_item.price)
            price_sim = max(0.0, 1.0 - (price_diff / max(self.price, other_item.price)))
        else:
            price_sim = 0.0
        
        # Combine content similarities
        weights = {
            'category': 0.35,
            'keywords': 0.35,
            'brand': 0.15,
            'price': 0.15
        }
        
        content_sim = (
            weights['category'] * category_sim +
            weights['keywords'] * keyword_sim +
            weights['brand'] * brand_sim +
            weights['price'] * price_sim
        )
        
        return content_sim
    
    def get_similar_items(self, 
                         other_items: List['ItemAgent'],
                         top_k: int = 10) -> List[Tuple['ItemAgent', float]]:
        """
        Find similar items.
        
        Args:
            other_items: List of other ItemAgent instances
            top_k: Number of similar items to return
        
        Returns:
            List of (item, similarity) tuples
        """
        similarities = []
        
        for other_item in other_items:
            if other_item.item_id == self.item_id:
                continue
            
            similarity = self.calculate_item_similarity(other_item)
            similarities.append((other_item, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Update similar items tracking
        self.collaborative_similar_items = [i.item_id for i, _ in similarities[:top_k]]
        self.similarity_scores = {i.item_id: s for i, s in similarities[:top_k]}
        
        return similarities[:top_k]
    
    def update_collaborative_memory(self, interactions: List[Dict]) -> bool:
        """
        Update collaborative memory with interactions from users.
        
        Args:
            interactions: List of interactions from users
        
        Returns:
            True if successful
        """
        if not interactions:
            return False
        
        # Aggregate collaborative information
        user_ids = [i.get('user_id') for i in interactions]
        ratings = [i.get('rating', 0) for i in interactions]
        
        collaborative_data = {
            'users': user_ids,
            'ratings': ratings,
            'avg_rating': np.mean(ratings) if ratings else 0,
            'num_interactions': len(interactions),
            'timestamp': datetime.now().isoformat()
        }
        
        # Update average rating
        self._update_average_rating()
        
        # Clear collaborative embedding cache
        self._collaborative_embedding_cache = None
        
        return self.update_memory(MemoryType.COLLABORATIVE.value, collaborative_data)
    
    def get_collaborative_signature(self) -> Dict[str, Any]:
        """
        Get collaborative signature of the item.
        
        Returns:
            Collaborative signature dictionary
        """
        # Get user rating distribution
        ratings = [r.get('rating', 0) for r in self.rating_history]
        rating_dist = Counter(ratings)
        
        # Get top users by rating
        high_rating_users = self.get_high_rating_users()
        low_rating_users = self.get_low_rating_users()
        
        return {
            'item_id': self.item_id,
            'num_interactions': len(self.rating_history),
            'avg_rating': self.average_rating,
            'rating_distribution': dict(rating_dist),
            'num_high_ratings': len(high_rating_users),
            'num_low_ratings': len(low_rating_users),
            'top_users': high_rating_users[:10],
            'timestamp': datetime.now().isoformat()
        }
    
    def is_cold_start(self, threshold: int = 5) -> bool:
        """
        Check if item is cold-start (few interactions).
        
        Args:
            threshold: Interaction threshold
        
        Returns:
            True if cold-start
        """
        return len(self.rating_history) < threshold
    
    def get_popularity_score(self) -> float:
        """
        Get popularity score based on interactions.
        
        Returns:
            Popularity score (0-1)
        """
        if self.num_ratings == 0:
            return 0.0
        
        # Combine number of ratings and average rating
        max_ratings = 1000  # Assumed maximum for normalization
        rating_count_score = min(1.0, self.num_ratings / max_ratings)
        rating_score = self.average_rating / 5.0
        
        # Weighted combination
        popularity = 0.6 * rating_count_score + 0.4 * rating_score
        return min(1.0, popularity)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ItemAgent to dictionary.
        
        Returns:
            Dictionary representation
        """
        data = super().to_dict()
        data.update({
            'item_id': self.item_id,
            'title': self.title,
            'description': self.description,
            'category': self.category,
            'brand': self.brand,
            'price': self.price,
            'average_rating': self.average_rating,
            'num_ratings': self.num_ratings,
            'content_features': self.content_features,
            'keywords': self.keywords,
            'summary': self.summary,
            'rating_history': self.rating_history,
            'user_history': self.user_history,
            'interaction_count': self.interaction_count,
            'collaborative_similar_items': self.collaborative_similar_items,
            'similarity_scores': self.similarity_scores,
            'last_interaction_time': self.last_interaction_time.isoformat() if self.last_interaction_time else None
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ItemAgent':
        """
        Create ItemAgent from dictionary.
        
        Args:
            data: Dictionary representation
        
        Returns:
            ItemAgent instance
        """
        item_id = data.get('item_id', data.get('agent_id'))
        config = data.get('config', {})
        
        # Create agent
        agent = cls(item_id, config)
        
        # Restore attributes
        agent.title = data.get('title', '')
        agent.description = data.get('description', '')
        agent.category = data.get('category', '')
        agent.brand = data.get('brand', '')
        agent.price = data.get('price', 0.0)
        agent.average_rating = data.get('average_rating', 0.0)
        agent.num_ratings = data.get('num_ratings', 0)
        agent.content_features = data.get('content_features', {})
        agent.keywords = data.get('keywords', [])
        agent.summary = data.get('summary', '')
        agent.rating_history = data.get('rating_history', [])
        agent.user_history = data.get('user_history', [])
        agent.interaction_count = data.get('interaction_count', len(agent.rating_history))
        agent.collaborative_similar_items = data.get('collaborative_similar_items', [])
        agent.similarity_scores = data.get('similarity_scores', {})
        
        if 'last_interaction_time' in data and data['last_interaction_time']:
            agent.last_interaction_time = datetime.fromisoformat(data['last_interaction_time'])
        
        # Restore base attributes
        if 'state' in data:
            agent.state = AgentState.from_dict(data['state'])
        
        if 'memory' in data and agent.memory:
            agent.memory.from_dict(data['memory'])
        
        agent.version = data.get('version', 1)
        
        if 'metadata' in data:
            agent.metadata = data['metadata']
        
        # Clear caches
        agent._content_embedding = None
        agent._collaborative_embedding_cache = None
        
        return agent
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ItemAgent(id={self.item_id}, title={self.title[:30]}..., interactions={len(self.rating_history)})"


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'memory': {
            'intrinsic': {'immutable': True, 'max_size': 5},
            'collaborative': {'immutable': False, 'max_size': 20, 'propagation_threshold': 0.3},
            'interaction': {'immutable': False, 'max_size': 15, 'buffer_size': 10}
        }
    }
    
    # Create item agent
    item = ItemAgent('item_001', config)
    
    # Set content
    item.set_content({
        'title': 'The Great Gatsby',
        'description': 'A novel by F. Scott Fitzgerald about the American dream.',
        'category': 'Books',
        'brand': 'Penguin',
        'price': 14.99,
        'keywords': ['classic', 'novel', 'american', 'literature']
    })
    
    # Add interactions
    item.add_interaction({
        'user_id': 'user_001',
        'rating': 5,
        'review_text': 'Amazing book!'
    })
    
    item.add_interaction({
        'user_id': 'user_002',
        'rating': 4,
        'review_text': 'Great read'
    })
    
    item.add_interaction({
        'user_id': 'user_003',
        'rating': 3
    })
    
    # Get statistics
    print(f"Item: {item}")
    print(f"Stats: {item.get_item_statistics()}")
    print(f"Text representation: {item.get_text_representation()}")
    
    # Check if cold-start
    print(f"Is cold-start? {item.is_cold_start(threshold=5)}")
    
    # Get popularity
    print(f"Popularity: {item.get_popularity_score():.2f}")
    
    # Get high rating users
    high_users = item.get_high_rating_users()
    print(f"High rating users: {high_users}")
    
    # Get collaborative signature
    signature = item.get_collaborative_signature()
    print(f"Collaborative signature: {signature}")