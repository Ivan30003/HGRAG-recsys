"""
user_agent.py - User agent implementation for H-GRAGrecsys

This module provides the UserAgent class that represents users in the
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


class UserAgent(BaseAgent):
    """
    User agent representing a user in the recommendation system.
    
    This class extends BaseAgent with user-specific functionality including
    preference management, collaborative filtering, and interaction history.
    """
    
    def __init__(self, user_id: str, config: Dict[str, Any]):
        """
        Initialize UserAgent.
        
        Args:
            user_id: Unique identifier for the user
            config: Configuration dictionary
        """
        super().__init__(user_id, config)
        
        # User-specific attributes
        self.user_id = user_id
        self.preferences: Dict[str, Any] = {}
        self.rating_history: List[Dict] = []
        self.item_history: List[str] = []
        
        # Interaction tracking
        self.interaction_count = 0
        self.last_interaction_time: Optional[datetime] = None
        
        # Collaborative filtering data
        self.collaborative_neighbors: List[str] = []
        self.neighbor_scores: Dict[str, float] = {}
        
        # Preference embedding cache
        self._preference_embedding: Optional[np.ndarray] = None
        
        logger.info(f"Initialized UserAgent: {user_id}")
    
    def get_embedding(self, component_type: Optional[str] = None) -> np.ndarray:
        """
        Get embedding representation of the user.
        
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
            elif component_type == 'preference':
                return self.get_preference_embedding()
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
        
        # Intrinsic embedding
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
        Get embedding from intrinsic memory.
        
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
        collaborative = self.get_collaborative_memory()
        if collaborative and hasattr(collaborative, 'get_embedding'):
            return collaborative.get_embedding()
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
    
    def get_preference_embedding(self) -> np.ndarray:
        """
        Get embedding of user preferences.
        
        Returns:
            Preference embedding vector
        """
        if self._preference_embedding is not None:
            return self._preference_embedding
        
        # Build preference embedding from ratings and interactions
        if not self.rating_history:
            return np.zeros(128)
        
        # Get item embeddings from items in history
        item_embeddings = []
        ratings = []
        
        for interaction in self.rating_history[-50:]:  # Limit to recent 50
            item_id = interaction.get('item_id')
            rating = interaction.get('rating', 0)
            if item_id and rating > 0:
                # In practice, we would get item embedding from external source
                # For now, use random embedding as placeholder
                # This should be replaced with actual item embeddings
                embedding = np.random.randn(128) * 0.1
                item_embeddings.append(embedding * rating)
                ratings.append(rating)
        
        if not item_embeddings:
            return np.zeros(128)
        
        # Weighted average by rating
        weights = np.array(ratings) / sum(ratings)
        preference_embedding = np.average(item_embeddings, axis=0, weights=weights)
        
        # Normalize
        norm = np.linalg.norm(preference_embedding)
        if norm > 0:
            preference_embedding = preference_embedding / norm
        
        self._preference_embedding = preference_embedding
        return preference_embedding
    
    def get_text_representation(self) -> str:
        """
        Get text representation of the user.
        
        Returns:
            Text representation
        """
        parts = [f"User {self.user_id}"]
        
        # Add preference information
        if self.preferences:
            categories = self.preferences.get('preferred_categories', {})
            if categories:
                top_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
                parts.append(f"Prefers: {', '.join([cat for cat, _ in top_cats])}")
        
        # Add interaction statistics
        parts.append(f"Interactions: {len(self.rating_history)}")
        
        # Add average rating
        if self.rating_history:
            avg_rating = np.mean([r.get('rating', 0) for r in self.rating_history])
            parts.append(f"Avg rating: {avg_rating:.2f}")
        
        return " | ".join(parts)
    
    def set_preferences(self, preferences: Dict[str, Any]) -> None:
        """
        Set user preferences.
        
        Args:
            preferences: Preference dictionary
        """
        self.preferences = preferences
        self._preference_embedding = None  # Clear cache
        
        # Update intrinsic memory
        self.update_memory(MemoryType.INTRINSIC.value, preferences)
        
        logger.info(f"Updated preferences for user {self.user_id}")
    
    def add_interaction(self, interaction_data: Dict[str, Any]) -> bool:
        """
        Add an interaction to the user's history.
        
        Args:
            interaction_data: Interaction data (item_id, rating, timestamp, etc.)
        
        Returns:
            True if successful
        """
        # Extract data
        item_id = interaction_data.get('item_id')
        rating = interaction_data.get('rating', 0)
        timestamp = interaction_data.get('timestamp', datetime.now().isoformat())
        review_text = interaction_data.get('review_text', '')
        summary = interaction_data.get('summary', '')
        
        if not item_id:
            logger.warning(f"Missing item_id in interaction for user {self.user_id}")
            return False
        
        # Create interaction entry
        interaction = {
            'item_id': item_id,
            'rating': rating,
            'timestamp': timestamp,
            'review_text': review_text,
            'summary': summary
        }
        
        # Add to history
        self.rating_history.append(interaction)
        self.item_history.append(item_id)
        self.interaction_count += 1
        self.last_interaction_time = datetime.now()
        
        # Update interaction memory
        self.add_interaction_to_memory(interaction)
        
        # Update state
        self.state.interaction_count = self.interaction_count
        self.state.updated_at = datetime.now().isoformat()
        
        # Clear embedding cache
        self._preference_embedding = None
        
        logger.debug(f"Added interaction for user {self.user_id}: item {item_id}, rating {rating}")
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
    
    def get_rating_for_item(self, item_id: str) -> Optional[float]:
        """
        Get rating for a specific item.
        
        Args:
            item_id: Item identifier
        
        Returns:
            Rating or None if not found
        """
        for interaction in reversed(self.rating_history):
            if interaction.get('item_id') == item_id:
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
    
    def get_high_rating_items(self, threshold: float = 4.0) -> List[str]:
        """
        Get items rated above threshold.
        
        Args:
            threshold: Rating threshold
        
        Returns:
            List of item IDs
        """
        return [
            interaction['item_id']
            for interaction in self.rating_history
            if interaction.get('rating', 0) >= threshold
        ]
    
    def get_low_rating_items(self, threshold: float = 2.0) -> List[str]:
        """
        Get items rated below threshold.
        
        Args:
            threshold: Rating threshold
        
        Returns:
            List of item IDs
        """
        return [
            interaction['item_id']
            for interaction in self.rating_history
            if interaction.get('rating', 0) <= threshold
        ]
    
    def get_preferred_categories(self, top_k: int = 5) -> List[Tuple[str, int]]:
        """
        Get preferred categories.
        
        Args:
            top_k: Number of top categories to return
        
        Returns:
            List of (category, count) tuples
        """
        # This should be extracted from interactions or preferences
        categories = self.preferences.get('preferred_categories', {})
        sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        return sorted_cats[:top_k]
    
    def get_user_statistics(self) -> Dict[str, Any]:
        """
        Get user statistics.
        
        Returns:
            Dictionary of statistics
        """
        ratings = [r.get('rating', 0) for r in self.rating_history]
        
        return {
            'user_id': self.user_id,
            'total_interactions': len(self.rating_history),
            'unique_items': len(set(self.item_history)),
            'avg_rating': np.mean(ratings) if ratings else 0,
            'std_rating': np.std(ratings) if ratings else 0,
            'min_rating': min(ratings) if ratings else 0,
            'max_rating': max(ratings) if ratings else 0,
            'rating_distribution': Counter(ratings),
            'preferred_categories': self.get_preferred_categories(),
            'last_interaction': self.last_interaction_time.isoformat() if self.last_interaction_time else None,
            'interaction_sparsity': 1 - (len(self.rating_history) / (len(self.item_history) + 1))
        }
    
    def calculate_user_similarity(self, other_user: 'UserAgent') -> float:
        """
        Calculate similarity to another user.
        
        Args:
            other_user: Other UserAgent instance
        
        Returns:
            Similarity score (0-1)
        """
        # Use base similarity
        base_similarity = self.calculate_similarity(other_user)
        
        # Enhance with interaction-based similarity
        if self.rating_history and other_user.rating_history:
            # Jaccard similarity of interacted items
            my_items = set(self.item_history)
            other_items = set(other_user.item_history)
            
            intersection = len(my_items & other_items)
            union = len(my_items | other_items)
            
            if union > 0:
                jaccard = intersection / union
            else:
                jaccard = 0.0
            
            # Combine similarities
            combined = 0.7 * base_similarity + 0.3 * jaccard
            return min(1.0, combined)
        
        return base_similarity
    
    def get_collaborative_neighbors(self, 
                                   other_users: List['UserAgent'],
                                   top_k: int = 10) -> List[Tuple['UserAgent', float]]:
        """
        Find collaborative neighbors among other users.
        
        Args:
            other_users: List of other UserAgent instances
            top_k: Number of neighbors to return
        
        Returns:
            List of (user, similarity) tuples
        """
        similarities = []
        
        for other_user in other_users:
            if other_user.user_id == self.user_id:
                continue
            
            similarity = self.calculate_user_similarity(other_user)
            similarities.append((other_user, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Update neighbor tracking
        self.collaborative_neighbors = [u.user_id for u, _ in similarities[:top_k]]
        self.neighbor_scores = {u.user_id: s for u, s in similarities[:top_k]}
        
        return similarities[:top_k]
    
    def get_recommendations_from_neighbors(self,
                                          neighbor_users: List['UserAgent'],
                                          exclude_items: Optional[Set[str]] = None,
                                          top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get item recommendations from neighbors.
        
        Args:
            neighbor_users: List of neighbor UserAgent instances
            exclude_items: Items to exclude (e.g., already interacted)
            top_k: Number of recommendations
        
        Returns:
            List of (item_id, score) tuples
        """
        if exclude_items is None:
            exclude_items = set(self.item_history)
        
        # Aggregate item scores from neighbors
        item_scores = defaultdict(float)
        
        for neighbor in neighbor_users:
            # Calculate neighbor weight
            weight = self.neighbor_scores.get(neighbor.user_id, 0.1)
            
            # Get neighbor's high-rated items
            neighbor_items = neighbor.get_high_rating_items()
            
            for item_id in neighbor_items:
                if item_id not in exclude_items:
                    # Weighted by neighbor similarity
                    item_scores[item_id] += weight
        
        # Sort by score
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]
    
    def update_collaborative_memory(self, interactions: List[Dict]) -> bool:
        """
        Update collaborative memory with new interactions.
        
        Args:
            interactions: List of interactions from neighbors
        
        Returns:
            True if successful
        """
        if not interactions:
            return False
        
        # Aggregate collaborative information
        collaborative_data = {
            'items': [i.get('item_id') for i in interactions],
            'ratings': [i.get('rating', 0) for i in interactions],
            'timestamp': datetime.now().isoformat()
        }
        
        return self.update_memory(MemoryType.COLLABORATIVE.value, collaborative_data)
    
    def reflect_on_interaction(self, 
                              item_id: str,
                              outcome: Dict[str, Any],
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on an interaction outcome.
        
        Args:
            item_id: Item identifier
            outcome: Interaction outcome
            context: Context information
        
        Returns:
            Reflection results
        """
        reflection = {
            'user_id': self.user_id,
            'item_id': item_id,
            'timestamp': datetime.now().isoformat(),
            'outcome': outcome,
            'context': context,
            'reflection_text': self._generate_reflection(outcome, context)
        }
        
        # Update memory based on reflection
        self.update_memory(MemoryType.INTERACTION.value, reflection)
        
        logger.info(f"Reflection completed for user {self.user_id} on item {item_id}")
        return reflection
    
    def _generate_reflection(self, outcome: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Generate reflection text based on outcome.
        
        Args:
            outcome: Interaction outcome
            context: Context information
        
        Returns:
            Reflection text
        """
        rating = outcome.get('rating', 0)
        is_positive = rating >= 4.0
        
        if is_positive:
            return f"User {self.user_id} liked this item (rating: {rating}). This aligns with their preferences."
        else:
            return f"User {self.user_id} disliked this item (rating: {rating}). This may indicate a change in preferences."
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert UserAgent to dictionary.
        
        Returns:
            Dictionary representation
        """
        data = super().to_dict()
        data.update({
            'user_id': self.user_id,
            'preferences': self.preferences,
            'rating_history': self.rating_history,
            'item_history': self.item_history,
            'interaction_count': self.interaction_count,
            'collaborative_neighbors': self.collaborative_neighbors,
            'neighbor_scores': self.neighbor_scores,
            'last_interaction_time': self.last_interaction_time.isoformat() if self.last_interaction_time else None
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserAgent':
        """
        Create UserAgent from dictionary.
        
        Args:
            data: Dictionary representation
        
        Returns:
            UserAgent instance
        """
        user_id = data.get('user_id', data.get('agent_id'))
        config = data.get('config', {})
        
        # Create agent
        agent = cls(user_id, config)
        
        # Restore attributes
        agent.preferences = data.get('preferences', {})
        agent.rating_history = data.get('rating_history', [])
        agent.item_history = data.get('item_history', [])
        agent.interaction_count = data.get('interaction_count', len(agent.rating_history))
        agent.collaborative_neighbors = data.get('collaborative_neighbors', [])
        agent.neighbor_scores = data.get('neighbor_scores', {})
        
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
        
        return agent
    
    def __repr__(self) -> str:
        """String representation."""
        return f"UserAgent(id={self.user_id}, interactions={len(self.rating_history)})"


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
    
    # Create user agent
    user = UserAgent('user_001', config)
    
    # Set preferences
    user.set_preferences({
        'preferred_categories': {'books': 5, 'electronics': 3},
        'age': 25,
        'location': 'NYC'
    })
    
    # Add interactions
    user.add_interaction({
        'item_id': 'item_001',
        'rating': 5,
        'review_text': 'Great product!'
    })
    
    user.add_interaction({
        'item_id': 'item_002',
        'rating': 3,
        'review_text': 'Average product'
    })
    
    user.add_interaction({
        'item_id': 'item_003',
        'rating': 4
    })
    
    # Get statistics
    print(f"User: {user}")
    print(f"Stats: {user.get_user_statistics()}")
    print(f"Text representation: {user.get_text_representation()}")
    
    # Get high rated items
    high_items = user.get_high_rating_items()
    print(f"High rated items: {high_items}")
    
    # Get preferences
    pref_cats = user.get_preferred_categories()
    print(f"Preferred categories: {pref_cats}")