"""
Item Agent Module
Implements item agents with hierarchical memory for the Hybrid-GraphRAG framework.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import logging

from hierarchy_memory_utils.intrinsic_memory import IntrinsicMemory
from hierarchy_memory_utils.collaborative_memory import CollaborativeMemory
from hierarchy_memory_utils.interaction_memory import InteractionMemory, InteractionTrace

logger = logging.getLogger(__name__)


@dataclass
class ItemProfile:
    """Additional item profile information beyond memory tiers."""
    
    # Popularity metrics
    total_interactions: int = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    interaction_velocity: float = 0.0  # Interactions per day
    
    # User diversity
    unique_users: Set[str] = field(default_factory=set)
    user_diversity_score: float = 0.0
    
    # Temporal patterns
    peak_interaction_times: List[str] = field(default_factory=list)
    seasonal_relevance: Dict[str, float] = field(default_factory=dict)
    
    # Quality signals
    avg_rating: float = 0.0
    rating_variance: float = 0.0
    return_rate: float = 0.0
    
    # Market position
    category_rank: int = 0
    price_tier: str = 'medium'
    brand_affinity: Dict[str, float] = field(default_factory=dict)
    
    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    lifecycle_stage: str = 'growth'  # 'new', 'growth', 'mature', 'decline'


class ItemAgent:
    """
    Item agent with hierarchical memory structure.
    
    Memory Tiers:
    1. Intrinsic (frozen): Title, category, description, brand, specifications
    2. Collaborative (mutable): Aggregated preferences of users who interact
    3. Interaction (volatile): Recent interaction buffer with user feedback
    
    Items are "activated" as agents that can:
    - Learn about their audience from interactions
    - Propagate preferences to similar items (cold-start help)
    - Participate in collaborative reflection
    """
    
    def __init__(self,
                 agent_id: str,
                 intrinsic_memory: IntrinsicMemory,
                 collaborative_memory: Optional[CollaborativeMemory] = None,
                 interaction_memory: Optional[InteractionMemory] = None,
                 profile: Optional[ItemProfile] = None):
        """
        Initialize item agent.
        
        Args:
            agent_id: Unique agent identifier
            intrinsic_memory: Immutable intrinsic memory
            collaborative_memory: Optional pre-initialized collaborative memory
            interaction_memory: Optional pre-initialized interaction memory
            profile: Optional item profile data
        """
        self.agent_id = agent_id
        self.agent_type = 'item'
        
        # Hierarchical memory tiers
        self.intrinsic_memory = intrinsic_memory
        self.collaborative_memory = collaborative_memory or CollaborativeMemory(
            agent_id=agent_id,
            agent_type='item'
        )
        self.interaction_memory = interaction_memory or InteractionMemory(
            agent_id=agent_id
        )
        
        # Additional profile
        self.profile = profile or ItemProfile()
        
        # Runtime state
        self.is_active = True
        self.last_interaction = datetime.now()
        
        # Embeddings cache (for GNN path)
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        # Optimization state
        self.reflection_history: List[Dict] = []
        self.audience_clarity: float = 0.3  # How well we understand our audience
        
        # Cold-start support
        self.warmup_sources: List[str] = []  # Items that helped warm up this item
        
        logger.debug(f"Item agent {agent_id} initialized")
    
    def get_full_memory_text(self) -> str:
        """
        Get concatenated text representation of all memory tiers.
        Used for LLM prompting in Graph RAG path.
        
        Returns:
            Formatted text of complete item memory
        """
        parts = []
        
        # Intrinsic memory (identity)
        parts.append("=== Item Information ===")
        parts.append(self.intrinsic_memory.to_prompt_text())
        
        # Collaborative memory (audience understanding)
        parts.append("\n=== Audience Insights ===")
        parts.append(self.collaborative_memory.to_prompt_text())
        
        # Interaction memory (recent feedback)
        parts.append("\n=== Recent Feedback ===")
        parts.append(self.interaction_memory.get_recent_context(5))
        
        # Popularity signals
        if self.profile.total_interactions > 0:
            pos_ratio = self.profile.positive_interactions / max(1, self.profile.total_interactions)
            parts.append(f"\n=== Popularity: {self.profile.total_interactions} interactions "
                        f"({pos_ratio:.0%} positive) ===")
        
        return "\n".join(parts)
    
    def get_description_for_candidate(self) -> str:
        """
        Get concise description for candidate ranking.
        Optimized for being one of many items in a recommendation list.
        
        Returns:
            Concise item description
        """
        parts = []
        
        # Core identity
        if self.intrinsic_memory.title:
            parts.append(f"Title: {self.intrinsic_memory.title}")
        if self.intrinsic_memory.category:
            parts.append(f"Category: {self.intrinsic_memory.category}")
        
        # Key audience insights
        if self.collaborative_memory.common_user_traits:
            parts.append(f"Popular with: {', '.join(self.collaborative_memory.common_user_traits[:3])}")
        
        # Quality signal
        if self.profile.total_interactions >= 5:
            pos_ratio = self.profile.positive_interactions / max(1, self.profile.total_interactions)
            if pos_ratio > 0.7:
                parts.append("Highly rated by users")
            elif pos_ratio < 0.3:
                parts.append("Mixed user reception")
        
        return " | ".join(parts)
    
    def record_interaction(self,
                           user_id: str,
                           user_preferences: str,
                           decision: str,
                           is_correct: bool,
                           explanation: str = ""):
        """
        Record an interaction with a user agent.
        
        Args:
            user_id: ID of the interacting user agent
            user_preferences: User's preference description
            decision: 'positive' or 'negative'
            is_correct: Whether the interaction was correctly predicted
            explanation: Explanation of the interaction
        """
        # Add to interaction memory
        self.interaction_memory.add_interaction(
            partner_id=user_id,
            partner_type='user',
            decision=decision,
            is_correct=is_correct,
            explanation=explanation
        )
        
        # Update profile
        self.profile.total_interactions += 1
        if decision == 'positive':
            self.profile.positive_interactions += 1
        else:
            self.profile.negative_interactions += 1
        
        self.profile.unique_users.add(user_id)
        self.profile.user_diversity_score = len(self.profile.unique_users) / max(1, self.profile.total_interactions)
        
        self.last_interaction = datetime.now()
        
        # Update audience understanding
        self._update_audience_understanding(user_preferences, decision)
    
    def _update_audience_understanding(self, 
                                        user_preferences: str, 
                                        decision: str):
        """
        Update collaborative memory based on user interaction.
        
        Args:
            user_preferences: User preference text
            decision: 'positive' or 'negative'
        """
        # Extract keywords from user preferences
        keywords = self._extract_keywords(user_preferences)
        
        if decision == 'positive':
            # Add to common user traits
            for kw in keywords[:3]:
                if kw not in self.collaborative_memory.common_user_traits:
                    self.collaborative_memory.common_user_traits.append(kw)
            
            # Update preference patterns
            pattern = f"Appeals to users interested in: {', '.join(keywords[:3])}"
            if pattern not in self.collaborative_memory.preference_patterns:
                self.collaborative_memory.preference_patterns.append(pattern)
        else:
            # Update dislike patterns
            pattern = f"Less suitable for users interested in: {', '.join(keywords[:3])}"
            if pattern not in self.collaborative_memory.dislike_patterns:
                self.collaborative_memory.dislike_patterns.append(pattern)
        
        # Prune old patterns
        max_patterns = 20
        if len(self.collaborative_memory.preference_patterns) > max_patterns:
            self.collaborative_memory.preference_patterns = \
                self.collaborative_memory.preference_patterns[-max_patterns:]
        
        # Update audience clarity
        self.audience_clarity = min(1.0, self.audience_clarity + 0.01)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text for pattern matching.
        
        Args:
            text: Input text
        
        Returns:
            List of keywords
        """
        # Simplified keyword extraction
        # In production, use NLP techniques
        words = text.lower().split()
        
        # Filter common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                      'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]
        
        # Count frequency
        from collections import Counter
        freq = Counter(keywords)
        
        return [word for word, _ in freq.most_common(10)]
    
    def learn_from_similar_item(self,
                                 similar_item: 'ItemAgent',
                                 similarity_score: float):
        """
        Learn audience patterns from a similar item.
        Used for cold-start warmup via item-item interaction.
        
        Args:
            similar_item: Another ItemAgent with established audience
            similarity_score: Content similarity score
        """
        if similarity_score < 0.5:
            return  # Not similar enough
        
        # Only learn if we have less data
        if self.profile.total_interactions >= similar_item.profile.total_interactions:
            return
        
        logger.info(f"Item {self.agent_id} learning from similar item {similar_item.agent_id} "
                   f"(similarity: {similarity_score:.2f})")
        
        # Transfer audience patterns with similarity weighting
        weight = similarity_score * 0.7  # Discount the transfer
        
        # Transfer common user traits
        for trait in similar_item.collaborative_memory.common_user_traits[:5]:
            if trait not in self.collaborative_memory.common_user_traits:
                # Only add if it makes sense for our category
                if self._trait_matches_category(trait):
                    self.collaborative_memory.common_user_traits.append(trait)
        
        # Transfer preference patterns (adapted)
        for pattern in similar_item.collaborative_memory.preference_patterns[:3]:
            adapted_pattern = pattern.replace(
                similar_item.intrinsic_memory.title or '',
                self.intrinsic_memory.title or ''
            )
            if adapted_pattern not in self.collaborative_memory.preference_patterns:
                self.collaborative_memory.preference_patterns.append(adapted_pattern)
        
        # Record warmup source
        self.warmup_sources.append(similar_item.agent_id)
        
        # Boost audience clarity
        self.audience_clarity = max(self.audience_clarity, similarity_score * 0.5)
    
    def _trait_matches_category(self, trait: str) -> bool:
        """
        Check if a user trait is relevant to this item's category.
        
        Args:
            trait: User trait description
        
        Returns:
            Whether trait matches category
        """
        if not self.intrinsic_memory.category:
            return True  # Can't verify, accept
        
        category_words = set(self.intrinsic_memory.category.lower().split())
        trait_words = set(trait.lower().split())
        
        # Check for overlap
        overlap = category_words & trait_words
        return len(overlap) > 0
    
    def receive_reflection_update(self,
                                   user_preferences: str,
                                   reflection_insight: str):
        """
        Update memory based on collaborative reflection.
        
        Args:
            user_preferences: User agent's preference text
            reflection_insight: Insight from reflection process
        """
        # Parse insight for new understanding
        if 'appeals to' in reflection_insight.lower():
            trait = reflection_insight.split('appeals to')[-1].strip(' .')
            if trait not in self.collaborative_memory.common_user_traits:
                self.collaborative_memory.common_user_traits.append(trait)
        
        if 'not suitable for' in reflection_insight.lower():
            trait = reflection_insight.split('not suitable for')[-1].strip(' .')
            pattern = f"Not ideal for users seeking: {trait}"
            if pattern not in self.collaborative_memory.dislike_patterns:
                self.collaborative_memory.dislike_patterns.append(pattern)
        
        # Record reflection
        self.reflection_history.append({
            'timestamp': datetime.now().isoformat(),
            'insight': reflection_insight[:200]
        })
        
        logger.debug(f"Item {self.agent_id} received reflection update")
    
    def get_intrinsic_embedding(self) -> Optional[np.ndarray]:
        """
        Get intrinsic embedding for GNN path.
        
        Returns:
            Intrinsic embedding array
        """
        if 'intrinsic' in self._embedding_cache:
            return self._embedding_cache['intrinsic']
        
        # Build text from intrinsic features
        text = self.intrinsic_memory.to_prompt_text()
        
        # Generate deterministic embedding
        import hashlib
        hash_bytes = hashlib.sha256(text.encode()).digest()
        embedding = np.zeros(256)
        for i in range(min(256, len(hash_bytes) * 8)):
            byte_idx = i // 8
            bit_idx = i % 8
            if byte_idx < len(hash_bytes):
                bit = (hash_bytes[byte_idx] >> bit_idx) & 1
                embedding[i] = bit * 2.0 - 1.0
        
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        self._embedding_cache['intrinsic'] = embedding
        return embedding
    
    def get_collaborative_embedding(self) -> Optional[np.ndarray]:
        """
        Get collaborative embedding for GNN path.
        
        Returns:
            Collaborative embedding array
        """
        if 'collaborative' in self._embedding_cache:
            return self._embedding_cache['collaborative']
        
        # Build text from collaborative patterns
        text = self.collaborative_memory.to_prompt_text()
        
        # If no patterns yet, use intrinsic as fallback
        if not text or text == "No collaborative patterns learned yet.":
            return self.get_intrinsic_embedding()
        
        import hashlib
        hash_bytes = hashlib.sha256(text.encode()).digest()
        embedding = np.zeros(256)
        for i in range(min(256, len(hash_bytes) * 8)):
            byte_idx = i // 8
            bit_idx = i % 8
            if byte_idx < len(hash_bytes):
                bit = (hash_bytes[byte_idx] >> bit_idx) & 1
                embedding[i] = bit * 2.0 - 1.0
        
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        self._embedding_cache['collaborative'] = embedding
        return embedding
    
    def compute_similarity_to(self, other_item: 'ItemAgent') -> float:
        """
        Compute content similarity with another item agent.
        Uses intrinsic memory for stable comparison.
        
        Args:
            other_item: Another ItemAgent instance
        
        Returns:
            Cosine similarity in [0, 1]
        """
        emb1 = self.get_intrinsic_embedding()
        emb2 = other_item.get_intrinsic_embedding()
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        sim = np.dot(emb1, emb2)
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))
    
    def get_gating_features(self) -> Dict[str, float]:
        """
        Compute features for the adaptive gating mechanism.
        
        Returns:
            Dictionary of gating features
        """
        # Audience clarity as confidence proxy
        confidence = self.audience_clarity
        
        # Memory staleness
        if self.collaborative_memory.last_updated:
            hours_since = (datetime.now() - 
                          self.collaborative_memory.last_updated).total_seconds() / 3600
            staleness = 1.0 - np.exp(-0.1 * hours_since)
        else:
            staleness = 1.0
        
        # Popularity ratio
        if self.profile.total_interactions > 0:
            pos_ratio = self.profile.positive_interactions / self.profile.total_interactions
        else:
            pos_ratio = 0.5
        
        return {
            'audience_clarity': confidence,
            'staleness': staleness,
            'interaction_count': self.profile.total_interactions,
            'positive_ratio': pos_ratio,
            'user_diversity': self.profile.user_diversity_score,
            'pattern_count': len(self.collaborative_memory.preference_patterns),
            'is_cold_start': 1.0 if self.profile.total_interactions < 5 else 0.0,
            'warmup_sources': len(self.warmup_sources)
        }
    
    def is_cold_start(self) -> bool:
        """
        Check if this item is in cold-start state.
        
        Returns:
            True if item has fewer than 5 interactions
        """
        return self.profile.total_interactions < 5
    
    def get_lifecycle_stage(self) -> str:
        """
        Determine item lifecycle stage based on interaction patterns.
        
        Returns:
            Lifecycle stage string
        """
        if self.profile.total_interactions < 5:
            return 'new'
        elif self.profile.total_interactions < 50:
            return 'growth'
        elif self.profile.interaction_velocity > 1.0:
            return 'mature'
        else:
            return 'decline'
    
    def to_dict(self) -> Dict:
        """
        Serialize agent state to dictionary.
        
        Returns:
            Dictionary representation of agent
        """
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'intrinsic_memory': self.intrinsic_memory.to_dict(),
            'collaborative_memory': self.collaborative_memory.to_dict(),
            'interaction_memory': self.interaction_memory.to_dict(),
            'profile': {
                'total_interactions': self.profile.total_interactions,
                'positive_interactions': self.profile.positive_interactions,
                'negative_interactions': self.profile.negative_interactions,
                'user_diversity_score': self.profile.user_diversity_score,
                'avg_rating': self.profile.avg_rating,
                'lifecycle_stage': self.get_lifecycle_stage(),
                'category_rank': self.profile.category_rank,
                'price_tier': self.profile.price_tier
            },
            'audience_clarity': self.audience_clarity,
            'is_active': self.is_active,
            'is_cold_start': self.is_cold_start(),
            'warmup_sources': self.warmup_sources,
            'last_interaction': self.last_interaction.isoformat(),
            'reflection_history_len': len(self.reflection_history)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ItemAgent':
        """
        Create agent from dictionary.
        
        Args:
            data: Dictionary with agent data
        
        Returns:
            ItemAgent instance
        """
        # Reconstruct intrinsic memory
        int_data = data['intrinsic_memory']
        intrinsic = IntrinsicMemory(
            agent_id=int_data['agent_id'],
            agent_type=int_data['agent_type'],
            title=int_data.get('title', ''),
            category=int_data.get('category', ''),
            description=int_data.get('description', ''),
            brand=int_data.get('brand')
        )
        
        # Reconstruct collaborative memory
        col_data = data['collaborative_memory']
        collaborative = CollaborativeMemory(
            agent_id=col_data['agent_id'],
            agent_type=col_data['agent_type']
        )
        collaborative.preference_patterns = col_data.get('preference_patterns', [])
        collaborative.dislike_patterns = col_data.get('dislike_patterns', [])
        collaborative.common_user_traits = col_data.get('common_user_traits', [])
        collaborative.update_count = col_data.get('update_count', 0)
        collaborative.confidence_score = col_data.get('confidence_score', 0.5)
        
        # Reconstruct interaction memory
        interaction = InteractionMemory(agent_id=data['agent_id'])
        interaction.total_interactions = data.get('interaction_memory', {}).get(
            'total_interactions', 0)
        interaction.correct_interactions = data.get('interaction_memory', {}).get(
            'correct_interactions', 0)
        
        # Reconstruct profile
        profile_data = data.get('profile', {})
        profile = ItemProfile(
            total_interactions=profile_data.get('total_interactions', 0),
            positive_interactions=profile_data.get('positive_interactions', 0),
            negative_interactions=profile_data.get('negative_interactions', 0),
            user_diversity_score=profile_data.get('user_diversity_score', 0.0),
            avg_rating=profile_data.get('avg_rating', 0.0),
            price_tier=profile_data.get('price_tier', 'medium')
        )
        
        agent = cls(
            agent_id=data['agent_id'],
            intrinsic_memory=intrinsic,
            collaborative_memory=collaborative,
            interaction_memory=interaction,
            profile=profile
        )
        
        agent.audience_clarity = data.get('audience_clarity', 0.3)
        agent.is_active = data.get('is_active', True)
        agent.warmup_sources = data.get('warmup_sources', [])
        
        return agent
    
    def __repr__(self) -> str:
        return (f"ItemAgent(id={self.agent_id}, "
                f"title={self.intrinsic_memory.title}, "
                f"interactions={self.profile.total_interactions}, "
                f"stage={self.get_lifecycle_stage()})")


class ColdStartItemAgent(ItemAgent):
    """
    Specialized item agent for cold-start scenarios.
    Inherits from ItemAgent with additional warmup capabilities.
    """
    
    def __init__(self,
                 agent_id: str,
                 intrinsic_memory: IntrinsicMemory,
                 **kwargs):
        """
        Initialize cold-start item agent.
        
        Args:
            agent_id: Unique agent identifier
            intrinsic_memory: Intrinsic memory
            **kwargs: Additional arguments for ItemAgent
        """
        super().__init__(agent_id, intrinsic_memory, **kwargs)
        self.warmup_status = 'pending'  # 'pending', 'in_progress', 'warmed_up'
        self.warmup_started_at: Optional[datetime] = None
        self.potential_audiences: List[Dict] = []  # Predicted audiences
    
    def start_warmup(self, similar_items: List['ItemAgent']):
        """
        Start warmup process using similar items.
        
        Args:
            similar_items: List of similar ItemAgents with established audiences
        """
        self.warmup_status = 'in_progress'
        self.warmup_started_at = datetime.now()
        
        logger.info(f"Starting warmup for cold-start item {self.agent_id} "
                   f"using {len(similar_items)} similar items")
        
        for similar_item in similar_items[:5]:  # Top 5 most similar
            similarity = self.compute_similarity_to(similar_item)
            self.learn_from_similar_item(similar_item, similarity)
        
        # Predict potential audiences
        self._predict_audiences(similar_items)
        
        self.warmup_status = 'warmed_up' if self.audience_clarity > 0.5 else 'in_progress'
    
    def _predict_audiences(self, similar_items: List['ItemAgent']):
        """
        Predict potential audience segments based on similar items.
        
        Args:
            similar_items: List of similar ItemAgents
        """
        # Aggregate audience traits from similar items
        audience_traits = {}
        
        for item in similar_items:
            for trait in item.collaborative_memory.common_user_traits:
                if trait not in audience_traits:
                    audience_traits[trait] = 0
                audience_traits[trait] += 1
        
        # Sort by frequency
        sorted_traits = sorted(audience_traits.items(), 
                              key=lambda x: x[1], 
                              reverse=True)
        
        self.potential_audiences = [
            {'trait': trait, 'confidence': min(0.9, count / len(similar_items))}
            for trait, count in sorted_traits[:5]
        ]
        
        logger.debug(f"Predicted {len(self.potential_audiences)} potential audiences "
                    f"for cold-start item {self.agent_id}")
    
    def validate_warmup(self, first_interactions: List[Dict]) -> float:
        """
        Validate warmup predictions against first real interactions.
        
        Args:
            first_interactions: List of first user interactions
        
        Returns:
            Warmup accuracy score
        """
        if not self.potential_audiences or not first_interactions:
            return 0.0
        
        # Check if predicted audiences match actual users
        predicted_traits = {p['trait'] for p in self.potential_audiences}
        actual_traits = set()
        
        for interaction in first_interactions:
            if interaction.get('decision') == 'positive':
                user_prefs = interaction.get('user_preferences', '')
                keywords = self._extract_keywords(user_prefs)
                actual_traits.update(keywords)
        
        if not actual_traits:
            return 0.0
        
        # Compute overlap
        overlap = predicted_traits & actual_traits
        accuracy = len(overlap) / max(1, len(predicted_traits))
        
        logger.info(f"Warmup validation for {self.agent_id}: {accuracy:.2f} accuracy")
        return accuracy
    
    def get_warmup_summary(self) -> Dict:
        """
        Get summary of warmup process.
        
        Returns:
            Warmup summary dictionary
        """
        return {
            'agent_id': self.agent_id,
            'warmup_status': self.warmup_status,
            'audience_clarity': self.audience_clarity,
            'warmup_sources_count': len(self.warmup_sources),
            'potential_audiences': self.potential_audiences,
            'interactions_since_warmup': self.profile.total_interactions,
            'warmup_duration_hours': (
                (datetime.now() - self.warmup_started_at).total_seconds() / 3600
                if self.warmup_started_at else 0
            )
        }