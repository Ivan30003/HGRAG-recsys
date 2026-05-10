"""
User Agent Module
Implements user agents with hierarchical memory for the Hybrid-GraphRAG framework.
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
class UserProfile:
    """Additional user profile information beyond memory tiers."""
    
    # Activity metrics
    total_interactions: int = 0
    active_sessions: int = 0
    avg_session_duration: float = 0.0
    
    # Preference diversity
    category_diversity: float = 0.0
    exploration_ratio: float = 0.0  # New categories vs familiar
    
    # Temporal patterns
    preferred_times: List[str] = field(default_factory=list)
    seasonal_preferences: Dict[str, List[str]] = field(default_factory=dict)
    
    # Social features
    trust_network: Set[str] = field(default_factory=set)
    influence_score: float = 0.0
    
    # Feedback history
    explicit_ratings: Dict[str, float] = field(default_factory=dict)
    implicit_signals: Dict[str, List[str]] = field(default_factory=dict)


class UserAgent:
    """
    User agent with hierarchical memory structure.
    
    Memory Tiers:
    1. Intrinsic (frozen): Demographics, explicit preferences, constraints
    2. Collaborative (mutable): Learned preference patterns from interactions
    3. Interaction (volatile): Recent interaction buffer with explanations
    
    The agent supports both LLM-based reasoning and GNN-based efficient inference.
    """
    
    def __init__(self,
                 agent_id: str,
                 intrinsic_memory: IntrinsicMemory,
                 collaborative_memory: Optional[CollaborativeMemory] = None,
                 interaction_memory: Optional[InteractionMemory] = None,
                 profile: Optional[UserProfile] = None):
        """
        Initialize user agent.
        
        Args:
            agent_id: Unique agent identifier
            intrinsic_memory: Immutable intrinsic memory
            collaborative_memory: Optional pre-initialized collaborative memory
            interaction_memory: Optional pre-initialized interaction memory
            profile: Optional user profile data
        """
        self.agent_id = agent_id
        self.agent_type = 'user'
        
        # Hierarchical memory tiers
        self.intrinsic_memory = intrinsic_memory
        self.collaborative_memory = collaborative_memory or CollaborativeMemory(
            agent_id=agent_id,
            agent_type='user'
        )
        self.interaction_memory = interaction_memory or InteractionMemory(
            agent_id=agent_id
        )
        
        # Additional profile
        self.profile = profile or UserProfile()
        
        # Runtime state
        self.is_active = True
        self.last_active = datetime.now()
        self.session_count = 0
        
        # Embeddings cache (for GNN path)
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        # Optimization state
        self.reflection_history: List[Dict] = []
        self.preference_confidence: float = 0.5
        
        logger.debug(f"User agent {agent_id} initialized")
    
    def get_full_memory_text(self) -> str:
        """
        Get concatenated text representation of all memory tiers.
        Used for LLM prompting in Graph RAG path.
        
        Returns:
            Formatted text of complete agent memory
        """
        parts = []
        
        # Intrinsic memory (identity)
        parts.append("=== User Profile ===")
        parts.append(self.intrinsic_memory.to_prompt_text())
        
        # Collaborative memory (learned preferences)
        parts.append("\n=== Learned Preferences ===")
        parts.append(self.collaborative_memory.to_prompt_text())
        
        # Interaction memory (recent history)
        parts.append("\n=== Recent Activity ===")
        parts.append(self.interaction_memory.get_recent_context(5))
        
        # Confidence indicator
        parts.append(f"\n=== Confidence: {self.preference_confidence:.2f} ===")
        
        return "\n".join(parts)
    
    def get_memory_for_retrieval(self) -> str:
        """
        Get concise memory text optimized for Graph RAG retrieval.
        Focuses on preference patterns and recent interactions.
        
        Returns:
            Concise text for retrieval context
        """
        parts = []
        
        # Key preferences
        if self.collaborative_memory.preference_patterns:
            parts.append("Preferences: " + 
                        "; ".join(self.collaborative_memory.preference_patterns[:5]))
        
        if self.collaborative_memory.dislike_patterns:
            parts.append("Dislikes: " + 
                        "; ".join(self.collaborative_memory.dislike_patterns[:3]))
        
        # Recent accuracy
        recent_acc = self.interaction_memory.get_recent_accuracy(10)
        parts.append(f"Recent decision accuracy: {recent_acc:.2f}")
        
        return "\n".join(parts)
    
    def decide_on_item(self,
                       item_memory: str,
                       graph_context: Optional[str] = None,
                       use_llm: bool = True,
                       llm_client: Optional[Any] = None) -> Dict:
        """
        Make a decision about an item.
        
        Args:
            item_memory: Item agent's memory text
            graph_context: Optional graph-retrieved context
            use_llm: Whether to use LLM for decision (vs GNN)
            llm_client: LLM client for text-based decision
        
        Returns:
            Decision dictionary with choice, explanation, and confidence
        """
        if use_llm and llm_client:
            return self._llm_decision(item_memory, graph_context, llm_client)
        else:
            return self._heuristic_decision(item_memory, graph_context)
    
    def _llm_decision(self,
                      item_memory: str,
                      graph_context: Optional[str],
                      llm_client: Any) -> Dict:
        """
        Use LLM for decision making.
        
        Args:
            item_memory: Item description text
            graph_context: Graph context text
            llm_client: LLM client
        
        Returns:
            Decision dictionary
        """
        # Build prompt
        prompt_parts = [
            "You are a user with the following preferences and history:",
            self.get_full_memory_text(),
            "",
            "You are considering the following item:",
            item_memory
        ]
        
        if graph_context:
            prompt_parts.extend([
                "",
                "Additional context from similar users:",
                graph_context
            ])
        
        prompt_parts.extend([
            "",
            "Based on your preferences, would you interact with this item?",
            "Respond with JSON: {\"decision\": \"positive\" or \"negative\", "
            "\"confidence\": 0.0-1.0, \"explanation\": \"...\"}"
        ])
        
        prompt = "\n".join(prompt_parts)
        
        try:
            response = llm_client.generate(prompt, json_mode=True)
            
            # Parse JSON response
            import json
            result = json.loads(response.text)
            
            return {
                'decision': result.get('decision', 'negative'),
                'confidence': result.get('confidence', 0.5),
                'explanation': result.get('explanation', ''),
                'method': 'llm',
                'tokens_used': response.tokens_used,
                'cost': response.cost_estimate
            }
            
        except Exception as e:
            logger.warning(f"LLM decision failed: {e}")
            return self._heuristic_decision(item_memory, graph_context)
    
    def _heuristic_decision(self,
                            item_memory: str,
                            graph_context: Optional[str] = None) -> Dict:
        """
        Make heuristic decision without LLM (fallback or GNN path).
        
        Args:
            item_memory: Item description text
            graph_context: Optional graph context
        
        Returns:
            Decision dictionary
        """
        # Count matching preferences
        match_score = 0.0
        total_patterns = 0
        
        for pattern in self.collaborative_memory.preference_patterns:
            total_patterns += 1
            if pattern.lower() in item_memory.lower():
                match_score += 1.0
        
        # Count dislikes
        for dislike in self.collaborative_memory.dislike_patterns:
            total_patterns += 1
            if dislike.lower() in item_memory.lower():
                match_score -= 1.0
        
        # Normalize
        if total_patterns > 0:
            confidence = max(0.1, min(0.9, match_score / total_patterns * 0.5 + 0.5))
        else:
            confidence = 0.5
        
        decision = 'positive' if confidence > 0.4 else 'negative'
        
        return {
            'decision': decision,
            'confidence': confidence,
            'explanation': f'Heuristic match score: {match_score:.2f}',
            'method': 'heuristic'
        }
    
    def reflect_on_interaction(self,
                               item_id: str,
                               decision: Dict,
                               ground_truth: str,
                               llm_client: Optional[Any] = None) -> Dict:
        """
        Reflect on an interaction outcome and update memory.
        
        Args:
            item_id: ID of the item agent
            decision: Decision dictionary from decide_on_item
            ground_truth: Whether user actually interacted ('positive'/'negative')
            llm_client: Optional LLM client for deep reflection
        
        Returns:
            Reflection result dictionary
        """
        is_correct = (decision['decision'] == ground_truth)
        
        # Record interaction
        condensed_candidates = self.interaction_memory.add_interaction(
            partner_id=item_id,
            partner_type='item',
            decision=decision['decision'],
            is_correct=is_correct,
            explanation=decision.get('explanation', '')
        )
        
        # Update profile
        self.profile.total_interactions += 1
        self.last_active = datetime.now()
        
        # If incorrect, perform deep reflection
        reflection_result = {'is_correct': is_correct, 'memory_updated': False}
        
        if not is_correct and llm_client:
            reflection_result = self._deep_reflection(
                item_id, decision, ground_truth, llm_client
            )
        
        # Condense old interaction traces into collaborative memory
        if condensed_candidates:
            self._condense_interactions(condensed_candidates, llm_client)
        
        # Update confidence
        self._update_confidence()
        
        # Record in history
        self.reflection_history.append({
            'timestamp': datetime.now().isoformat(),
            'item_id': item_id,
            'decision': decision['decision'],
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'reflection': reflection_result
        })
        
        return reflection_result
    
    def _deep_reflection(self,
                         item_id: str,
                         decision: Dict,
                         ground_truth: str,
                         llm_client: Any) -> Dict:
        """
        Perform deep reflection using LLM to update memory.
        
        Args:
            item_id: Item agent ID
            decision: Incorrect decision
            ground_truth: Actual outcome
            llm_client: LLM client
        
        Returns:
            Reflection result
        """
        prompt = f"""
You are reflecting on an incorrect recommendation decision.

Your profile:
{self.get_full_memory_text()}

You made a {decision['decision']} decision but the ground truth was {ground_truth}.
Your explanation was: {decision.get('explanation', 'N/A')}

Please analyze:
1. Why did you make this incorrect choice?
2. What preference patterns should be updated?
3. What new patterns should be learned?

Respond with JSON:
{{
    "analysis": "...",
    "new_preferences": ["..."],
    "new_dislikes": ["..."],
    "remove_preferences": ["..."],
    "confidence_adjustment": 0.0
}}
"""
        
        try:
            response = llm_client.generate(prompt, json_mode=True)
            import json
            result = json.loads(response.text)
            
            # Apply updates to collaborative memory
            new_prefs = result.get('new_preferences', [])
            new_dislikes = result.get('new_dislikes', [])
            remove_prefs = result.get('remove_preferences', [])
            
            self.collaborative_memory.update_from_reflection(
                new_patterns=new_prefs,
                new_dislikes=new_dislikes,
                partner_id=item_id
            )
            
            # Remove outdated preferences
            for pref in remove_prefs:
                if pref in self.collaborative_memory.preference_patterns:
                    self.collaborative_memory.preference_patterns.remove(pref)
            
            # Adjust confidence
            conf_adj = result.get('confidence_adjustment', 0.0)
            self.preference_confidence = max(0.1, min(0.95, 
                self.preference_confidence + conf_adj))
            
            return {
                'is_correct': False,
                'memory_updated': True,
                'analysis': result.get('analysis', ''),
                'new_preferences': new_prefs,
                'new_dislikes': new_dislikes,
                'confidence_adjusted': conf_adj
            }
            
        except Exception as e:
            logger.warning(f"Deep reflection failed: {e}")
            return {'is_correct': False, 'memory_updated': False, 'error': str(e)}
    
    def _condense_interactions(self,
                                traces: List[InteractionTrace],
                                llm_client: Optional[Any] = None):
        """
        Condense old interaction traces into collaborative memory.
        
        Args:
            traces: Interaction traces to condense
            llm_client: Optional LLM client for summarization
        """
        if not traces:
            return
        
        # Extract patterns from traces
        positive_items = []
        negative_items = []
        
        for trace in traces:
            if trace.is_correct:
                if trace.decision == 'positive':
                    positive_items.append(f"Interacted with {trace.partner_id}")
                else:
                    negative_items.append(f"Avoided {trace.partner_id}")
        
        # Update collaborative patterns
        if positive_items:
            summary = f"Liked items similar to: {', '.join(positive_items[:3])}"
            if summary not in self.collaborative_memory.preference_patterns:
                self.collaborative_memory.preference_patterns.append(summary)
        
        if negative_items:
            summary = f"Disliked items similar to: {', '.join(negative_items[:3])}"
            if summary not in self.collaborative_memory.dislike_patterns:
                self.collaborative_memory.dislike_patterns.append(summary)
    
    def _update_confidence(self):
        """Update preference confidence based on interaction accuracy."""
        recent_acc = self.interaction_memory.get_recent_accuracy(20)
        total_int = self.interaction_memory.total_interactions
        
        # Blend recent accuracy with historical confidence
        if total_int > 10:
            weight_recent = min(0.5, total_int / 100)
            self.preference_confidence = (
                weight_recent * recent_acc +
                (1 - weight_recent) * self.preference_confidence
            )
        else:
            self.preference_confidence = max(0.3, recent_acc)
    
    def receive_neighborhood_update(self,
                                     neighbor_memories: List[CollaborativeMemory],
                                     weights: Optional[List[float]] = None):
        """
        Receive and fuse collaborative signals from similar users.
        Called during lazy graph propagation.
        
        Args:
            neighbor_memories: Collaborative memories of neighbor user agents
            weights: Optional importance weights for each neighbor
        """
        self.collaborative_memory.fuse_from_neighbors(neighbor_memories, weights)
        logger.debug(f"Agent {self.agent_id} received neighborhood update "
                    f"from {len(neighbor_memories)} neighbors")
    
    def get_preference_embedding(self) -> Optional[np.ndarray]:
        """
        Get preference embedding for GNN path.
        Uses cached value if available.
        
        Returns:
            Preference embedding array or None
        """
        if 'preference' in self._embedding_cache:
            return self._embedding_cache['preference']
        
        # Build text representation
        text = self.get_full_memory_text()
        
        # In production, this calls the embedding API
        # For now, return a deterministic hash-based embedding
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
        
        self._embedding_cache['preference'] = embedding
        return embedding
    
    def compute_similarity_to(self, other_agent: 'UserAgent') -> float:
        """
        Compute preference similarity with another user agent.
        
        Args:
            other_agent: Another UserAgent instance
        
        Returns:
            Cosine similarity in [0, 1]
        """
        emb1 = self.get_preference_embedding()
        emb2 = other_agent.get_preference_embedding()
        
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
        # Memory staleness
        if self.collaborative_memory.last_updated:
            hours_since = (datetime.now() - 
                          self.collaborative_memory.last_updated).total_seconds() / 3600
            staleness = 1.0 - np.exp(-0.1 * hours_since)
        else:
            staleness = 1.0
        
        return {
            'confidence': self.preference_confidence,
            'staleness': staleness,
            'interaction_count': self.profile.total_interactions,
            'recent_accuracy': self.interaction_memory.get_recent_accuracy(10),
            'pattern_count': len(self.collaborative_memory.preference_patterns),
            'dislike_count': len(self.collaborative_memory.dislike_patterns)
        }
    
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
                'exploration_ratio': self.profile.exploration_ratio,
                'influence_score': self.profile.influence_score
            },
            'preference_confidence': self.preference_confidence,
            'is_active': self.is_active,
            'last_active': self.last_active.isoformat(),
            'session_count': self.session_count,
            'reflection_history_len': len(self.reflection_history)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserAgent':
        """
        Create agent from dictionary.
        
        Args:
            data: Dictionary with agent data
        
        Returns:
            UserAgent instance
        """
        from hierarchy_memory_utils.intrinsic_memory import IntrinsicMemory
        
        # Reconstruct intrinsic memory
        int_data = data['intrinsic_memory']
        intrinsic = IntrinsicMemory(
            agent_id=int_data['agent_id'],
            agent_type=int_data['agent_type'],
            explicit_preferences=int_data.get('explicit_preferences', []),
            stated_constraints=int_data.get('stated_constraints', []),
            demographic_info=int_data.get('demographic_info')
        )
        
        # Reconstruct collaborative memory
        col_data = data['collaborative_memory']
        collaborative = CollaborativeMemory(
            agent_id=col_data['agent_id'],
            agent_type=col_data['agent_type']
        )
        collaborative.preference_patterns = col_data.get('preference_patterns', [])
        collaborative.dislike_patterns = col_data.get('dislike_patterns', [])
        collaborative.update_count = col_data.get('update_count', 0)
        collaborative.confidence_score = col_data.get('confidence_score', 0.5)
        
        # Reconstruct interaction memory
        interaction = InteractionMemory(
            agent_id=data['agent_id']
        )
        interaction.total_interactions = data.get('interaction_memory', {}).get(
            'total_interactions', 0)
        interaction.correct_interactions = data.get('interaction_memory', {}).get(
            'correct_interactions', 0)
        
        agent = cls(
            agent_id=data['agent_id'],
            intrinsic_memory=intrinsic,
            collaborative_memory=collaborative,
            interaction_memory=interaction
        )
        
        agent.preference_confidence = data.get('preference_confidence', 0.5)
        agent.is_active = data.get('is_active', True)
        
        return agent
    
    def __repr__(self) -> str:
        return (f"UserAgent(id={self.agent_id}, "
                f"interactions={self.profile.total_interactions}, "
                f"confidence={self.preference_confidence:.2f})")


class ColdStartUserAgent(UserAgent):
    """
    Specialized user agent for cold-start scenarios.
    Inherits from UserAgent with additional cold-start handling.
    """
    
    def __init__(self,
                 agent_id: str,
                 intrinsic_memory: IntrinsicMemory,
                 **kwargs):
        """
        Initialize cold-start user agent.
        
        Args:
            agent_id: Unique agent identifier
            intrinsic_memory: Intrinsic memory
            **kwargs: Additional arguments for UserAgent
        """
        super().__init__(agent_id, intrinsic_memory, **kwargs)
        self.is_cold_start = True
        self.warmup_progress = 0.0
        self.onboarding_completed = False
    
    def complete_onboarding(self, 
                            initial_preferences: List[str],
                            initial_dislikes: List[str]):
        """
        Complete onboarding with explicit preference elicitation.
        
        Args:
            initial_preferences: Explicitly stated preferences
            initial_dislikes: Explicitly stated dislikes
        """
        self.collaborative_memory.update_from_reflection(
            new_patterns=initial_preferences,
            new_dislikes=initial_dislikes,
            partner_id='onboarding'
        )
        self.onboarding_completed = True
        self.preference_confidence = 0.7
        logger.info(f"Cold-start user {self.agent_id} completed onboarding")
    
    def update_warmup_progress(self):
        """Update warmup progress based on interaction count."""
        min_interactions = 5
        max_interactions = 20
        
        progress = min(1.0, 
                      (self.profile.total_interactions - min_interactions) / 
                      (max_interactions - min_interactions))
        
        self.warmup_progress = max(0.0, progress)
        
        if self.warmup_progress >= 1.0:
            self.is_cold_start = False
    
    def get_gating_features(self) -> Dict[str, float]:
        """Get gating features with cold-start awareness."""
        features = super().get_gating_features()
        features['is_cold_start'] = 1.0 if self.is_cold_start else 0.0
        features['warmup_progress'] = self.warmup_progress
        return features