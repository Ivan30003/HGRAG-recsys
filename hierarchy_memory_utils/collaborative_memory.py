"""
Collaborative Memory Module
Handles learned preference patterns from agent interactions.
This is the primary locus of optimization in the framework.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from datetime import datetime
import numpy as np


@dataclass
class CollaborativeMemory:
    """
    Mutable memory tier that captures collaborative patterns.
    Updated through agent interactions and neighborhood propagation.
    """
    
    agent_id: str
    agent_type: str  # 'user' or 'item'
    
    # For item agents: "who likes this item and why"
    # For user agents: "what types of items does this user prefer"
    preference_patterns: List[str] = field(default_factory=list)
    dislike_patterns: List[str] = field(default_factory=list)
    
    # Aggregated characteristics
    common_user_traits: List[str] = field(default_factory=list)  # items only
    preferred_categories: List[str] = field(default_factory=list)  # users only
    
    # Numerical representation (updated via distillation or encoding)
    text_embedding: Optional[List[float]] = None
    
    # Metadata
    update_count: int = 0
    last_updated: Optional[datetime] = None
    interaction_partners: Set[str] = field(default_factory=set)
    confidence_score: float = 0.5  # How confident we are in this memory
    
    def update_from_reflection(self, 
                                new_patterns: List[str], 
                                new_dislikes: List[str],
                                partner_id: str):
        """
        Update collaborative memory based on reflection output.
        
        Args:
            new_patterns: Newly discovered preference patterns
            new_dislikes: Newly discovered dislikes
            partner_id: ID of the interaction partner (user/item)
        """
        # Add new patterns, avoiding duplicates
        for pattern in new_patterns:
            if pattern not in self.preference_patterns:
                self.preference_patterns.append(pattern)
        
        for dislike in new_dislikes:
            if dislike not in self.dislike_patterns:
                self.dislike_patterns.append(dislike)
        
        # Prune old patterns if exceeding capacity
        max_patterns = 20
        if len(self.preference_patterns) > max_patterns:
            # Keep most recent patterns (they're appended at the end)
            self.preference_patterns = self.preference_patterns[-max_patterns:]
        
        if len(self.dislike_patterns) > max_patterns:
            self.dislike_patterns = self.dislike_patterns[-max_patterns:]
        
        # Update metadata
        self.update_count += 1
        self.last_updated = datetime.now()
        self.interaction_partners.add(partner_id)
        self.confidence_score = min(1.0, self.confidence_score + 0.02)
    
    def fuse_from_neighbors(self, neighbor_memories: List['CollaborativeMemory'], 
                           weights: Optional[List[float]] = None):
        """
        Fuse collaborative signals from neighbor agents.
        Used during neighborhood propagation.
        
        Args:
            neighbor_memories: List of collaborative memories from neighbors
            weights: Optional importance weights for each neighbor
        """
        if not neighbor_memories:
            return
        
        if weights is None:
            weights = [1.0 / len(neighbor_memories)] * len(neighbor_memories)
        
        # Collect patterns from neighbors
        all_patterns = []
        all_dislikes = []
        
        for memory, weight in zip(neighbor_memories, weights):
            # Weighted sampling: more important neighbors contribute more
            n_patterns = max(1, int(len(memory.preference_patterns) * weight))
            all_patterns.extend(memory.preference_patterns[:n_patterns])
            
            n_dislikes = max(1, int(len(memory.dislike_patterns) * weight))
            all_dislikes.extend(memory.dislike_patterns[:n_dislikes])
        
        # Merge with current patterns
        self.update_from_reflection(all_patterns, all_dislikes, 'neighborhood')
    
    def set_embedding(self, embedding: List[float]):
        """Update the numerical representation."""
        self.text_embedding = embedding
    
    def to_prompt_text(self) -> str:
        """Generate text representation for LLM prompts."""
        parts = []
        
        if self.preference_patterns:
            parts.append("Preferences: " + "; ".join(self.preference_patterns[:5]))
        
        if self.dislike_patterns:
            parts.append("Dislikes: " + "; ".join(self.dislike_patterns[:5]))
        
        if self.common_user_traits:
            parts.append("Common user traits: " + "; ".join(self.common_user_traits[:3]))
        
        if self.preferred_categories:
            parts.append("Preferred categories: " + ", ".join(self.preferred_categories[:5]))
        
        return "\n".join(parts) if parts else "No collaborative patterns learned yet."
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'preference_patterns': self.preference_patterns,
            'dislike_patterns': self.dislike_patterns,
            'common_user_traits': self.common_user_traits,
            'preferred_categories': self.preferred_categories,
            'update_count': self.update_count,
            'last_updated': str(self.last_updated) if self.last_updated else None,
            'interaction_partners': list(self.interaction_partners),
            'confidence_score': self.confidence_score
        }
    
    def get_entropy(self) -> float:
        """
        Calculate uncertainty of this memory.
        Higher entropy = less confident = more likely to need LLM path.
        """
        if not self.preference_patterns:
            return 1.0
        
        # More patterns with high confidence = lower entropy
        diversity = min(1.0, len(set(self.preference_patterns)) / 20.0)
        return 1.0 - (self.confidence_score * (1.0 - diversity * 0.5))
