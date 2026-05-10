"""
Interaction Memory Module
Buffers recent interaction traces with explanations.
Periodically condensed into collaborative memory.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import math


@dataclass
class InteractionTrace:
    """Single interaction record with metadata."""
    timestamp: datetime
    partner_id: str
    partner_type: str  # 'user' or 'item'
    decision: str  # 'positive' or 'negative'
    is_correct: bool  # Did agent choose correctly?
    explanation: str
    graph_context_used: Optional[Dict] = None
    influential_paths: Optional[List[str]] = None
    
    def get_age_hours(self) -> float:
        """Get age of this trace in hours."""
        delta = datetime.now() - self.timestamp
        return delta.total_seconds() / 3600.0


@dataclass
class InteractionMemory:
    """
    Volatile memory tier that buffers recent interactions.
    Provides short-term context for agent decisions.
    """
    
    agent_id: str
    max_buffer_size: int = 10
    decay_lambda: float = 0.1  # Decay rate for retention probability
    
    # Storage
    interaction_traces: List[InteractionTrace] = field(default_factory=list)
    
    # Summary statistics
    total_interactions: int = 0
    correct_interactions: int = 0
    accuracy_history: List[float] = field(default_factory=list)
    
    def add_interaction(self, 
                        partner_id: str,
                        partner_type: str,
                        decision: str,
                        is_correct: bool,
                        explanation: str,
                        graph_context: Optional[Dict] = None,
                        influential_paths: Optional[List[str]] = None):
        """
        Add a new interaction trace to the buffer.
        
        Args:
            partner_id: ID of the interaction partner
            partner_type: 'user' or 'item'
            decision: 'positive' or 'negative'
            is_correct: Whether the agent's decision matched ground truth
            explanation: Natural language explanation of the decision
            graph_context: Graph context used for this decision
            influential_paths: Metapaths that most influenced the decision
        """
        trace = InteractionTrace(
            timestamp=datetime.now(),
            partner_id=partner_id,
            partner_type=partner_type,
            decision=decision,
            is_correct=is_correct,
            explanation=explanation,
            graph_context_used=graph_context,
            influential_paths=influential_paths
        )
        
        self.interaction_traces.append(trace)
        self.total_interactions += 1
        
        if is_correct:
            self.correct_interactions += 1
        
        # Update accuracy history
        current_accuracy = self.correct_interactions / self.total_interactions
        self.accuracy_history.append(current_accuracy)
        
        # Keep only recent history
        if len(self.accuracy_history) > 100:
            self.accuracy_history = self.accuracy_history[-50:]
        
        # Prune old traces based on decay
        self._prune_old_traces()
        
        # Trigger condensation if buffer is full
        if len(self.interaction_traces) > self.max_buffer_size:
            return self._get_condensation_candidates()
        
        return []
    
    def _prune_old_traces(self):
        """Remove traces with low retention probability."""
        retained = []
        for trace in self.interaction_traces:
            age_hours = trace.get_age_hours()
            retention_prob = math.exp(-self.decay_lambda * age_hours)
            
            # Always keep correct interactions slightly longer
            if trace.is_correct:
                retention_prob *= 1.5
            
            if retention_prob > 0.1:  # Keep if probability > 10%
                retained.append(trace)
        
        self.interaction_traces = retained
    
    def _get_condensation_candidates(self) -> List[InteractionTrace]:
        """
        Identify traces that should be condensed into collaborative memory.
        Returns the oldest traces that exceed buffer capacity.
        """
        if len(self.interaction_traces) <= self.max_buffer_size:
            return []
        
        # Sort by age (oldest first)
        sorted_traces = sorted(self.interaction_traces, 
                              key=lambda t: t.timestamp)
        
        # Return traces beyond buffer capacity
        num_to_condense = len(self.interaction_traces) - self.max_buffer_size
        candidates = sorted_traces[:num_to_condense]
        
        # Remove condensed traces from buffer
        self.interaction_traces = sorted_traces[num_to_condense:]
        
        return candidates
    
    def get_recent_context(self, n: int = 5) -> str:
        """
        Get text summary of recent interactions for LLM prompts.
        
        Args:
            n: Number of most recent interactions to include
        
        Returns:
            Formatted text of recent interaction history
        """
        if not self.interaction_traces:
            return "No recent interactions."
        
        recent = sorted(self.interaction_traces, 
                       key=lambda t: t.timestamp, 
                       reverse=True)[:n]
        
        parts = ["Recent interactions:"]
        for i, trace in enumerate(recent):
            outcome = "✓" if trace.is_correct else "✗"
            parts.append(
                f"  {i+1}. [{outcome}] With {trace.partner_type} {trace.partner_id}: "
                f"{trace.decision} - {trace.explanation[:100]}..."
            )
        
        return "\n".join(parts)
    
    def get_correct_ratio(self) -> float:
        """Get ratio of correct interactions."""
        if self.total_interactions == 0:
            return 0.5
        return self.correct_interactions / self.total_interactions
    
    def get_recent_accuracy(self, window: int = 10) -> float:
        """
        Get accuracy over recent window of interactions.
        
        Args:
            window: Number of recent interactions to consider
        
        Returns:
            Accuracy in recent window
        """
        if len(self.accuracy_history) < window:
            return self.get_correct_ratio()
        
        recent = self.accuracy_history[-window:]
        return sum(recent) / len(recent)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'agent_id': self.agent_id,
            'total_interactions': self.total_interactions,
            'correct_interactions': self.correct_interactions,
            'current_accuracy': self.get_correct_ratio(),
            'buffer_size': len(self.interaction_traces),
            'recent_context': self.get_recent_context(3)
        }