"""
Reflection Engine Module for H-GRAGrecsys

This module implements the reflection mechanism for agents, enabling them to
analyze interactions, generate insights, update memories, and provide
explanations for recommendations.
"""

import torch
import json
import re
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.llm.llm_interface import LLMInterface, LLMResponse
from models.llm.prompt_templates import PromptTemplates
from models.agent.user_agent import UserAgent
from models.agent.item_agent import ItemAgent
from models.agent.memory import AgentMemory
from utils.logger import Logger
from utils.config_loader import ConfigLoader


@dataclass
class ReflectionResult:
    """
    Result of a reflection process.
    
    Attributes:
        reflection_id: Unique identifier for the reflection
        timestamp: Timestamp of reflection
        user_id: User ID
        item_id: Item ID
        outcome: Outcome of the interaction
        preference_signals: Extracted preference signals
        item_assessment: Assessment of the item
        patterns: Identified patterns
        recommendations: Future recommendations
        confidence: Confidence score
        explanation: Generated explanation
        metadata: Additional metadata
    """
    reflection_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    user_id: str = ""
    item_id: str = ""
    outcome: str = ""
    preference_signals: List[Dict[str, Any]] = field(default_factory=list)
    item_assessment: Dict[str, Any] = field(default_factory=dict)
    patterns: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReflectionEngine:
    """
    Engine for generating and processing reflections.
    
    This class handles:
    - Reflecting on user-item interactions
    - Generating explanations for recommendations
    - Updating agent memories based on reflections
    - Verifying reflection quality
    - Aggregating multiple reflections
    - Generating insights from reflection history
    """
    
    def __init__(self, llm: LLMInterface, config: Dict[str, Any]):
        """
        Initialize the reflection engine.
        
        Args:
            llm: LLMInterface instance
            config: Configuration dictionary
        """
        self.llm = llm
        self.config = config
        self.logger = Logger.get_instance(log_dir='logs', name='reflection_engine')
        
        # Extract configuration
        reflection_config = config.get('model', {}).get('llm', {}).get('reflection', {})
        self.max_reflection_length = reflection_config.get('max_length', 500)
        self.min_confidence_threshold = reflection_config.get('min_confidence', 0.5)
        self.reflection_buffer_size = reflection_config.get('buffer_size', 100)
        self.aggregation_strategy = reflection_config.get('aggregation_strategy', 'weighted_average')
        
        # Initialize prompt templates
        self.prompt_templates = PromptTemplates(config)
        
        # Reflection storage
        self.reflection_history: Dict[str, List[ReflectionResult]] = defaultdict(list)
        self.reflection_cache: Dict[str, ReflectionResult] = {}
        
        # Statistics
        self.reflection_stats = {
            'total_reflections': 0,
            'successful_reflections': 0,
            'failed_reflections': 0,
            'average_confidence': 0.0,
            'reflections_by_type': defaultdict(int),
            'reflection_times': []
        }
        
        self.logger.log_info("Initialized ReflectionEngine")
    
    def reflect(self, user_agent: UserAgent, 
                item_agent: ItemAgent,
                outcome: Union[str, float],
                context: Dict[str, Any]) -> ReflectionResult:
        """
        Generate a reflection on a user-item interaction.
        
        Args:
            user_agent: UserAgent instance
            item_agent: ItemAgent instance
            outcome: Outcome of interaction (success/failure or rating)
            context: Interaction context
            
        Returns:
            ReflectionResult: Generated reflection
        """
        self.logger.log_info(f"Reflecting on interaction: user={user_agent.agent_id}, item={item_agent.agent_id}")
        
        start_time = datetime.now().timestamp()
        
        # Prepare user and item information
        user_info = self._prepare_user_info(user_agent)
        item_info = self._prepare_item_info(item_agent)
        
        # Get reflection prompt
        prompt = self.prompt_templates.get_reflection_prompt(
            user=user_info,
            item=item_info,
            context={
                'rating': outcome if isinstance(outcome, (int, float)) else None,
                'outcome': outcome if isinstance(outcome, str) else None,
                'context': context
            }
        )
        
        # Generate reflection
        try:
            response = self.llm.generate(
                prompt,
                max_tokens=self.max_reflection_length,
                temperature=0.7
            )
            
            # Parse response
            reflection = self._parse_reflection_response(
                response.content,
                user_agent.agent_id,
                item_agent.agent_id,
                outcome
            )
            
            # Add metadata
            reflection.reflection_id = self._generate_reflection_id()
            reflection.timestamp = datetime.now().isoformat()
            reflection.metadata['context'] = context
            reflection.metadata['response_tokens'] = response.tokens_used
            reflection.metadata['latency'] = response.latency
            
            # Verify reflection quality
            if not self.verify_reflection(reflection):
                self.logger.log_warning(f"Reflection verification failed for {reflection.reflection_id}")
                reflection.confidence *= 0.7  # Reduce confidence
            
            # Store reflection
            self._store_reflection(reflection, user_agent.agent_id)
            
            # Update statistics
            self.reflection_stats['total_reflections'] += 1
            self.reflection_stats['successful_reflections'] += 1
            self.reflection_stats['average_confidence'] = (
                (self.reflection_stats['average_confidence'] * 
                 (self.reflection_stats['successful_reflections'] - 1) + 
                 reflection.confidence) / self.reflection_stats['successful_reflections']
            )
            self.reflection_stats['reflection_times'].append(
                datetime.now().timestamp() - start_time
            )
            
            self.logger.log_info(f"Generated reflection {reflection.reflection_id} with confidence {reflection.confidence:.2f}")
            return reflection
            
        except Exception as e:
            self.logger.log_error(f"Reflection generation failed: {str(e)}")
            self.reflection_stats['failed_reflections'] += 1
            
            # Return a fallback reflection
            return self._create_fallback_reflection(
                user_agent.agent_id,
                item_agent.agent_id,
                outcome
            )
    
    def _prepare_user_info(self, user_agent: UserAgent) -> str:
        """
        Prepare user information for reflection prompt.
        
        Args:
            user_agent: UserAgent instance
            
        Returns:
            str: Formatted user information
        """
        info = []
        info.append(f"User ID: {user_agent.agent_id}")
        
        # Get preferences
        preferences = user_agent.get_preference_memory()
        if preferences:
            info.append(f"Preferences: {json.dumps(preferences, indent=2)}")
        
        # Get interaction history
        history = user_agent.get_recommendation_history()
        if history:
            info.append(f"Interaction History: {len(history)} interactions")
        
        # Get embedding (if available)
        embedding = user_agent.get_embedding()
        if embedding is not None:
            info.append(f"Embedding Dimension: {embedding.shape[0]}")
        
        return "\n".join(info)
    
    def _prepare_item_info(self, item_agent: ItemAgent) -> str:
        """
        Prepare item information for reflection prompt.
        
        Args:
            item_agent: ItemAgent instance
            
        Returns:
            str: Formatted item information
        """
        info = []
        info.append(f"Item ID: {item_agent.agent_id}")
        
        # Get metadata
        metadata = item_agent.get_item_metadata()
        if metadata:
            info.append(f"Metadata: {json.dumps(metadata, indent=2)}")
        
        # Get popularity
        popularity = item_agent.get_popularity_score()
        info.append(f"Popularity Score: {popularity:.2f}")
        
        # Get content embedding (if available)
        embedding = item_agent.get_content_embedding()
        if embedding is not None:
            info.append(f"Content Embedding Dimension: {embedding.shape[0]}")
        
        return "\n".join(info)
    
    def _parse_reflection_response(self, response: str,
                                  user_id: str,
                                  item_id: str,
                                  outcome: Union[str, float]) -> ReflectionResult:
        """
        Parse LLM response into ReflectionResult.
        
        Args:
            response: LLM response text
            user_id: User ID
            item_id: Item ID
            outcome: Interaction outcome
            
        Returns:
            ReflectionResult: Parsed reflection
        """
        result = ReflectionResult(
            user_id=user_id,
            item_id=item_id,
            outcome=str(outcome)
        )
        
        # Try to parse structured output
        try:
            # Extract sections using regex patterns
            patterns = {
                'preference_signals': r'(?:Preference Signal|Preferences?):\s*(.+?)(?=(?:Item Assessment|Item|Assessment|\d\.|$))',
                'item_assessment': r'(?:Item Assessment|Item|Assessment):\s*(.+?)(?=(?:Pattern|Patterns|\d\.|$))',
                'patterns': r'(?:Pattern|Patterns|Pattern Identification):\s*(.+?)(?=(?:Future Recommendations|Recommendations|Confidence|\d\.|$))',
                'recommendations': r'(?:Future Recommendations|Recommendations?):\s*(.+?)(?=(?:Confidence|Confidence Score|\d\.|$))',
                'confidence': r'(?:Confidence|Confidence Score):\s*([\d.]+)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    content = match.group(1).strip()
                    
                    if key == 'preference_signals':
                        result.preference_signals = self._parse_preference_signals(content)
                    elif key == 'item_assessment':
                        result.item_assessment = self._parse_item_assessment(content)
                    elif key == 'patterns':
                        result.patterns = self._parse_patterns(content)
                    elif key == 'recommendations':
                        result.recommendations = self._parse_recommendations(content)
                    elif key == 'confidence':
                        try:
                            result.confidence = float(content)
                        except:
                            result.confidence = 0.5
            
            # If confidence not found, try to infer
            if result.confidence == 0.0:
                confidence_match = re.search(r'(\d+\.?\d*)', response)
                if confidence_match:
                    result.confidence = min(1.0, float(confidence_match.group(1)))
                else:
                    result.confidence = 0.5
            
            # Store full explanation
            result.explanation = response[:500]  # Truncate
            
        except Exception as e:
            self.logger.log_warning(f"Failed to parse reflection response: {str(e)}")
            result.explanation = response[:200]
            result.confidence = 0.3
        
        return result
    
    def _parse_preference_signals(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse preference signals from text.
        
        Args:
            content: Text containing preference signals
            
        Returns:
            List[Dict[str, Any]]: Parsed preference signals
        """
        signals = []
        # Try to find bullet points or numbered lists
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('•') or re.match(r'^\d+\.', line):
                # Extract signal
                signal_text = re.sub(r'^[-\•\d+\.\s]+', '', line).strip()
                if signal_text:
                    signals.append({
                        'signal': signal_text,
                        'weight': 1.0
                    })
        
        if not signals:
            # If no bullet points, use the whole content
            signals.append({
                'signal': content[:200],
                'weight': 1.0
            })
        
        return signals
    
    def _parse_item_assessment(self, content: str) -> Dict[str, Any]:
        """
        Parse item assessment from text.
        
        Args:
            content: Text containing item assessment
            
        Returns:
            Dict[str, Any]: Parsed item assessment
        """
        assessment = {
            'description': content[:200],
            'score': 0.5
        }
        
        # Try to extract score
        score_match = re.search(r'(\d+\.?\d*)', content)
        if score_match:
            assessment['score'] = min(1.0, float(score_match.group(1)))
        
        return assessment
    
    def _parse_patterns(self, content: str) -> List[str]:
        """
        Parse patterns from text.
        
        Args:
            content: Text containing patterns
            
        Returns:
            List[str]: Parsed patterns
        """
        patterns = []
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('•') or re.match(r'^\d+\.', line):
                pattern = re.sub(r'^[-\•\d+\.\s]+', '', line).strip()
                if pattern:
                    patterns.append(pattern)
        
        if not patterns:
            patterns = [content[:200]]
        
        return patterns
    
    def _parse_recommendations(self, content: str) -> List[str]:
        """
        Parse recommendations from text.
        
        Args:
            content: Text containing recommendations
            
        Returns:
            List[str]: Parsed recommendations
        """
        recommendations = []
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('•') or re.match(r'^\d+\.', line):
                rec = re.sub(r'^[-\•\d+\.\s]+', '', line).strip()
                if rec:
                    recommendations.append(rec)
        
        if not recommendations:
            recommendations = [content[:200]]
        
        return recommendations
    
    def _generate_reflection_id(self) -> str:
        """
        Generate a unique reflection ID.
        
        Returns:
            str: Reflection ID
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"ref_{timestamp}_{hash(str(datetime.now())) % 10000:04d}"
    
    def _store_reflection(self, reflection: ReflectionResult, user_id: str) -> None:
        """
        Store a reflection in history.
        
        Args:
            reflection: ReflectionResult instance
            user_id: User ID
        """
        # Store in history
        self.reflection_history[user_id].append(reflection)
        
        # Limit buffer size
        if len(self.reflection_history[user_id]) > self.reflection_buffer_size:
            # Remove oldest reflections
            self.reflection_history[user_id] = self.reflection_history[user_id][-self.reflection_buffer_size:]
        
        # Store in cache
        self.reflection_cache[reflection.reflection_id] = reflection
    
    def _create_fallback_reflection(self, user_id: str, 
                                   item_id: str,
                                   outcome: Union[str, float]) -> ReflectionResult:
        """
        Create a fallback reflection when generation fails.
        
        Args:
            user_id: User ID
            item_id: Item ID
            outcome: Interaction outcome
            
        Returns:
            ReflectionResult: Fallback reflection
        """
        return ReflectionResult(
            reflection_id=self._generate_reflection_id(),
            user_id=user_id,
            item_id=item_id,
            outcome=str(outcome),
            preference_signals=[{'signal': f'Interaction with item {item_id}', 'weight': 0.5}],
            item_assessment={'description': f'Item {item_id}', 'score': 0.5},
            patterns=[f'Standard interaction pattern'],
            recommendations=[f'Consider similar items'],
            confidence=0.3,
            explanation=f'Fallback reflection for interaction between {user_id} and {item_id}',
            metadata={'fallback': True}
        )
    
    def update_memory(self, user_agent: UserAgent, 
                     item_agent: ItemAgent,
                     reflection: ReflectionResult) -> None:
        """
        Update agent memories based on reflection.
        
        Args:
            user_agent: UserAgent instance
            item_agent: ItemAgent instance
            reflection: ReflectionResult instance
        """
        self.logger.log_info(f"Updating memories from reflection {reflection.reflection_id}")
        
        # Update user memory
        if reflection.preference_signals:
            user_agent.update_preference({
                'reflection_id': reflection.reflection_id,
                'signals': reflection.preference_signals,
                'timestamp': reflection.timestamp,
                'confidence': reflection.confidence
            })
        
        # Update item memory
        if reflection.item_assessment:
            item_agent.update_collaborative_pattern(
                [reflection.user_id],
                {
                    'reflection_id': reflection.reflection_id,
                    'assessment': reflection.item_assessment,
                    'timestamp': reflection.timestamp,
                    'confidence': reflection.confidence
                }
            )
    
    def generate_explanation(self, user_agent: UserAgent,
                           item_agent: ItemAgent,
                           reflection: Optional[ReflectionResult] = None) -> str:
        """
        Generate an explanation for a recommendation.
        
        Args:
            user_agent: UserAgent instance
            item_agent: ItemAgent instance
            reflection: Optional reflection to use
            
        Returns:
            str: Generated explanation
        """
        self.logger.log_info(f"Generating explanation for user={user_agent.agent_id}, item={item_agent.agent_id}")
        
        # Prepare user and item information
        user_info = self._prepare_user_info(user_agent)
        item_info = self._prepare_item_info(item_agent)
        
        # Use reflection if provided
        reflection_info = ""
        if reflection:
            reflection_info = f"""
Reflection Information:
- Confidence: {reflection.confidence:.2f}
- Patterns: {', '.join(reflection.patterns)}
- Recommendations: {', '.join(reflection.recommendations)}
"""
        
        # Get explanation prompt
        prompt = self.prompt_templates.get_explanation_prompt(
            user=user_info,
            item=item_info,
            recommendation={
                'reason': 'Based on user preferences and item characteristics',
                'context': {
                    'reflection': reflection_info,
                    'user_history': len(user_agent.get_recommendation_history()),
                    'item_popularity': item_agent.get_popularity_score()
                },
                'audience': 'user',
                'tone': 'friendly',
                'length': 'medium',
                'key_points': ['relevance', 'quality', 'user_fit']
            }
        )
        
        try:
            response = self.llm.generate(
                prompt,
                max_tokens=300,
                temperature=0.6
            )
            
            explanation = response.content.strip()
            
            # If explanation is too long, truncate
            if len(explanation) > 500:
                explanation = explanation[:500] + "..."
            
            return explanation
            
        except Exception as e:
            self.logger.log_error(f"Explanation generation failed: {str(e)}")
            return f"Recommended based on your preferences and the quality of {item_agent.agent_id}."
    
    def verify_reflection(self, reflection: ReflectionResult) -> bool:
        """
        Verify the quality of a reflection.
        
        Args:
            reflection: ReflectionResult instance
            
        Returns:
            bool: True if reflection passes verification
        """
        # Check confidence threshold
        if reflection.confidence < self.min_confidence_threshold:
            return False
        
        # Check if essential fields are populated
        if not reflection.user_id or not reflection.item_id:
            return False
        
        if not reflection.preference_signals and not reflection.patterns:
            return False
        
        # Check for coherent structure
        if not reflection.explanation and not reflection.recommendations:
            return False
        
        return True
    
    def aggregate_reflections(self, reflections: List[ReflectionResult],
                            strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Aggregate multiple reflections into a summary.
        
        Args:
            reflections: List of ReflectionResult instances
            strategy: Aggregation strategy
            
        Returns:
            Dict[str, Any]: Aggregated reflection summary
        """
        strategy = strategy or self.aggregation_strategy
        
        if not reflections:
            return {}
        
        self.logger.log_info(f"Aggregating {len(reflections)} reflections using {strategy}")
        
        if strategy == 'weighted_average':
            return self._aggregate_weighted_average(reflections)
        elif strategy == 'consensus':
            return self._aggregate_consensus(reflections)
        elif strategy == 'recent':
            return self._aggregate_recent(reflections)
        else:
            return self._aggregate_simple(reflections)
    
    def _aggregate_weighted_average(self, reflections: List[ReflectionResult]) -> Dict[str, Any]:
        """
        Aggregate reflections using weighted average.
        
        Args:
            reflections: List of ReflectionResult instances
            
        Returns:
            Dict[str, Any]: Aggregated summary
        """
        total_weight = sum(r.confidence for r in reflections)
        if total_weight == 0:
            return self._aggregate_simple(reflections)
        
        # Aggregate preference signals
        all_signals = []
        for r in reflections:
            for signal in r.preference_signals:
                signal['weight'] = signal.get('weight', 1.0) * r.confidence
                all_signals.append(signal)
        
        # Sort signals by weight
        all_signals.sort(key=lambda x: x['weight'], reverse=True)
        
        # Aggregate patterns
        pattern_counts = defaultdict(int)
        for r in reflections:
            for pattern in r.patterns:
                pattern_counts[pattern] += 1
        
        # Calculate average confidence
        avg_confidence = sum(r.confidence for r in reflections) / len(reflections)
        
        return {
            'preference_signals': all_signals[:10],
            'patterns': sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'average_confidence': avg_confidence,
            'num_reflections': len(reflections),
            'strategy': 'weighted_average',
            'timestamp': datetime.now().isoformat()
        }
    
    def _aggregate_consensus(self, reflections: List[ReflectionResult]) -> Dict[str, Any]:
        """
        Aggregate reflections using consensus.
        
        Args:
            reflections: List of ReflectionResult instances
            
        Returns:
            Dict[str, Any]: Aggregated summary
        """
        # Find patterns that appear in most reflections
        pattern_counts = defaultdict(int)
        signal_counts = defaultdict(int)
        
        for r in reflections:
            for pattern in r.patterns:
                pattern_counts[pattern] += 1
            for signal in r.preference_signals:
                signal_counts[signal.get('signal', '')] += 1
        
        # Threshold for consensus (appears in > 50% of reflections)
        consensus_threshold = len(reflections) * 0.5
        
        consensus_patterns = [
            pattern for pattern, count in pattern_counts.items()
            if count >= consensus_threshold
        ]
        
        consensus_signals = [
            signal for signal, count in signal_counts.items()
            if count >= consensus_threshold
        ]
        
        return {
            'consensus_patterns': consensus_patterns,
            'consensus_signals': consensus_signals[:5],
            'total_reflections': len(reflections),
            'consensus_ratio': len(consensus_patterns) / len(pattern_counts) if pattern_counts else 0,
            'strategy': 'consensus',
            'timestamp': datetime.now().isoformat()
        }
    
    def _aggregate_recent(self, reflections: List[ReflectionResult]) -> Dict[str, Any]:
        """
        Aggregate reflections focusing on recent ones.
        
        Args:
            reflections: List of ReflectionResult instances
            
        Returns:
            Dict[str, Any]: Aggregated summary
        """
        # Sort by timestamp (most recent first)
        sorted_reflections = sorted(
            reflections,
            key=lambda x: x.timestamp,
            reverse=True
        )
        
        # Take last 5 reflections
        recent = sorted_reflections[:5]
        
        return {
            'recent_reflections': [
                {
                    'id': r.reflection_id,
                    'timestamp': r.timestamp,
                    'confidence': r.confidence,
                    'patterns': r.patterns[:3]
                }
                for r in recent
            ],
            'num_recent': len(recent),
            'total_reflections': len(reflections),
            'strategy': 'recent',
            'timestamp': datetime.now().isoformat()
        }
    
    def _aggregate_simple(self, reflections: List[ReflectionResult]) -> Dict[str, Any]:
        """
        Simple aggregation of reflections.
        
        Args:
            reflections: List of ReflectionResult instances
            
        Returns:
            Dict[str, Any]: Aggregated summary
        """
        all_patterns = []
        all_signals = []
        avg_confidence = 0
        
        for r in reflections:
            all_patterns.extend(r.patterns)
            all_signals.extend(r.preference_signals)
            avg_confidence += r.confidence
        
        avg_confidence /= len(reflections) if reflections else 1
        
        return {
            'patterns': all_patterns[:10],
            'preference_signals': all_signals[:5],
            'average_confidence': avg_confidence,
            'num_reflections': len(reflections),
            'strategy': 'simple',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_reflection_history(self, user_id: str, 
                              limit: int = 10) -> List[ReflectionResult]:
        """
        Get reflection history for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of reflections to return
            
        Returns:
            List[ReflectionResult]: Reflection history
        """
        reflections = self.reflection_history.get(user_id, [])
        return reflections[-limit:]
    
    def get_reflection_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get a summary of reflections for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dict[str, Any]: Reflection summary
        """
        reflections = self.reflection_history.get(user_id, [])
        if not reflections:
            return {
                'user_id': user_id,
                'total_reflections': 0,
                'message': 'No reflections available'
            }
        
        # Aggregate reflections
        summary = self.aggregate_reflections(
            reflections[-50:],  # Use recent 50 reflections
            strategy='weighted_average'
        )
        
        summary['user_id'] = user_id
        summary['total_reflections'] = len(reflections)
        summary['average_confidence'] = sum(r.confidence for r in reflections) / len(reflections)
        
        return summary
    
    def clear_history(self, user_id: Optional[str] = None) -> None:
        """
        Clear reflection history.
        
        Args:
            user_id: Specific user ID or None for all
        """
        if user_id:
            if user_id in self.reflection_history:
                self.reflection_history[user_id].clear()
                self.logger.log_info(f"Cleared reflection history for user {user_id}")
        else:
            self.reflection_history.clear()
            self.logger.log_info("Cleared all reflection history")
    
    def get_reflection_stats(self) -> Dict[str, Any]:
        """
        Get reflection statistics.
        
        Returns:
            Dict[str, Any]: Reflection statistics
        """
        stats = self.reflection_stats.copy()
        stats['total_users'] = len(self.reflection_history)
        stats['total_reflections_stored'] = sum(len(refs) for refs in self.reflection_history.values())
        stats['cache_size'] = len(self.reflection_cache)
        stats['average_reflection_time'] = (
            np.mean(self.reflection_stats['reflection_times']) 
            if self.reflection_stats['reflection_times'] else 0.0
        )
        stats['success_rate'] = (
            stats['successful_reflections'] / stats['total_reflections']
            if stats['total_reflections'] > 0 else 0.0
        )
        
        return stats
    
    def __str__(self) -> str:
        """String representation."""
        return (f"ReflectionEngine(reflections={self.reflection_stats['total_reflections']}, "
                f"users={len(self.reflection_history)})")


# Example usage
if __name__ == "__main__":
    # Load configuration
    config_path = "config/default_config.yaml"
    config_loader = ConfigLoader(config_path)
    config = config_loader.load_config()
    
    # Create LLM interface (mock for testing)
    from models.llm.llm_interface import LLMInterface
    llm = LLMInterface(config)
    
    # Create reflection engine
    reflection_engine = ReflectionEngine(llm, config)
    
    # Create mock agents
    user_agent = UserAgent('user_1', config)
    item_agent = ItemAgent('item_1', config)
    
    # Test reflection
    reflection = reflection_engine.reflect(
        user_agent,
        item_agent,
        outcome='success',
        context={
            'rating': 4.5,
            'timestamp': '2024-01-15T10:30:00',
            'session_id': 'session_123'
        }
    )
    
    print(f"Reflection ID: {reflection.reflection_id}")
    print(f"Confidence: {reflection.confidence:.2f}")
    print(f"Patterns: {reflection.patterns[:3]}")
    print(f"Explanation: {reflection.explanation[:100]}...")
    
    # Update memory
    reflection_engine.update_memory(user_agent, item_agent, reflection)
    
    # Generate explanation
    explanation = reflection_engine.generate_explanation(user_agent, item_agent, reflection)
    print(f"\nExplanation: {explanation}")
    
    # Get reflection summary
    summary = reflection_engine.get_reflection_summary('user_1')
    print(f"\nReflection Summary: {summary}")
    
    # Get statistics
    stats = reflection_engine.get_reflection_stats()
    print(f"\nReflection Statistics: {stats}")