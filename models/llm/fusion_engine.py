"""
Fusion Engine Module for H-GRAGrecsys

This module implements the fusion mechanism for combining multiple sources of
information including collaborative memories, user preferences, and item patterns
into unified representations for recommendation and reasoning.
"""

import torch
import numpy as np
import json
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
from models.agent.memory import AgentMemory, HierarchicalMemory
from models.agent.memory_components import MemoryComponent, IntrinsicMemory, CollaborativeMemory, InteractionMemory
from utils.logger import Logger
from utils.config_loader import ConfigLoader


@dataclass
class FusionResult:
    """
    Result of a fusion operation.
    
    Attributes:
        fusion_id: Unique identifier for the fusion
        timestamp: Timestamp of fusion
        source_type: Type of sources fused ('memories', 'preferences', 'patterns')
        fused_representation: The unified representation
        insights: Key insights from fusion
        confidence: Confidence score
        metadata: Additional metadata
    """
    fusion_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source_type: str = ""
    fused_representation: Dict[str, Any] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class FusionEngine:
    """
    Engine for fusing multiple sources of information.
    
    This class handles:
    - Fusing collaborative memories from multiple agents
    - Fusing user preferences from various signals
    - Fusing item patterns from interactions
    - Generating collaborative memory for agents
    - Evaluating fusion quality
    - Multiple fusion strategies
    """
    
    def __init__(self, llm: LLMInterface, config: Dict[str, Any]):
        """
        Initialize the fusion engine.
        
        Args:
            llm: LLMInterface instance
            config: Configuration dictionary
        """
        self.llm = llm
        self.config = config
        self.logger = Logger.get_instance(log_dir='logs', name='fusion_engine')
        
        # Extract configuration
        fusion_config = config.get('model', {}).get('llm', {}).get('fusion', {})
        self.max_fusion_length = fusion_config.get('max_length', 500)
        self.min_confidence_threshold = fusion_config.get('min_confidence', 0.5)
        self.fusion_strategy = fusion_config.get('strategy', 'weighted')
        self.weight_alpha = fusion_config.get('weight_alpha', 0.7)
        self.weight_beta = fusion_config.get('weight_beta', 0.3)
        self.similarity_threshold = fusion_config.get('similarity_threshold', 0.5)
        
        # Initialize prompt templates
        self.prompt_templates = PromptTemplates(config)
        
        # Fusion cache
        self.fusion_cache: Dict[str, FusionResult] = {}
        self.cache_size = fusion_config.get('cache_size', 100)
        
        # Statistics
        self.fusion_stats = {
            'total_fusions': 0,
            'successful_fusions': 0,
            'failed_fusions': 0,
            'fusions_by_type': defaultdict(int),
            'average_confidence': 0.0,
            'fusion_times': []
        }
        
        self.logger.log_info(f"Initialized FusionEngine with strategy={self.fusion_strategy}")
    
    def fuse_collaborative_memories(self, memories: List[Dict[str, Any]], 
                                   instruction: str,
                                   source_ids: Optional[List[str]] = None) -> FusionResult:
        """
        Fuse multiple collaborative memories into a unified representation.
        
        Args:
            memories: List of memory dictionaries
            instruction: Fusion instruction/goal
            source_ids: Optional list of source agent IDs
            
        Returns:
            FusionResult: Fused representation
        """
        self.logger.log_info(f"Fusing {len(memories)} collaborative memories")
        
        start_time = datetime.now().timestamp()
        
        # Generate fusion ID
        fusion_id = self._generate_fusion_id()
        
        # Prepare memories for fusion
        memory_text = self._prepare_memories_for_fusion(memories)
        memory_types = [m.get('type', 'unknown') for m in memories]
        weights = [m.get('weight', 1.0) for m in memories]
        
        # Get fusion prompt
        prompt = self.prompt_templates.get_fusion_prompt(
            memories=memory_text,
            instruction=f"{instruction}\n\nMemory Types: {', '.join(memory_types)}\nWeights: {weights}"
        )
        
        try:
            response = self.llm.generate(
                prompt,
                max_tokens=self.max_fusion_length,
                temperature=0.6
            )
            
            # Parse fusion response
            fused = self._parse_fusion_response(
                response.content,
                'memories',
                source_ids
            )
            
            # Add metadata
            fused.fusion_id = fusion_id
            fused.timestamp = datetime.now().isoformat()
            fused.metadata['num_memories'] = len(memories)
            fused.metadata['memory_types'] = memory_types
            fused.metadata['weights'] = weights
            fused.metadata['response_tokens'] = response.tokens_used
            fused.metadata['latency'] = response.latency
            
            # Store in cache
            self._cache_fusion(fusion_id, fused)
            
            # Update statistics
            self._update_statistics(fused, 'memories', start_time)
            
            self.logger.log_info(f"Fused {len(memories)} memories with confidence {fused.confidence:.2f}")
            return fused
            
        except Exception as e:
            self.logger.log_error(f"Memory fusion failed: {str(e)}")
            self.fusion_stats['failed_fusions'] += 1
            return self._create_fallback_fusion('memories', memories, source_ids)
    
    def fuse_user_preferences(self, preferences: Dict[str, Any],
                             source_info: Optional[Dict[str, Any]] = None) -> FusionResult:
        """
        Fuse user preferences from various sources.
        
        Args:
            preferences: Dictionary containing preference data
            source_info: Additional source information
            
        Returns:
            FusionResult: Fused preferences
        """
        self.logger.log_info("Fusing user preferences")
        
        start_time = datetime.now().timestamp()
        
        # Prepare preference data
        explicit = preferences.get('explicit', {})
        implicit = preferences.get('implicit', {})
        contextual = preferences.get('contextual', {})
        historical = preferences.get('historical', {})
        
        # Get fusion prompt
        prompt = self.prompt_templates.get_fusion_prompt(
            memories=[
                {'type': 'explicit', 'content': json.dumps(explicit, indent=2)},
                {'type': 'implicit', 'content': json.dumps(implicit, indent=2)},
                {'type': 'contextual', 'content': json.dumps(contextual, indent=2)},
                {'type': 'historical', 'content': json.dumps(historical, indent=2)}
            ],
            instruction="Fuse user preferences from all sources into a unified profile"
        )
        
        try:
            response = self.llm.generate(
                prompt,
                max_tokens=self.max_fusion_length,
                temperature=0.5
            )
            
            # Parse fusion response
            fused = self._parse_fusion_response(
                response.content,
                'preferences',
                source_info
            )
            
            # Add structured preference data
            fused.fused_representation['explicit'] = explicit
            fused.fused_representation['implicit'] = implicit
            fused.fused_representation['contextual'] = contextual
            fused.fused_representation['historical'] = historical
            
            # Generate fusion ID
            fused.fusion_id = self._generate_fusion_id()
            fused.timestamp = datetime.now().isoformat()
            fused.metadata['response_tokens'] = response.tokens_used
            fused.metadata['latency'] = response.latency
            
            # Store in cache
            self._cache_fusion(fused.fusion_id, fused)
            
            # Update statistics
            self._update_statistics(fused, 'preferences', start_time)
            
            self.logger.log_info(f"Fused user preferences with confidence {fused.confidence:.2f}")
            return fused
            
        except Exception as e:
            self.logger.log_error(f"Preference fusion failed: {str(e)}")
            self.fusion_stats['failed_fusions'] += 1
            return self._create_fallback_fusion('preferences', [preferences], [source_info])
    
    def fuse_item_patterns(self, patterns: List[Dict[str, Any]],
                          item_id: Optional[str] = None) -> FusionResult:
        """
        Fuse item patterns from interactions.
        
        Args:
            patterns: List of interaction patterns
            item_id: Optional item ID
            
        Returns:
            FusionResult: Fused patterns
        """
        self.logger.log_info(f"Fusing {len(patterns)} item patterns for item {item_id}")
        
        start_time = datetime.now().timestamp()
        
        # Prepare patterns for fusion
        pattern_text = self._prepare_patterns_for_fusion(patterns)
        
        # Get fusion prompt
        prompt = self.prompt_templates.get_fusion_prompt(
            memories=pattern_text,
            instruction=f"Fuse item patterns for item {item_id if item_id else 'unknown'} into a unified representation"
        )
        
        try:
            response = self.llm.generate(
                prompt,
                max_tokens=self.max_fusion_length,
                temperature=0.6
            )
            
            # Parse fusion response
            fused = self._parse_fusion_response(
                response.content,
                'patterns',
                {'item_id': item_id}
            )
            
            # Add pattern data
            fused.fused_representation['patterns'] = patterns
            fused.fused_representation['item_id'] = item_id
            
            # Generate fusion ID
            fused.fusion_id = self._generate_fusion_id()
            fused.timestamp = datetime.now().isoformat()
            fused.metadata['num_patterns'] = len(patterns)
            fused.metadata['response_tokens'] = response.tokens_used
            fused.metadata['latency'] = response.latency
            
            # Store in cache
            self._cache_fusion(fused.fusion_id, fused)
            
            # Update statistics
            self._update_statistics(fused, 'patterns', start_time)
            
            self.logger.log_info(f"Fused {len(patterns)} patterns with confidence {fused.confidence:.2f}")
            return fused
            
        except Exception as e:
            self.logger.log_error(f"Pattern fusion failed: {str(e)}")
            self.fusion_stats['failed_fusions'] += 1
            return self._create_fallback_fusion('patterns', patterns, {'item_id': item_id})
    
    def generate_collaborative_memory(self, agent: Any,
                                     neighbors: List[Any],
                                     max_neighbors: int = 5) -> Dict[str, Any]:
        """
        Generate a collaborative memory for an agent from its neighbors.
        
        Args:
            agent: UserAgent or ItemAgent instance
            neighbors: List of neighboring agents
            max_neighbors: Maximum neighbors to use
            
        Returns:
            Dict[str, Any]: Collaborative memory
        """
        self.logger.log_info(f"Generating collaborative memory for {agent.agent_id}")
        
        # Select top neighbors
        selected_neighbors = neighbors[:max_neighbors]
        
        if not selected_neighbors:
            return {
                'type': 'collaborative',
                'content': 'No collaborative information available',
                'confidence': 0.0
            }
        
        # Extract neighbor information
        neighbor_info = []
        for neighbor in selected_neighbors:
            if hasattr(neighbor, 'get_embedding'):
                embedding = neighbor.get_embedding()
                info = {
                    'id': neighbor.agent_id,
                    'type': type(neighbor).__name__,
                    'has_embedding': embedding is not None
                }
                
                # Get preferences if available
                if hasattr(neighbor, 'get_preference_memory'):
                    info['preferences'] = neighbor.get_preference_memory()
                
                neighbor_info.append(info)
        
        # Prepare fusion prompt
        prompt = f"""
Generate a collaborative memory for agent {agent.agent_id} based on these neighbors:

Neighbors:
{json.dumps(neighbor_info, indent=2)}

Please provide:
1. Collaborative Memory: A unified representation of the collaborative information
2. Key Patterns: What patterns emerge from the neighbors?
3. Confidence Score: How confident are you in this collaborative memory? (0-1)
"""
        
        try:
            response = self.llm.generate(
                prompt,
                max_tokens=300,
                temperature=0.6
            )
            
            # Parse response
            collaborative_memory = {
                'type': 'collaborative',
                'content': response.content,
                'confidence': 0.7,  # Default
                'neighbors_used': len(selected_neighbors),
                'timestamp': datetime.now().isoformat()
            }
            
            # Extract confidence from response
            confidence_match = re.search(r'Confidence Score:\s*([\d.]+)', response.content)
            if confidence_match:
                collaborative_memory['confidence'] = min(1.0, float(confidence_match.group(1)))
            
            return collaborative_memory
            
        except Exception as e:
            self.logger.log_error(f"Collaborative memory generation failed: {str(e)}")
            return {
                'type': 'collaborative',
                'content': 'Generated from neighbors',
                'confidence': 0.3,
                'neighbors_used': len(selected_neighbors)
            }
    
    def _prepare_memories_for_fusion(self, memories: List[Dict[str, Any]]) -> str:
        """
        Prepare memories for fusion prompt.
        
        Args:
            memories: List of memory dictionaries
            
        Returns:
            str: Formatted memory text
        """
        formatted = []
        for i, memory in enumerate(memories):
            formatted.append(f"Memory {i+1}:")
            formatted.append(f"Type: {memory.get('type', 'unknown')}")
            formatted.append(f"Content: {json.dumps(memory.get('content', ''), indent=2)}")
            formatted.append("---")
        
        return "\n".join(formatted)
    
    def _prepare_patterns_for_fusion(self, patterns: List[Dict[str, Any]]) -> str:
        """
        Prepare patterns for fusion prompt.
        
        Args:
            patterns: List of pattern dictionaries
            
        Returns:
            str: Formatted pattern text
        """
        formatted = []
        for i, pattern in enumerate(patterns):
            formatted.append(f"Pattern {i+1}:")
            formatted.append(json.dumps(pattern, indent=2))
            formatted.append("---")
        
        return "\n".join(formatted)
    
    def _parse_fusion_response(self, response: str,
                              source_type: str,
                              source_ids: Optional[Any] = None) -> FusionResult:
        """
        Parse LLM response into FusionResult.
        
        Args:
            response: LLM response text
            source_type: Type of sources fused
            source_ids: Source identifiers
            
        Returns:
            FusionResult: Parsed fusion result
        """
        result = FusionResult(
            source_type=source_type,
            fused_representation={},
            insights=[],
            confidence=0.5
        )
        
        if source_ids:
            result.metadata['source_ids'] = source_ids
        
        # Try to parse structured output
        try:
            # Extract insights
            insights_match = re.search(r'(?:Key Insights|Insights?):\s*(.+?)(?=(?:Confidence|\d\.|$))', 
                                      response, re.IGNORECASE | re.DOTALL)
            if insights_match:
                insights_text = insights_match.group(1).strip()
                result.insights = self._parse_insights(insights_text)
            
            # Extract confidence
            confidence_match = re.search(r'(?:Confidence|Confidence Score):\s*([\d.]+)', 
                                        response, re.IGNORECASE)
            if confidence_match:
                result.confidence = min(1.0, float(confidence_match.group(1)))
            else:
                # Try to infer confidence from response
                result.confidence = self._infer_confidence(response)
            
            # Extract key representations
            representations_match = re.search(r'(?:Fused Representation|Representation|Summary):\s*(.+?)(?=(?:Key Insights|Confidence|\d\.|$))', 
                                             response, re.IGNORECASE | re.DOTALL)
            if representations_match:
                result.fused_representation['summary'] = representations_match.group(1).strip()
            else:
                result.fused_representation['summary'] = response[:200]
            
            # Try to extract structured data
            # Look for JSON-like structures
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    result.fused_representation.update(data)
                except:
                    pass
            
        except Exception as e:
            self.logger.log_warning(f"Failed to parse fusion response: {str(e)}")
            result.fused_representation['summary'] = response[:200]
            result.confidence = 0.4
        
        # Ensure confidence is in [0, 1]
        result.confidence = max(0.0, min(1.0, result.confidence))
        
        return result
    
    def _parse_insights(self, insights_text: str) -> List[str]:
        """
        Parse insights from text.
        
        Args:
            insights_text: Text containing insights
            
        Returns:
            List[str]: Parsed insights
        """
        insights = []
        lines = insights_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('•') or re.match(r'^\d+\.', line):
                insight = re.sub(r'^[-\•\d+\.\s]+', '', line).strip()
                if insight:
                    insights.append(insight)
        
        if not insights:
            insights = [insights_text[:200]]
        
        return insights
    
    def _infer_confidence(self, response: str) -> float:
        """
        Infer confidence score from response text.
        
        Args:
            response: Response text
            
        Returns:
            float: Inferred confidence
        """
        # Look for confidence indicators
        positive_indicators = ['certain', 'confident', 'clear', 'definitely', 'strong']
        negative_indicators = ['uncertain', 'unclear', 'maybe', 'perhaps', 'possibly']
        
        response_lower = response.lower()
        positive_score = sum(1 for word in positive_indicators if word in response_lower)
        negative_score = sum(1 for word in negative_indicators if word in response_lower)
        
        if positive_score + negative_score > 0:
            confidence = positive_score / (positive_score + negative_score)
        else:
            confidence = 0.5
        
        return min(1.0, max(0.0, confidence))
    
    def _generate_fusion_id(self) -> str:
        """
        Generate a unique fusion ID.
        
        Returns:
            str: Fusion ID
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"fusion_{timestamp}_{hash(str(datetime.now())) % 10000:04d}"
    
    def _cache_fusion(self, fusion_id: str, fusion: FusionResult) -> None:
        """
        Cache a fusion result.
        
        Args:
            fusion_id: Fusion ID
            fusion: FusionResult instance
        """
        if len(self.fusion_cache) >= self.cache_size:
            # Remove oldest entry
            self.fusion_cache.pop(next(iter(self.fusion_cache)))
        
        self.fusion_cache[fusion_id] = fusion
    
    def _update_statistics(self, fusion: FusionResult, 
                          fusion_type: str, start_time: float) -> None:
        """
        Update fusion statistics.
        
        Args:
            fusion: FusionResult instance
            fusion_type: Type of fusion
            start_time: Start time of fusion
        """
        self.fusion_stats['total_fusions'] += 1
        self.fusion_stats['successful_fusions'] += 1
        self.fusion_stats['fusions_by_type'][fusion_type] += 1
        self.fusion_stats['average_confidence'] = (
            (self.fusion_stats['average_confidence'] * 
             (self.fusion_stats['successful_fusions'] - 1) + 
             fusion.confidence) / self.fusion_stats['successful_fusions']
        )
        self.fusion_stats['fusion_times'].append(
            datetime.now().timestamp() - start_time
        )
    
    def _create_fallback_fusion(self, source_type: str,
                               sources: List[Any],
                               source_ids: Optional[Any] = None) -> FusionResult:
        """
        Create a fallback fusion when generation fails.
        
        Args:
            source_type: Type of sources
            sources: Source data
            source_ids: Source identifiers
            
        Returns:
            FusionResult: Fallback fusion
        """
        return FusionResult(
            fusion_id=self._generate_fusion_id(),
            source_type=source_type,
            fused_representation={
                'summary': f'Fallback fusion for {len(sources)} sources',
                'sources': str(sources)[:200]
            },
            insights=['Fallback fusion due to processing error'],
            confidence=0.2,
            metadata={
                'fallback': True,
                'source_ids': source_ids,
                'num_sources': len(sources)
            }
        )
    
    def evaluate_fusion_quality(self, original: Dict[str, Any],
                               fused: Dict[str, Any]) -> float:
        """
        Evaluate the quality of a fusion by comparing with original.
        
        Args:
            original: Original representation
            fused: Fused representation
            
        Returns:
            float: Quality score (0-1)
        """
        # Check if both are dictionaries
        if not isinstance(original, dict) or not isinstance(fused, dict):
            return 0.0
        
        # Calculate coverage: how much of original is represented in fused
        original_keys = set(original.keys())
        fused_keys = set(fused.keys())
        overlap = len(original_keys & fused_keys) / len(original_keys) if original_keys else 0
        
        # Calculate structural similarity
        if overlap < self.similarity_threshold:
            return overlap
        
        # Calculate content similarity (simplified)
        # If both have summary, compare length and content
        if 'summary' in original and 'summary' in fused:
            orig_summary = str(original['summary'])
            fused_summary = str(fused['summary'])
            # Jaccard similarity of words
            orig_words = set(orig_summary.lower().split())
            fused_words = set(fused_summary.lower().split())
            word_sim = len(orig_words & fused_words) / len(orig_words | fused_words) if (orig_words | fused_words) else 0
        else:
            word_sim = 0.5
        
        # Combine scores
        quality = 0.6 * overlap + 0.4 * word_sim
        
        return min(1.0, quality)
    
    def get_fusion_by_id(self, fusion_id: str) -> Optional[FusionResult]:
        """
        Get a fusion result by ID.
        
        Args:
            fusion_id: Fusion ID
            
        Returns:
            Optional[FusionResult]: Fusion result or None
        """
        return self.fusion_cache.get(fusion_id)
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """
        Get fusion statistics.
        
        Returns:
            Dict[str, Any]: Fusion statistics
        """
        stats = self.fusion_stats.copy()
        stats['cache_size'] = len(self.fusion_cache)
        stats['success_rate'] = (
            stats['successful_fusions'] / stats['total_fusions']
            if stats['total_fusions'] > 0 else 0.0
        )
        stats['average_fusion_time'] = (
            np.mean(self.fusion_stats['fusion_times']) 
            if self.fusion_stats['fusion_times'] else 0.0
        )
        stats['recent_fusion_times'] = self.fusion_stats['fusion_times'][-10:]
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the fusion cache."""
        self.fusion_cache.clear()
        self.logger.log_info("Cleared fusion cache")
    
    def reset_statistics(self) -> None:
        """Reset fusion statistics."""
        self.fusion_stats = {
            'total_fusions': 0,
            'successful_fusions': 0,
            'failed_fusions': 0,
            'fusions_by_type': defaultdict(int),
            'average_confidence': 0.0,
            'fusion_times': []
        }
        self.logger.log_info("Reset fusion statistics")
    
    def __str__(self) -> str:
        """String representation."""
        return (f"FusionEngine(fusions={self.fusion_stats['total_fusions']}, "
                f"cache={len(self.fusion_cache)}, "
                f"success_rate={self.fusion_stats['successful_fusions']/self.fusion_stats['total_fusions']:.2f})")


# Example usage
if __name__ == "__main__":
    # Load configuration
    config_path = "config/default_config.yaml"
    config_loader = ConfigLoader(config_path)
    config = config_loader.load_config()
    
    # Create LLM interface
    from models.llm.llm_interface import LLMInterface
    llm = LLMInterface(config)
    
    # Create fusion engine
    fusion_engine = FusionEngine(llm, config)
    
    # Test memory fusion
    memories = [
        {'type': 'preference', 'content': 'User likes action movies', 'weight': 0.8},
        {'type': 'interaction', 'content': 'User watched 5 action movies in the last month', 'weight': 0.9},
        {'type': 'collaborative', 'content': 'Similar users liked sci-fi movies', 'weight': 0.6}
    ]
    
    result = fusion_engine.fuse_collaborative_memories(
        memories,
        instruction="Fuse memories to understand user preferences",
        source_ids=['user_1', 'user_2']
    )
    
    print(f"Fusion ID: {result.fusion_id}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Insights: {result.insights[:3]}")
    print(f"Summary: {result.fused_representation.get('summary', '')[:100]}...")
    
    # Test preference fusion
    preferences = {
        'explicit': {'genres': ['action', 'sci-fi'], 'rating': 4.5},
        'implicit': {'watch_time': 120, 'completion_rate': 0.8},
        'contextual': {'time_of_day': 'evening', 'day_of_week': 'weekend'},
        'historical': ['Movie A', 'Movie B', 'Movie C']
    }
    
    pref_result = fusion_engine.fuse_user_preferences(
        preferences,
        source_info={'user_id': 'user_1', 'source': 'multiple'}
    )
    
    print(f"\nPreference Fusion ID: {pref_result.fusion_id}")
    print(f"Confidence: {pref_result.confidence:.2f}")
    
    # Test fusion quality evaluation
    original = {'summary': 'User likes action movies', 'type': 'user'}
    fused = {'summary': 'User prefers action movies', 'confidence': 0.8}
    quality = fusion_engine.evaluate_fusion_quality(original, fused)
    print(f"\nFusion Quality: {quality:.2f}")
    
    # Get statistics
    stats = fusion_engine.get_fusion_statistics()
    print(f"\nFusion Statistics: {stats}")