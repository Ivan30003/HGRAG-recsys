"""
Context Builder Module
Builds structured context for LLM prompts from graph retrieval results.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds formatted context strings for LLM prompts.
    
    Converts raw graph retrieval results into structured, readable
    natural language context that helps LLMs reason about
    collaborative patterns and relationships.
    """
    
    def __init__(self, max_context_length: int = 2000):
        """
        Initialize context builder.
        
        Args:
            max_context_length: Maximum length of built context in characters
        """
        self.max_context_length = max_context_length
    
    def build_context_prompt(self,
                             graph_context: Dict,
                             user_memory: str = '',
                             pos_item_memory: str = '',
                             neg_item_memory: str = '',
                             include_paths: bool = True,
                             include_statistics: bool = True) -> str:
        """
        Build a comprehensive context prompt for agent decision making.
        
        Args:
            graph_context: Raw graph context from MetapathExtractor
            user_memory: User agent's memory text
            pos_item_memory: Positive item's memory text
            neg_item_memory: Negative item's memory text
            include_paths: Whether to include metapath descriptions
            include_statistics: Whether to include graph statistics
        
        Returns:
            Formatted context string for LLM prompt
        """
        sections = []
        
        # Section 1: Similar users context
        similar_users_section = self._build_similar_users_section(graph_context)
        if similar_users_section:
            sections.append(similar_users_section)
        
        # Section 2: Item relationships
        item_relations_section = self._build_item_relations_section(graph_context)
        if item_relations_section:
            sections.append(item_relations_section)
        
        # Section 3: Metapath reasoning paths
        if include_paths:
            paths_section = self._build_metapaths_section(graph_context)
            if paths_section:
                sections.append(paths_section)
        
        # Section 4: Collaborative statistics
        if include_statistics:
            stats_section = self._build_statistics_section(graph_context)
            if stats_section:
                sections.append(stats_section)
        
        # Combine sections
        context = "\n\n".join(sections)
        
        # Truncate if too long
        if len(context) > self.max_context_length:
            context = self._truncate_context(context)
        
        return context
    
    def build_inference_prompt(self,
                               user_id: str,
                               user_memory: str,
                               candidate_items: Dict[str, str],
                               graph_context: Dict) -> str:
        """
        Build context for Phase 3 inference ranking.
        
        Args:
            user_id: User agent ID
            user_memory: User's memory text
            candidate_items: Dict mapping item_id -> item description
            graph_context: Graph context from retrieval
        
        Returns:
            Formatted inference context
        """
        sections = []
        
        # User overview
        sections.append(f"=== User Context ===\n{user_memory}")
        
        # Candidate items
        items_section = self._build_candidate_items_section(candidate_items)
        sections.append(f"=== Candidate Items ===\n{items_section}")
        
        # Graph insights
        graph_section = self._build_inference_graph_section(graph_context)
        if graph_section:
            sections.append(f"=== Collaborative Insights ===\n{graph_section}")
        
        return "\n\n".join(sections)
    
    def _build_similar_users_section(self, graph_context: Dict) -> Optional[str]:
        """
        Build section about similar users and their preferences.
        
        Args:
            graph_context: Graph context dictionary
        
        Returns:
            Formatted similar users section
        """
        similar_users = graph_context.get('similar_users', [])
        if not similar_users:
            return None
        
        lines = ["=== Similar Users ==="]
        lines.append("The following users have preferences similar to yours:")
        
        for i, user_info in enumerate(similar_users[:5], 1):
            user_id = user_info.get('user_id', f'user_{i}')
            preferences = user_info.get('preferences', 'Unknown preferences')
            shared_items = user_info.get('shared_items', [])
            similarity = user_info.get('similarity', 0.0)
            
            lines.append(f"\n{i}. User {user_id} (similarity: {similarity:.0%})")
            
            if shared_items:
                lines.append(f"   Shares interactions with: {', '.join(shared_items[:3])}")
            
            if preferences:
                prefs_str = self._truncate_text(str(preferences), 200)
                lines.append(f"   Preferences: {prefs_str}")
        
        return "\n".join(lines)
    
    def _build_item_relations_section(self, graph_context: Dict) -> Optional[str]:
        """
        Build section about item-to-item relationships.
        
        Args:
            graph_context: Graph context dictionary
        
        Returns:
            Formatted item relations section
        """
        item_relations = graph_context.get('item_relations', [])
        if not item_relations:
            return None
        
        lines = ["=== Item Relationships ==="]
        
        for relation in item_relations[:5]:
            source = relation.get('source_item', 'Unknown')
            target = relation.get('target_item', 'Unknown')
            rel_type = relation.get('relation_type', 'similar')
            strength = relation.get('strength', 0.0)
            co_users = relation.get('co_interacting_users', 0)
            
            if rel_type == 'content_sim':
                lines.append(
                    f"• '{source}' is content-similar to '{target}' "
                    f"(similarity: {strength:.0%})"
                )
            elif rel_type == 'co_interact':
                lines.append(
                    f"• '{source}' and '{target}' are often interacted with "
                    f"by the same users ({co_users} shared users)"
                )
        
        return "\n".join(lines)
    
    def _build_metapaths_section(self, graph_context: Dict) -> Optional[str]:
        """
        Build section describing reasoning paths from graph.
        
        Args:
            graph_context: Graph context dictionary
        
        Returns:
            Formatted metapaths section
        """
        metapaths = graph_context.get('metapaths', [])
        if not metapaths:
            return None
        
        lines = ["=== Reasoning Paths ==="]
        lines.append("The following relational patterns were found in the interaction graph:")
        
        # Group by metapath type
        paths_by_type = defaultdict(list)
        for path in metapaths:
            path_type = path.get('type', 'unknown')
            paths_by_type[path_type].append(path)
        
        # User-Item-User paths
        if 'user-item-user' in paths_by_type:
            lines.append("\nUsers who interacted with similar items:")
            for path in paths_by_type['user-item-user'][:3]:
                description = path.get('description', '')
                lines.append(f"  → {self._truncate_text(description, 150)}")
        
        # User-User-Item paths
        if 'user-user-item' in paths_by_type:
            lines.append("\nWhat similar users chose:")
            for path in paths_by_type['user-user-item'][:3]:
                description = path.get('description', '')
                lines.append(f"  → {self._truncate_text(description, 150)}")
        
        # Item-Item-Analogy paths
        if 'item-item-analogy' in paths_by_type:
            lines.append("\nItems similar to ones you've enjoyed:")
            for path in paths_by_type['item-item-analogy'][:3]:
                description = path.get('description', '')
                lines.append(f"  → {self._truncate_text(description, 150)}")
        
        return "\n".join(lines)
    
    def _build_statistics_section(self, graph_context: Dict) -> Optional[str]:
        """
        Build section with collaborative statistics.
        
        Args:
            graph_context: Graph context dictionary
        
        Returns:
            Formatted statistics section
        """
        stats = graph_context.get('statistics', {})
        if not stats:
            return None
        
        lines = ["=== Graph Statistics ==="]
        
        if 'num_similar_users' in stats:
            lines.append(f"• {stats['num_similar_users']} users with similar preferences found")
        
        if 'num_shared_items' in stats:
            lines.append(f"• {stats['num_shared_items']} items shared with similar users")
        
        if 'popularity_percentile' in stats:
            lines.append(f"• Item popularity: {stats['popularity_percentile']}th percentile")
        
        if 'avg_rating' in stats:
            lines.append(f"• Average user rating: {stats['avg_rating']:.1f}/5.0")
        
        return "\n".join(lines)
    
    def _build_candidate_items_section(self, candidate_items: Dict[str, str]) -> str:
        """
        Build section listing candidate items.
        
        Args:
            candidate_items: Dict mapping item_id -> description
        
        Returns:
            Formatted candidate items section
        """
        lines = []
        
        for i, (item_id, description) in enumerate(candidate_items.items(), 1):
            desc_short = self._truncate_text(description, 150)
            lines.append(f"{i}. [{item_id}] {desc_short}")
        
        return "\n".join(lines)
    
    def _build_inference_graph_section(self, graph_context: Dict) -> Optional[str]:
        """
        Build graph insights section for inference.
        
        Args:
            graph_context: Graph context dictionary
        
        Returns:
            Formatted inference graph section
        """
        parts = []
        
        # Similar users' preferences
        similar_users = graph_context.get('similar_users', [])
        if similar_users:
            liked_items = []
            for user in similar_users[:3]:
                for item in user.get('liked_items', [])[:2]:
                    if item not in liked_items:
                        liked_items.append(item)
            
            if liked_items:
                parts.append(
                    f"Users similar to you also liked: {', '.join(liked_items[:5])}"
                )
        
        # Popular in neighborhood
        popular_items = graph_context.get('popular_in_neighborhood', [])
        if popular_items:
            parts.append(
                f"Popular among your peers: {', '.join(popular_items[:5])}"
            )
        
        # Trending items
        trending = graph_context.get('trending_items', [])
        if trending:
            parts.append(
                f"Currently trending: {', '.join(trending[:3])}"
            )
        
        if not parts:
            return None
        
        return "\n".join(parts)
    
    def build_reflection_context(self,
                                  user_id: str,
                                  decision: Dict,
                                  graph_context: Dict) -> str:
        """
        Build context specifically for reflection analysis.
        
        Args:
            user_id: User agent ID
            decision: Decision dictionary
            graph_context: Graph context
        
        Returns:
            Formatted reflection context
        """
        sections = []
        
        # What went wrong
        sections.append("=== Decision Analysis ===")
        sections.append(f"Decision: {decision.get('decision', 'unknown')}")
        sections.append(f"Confidence: {decision.get('confidence', 0.0):.2f}")
        sections.append(f"Method: {decision.get('method', 'unknown')}")
        
        # Available context that was used/missed
        sections.append("\n=== Context That Was Available ===")
        
        metapaths = graph_context.get('metapaths', [])
        if metapaths:
            sections.append("The following graph paths were retrieved but may not have been used:")
            for path in metapaths[:5]:
                path_type = path.get('type', 'unknown')
                path_desc = path.get('description', 'No description')
                sections.append(f"  [{path_type}] {self._truncate_text(path_desc, 150)}")
        
        # Similar users that could have helped
        similar_users = graph_context.get('similar_users', [])
        if similar_users:
            sections.append("\nSimilar users whose preferences could have informed the decision:")
            for user in similar_users[:3]:
                sections.append(f"  • User with {user.get('similarity', 0):.0%} preference similarity")
        
        return "\n".join(sections)
    
    def build_propagation_context(self,
                                   agent_id: str,
                                   neighbor_signals: List[Dict],
                                   propagation_paths: List[Dict]) -> str:
        """
        Build context for neighborhood propagation.
        
        Args:
            agent_id: Agent receiving signals
            neighbor_signals: Signals from neighbors
            propagation_paths: Paths through which signals propagated
        
        Returns:
            Formatted propagation context
        """
        sections = []
        
        sections.append(f"=== Propagation Context for Agent {agent_id} ===")
        
        # Neighbor signals
        sections.append(f"\nReceived signals from {len(neighbor_signals)} neighbors:")
        
        for i, signal in enumerate(neighbor_signals[:5], 1):
            source = signal.get('source_agent', 'unknown')
            signal_type = signal.get('signal_type', 'preference')
            strength = signal.get('strength', 0.0)
            content = signal.get('content', '')
            
            sections.append(
                f"\n{i}. From {source} (strength: {strength:.2f})"
            )
            sections.append(f"   Type: {signal_type}")
            sections.append(f"   Content: {self._truncate_text(str(content), 200)}")
        
        # Propagation paths
        if propagation_paths:
            sections.append(f"\nSignals arrived via {len(propagation_paths)} paths:")
            for path in propagation_paths[:3]:
                path_str = " → ".join(path.get('nodes', ['unknown']))
                sections.append(f"  • {path_str}")
        
        return "\n".join(sections)
    
    def build_cold_start_context(self,
                                  user_id: str,
                                  similar_users: List[Dict],
                                  popular_items: List[str]) -> str:
        """
        Build context for cold-start recommendation.
        
        Args:
            user_id: Cold-start user ID
            similar_users: Demographically similar users
            popular_items: Popular items in preferred categories
        
        Returns:
            Formatted cold-start context
        """
        sections = []
        
        sections.append(f"=== Cold-Start Context for User {user_id} ===")
        
        if similar_users:
            sections.append(f"\nFound {len(similar_users)} demographically similar users.")
            
            # Aggregate their preferences
            all_prefs = []
            all_categories = []
            for user in similar_users[:10]:
                all_prefs.extend(user.get('preferences', []))
                all_categories.extend(user.get('preferred_categories', []))
            
            if all_prefs:
                from collections import Counter
                pref_counts = Counter(all_prefs)
                top_prefs = pref_counts.most_common(5)
                sections.append(
                    f"Common preferences: {', '.join([p for p, _ in top_prefs])}"
                )
            
            if all_categories:
                cat_counts = Counter(all_categories)
                top_cats = cat_counts.most_common(3)
                sections.append(
                    f"Popular categories: {', '.join([c for c, _ in top_cats])}"
                )
        
        if popular_items:
            sections.append(
                f"\nPopular items in relevant categories: {', '.join(popular_items[:5])}"
            )
        
        return "\n".join(sections)
    
    def build_item_warmup_context(self,
                                   item_id: str,
                                   similar_items: List[Dict],
                                   audience_predictions: List[Dict]) -> str:
        """
        Build context for item cold-start warmup.
        
        Args:
            item_id: Cold-start item ID
            similar_items: Content-similar items with established audiences
            audience_predictions: Predicted audience segments
        
        Returns:
            Formatted warmup context
        """
        sections = []
        
        sections.append(f"=== Item Warmup Context for {item_id} ===")
        
        if similar_items:
            sections.append(f"\nFound {len(similar_items)} similar items with established audiences:")
            for item in similar_items[:5]:
                item_title = item.get('title', 'Unknown')
                audience_size = item.get('total_interactions', 0)
                key_audience = item.get('common_user_traits', [])
                
                sections.append(f"\n  • {item_title} ({audience_size} interactions)")
                if key_audience:
                    sections.append(f"    Popular with: {', '.join(key_audience[:3])}")
        
        if audience_predictions:
            sections.append(f"\nPredicted audience segments:")
            for pred in audience_predictions[:5]:
                trait = pred.get('trait', '')
                confidence = pred.get('confidence', 0.0)
                sections.append(f"  • {trait} (confidence: {confidence:.0%})")
        
        return "\n".join(sections)
    
    def build_advertisement_context(self,
                                     item_details: str,
                                     target_audience: List[str],
                                     competitor_ads: Optional[List[str]] = None) -> str:
        """
        Build context for advertisement generation.
        
        Args:
            item_details: Item description
            target_audience: Target audience segments
            competitor_ads: Optional competitor advertisements
        
        Returns:
            Formatted advertisement context
        """
        sections = []
        
        sections.append("=== Advertisement Context ===")
        sections.append(f"\nItem: {item_details}")
        
        sections.append(f"\nTarget Audience:")
        for audience in target_audience:
            sections.append(f"  • {audience}")
        
        if competitor_ads:
            sections.append(f"\nCompetitor Advertisements (for reference):")
            for i, ad in enumerate(competitor_ads[:3], 1):
                sections.append(f"\n  Competitor {i}:")
                sections.append(f"  {self._truncate_text(ad, 200)}")
        
        return "\n".join(sections)
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """
        Truncate text to maximum length with ellipsis.
        
        Args:
            text: Input text
            max_length: Maximum length
        
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        
        return text[:max_length - 3] + "..."
    
    def _truncate_context(self, context: str) -> str:
        """
        Truncate full context to maximum length.
        Preserves section headers.
        
        Args:
            context: Full context text
        
        Returns:
            Truncated context
        """
        if len(context) <= self.max_context_length:
            return context
        
        # Split by sections
        sections = context.split("\n\n")
        result = []
        current_length = 0
        
        for section in sections:
            section_length = len(section) + 2  # +2 for "\n\n"
            
            if current_length + section_length <= self.max_context_length - 50:
                result.append(section)
                current_length += section_length
            else:
                # Add truncated note
                result.append(f"\n[... {len(sections) - len(result)} more sections truncated ...]")
                break
        
        return "\n\n".join(result)
    
    def estimate_context_quality(self, graph_context: Dict) -> Dict[str, float]:
        """
        Estimate the quality and informativeness of graph context.
        
        Args:
            graph_context: Graph context dictionary
        
        Returns:
            Dictionary with quality metrics
        """
        metrics = {}
        
        # Number of similar users found
        num_similar = len(graph_context.get('similar_users', []))
        metrics['similar_users_count'] = num_similar
        metrics['has_similar_users'] = float(num_similar > 0)
        
        # Number of metapaths
        num_paths = len(graph_context.get('metapaths', []))
        metrics['metapaths_count'] = num_paths
        metrics['has_metapaths'] = float(num_paths > 0)
        
        # Path type diversity
        path_types = set()
        for path in graph_context.get('metapaths', []):
            path_types.add(path.get('type', 'unknown'))
        metrics['path_type_diversity'] = len(path_types) / 3.0  # Max 3 types
        
        # Average similarity of similar users
        similarities = [
            u.get('similarity', 0.0) 
            for u in graph_context.get('similar_users', [])
        ]
        metrics['avg_similarity'] = sum(similarities) / max(1, len(similarities))
        
        # Overall quality score
        quality_score = (
            0.3 * metrics['has_similar_users'] +
            0.3 * metrics['has_metapaths'] +
            0.2 * min(1.0, metrics['similar_users_count'] / 10) +
            0.1 * metrics['path_type_diversity'] +
            0.1 * metrics['avg_similarity']
        )
        metrics['overall_quality'] = quality_score
        
        return metrics