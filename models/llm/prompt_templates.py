"""
Prompt Templates Module for H-GRAGrecsys

This module provides comprehensive prompt templates for various LLM tasks
including reflection, fusion, ranking, explanation, and summarization.
"""

import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger import Logger
from utils.config_loader import ConfigLoader


@dataclass
class PromptTemplate:
    """
    Represents a prompt template with metadata.
    
    Attributes:
        name: Template name
        system_prompt: System prompt for the template
        user_prompt: User prompt template
        description: Template description
        variables: Required variables
        version: Template version
    """
    name: str
    system_prompt: str
    user_prompt: str
    description: str = ""
    variables: List[str] = field(default_factory=list)
    version: str = "1.0"


class PromptTemplates:
    """
    Comprehensive prompt templates for LLM interactions.
    
    This class provides:
    - Reflection prompts for agent reflection
    - Fusion prompts for memory fusion
    - Ranking prompts for recommendation ranking
    - Explanation prompts for recommendation explanations
    - Summarization prompts for text summarization
    - Metapath verbalization prompts
    - System prompts for various tasks
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize prompt templates.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = Logger.get_instance(log_dir='logs', name='prompt_templates')
        
        # Extract configuration
        llm_config = config.get('model', {}).get('llm', {})
        self.max_prompt_length = llm_config.get('max_prompt_length', 4096)
        self.template_version = llm_config.get('template_version', '1.0')
        
        # Initialize template registry
        self.templates: Dict[str, PromptTemplate] = {}
        self._initialize_templates()
        
        # Statistics
        self.usage_stats = {
            'total_uses': 0,
            'template_uses': defaultdict(int),
            'template_variables': defaultdict(set)
        }
        
        self.logger.log_info(f"Initialized PromptTemplates with {len(self.templates)} templates")
    
    def _initialize_templates(self) -> None:
        """
        Initialize all prompt templates.
        """
        # System prompts
        self._initialize_system_prompts()
        
        # Task-specific templates
        self._initialize_reflection_templates()
        self._initialize_fusion_templates()
        self._initialize_ranking_templates()
        self._initialize_explanation_templates()
        self._initialize_summarization_templates()
        self._initialize_metapath_templates()
        self._initialize_recommendation_templates()
        self._initialize_evaluation_templates()
    
    def _initialize_system_prompts(self) -> None:
        """
        Initialize system prompts for different roles.
        """
        self.SYSTEM_REFLECTION = """You are an AI assistant specialized in analyzing interactions and generating reflective insights for a recommendation system. You should:
1. Analyze user-item interactions carefully
2. Identify patterns and preferences
3. Generate meaningful reflections
4. Consider both explicit and implicit signals
5. Provide structured and actionable insights"""

        self.SYSTEM_FUSION = """You are an AI assistant specialized in fusing multiple sources of information into coherent representations. You should:
1. Combine information from different sources
2. Identify complementary patterns
3. Resolve conflicts between sources
4. Generate unified representations
5. Preserve important details"""

        self.SYSTEM_RANKING = """You are an AI assistant specialized in ranking items for recommendation. You should:
1. Consider user preferences and context
2. Evaluate item relevance
3. Provide explainable rankings
4. Consider diversity and novelty
5. Generate confidence scores"""

        self.SYSTEM_EXPLANATION = """You are an AI assistant specialized in generating explanations for recommendations. You should:
1. Provide clear and concise explanations
2. Connect recommendations to user preferences
3. Use natural and engaging language
4. Include relevant context
5. Build trust through transparency"""

        self.SYSTEM_EVALUATION = """You are an AI assistant specialized in evaluating recommendations and providing constructive feedback. You should:
1. Assess recommendation quality
2. Identify strengths and weaknesses
3. Provide actionable suggestions
4. Consider multiple evaluation criteria
5. Generate balanced assessments"""
    
    def _initialize_reflection_templates(self) -> None:
        """
        Initialize reflection prompt templates.
        """
        self.templates['reflection'] = PromptTemplate(
            name='reflection',
            system_prompt=self.SYSTEM_REFLECTION,
            user_prompt="""Analyze the following interaction and generate a reflection:

User: {user_info}
Item: {item_info}
Interaction Context: {context}

Interaction Details:
- Rating: {rating}
- Timestamp: {timestamp}
- Outcome: {outcome}

Based on this interaction, please provide:
1. Preference Signal: What does this interaction reveal about the user's preferences?
2. Item Assessment: How suitable was this item for the user?
3. Pattern Identification: What patterns can you identify from this interaction?
4. Future Recommendations: What would you recommend based on this interaction?
5. Confidence Score: How confident are you in this assessment? (0-1)

Please provide your reflection in a structured format.""",
            description="Template for generating reflections from interactions",
            variables=['user_info', 'item_info', 'context', 'rating', 'timestamp', 'outcome']
        )
        
        self.templates['reflection_batch'] = PromptTemplate(
            name='reflection_batch',
            system_prompt=self.SYSTEM_REFLECTION,
            user_prompt="""Analyze the following interactions and generate aggregated reflections:

Interactions:
{interactions}

User History: {user_history}

Please provide:
1. Overall Preference Summary: What are the user's key preferences?
2. Preference Evolution: How have preferences evolved over time?
3. Key Patterns: What are the most important patterns?
4. Personalization Insights: What makes this user unique?
5. Recommendations: What would you recommend next?
6. Confidence Score: How confident are you in this assessment? (0-1)""",
            description="Template for generating aggregated reflections",
            variables=['interactions', 'user_history']
        )
    
    def _initialize_fusion_templates(self) -> None:
        """
        Initialize fusion prompt templates.
        """
        self.templates['fusion_memories'] = PromptTemplate(
            name='fusion_memories',
            system_prompt=self.SYSTEM_FUSION,
            user_prompt="""Fuse the following memories into a unified representation:

Memories to Fuse:
{memories}

Fusion Context:
- Memory Types: {memory_types}
- Importance Weights: {weights}
- Goal: {goal}

Please provide:
1. Fused Representation: A unified representation of all memories
2. Key Insights: What are the most important insights?
3. Conflicts Resolved: How were conflicts resolved?
4. Confidence Score: How confident are you in this fusion? (0-1)
5. Additional Notes: Any additional observations""",
            description="Template for fusing multiple memories",
            variables=['memories', 'memory_types', 'weights', 'goal']
        )
        
        self.templates['fusion_user_preferences'] = PromptTemplate(
            name='fusion_user_preferences',
            system_prompt=self.SYSTEM_FUSION,
            user_prompt="""Fuse the following user preferences into a coherent profile:

User Preferences:
{preferences}

Preference Types:
- Explicit: {explicit}
- Implicit: {implicit}
- Contextual: {contextual}
- Historical: {historical}

Please provide:
1. Unified Preference Profile: A coherent representation
2. Preference Strength: How strong are these preferences? (0-1)
3. Preference Stability: How stable are these preferences?
4. Key Preferences: What are the top 5 preferences?
5. Preference Conflicts: Are there any conflicts?
6. Additional Insights: Any other important observations""",
            description="Template for fusing user preferences",
            variables=['preferences', 'explicit', 'implicit', 'contextual', 'historical']
        )
        
        self.templates['fusion_item_patterns'] = PromptTemplate(
            name='fusion_item_patterns',
            system_prompt=self.SYSTEM_FUSION,
            user_prompt="""Fuse the following item interaction patterns:

Item: {item}
Interaction Patterns: {patterns}
User Demographics: {demographics}

Please provide:
1. Unified Pattern Representation
2. User Engagement: How engaged are users with this item?
3. Item Characteristics: What are the key characteristics?
4. Popularity Context: How popular is this item in different contexts?
5. Recommendation Potential: How recommendable is this item?
6. Additional Insights""",
            description="Template for fusing item patterns",
            variables=['item', 'patterns', 'demographics']
        )
    
    def _initialize_ranking_templates(self) -> None:
        """
        Initialize ranking prompt templates.
        """
        self.templates['ranking'] = PromptTemplate(
            name='ranking',
            system_prompt=self.SYSTEM_RANKING,
            user_prompt="""Rank the following candidate items for the user:

User Profile:
{user_profile}

Candidate Items:
{candidates}

Context:
{context}

Ranking Criteria:
- Relevance to User: {relevance}
- Item Quality: {quality}
- Diversity: {diversity}
- Novelty: {novelty}

Please provide:
1. Ranked Items: Items sorted by relevance (include scores)
2. Top Recommendation: The top recommended item
3. Ranking Justification: Why are these items ranked this way?
4. Confidence Scores: Confidence for each ranked item
5. Alternative Recommendations: Alternatives if top items are not suitable""",
            description="Template for ranking items",
            variables=['user_profile', 'candidates', 'context', 'relevance', 'quality', 'diversity', 'novelty']
        )
        
        self.templates['ranking_metapath'] = PromptTemplate(
            name='ranking_metapath',
            system_prompt=self.SYSTEM_RANKING,
            user_prompt="""Rank candidate items using metapath-based reasoning:

User Profile: {user_profile}
Candidate Items: {candidates}
Metapaths: {metapaths}
Path Weights: {path_weights}

For each metapath, consider:
1. Path Relevance: How relevant is this path?
2. Path Confidence: How confident are we in this path?
3. Path Diversity: Does this path provide diverse perspectives?

Please provide:
1. Metapath-Based Rankings: Items ranked by each metapath
2. Aggregated Ranking: Combined ranking across all metapaths
3. Most Influential Path: Which path had the most influence?
4. Confidence Scores: Overall confidence in the ranking""",
            description="Template for metapath-based ranking",
            variables=['user_profile', 'candidates', 'metapaths', 'path_weights']
        )
    
    def _initialize_explanation_templates(self) -> None:
        """
        Initialize explanation prompt templates.
        """
        self.templates['explanation'] = PromptTemplate(
            name='explanation',
            system_prompt=self.SYSTEM_EXPLANATION,
            user_prompt="""Generate an explanation for the following recommendation:

User Profile: {user_profile}
Recommended Item: {item}
Reason for Recommendation: {reason}
Context: {context}

Explanation Requirements:
- Target Audience: {audience}
- Tone: {tone}
- Length: {length}
- Key Points to Cover: {key_points}

Please provide:
1. Concise Explanation: A clear explanation of why this item was recommended
2. Detailed Explanation: A more detailed explanation with supporting reasons
3. User Benefit: How does this recommendation benefit the user?
4. Alternative Perspective: Any alternative considerations
5. Trust Building: Why the user should trust this recommendation""",
            description="Template for generating explanations",
            variables=['user_profile', 'item', 'reason', 'context', 'audience', 'tone', 'length', 'key_points']
        )
        
        self.templates['explanation_comparative'] = PromptTemplate(
            name='explanation_comparative',
            system_prompt=self.SYSTEM_EXPLANATION,
            user_prompt="""Generate a comparative explanation for the following recommendations:

User Profile: {user_profile}
Top Recommendations: {top_items}
Alternative Items: {alternatives}
Comparison Criteria: {criteria}

Please provide:
1. Why Top Items: Why were these items recommended?
2. Why Alternatives: Why were alternatives not recommended?
3. Key Differences: What are the key differences?
4. User Fit: Which recommendation best fits the user?
5. Recommendation Confidence: Confidence in the recommendation (0-1)""",
            description="Template for comparative explanations",
            variables=['user_profile', 'top_items', 'alternatives', 'criteria']
        )
    
    def _initialize_summarization_templates(self) -> None:
        """
        Initialize summarization prompt templates.
        """
        self.templates['summarization'] = PromptTemplate(
            name='summarization',
            system_prompt="You are an AI assistant specialized in summarizing text.",
            user_prompt="""Summarize the following text:

Text to Summarize:
{text}

Summarization Requirements:
- Maximum Length: {max_length}
- Key Points: {key_points}
- Format: {format}
- Audience: {audience}

Please provide a concise summary that captures the main points.""",
            description="Template for text summarization",
            variables=['text', 'max_length', 'key_points', 'format', 'audience']
        )
        
        self.templates['summarization_interactions'] = PromptTemplate(
            name='summarization_interactions',
            system_prompt="You are an AI assistant specialized in summarizing user interaction history.",
            user_prompt="""Summarize the following user interaction history:

User: {user}
Interactions:
{interactions}

Interaction Types:
- Explicit Ratings: {explicit_ratings}
- Implicit Signals: {implicit_signals}
- Contextual Information: {contextual_info}

Please provide:
1. Interaction Summary: A brief overview of the interactions
2. Key Patterns: What patterns emerge?
3. User Preferences: What preferences are revealed?
4. Interaction Quality: How meaningful are these interactions?
5. Recommendation Implications: What does this mean for recommendations?""",
            description="Template for summarizing interaction history",
            variables=['user', 'interactions', 'explicit_ratings', 'implicit_signals', 'contextual_info']
        )
    
    def _initialize_metapath_templates(self) -> None:
        """
        Initialize metapath verbalization templates.
        """
        self.templates['metapath'] = PromptTemplate(
            name='metapath',
            system_prompt="You are an AI assistant specialized in explaining graph structures.",
            user_prompt="""Verbalize the following metapath:

Metapath: {metapath}
Path Nodes: {nodes}
Path Relations: {relations}
Path Type: {path_type}

Please provide:
1. Natural Language Description: A clear description of this metapath
2. Relevance Explanation: Why is this metapath relevant?
3. User-Item Connection: How does this connect users and items?
4. Recommendation Implication: What does this imply for recommendations?
5. Confidence Score: How confident are you in this interpretation? (0-1)""",
            description="Template for verbalizing metapaths",
            variables=['metapath', 'nodes', 'relations', 'path_type']
        )
        
        self.templates['metapath_batch'] = PromptTemplate(
            name='metapath_batch',
            system_prompt="You are an AI assistant specialized in analyzing graph patterns.",
            user_prompt="""Analyze and verbalize the following metapaths:

Metapaths:
{metapaths}

Analysis Requirements:
- Compare Patterns: {compare_patterns}
- Identify Commonalities: {identify_commonalities}
- Highlight Important Paths: {highlight_paths}

Please provide:
1. Individual Descriptions: Description of each metapath
2. Pattern Analysis: What patterns emerge across metapaths?
3. Most Important Path: Which metapath is most important and why?
4. Recommendation Insights: What insights for recommendations?
5. Confidence Assessment: How confident are you in this analysis? (0-1)""",
            description="Template for analyzing multiple metapaths",
            variables=['metapaths', 'compare_patterns', 'identify_commonalities', 'highlight_paths']
        )
    
    def _initialize_recommendation_templates(self) -> None:
        """
        Initialize recommendation-specific templates.
        """
        self.templates['recommendation_general'] = PromptTemplate(
            name='recommendation_general',
            system_prompt="You are an AI assistant specialized in providing personalized recommendations.",
            user_prompt="""Provide personalized recommendations for the following user:

User Profile:
{user_profile}

Available Items:
{available_items}

Recommendation Constraints:
- Number of Recommendations: {num_recommendations}
- Diversity Requirement: {diversity}
- Novelty Requirement: {novelty}
- Serendipity: {serendipity}

Please provide:
1. Top Recommendations: List of recommended items
2. Recommendation Rationale: Why these items?
3. Explanation for Each: Brief explanation for each recommendation
4. Confidence Scores: Confidence for each recommendation (0-1)
5. Alternative Sets: Alternative recommendations if needed""",
            description="Template for general recommendations",
            variables=['user_profile', 'available_items', 'num_recommendations', 'diversity', 'novelty', 'serendipity']
        )
        
        self.templates['recommendation_hybrid'] = PromptTemplate(
            name='recommendation_hybrid',
            system_prompt=self.SYSTEM_RANKING,
            user_prompt="""Generate hybrid recommendations combining collaborative and content-based approaches:

User Profile:
{user_profile}

Collaborative Recommendations: {collaborative}
Content-Based Recommendations: {content_based}
Hybrid Strategy: {hybrid_strategy}

Please provide:
1. Hybrid Recommendations: Combined recommendations
2. Fusion Method: How were the approaches combined?
3. Balance Analysis: How balanced are the different approaches?
4. Final Ranking: Ranked list with scores
5. Explanation: Why these recommendations?
6. Confidence: Overall confidence (0-1)""",
            description="Template for hybrid recommendations",
            variables=['user_profile', 'collaborative', 'content_based', 'hybrid_strategy']
        )
    
    def _initialize_evaluation_templates(self) -> None:
        """
        Initialize evaluation prompt templates.
        """
        self.templates['evaluation'] = PromptTemplate(
            name='evaluation',
            system_prompt=self.SYSTEM_EVALUATION,
            user_prompt="""Evaluate the following recommendation:

User: {user}
Recommended Items: {items}
Recommendation Context: {context}
Evaluation Criteria: {criteria}

Please provide:
1. Overall Assessment: How good is this recommendation? (0-1)
2. Strengths: What are the strengths?
3. Weaknesses: What are the weaknesses?
4. User Fit: How well does this fit the user?
5. Improvement Suggestions: How could it be improved?
6. Actionable Feedback: What changes would you suggest?""",
            description="Template for evaluating recommendations",
            variables=['user', 'items', 'context', 'criteria']
        )
        
        self.templates['evaluation_batch'] = PromptTemplate(
            name='evaluation_batch',
            system_prompt=self.SYSTEM_EVALUATION,
            user_prompt="""Evaluate the following batch of recommendations:

User: {user}
Recommendations: {recommendations}
Evaluation Metrics: {metrics}
Baseline Performance: {baseline}

Please provide:
1. Overall Performance: How well does this perform? (0-1)
2. Per-Item Evaluation: Evaluation of each recommendation
3. Trends: Any notable trends?
4. Comparison with Baseline: How does this compare?
5. Recommendations for Improvement: What would you suggest?
6. Summary: Brief evaluation summary""",
            description="Template for batch evaluation",
            variables=['user', 'recommendations', 'metrics', 'baseline']
        )
    
    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        """
        Get a prompt template by name.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Optional[PromptTemplate]: The template or None
        """
        template = self.templates.get(template_name)
        if template:
            self.usage_stats['total_uses'] += 1
            self.usage_stats['template_uses'][template_name] += 1
        return template
    
    def render_template(self, template_name: str, 
                       variables: Dict[str, Any]) -> str:
        """
        Render a template with variables.
        
        Args:
            template_name: Name of the template
            variables: Variables for the template
            
        Returns:
            str: Rendered template
            
        Raises:
            ValueError: If template not found or variables missing
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Check required variables
        missing = [v for v in template.variables if v not in variables]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        # Format user prompt
        user_prompt = template.user_prompt.format(**variables)
        
        # Track variables used
        for var in variables.keys():
            self.usage_stats['template_variables'][template_name].add(var)
        
        # Combine with system prompt if needed
        if template.system_prompt:
            return f"{template.system_prompt}\n\n{user_prompt}"
        
        return user_prompt
    
    def get_reflection_prompt(self, user: str, item: str, 
                             context: Dict[str, Any]) -> str:
        """
        Get reflection prompt for user-item interaction.
        
        Args:
            user: User information
            item: Item information
            context: Interaction context
            
        Returns:
            str: Reflection prompt
        """
        variables = {
            'user_info': user,
            'item_info': item,
            'context': json.dumps(context, indent=2),
            'rating': context.get('rating', 'unknown'),
            'timestamp': context.get('timestamp', datetime.now().isoformat()),
            'outcome': context.get('outcome', 'unknown')
        }
        return self.render_template('reflection', variables)
    
    def get_fusion_prompt(self, memories: List[Dict[str, Any]], 
                         instruction: str) -> str:
        """
        Get fusion prompt for memories.
        
        Args:
            memories: List of memory dictionaries
            instruction: Fusion instruction
            
        Returns:
            str: Fusion prompt
        """
        variables = {
            'memories': json.dumps(memories, indent=2),
            'memory_types': [m.get('type', 'unknown') for m in memories],
            'weights': [m.get('weight', 1.0) for m in memories],
            'goal': instruction
        }
        return self.render_template('fusion_memories', variables)
    
    def get_ranking_prompt(self, user: str, candidates: List[str], 
                          context: Dict[str, Any]) -> str:
        """
        Get ranking prompt for recommendation.
        
        Args:
            user: User profile information
            candidates: List of candidate items
            context: Ranking context
            
        Returns:
            str: Ranking prompt
        """
        variables = {
            'user_profile': user,
            'candidates': json.dumps(candidates, indent=2),
            'context': json.dumps(context, indent=2),
            'relevance': context.get('relevance', 'high'),
            'quality': context.get('quality', 'high'),
            'diversity': context.get('diversity', 'medium'),
            'novelty': context.get('novelty', 'medium')
        }
        return self.render_template('ranking', variables)
    
    def get_explanation_prompt(self, user: str, item: str, 
                             recommendation: Dict[str, Any]) -> str:
        """
        Get explanation prompt for recommendation.
        
        Args:
            user: User profile information
            item: Recommended item
            recommendation: Recommendation details
            
        Returns:
            str: Explanation prompt
        """
        variables = {
            'user_profile': user,
            'item': item,
            'reason': recommendation.get('reason', ''),
            'context': json.dumps(recommendation.get('context', {}), indent=2),
            'audience': recommendation.get('audience', 'general'),
            'tone': recommendation.get('tone', 'friendly'),
            'length': recommendation.get('length', 'medium'),
            'key_points': ', '.join(recommendation.get('key_points', ['relevance', 'quality']))
        }
        return self.render_template('explanation', variables)
    
    def get_summarization_prompt(self, text: str, 
                                max_length: Optional[int] = None) -> str:
        """
        Get summarization prompt for text.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length
            
        Returns:
            str: Summarization prompt
        """
        variables = {
            'text': text,
            'max_length': max_length or 100,
            'key_points': ['main_idea', 'important_details'],
            'format': 'paragraph',
            'audience': 'general'
        }
        return self.render_template('summarization', variables)
    
    def get_metapath_verbalization_prompt(self, metapath: Dict[str, Any]) -> str:
        """
        Get prompt for verbalizing a metapath.
        
        Args:
            metapath: Metapath dictionary
            
        Returns:
            str: Metapath verbalization prompt
        """
        variables = {
            'metapath': json.dumps(metapath, indent=2),
            'nodes': metapath.get('path', []),
            'relations': metapath.get('relations', []),
            'path_type': metapath.get('path_type', 'unknown')
        }
        return self.render_template('metapath', variables)
    
    def get_hybrid_recommendation_prompt(self, user: str,
                                        collaborative: List[str],
                                        content_based: List[str],
                                        strategy: str) -> str:
        """
        Get hybrid recommendation prompt.
        
        Args:
            user: User profile
            collaborative: Collaborative filtering recommendations
            content_based: Content-based recommendations
            strategy: Hybridization strategy
            
        Returns:
            str: Hybrid recommendation prompt
        """
        variables = {
            'user_profile': user,
            'collaborative': json.dumps(collaborative, indent=2),
            'content_based': json.dumps(content_based, indent=2),
            'hybrid_strategy': strategy
        }
        return self.render_template('recommendation_hybrid', variables)
    
    def get_evaluation_prompt(self, user: str, items: List[str],
                            context: Dict[str, Any]) -> str:
        """
        Get evaluation prompt for recommendations.
        
        Args:
            user: User profile
            items: Recommended items
            context: Evaluation context
            
        Returns:
            str: Evaluation prompt
        """
        variables = {
            'user': user,
            'items': json.dumps(items, indent=2),
            'context': json.dumps(context, indent=2),
            'criteria': context.get('criteria', ['relevance', 'diversity', 'novelty'])
        }
        return self.render_template('evaluation', variables)
    
    def create_custom_prompt(self, system_prompt: str, 
                           user_prompt: str,
                           name: str = 'custom',
                           variables: Optional[List[str]] = None) -> str:
        """
        Create a custom prompt template.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt template
            name: Template name
            variables: Required variables
            
        Returns:
            str: Combined prompt
        """
        template = PromptTemplate(
            name=name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            description="Custom template",
            variables=variables or []
        )
        
        self.templates[name] = template
        self.logger.log_info(f"Created custom template: {name}")
        
        return f"{system_prompt}\n\n{user_prompt}"
    
    def get_template_usage_stats(self) -> Dict[str, Any]:
        """
        Get template usage statistics.
        
        Returns:
            Dict[str, Any]: Usage statistics
        """
        stats = {
            'total_uses': self.usage_stats['total_uses'],
            'template_uses': dict(self.usage_stats['template_uses']),
            'template_variables': {
                k: list(v) for k, v in self.usage_stats['template_variables'].items()
            },
            'most_used_template': max(
                self.usage_stats['template_uses'].items(),
                key=lambda x: x[1]
            )[0] if self.usage_stats['template_uses'] else None
        }
        return stats
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available templates.
        
        Returns:
            List[Dict[str, Any]]: Template information
        """
        return [
            {
                'name': t.name,
                'description': t.description,
                'variables': t.variables,
                'version': t.version
            }
            for t in self.templates.values()
        ]
    
    def get_template_metadata(self, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Optional[Dict[str, Any]]: Template metadata
        """
        template = self.get_template(template_name)
        if template:
            return {
                'name': template.name,
                'description': template.description,
                'variables': template.variables,
                'version': template.version,
                'system_prompt_length': len(template.system_prompt),
                'user_prompt_length': len(template.user_prompt),
                'usage_count': self.usage_stats['template_uses'].get(template_name, 0)
            }
        return None
    
    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.usage_stats = {
            'total_uses': 0,
            'template_uses': defaultdict(int),
            'template_variables': defaultdict(set)
        }
        self.logger.log_info("Reset template usage statistics")
    
    def __str__(self) -> str:
        """String representation."""
        return f"PromptTemplates(templates={len(self.templates)}, uses={self.usage_stats['total_uses']})"


# Example usage
if __name__ == "__main__":
    # Load configuration
    config_path = "config/default_config.yaml"
    config_loader = ConfigLoader(config_path)
    config = config_loader.load_config()
    
    # Create prompt templates
    prompt_templates = PromptTemplates(config)
    
    # List available templates
    templates = prompt_templates.list_templates()
    print(f"Available templates: {len(templates)}")
    for template in templates[:5]:
        print(f"  - {template['name']}: {template['description']}")
    
    # Test reflection prompt
    reflection_prompt = prompt_templates.get_reflection_prompt(
        user="User: Alice, age: 25, preferences: electronics",
        item="Item: Product A, category: electronics, price: $299",
        context={
            'rating': 4.5,
            'timestamp': '2024-01-15T10:30:00',
            'outcome': 'purchase'
        }
    )
    print("\nReflection Prompt (first 200 chars):")
    print(reflection_prompt[:200] + "...")
    
    # Test ranking prompt
    ranking_prompt = prompt_templates.get_ranking_prompt(
        user="User: Bob, age: 30, preferences: books",
        candidates=['Book A', 'Book B', 'Book C', 'Book D'],
        context={
            'relevance': 'high',
            'quality': 'medium',
            'diversity': 'high',
            'novelty': 'medium'
        }
    )
    print("\nRanking Prompt (first 200 chars):")
    print(ranking_prompt[:200] + "...")
    
    # Test explanation prompt
    explanation_prompt = prompt_templates.get_explanation_prompt(
        user="User: Charlie, age: 35, preferences: movies",
        item="Movie: Inception",
        recommendation={
            'reason': 'Similar to movies you enjoyed',
            'context': {'genre': 'Sci-Fi', 'rating': 4.5},
            'audience': 'general',
            'tone': 'friendly',
            'length': 'medium',
            'key_points': ['genre', 'director', 'actors']
        }
    )
    print("\nExplanation Prompt (first 200 chars):")
    print(explanation_prompt[:200] + "...")
    
    # Test summarization prompt
    summarization_prompt = prompt_templates.get_summarization_prompt(
        text="This is a long text about user interactions and preferences...",
        max_length=50
    )
    print("\nSummarization Prompt (first 200 chars):")
    print(summarization_prompt[:200] + "...")
    
    # Test metapath verbalization
    metapath_prompt = prompt_templates.get_metapath_verbalization_prompt({
        'path': ['user_1', 'item_1', 'user_2', 'item_2'],
        'relations': ['interact', 'similar_pref', 'interact'],
        'path_type': 'user_item_user_item'
    })
    print("\nMetapath Verbalization Prompt (first 200 chars):")
    print(metapath_prompt[:200] + "...")
    
    # Get template usage statistics
    stats = prompt_templates.get_template_usage_stats()
    print(f"\nTemplate Usage Statistics: {stats}")
    
    # Test custom template creation
    custom_prompt = prompt_templates.create_custom_prompt(
        system_prompt="You are a helpful assistant.",
        user_prompt="Please help me with: {question}",
        name='custom_help',
        variables=['question']
    )
    print("\nCustom Prompt:")
    print(custom_prompt)