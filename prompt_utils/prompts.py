"""
Prompt Templates Module
Contains all prompt templates for LLM interactions across all three phases.
"""

from typing import Dict, List, Optional, Any
import json


class PromptTemplates:
    """
    Collection of prompt templates for different LLM tasks.
    
    Templates are organized by phase and task type:
    - Phase 1: Agent initialization, decisions, reflection, propagation
    - Phase 2: (No LLM calls - distillation training)
    - Phase 3: Ranking, explanation, memory refresh
    """
    
    def __init__(self):
        """Initialize prompt templates."""
        self._templates = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load all prompt templates."""
        self._templates = {
            # ===== Phase 1: Bootstrap Templates =====
            
            'user_summary': self._get_user_summary_template(),
            'item_summary': self._get_item_summary_template(),
            'decision': self._get_decision_template(),
            'reflection': self._get_reflection_template(),
            'propagation': self._get_propagation_template(),
            'memory_fusion': self._get_memory_fusion_template(),
            
            # ===== Phase 3: Inference Templates =====
            
            'ranking': self._get_ranking_template(),
            'ranking_with_explanation': self._get_ranking_with_explanation_template(),
            'memory_refresh': self._get_memory_refresh_template(),
            'cold_start_recommend': self._get_cold_start_recommend_template(),
            
            # ===== Review and Social Templates =====
            
            'review_positive': self._get_review_positive_template(),
            'review_negative': self._get_review_negative_template(),
            'discussion_start': self._get_discussion_start_template(),
            'discussion_response': self._get_discussion_response_template(),
            
            # ===== Advertisement Templates =====
            
            'ad_draft': self._get_ad_draft_template(),
            'ad_critique': self._get_ad_critique_template(),
            'ad_refinement': self._get_ad_refinement_template(),
        }
    
    # ============================================
    # Phase 1: Bootstrap Templates
    # ============================================
    
    def _get_user_summary_template(self) -> str:
        """Template for generating user intrinsic memory summary."""
        return """
You are creating a concise profile summary for a user in a recommendation system.

User Information:
{user_data}

Please generate a summary that captures:
1. The user's explicit preferences and interests
2. Any stated constraints or deal-breakers
3. Demographic context that might influence preferences
4. General taste profile (e.g., "mainstream pop fan", "niche indie explorer")

Guidelines:
- Keep the summary under 80 words
- Be specific and personalizable
- Avoid generic statements that apply to all users
- Focus on distinguishing characteristics

Output format:
User profile summary: [Your summary here]
"""
    
    def _get_item_summary_template(self) -> str:
        """Template for generating item intrinsic memory summary."""
        return """
You are creating a concise description summary for an item in a recommendation system.

Item Information:
{item_data}

Please generate a summary that captures:
1. The core identity and category of the item
2. Key features that distinguish it from similar items
3. The primary audience or use case
4. Any unique selling points

Guidelines:
- Keep the summary under 80 words
- Focus on factual, distinguishing characteristics
- Avoid marketing language or subjective claims
- Include specific details (genre, style, brand, specs)

Output format:
Item description summary: [Your summary here]
"""
    
    def _get_decision_template(self) -> str:
        """Template for agent decision making in Phase 1."""
        return """
You are a {agent_type} agent with the following profile and preferences:

{agent_memory}

You are considering the following item:

{item_memory}

{graph_context}

Based on your preferences and the provided context, please make a decision:

1. Would you interact with this item? (positive/negative)
2. What is your confidence level? (0.0 to 1.0)
3. Explain your reasoning, referencing specific aspects of your preferences and the item's features.

{hard_negative_note}

Respond in the following JSON format:
{{
    "decision": "positive" or "negative",
    "confidence": 0.0-1.0,
    "explanation": "Your detailed reasoning here",
    "key_factors": ["factor1", "factor2"],
    "graph_context_used": true/false
}}
"""
    
    def _get_reflection_template(self) -> str:
        """Template for collaborative reflection in Phase 1."""
        return """
You need to reflect on an incorrect recommendation decision and update your understanding.

YOUR CURRENT PROFILE:
{user_memory}

ITEM YOU INCORRECTLY {wrong_decision}:
{chosen_item_memory}

ITEM YOU SHOULD HAVE CHOSEN:
{correct_item_memory}

YOUR EXPLANATION FOR THE WRONG CHOICE:
{explanation}

GRAPH CONTEXT THAT WAS AVAILABLE:
{graph_context}

Please analyze:
1. Why did you make this incorrect choice? What specific features or patterns misled you?
2. What should you have noticed about the correct item that you missed?
3. How should your preferences be updated to avoid similar mistakes?

For the items involved:
- Update your understanding of what kinds of users each item appeals to
- Identify distinguishing features between the two items

Provide your reflection in the following JSON format:
{{
    "analysis": "Detailed analysis of the mistake",
    "missed_patterns": ["pattern1", "pattern2"],
    "new_user_preferences": ["preference1", "preference2"],
    "new_user_dislikes": ["dislike1", "dislike2"],
    "correct_item_insight": "What makes this item actually suitable",
    "incorrect_item_insight": "Why this item was actually unsuitable",
    "confidence_adjustment": -0.1 to 0.1,
    "suggested_graph_paths": ["path_type1", "path_type2"]
}}
"""
    
    def _get_propagation_template(self) -> str:
        """Template for lazy neighborhood propagation."""
        return """
You are updating your understanding based on signals from similar agents in your network.

YOUR CURRENT UNDERSTANDING:
{agent_memory}

SIGNALS FROM NEIGHBORING AGENTS:
{neighbor_signals}

These signals come from agents that share similarities with you through:
{propagation_paths}

Please update your understanding by:
1. Incorporating relevant new patterns from neighbors
2. Strengthening existing patterns that are confirmed by neighbors
3. Weakening patterns that contradict neighbor signals
4. Maintaining your unique characteristics

Guidelines:
- Only adopt patterns that are consistent with your core identity
- Give more weight to signals from highly similar neighbors
- Preserve patterns that are strongly confirmed by your own experience

Provide your updated understanding in JSON format:
{{
    "adopted_patterns": ["pattern1", "pattern2"],
    "strengthened_patterns": ["pattern3"],
    "weakened_patterns": ["pattern4"],
    "maintained_unique_patterns": ["pattern5"],
    "confidence_update": 0.0-0.1,
    "propagation_summary": "Brief summary of changes"
}}
"""
    
    def _get_memory_fusion_template(self) -> str:
        """Template for fusing multiple memory signals."""
        return """
You need to fuse multiple perspectives into a single, coherent understanding.

CURRENT UNDERSTANDING:
{current_memory}

NEW PERSPECTIVES TO FUSE:
{new_perspectives}

Please create an updated understanding that:
1. Incorporates the most important new information
2. Resolves any contradictions between perspectives
3. Maintains consistency with established patterns
4. Removes outdated or superseded information
5. Is concise and specific

Guidelines:
- Prioritize patterns confirmed by multiple perspectives
- Generalize specific observations into broader patterns when appropriate
- Keep the total description under {max_words} words
- Focus on actionable, distinguishing characteristics

Output format:
Fused understanding: [Your updated description here]
"""
    
    # ============================================
    # Phase 3: Inference Templates
    # ============================================
    
    def _get_ranking_template(self) -> str:
        """Template for LLM-based ranking in Phase 3."""
        return """
You are a recommendation system ranking items for a user.

USER PROFILE:
{user_memory}

CANDIDATE ITEMS:
{candidate_items}

GRAPH CONTEXT (from similar users and items):
{graph_context}

Please rank the top {top_k} items for this user from the candidates listed above.

For each recommended item, provide:
1. The item ID
2. A relevance score (0.0-1.0)
3. A brief explanation of why it matches the user's preferences

Consider:
- Direct preference matches
- Patterns from similar users (collaborative filtering)
- Item-to-item relationships
- The user's recent interaction history

Respond in the following JSON format:
{{
    "rankings": [
        {{
            "item_id": "item_xxx",
            "score": 0.95,
            "explanation": "This item matches the user's preference for..."
        }}
    ],
    "overall_strategy": "Brief description of ranking strategy used"
}}
"""
    
    def _get_ranking_with_explanation_template(self) -> str:
        """Template for user-facing ranking with detailed explanations."""
        return """
You are generating personalized recommendations with explanations for a real user.

USER CONTEXT:
{user_memory}

AVAILABLE ITEMS:
{candidate_items}

INSIGHTS FROM SIMILAR USERS:
{graph_context}

Please generate {top_k} personalized recommendations. For each recommendation:

1. Explain WHY this specific item matches the user's tastes
2. Reference specific aspects of the user's preferences
3. Mention similar items the user has enjoyed (if applicable)
4. Note what similar users thought about this item
5. Use natural, conversational language

Format each recommendation as:
---
**{rank}. {item_title}** (Score: {score})
{personalized_explanation}
---

Respond in JSON format:
{{
    "recommendations": [
        {{
            "rank": 1,
            "item_id": "item_xxx",
            "score": 0.95,
            "explanation": "Natural language explanation"
        }}
    ],
    "intro_message": "Optional personalized greeting",
    "theme": "Overall theme of recommendations"
}}
"""
    
    def _get_memory_refresh_template(self) -> str:
        """Template for refreshing agent memory from embeddings."""
        return """
You are regenerating your self-description from compressed representations.

CURRENT COMPRESSED STATE:
{embedding_summary}

PREVIOUS TEXT DESCRIPTION (may be outdated):
{previous_text}

Please regenerate a fresh text description that:
1. Captures the most important patterns from the compressed state
2. Is consistent with your core identity
3. Reflects any preference evolution
4. Is specific and distinguishing

Key patterns detected:
{key_patterns}

Guidelines:
- Keep under {max_words} words
- Be specific and personalizable
- Focus on what makes you unique
- Include both preferences and dislikes

Output format:
Refreshed self-description: [Your updated description here]
"""
    
    def _get_cold_start_recommend_template(self) -> str:
        """Template for cold-start recommendations."""
        return """
You are recommending items for a new user with limited interaction history.

NEW USER PROFILE:
{user_memory}

LIMITED INTERACTION HISTORY:
{interaction_history}

SIMILAR USERS' PREFERENCES:
{similar_users_context}

AVAILABLE ITEMS:
{candidate_items}

Since this user has limited history:
1. Rely more on explicit preferences and demographics
2. Use patterns from demographically similar users
3. Recommend popular, well-rated items in preferred categories
4. Include some diverse options for exploration
5. Be more conservative in niche recommendations

For each recommendation, explain how it relates to:
- The user's explicit preferences
- What similar users enjoyed
- Why it's a safe/good starting point

Respond in JSON format:
{{
    "recommendations": [...],
    "exploration_items": [...],
    "strategy": "cold_start"
}}
"""
    
    # ============================================
    # Review and Social Templates
    # ============================================
    
    def _get_review_positive_template(self) -> str:
        """Template for generating positive reviews."""
        return """
You are writing a review for an item you enjoyed.

YOUR PREFERENCES:
{user_preferences}

ITEM DETAILS:
{item_details}

YOUR EXPERIENCE:
{experience_summary}

Please write a review that:
1. Describes what you specifically enjoyed
2. Relates the item to your personal preferences
3. Mentions specific features that stood out
4. Is helpful for other users with similar tastes
5. Is authentic and personal

Guidelines:
- Keep under {max_words} words
- Be specific, not generic
- Mention both what you liked and any minor drawbacks
- Write naturally, as a real person would

Output format:
Review: [Your review here]
"""
    
    def _get_review_negative_template(self) -> str:
        """Template for generating negative reviews."""
        return """
You are writing a review for an item that did not meet your expectations.

YOUR PREFERENCES:
{user_preferences}

ITEM DETAILS:
{item_details}

YOUR EXPERIENCE:
{experience_summary}

Please write a review that:
1. Explains specifically what didn't work for you
2. Relates the disappointment to your personal preferences
3. Is fair and constructive (not just complaining)
4. Helps similar users avoid the same mismatch
5. Mentions if the item might work for different tastes

Guidelines:
- Keep under {max_words} words
- Be specific about what was lacking
- Acknowledge if the item has merits for other users
- Write constructively, not aggressively

Output format:
Review: [Your review here]
"""
    
    def _get_discussion_start_template(self) -> str:
        """Template for starting item discussions between agents."""
        return """
You are discussing an item with a friend who is considering purchasing it.

YOUR PROFILE:
{your_preferences}

YOUR EXPERIENCE WITH THE ITEM:
{your_experience}

ITEM DETAILS:
{item_details}

Start a conversation where you:
1. Share your honest opinion about the item
2. Ask about your friend's specific preferences
3. Offer to help them decide if it's right for them
4. Share specific pros and cons based on your experience

Be conversational and natural. This is a chat with a friend.

Output format:
Your message: [Your conversation starter here]
"""
    
    def _get_discussion_response_template(self) -> str:
        """Template for responding in item discussions."""
        return """
Continue the conversation about the item.

YOUR PROFILE:
{your_preferences}

FRIEND'S PROFILE:
{friend_preferences}

ITEM DETAILS:
{item_details}

CONVERSATION HISTORY:
{conversation_history}

FRIEND'S LATEST MESSAGE:
{friend_message}

Respond naturally, considering:
1. Your friend's specific tastes and concerns
2. Your own experience with the item
3. Whether you think it's a good match for them
4. Any alternatives you might suggest

Be helpful and conversational.

Output format:
Your response: [Your message here]
"""
    
    # ============================================
    # Advertisement Templates
    # ============================================
    
    def _get_ad_draft_template(self) -> str:
        """Template for generating advertisement drafts."""
        return """
You are writing an advertisement for the following item:

ITEM DETAILS:
{item_details}

TARGET AUDIENCE:
{target_audience}

KEY SELLING POINTS:
{selling_points}

ADVERTISING GOALS:
{ad_goals}

Please write an advertisement draft that:
1. Captures attention in the first sentence
2. Highlights the most compelling features
3. Speaks directly to the target audience
4. Includes a clear call to action
5. Is truthful and not misleading

Guidelines:
- Keep under {max_words} words
- Use persuasive but honest language
- Focus on benefits, not just features
- Create emotional connection

Output format:
Advertisement draft: [Your advertisement here]
"""
    
    def _get_ad_critique_template(self) -> str:
        """Template for critiquing advertisements."""
        return """
You are reviewing a colleague's advertisement draft from your area of expertise.

YOUR EXPERTISE: {expertise_area}

ITEM BEING ADVERTISED:
{item_details}

DRAFT ADVERTISEMENT:
{ad_draft}

From your expertise in {expertise_area}, provide constructive feedback:
1. What works well?
2. What could be improved regarding {expertise_area}?
3. Specific, actionable suggestions

Guidelines:
- Keep suggestions under {max_words} words
- Be specific and constructive
- Focus on your area of expertise
- Don't rewrite the ad, just suggest improvements

Output format:
My suggested revisions: [Your suggestions here]
"""
    
    def _get_ad_refinement_template(self) -> str:
        """Template for refining advertisements based on feedback."""
        return """
Refine your advertisement based on colleague feedback.

ORIGINAL DRAFT:
{original_draft}

COLLEAGUE FEEDBACK:
{feedback_list}

ITEM DETAILS:
{item_details}

Please create an improved version that:
1. Incorporates the most valuable feedback
2. Maintains your original vision and voice
3. Addresses all major concerns raised
4. Stays within the word limit

Guidelines:
- Keep under {max_words} words
- Prioritize changes that improve effectiveness
- You may ignore feedback that doesn't align with your vision

Output format:
Revised advertisement: [Your revised advertisement here]
"""
    
    # ============================================
    # Template Formatting Methods
    # ============================================
    
    def get_user_summary_prompt(self, user_data: Dict) -> str:
        """Format user summary prompt."""
        template = self._templates['user_summary']
        return template.format(user_data=json.dumps(user_data, indent=2))
    
    def get_item_summary_prompt(self, item_data: Dict) -> str:
        """Format item summary prompt."""
        template = self._templates['item_summary']
        return template.format(item_data=json.dumps(item_data, indent=2))
    
    def get_decision_prompt(self,
                            agent_type: str = 'user',
                            agent_memory: str = '',
                            item_memory: str = '',
                            graph_context: str = '',
                            is_hard_negative: bool = False) -> str:
        """Format decision prompt."""
        template = self._templates['decision']
        
        hard_negative_note = ""
        if is_hard_negative:
            hard_negative_note = """
NOTE: This is a challenging decision. The items may appear similar,
so pay close attention to subtle differences that match your preferences.
"""
        
        return template.format(
            agent_type=agent_type,
            agent_memory=agent_memory,
            item_memory=item_memory,
            graph_context=graph_context if graph_context else "No additional context available.",
            hard_negative_note=hard_negative_note
        )
    
    def get_reflection_prompt(self,
                              user_memory: str = '',
                              pos_item_memory: str = '',
                              neg_item_memory: str = '',
                              wrong_choice: str = '',
                              explanation: str = '',
                              graph_context: str = '') -> str:
        """Format collaborative reflection prompt."""
        template = self._templates['reflection']
        
        wrong_decision = 'CHOSE' if wrong_choice else 'REJECTED'
        
        return template.format(
            user_memory=user_memory,
            wrong_decision=wrong_decision,
            chosen_item_memory=neg_item_memory if wrong_choice else pos_item_memory,
            correct_item_memory=pos_item_memory if wrong_choice else neg_item_memory,
            explanation=explanation,
            graph_context=graph_context if graph_context else "No graph context was used."
        )
    
    def get_ranking_prompt(self,
                           user_memory: str = '',
                           candidate_items: str = '',
                           graph_context: str = '',
                           top_k: int = 10,
                           include_explanation: bool = False) -> str:
        """Format ranking prompt for Phase 3."""
        if include_explanation:
            template = self._templates['ranking_with_explanation']
        else:
            template = self._templates['ranking']
        
        return template.format(
            user_memory=user_memory,
            candidate_items=candidate_items,
            graph_context=graph_context if graph_context else "No collaborative context available.",
            top_k=top_k
        )
    
    def get_propagation_prompt(self,
                               agent_memory: str = '',
                               neighbor_signals: str = '',
                               propagation_paths: str = '') -> str:
        """Format propagation prompt."""
        template = self._templates['propagation']
        return template.format(
            agent_memory=agent_memory,
            neighbor_signals=neighbor_signals,
            propagation_paths=propagation_paths
        )
    
    def get_memory_fusion_prompt(self,
                                  current_memory: str = '',
                                  new_perspectives: str = '',
                                  max_words: int = 180) -> str:
        """Format memory fusion prompt."""
        template = self._templates['memory_fusion']
        return template.format(
            current_memory=current_memory,
            new_perspectives=new_perspectives,
            max_words=max_words
        )
    
    def get_review_prompt(self,
                          user_preferences: str = '',
                          item_details: str = '',
                          experience_summary: str = '',
                          is_positive: bool = True,
                          max_words: int = 80) -> str:
        """Format review generation prompt."""
        if is_positive:
            template = self._templates['review_positive']
        else:
            template = self._templates['review_negative']
        
        return template.format(
            user_preferences=user_preferences,
            item_details=item_details,
            experience_summary=experience_summary,
            max_words=max_words
        )
    
    def get_advertisement_prompts(self,
                                   item_details: str = '',
                                   target_audience: str = '',
                                   selling_points: str = '',
                                   ad_goals: str = 'Increase sales') -> Dict[str, str]:
        """Get all advertisement-related prompts."""
        return {
            'draft': self._templates['ad_draft'].format(
                item_details=item_details,
                target_audience=target_audience,
                selling_points=selling_points,
                ad_goals=ad_goals,
                max_words=150
            ),
            'critique_personalization': self._templates['ad_critique'].format(
                expertise_area='personalization',
                item_details=item_details,
                ad_draft='{ad_draft}',
                max_words=40
            ),
            'critique_creativity': self._templates['ad_critique'].format(
                expertise_area='creativity',
                item_details=item_details,
                ad_draft='{ad_draft}',
                max_words=40
            ),
            'critique_attractiveness': self._templates['ad_critique'].format(
                expertise_area='attractiveness and persuasion',
                item_details=item_details,
                ad_draft='{ad_draft}',
                max_words=40
            )
        }
    
    def get_cold_start_prompt(self,
                              user_memory: str = '',
                              interaction_history: str = '',
                              similar_users_context: str = '',
                              candidate_items: str = '') -> str:
        """Format cold-start recommendation prompt."""
        template = self._templates['cold_start_recommend']
        return template.format(
            user_memory=user_memory,
            interaction_history=interaction_history,
            similar_users_context=similar_users_context,
            candidate_items=candidate_items
        )
    
    def get_memory_refresh_prompt(self,
                                   embedding_summary: str = '',
                                   previous_text: str = '',
                                   key_patterns: str = '',
                                   max_words: int = 180) -> str:
        """Format memory refresh prompt."""
        template = self._templates['memory_refresh']
        return template.format(
            embedding_summary=embedding_summary,
            previous_text=previous_text,
            key_patterns=key_patterns,
            max_words=max_words
        )