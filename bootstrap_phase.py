"""
Bootstrap Phase: Full Text Agents with Graph RAG
Phase 1 of the Hybrid-GraphRAG pipeline.
"""

import argparse
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np

from agents_utils.user_agent import UserAgent
from agents_utils.item_agent import ItemAgent
from graph_utils.heterogeneous_graph import HeterogeneousGraph
from graph_utils.metapath_extractor import MetapathExtractor
from llm_utils.llm_client import LLMClient
from llm_utils.embedding_client import EmbeddingClient
from prompt_utils.prompts import PromptTemplates
from prompt_utils.context_builder import ContextBuilder
from datasets_utils.amazon_loader import AmazonDataLoader
from hierarchy_memory_utils.intrinsic_memory import (
    create_intrinsic_memory_from_item,
    create_intrinsic_memory_from_user
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BootstrapPhase:
    """
    Phase 1: Bootstrap with full text Graph RAG.
    Collects reflection traces for Phase 2 distillation.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize bootstrap phase.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.bootstrap_config = config['bootstrap']
        
        # Initialize components
        self.llm_client = LLMClient()
        self.embedding_client = EmbeddingClient()
        self.prompt_templates = PromptTemplates()
        self.context_builder = ContextBuilder()
        self.metapath_extractor = MetapathExtractor()
        
        # Storage
        self.user_agents: Dict[str, UserAgent] = {}
        self.item_agents: Dict[str, ItemAgent] = {}
        self.graph = HeterogeneousGraph()
        self.reflection_traces: List[Dict] = []
        
        # Statistics
        self.stats = {
            'total_interactions': 0,
            'correct_decisions': 0,
            'reflections_performed': 0,
            'llm_calls': 0,
            'total_cost': 0.0
        }
    
    def initialize_agents(self, users_data: Dict, items_data: Dict):
        """
        Initialize user and item agents with hierarchical memory.
        
        Args:
            users_data: Dictionary of user_id -> user features
            items_data: Dictionary of item_id -> item features
        """
        logger.info(f"Initializing {len(users_data)} user agents...")
        for user_id, user_data in users_data.items():
            intrinsic_memory = create_intrinsic_memory_from_user(user_data, user_id)
            intrinsic_memory.summary_text = self._generate_user_summary(user_data)
            
            # Encode intrinsic memory
            embedding = self.embedding_client.encode(intrinsic_memory.to_prompt_text())
            intrinsic_memory.set_embedding(embedding.tolist())
            
            agent = UserAgent(
                agent_id=user_id,
                intrinsic_memory=intrinsic_memory
            )
            self.user_agents[user_id] = agent
            self.graph.add_user(user_id)
        
        logger.info(f"Initializing {len(items_data)} item agents...")
        for item_id, item_data in items_data.items():
            intrinsic_memory = create_intrinsic_memory_from_item(item_data, item_id)
            intrinsic_memory.summary_text = self._generate_item_summary(item_data)
            
            # Encode intrinsic memory
            embedding = self.embedding_client.encode(intrinsic_memory.to_prompt_text())
            intrinsic_memory.set_embedding(embedding.tolist())
            
            agent = ItemAgent(
                agent_id=item_id,
                intrinsic_memory=intrinsic_memory
            )
            self.item_agents[item_id] = agent
            self.graph.add_item(item_id)
    
    def _generate_user_summary(self, user_data: Dict) -> str:
        """Generate summary text for user intrinsic memory."""
        prompt = self.prompt_templates.get_user_summary_prompt(user_data)
        response = self.llm_client.generate(prompt, max_tokens=100)
        self.stats['llm_calls'] += 1
        self.stats['total_cost'] += response.cost_estimate
        return response.text
    
    def _generate_item_summary(self, item_data: Dict) -> str:
        """Generate summary text for item intrinsic memory."""
        prompt = self.prompt_templates.get_item_summary_prompt(item_data)
        response = self.llm_client.generate(prompt, max_tokens=100)
        self.stats['llm_calls'] += 1
        self.stats['total_cost'] += response.cost_estimate
        return response.text
    
    def construct_graph(self):
        """Build the initial heterogeneous graph with similarity edges."""
        logger.info("Constructing heterogeneous graph...")
        
        graph_config = self.bootstrap_config['graph']
        
        # Add interaction edges from training data
        # (These would come from the actual dataset)
        
        # Add user-user similarity edges
        user_ids = list(self.user_agents.keys())
        for i, u1 in enumerate(user_ids):
            for u2 in user_ids[i+1:]:
                sim = self._compute_user_similarity(u1, u2)
                if sim > graph_config['similarity_threshold_user']:
                    self.graph.add_edge(u1, u2, 'similar_pref', weight=sim)
        
        # Add item-item content similarity edges
        item_ids = list(self.item_agents.keys())
        for i, item1 in enumerate(item_ids):
            for j, item2 in enumerate(item_ids[i+1:i+50]):  # Limit comparisons
                sim = self._compute_item_similarity(item1, item2)
                if sim > graph_config['similarity_threshold_item']:
                    self.graph.add_edge(item1, item2, 'content_sim', weight=sim)
        
        logger.info(f"Graph constructed: {self.graph.get_graph_statistics()}")
    
    def _compute_user_similarity(self, user1_id: str, user2_id: str) -> float:
        """Compute similarity between two users."""
        agent1 = self.user_agents[user1_id]
        agent2 = self.user_agents[user2_id]
        
        text1 = agent1.intrinsic_memory.to_prompt_text() + " " + \
                agent1.collaborative_memory.to_prompt_text()
        text2 = agent2.intrinsic_memory.to_prompt_text() + " " + \
                agent2.collaborative_memory.to_prompt_text()
        
        return self.embedding_client.compute_similarity(text1, text2)
    
    def _compute_item_similarity(self, item1_id: str, item2_id: str) -> float:
        """Compute similarity between two items."""
        agent1 = self.item_agents[item1_id]
        agent2 = self.item_agents[item2_id]
        
        text1 = agent1.intrinsic_memory.to_prompt_text()
        text2 = agent2.intrinsic_memory.to_prompt_text()
        
        return self.embedding_client.compute_similarity(text1, text2)
    
    def run_interaction_step(self, 
                             user_id: str, 
                             positive_item_id: str,
                             negative_item_id: str) -> Dict:
        """
        Run a single interaction step with Graph RAG retrieval and reflection.
        
        Args:
            user_id: User agent ID
            positive_item_id: Ground truth positive item
            negative_item_id: Sampled negative item
        
        Returns:
            Reflection trace dictionary
        """
        user_agent = self.user_agents[user_id]
        pos_item = self.item_agents[positive_item_id]
        neg_item = self.item_agents[negative_item_id]
        
        # Step 1: Graph RAG retrieval
        rag_config = self.bootstrap_config['graph_rag']
        graph_context = self.metapath_extractor.extract_context(
            user_id=user_id,
            candidate_ids=[positive_item_id, negative_item_id],
            graph=self.graph,
            max_hops=rag_config['max_hops'],
            top_k=rag_config['top_k_paths']
        )
        
        # Step 2: Build context-aware prompt
        context_text = self.context_builder.build_context_prompt(
            graph_context=graph_context,
            user_memory=user_agent.get_full_memory_text(),
            pos_item_memory=pos_item.get_full_memory_text(),
            neg_item_memory=neg_item.get_full_memory_text()
        )
        
        # Step 3: Agent decision
        decision_prompt = self.prompt_templates.get_decision_prompt(context_text)
        response = self.llm_client.generate(decision_prompt)
        self.stats['llm_calls'] += 1
        self.stats['total_cost'] += response.cost_estimate
        
        # Parse decision
        decision = self._parse_decision(response.text)
        is_correct = (decision['chosen_item'] == positive_item_id)
        
        self.stats['total_interactions'] += 1
        if is_correct:
            self.stats['correct_decisions'] += 1
        
        # Step 4: Collaborative reflection (if incorrect)
        reflection_result = None
        if not is_correct:
            reflection_prompt = self.prompt_templates.get_reflection_prompt(
                user_memory=user_agent.get_full_memory_text(),
                pos_item_memory=pos_item.get_full_memory_text(),
                neg_item_memory=neg_item.get_full_memory_text(),
                wrong_choice=decision['chosen_item'],
                explanation=decision['explanation']
            )
            
            reflection_response = self.llm_client.generate(reflection_prompt)
            self.stats['llm_calls'] += 1
            self.stats['total_cost'] += reflection_response.cost_estimate
            self.stats['reflections_performed'] += 1
            
            # Update agent memories
            reflection_result = self._apply_reflection(
                user_agent, pos_item, neg_item, reflection_response.text
            )
            
            # Update graph edge weights
            self.graph.update_edge_weight(
                user_id, positive_item_id, 'interact', delta=0.1
            )
        
        # Step 5: Collect trace
        trace = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'positive_item_id': positive_item_id,
            'negative_item_id': negative_item_id,
            'graph_context': graph_context,
            'decision': decision,
            'is_correct': is_correct,
            'reflection_result': reflection_result,
            'influential_paths': graph_context.get('influential_paths', []),
            'user_memory_before': user_agent.to_dict(),
            'pos_item_memory_before': pos_item.to_dict(),
            'neg_item_memory_before': neg_item.to_dict()
        }
        
        self.reflection_traces.append(trace)
        
        # Update interaction memories
        user_agent.interaction_memory.add_interaction(
            partner_id=positive_item_id,
            partner_type='item',
            decision='positive' if is_correct else 'negative',
            is_correct=is_correct,
            explanation=decision.get('explanation', ''),
            graph_context=graph_context
        )
        
        return trace
    
    def _parse_decision(self, llm_output: str) -> Dict:
        """Parse LLM decision output."""
        # Simplified parsing - in production, use structured output
        return {
            'chosen_item': 'item_1' if 'first' in llm_output.lower() else 'item_2',
            'explanation': llm_output[:200],
            'confidence': 0.8
        }
    
    def _apply_reflection(self, 
                          user_agent: 'UserAgent',
                          pos_item: 'ItemAgent',
                          neg_item: 'ItemAgent',
                          reflection_text: str) -> Dict:
        """Apply reflection results to agent memories."""
        # Parse reflection to extract new patterns
        # Simplified - in production, parse structured output
        
        # Update user preferences
        user_agent.collaborative_memory.update_from_reflection(
            new_patterns=["Discovered preference from reflection"],
            new_dislikes=["Discovered dislike from reflection"],
            partner_id=pos_item.agent_id
        )
        
        # Update item collaborative memory
        pos_item.collaborative_memory.update_from_reflection(
            new_patterns=["User type that likes this item"],
            new_dislikes=[],
            partner_id=user_agent.agent_id
        )
        
        return {
            'user_update': 'applied',
            'pos_item_update': 'applied',
            'reflection_summary': reflection_text[:100]
        }
    
    def run(self, 
            users_data: Dict, 
            items_data: Dict,
            interactions: List[Tuple[str, str]],
            max_steps: int = 3) -> Dict:
        """
        Run the complete bootstrap phase.
        
        Args:
            users_data: User features
            items_data: Item features
            interactions: List of (user_id, item_id) pairs
            max_steps: Maximum optimization steps per interaction
        
        Returns:
            Bootstrap results including traces and statistics
        """
        logger.info("Starting bootstrap phase...")
        
        # Initialize agents and graph
        self.initialize_agents(users_data, items_data)
        self.construct_graph()
        
        # Process interactions
        for step in range(max_steps):
            logger.info(f"Optimization step {step + 1}/{max_steps}")
            
            for user_id, item_id in interactions[:100]:  # Limit for demo
                # Sample negative
                negative_id = self._sample_negative(user_id, item_id)
                
                if negative_id:
                    trace = self.run_interaction_step(user_id, item_id, negative_id)
            
            # Lazy graph propagation
            self._propagate_signals()
            
            # Log progress
            accuracy = self.stats['correct_decisions'] / max(1, self.stats['total_interactions'])
            logger.info(f"  Accuracy: {accuracy:.3f}, "
                       f"Reflections: {self.stats['reflections_performed']}, "
                       f"LLM Calls: {self.stats['llm_calls']}")
        
        # Save results
        return {
            'traces': self.reflection_traces,
            'statistics': self.stats,
            'graph_stats': self.graph.get_graph_statistics(),
            'num_agents': {
                'users': len(self.user_agents),
                'items': len(self.item_agents)
            }
        }
    
    def _sample_negative(self, user_id: str, positive_id: str) -> Optional[str]:
        """Sample a negative item for contrastive learning."""
        # Simplified random sampling
        available = [iid for iid in self.item_agents.keys() 
                    if iid != positive_id]
        if available:
            return np.random.choice(available)
        return None
    
    def _propagate_signals(self):
        """Perform lazy neighborhood propagation."""
        threshold = self.bootstrap_config['lazy_propagation_threshold']
        # Simplified - check accumulated signals and propagate
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiment_launch_confg.yaml')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data (mock for demo)
    users_data = {
        f'user_{i}': {'preferences': ['rock', 'jazz'], 'constraints': []}
        for i in range(config['bootstrap']['num_users'])
    }
    
    items_data = {
        f'item_{i}': {
            'title': f'Product {i}',
            'category': np.random.choice(['Books', 'Music', 'Electronics']),
            'description': f'Description for item {i}'
        }
        for i in range(100)
    }
    
    interactions = [(f'user_{i}', f'item_{i % 100}') for i in range(500)]
    
    # Run bootstrap phase
    bootstrap = BootstrapPhase(config)
    results = bootstrap.run(users_data, items_data, interactions)
    
    # Save results
    output_dir = Path(config['output']['save_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'bootstrap_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Bootstrap phase complete. Results saved to {output_dir}")
    logger.info(f"Statistics: {results['statistics']}")


if __name__ == '__main__':
    main()