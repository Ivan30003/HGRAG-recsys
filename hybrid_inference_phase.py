"""
Hybrid Inference Phase: Adaptive Gating with Dual-Path Execution
Phase 3 of the Hybrid-GraphRAG pipeline.

Deploys trained models with adaptive gating between 
efficient GNN path and full LLM Graph RAG path.
"""

import argparse
import yaml
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
from collections import defaultdict
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import ndcg_score

from gnn_utils.hgnn import HeterogeneousGNN, LightDecoder
from graph_utils.heterogeneous_graph import HeterogeneousGraph
from graph_utils.metapath_extractor import MetapathExtractor
from llm_utils.llm_client import LLMClient
from llm_utils.embedding_client import EmbeddingClient
from prompt_utils.prompts import PromptTemplates
from prompt_utils.context_builder import ContextBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptiveGate:
    """
    Adaptive gating mechanism for routing decisions.
    Learns when to use GNN vs LLM path.
    """
    
    def __init__(self, 
                 threshold: float = 0.3,
                 weights: Optional[np.ndarray] = None,
                 bias: float = -0.5):
        """
        Initialize adaptive gate.
        
        Args:
            threshold: Decision threshold (0-1)
            weights: Feature weights [confidence, density, criticality, staleness]
            bias: Bias term
        """
        self.threshold = threshold
        
        if weights is None:
            # Default weights learned from Phase 1 traces
            self.weights = np.array([0.35, 0.25, 0.25, 0.15])
        else:
            self.weights = weights
        
        self.bias = bias
        
        # Statistics
        self.gnn_path_count = 0
        self.llm_path_count = 0
        self.total_decisions = 0
    
    def compute_gate_score(self,
                           confidence: float,
                           density: float,
                           criticality: float,
                           staleness: float) -> float:
        """
        Compute gating score from four features.
        
        Args:
            confidence: GNN prediction confidence (0-1, higher = more confident)
            density: Graph neighborhood density (0-1)
            criticality: Whether decision needs explanation (0 or 1)
            staleness: Time since last text refresh (0-1)
        
        Returns:
            Gate score (0-1). Score > threshold -> LLM path
        """
        features = np.array([confidence, density, criticality, staleness])
        score = np.dot(self.weights, features) + self.bias
        
        # Apply sigmoid
        score = 1.0 / (1.0 + np.exp(-score))
        
        return score
    
    def decide_path(self,
                    confidence: float,
                    density: float,
                    criticality: bool,
                    staleness: float) -> str:
        """
        Decide which path to use.
        
        Args:
            confidence: GNN confidence
            density: Graph density
            criticality: Is user-facing?
            staleness: Memory staleness
        
        Returns:
            'llm' or 'gnn'
        """
        crit_val = 1.0 if criticality else 0.0
        score = self.compute_gate_score(confidence, density, crit_val, staleness)
        
        self.total_decisions += 1
        
        if score > self.threshold:
            self.llm_path_count += 1
            return 'llm'
        else:
            self.gnn_path_count += 1
            return 'gnn'
    
    def get_statistics(self) -> Dict:
        """Get gating statistics."""
        total = max(1, self.total_decisions)
        return {
            'total_decisions': self.total_decisions,
            'gnn_path_ratio': self.gnn_path_count / total,
            'llm_path_ratio': self.llm_path_count / total,
            'gnn_count': self.gnn_path_count,
            'llm_count': self.llm_path_count,
            'threshold': self.threshold,
            'weights': self.weights.tolist()
        }


class HybridInferenceEngine:
    """
    Phase 3: Hybrid inference with adaptive gating.
    Combines efficient GNN path with full LLM Graph RAG.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize hybrid inference engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.inference_config = config['inference']
        self.gating_config = self.inference_config['gating']
        
        # Set device
        self.device = torch.device(
            config['experiment']['device'] 
            if torch.cuda.is_available() 
            else 'cpu'
        )
        
        # Initialize adaptive gate
        self.gate = AdaptiveGate(
            threshold=self.gating_config['default_threshold']
        )
        
        # Initialize components
        self.llm_client = LLMClient()
        self.embedding_client = EmbeddingClient()
        self.prompt_templates = PromptTemplates()
        self.context_builder = ContextBuilder()
        self.metapath_extractor = MetapathExtractor()
        
        # Models (loaded from Phase 2)
        self.hgnn: Optional[HeterogeneousGNN] = None
        self.decoder: Optional[LightDecoder] = None
        
        # Graph
        self.graph = HeterogeneousGraph()
        
        # Agent storage
        self.user_agents: Dict = {}
        self.item_agents: Dict = {}
        
        # Statistics
        self.stats = {
            'total_recommendations': 0,
            'gnn_path_calls': 0,
            'llm_path_calls': 0,
            'total_llm_cost': 0.0,
            'total_latency': 0.0,
            'cold_start_improvements': []
        }
    
    def load_models(self, checkpoint_path: str):
        """
        Load trained HGNN and decoder from Phase 2.
        
        Args:
            checkpoint_path: Path to model checkpoint
        """
        logger.info(f"Loading models from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize HGNN
        hgnn_config = self.config['distillation']['hgnn']
        self.hgnn = HeterogeneousGNN(
            input_dim=hgnn_config['input_dim'],
            hidden_dim=hgnn_config['hidden_dim'],
            output_dim=hgnn_config['output_dim'],
            num_layers=hgnn_config['num_layers'],
            dropout=0.0  # No dropout during inference
        ).to(self.device)
        
        self.hgnn.load_state_dict(checkpoint['hgnn_state_dict'])
        self.hgnn.eval()
        
        # Initialize decoder
        self.decoder = LightDecoder(
            embedding_dim=hgnn_config['output_dim'],
            hidden_dim=512,
            vocab_size=50000,
            max_length=100
        ).to(self.device)
        
        if 'decoder_state_dict' in checkpoint:
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.decoder.eval()
        
        logger.info("Models loaded successfully")
    
    def gnn_path_recommend(self,
                           user_id: str,
                           candidate_ids: List[str],
                           top_k: int = 10) -> Tuple[List[str], List[float]]:
        """
        Generate recommendations using efficient GNN path.
        
        Args:
            user_id: User agent ID
            candidate_ids: List of candidate item IDs
            top_k: Number of recommendations to return
        
        Returns:
            Tuple of (ranked_item_ids, scores)
        """
        self.stats['gnn_path_calls'] += 1
        start_time = time.time()
        
        # Get user embeddings
        user_int_emb = self.graph.get_node_embedding(user_id, 'intrinsic')
        user_col_emb = self.graph.get_node_embedding(user_id, 'collaborative')
        user_pref_emb = self.graph.get_node_embedding(user_id, 'preference')
        
        if user_col_emb is None:
            # Fallback: use intrinsic embedding
            user_col_emb = user_int_emb
        
        user_col_tensor = torch.tensor(user_col_emb, device=self.device)
        user_int_tensor = torch.tensor(user_int_emb, device=self.device) if user_int_emb is not None else user_col_tensor
        
        # Get graph context embedding (aggregated neighbors)
        context_emb = self._get_graph_context_embedding(user_id)
        
        # Score candidates
        scores = []
        for item_id in candidate_ids:
            item_int_emb = self.graph.get_node_embedding(item_id, 'intrinsic')
            item_col_emb = self.graph.get_node_embedding(item_id, 'collaborative')
            
            if item_col_emb is None:
                item_col_emb = item_int_emb
            
            item_col_tensor = torch.tensor(item_col_emb, device=self.device)
            item_int_tensor = torch.tensor(item_int_emb, device=self.device) if item_int_emb is not None else item_col_tensor
            
            # Compute score components
            collab_score = torch.dot(user_col_tensor, item_col_tensor).item()
            content_score = torch.dot(user_int_tensor, item_int_tensor).item()
            context_score = torch.dot(context_emb, item_col_tensor).item() if context_emb is not None else 0.0
            
            # Weighted combination
            gnn_config = self.inference_config['gnn_path']
            total_score = (
                gnn_config['collaborative_weight'] * collab_score +
                gnn_config['content_weight'] * content_score +
                gnn_config['graph_context_weight'] * context_score
            )
            
            scores.append((item_id, total_score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        ranked_items = [item_id for item_id, _ in scores[:top_k]]
        ranked_scores = [score for _, score in scores[:top_k]]
        
        latency = time.time() - start_time
        self.stats['total_latency'] += latency
        
        return ranked_items, ranked_scores
    
    def _get_graph_context_embedding(self, 
                                      user_id: str) -> Optional[torch.Tensor]:
        """
        Get aggregated graph context embedding for a user.
        
        Args:
            user_id: User agent ID
        
        Returns:
            Aggregated context embedding tensor
        """
        # Get similar users
        similar_users = self.graph.adjacency.get('similar_pref', {}).get(user_id, set())
        
        if not similar_users:
            return None
        
        # Aggregate their collaborative embeddings
        embeddings = []
        for su_id in list(similar_users)[:10]:  # Top 10 similar users
            emb = self.graph.get_node_embedding(su_id, 'collaborative')
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            return None
        
        # Average pooling
        avg_emb = np.mean(embeddings, axis=0)
        return torch.tensor(avg_emb, device=self.device)
    
    def llm_path_recommend(self,
                           user_id: str,
                           candidate_ids: List[str],
                           top_k: int = 10) -> Tuple[List[str], List[float], str]:
        """
        Generate recommendations using full LLM Graph RAG path.
        
        Args:
            user_id: User agent ID
            candidate_ids: List of candidate item IDs
            top_k: Number of recommendations
        
        Returns:
            Tuple of (ranked_item_ids, scores, explanation)
        """
        self.stats['llm_path_calls'] += 1
        start_time = time.time()
        
        # Get agent information
        user_agent = self.user_agents.get(user_id)
        
        # Extract graph context via metapath
        graph_context = self.metapath_extractor.extract_context(
            user_id=user_id,
            candidate_ids=candidate_ids,
            graph=self.graph,
            max_hops=2,
            top_k=15
        )
        
        # Build LLM prompt
        context_text = self.context_builder.build_inference_prompt(
            user_id=user_id,
            user_memory=self._get_user_memory_text(user_id),
            candidate_items=self._get_candidate_items_text(candidate_ids),
            graph_context=graph_context
        )
        
        # Get LLM recommendation
        prompt = self.prompt_templates.get_ranking_prompt(context_text, top_k)
        response = self.llm_client.generate(prompt)
        
        self.stats['total_llm_cost'] += response.cost_estimate
        
        # Parse LLM output
        ranked_items, scores, explanation = self._parse_llm_ranking(
            response.text, candidate_ids, top_k
        )
        
        # Update memory staleness
        self._update_memory_sync(user_id)
        
        latency = time.time() - start_time
        self.stats['total_latency'] += latency
        
        return ranked_items, scores, explanation
    
    def _get_user_memory_text(self, user_id: str) -> str:
        """Get user memory text."""
        agent = self.user_agents.get(user_id)
        if agent:
            return agent.get('full_memory_text', f'User {user_id}')
        return f'User {user_id}'
    
    def _get_candidate_items_text(self, candidate_ids: List[str]) -> Dict[str, str]:
        """Get text descriptions for candidate items."""
        items_text = {}
        for item_id in candidate_ids:
            agent = self.item_agents.get(item_id)
            if agent:
                items_text[item_id] = agent.get('description', f'Item {item_id}')
            else:
                items_text[item_id] = f'Item {item_id}'
        return items_text
    
    def _parse_llm_ranking(self, 
                           llm_output: str,
                           candidate_ids: List[str],
                           top_k: int) -> Tuple[List[str], List[float], str]:
        """
        Parse LLM ranking output.
        
        Args:
            llm_output: Raw LLM response
            candidate_ids: Original candidate IDs
            top_k: Number of items to extract
        
        Returns:
            Tuple of (ranked_items, scores, explanation)
        """
        # Simplified parsing
        # In production, use structured output
        ranked = candidate_ids[:top_k]
        scores = [1.0 - i/len(candidate_ids) for i in range(top_k)]
        explanation = llm_output[:500]
        
        return ranked, scores, explanation
    
    def _update_memory_sync(self, user_id: str):
        """Synchronize text and embedding representations."""
        sync_period = self.gating_config.get('sync_period', 20)
        
        # Track sync counter
        if not hasattr(self, '_sync_counters'):
            self._sync_counters = defaultdict(int)
        
        self._sync_counters[user_id] += 1
        
        if self._sync_counters[user_id] >= sync_period:
            # Regenerate embeddings from text
            self._sync_embeddings_to_text(user_id)
            self._sync_counters[user_id] = 0
    
    def _sync_embeddings_to_text(self, user_id: str):
        """Update embeddings to match current text memory."""
        user_agent = self.user_agents.get(user_id)
        if user_agent:
            text = user_agent.get('full_memory_text', '')
            embedding = self.embedding_client.encode(text)
            self.graph.set_node_embedding(user_id, 'collaborative', embedding)
    
    def recommend(self,
                  user_id: str,
                  candidate_ids: List[str],
                  top_k: int = 10,
                  is_user_facing: bool = False) -> Dict:
        """
        Generate recommendations with adaptive gating.
        
        Args:
            user_id: User agent ID
            candidate_ids: Candidate item IDs
            top_k: Number of recommendations
            is_user_facing: Whether this is a user-facing request
        
        Returns:
            Recommendation result dictionary
        """
        self.stats['total_recommendations'] += 1
        
        # Compute gating features
        confidence = self._compute_gnn_confidence(user_id, candidate_ids)
        density = self._compute_graph_density(user_id)
        staleness = self._compute_memory_staleness(user_id)
        
        # Decide path
        path = self.gate.decide_path(
            confidence=confidence,
            density=density,
            criticality=is_user_facing,
            staleness=staleness
        )
        
        if path == 'gnn':
            ranked_items, scores = self.gnn_path_recommend(
                user_id, candidate_ids, top_k
            )
            explanation = f"GNN-based recommendation (confidence: {confidence:.2f})"
            path_used = 'gnn'
        else:
            ranked_items, scores, explanation = self.llm_path_recommend(
                user_id, candidate_ids, top_k
            )
            path_used = 'llm'
        
        return {
            'user_id': user_id,
            'ranked_items': ranked_items,
            'scores': scores,
            'explanation': explanation,
            'path_used': path_used,
            'gate_features': {
                'confidence': confidence,
                'density': density,
                'staleness': staleness,
                'is_user_facing': is_user_facing
            }
        }
    
    def _compute_gnn_confidence(self, user_id: str, candidate_ids: List[str]) -> float:
        """
        Compute GNN prediction confidence.
        
        Args:
            user_id: User agent ID
            candidate_ids: Candidate item IDs
        
        Returns:
            Confidence score (0-1, higher = more confident)
        """
        user_col_emb = self.graph.get_node_embedding(user_id, 'collaborative')
        
        if user_col_emb is None:
            return 0.1  # Very low confidence without embeddings
        
        # Compute entropy of scores across candidates
        user_tensor = torch.tensor(user_col_emb, device=self.device)
        scores = []
        
        for item_id in candidate_ids[:20]:  # Sample for efficiency
            item_emb = self.graph.get_node_embedding(item_id, 'collaborative')
            if item_emb is not None:
                item_tensor = torch.tensor(item_emb, device=self.device)
                score = torch.dot(user_tensor, item_tensor).item()
                scores.append(score)
        
        if not scores:
            return 0.5
        
        # Normalize scores
        scores = np.array(scores)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        # Confidence = 1 - entropy
        entropy = -np.sum(scores * np.log(scores + 1e-8)) / np.log(len(scores))
        confidence = 1.0 - entropy
        
        return float(confidence)
    
    def _compute_graph_density(self, user_id: str) -> float:
        """
        Compute graph neighborhood density.
        
        Args:
            user_id: User agent ID
        
        Returns:
            Density score (0-1, higher = denser neighborhood)
        """
        total_neighbors = 0
        for edge_type in ['interact', 'similar_pref']:
            neighbors = self.graph.adjacency.get(edge_type, {}).get(user_id, set())
            total_neighbors += len(neighbors)
        
        # Normalize
        max_neighbors = 50
        density = min(1.0, total_neighbors / max_neighbors)
        
        return density
    
    def _compute_memory_staleness(self, user_id: str) -> float:
        """
        Compute memory staleness.
        
        Args:
            user_id: User agent ID
        
        Returns:
            Staleness score (0-1, higher = more stale)
        """
        user_agent = self.user_agents.get(user_id, {})
        last_updated = user_agent.get('last_updated')
        
        if last_updated is None:
            return 1.0  # Very stale
        
        # Parse timestamp
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated)
        
        # Hours since update
        hours_since = (datetime.now() - last_updated).total_seconds() / 3600
        
        # Staleness = 1 - exp(-lambda * time)
        staleness = 1.0 - np.exp(-0.1 * hours_since)
        
        return float(staleness)
    
    def evaluate(self,
                test_users: List[str],
                test_interactions: Dict[str, str],
                all_items: List[str],
                k_values: List[int] = [1, 5, 10],
                num_negatives: int = 99) -> Dict:
        """
        Evaluate recommendation performance.
        
        Args:
            test_users: List of test user IDs
            test_interactions: Dict mapping user_id -> ground truth item_id
            all_items: List of all item IDs
            k_values: K values for NDCG
            num_negatives: Number of negative samples per user
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating on {len(test_users)} test users...")
        
        metrics = defaultdict(list)
        cold_start_results = []
        
        for user_id in test_users:
            ground_truth = test_interactions.get(user_id)
            if not ground_truth:
                continue
            
            # Sample negative items
            negatives = self._sample_negatives(user_id, ground_truth, all_items, num_negatives)
            
            # Create candidate list (positive + negatives)
            candidates = [ground_truth] + negatives
            np.random.shuffle(candidates)
            
            # Get recommendations
            result = self.recommend(
                user_id=user_id,
                candidate_ids=candidates,
                top_k=max(k_values),
                is_user_facing=False  # Evaluation is backend
            )
            
            # Compute metrics
            ranked_items = result['ranked_items']
            
            for k in k_values:
                # Hit Rate
                hit = 1 if ground_truth in ranked_items[:k] else 0
                metrics[f'HR@{k}'].append(hit)
                
                # NDCG
                relevance = [1 if item == ground_truth else 0 for item in ranked_items[:k]]
                if sum(relevance) > 0:
                    ndcg = ndcg_score([relevance], [list(range(k, 0, -1))])
                    metrics[f'NDCG@{k}'].append(ndcg)
                else:
                    metrics[f'NDCG@{k}'].append(0.0)
            
            # MRR
            try:
                rank = ranked_items.index(ground_truth) + 1
                metrics['MRR'].append(1.0 / rank)
            except ValueError:
                metrics['MRR'].append(0.0)
            
            # Cold-start analysis
            interaction_count = self._get_user_interaction_count(user_id)
            cold_start_results.append({
                'user_id': user_id,
                'interaction_count': interaction_count,
                'path_used': result['path_used'],
                'hit': hit
            })
        
        # Aggregate metrics
        results = {}
        for metric_name, values in metrics.items():
            results[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # Cold-start breakdown
        cold_thresholds = self.config['evaluation'].get('cold_start', {}).get('thresholds', [5, 10])
        for threshold in cold_thresholds:
            cold_users = [r for r in cold_start_results if r['interaction_count'] < threshold]
            warm_users = [r for r in cold_start_results if r['interaction_count'] >= threshold]
            
            if cold_users:
                results[f'cold_start_lt{threshold}_HR@10'] = np.mean([r['hit'] for r in cold_users])
            if warm_users:
                results[f'warm_ge{threshold}_HR@10'] = np.mean([r['hit'] for r in warm_users])
        
        # Add gating statistics
        results['gate_stats'] = self.gate.get_statistics()
        results['cost_stats'] = {
            'total_llm_cost': self.stats['total_llm_cost'],
            'total_latency': self.stats['total_latency'],
            'avg_latency_per_rec': self.stats['total_latency'] / max(1, self.stats['total_recommendations'])
        }
        
        return results
    
    def _sample_negatives(self, 
                          user_id: str, 
                          positive_id: str,
                          all_items: List[str],
                          num_negatives: int) -> List[str]:
        """Sample negative items for evaluation."""
        available = [i for i in all_items if i != positive_id]
        
        if len(available) < num_negatives:
            return available
        
        return list(np.random.choice(available, num_negatives, replace=False))
    
    def _get_user_interaction_count(self, user_id: str) -> int:
        """Get number of interactions for a user."""
        neighbors = self.graph.adjacency.get('interact', {}).get(user_id, set())
        return len(neighbors)
    
    def get_statistics(self) -> Dict:
        """Get comprehensive inference statistics."""
        return {
            **self.stats,
            'gate_stats': self.gate.get_statistics()
        }


def main():
    parser = argparse.ArgumentParser(description='Phase 3: Hybrid Inference')
    parser.add_argument('--config', type=str, default='experiment_launch_confg.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint from Phase 2')
    parser.add_argument('--bootstrap_results', type=str, default='results/bootstrap_results.json',
                       help='Path to bootstrap results for agent data')
    parser.add_argument('--output', type=str, default='results/inference_results.json',
                       help='Path for inference results')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if inference is enabled
    if not config.get('inference', {}).get('enabled', True):
        logger.info("Inference phase disabled in config. Skipping.")
        return
    
    # Load bootstrap results for agent data
    with open(args.bootstrap_results, 'r') as f:
        bootstrap_results = json.load(f)
    
    # Initialize engine
    engine = HybridInferenceEngine(config)
    
    # Load trained models
    engine.load_models(args.model_checkpoint)
    
    # Load agent data and graph from bootstrap
    # (Simplified - in production, load from saved state)
    
    # Create test data
    num_test_users = config['inference']['num_test_users']
    test_users = [f'user_{i}' for i in range(num_test_users)]
    
    test_interactions = {
        f'user_{i}': f'item_{np.random.randint(100)}' 
        for i in range(num_test_users)
    }
    
    all_items = [f'item_{i}' for i in range(200)]
    
    # Pre-populate graph with test data
    for user_id in test_users:
        engine.graph.add_user(user_id)
        engine.user_agents[user_id] = {
            'full_memory_text': f'User {user_id} preferences',
            'last_updated': datetime.now().isoformat()
        }
    
    for item_id in all_items:
        engine.graph.add_item(item_id)
        engine.item_agents[item_id] = {
            'description': f'Item {item_id} description'
        }
    
    # Add some interaction edges for the graph
    for user_id in test_users[:50]:  # Some users have interactions
        for _ in range(np.random.randint(5, 20)):
            item_id = np.random.choice(all_items)
            engine.graph.add_edge(user_id, item_id, 'interact', weight=np.random.uniform(0.5, 1.0))
    
    # Add similarity edges
    for i, u1 in enumerate(test_users):
        for u2 in test_users[i+1:i+5]:
            if u2 in test_users:
                engine.graph.add_edge(u1, u2, 'similar_pref', weight=np.random.uniform(0.7, 0.95))
    
    # Set random embeddings for testing
    for node_id in list(engine.graph.user_nodes) + list(engine.graph.item_nodes):
        emb = np.random.randn(256).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        engine.graph.set_node_embedding(node_id, 'intrinsic', emb)
        engine.graph.set_node_embedding(node_id, 'collaborative', emb.copy())
    
    # Run evaluation
    logger.info("Running evaluation...")
    results = engine.evaluate(
        test_users=test_users,
        test_interactions=test_interactions,
        all_items=all_items,
        k_values=config['evaluation']['k_values'],
        num_negatives=config['inference']['num_negatives']
    )
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    
    for metric_name, metric_data in results.items():
        if isinstance(metric_data, dict) and 'mean' in metric_data:
            logger.info(f"{metric_name}: {metric_data['mean']:.4f} ± {metric_data['std']:.4f}")
    
    logger.info("\nGating Statistics:")
    gate_stats = results.get('gate_stats', {})
    logger.info(f"  Total decisions: {gate_stats.get('total_decisions', 0)}")
    logger.info(f"  GNN path ratio: {gate_stats.get('gnn_path_ratio', 0):.2%}")
    logger.info(f"  LLM path ratio: {gate_stats.get('llm_path_ratio', 0):.2%}")
    
    logger.info("\nCost Statistics:")
    cost_stats = results.get('cost_stats', {})
    logger.info(f"  Total LLM cost: ${cost_stats.get('total_llm_cost', 0):.4f}")
    logger.info(f"  Avg latency/recommendation: {cost_stats.get('avg_latency_per_rec', 0):.4f}s")
    
    # Cold-start results
    cold_start_keys = [k for k in results.keys() if k.startswith('cold_start')]
    if cold_start_keys:
        logger.info("\nCold-Start Analysis:")
        for key in cold_start_keys:
            logger.info(f"  {key}: {results[key]:.4f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy values for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(output_path, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")
    logger.info("Phase 3 complete!")


if __name__ == '__main__':
    main()