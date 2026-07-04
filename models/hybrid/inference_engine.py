"""
Hybrid Inference Engine for H-GRAGrecsys

This module implements the hybrid inference engine that orchestrates GNN and LLM
paths for recommendation and ranking tasks. The engine provides:
- Single and batch inference for recommendations
- Ranking items for users with hybrid reasoning
- Explanation generation with hybrid path selection
- Quality assessment and confidence estimation
- Performance monitoring and optimization

The inference engine leverages the adaptive router to dynamically choose
between GNN and LLM paths based on context and uncertainty.
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import time
import math
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from sibling modules
from models.hybrid.adaptive_gate import AdaptiveGate, GatingFeatures
from models.hybrid.router import Router, RoutingDecision

# Import from GNN module
from models.gnn.gnn_encoder import GNNEncoder

# Import from LLM module
from models.llm.llm_interface import LLMInterface
from models.llm.prompt_templates import PromptTemplates

# Import from graph module
from models.graph.heterogeneous_graph import HeterogeneousGraph

# Import from evaluation
from evaluation.metrics import Metrics

# Import from utils
from utils.config_loader import ConfigLoader
from utils.logger import Logger
from utils.seed_manager import SeedManager
from utils.timer import Timer


@dataclass
class InferenceResult:
    """
    Dataclass for inference results.
    
    Attributes:
        predictions: Prediction tensor or list.
        routing_decision: Routing decision used.
        confidence: Prediction confidence.
        execution_time: Time taken for inference.
        explanations: Optional explanations.
        metadata: Additional metadata.
    """
    predictions: Any
    routing_decision: RoutingDecision
    confidence: float
    execution_time: float
    explanations: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HybridInferenceEngine(nn.Module):
    """
    Hybrid inference engine for H-GRAGrecsys.
    
    This class orchestrates the hybrid inference process, combining GNN and LLM
    paths through adaptive routing for recommendations, ranking, and explanations.
    """
    
    def __init__(
        self,
        router: Optional[Router] = None,
        gnn_encoder: Optional[GNNEncoder] = None,
        llm_interface: Optional[LLMInterface] = None,
        prompt_templates: Optional[PromptTemplates] = None,
        config: Optional[Union[str, Dict, ConfigLoader]] = None
    ):
        """
        Initialize the hybrid inference engine.
        
        Args:
            router: Optional Router instance. If None, creates from config.
            gnn_encoder: Optional GNNEncoder instance. If None, creates from config.
            llm_interface: Optional LLMInterface instance. If None, creates from config.
            prompt_templates: Optional PromptTemplates instance. If None, creates from config.
            config: Configuration object or path to config file.
                   Can be a string path, dict, or ConfigLoader instance.
        
        Raises:
            ValueError: If config is invalid or missing required fields.
        """
        super(HybridInferenceEngine, self).__init__()
        
        # Load configuration
        if config is None:
            self.config = {
                'model': {
                    'hybrid': {
                        'gate_threshold': 0.3,
                        'uniform_llm_rate': 0.15,
                        'fallback_strategy': 'gnn',
                        'combine_predictions': True,
                        'combine_weight': 0.5,
                        'max_llm_candidates': 20,
                        'batch_size': 32
                    }
                },
                'evaluation': {
                    'k_values': [1, 5, 10],
                    'num_negatives': 99
                }
            }
        elif isinstance(config, str):
            self.config_loader = ConfigLoader(config)
            self.config = self.config_loader.load_config()
        elif isinstance(config, dict):
            self.config = config
            self.config_loader = None
        elif isinstance(config, ConfigLoader):
            self.config_loader = config
            self.config = config.load_config()
        else:
            raise ValueError(f"Invalid config type: {type(config)}")
        
        # Setup logger
        self.logger = Logger(
            log_dir=self.config.get('logging', {}).get('log_dir', './logs'),
            name='inference_engine'
        )
        
        # Extract configuration
        hybrid_config = self.config.get('model', {}).get('hybrid', {})
        evaluation_config = self.config.get('evaluation', {})
        
        self.threshold = hybrid_config.get('gate_threshold', 0.3)
        self.uniform_llm_rate = hybrid_config.get('uniform_llm_rate', 0.15)
        self.fallback_strategy = hybrid_config.get('fallback_strategy', 'gnn')
        self.combine_predictions = hybrid_config.get('combine_predictions', True)
        self.combine_weight = hybrid_config.get('combine_weight', 0.5)
        self.max_llm_candidates = hybrid_config.get('max_llm_candidates', 20)
        self.batch_size = hybrid_config.get('batch_size', 32)
        self.k_values = evaluation_config.get('k_values', [1, 5, 10])
        self.num_negatives = evaluation_config.get('num_negatives', 99)
        
        # Initialize components
        self.router = router if router is not None else self._create_router()
        self.gnn_encoder = gnn_encoder if gnn_encoder is not None else self._create_gnn_encoder()
        self.llm_interface = llm_interface if llm_interface is not None else self._create_llm_interface()
        self.prompt_templates = prompt_templates if prompt_templates is not None else self._create_prompt_templates()
        
        # Metrics calculator
        self.metrics = Metrics(self.config)
        
        # Inference statistics
        self.inference_stats = {
            'total_inferences': 0,
            'gnn_inferences': 0,
            'llm_inferences': 0,
            'combined_inferences': 0,
            'avg_inference_time': 0.0,
            'total_llm_calls': 0,
            'total_gnn_calls': 0
        }
        
        # Cache for embeddings
        self.embedding_cache = {}
        self.cache_enabled = True
        
        self.logger.log_info(
            f"HybridInferenceEngine initialized: threshold={self.threshold}, "
            f"max_llm_candidates={self.max_llm_candidates}, "
            f"batch_size={self.batch_size}"
        )
    
    def _create_router(self) -> Router:
        """Create router from configuration."""
        return Router(config=self.config)
    
    def _create_gnn_encoder(self) -> GNNEncoder:
        """Create GNN encoder from configuration."""
        return GNNEncoder(config=self.config)
    
    def _create_llm_interface(self) -> LLMInterface:
        """Create LLM interface from configuration."""
        return LLMInterface(config=self.config)
    
    def _create_prompt_templates(self) -> PromptTemplates:
        """Create prompt templates from configuration."""
        return PromptTemplates(self.config)
    
    def infer(
        self,
        user: Any,
        candidates: List[Any],
        graph: Optional[Union[HeteroGraph, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        force_path: Optional[str] = None,
        return_explanations: bool = False
    ) -> InferenceResult:
        """
        Perform inference for a single user and candidate items.
        
        Args:
            user: User object or ID.
            candidates: List of candidate items.
            graph: Optional graph object.
            context: Optional context dictionary.
            force_path: Optional force routing to specific path.
            return_explanations: Whether to return explanations.
        
        Returns:
            InferenceResult containing predictions and metadata.
        
        Raises:
            ValueError: If candidates list is empty.
        """
        if not candidates:
            raise ValueError("Candidates list cannot be empty")
        
        start_time = time.time()
        
        # Prepare context
        if context is None:
            context = {}
        context['num_candidates'] = len(candidates)
        context['user_id'] = user if isinstance(user, (int, str)) else getattr(user, 'id', None)
        
        # Get GNN predictions
        gnn_pred = None
        if self.gnn_encoder is not None:
            with Timer() as timer:
                gnn_pred = self._get_gnn_predictions(user, candidates, graph)
            self.inference_stats['total_gnn_calls'] += 1
        
        # Get LLM predictions if needed
        llm_pred = None
        llm_candidates = candidates
        if len(candidates) > self.max_llm_candidates:
            # Sample candidates for LLM
            indices = np.random.choice(
                len(candidates),
                min(self.max_llm_candidates, len(candidates)),
                replace=False
            )
            llm_candidates = [candidates[i] for i in indices]
        
        if self.llm_interface is not None:
            with Timer() as timer:
                llm_pred = self._get_llm_predictions(user, llm_candidates, context)
            self.inference_stats['total_llm_calls'] += 1
        
        # Route and get final prediction
        final_pred, decision, metadata = self.router.forward(
            node=user,
            context=context,
            gnn_pred=gnn_pred,
            llm_pred=llm_pred,
            graph=graph,
            force_path=force_path
        )
        
        # Compute confidence
        confidence = self._compute_confidence(final_pred, decision)
        
        # Get explanations if requested
        explanations = None
        if return_explanations and decision.path == 'llm':
            explanations = self._generate_explanations(user, candidates, final_pred, context)
        
        # Update statistics
        self.inference_stats['total_inferences'] += 1
        if decision.path == 'gnn':
            self.inference_stats['gnn_inferences'] += 1
        elif decision.path == 'llm':
            self.inference_stats['llm_inferences'] += 1
        else:
            self.inference_stats['combined_inferences'] += 1
        
        execution_time = time.time() - start_time
        alpha = 0.1
        self.inference_stats['avg_inference_time'] = (
            (1 - alpha) * self.inference_stats['avg_inference_time'] +
            alpha * execution_time
        )
        
        return InferenceResult(
            predictions=final_pred,
            routing_decision=decision,
            confidence=confidence,
            execution_time=execution_time,
            explanations=explanations,
            metadata=metadata
        )
    
    def _get_gnn_predictions(
        self,
        user: Any,
        candidates: List[Any],
        graph: Optional[Union[HeteroGraph, Any]]
    ) -> torch.Tensor:
        """
        Get GNN predictions for user and candidates.
        
        Args:
            user: User object or ID.
            candidates: List of candidate items.
            graph: Optional graph object.
        
        Returns:
            Tensor of GNN predictions.
        """
        if self.gnn_encoder is None:
            return None
        
        # Get user embedding
        user_id = user if isinstance(user, (int, str)) else getattr(user, 'id', None)
        user_type = 'user'
        
        # Get candidate embeddings
        candidate_ids = [
            c if isinstance(c, (int, str)) else getattr(c, 'id', None)
            for c in candidates
        ]
        
        # Encode graph if available
        if graph is not None:
            # Encode all nodes
            embeddings = self.gnn_encoder.encode_graph(graph)
            
            # Get user and item embeddings
            if user_type in embeddings:
                user_emb = embeddings[user_type][user_id]
            else:
                # Use cache or fallback
                user_emb = self._get_embedding_fallback(user_id, user_type)
            
            # Get candidate embeddings
            item_type = 'item'
            if item_type in embeddings:
                candidate_embs = embeddings[item_type][candidate_ids]
            else:
                # Use fallback
                candidate_embs = torch.stack([
                    self._get_embedding_fallback(cid, item_type)
                    for cid in candidate_ids
                ])
            
            # Compute scores (dot product)
            scores = torch.matmul(user_emb.unsqueeze(0), candidate_embs.T).squeeze(0)
            return scores
        
        # Fallback: random predictions
        return torch.randn(len(candidates))
    
    def _get_embedding_fallback(self, node_id: Any, node_type: str) -> torch.Tensor:
        """
        Get embedding from cache or create random embedding.
        
        Args:
            node_id: Node ID.
            node_type: Node type.
        
        Returns:
            Embedding tensor.
        """
        cache_key = f"{node_type}_{node_id}"
        if self.cache_enabled and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Create random embedding
        embedding_dim = self.config.get('model', {}).get('gnn', {}).get('output_dim', 128)
        embedding = torch.randn(embedding_dim)
        
        if self.cache_enabled:
            self.embedding_cache[cache_key] = embedding
        
        return embedding
    
    def _get_llm_predictions(
        self,
        user: Any,
        candidates: List[Any],
        context: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Get LLM predictions for user and candidates.
        
        Args:
            user: User object or ID.
            candidates: List of candidate items.
            context: Context dictionary.
        
        Returns:
            Tensor of LLM predictions.
        """
        if self.llm_interface is None or not candidates:
            return None
        
        try:
            # Prepare prompt
            user_info = str(user) if not isinstance(user, (int, str)) else user
            candidate_info = [str(c) if not isinstance(c, (int, str)) else c for c in candidates]
            
            prompt = self.prompt_templates.get_ranking_prompt(
                user=user_info,
                candidates=candidate_info,
                context=context
            )
            
            # Generate predictions
            llm_output = self.llm_interface.generate(prompt)
            
            # Parse predictions
            predictions = self._parse_llm_predictions(llm_output, len(candidates))
            return predictions
            
        except Exception as e:
            self.logger.log_error(f"LLM prediction failed: {e}")
            return None
    
    def _parse_llm_predictions(
        self,
        llm_output: str,
        num_candidates: int
    ) -> torch.Tensor:
        """
        Parse LLM output to extract predictions.
        
        Args:
            llm_output: LLM generated text.
            num_candidates: Number of candidates.
        
        Returns:
            Tensor of predictions.
        """
        # Simple parsing: look for scores or rankings
        scores = torch.zeros(num_candidates)
        
        # Try to extract numerical scores
        try:
            # Look for pattern: "score: X" or "rating: X"
            import re
            score_pattern = r'(?:score|rating|prediction)s?:?\s*([\d.]+)'
            matches = re.findall(score_pattern, llm_output, re.IGNORECASE)
            
            if matches:
                # Use first N matches
                num_matches = min(len(matches), num_candidates)
                scores[:num_matches] = torch.tensor([float(m) for m in matches[:num_matches]])
                return scores
        except Exception:
            pass
        
        # Fallback: try to extract ranking order
        try:
            # Look for ranked list
            import re
            rank_pattern = r'(\d+)\.?\s*(?:item|option)?'
            matches = re.findall(rank_pattern, llm_output)
            
            if matches:
                # Assign scores based on rank
                for i, match in enumerate(matches[:num_candidates]):
                    try:
                        rank = int(match)
                        if 1 <= rank <= num_candidates:
                            scores[rank - 1] = 1.0 - (i / num_candidates)
                    except ValueError:
                        continue
                return scores
        except Exception:
            pass
        
        # Random scores as fallback
        return torch.randn(num_candidates)
    
    def _compute_confidence(
        self,
        predictions: torch.Tensor,
        decision: RoutingDecision
    ) -> float:
        """
        Compute confidence for predictions.
        
        Args:
            predictions: Prediction tensor.
            decision: Routing decision.
        
        Returns:
            Confidence score.
        """
        if predictions is None:
            return 0.0
        
        try:
            if isinstance(predictions, torch.Tensor):
                # Use softmax confidence
                probs = F.softmax(predictions, dim=-1)
                confidence = probs.max().item()
                return float(confidence)
            else:
                return float(decision.confidence)
        except Exception:
            return 0.5
    
    def _generate_explanations(
        self,
        user: Any,
        candidates: List[Any],
        predictions: torch.Tensor,
        context: Dict[str, Any]
    ) -> List[str]:
        """
        Generate explanations for predictions.
        
        Args:
            user: User object or ID.
            candidates: List of candidate items.
            predictions: Prediction tensor.
            context: Context dictionary.
        
        Returns:
            List of explanation strings.
        """
        if self.llm_interface is None:
            return None
        
        explanations = []
        
        # Get top candidates
        if isinstance(predictions, torch.Tensor):
            top_indices = torch.topk(predictions, min(5, len(candidates))).indices
        else:
            top_indices = range(min(5, len(candidates)))
        
        for idx in top_indices:
            if idx >= len(candidates):
                continue
            
            item = candidates[idx]
            try:
                prompt = self.prompt_templates.get_explanation_prompt(
                    user=str(user),
                    item=str(item),
                    recommendation=predictions[idx].item() if isinstance(predictions, torch.Tensor) else None
                )
                
                explanation = self.llm_interface.generate(prompt)
                explanations.append(explanation)
            except Exception as e:
                self.logger.log_warning(f"Explanation generation failed: {e}")
                explanations.append(f"Recommendation based on user preferences and item similarity")
        
        return explanations
    
    def rank_items(
        self,
        user: Any,
        items: List[Any],
        graph: Optional[Union[HeteroGraph, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> List[Tuple[Any, float]]:
        """
        Rank items for a user.
        
        Args:
            user: User object or ID.
            items: List of items to rank.
            graph: Optional graph object.
            context: Optional context dictionary.
            top_k: Number of top items to return.
        
        Returns:
            List of (item, score) tuples sorted by score.
        """
        # Get inference results
        result = self.infer(
            user=user,
            candidates=items,
            graph=graph,
            context=context
        )
        
        # Extract predictions
        predictions = result.predictions
        if predictions is None:
            return [(items[i], 0.0) for i in range(min(top_k, len(items)))]
        
        # Sort items by prediction score
        if isinstance(predictions, torch.Tensor):
            if len(predictions) != len(items):
                # Predictions may be for subset of items
                # Use scores for subset, random for others
                scores = torch.zeros(len(items))
                scores[:len(predictions)] = predictions
                predictions = scores
            
            # Get top k indices
            top_indices = torch.topk(predictions, min(top_k, len(items))).indices
            ranked_items = [(items[idx.item()], predictions[idx].item()) for idx in top_indices]
        else:
            # Fallback: random ranking
            ranked_items = [(items[i], 0.0) for i in range(min(top_k, len(items)))]
            np.random.shuffle(ranked_items)
        
        return ranked_items
    
    def process_batch(
        self,
        users: List[Any],
        items: List[Any],
        graph: Optional[Union[HeteroGraph, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[InferenceResult]:
        """
        Process batch inference for multiple users.
        
        Args:
            users: List of users.
            items: List of items for each user.
            graph: Optional graph object.
            context: Optional context dictionary.
        
        Returns:
            List of InferenceResult objects.
        """
        results = []
        
        for user, user_items in zip(users, items):
            result = self.infer(
                user=user,
                candidates=user_items,
                graph=graph,
                context=context
            )
            results.append(result)
        
        return results
    
    def get_inference_metrics(self) -> Dict[str, Any]:
        """
        Get inference metrics and statistics.
        
        Returns:
            Dict containing inference metrics.
        """
        total = self.inference_stats['total_inferences']
        if total == 0:
            return {
                'total_inferences': 0,
                'gnn_ratio': 0.0,
                'llm_ratio': 0.0,
                'combined_ratio': 0.0,
                'avg_inference_time': 0.0,
                'total_llm_calls': 0,
                'total_gnn_calls': 0,
                'llm_call_ratio': 0.0
            }
        
        router_stats = self.router.get_routing_statistics() if self.router else {}
        
        return {
            'total_inferences': total,
            'gnn_ratio': self.inference_stats['gnn_inferences'] / total,
            'llm_ratio': self.inference_stats['llm_inferences'] / total,
            'combined_ratio': self.inference_stats['combined_inferences'] / total,
            'avg_inference_time': self.inference_stats['avg_inference_time'],
            'total_llm_calls': self.inference_stats['total_llm_calls'],
            'total_gnn_calls': self.inference_stats['total_gnn_calls'],
            'llm_call_ratio': (
                self.inference_stats['total_llm_calls'] / 
                max(1, self.inference_stats['total_gnn_calls'] + self.inference_stats['total_llm_calls'])
            ),
            'router_stats': router_stats
        }
    
    def compute_quality_assessment(
        self,
        user: Any,
        item: Any,
        prediction: float,
        ground_truth: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute quality assessment for a prediction.
        
        Args:
            user: User object or ID.
            item: Item object or ID.
            prediction: Prediction score.
            ground_truth: Optional ground truth score.
        
        Returns:
            Dict containing quality metrics.
        """
        quality = {
            'prediction': prediction,
            'confidence': 0.5,
            'uncertainty': 0.5
        }
        
        if ground_truth is not None:
            # Compute error metrics
            error = abs(prediction - ground_truth)
            quality['error'] = error
            quality['relative_error'] = error / max(1e-6, abs(ground_truth))
        
        # Confidence based on prediction magnitude
        quality['confidence'] = min(1.0, abs(prediction) / 5.0)
        quality['uncertainty'] = 1.0 - quality['confidence']
        
        return quality
    
    def warm_start(
        self,
        new_user: Any,
        interactions: List[Tuple[Any, float]],
        graph: Optional[Union[HeteroGraph, Any]] = None
    ) -> Dict[str, Any]:
        """
        Warm start for a new user with interactions.
        
        Args:
            new_user: New user object.
            interactions: List of (item, rating) tuples.
            graph: Optional graph object.
        
        Returns:
            Dict containing warm start results.
        """
        self.logger.log_info(f"Warm start for user: {new_user}")
        
        # Initialize user embedding from interactions
        if graph is not None and self.gnn_encoder is not None:
            # Use GNN to initialize
            user_id = new_user if isinstance(new_user, (int, str)) else getattr(new_user, 'id', None)
            
            # Get item embeddings for interacted items
            item_ids = [item for item, _ in interactions]
            item_embeddings = []
            item_weights = []
            
            for item, rating in interactions:
                item_id = item if isinstance(item, (int, str)) else getattr(item, 'id', None)
                try:
                    # Get embedding from graph
                    embedding = self._get_embedding_fallback(item_id, 'item')
                    item_embeddings.append(embedding)
                    item_weights.append(rating)
                except Exception:
                    continue
            
            if item_embeddings:
                # Compute weighted average of item embeddings
                item_embeddings = torch.stack(item_embeddings)
                item_weights = torch.tensor(item_weights, dtype=torch.float32)
                item_weights = F.softmax(item_weights, dim=0)
                
                user_embedding = torch.matmul(item_weights, item_embeddings)
                
                # Cache user embedding
                cache_key = f"user_{user_id}"
                self.embedding_cache[cache_key] = user_embedding
                
                self.logger.log_info(f"Warm start completed for user: {new_user}")
                return {
                    'user_id': user_id,
                    'embedding': user_embedding,
                    'num_interactions': len(interactions),
                    'status': 'success'
                }
        
        self.logger.log_warning(f"Warm start failed for user: {new_user}")
        return {
            'user_id': new_user if isinstance(new_user, (int, str)) else getattr(new_user, 'id', None),
            'status': 'failed',
            'num_interactions': len(interactions)
        }
    
    def enable_cache(self):
        """Enable embedding caching."""
        self.cache_enabled = True
        self.logger.log_info("Cache enabled")
    
    def disable_cache(self):
        """Disable embedding caching."""
        self.cache_enabled = False
        self.logger.log_info("Cache disabled")
    
    def clear_cache(self):
        """Clear embedding cache."""
        self.embedding_cache = {}
        self.logger.log_info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache statistics.
        """
        return {
            'cache_size': len(self.embedding_cache),
            'cache_enabled': self.cache_enabled
        }
    
    def reset_statistics(self):
        """Reset all inference statistics."""
        self.inference_stats = {
            'total_inferences': 0,
            'gnn_inferences': 0,
            'llm_inferences': 0,
            'combined_inferences': 0,
            'avg_inference_time': 0.0,
            'total_llm_calls': 0,
            'total_gnn_calls': 0
        }
        if self.router:
            self.router.reset_statistics()
        self.logger.log_info("Statistics reset")
    
    def save_engine(self, save_path: str):
        """
        Save inference engine state.
        
        Args:
            save_path: Path to save the engine.
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            state_dict = {
                'router': self.router.state_dict() if self.router else None,
                'gnn_encoder': self.gnn_encoder.state_dict() if self.gnn_encoder else None,
                'config': self.config,
                'threshold': self.threshold,
                'inference_stats': self.inference_stats,
                'version': __version__
            }
            
            torch.save(state_dict, save_path)
            self.logger.log_info(f"Engine saved to {save_path}")
        
        except Exception as e:
            self.logger.log_error(f"Failed to save engine: {e}")
            raise
    
    def load_engine(self, load_path: str):
        """
        Load inference engine state.
        
        Args:
            load_path: Path to load the engine from.
        
        Raises:
            FileNotFoundError: If checkpoint not found.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Engine checkpoint not found: {load_path}")
        
        try:
            checkpoint = torch.load(load_path, map_location='cpu')
            
            if checkpoint.get('router') is not None and self.router:
                self.router.load_state_dict(checkpoint['router'])
            
            if checkpoint.get('gnn_encoder') is not None and self.gnn_encoder:
                self.gnn_encoder.load_state_dict(checkpoint['gnn_encoder'])
            
            self.threshold = checkpoint.get('threshold', self.threshold)
            self.inference_stats = checkpoint.get('inference_stats', self.inference_stats)
            
            if 'config' in checkpoint:
                self.config = checkpoint['config']
            
            self.logger.log_info(f"Engine loaded from {load_path}")
        
        except Exception as e:
            self.logger.log_error(f"Failed to load engine: {e}")
            raise
    
    def get_parameters(self) -> Dict[str, int]:
        """
        Get parameter statistics for all components.
        
        Returns:
            Dict with parameter counts.
        """
        params = {'total': 0}
        
        if self.router:
            router_params = self.router.get_parameters()
            params['router'] = router_params['total']
            params['total'] += router_params['total']
        
        if self.gnn_encoder:
            gnn_params = self.gnn_encoder.get_parameters()
            params['gnn_encoder'] = gnn_params['total']
            params['total'] += gnn_params['total']
        
        if self.llm_interface:
            llm_params = self.llm_interface.get_parameters()
            params['llm_interface'] = llm_params['total']
            params['total'] += llm_params['total']
        
        return params
    
    def to_device(self, device: torch.device) -> 'HybridInferenceEngine':
        """
        Move all components to specified device.
        
        Args:
            device: PyTorch device.
        
        Returns:
            Self with components moved to device.
        """
        if self.router:
            self.router.to_device(device)
        if self.gnn_encoder:
            self.gnn_encoder.to_device(device)
        if self.llm_interface:
            self.llm_interface.to_device(device)
        
        self.to(device)
        self.logger.log_info(f"Engine moved to device: {device}")
        
        return self
    
    def forward(
        self,
        user: Any,
        candidates: List[Any],
        graph: Optional[Union[HeteroGraph, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        force_path: Optional[str] = None
    ) -> InferenceResult:
        """
        Forward pass for inference (alias for infer).
        
        Args:
            user: User object or ID.
            candidates: List of candidate items.
            graph: Optional graph object.
            context: Optional context dictionary.
            force_path: Optional force routing to specific path.
        
        Returns:
            InferenceResult containing predictions and metadata.
        """
        return self.infer(
            user=user,
            candidates=candidates,
            graph=graph,
            context=context,
            force_path=force_path
        )


# Module level variables and exports
__all__ = [
    'InferenceResult',
    'HybridInferenceEngine',
    '__doc__'
]

# Version information
__version__ = '1.0.0'
__author__ = 'H-GRAGrecsys Team'


def create_inference_engine(
    config_path: Optional[str] = None,
    router: Optional[Router] = None,
    gnn_encoder: Optional[GNNEncoder] = None,
    llm_interface: Optional[LLMInterface] = None,
    device: Optional[torch.device] = None
) -> HybridInferenceEngine:
    """
    Factory function to create a HybridInferenceEngine instance.
    
    Args:
        config_path: Optional path to configuration file.
        router: Optional Router instance.
        gnn_encoder: Optional GNNEncoder instance.
        llm_interface: Optional LLMInterface instance.
        device: Optional device to move engine to. Defaults to CUDA if available.
    
    Returns:
        Initialized HybridInferenceEngine instance.
    
    Example:
        >>> engine = create_inference_engine(
        ...     config_path='config/default_config.yaml',
        ...     router=router,
        ...     gnn_encoder=encoder,
        ...     llm_interface=llm
        ... )
        >>> engine.to_device(torch.device('cuda'))
    """
    engine = HybridInferenceEngine(
        router=router,
        gnn_encoder=gnn_encoder,
        llm_interface=llm_interface,
        config=config_path
    )
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return engine.to_device(device)


def create_inference_result(
    predictions: Any,
    routing_decision: RoutingDecision,
    confidence: float,
    execution_time: float,
    explanations: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> InferenceResult:
    """
    Factory function to create an InferenceResult object.
    
    Args:
        predictions: Prediction tensor or list.
        routing_decision: Routing decision.
        confidence: Prediction confidence.
        execution_time: Execution time.
        explanations: Optional explanations.
        metadata: Optional metadata.
    
    Returns:
        InferenceResult object.
    
    Example:
        >>> result = create_inference_result(
        ...     predictions=scores,
        ...     routing_decision=decision,
        ...     confidence=0.9,
        ...     execution_time=0.1
        ... )
    """
    return InferenceResult(
        predictions=predictions,
        routing_decision=routing_decision,
        confidence=confidence,
        execution_time=execution_time,
        explanations=explanations,
        metadata=metadata or {}
    )