"""
Adaptive Gate Module for H-GRAGrecsys

This module implements the adaptive gating mechanism for hybrid inference
in H-GRAGrecsys. The gate dynamically routes requests between GNN and LLM
paths based on:
- Prediction confidence
- Graph density
- Context criticality
- Node staleness
- Computational cost

The adaptive gate enables efficient trade-offs between accuracy and efficiency
by selectively invoking the LLM only when necessary.
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from utils
from utils.config_loader import ConfigLoader
from utils.logger import Logger
from utils.seed_manager import SeedManager
from utils.timer import Timer

# Import from graph module
from models.graph.heterogeneous_graph import HeterogeneousGraph
from models.graph.relation_types import RelationType


@dataclass
class GatingFeatures:
    """
    Features used for gating decision computation.
    
    Attributes:
        confidence: Prediction confidence score (0-1).
        graph_density: Density of local graph neighborhood (0-1).
        context_criticality: Criticality of the context (0-1).
        node_staleness: Staleness of node information (0-1).
        interaction_history: Recent interaction history features.
        uncertainty: Prediction uncertainty score.
        complexity: Task complexity score.
    """
    confidence: float = 0.5
    graph_density: float = 0.5
    context_criticality: float = 0.5
    node_staleness: float = 0.0
    interaction_history: Optional[torch.Tensor] = None
    uncertainty: float = 0.0
    complexity: float = 0.5
    
    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Convert gating features to tensor.
        
        Args:
            device: Device to place tensor on.
        
        Returns:
            Tensor of gating features.
        """
        features = [
            self.confidence,
            self.graph_density,
            self.context_criticality,
            self.node_staleness,
            self.uncertainty,
            self.complexity
        ]
        return torch.tensor(features, device=device, dtype=torch.float32)


class GatingFeaturesExtractor(nn.Module):
    """
    Extracts gating features from graph, node, and prediction context.
    
    This class computes the features needed for the adaptive gate to make
    routing decisions.
    """
    
    def __init__(self, config: Optional[Union[str, Dict, ConfigLoader]] = None):
        """
        Initialize the gating features extractor.
        
        Args:
            config: Configuration object or path to config file.
                   Can be a string path, dict, or ConfigLoader instance.
        """
        super(GatingFeaturesExtractor, self).__init__()
        
        # Load configuration
        if config is None:
            self.config = {}
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
        
        # Extract configuration
        hybrid_config = self.config.get('model', {}).get('hybrid', {})
        self.staleness_lambda = hybrid_config.get('staleness_lambda', 0.1)
        self.uniform_llm_rate = hybrid_config.get('uniform_llm_rate', 0.15)
        
        # Setup logger
        self.logger = Logger(
            log_dir=self.config.get('logging', {}).get('log_dir', './logs'),
            name='gating_features_extractor'
        )
        
        # Feature projection (if needed)
        self.feature_proj = nn.Linear(6, 6)  # Simple projection for feature transformation
        
        self.logger.log_info("GatingFeaturesExtractor initialized")
    
    def get_confidence(
        self,
        prediction_logits: torch.Tensor,
        use_softmax: bool = True
    ) -> float:
        """
        Compute prediction confidence from logits.
        
        Args:
            prediction_logits: Model prediction logits.
            use_softmax: Whether to apply softmax first.
        
        Returns:
            Confidence score between 0 and 1.
        """
        if use_softmax:
            probs = F.softmax(prediction_logits, dim=-1)
            confidence = probs.max().item()
        else:
            # Simple confidence from logit magnitude
            confidence = torch.sigmoid(prediction_logits.max()).item()
        
        return float(confidence)
    
    def get_graph_density(
        self,
        graph: Union[HeteroGraph, Any],
        node_id: Optional[int] = None,
        node_type: Optional[str] = None,
        radius: int = 2
    ) -> float:
        """
        Compute graph density in the neighborhood of a node.
        
        Args:
            graph: Graph object.
            node_id: Optional node ID to compute local density.
            node_type: Node type if node_id is provided.
            radius: Neighborhood radius for local density.
        
        Returns:
            Graph density score between 0 and 1.
        """
        if graph is None:
            return 0.5  # Default density
        
        try:
            if node_id is not None and node_type is not None:
                # Compute local density
                if hasattr(graph, 'get_neighbors'):
                    neighbors = graph.get_neighbors(node_id, radius=radius)
                    num_neighbors = len(neighbors) if neighbors else 0
                    
                    # Normalize by expected density (adjust based on graph size)
                    max_neighbors = 100  # Configurable
                    density = min(num_neighbors / max_neighbors, 1.0)
                    return density
            else:
                # Compute global graph density
                if hasattr(graph, 'get_graph_statistics'):
                    stats = graph.get_graph_statistics()
                    if 'density' in stats:
                        return min(stats['density'], 1.0)
        except Exception as e:
            self.logger.log_warning(f"Failed to compute graph density: {e}")
        
        return 0.5
    
    def get_context_criticality(
        self,
        context: Dict[str, Any],
        prediction: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute context criticality score.
        
        Args:
            context: Context dictionary containing query information.
            prediction: Optional prediction for context-aware criticality.
        
        Returns:
            Criticality score between 0 and 1.
        """
        criticality = 0.0
        
        # Check for cold start
        if context.get('is_cold_start', False):
            criticality += 0.3
        
        # Check for rare items
        if context.get('item_popularity', 0) < 0.1:
            criticality += 0.2
        
        # Check for diverse preferences
        if context.get('preference_diversity', 0) > 0.7:
            criticality += 0.2
        
        # Check for ambiguous context
        if context.get('is_ambiguous', False):
            criticality += 0.3
        
        # Check for high-value interaction
        if context.get('is_high_value', False):
            criticality += 0.2
        
        return min(criticality, 1.0)
    
    def get_node_staleness(
        self,
        node: Any,
        current_time: Optional[float] = None
    ) -> float:
        """
        Compute node staleness based on last update time.
        
        Args:
            node: Node object with timestamp information.
            current_time: Current time for staleness computation.
        
        Returns:
            Staleness score between 0 and 1.
        """
        if node is None:
            return 0.0
        
        try:
            # Try to get last update time
            if hasattr(node, 'last_update'):
                last_update = node.last_update
                if current_time is None:
                    current_time = time.time()
                
                # Compute staleness
                time_diff = current_time - last_update
                staleness = 1.0 - math.exp(-self.staleness_lambda * time_diff)
                return min(staleness, 1.0)
        except Exception:
            pass
        
        # Default staleness based on number of interactions
        if hasattr(node, 'num_interactions'):
            interactions = node.num_interactions
            if interactions == 0:
                return 1.0
            elif interactions < 5:
                return 0.8
            elif interactions < 20:
                return 0.5
            else:
                return 0.2
        
        return 0.0
    
    def get_uncertainty(
        self,
        prediction: torch.Tensor,
        use_entropy: bool = True
    ) -> float:
        """
        Compute prediction uncertainty.
        
        Args:
            prediction: Model prediction tensor.
            use_entropy: Whether to use entropy for uncertainty.
        
        Returns:
            Uncertainty score between 0 and 1.
        """
        if prediction is None:
            return 0.0
        
        try:
            if use_entropy:
                # Compute entropy of softmax probabilities
                probs = F.softmax(prediction, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
                max_entropy = math.log(prediction.size(-1))
                uncertainty = (entropy / max_entropy).mean().item()
                return float(uncertainty)
            else:
                # Use variance
                uncertainty = prediction.var().item()
                return min(uncertainty, 1.0)
        except Exception:
            return 0.0
    
    def get_complexity(
        self,
        context: Dict[str, Any],
        task_type: Optional[str] = None
    ) -> float:
        """
        Compute task complexity score.
        
        Args:
            context: Context dictionary.
            task_type: Type of task ('recommendation', 'explanation', etc.).
        
        Returns:
            Complexity score between 0 and 1.
        """
        complexity = 0.0
        
        # Task type complexity
        if task_type == 'explanation':
            complexity += 0.4
        elif task_type == 'ranking':
            complexity += 0.3
        elif task_type == 'recommendation':
            complexity += 0.2
        
        # Context complexity
        if context.get('num_candidates', 0) > 50:
            complexity += 0.2
        if context.get('diverse_items', False):
            complexity += 0.2
        
        return min(complexity, 1.0)
    
    def extract_features(
        self,
        node: Any,
        context: Dict[str, Any],
        prediction: Optional[torch.Tensor] = None,
        graph: Optional[Union[HeteroGraph, Any]] = None,
        current_time: Optional[float] = None
    ) -> GatingFeatures:
        """
        Extract all gating features.
        
        Args:
            node: Target node (user or item).
            context: Context dictionary.
            prediction: Optional model prediction.
            graph: Optional graph for density computation.
            current_time: Optional current time for staleness.
        
        Returns:
            GatingFeatures object with all extracted features.
        """
        # Extract each feature
        confidence = self.get_confidence(prediction) if prediction is not None else 0.5
        graph_density = self.get_graph_density(
            graph,
            node_id=context.get('node_id'),
            node_type=context.get('node_type')
        ) if graph is not None else 0.5
        context_criticality = self.get_context_criticality(context, prediction)
        node_staleness = self.get_node_staleness(node, current_time)
        uncertainty = self.get_uncertainty(prediction) if prediction is not None else 0.0
        complexity = self.get_complexity(context)
        
        return GatingFeatures(
            confidence=confidence,
            graph_density=graph_density,
            context_criticality=context_criticality,
            node_staleness=node_staleness,
            uncertainty=uncertainty,
            complexity=complexity
        )
    
    def forward(
        self,
        node: Any,
        context: Dict[str, Any],
        prediction: Optional[torch.Tensor] = None,
        graph: Optional[Union[HeteroGraph, Any]] = None,
        current_time: Optional[float] = None
    ) -> GatingFeatures:
        """
        Forward pass to extract gating features.
        
        Args:
            node: Target node.
            context: Context dictionary.
            prediction: Optional prediction.
            graph: Optional graph.
            current_time: Optional current time.
        
        Returns:
            GatingFeatures object.
        """
        return self.extract_features(node, context, prediction, graph, current_time)


class AdaptiveGate(nn.Module):
    """
    Adaptive gate for routing decisions between GNN and LLM paths.
    
    This class implements the core gating mechanism that computes gating scores
    and makes routing decisions based on input features.
    """
    
    def __init__(self, config: Optional[Union[str, Dict, ConfigLoader]] = None):
        """
        Initialize adaptive gate.
        
        Args:
            config: Configuration object or path to config file.
                   Can be a string path, dict, or ConfigLoader instance.
        
        Raises:
            ValueError: If config is invalid.
        """
        super(AdaptiveGate, self).__init__()
        
        # Load configuration
        if config is None:
            self.config = {
                'model': {
                    'hybrid': {
                        'gate_threshold': 0.3,
                        'staleness_lambda': 0.1,
                        'uniform_llm_rate': 0.15,
                        'gate_hidden_dim': 64,
                        'gate_num_layers': 2,
                        'gate_dropout': 0.1
                    }
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
            name='adaptive_gate'
        )
        
        # Extract configuration
        hybrid_config = self.config.get('model', {}).get('hybrid', {})
        self.threshold = hybrid_config.get('gate_threshold', 0.3)
        self.staleness_lambda = hybrid_config.get('staleness_lambda', 0.1)
        self.uniform_llm_rate = hybrid_config.get('uniform_llm_rate', 0.15)
        self.hidden_dim = hybrid_config.get('gate_hidden_dim', 64)
        self.num_layers = hybrid_config.get('gate_num_layers', 2)
        self.dropout = hybrid_config.get('gate_dropout', 0.1)
        
        # Build gating network
        self._build_gate_network()
        
        # Feature extractor
        self.feature_extractor = GatingFeaturesExtractor(config)
        
        # Tracking
        self.routing_history = []
        self.gate_stats = {
            'total_decisions': 0,
            'gnn_decisions': 0,
            'llm_decisions': 0,
            'avg_gate_score': 0.0
        }
        
        self.logger.log_info(
            f"AdaptiveGate initialized: threshold={self.threshold}, "
            f"uniform_llm_rate={self.uniform_llm_rate}"
        )
    
    def _build_gate_network(self):
        """
        Build the gating network architecture.
        """
        layers = []
        input_dim = 6  # Number of GatingFeatures
        
        # Input layer
        layers.append(nn.Linear(input_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout))
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
        
        # Output layer (single score)
        layers.append(nn.Linear(self.hidden_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.gate_network = nn.Sequential(*layers)
    
    def compute_gating_score(
        self,
        features: Union[GatingFeatures, torch.Tensor]
    ) -> float:
        """
        Compute gating score from features.
        
        Args:
            features: GatingFeatures object or feature tensor.
        
        Returns:
            Gating score between 0 and 1.
        """
        # Convert features to tensor if needed
        if isinstance(features, GatingFeatures):
            feature_tensor = features.to_tensor()
        elif isinstance(features, torch.Tensor):
            feature_tensor = features
        else:
            raise ValueError(f"Unsupported feature type: {type(features)}")
        
        # Add batch dimension if needed
        if len(feature_tensor.shape) == 1:
            feature_tensor = feature_tensor.unsqueeze(0)
        
        # Compute gate score
        with torch.no_grad():
            gate_score = self.gate_network(feature_tensor)
            gate_score = gate_score.squeeze().item()
        
        # Add small random noise for exploration
        if self.uniform_llm_rate > 0:
            if np.random.random() < self.uniform_llm_rate:
                gate_score = 0.0  # Force LLM path
        
        return float(gate_score)
    
    def decide_path(
        self,
        gate_score: float,
        threshold: Optional[float] = None
    ) -> str:
        """
        Make routing decision based on gate score.
        
        Args:
            gate_score: Gating score between 0 and 1.
            threshold: Optional custom threshold. If None, uses default.
        
        Returns:
            Routing decision: 'gnn' or 'llm'.
        """
        if threshold is None:
            threshold = self.threshold
        
        # Lower score means LLM path (more uncertain)
        if gate_score > threshold:
            decision = 'gnn'
        else:
            decision = 'llm'
        
        # Update statistics
        self.gate_stats['total_decisions'] += 1
        if decision == 'gnn':
            self.gate_stats['gnn_decisions'] += 1
        else:
            self.gate_stats['llm_decisions'] += 1
        
        # Update average gate score
        alpha = 0.1
        self.gate_stats['avg_gate_score'] = (
            (1 - alpha) * self.gate_stats['avg_gate_score'] + alpha * gate_score
        )
        
        # Store decision
        self.routing_history.append({
            'score': gate_score,
            'decision': decision,
            'threshold': threshold
        })
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
        
        return decision
    
    def compute_conf(
        self,
        prediction: torch.Tensor
    ) -> float:
        """
        Compute confidence from prediction (alias for backward compatibility).
        
        Args:
            prediction: Model prediction logits.
        
        Returns:
            Confidence score.
        """
        return self.feature_extractor.get_confidence(prediction)
    
    def compute_density(
        self,
        graph: Union[HeteroGraph, Any],
        node_id: Optional[int] = None,
        node_type: Optional[str] = None
    ) -> float:
        """
        Compute graph density (alias for backward compatibility).
        
        Args:
            graph: Graph object.
            node_id: Optional node ID.
            node_type: Optional node type.
        
        Returns:
            Graph density score.
        """
        return self.feature_extractor.get_graph_density(graph, node_id, node_type)
    
    def compute_criticality(
        self,
        context: Dict[str, Any]
    ) -> float:
        """
        Compute context criticality (alias for backward compatibility).
        
        Args:
            context: Context dictionary.
        
        Returns:
            Criticality score.
        """
        return self.feature_extractor.get_context_criticality(context)
    
    def compute_staleness(
        self,
        node: Any,
        current_time: Optional[float] = None
    ) -> float:
        """
        Compute node staleness (alias for backward compatibility).
        
        Args:
            node: Node object.
            current_time: Optional current time.
        
        Returns:
            Staleness score.
        """
        return self.feature_extractor.get_node_staleness(node, current_time)
    
    def get_gate_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get learnable gate network weights.
        
        Returns:
            Dict mapping layer names to weight tensors.
        """
        weights = {}
        for i, layer in enumerate(self.gate_network):
            if isinstance(layer, nn.Linear):
                weights[f'layer_{i}_weight'] = layer.weight
                weights[f'layer_{i}_bias'] = layer.bias
        
        return weights
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get routing statistics.
        
        Returns:
            Dict with routing statistics.
        """
        total = self.gate_stats['total_decisions']
        if total == 0:
            return {
                'total_decisions': 0,
                'gnn_ratio': 0.0,
                'llm_ratio': 0.0,
                'avg_gate_score': 0.0
            }
        
        return {
            'total_decisions': total,
            'gnn_ratio': self.gate_stats['gnn_decisions'] / total,
            'llm_ratio': self.gate_stats['llm_decisions'] / total,
            'avg_gate_score': self.gate_stats['avg_gate_score']
        }
    
    def reset_stats(self):
        """Reset routing statistics."""
        self.gate_stats = {
            'total_decisions': 0,
            'gnn_decisions': 0,
            'llm_decisions': 0,
            'avg_gate_score': 0.0
        }
        self.routing_history = []
    
    def forward(
        self,
        node: Any,
        context: Dict[str, Any],
        prediction: torch.Tensor,
        graph: Optional[Union[HeteroGraph, Any]] = None,
        current_time: Optional[float] = None,
        threshold: Optional[float] = None
    ) -> Tuple[str, float, GatingFeatures]:
        """
        Full forward pass: extract features, compute score, decide path.
        
        Args:
            node: Target node.
            context: Context dictionary.
            prediction: Model prediction.
            graph: Optional graph object.
            current_time: Optional current time.
            threshold: Optional custom threshold.
        
        Returns:
            Tuple of (decision, gate_score, features).
        """
        # Extract features
        features = self.feature_extractor(
            node=node,
            context=context,
            prediction=prediction,
            graph=graph,
            current_time=current_time
        )
        
        # Compute gate score
        gate_score = self.compute_gating_score(features)
        
        # Make decision
        decision = self.decide_path(gate_score, threshold)
        
        return decision, gate_score, features
    
    def set_threshold(self, threshold: float):
        """
        Set new gating threshold.
        
        Args:
            threshold: New threshold value between 0 and 1.
        
        Raises:
            ValueError: If threshold is out of range.
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        
        self.threshold = threshold
        self.logger.log_info(f"Gate threshold updated to {threshold}")
    
    def get_threshold(self) -> float:
        """Get current gating threshold."""
        return self.threshold
    
    def optimize_threshold(
        self,
        validation_data: List[Tuple[Any, Dict[str, Any], torch.Tensor]],
        llm_performance: float,
        gnn_performance: float,
        cost_function: Optional[Callable] = None
    ) -> float:
        """
        Optimize gating threshold on validation data.
        
        Args:
            validation_data: List of (node, context, prediction) tuples.
            llm_performance: LLM path performance metric.
            gnn_performance: GNN path performance metric.
            cost_function: Optional cost function for optimization.
        
        Returns:
            Optimal threshold value.
        """
        self.logger.log_info("Optimizing gate threshold")
        
        if not validation_data:
            self.logger.log_warning("No validation data provided, keeping current threshold")
            return self.threshold
        
        best_threshold = self.threshold
        best_score = -float('inf')
        
        # Try different thresholds
        for threshold in np.linspace(0.1, 0.9, 20):
            # Simulate performance on validation data
            gnn_used = 0
            llm_used = 0
            
            for node, context, prediction in validation_data:
                # Compute gate score
                features = self.feature_extractor(
                    node=node,
                    context=context,
                    prediction=prediction
                )
                gate_score = self.compute_gating_score(features)
                decision = self.decide_path(gate_score, threshold)
                
                if decision == 'gnn':
                    gnn_used += 1
                else:
                    llm_used += 1
            
            # Compute optimization score
            total = len(validation_data)
            if total == 0:
                continue
            
            gnn_ratio = gnn_used / total
            llm_ratio = llm_used / total
            
            # Expected performance
            expected_performance = (
                gnn_ratio * gnn_performance +
                llm_ratio * llm_performance
            )
            
            # Cost factor (if provided)
            if cost_function is not None:
                cost = cost_function(gnn_used, llm_used)
                score = expected_performance - 0.01 * cost
            else:
                # Simple: maximize performance with LLM usage penalty
                score = expected_performance - 0.1 * llm_ratio
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.logger.log_info(f"Optimized threshold: {best_threshold:.3f}")
        self.set_threshold(best_threshold)
        
        return best_threshold
    
    def save_gate(self, save_path: str):
        """
        Save gate parameters.
        
        Args:
            save_path: Path to save the gate.
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            state_dict = {
                'gate_network': self.gate_network.state_dict(),
                'threshold': self.threshold,
                'gate_stats': self.gate_stats,
                'config': self.config
            }
            
            torch.save(state_dict, save_path)
            self.logger.log_info(f"Gate saved to {save_path}")
        
        except Exception as e:
            self.logger.log_error(f"Failed to save gate: {e}")
            raise
    
    def load_gate(self, load_path: str):
        """
        Load gate parameters.
        
        Args:
            load_path: Path to load the gate from.
        
        Raises:
            FileNotFoundError: If checkpoint not found.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Gate checkpoint not found: {load_path}")
        
        try:
            checkpoint = torch.load(load_path, map_location='cpu')
            
            self.gate_network.load_state_dict(checkpoint['gate_network'])
            self.threshold = checkpoint.get('threshold', self.threshold)
            self.gate_stats = checkpoint.get('gate_stats', self.gate_stats)
            
            if 'config' in checkpoint:
                self.config = checkpoint['config']
            
            self.logger.log_info(f"Gate loaded from {load_path}")
        
        except Exception as e:
            self.logger.log_error(f"Failed to load gate: {e}")
            raise
    
    def get_parameters(self) -> Dict[str, int]:
        """
        Get parameter statistics.
        
        Returns:
            Dict with parameter counts.
        """
        return {
            'gate_network': sum(p.numel() for p in self.gate_network.parameters()),
            'total': sum(p.numel() for p in self.parameters())
        }
    
    def to_device(self, device: torch.device) -> 'AdaptiveGate':
        """
        Move gate to specified device.
        
        Args:
            device: PyTorch device.
        
        Returns:
            Self with gate moved to device.
        """
        self.to(device)
        return self


# Module level variables and exports
__all__ = [
    'GatingFeatures',
    'GatingFeaturesExtractor',
    'AdaptiveGate',
    '__doc__'
]

# Version information
__version__ = '1.0.0'
__author__ = 'H-GRAGrecsys Team'


def create_adaptive_gate(
    config_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> AdaptiveGate:
    """
    Factory function to create an AdaptiveGate instance.
    
    Args:
        config_path: Optional path to configuration file.
        device: Optional device to move gate to. Defaults to CUDA if available.
    
    Returns:
        Initialized AdaptiveGate instance.
    
    Example:
        >>> gate = create_adaptive_gate('config/default_config.yaml')
        >>> gate.to_device(torch.device('cuda'))
    """
    gate = AdaptiveGate(config_path)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return gate.to_device(device)


def create_gating_features(
    confidence: float = 0.5,
    graph_density: float = 0.5,
    context_criticality: float = 0.5,
    node_staleness: float = 0.0,
    uncertainty: float = 0.0,
    complexity: float = 0.5,
    device: Optional[torch.device] = None
) -> GatingFeatures:
    """
    Factory function to create a GatingFeatures object.
    
    Args:
        confidence: Confidence score (0-1).
        graph_density: Graph density (0-1).
        context_criticality: Context criticality (0-1).
        node_staleness: Node staleness (0-1).
        uncertainty: Prediction uncertainty (0-1).
        complexity: Task complexity (0-1).
        device: Optional device for tensor conversion.
    
    Returns:
        GatingFeatures object.
    
    Example:
        >>> features = create_gating_features(
        ...     confidence=0.8,
        ...     graph_density=0.6,
        ...     context_criticality=0.3
        ... )
    """
    return GatingFeatures(
        confidence=confidence,
        graph_density=graph_density,
        context_criticality=context_criticality,
        node_staleness=node_staleness,
        uncertainty=uncertainty,
        complexity=complexity
    )