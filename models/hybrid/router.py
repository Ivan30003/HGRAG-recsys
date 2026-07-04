"""
Router Module for H-GRAGrecsys

This module implements the hybrid router that coordinates between GNN and LLM
paths based on adaptive gating decisions. The router handles:
- Request routing between GNN and LLM inference paths
- Prediction aggregation and combination
- Fallback mechanisms for failed predictions
- Routing statistics and monitoring
- Dynamic path switching based on context

The router enables efficient hybrid inference by leveraging GNN for simple
cases and LLM for complex, uncertain, or critical cases.
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from sibling modules
from models.hybrid.adaptive_gate import AdaptiveGate, GatingFeatures, GatingFeaturesExtractor

# Import from graph module
from models.graph.heterogeneous_graph import HeterogeneousGraph
from models.graph.relation_types import RelationType

# Import from GNN module
from models.gnn.gnn_encoder import GNNEncoder

# Import from LLM module
from models.llm.llm_interface import LLMInterface

# Import from utils
from utils.config_loader import ConfigLoader
from utils.logger import Logger
from utils.seed_manager import SeedManager
from utils.timer import Timer


@dataclass
class RoutingDecision:
    """
    Dataclass for routing decision information.
    
    Attributes:
        path: Selected path ('gnn' or 'llm').
        gate_score: Gating score used for decision.
        confidence: Prediction confidence.
        explanation: Optional explanation for the decision.
        timestamp: Decision timestamp.
        metadata: Additional metadata.
    """
    path: str
    gate_score: float
    confidence: float
    explanation: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Router(nn.Module):
    """
    Hybrid router for coordinating GNN and LLM inference paths.
    
    This class manages the routing of requests between GNN and LLM paths
    based on adaptive gating decisions. It handles prediction aggregation,
    fallback mechanisms, and performance monitoring.
    """
    
    def __init__(
        self,
        gate: Optional[AdaptiveGate] = None,
        gnn_encoder: Optional[GNNEncoder] = None,
        llm_interface: Optional[LLMInterface] = None,
        config: Optional[Union[str, Dict, ConfigLoader]] = None
    ):
        """
        Initialize the hybrid router.
        
        Args:
            gate: Optional AdaptiveGate instance. If None, creates from config.
            gnn_encoder: Optional GNNEncoder instance. If None, creates from config.
            llm_interface: Optional LLMInterface instance. If None, creates from config.
            config: Configuration object or path to config file.
                   Can be a string path, dict, or ConfigLoader instance.
        
        Raises:
            ValueError: If config is invalid or missing required fields.
        """
        super(Router, self).__init__()
        
        # Load configuration
        if config is None:
            self.config = {
                'model': {
                    'hybrid': {
                        'gate_threshold': 0.3,
                        'uniform_llm_rate': 0.15,
                        'fallback_strategy': 'gnn',
                        'combine_predictions': True,
                        'combine_weight': 0.5
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
            name='router'
        )
        
        # Extract configuration
        hybrid_config = self.config.get('model', {}).get('hybrid', {})
        self.threshold = hybrid_config.get('gate_threshold', 0.3)
        self.uniform_llm_rate = hybrid_config.get('uniform_llm_rate', 0.15)
        self.fallback_strategy = hybrid_config.get('fallback_strategy', 'gnn')
        self.combine_predictions = hybrid_config.get('combine_predictions', True)
        self.combine_weight = hybrid_config.get('combine_weight', 0.5)
        
        # Initialize components
        self.gate = gate if gate is not None else self._create_gate()
        self.gnn_encoder = gnn_encoder if gnn_encoder is not None else self._create_gnn_encoder()
        self.llm_interface = llm_interface if llm_interface is not None else self._create_llm_interface()
        
        # Routing state
        self.routing_history: List[RoutingDecision] = []
        self.routing_stats = {
            'total_requests': 0,
            'gnn_requests': 0,
            'llm_requests': 0,
            'fallback_requests': 0,
            'combined_requests': 0,
            'avg_gnn_time': 0.0,
            'avg_llm_time': 0.0,
            'avg_combined_time': 0.0
        }
        
        # Performance tracking
        self.gnn_performance = []
        self.llm_performance = []
        self.combined_performance = []
        
        self.logger.log_info(
            f"Router initialized: threshold={self.threshold}, "
            f"fallback={self.fallback_strategy}, combine={self.combine_predictions}"
        )
    
    def _create_gate(self) -> AdaptiveGate:
        """Create adaptive gate from configuration."""
        return AdaptiveGate(self.config)
    
    def _create_gnn_encoder(self) -> GNNEncoder:
        """Create GNN encoder from configuration."""
        return GNNEncoder(config=self.config)
    
    def _create_llm_interface(self) -> LLMInterface:
        """Create LLM interface from configuration."""
        return LLMInterface(config=self.config)
    
    def route(
        self,
        node: Any,
        context: Dict[str, Any],
        prediction: Optional[torch.Tensor] = None,
        graph: Optional[Union[HeteroGraph, Any]] = None,
        current_time: Optional[float] = None,
        force_path: Optional[str] = None
    ) -> RoutingDecision:
        """
        Route a request to GNN or LLM path.
        
        Args:
            node: Target node (user or item).
            context: Context dictionary containing query information.
            prediction: Optional model prediction.
            graph: Optional graph object.
            current_time: Optional current time for staleness.
            force_path: Optional force routing to specific path ('gnn' or 'llm').
        
        Returns:
            RoutingDecision object containing routing information.
        
        Raises:
            ValueError: If force_path is invalid.
        """
        self.routing_stats['total_requests'] += 1
        
        # Force path if specified
        if force_path is not None:
            if force_path not in ['gnn', 'llm']:
                raise ValueError(f"Invalid force_path: {force_path}. Must be 'gnn' or 'llm'")
            
            self.logger.log_info(f"Forcing routing to {force_path} path")
            decision = RoutingDecision(
                path=force_path,
                gate_score=1.0 if force_path == 'gnn' else 0.0,
                confidence=1.0,
                explanation=f"Forced to {force_path} path"
            )
            self._update_stats(decision)
            return decision
        
        # Get gating decision
        decision_path, gate_score, features = self.gate.forward(
            node=node,
            context=context,
            prediction=prediction,
            graph=graph,
            current_time=current_time
        )
        
        # Compute confidence
        confidence = features.confidence if features else 0.5
        
        # Create routing decision
        decision = RoutingDecision(
            path=decision_path,
            gate_score=gate_score,
            confidence=confidence,
            metadata={
                'features': features,
                'context': context
            }
        )
        
        # Update statistics
        self._update_stats(decision)
        
        # Store in history
        self.routing_history.append(decision)
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
        
        self.logger.log_info(
            f"Routing decision: {decision_path} (score={gate_score:.3f}, "
            f"confidence={confidence:.3f})"
        )
        
        return decision
    
    def _update_stats(self, decision: RoutingDecision):
        """Update routing statistics."""
        if decision.path == 'gnn':
            self.routing_stats['gnn_requests'] += 1
        else:
            self.routing_stats['llm_requests'] += 1
    
    def apply_routing_prediction(
        self,
        gnn_pred: torch.Tensor,
        llm_pred: torch.Tensor,
        decision: RoutingDecision,
        combine: Optional[bool] = None
    ) -> Tuple[torch.Tensor, str, Dict[str, Any]]:
        """
        Apply routing decision to get final prediction.
        
        Args:
            gnn_pred: GNN model prediction.
            llm_pred: LLM model prediction.
            decision: Routing decision.
            combine: Optional override for combination flag.
        
        Returns:
            Tuple of (final_prediction, used_path, metadata).
        """
        used_path = decision.path
        final_pred = None
        metadata = {
            'gnn_pred': gnn_pred,
            'llm_pred': llm_pred,
            'gate_score': decision.gate_score,
            'confidence': decision.confidence
        }
        
        combine_predictions = combine if combine is not None else self.combine_predictions
        
        if combine_predictions and gnn_pred is not None and llm_pred is not None:
            # Combine predictions
            weight = self.combine_weight
            final_pred = weight * gnn_pred + (1 - weight) * llm_pred
            used_path = 'combined'
            self.routing_stats['combined_requests'] += 1
            metadata['combine_weight'] = weight
            self.logger.log_info("Combined GNN and LLM predictions")
        elif decision.path == 'gnn':
            if gnn_pred is not None:
                final_pred = gnn_pred
            else:
                # Fallback
                final_pred, fallback_path = self._handle_fallback(gnn_pred, llm_pred)
                used_path = fallback_path
                metadata['fallback'] = True
        else:  # llm path
            if llm_pred is not None:
                final_pred = llm_pred
            else:
                # Fallback
                final_pred, fallback_path = self._handle_fallback(gnn_pred, llm_pred)
                used_path = fallback_path
                metadata['fallback'] = True
        
        return final_pred, used_path, metadata
    
    def _handle_fallback(
        self,
        gnn_pred: Optional[torch.Tensor],
        llm_pred: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, str]:
        """
        Handle fallback when requested path prediction is unavailable.
        
        Args:
            gnn_pred: GNN prediction (may be None).
            llm_pred: LLM prediction (may be None).
        
        Returns:
            Tuple of (fallback_prediction, fallback_path).
        
        Raises:
            RuntimeError: If both predictions are None.
        """
        self.routing_stats['fallback_requests'] += 1
        
        if self.fallback_strategy == 'gnn':
            if gnn_pred is not None:
                self.logger.log_info("Fallback to GNN path")
                return gnn_pred, 'gnn_fallback'
            elif llm_pred is not None:
                self.logger.log_info("Fallback to LLM path")
                return llm_pred, 'llm_fallback'
        elif self.fallback_strategy == 'llm':
            if llm_pred is not None:
                self.logger.log_info("Fallback to LLM path")
                return llm_pred, 'llm_fallback'
            elif gnn_pred is not None:
                self.logger.log_info("Fallback to GNN path")
                return gnn_pred, 'gnn_fallback'
        else:  # 'ensemble' or other
            if gnn_pred is not None and llm_pred is not None:
                # Use average
                avg_pred = (gnn_pred + llm_pred) / 2
                self.logger.log_info("Fallback to ensemble (average)")
                return avg_pred, 'ensemble_fallback'
            elif gnn_pred is not None:
                return gnn_pred, 'gnn_fallback'
            elif llm_pred is not None:
                return llm_pred, 'llm_fallback'
        
        # Both predictions are None
        raise RuntimeError("Both GNN and LLM predictions are None, cannot handle fallback")
    
    def get_routing_decision(
        self,
        gate_score: float,
        threshold: Optional[float] = None
    ) -> str:
        """
        Get routing decision from gate score (alias for compatibility).
        
        Args:
            gate_score: Gating score.
            threshold: Optional custom threshold.
        
        Returns:
            Routing decision: 'gnn' or 'llm'.
        """
        if threshold is None:
            threshold = self.threshold
        
        return self.gate.decide_path(gate_score, threshold)
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive routing statistics.
        
        Returns:
            Dict containing routing statistics.
        """
        total = self.routing_stats['total_requests']
        if total == 0:
            return {
                'total_requests': 0,
                'gnn_ratio': 0.0,
                'llm_ratio': 0.0,
                'fallback_ratio': 0.0,
                'combined_ratio': 0.0,
                'avg_gate_score': 0.0
            }
        
        gate_stats = self.gate.get_routing_stats() if self.gate else {}
        
        return {
            'total_requests': total,
            'gnn_ratio': self.routing_stats['gnn_requests'] / total,
            'llm_ratio': self.routing_stats['llm_requests'] / total,
            'fallback_ratio': self.routing_stats['fallback_requests'] / total,
            'combined_ratio': self.routing_stats['combined_requests'] / total,
            'avg_gnn_time': self.routing_stats['avg_gnn_time'],
            'avg_llm_time': self.routing_stats['avg_llm_time'],
            'avg_combined_time': self.routing_stats['avg_combined_time'],
            'gate_stats': gate_stats
        }
    
    def update_performance_metrics(
        self,
        path: str,
        execution_time: float,
        quality_score: Optional[float] = None
    ):
        """
        Update performance metrics for a path.
        
        Args:
            path: Path used ('gnn', 'llm', 'combined').
            execution_time: Execution time in seconds.
            quality_score: Optional quality metric.
        """
        if path == 'gnn':
            self.gnn_performance.append({
                'time': execution_time,
                'quality': quality_score
            })
            # Update average
            alpha = 0.1
            self.routing_stats['avg_gnn_time'] = (
                (1 - alpha) * self.routing_stats['avg_gnn_time'] +
                alpha * execution_time
            )
        elif path == 'llm':
            self.llm_performance.append({
                'time': execution_time,
                'quality': quality_score
            })
            alpha = 0.1
            self.routing_stats['avg_llm_time'] = (
                (1 - alpha) * self.routing_stats['avg_llm_time'] +
                alpha * execution_time
            )
        elif path == 'combined':
            self.combined_performance.append({
                'time': execution_time,
                'quality': quality_score
            })
            alpha = 0.1
            self.routing_stats['avg_combined_time'] = (
                (1 - alpha) * self.routing_stats['avg_combined_time'] +
                alpha * execution_time
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dict with performance statistics.
        """
        def _compute_stats(data):
            if not data:
                return {'count': 0, 'avg_time': 0.0, 'avg_quality': 0.0}
            
            times = [d['time'] for d in data]
            qualities = [d['quality'] for d in data if d['quality'] is not None]
            
            return {
                'count': len(data),
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'avg_quality': np.mean(qualities) if qualities else None,
                'std_quality': np.std(qualities) if qualities else None
            }
        
        return {
            'gnn': _compute_stats(self.gnn_performance),
            'llm': _compute_stats(self.llm_performance),
            'combined': _compute_stats(self.combined_performance)
        }
    
    def reset_statistics(self):
        """Reset all routing and performance statistics."""
        self.routing_stats = {
            'total_requests': 0,
            'gnn_requests': 0,
            'llm_requests': 0,
            'fallback_requests': 0,
            'combined_requests': 0,
            'avg_gnn_time': 0.0,
            'avg_llm_time': 0.0,
            'avg_combined_time': 0.0
        }
        self.routing_history = []
        self.gnn_performance = []
        self.llm_performance = []
        self.combined_performance = []
        
        if self.gate:
            self.gate.reset_stats()
        
        self.logger.log_info("Statistics reset")
    
    def set_threshold(self, threshold: float):
        """
        Set gating threshold.
        
        Args:
            threshold: New threshold value between 0 and 1.
        """
        if self.gate:
            self.gate.set_threshold(threshold)
            self.threshold = threshold
        else:
            self.threshold = threshold
        
        self.logger.log_info(f"Threshold updated to {threshold}")
    
    def optimize_gating(
        self,
        validation_data: List[Tuple[Any, Dict[str, Any], torch.Tensor]],
        llm_performance: float,
        gnn_performance: float,
        cost_function: Optional[Callable] = None
    ) -> float:
        """
        Optimize gating threshold using validation data.
        
        Args:
            validation_data: List of (node, context, prediction) tuples.
            llm_performance: LLM path performance metric.
            gnn_performance: GNN path performance metric.
            cost_function: Optional cost function for optimization.
        
        Returns:
            Optimized threshold value.
        """
        if self.gate:
            optimal_threshold = self.gate.optimize_threshold(
                validation_data=validation_data,
                llm_performance=llm_performance,
                gnn_performance=gnn_performance,
                cost_function=cost_function
            )
            self.threshold = optimal_threshold
            return optimal_threshold
        else:
            self.logger.log_warning("No gate available for optimization")
            return self.threshold
    
    def save_router(self, save_path: str):
        """
        Save router state including gate and components.
        
        Args:
            save_path: Path to save the router.
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            state_dict = {
                'gate': self.gate.state_dict() if self.gate else None,
                'threshold': self.threshold,
                'routing_stats': self.routing_stats,
                'config': self.config,
                'version': __version__
            }
            
            torch.save(state_dict, save_path)
            self.logger.log_info(f"Router saved to {save_path}")
        
        except Exception as e:
            self.logger.log_error(f"Failed to save router: {e}")
            raise
    
    def load_router(self, load_path: str):
        """
        Load router state.
        
        Args:
            load_path: Path to load the router from.
        
        Raises:
            FileNotFoundError: If checkpoint not found.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Router checkpoint not found: {load_path}")
        
        try:
            checkpoint = torch.load(load_path, map_location='cpu')
            
            if checkpoint.get('gate') is not None and self.gate:
                self.gate.load_state_dict(checkpoint['gate'])
            
            self.threshold = checkpoint.get('threshold', self.threshold)
            self.routing_stats = checkpoint.get('routing_stats', self.routing_stats)
            
            if 'config' in checkpoint:
                self.config = checkpoint['config']
            
            self.logger.log_info(f"Router loaded from {load_path}")
        
        except Exception as e:
            self.logger.log_error(f"Failed to load router: {e}")
            raise
    
    def get_gate_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get gate network weights.
        
        Returns:
            Dict of weight tensors.
        """
        if self.gate:
            return self.gate.get_gate_weights()
        else:
            return {}
    
    def get_parameters(self) -> Dict[str, int]:
        """
        Get parameter statistics for all components.
        
        Returns:
            Dict with parameter counts.
        """
        params = {'total': 0}
        
        if self.gate:
            gate_params = self.gate.get_parameters()
            params['gate'] = gate_params['total']
            params['total'] += gate_params['total']
        
        if self.gnn_encoder:
            gnn_params = self.gnn_encoder.get_parameters()
            params['gnn_encoder'] = gnn_params['total']
            params['total'] += gnn_params['total']
        
        if self.llm_interface:
            llm_params = self.llm_interface.get_parameters()
            params['llm_interface'] = llm_params['total']
            params['total'] += llm_params['total']
        
        return params
    
    def to_device(self, device: torch.device) -> 'Router':
        """
        Move all components to specified device.
        
        Args:
            device: PyTorch device.
        
        Returns:
            Self with components moved to device.
        """
        if self.gate:
            self.gate.to_device(device)
        if self.gnn_encoder:
            self.gnn_encoder.to_device(device)
        if self.llm_interface:
            self.llm_interface.to_device(device)
        
        self.to(device)
        self.logger.log_info(f"Router moved to device: {device}")
        
        return self
    
    def forward(
        self,
        node: Any,
        context: Dict[str, Any],
        gnn_pred: Optional[torch.Tensor] = None,
        llm_pred: Optional[torch.Tensor] = None,
        graph: Optional[Union[HeteroGraph, Any]] = None,
        current_time: Optional[float] = None,
        force_path: Optional[str] = None,
        combine: Optional[bool] = None
    ) -> Tuple[torch.Tensor, RoutingDecision, Dict[str, Any]]:
        """
        Full forward pass: route and apply prediction.
        
        Args:
            node: Target node.
            context: Context dictionary.
            gnn_pred: GNN prediction (optional).
            llm_pred: LLM prediction (optional).
            graph: Optional graph object.
            current_time: Optional current time.
            force_path: Optional force routing to specific path.
            combine: Optional override for combination flag.
        
        Returns:
            Tuple of (final_prediction, routing_decision, metadata).
        """
        # Get routing decision
        decision = self.route(
            node=node,
            context=context,
            prediction=gnn_pred or llm_pred,
            graph=graph,
            current_time=current_time,
            force_path=force_path
        )
        
        # Apply routing decision
        final_pred, used_path, metadata = self.apply_routing_prediction(
            gnn_pred=gnn_pred,
            llm_pred=llm_pred,
            decision=decision,
            combine=combine
        )
        
        # Update metadata
        metadata['used_path'] = used_path
        metadata['routing_decision'] = decision
        
        return final_pred, decision, metadata


# Module level variables and exports
__all__ = [
    'RoutingDecision',
    'Router',
    '__doc__'
]

# Version information
__version__ = '1.0.0'
__author__ = 'H-GRAGrecsys Team'


def create_router(
    config_path: Optional[str] = None,
    gate: Optional[AdaptiveGate] = None,
    gnn_encoder: Optional[GNNEncoder] = None,
    llm_interface: Optional[LLMInterface] = None,
    device: Optional[torch.device] = None
) -> Router:
    """
    Factory function to create a Router instance.
    
    Args:
        config_path: Optional path to configuration file.
        gate: Optional AdaptiveGate instance.
        gnn_encoder: Optional GNNEncoder instance.
        llm_interface: Optional LLMInterface instance.
        device: Optional device to move router to. Defaults to CUDA if available.
    
    Returns:
        Initialized Router instance.
    
    Example:
        >>> router = create_router(
        ...     config_path='config/default_config.yaml',
        ...     gate=gate,
        ...     gnn_encoder=encoder,
        ...     llm_interface=llm
        ... )
        >>> router.to_device(torch.device('cuda'))
    """
    router = Router(
        gate=gate,
        gnn_encoder=gnn_encoder,
        llm_interface=llm_interface,
        config=config_path
    )
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return router.to_device(device)


def create_routing_decision(
    path: str,
    gate_score: float,
    confidence: float,
    explanation: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> RoutingDecision:
    """
    Factory function to create a RoutingDecision object.
    
    Args:
        path: Selected path ('gnn' or 'llm').
        gate_score: Gating score.
        confidence: Prediction confidence.
        explanation: Optional explanation.
        metadata: Optional metadata.
    
    Returns:
        RoutingDecision object.
    
    Example:
        >>> decision = create_routing_decision(
        ...     path='gnn',
        ...     gate_score=0.8,
        ...     confidence=0.9,
        ...     explanation='High confidence GNN prediction'
        ... )
    """
    return RoutingDecision(
        path=path,
        gate_score=gate_score,
        confidence=confidence,
        explanation=explanation,
        metadata=metadata or {}
    )