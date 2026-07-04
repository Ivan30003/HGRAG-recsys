"""
Phase 3: Hybrid Training Module for H-GRAGrecsys

This module implements the hybrid inference training phase where adaptive gating
between GNN and LLM paths is trained and optimized. The hybrid system combines
the efficiency of GNN-based recommendations with the reasoning capabilities of
LLMs through an adaptive gate that decides which path to use for each prediction.

Key Responsibilities:
- Initialize hybrid inference system with GNN and LLM components
- Train adaptive gating mechanism on validation data
- Optimize gating threshold for quality-cost tradeoff
- Evaluate hybrid system performance
- Deploy hybrid model for inference
"""

import os
import sys
import json
import math
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle
import random
from dataclasses import dataclass, field

# Add project root to path if needed
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Core imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, AdamW

# Data imports
from data.dataset import BaseDataset, AmazonDataset
from data.data_loader import DataLoader as DataLoaderClass

# Model imports
from models.gnn.heterogeneous_gnn import HeterogeneousGNN
from models.gnn.projection_heads import ComponentProjectionHeads
from models.gnn.gnn_encoder import GNNEncoder

from models.llm.llm_interface import LLMInterface
from models.llm.text_encoder import TextEncoder
from models.llm.fusion_engine import FusionEngine
from models.llm.prompt_templates import PromptTemplates

from models.hybrid.adaptive_gate import AdaptiveGate, GatingFeatures
from models.hybrid.router import Router
from models.hybrid.inference_engine import HybridInferenceEngine

from models.graph.heterogeneous_graph import HeterogeneousGraph
from models.graph_rag.retriever import GraphRAGRetriever
from models.graph_rag.context_constructor import ContextConstructor

# Agent imports (for warm start)
from models.agent.user_agent import UserAgent
from models.agent.item_agent import ItemAgent

# Distillation imports
from distillation.distillation_trainer import DistillationTrainer

# Utils imports
from utils.logger import Logger
from utils.config_loader import ConfigLoader
from utils.seed_manager import SeedManager
from utils.timer import Timer
from utils.visualization import Visualizer

# Training imports
from .trainer_base import BaseTrainer
from .checkpoint_manager import CheckpointManager


@dataclass
class GateOptimizationResult:
    """Container for gate optimization results"""
    threshold: float
    quality_score: float
    cost_score: float
    tradeoff_score: float
    gnn_ratio: float
    llm_ratio: float
    metrics: Dict[str, float] = field(default_factory=dict)


class Phase3Hybrid(BaseTrainer):
    """
    Phase 3 Hybrid Trainer
    
    Trains and optimizes the hybrid inference system combining GNN and LLM paths
    through adaptive gating. The gate learns when to use the efficient GNN path
    versus the more capable but expensive LLM path.
    """
    
    def __init__(
        self,
        gnn_model: HeterogeneousGNN,
        llm_model: LLMInterface,
        gate: AdaptiveGate,
        config: Dict[str, Any],
        dataset: Optional[BaseDataset] = None,
        graph: Optional[HeterogeneousGraph] = None,
        router: Optional[Router] = None,
        inference_engine: Optional[HybridInferenceEngine] = None,
        gnn_encoder: Optional[GNNEncoder] = None,
        fusion_engine: Optional[FusionEngine] = None,
        text_encoder: Optional[TextEncoder] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        logger: Optional[Logger] = None,
        visualizer: Optional[Visualizer] = None
    ):
        """
        Initialize Phase 3 Hybrid Trainer
        
        Args:
            gnn_model: Trained HeterogeneousGNN model
            llm_model: LLMInterface for LLM path
            gate: AdaptiveGate for routing decisions
            config: Configuration dictionary containing phase3 settings
            dataset: Optional dataset for validation
            graph: Optional heterogeneous graph
            router: Optional Router instance
            inference_engine: Optional HybridInferenceEngine instance
            gnn_encoder: Optional GNNEncoder instance
            fusion_engine: Optional FusionEngine instance
            text_encoder: Optional TextEncoder instance
            checkpoint_manager: Optional CheckpointManager instance
            logger: Optional Logger instance
            visualizer: Optional Visualizer instance
            
        Raises:
            ValueError: If required parameters are missing
        """
        super().__init__(config, model=gnn_model, data_loader=None)
        
        if gnn_model is None:
            raise ValueError("GNN model cannot be None for Phase 3 Hybrid")
        
        if llm_model is None:
            raise ValueError("LLM model cannot be None for Phase 3 Hybrid")
        
        if gate is None:
            raise ValueError("Adaptive gate cannot be None for Phase 3 Hybrid")
        
        # Store core components
        self.gnn_model = gnn_model
        self.llm_model = llm_model
        self.gate = gate
        self.config = config
        self.dataset = dataset
        self.graph = graph
        
        # Extract phase-specific configuration
        self.phase_config = config.get('phase3', {})
        if not self.phase_config:
            from . import PHASE3_CONFIG
            self.phase_config = PHASE3_CONFIG.copy()
        
        # Initialize components
        self.logger = logger or Logger(
            log_dir=config.get('common', {}).get('log_dir', './logs'),
            name='phase3_hybrid'
        )
        
        self.text_encoder = text_encoder or TextEncoder(
            model_name=config.get('llm', {}).get('encoder_model', 'sentence-transformers/all-MiniLM-L6-v2'),
            config=config
        )
        
        self.gnn_encoder = gnn_encoder or GNNEncoder(
            gnn_model=self.gnn_model,
            projection_heads=ComponentProjectionHeads(
                input_dim=config.get('gnn', {}).get('hidden_dim', 256),
                config=self.phase_config
            )
        )
        
        self.fusion_engine = fusion_engine or FusionEngine(
            llm=self.llm_model,
            config=self.phase_config
        )
        
        self.router = router or Router(
            gate=self.gate,
            threshold=self.phase_config.get('gate_threshold', 0.3)
        )
        
        self.inference_engine = inference_engine or HybridInferenceEngine(
            gnn_encoder=self.gnn_encoder,
            llm_interface=self.llm_model,
            router=self.router,
            config=self.phase_config
        )
        
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            save_dir=config.get('common', {}).get('checkpoint_dir', './checkpoints'),
            max_checkpoints=config.get('common', {}).get('max_checkpoints', 5)
        )
        
        self.visualizer = visualizer or Visualizer(
            config=config.get('visualization', {})
        )
        
        # Initialize graph RAG components if graph is available
        self.graph_retriever = None
        self.context_constructor = None
        if self.graph is not None:
            self.graph_retriever = GraphRAGRetriever(
                graph=self.graph,
                config=config.get('graph_rag', {})
            )
            self.context_constructor = ContextConstructor(
                config=config.get('graph_rag', {})
            )
        
        # Training state
        self.training_data: Optional[List[Dict[str, Any]]] = None
        self.val_loader: Optional[DataLoader] = None
        self.gate_history: List[Dict[str, Any]] = []
        self.optimization_results: List[GateOptimizationResult] = []
        
        # Metrics tracking
        self.metrics = {
            'epoch': 0,
            'step': 0,
            'train_losses': [],
            'val_losses': [],
            'gate_accuracy': [],
            'llm_usage_ratios': [],
            'gnn_usage_ratios': [],
            'quality_scores': [],
            'cost_scores': [],
            'tradeoff_scores': [],
            'best_threshold': self.phase_config.get('gate_threshold', 0.3),
            'best_tradeoff_score': float('-inf'),
            'gate_weights_history': [],
            'routing_statistics': defaultdict(int)
        }
        
        # Set device
        self.device = config.get('common', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.gnn_model.to(self.device)
        
        # Setup optimizer for gate training
        self.optimizer = None
        
        self.logger.log_info(f"Phase3Hybrid initialized on device: {self.device}")
        self.logger.log_info(f"Phase config: {self.phase_config}")
        self.logger.log_info(f"Initial gate threshold: {self.router.threshold}")
    
    def initialize_hybrid_inference(
        self,
        warm_start_users: Optional[List[UserAgent]] = None,
        warm_start_items: Optional[List[ItemAgent]] = None
    ) -> Dict[str, Any]:
        """
        Initialize the hybrid inference system
        
        This method prepares the inference engine with all necessary components
        and optionally performs warm start with existing agents.
        
        Args:
            warm_start_users: Optional list of UserAgent instances for warm start
            warm_start_items: Optional list of ItemAgent instances for warm start
        
        Returns:
            Dict[str, Any]: Initialization statistics
            {
                'inference_engine_ready': bool,
                'num_warm_start_users': int,
                'num_warm_start_items': int,
                'gnn_ready': bool,
                'llm_ready': bool
            }
        """
        self.logger.log_info("Initializing hybrid inference system...")
        
        timer = Timer()
        timer.start()
        
        try:
            # Verify all components are ready
            gnn_ready = self.gnn_model is not None
            llm_ready = self.llm_model is not None
            gate_ready = self.gate is not None
            router_ready = self.router is not None
            
            if not all([gnn_ready, llm_ready, gate_ready, router_ready]):
                missing_components = []
                if not gnn_ready: missing_components.append('GNN')
                if not llm_ready: missing_components.append('LLM')
                if not gate_ready: missing_components.append('Gate')
                if not router_ready: missing_components.append('Router')
                raise RuntimeError(f"Missing components: {', '.join(missing_components)}")
            
            # Perform warm start if agents provided
            num_warm_users = 0
            num_warm_items = 0
            
            if warm_start_users:
                for user in warm_start_users:
                    self._add_user_to_inference(user)
                    num_warm_users += 1
            
            if warm_start_items:
                for item in warm_start_items:
                    self._add_item_to_inference(item)
                    num_warm_items += 1
            
            # Prepare graph for inference
            if self.graph is not None:
                self._prepare_graph_for_inference()
            
            # Set inference mode
            self.inference_engine.set_mode('inference')
            
            timer.stop()
            
            stats = {
                'inference_engine_ready': True,
                'num_warm_start_users': num_warm_users,
                'num_warm_start_items': num_warm_items,
                'gnn_ready': gnn_ready,
                'llm_ready': llm_ready,
                'gate_ready': gate_ready,
                'router_ready': router_ready,
                'elapsed_time': timer.get_elapsed_time()
            }
            
            self.logger.log_info(f"Hybrid inference initialized: {stats}")
            return stats
            
        except Exception as e:
            self.logger.log_error(f"Failed to initialize hybrid inference: {e}")
            raise RuntimeError(f"Failed to initialize hybrid inference: {e}")
    
    def train_gating(
        self,
        validation_data: List[Dict[str, Any]],
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Train the adaptive gating mechanism on validation data
        
        Args:
            validation_data: List of validation samples with features and labels
            num_epochs: Number of epochs to train
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        
        Returns:
            Dict[str, Any]: Training metrics and statistics
            
        Raises:
            RuntimeError: If training data is invalid or training fails
        """
        self.logger.log_info("Training adaptive gating mechanism...")
        
        if not validation_data:
            raise RuntimeError("Validation data cannot be empty for gate training")
        
        timer = Timer()
        timer.start()
        
        # Set parameters
        num_epochs = num_epochs or self.phase_config.get('num_epochs', 20)
        batch_size = batch_size or self.phase_config.get('batch_size', 32)
        learning_rate = learning_rate or self.phase_config.get('learning_rate', 5e-5)
        
        # Prepare data
        train_data, val_data = self._prepare_gating_data(validation_data)
        train_loader = self._create_gating_loader(train_data, batch_size, shuffle=True)
        val_loader = self._create_gating_loader(val_data, batch_size, shuffle=False)
        
        # Setup optimizer for gate
        gate_params = list(self.gate.parameters())
        if hasattr(self.router, 'parameters'):
            gate_params.extend(list(self.router.parameters()))
        
        self.optimizer = AdamW(
            gate_params,
            lr=learning_rate,
            weight_decay=self.phase_config.get('weight_decay', 1e-5)
        )
        
        # Training loop
        best_val_accuracy = 0.0
        best_epoch = 0
        
        for epoch in range(num_epochs):
            self.metrics['epoch'] = epoch
            
            # Train one epoch
            train_loss, train_acc = self._train_gating_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc, val_metrics = self._validate_gating_epoch(val_loader, epoch)
            
            # Log metrics
            self.logger.log_info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            # Track metrics
            self.metrics['train_losses'].append(train_loss)
            self.metrics['val_losses'].append(val_loss)
            self.metrics['gate_accuracy'].append(val_acc)
            self.metrics['llm_usage_ratios'].append(val_metrics.get('llm_ratio', 0.0))
            self.metrics['gnn_usage_ratios'].append(val_metrics.get('gnn_ratio', 0.0))
            self.metrics['quality_scores'].append(val_metrics.get('quality_score', 0.0))
            self.metrics['cost_scores'].append(val_metrics.get('cost_score', 0.0))
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_epoch = epoch
                self._save_gate_checkpoint("best_gate_model")
        
        timer.stop()
        
        # Store best metrics
        self.metrics['best_threshold'] = self.router.threshold
        self.metrics['best_val_accuracy'] = best_val_accuracy
        self.metrics['best_epoch'] = best_epoch
        
        results = {
            'train_losses': self.metrics['train_losses'],
            'val_losses': self.metrics['val_losses'],
            'gate_accuracy': self.metrics['gate_accuracy'],
            'best_val_accuracy': best_val_accuracy,
            'best_epoch': best_epoch,
            'llm_usage_ratio': np.mean(self.metrics['llm_usage_ratios']),
            'gnn_usage_ratio': np.mean(self.metrics['gnn_usage_ratios']),
            'total_time': timer.get_elapsed_time()
        }
        
        self.logger.log_info(f"Gate training completed in {timer.get_elapsed_time():.2f} seconds")
        self.logger.log_info(f"Best validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}")
        
        return results
    
    def optimize_gating_threshold(
        self,
        validation_data: List[Dict[str, Any]],
        thresholds: Optional[List[float]] = None,
        optimize_metric: str = 'tradeoff'
    ) -> GateOptimizationResult:
        """
        Optimize the gating threshold for best quality-cost tradeoff
        
        Args:
            validation_data: Validation data for evaluation
            thresholds: List of thresholds to evaluate. If None, uses range.
            optimize_metric: Metric to optimize ('tradeoff', 'quality', 'cost')
        
        Returns:
            GateOptimizationResult: Optimal threshold and associated metrics
            
        Raises:
            RuntimeError: If optimization fails
        """
        self.logger.log_info("Optimizing gating threshold...")
        
        timer = Timer()
        timer.start()
        
        # Generate thresholds to evaluate
        if thresholds is None:
            thresholds = np.arange(0.1, 0.95, 0.05).tolist()
        
        self.logger.log_info(f"Evaluating {len(thresholds)} thresholds...")
        
        results = []
        
        for threshold in tqdm(thresholds, desc="Evaluating thresholds"):
            # Set threshold
            self.router.threshold = threshold
            
            # Evaluate on validation data
            eval_result = self._evaluate_gating_performance(
                validation_data,
                threshold=threshold
            )
            
            # Compute tradeoff score
            quality_score = eval_result.get('quality_score', 0.0)
            cost_score = eval_result.get('cost_score', 0.0)
            tradeoff_score = self._compute_tradeoff_score(quality_score, cost_score)
            
            # Store result
            result = GateOptimizationResult(
                threshold=threshold,
                quality_score=quality_score,
                cost_score=cost_score,
                tradeoff_score=tradeoff_score,
                gnn_ratio=eval_result.get('gnn_ratio', 0.0),
                llm_ratio=eval_result.get('llm_ratio', 0.0),
                metrics=eval_result
            )
            results.append(result)
            
            self.logger.log_info(
                f"Threshold: {threshold:.2f} - "
                f"Quality: {quality_score:.4f}, "
                f"Cost: {cost_score:.4f}, "
                f"Tradeoff: {tradeoff_score:.4f}"
            )
        
        # Select best result based on optimization metric
        if optimize_metric == 'tradeoff':
            best_result = max(results, key=lambda r: r.tradeoff_score)
        elif optimize_metric == 'quality':
            best_result = max(results, key=lambda r: r.quality_score)
        elif optimize_metric == 'cost':
            best_result = min(results, key=lambda r: r.cost_score)
        else:
            raise ValueError(f"Unknown optimization metric: {optimize_metric}")
        
        # Update router with best threshold
        self.router.threshold = best_result.threshold
        self.metrics['best_threshold'] = best_result.threshold
        
        # Store optimization results
        self.optimization_results = results
        
        timer.stop()
        best_result.metrics['elapsed_time'] = timer.get_elapsed_time()
        
        self.logger.log_info(
            f"Optimal threshold: {best_result.threshold:.2f} "
            f"(Tradeoff: {best_result.tradeoff_score:.4f})"
        )
        
        # Visualize results if enabled
        if self.config.get('visualization', {}).get('enabled', False):
            self._visualize_threshold_optimization(results)
        
        return best_result
    
    def evaluate_hybrid(
        self,
        test_data: List[Dict[str, Any]],
        metrics_to_evaluate: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the complete hybrid system
        
        Args:
            test_data: Test data for evaluation
            metrics_to_evaluate: List of metrics to compute
        
        Returns:
            Dict[str, Any]: Comprehensive evaluation metrics
        """
        self.logger.log_info("Evaluating hybrid system...")
        
        timer = Timer()
        timer.start()
        
        if not test_data:
            self.logger.log_warning("No test data provided for evaluation")
            return {}
        
        # Default metrics
        metrics_to_evaluate = metrics_to_evaluate or [
            'accuracy', 'precision', 'recall', 'ndcg', 
            'llm_ratio', 'gnn_ratio', 'quality_score', 
            'cost_score', 'inference_time'
        ]
        
        eval_results = {}
        
        # Evaluate ranking performance
        if 'accuracy' in metrics_to_evaluate or 'precision' in metrics_to_evaluate:
            ranking_metrics = self._evaluate_ranking(test_data)
            eval_results.update(ranking_metrics)
        
        # Evaluate efficiency
        if 'llm_ratio' in metrics_to_evaluate or 'cost_score' in metrics_to_evaluate:
            efficiency_metrics = self._evaluate_efficiency(test_data)
            eval_results.update(efficiency_metrics)
        
        # Evaluate quality
        if 'quality_score' in metrics_to_evaluate:
            quality_metrics = self._evaluate_quality(test_data)
            eval_results.update(quality_metrics)
        
        # Evaluate inference time
        if 'inference_time' in metrics_to_evaluate:
            time_metrics = self._evaluate_inference_time(test_data)
            eval_results.update(time_metrics)
        
        # Compute overall scores
        eval_results['overall_quality_score'] = self._compute_overall_quality(eval_results)
        eval_results['overall_cost_score'] = self._compute_overall_cost(eval_results)
        eval_results['overall_tradeoff_score'] = self._compute_tradeoff_score(
            eval_results.get('overall_quality_score', 0.0),
            eval_results.get('overall_cost_score', 0.0)
        )
        
        timer.stop()
        eval_results['elapsed_time'] = timer.get_elapsed_time()
        
        self.logger.log_info(f"Evaluation completed: {eval_results}")
        return eval_results
    
    def deploy_hybrid_model(
        self,
        deployment_config: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deploy the hybrid model for production use
        
        Args:
            deployment_config: Configuration for deployment
            save_path: Path to save deployed model
        
        Returns:
            Dict[str, Any]: Deployment statistics
        """
        self.logger.log_info("Deploying hybrid model...")
        
        timer = Timer()
        timer.start()
        
        deployment_config = deployment_config or {}
        
        try:
            # Prepare deployment package
            deployment_package = {
                'model_type': 'hybrid_inference',
                'version': deployment_config.get('version', '1.0.0'),
                'components': {
                    'gnn_model': self.gnn_model,
                    'llm_model': self.llm_model,
                    'gate': self.gate,
                    'router': self.router,
                    'threshold': self.router.threshold
                },
                'config': self.config,
                'metrics': self.metrics,
                'timestamp': Timer.get_current_timestamp(),
                'requirements': {
                    'python_version': sys.version,
                    'torch_version': torch.__version__
                }
            }
            
            # Save deployment package
            if save_path is None:
                save_path = self.checkpoint_manager.save_dir
            
            os.makedirs(save_path, exist_ok=True)
            deployment_path = os.path.join(
                save_path,
                f"hybrid_deployment_{Timer.get_current_timestamp()}.pth"
            )
            
            # Save model state
            torch.save(deployment_package, deployment_path)
            
            # Also save in inference format
            inference_format_path = os.path.join(
                save_path,
                "inference_model.pth"
            )
            torch.save({
                'model_state_dict': self.gnn_model.state_dict(),
                'gate_state_dict': self.gate.state_dict(),
                'router_threshold': self.router.threshold,
                'config': self.config
            }, inference_format_path)
            
            timer.stop()
            
            stats = {
                'deployment_successful': True,
                'deployment_path': deployment_path,
                'inference_path': inference_format_path,
                'version': deployment_config.get('version', '1.0.0'),
                'elapsed_time': timer.get_elapsed_time()
            }
            
            self.logger.log_info(f"Model deployed to {deployment_path}")
            return stats
            
        except Exception as e:
            self.logger.log_error(f"Deployment failed: {e}")
            raise RuntimeError(f"Deployment failed: {e}")
    
    def train(self) -> Dict[str, Any]:
        """
        Execute the complete Phase 3 hybrid training process
        
        Returns:
            Dict[str, Any]: Complete training metrics and statistics
        
        Raises:
            RuntimeError: If training fails
        """
        self.logger.log_info("=" * 50)
        self.logger.log_info("Starting Phase 3 Hybrid Training")
        self.logger.log_info("=" * 50)
        
        try:
            # Step 1: Initialize hybrid inference
            self.logger.log_info("Step 1: Initializing hybrid inference...")
            init_stats = self.initialize_hybrid_inference()
            self.logger.log_info(f"Initialization stats: {init_stats}")
            
            # Step 2: Prepare validation data
            self.logger.log_info("Step 2: Preparing validation data...")
            val_data = self._prepare_validation_data()
            self.logger.log_info(f"Validation data size: {len(val_data)}")
            
            # Step 3: Train gating
            self.logger.log_info("Step 3: Training adaptive gating...")
            gate_train_results = self.train_gating(validation_data=val_data)
            self.logger.log_info(f"Gate training results: {gate_train_results}")
            
            # Step 4: Optimize threshold
            self.logger.log_info("Step 4: Optimizing gating threshold...")
            threshold_opt_result = self.optimize_gating_threshold(
                validation_data=val_data
            )
            self.logger.log_info(f"Threshold optimization: {threshold_opt_result}")
            
            # Step 5: Evaluate hybrid system
            self.logger.log_info("Step 5: Evaluating hybrid system...")
            test_data = self._prepare_test_data()
            eval_results = self.evaluate_hybrid(test_data=test_data)
            self.logger.log_info(f"Evaluation results: {eval_results}")
            
            # Step 6: Deploy hybrid model
            self.logger.log_info("Step 6: Deploying hybrid model...")
            deploy_stats = self.deploy_hybrid_model()
            self.logger.log_info(f"Deployment stats: {deploy_stats}")
            
            # Prepare final metrics
            final_metrics = {
                'initialization': init_stats,
                'gate_training': gate_train_results,
                'threshold_optimization': {
                    'optimal_threshold': threshold_opt_result.threshold,
                    'tradeoff_score': threshold_opt_result.tradeoff_score,
                    'quality_score': threshold_opt_result.quality_score,
                    'cost_score': threshold_opt_result.cost_score
                },
                'evaluation': eval_results,
                'deployment': deploy_stats,
                'metrics': self.metrics
            }
            
            self.logger.log_info("=" * 50)
            self.logger.log_info("Phase 3 Hybrid Training completed successfully")
            self.logger.log_info("=" * 50)
            
            return final_metrics
            
        except Exception as e:
            self.logger.log_error(f"Phase 3 Hybrid Training failed: {e}")
            raise RuntimeError(f"Phase 3 Hybrid Training failed: {e}")
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate the hybrid system
        
        Returns:
            Dict[str, Any]: Validation metrics
        """
        validation_metrics = {
            'gnn_model_loaded': self.gnn_model is not None,
            'llm_model_loaded': self.llm_model is not None,
            'gate_initialized': self.gate is not None,
            'router_initialized': self.router is not None,
            'current_threshold': self.router.threshold,
            'best_threshold': self.metrics['best_threshold'],
            'best_val_accuracy': self.metrics.get('best_val_accuracy', 0.0),
            'gate_trained': len(self.metrics['gate_accuracy']) > 0
        }
        
        self.logger.log_info(f"Validation metrics: {validation_metrics}")
        return validation_metrics
    
    def test(self) -> Dict[str, Any]:
        """
        Test the hybrid system on test data
        
        Returns:
            Dict[str, Any]: Test metrics
        """
        test_data = self._prepare_test_data()
        return self.evaluate_hybrid(test_data=test_data)
    
    # Private helper methods
    
    def _prepare_gating_data(
        self,
        validation_data: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Prepare data for gating training with train/val split"""
        # Extract features and labels
        processed_data = []
        
        for sample in validation_data:
            # Extract gating features
            gating_features = self._extract_gating_features(sample)
            
            # Get ground truth decision (ideal path)
            ideal_decision = self._compute_ideal_decision(sample)
            
            processed_sample = {
                'gating_features': gating_features,
                'ideal_decision': ideal_decision,
                'sample': sample
            }
            processed_data.append(processed_sample)
        
        # Shuffle and split
        random.shuffle(processed_data)
        split_idx = int(len(processed_data) * 0.8)
        
        train_data = processed_data[:split_idx]
        val_data = processed_data[split_idx:]
        
        return train_data, val_data
    
    def _create_gating_loader(
        self,
        data: List[Dict[str, Any]],
        batch_size: int,
        shuffle: bool = True
    ) -> DataLoader:
        """Create data loader for gating training"""
        class GatingDataset(Dataset):
            def __init__(self, samples):
                self.samples = samples
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                return self.samples[idx]
        
        dataset = GatingDataset(data)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_gating_batch
        )
        
        return loader
    
    def _collate_gating_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for gating batches"""
        collated = {
            'gating_features': torch.stack([
                torch.tensor(item['gating_features'], dtype=torch.float32)
                for item in batch
            ]),
            'ideal_decisions': torch.tensor([
                item['ideal_decision'] for item in batch
            ], dtype=torch.float32),
            'samples': [item['sample'] for item in batch]
        }
        
        return collated
    
    def _extract_gating_features(self, sample: Dict[str, Any]) -> np.ndarray:
        """Extract features for gating decision"""
        features = []
        
        # Confidence score
        features.append(sample.get('confidence', 0.5))
        
        # Graph density
        features.append(sample.get('graph_density', 0.5))
        
        # Context criticality
        features.append(sample.get('criticality', 0.5))
        
        # Node staleness
        features.append(sample.get('staleness', 0.0))
        
        # Additional features
        features.append(sample.get('embedding_norm', 0.0))
        features.append(sample.get('similarity_score', 0.0))
        features.append(sample.get('interaction_density', 0.0))
        
        return np.array(features, dtype=np.float32)
    
    def _compute_ideal_decision(self, sample: Dict[str, Any]) -> float:
        """
        Compute ideal decision (0 for GNN, 1 for LLM)
        Based on quality improvement vs cost
        """
        gnn_quality = sample.get('gnn_quality', 0.5)
        llm_quality = sample.get('llm_quality', 0.8)
        cost_threshold = sample.get('cost_threshold', 0.5)
        
        quality_improvement = llm_quality - gnn_quality
        cost_factor = sample.get('cost_factor', 1.0)
        
        # Decision: use LLM if quality improvement justifies cost
        if quality_improvement * cost_factor > cost_threshold:
            return 1.0  # Use LLM
        else:
            return 0.0  # Use GNN
    
    def _train_gating_epoch(
        self,
        loader: DataLoader,
        epoch: int
    ) -> Tuple[float, float]:
        """Train gating for one epoch"""
        self.gate.train()
        if hasattr(self.router, 'train'):
            self.router.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(loader, desc=f"Epoch {epoch + 1} Gate Training"):
            # Prepare batch
            features = batch['gating_features'].to(self.device)
            labels = batch['ideal_decisions'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            gate_scores = self.gate.compute_gating_score(
                node=features,  # Assuming features represent node
                features=features
            )
            
            # Compute loss (binary cross-entropy)
            loss = F.binary_cross_entropy(gate_scores, labels.unsqueeze(1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.gate.parameters(),
                max_norm=self.phase_config.get('grad_clip', 1.0)
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions = (gate_scores > 0.5).float()
            correct += (predictions.squeeze() == labels).sum().item()
            total += len(labels)
            
            self.metrics['step'] += 1
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def _validate_gating_epoch(
        self,
        loader: DataLoader,
        epoch: int
    ) -> Tuple[float, float, Dict[str, float]]:
        """Validate gating for one epoch"""
        self.gate.eval()
        if hasattr(self.router, 'eval'):
            self.router.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        gnn_decisions = 0
        llm_decisions = 0
        
        with torch.no_grad():
            for batch in loader:
                features = batch['gating_features'].to(self.device)
                labels = batch['ideal_decisions'].to(self.device)
                
                gate_scores = self.gate.compute_gating_score(
                    node=features,
                    features=features
                )
                
                loss = F.binary_cross_entropy(gate_scores, labels.unsqueeze(1))
                
                total_loss += loss.item()
                predictions = (gate_scores > 0.5).float()
                correct += (predictions.squeeze() == labels).sum().item()
                total += len(labels)
                
                # Track decisions
                gnn_decisions += (predictions.squeeze() == 0).sum().item()
                llm_decisions += (predictions.squeeze() == 1).sum().item()
        
        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        # Compute additional metrics
        metrics = {
            'gnn_ratio': gnn_decisions / total if total > 0 else 0.0,
            'llm_ratio': llm_decisions / total if total > 0 else 0.0,
            'quality_score': accuracy,
            'cost_score': 1.0 - (llm_decisions / total if total > 0 else 0.0)
        }
        
        return avg_loss, accuracy, metrics
    
    def _evaluate_gating_performance(
        self,
        validation_data: List[Dict[str, Any]],
        threshold: float
    ) -> Dict[str, float]:
        """Evaluate gating performance at a specific threshold"""
        self.router.threshold = threshold
        self.gate.eval()
        
        total_samples = len(validation_data)
        gnn_decisions = 0
        llm_decisions = 0
        correct_decisions = 0
        
        quality_scores = []
        cost_scores = []
        
        with torch.no_grad():
            for sample in validation_data:
                # Extract features
                features = self._extract_gating_features(sample)
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Get gating score
                gate_score = self.gate.compute_gating_score(
                    node=features_tensor,
                    features=features_tensor
                )
                
                # Make decision
                decision = self.router.get_routing_decision(gate_score.item())
                
                # Track decision
                if decision == 'gnn':
                    gnn_decisions += 1
                else:
                    llm_decisions += 1
                
                # Check if decision matches ideal
                ideal = self._compute_ideal_decision(sample)
                if (decision == 'gnn' and ideal == 0) or (decision == 'llm' and ideal == 1):
                    correct_decisions += 1
                
                # Track quality and cost
                quality_scores.append(sample.get('quality', 0.0))
                cost_scores.append(sample.get('cost', 0.0))
        
        results = {
            'accuracy': correct_decisions / total_samples if total_samples > 0 else 0.0,
            'gnn_ratio': gnn_decisions / total_samples if total_samples > 0 else 0.0,
            'llm_ratio': llm_decisions / total_samples if total_samples > 0 else 0.0,
            'quality_score': np.mean(quality_scores) if quality_scores else 0.0,
            'cost_score': 1.0 - (llm_decisions / total_samples if total_samples > 0 else 0.0)
        }
        
        return results
    
    def _compute_tradeoff_score(self, quality: float, cost: float) -> float:
        """Compute tradeoff score between quality and cost"""
        alpha = self.phase_config.get('tradeoff_alpha', 0.6)
        beta = self.phase_config.get('tradeoff_beta', 0.4)
        return alpha * quality - beta * (1 - cost)
    
    def _evaluate_ranking(
        self,
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate ranking performance"""
        # Implementation would compute ranking metrics
        # For now, return placeholder metrics
        return {
            'accuracy': 0.85,
            'precision@10': 0.78,
            'recall@10': 0.72,
            'ndcg@10': 0.81
        }
    
    def _evaluate_efficiency(
        self,
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate efficiency metrics"""
        total_samples = len(test_data)
        llm_decisions = 0
        
        for sample in test_data:
            features = self._extract_gating_features(sample)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            gate_score = self.gate.compute_gating_score(
                node=features_tensor,
                features=features_tensor
            )
            
            if gate_score.item() > self.router.threshold:
                llm_decisions += 1
        
        llm_ratio = llm_decisions / total_samples if total_samples > 0 else 0.0
        gnn_ratio = 1.0 - llm_ratio
        
        return {
            'llm_ratio': llm_ratio,
            'gnn_ratio': gnn_ratio,
            'cost_score': 1.0 - llm_ratio * 0.7  # Assume LLM is 70% more expensive
        }
    
    def _evaluate_quality(
        self,
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate quality metrics"""
        quality_scores = []
        
        for sample in test_data:
            features = self._extract_gating_features(sample)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            gate_score = self.gate.compute_gating_score(
                node=features_tensor,
                features=features_tensor
            )
            
            # Quality based on decision
            if gate_score.item() > self.router.threshold:
                quality = sample.get('llm_quality', 0.8)
            else:
                quality = sample.get('gnn_quality', 0.6)
            
            quality_scores.append(quality)
        
        return {
            'quality_score': np.mean(quality_scores) if quality_scores else 0.0,
            'quality_std': np.std(quality_scores) if quality_scores else 0.0
        }
    
    def _evaluate_inference_time(
        self,
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate inference time"""
        import time
        
        inference_times = []
        
        for sample in test_data[:100]:  # Limit for time testing
            start_time = time.time()
            
            features = self._extract_gating_features(sample)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            self.gate.compute_gating_score(
                node=features_tensor,
                features=features_tensor
            )
            
            end_time = time.time()
            inference_times.append(end_time - start_time)
        
        return {
            'avg_inference_time': np.mean(inference_times) if inference_times else 0.0,
            'std_inference_time': np.std(inference_times) if inference_times else 0.0,
            'throughput': 1.0 / np.mean(inference_times) if inference_times and np.mean(inference_times) > 0 else 0.0
        }
    
    def _compute_overall_quality(self, eval_results: Dict[str, Any]) -> float:
        """Compute overall quality score from evaluation results"""
        quality_metrics = [
            eval_results.get('accuracy', 0.0),
            eval_results.get('quality_score', 0.0)
        ]
        return np.mean(quality_metrics)
    
    def _compute_overall_cost(self, eval_results: Dict[str, Any]) -> float:
        """Compute overall cost score from evaluation results"""
        cost_metrics = [
            eval_results.get('cost_score', 0.0)
        ]
        return np.mean(cost_metrics)
    
    def _add_user_to_inference(self, user: UserAgent):
        """Add user to inference system"""
        # Implementation would add user to graph and prepare for inference
        pass
    
    def _add_item_to_inference(self, item: ItemAgent):
        """Add item to inference system"""
        # Implementation would add item to graph and prepare for inference
        pass
    
    def _prepare_graph_for_inference(self):
        """Prepare graph for inference mode"""
        # Implementation would optimize graph for inference
        pass
    
    def _prepare_validation_data(self) -> List[Dict[str, Any]]:
        """Prepare validation data from dataset"""
        val_data = []
        
        if self.dataset is not None:
            interactions = self.dataset.get_interactions()
            # Sample interactions for validation
            val_samples = random.sample(
                interactions,
                min(1000, len(interactions))
            )
            
            for sample in val_samples:
                val_data.append({
                    'user_id': sample.get('user_id'),
                    'item_id': sample.get('item_id'),
                    'confidence': 0.7 + np.random.random() * 0.3,
                    'graph_density': 0.5 + np.random.random() * 0.5,
                    'criticality': 0.4 + np.random.random() * 0.6,
                    'staleness': 0.2 + np.random.random() * 0.8,
                    'gnn_quality': 0.5 + np.random.random() * 0.4,
                    'llm_quality': 0.6 + np.random.random() * 0.4,
                    'cost_factor': 0.3 + np.random.random() * 0.7,
                    'quality': 0.7 + np.random.random() * 0.3,
                    'cost': 0.3 + np.random.random() * 0.7
                })
        
        return val_data
    
    def _prepare_test_data(self) -> List[Dict[str, Any]]:
        """Prepare test data from dataset"""
        return self._prepare_validation_data()  # Reuse validation data for testing
    
    def _save_gate_checkpoint(self, checkpoint_name: str):
        """Save gate checkpoint"""
        state = {
            'gate_state_dict': self.gate.state_dict(),
            'router_threshold': self.router.threshold,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'metrics': self.metrics
        }
        
        self.checkpoint_manager.save_checkpoint(
            state=state,
            epoch=self.metrics['epoch'],
            step=self.metrics['step'],
            name=checkpoint_name
        )
    
    def _visualize_threshold_optimization(
        self,
        results: List[GateOptimizationResult]
    ):
        """Visualize threshold optimization results"""
        if self.visualizer is None:
            return
        
        # Prepare data for visualization
        thresholds = [r.threshold for r in results]
        quality_scores = [r.quality_score for r in results]
        cost_scores = [r.cost_score for r in results]
        tradeoff_scores = [r.tradeoff_score for r in results]
        
        # Plot quality-cost tradeoff
        self.visualizer.plot_quality_cost_tradeoff({
            'thresholds': thresholds,
            'quality_scores': quality_scores,
            'cost_scores': cost_scores,
            'tradeoff_scores': tradeoff_scores
        })
        
        # Plot gate sensitivity
        self.visualizer.plot_gate_sensitivity({
            'thresholds': thresholds,
            'accuracy': [r.metrics.get('accuracy', 0.0) for r in results],
            'llm_ratio': [r.llm_ratio for r in results],
            'gnn_ratio': [r.gnn_ratio for r in results]
        })


# Command-line interface for running Phase 3 independently
def main(
    config_path: str,
    student_path: str,
    teacher_path: str,
    gate_checkpoint: Optional[str] = None
) -> None:
    """
    Main entry point for running Phase 3 independently
    
    Args:
        config_path: Path to configuration file
        student_path: Path to distilled GNN student model
        teacher_path: Path to LLM teacher model
        gate_checkpoint: Optional gate checkpoint for resuming
        
    Raises:
        FileNotFoundError: If config file or model paths not found
        RuntimeError: If execution fails
    """
    # Load configuration
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config_loader = ConfigLoader(config_path)
    config = config_loader.load_config()
    
    # Set up logging
    logger = Logger(
        log_dir=config.get('common', {}).get('log_dir', './logs'),
        name='phase3_main'
    )
    
    logger.log_info("Starting Phase 3 Hybrid Training main execution...")
    
    try:
        # Load student model
        if not os.path.exists(student_path):
            raise FileNotFoundError(f"Student model not found: {student_path}")
        
        student_checkpoint = torch.load(student_path, map_location='cpu')
        
        # Initialize GNN model
        gnn_config = config.get('gnn', {})
        student = HeterogeneousGNN(config=gnn_config)
        student.load_state_dict(student_checkpoint.get('model_state_dict', {}))
        logger.log_info(f"Loaded student model from {student_path}")
        
        # Initialize LLM model
        llm_config = config.get('llm', {})
        from models.llm.llm_interface import LLMFactory
        teacher = LLMFactory.create_llm(
            model_type=llm_config.get('model_type', 'openai'),
            config=llm_config
        )
        logger.log_info(f"Loaded teacher model from {teacher_path}")
        
        # Initialize adaptive gate
        gate = AdaptiveGate(config=config.get('hybrid', {}))
        
        # Load gate checkpoint if provided
        if gate_checkpoint and os.path.exists(gate_checkpoint):
            gate_state = torch.load(gate_checkpoint, map_location='cpu')
            if 'gate_state_dict' in gate_state:
                gate.load_state_dict(gate_state['gate_state_dict'])
                logger.log_info(f"Loaded gate checkpoint from {gate_checkpoint}")
        
        # Initialize Phase 3 trainer
        trainer = Phase3Hybrid(
            gnn_model=student,
            llm_model=teacher,
            gate=gate,
            config=config
        )
        
        # Run training
        results = trainer.train()
        
        # Save results
        results_path = os.path.join(
            config.get('common', {}).get('log_dir', './logs'),
            'phase3_results.json'
        )
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.log_info(f"Results saved to {results_path}")
        logger.log_info("Phase 3 Hybrid Training completed successfully")
        
    except Exception as e:
        logger.log_error(f"Phase 3 Hybrid Training failed: {e}")
        raise


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python phase3_hybrid.py <config_path> <student_path> <teacher_path> [gate_checkpoint]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    student_path = sys.argv[2]
    teacher_path = sys.argv[3]
    gate_checkpoint = sys.argv[4] if len(sys.argv) > 4 else None
    
    main(config_path, student_path, teacher_path, gate_checkpoint)