"""
scripts/train_phase3.py

Phase 3 Training Script for H-GRAGrecsys - Hybrid Inference with Adaptive Gating

This script implements the third phase of the H-GRAGrecsys training pipeline:
1. Load Phase 2 distilled GNN model and Phase 1 teacher LLM
2. Initialize adaptive gating mechanism
3. Train the hybrid inference engine
4. Optimize gating thresholds
5. Balance quality and efficiency
6. Deploy the final hybrid model

Features:
- Adaptive gating training
- Threshold optimization
- Quality-efficiency tradeoff
- Hybrid inference
- Checkpoint saving and loading
- Progress logging and visualization
- GPU support
- Deployment-ready model export
"""

import os
import sys
import json
import yaml
import argparse
import pickle
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple
from datetime import datetime
import traceback
import shutil
import numpy as np

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.logger import get_logger
from utils.config_loader import ConfigLoader, load_config
from utils.seed_manager import create_seed_manager
from utils.timer import Timer, global_timer
from utils.visualizer import create_visualizer

# Import training module
from training.phase3_hybrid import Phase3Hybrid

# Import hybrid modules
from models.hybrid.adaptive_gate import AdaptiveGate, GatingFeatures
from models.hybrid.router import Router
from models.hybrid.inference_engine import HybridInferenceEngine

# Import model components
from models.gnn.gnn_encoder import GNNEncoder
from models.gnn.heterogeneous_gnn import HeterogeneousGNN
from models.llm.llm_interface import LLMInterface
from models.graph.graph_builder import GraphBuilder
from models.graph.heterogeneous_graph import HeterogeneousGraph

# Import data modules
from data.amazon_dataset import AmazonDataset
from data.data_loader import DataLoader

# Import evaluation
from evaluation.evaluator import Evaluator
from evaluation.efficiency_evaluator import EfficiencyEvaluator

# Try to import torch for GPU support
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader as TorchDataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class Phase3Trainer:
    """
    Phase 3 Trainer for H-GRAGrecsys - Hybrid Inference with Adaptive Gating.
    
    Features:
    - Adaptive gating training
    - Threshold optimization
    - Quality-efficiency tradeoff
    - Hybrid inference
    - Checkpoint saving and loading
    - Progress monitoring and visualization
    - Deployment-ready model export
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        gnn_path: Optional[Union[str, Path]] = None,
        llm_path: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        resume_from: Optional[Union[str, Path]] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        logger: Optional['Logger'] = None,
        verbose: bool = True
    ):
        """
        Initialize the Phase 3 Trainer.
        
        Args:
            config_path (str, Path, optional): Path to configuration file
            gnn_path (str, Path, optional): Path to Phase 2 GNN model
            llm_path (str, Path, optional): Path to Phase 1 LLM model
            checkpoint_dir (str, Path, optional): Directory to save checkpoints
            output_dir (str, Path, optional): Directory for outputs
            resume_from (str, Path, optional): Checkpoint to resume from
            seed (int, optional): Random seed for reproducibility
            device (str, optional): Device to use ('cpu', 'cuda')
            logger (Logger, optional): Logger instance
            verbose (bool): Whether to enable verbose output
        
        Example:
            trainer = Phase3Trainer(
                config_path='config/default_config.yaml',
                gnn_path='experiments/phase2/checkpoints/distilled_model.pt',
                llm_path='experiments/phase1/checkpoints/phase1_best.pt',
                output_dir='experiments/phase3'
            )
            trainer.train()
        """
        # Setup paths
        self.config_path = Path(config_path) if config_path else None
        self.gnn_path = Path(gnn_path) if gnn_path else None
        self.llm_path = Path(llm_path) if llm_path else None
        self.output_dir = Path(output_dir) if output_dir else Path("experiments/phase3")
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        if logger is None:
            self.logger = get_logger(
                log_dir=self.log_dir,
                name="phase3_trainer",
                verbose=verbose
            )
        else:
            self.logger = logger
        
        # Load configuration
        if self.config_path and self.config_path.exists():
            self.config_loader = ConfigLoader(
                config_path=self.config_path,
                logger=self.logger
            )
            self.config = self.config_loader.config
        else:
            self.config_loader = ConfigLoader(load_defaults=True)
            self.config = self.config_loader.config
        
        # Save config to output directory
        self.config_loader.save_config(path=self.output_dir / "config.yaml")
        
        # Setup seed manager
        self.seed = seed or self.config.get('seed', 42)
        self.seed_manager = create_seed_manager(
            seed=self.seed,
            config_path=self.config_path
        )
        self.seed_manager.set_all_seeds()
        
        # Setup timer
        self.timer = Timer(
            name="phase3_training",
            logger=self.logger,
            track_memory=True,
            track_gpu=True,
            save_report=True,
            report_dir=self.output_dir / "timing"
        )
        
        # Setup visualizer
        self.visualizer = create_visualizer(
            config_path=self.config_path,
            output_dir=self.output_dir / "plots",
            interactive=False
        )
        
        # Set device
        self.device = device or self._get_default_device()
        self.logger.log_info(f"Using device: {self.device}")
        
        # Resume from checkpoint
        self.resume_from = Path(resume_from) if resume_from else None
        self.start_epoch = 0
        
        # Initialize training components
        self.phase3 = None
        self.dataset = None
        self.data_loader = None
        self.gnn_model = None
        self.llm_model = None
        self.gate = None
        self.router = None
        self.hybrid_engine = None
        
        # Training state
        self.state = {
            'epoch': 0,
            'best_metrics': {},
            'training_completed': False,
            'gate_stats': {},
            'threshold_history': []
        }
        
        self.logger.log_info("Phase3Trainer initialized")
        self.logger.log_info(f"Output directory: {self.output_dir}")
        self.logger.log_info(f"Checkpoint directory: {self.checkpoint_dir}")
        self.logger.log_info(f"GNN path: {self.gnn_path}")
        self.logger.log_info(f"LLM path: {self.llm_path}")
    
    def _get_default_device(self) -> str:
        """
        Get the default device (GPU if available, else CPU).
        
        Returns:
            str: Device name
        """
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def train(self) -> Dict[str, Any]:
        """
        Run Phase 3 hybrid training.
        
        Returns:
            Dict[str, Any]: Training results and metrics
        
        Example:
            results = trainer.train()
            print(f"Hybrid training completed: {results['epochs_completed']} epochs")
        """
        self.logger.log_info("=" * 80)
        self.logger.log_info("PHASE 3: Hybrid Inference with Adaptive Gating")
        self.logger.log_info("=" * 80)
        
        with self.timer.measure("phase3_training"):
            # Step 1: Load data
            self._load_data()
            
            # Step 2: Load GNN model (Phase 2)
            self._load_gnn_model()
            
            # Step 3: Load LLM model (Phase 1)
            self._load_llm_model()
            
            # Step 4: Initialize adaptive gate
            self._initialize_gate()
            
            # Step 5: Initialize hybrid engine
            self._initialize_hybrid_engine()
            
            # Step 6: Initialize Phase 3
            self._initialize_phase3()
            
            # Step 7: Resume if requested
            if self.resume_from:
                self._resume_training()
            
            # Step 8: Run training
            results = self._run_training()
            
            # Step 9: Optimize gating threshold
            results['threshold_optimization'] = self._optimize_gating_threshold()
            
            # Step 10: Deploy hybrid model
            self._deploy_hybrid_model()
            
            # Step 11: Save final model
            self._save_final_model()
            
            # Step 12: Generate summary
            self._generate_summary(results)
        
        self.logger.log_info("=" * 80)
        self.logger.log_info("Phase 3 Hybrid Training Completed")
        self.logger.log_info("=" * 80)
        
        return results
    
    def _load_data(self) -> None:
        """
        Load and prepare dataset.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("LOADING DATA")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("load_data"):
            # Get dataset name from config
            dataset_name = self.config.get('data', {}).get('dataset_name', 'Amazon_Books')
            
            # Load dataset
            self.dataset = AmazonDataset(dataset_name, self.config)
            self.dataset.load_data()
            
            # Get statistics
            stats = self.dataset.get_statistics()
            self.logger.log_info(f"Dataset: {dataset_name}")
            self.logger.log_info(f"Users: {stats.get('num_users', 0):,}")
            self.logger.log_info(f"Items: {stats.get('num_items', 0):,}")
            self.logger.log_info(f"Interactions: {stats.get('num_interactions', 0):,}")
            
            # Create data loader
            batch_size = self.config.get('training', {}).get('phase3', {}).get('batch_size', 32)
            self.data_loader = DataLoader(
                dataset=self.dataset,
                batch_size=batch_size,
                shuffle=True
            )
            
            self.logger.log_info(f"Batch size: {batch_size}")
            self.logger.log_info(f"Number of batches: {len(self.data_loader.get_train_batches())}")
    
    def _load_gnn_model(self) -> None:
        """
        Load GNN model from Phase 2 checkpoint.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("LOADING GNN MODEL")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("load_gnn"):
            # Initialize GNN encoder
            self.gnn_model = GNNEncoder(self.config)
            
            if self.gnn_path and self.gnn_path.exists():
                try:
                    # Load checkpoint
                    with open(self.gnn_path, 'rb') as f:
                        checkpoint = pickle.load(f)
                    
                    # Load model state
                    if TORCH_AVAILABLE and 'model_state_dict' in checkpoint:
                        self.gnn_model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'student_model' in checkpoint:
                        self.gnn_model = checkpoint['student_model']
                    
                    self.logger.log_info(f"GNN model loaded from: {self.gnn_path}")
                    
                except Exception as e:
                    self.logger.log_error(f"Failed to load GNN model: {e}")
                    self.logger.log_info("Using newly initialized GNN model")
            else:
                self.logger.log_warning("GNN model path not found. Using newly initialized model.")
            
            self.logger.log_info(f"GNN model initialized")
    
    def _load_llm_model(self) -> None:
        """
        Load LLM model from Phase 1 checkpoint.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("LOADING LLM MODEL")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("load_llm"):
            # Initialize LLM interface
            model_name = self.config.get('llm', {}).get('model_name', 'gpt-3.5-turbo')
            self.llm_model = LLMInterface(
                model_name=model_name,
                config=self.config
            )
            
            if self.llm_path and self.llm_path.exists():
                try:
                    # Load checkpoint
                    with open(self.llm_path, 'rb') as f:
                        checkpoint = pickle.load(f)
                    
                    # Load agent states if available
                    if 'state' in checkpoint:
                        self.llm_agents = checkpoint['state'].get('user_agents', {})
                        self.llm_item_agents = checkpoint['state'].get('item_agents', {})
                    
                    self.logger.log_info(f"LLM model loaded from: {self.llm_path}")
                    
                except Exception as e:
                    self.logger.log_error(f"Failed to load LLM model: {e}")
                    self.logger.log_info("Using newly initialized LLM model")
            else:
                self.logger.log_warning("LLM model path not found. Using newly initialized model.")
            
            self.logger.log_info(f"LLM model initialized: {model_name}")
    
    def _initialize_gate(self) -> None:
        """
        Initialize adaptive gate.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("INITIALIZING ADAPTIVE GATE")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("initialize_gate"):
            # Get gate config
            gate_config = self.config.get('model', {}).get('hybrid', {})
            threshold = gate_config.get('gate_threshold', 0.3)
            staleness_lambda = gate_config.get('staleness_lambda', 0.1)
            
            # Initialize gate
            self.gate = AdaptiveGate(self.config)
            self.gate.threshold = threshold
            
            # Initialize gating features
            self.gating_features = GatingFeatures(
                graph=self._get_graph(),
                config=self.config
            )
            
            # Initialize router
            self.router = Router(
                gate=self.gate,
                threshold=threshold
            )
            
            self.logger.log_info(f"Gate initialized with threshold: {threshold}")
            self.logger.log_info(f"Staleness lambda: {staleness_lambda}")
    
    def _get_graph(self) -> HeterogeneousGraph:
        """
        Get graph for gating features.
        
        Returns:
            HeterogeneousGraph: Graph
        """
        # Build graph from dataset
        graph_builder = GraphBuilder(self.config)
        
        # Create agents
        agents = []
        users = list(self.dataset.get_user_items().keys())[:100]
        items = list(self.dataset.get_item_features().keys())[:100]
        
        for user_id in users:
            agents.append(UserAgent(user_id, self.config))
        
        for item_id in items:
            agents.append(ItemAgent(item_id, self.config))
        
        # Build graph
        graph = graph_builder.build_graph(
            agents=agents,
            interactions=self.dataset.get_interactions()
        )
        
        return graph
    
    def _initialize_hybrid_engine(self) -> None:
        """
        Initialize hybrid inference engine.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("INITIALIZING HYBRID ENGINE")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("initialize_hybrid"):
            # Create hybrid engine
            self.hybrid_engine = HybridInferenceEngine(
                gnn_encoder=self.gnn_model,
                llm_interface=self.llm_model,
                gate=self.gate,
                config=self.config
            )
            
            self.logger.log_info("Hybrid engine initialized")
    
    def _initialize_phase3(self) -> None:
        """
        Initialize Phase 3 training module.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("INITIALIZING PHASE 3 TRAINING")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("initialize_phase3"):
            self.phase3 = Phase3Hybrid(
                gnn_model=self.gnn_model,
                llm_model=self.llm_model,
                gate=self.gate,
                config=self.config,
                logger=self.logger,
                seed_manager=self.seed_manager
            )
            
            self.logger.log_info("Phase 3 training module initialized")
    
    def _run_training(self) -> Dict[str, Any]:
        """
        Run the Phase 3 hybrid training loop.
        
        Returns:
            Dict[str, Any]: Training results
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("RUNNING HYBRID TRAINING")
        self.logger.log_info("-" * 50)
        
        # Get training parameters
        train_config = self.config.get('training', {}).get('phase3', {})
        num_epochs = train_config.get('num_epochs', 20)
        learning_rate = train_config.get('learning_rate', 5e-5)
        
        self.logger.log_info(f"Number of epochs: {num_epochs}")
        self.logger.log_info(f"Learning rate: {learning_rate}")
        
        # Training loop
        training_results = {
            'epochs_completed': 0,
            'metrics_history': [],
            'gate_history': [],
            'loss_history': [],
            'best_metrics': {},
            'threshold_history': []
        }
        
        for epoch in range(self.start_epoch, num_epochs):
            self.logger.log_info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            with self.timer.measure(f"epoch_{epoch}"):
                # Train one epoch
                epoch_loss, epoch_metrics, gate_stats = self._train_epoch(epoch)
                
                # Store metrics
                training_results['metrics_history'].append(epoch_metrics)
                training_results['gate_history'].append(gate_stats)
                training_results['loss_history'].append(epoch_loss)
                training_results['epochs_completed'] = epoch + 1
                
                # Update best metrics
                if not training_results['best_metrics'] or epoch_metrics.get('ndcg@10', 0) > training_results['best_metrics'].get('ndcg@10', 0):
                    training_results['best_metrics'] = epoch_metrics
                    self._save_checkpoint(epoch, epoch_metrics, epoch_loss, is_best=True)
                
                # Save checkpoint
                if (epoch + 1) % train_config.get('save_interval', 5) == 0:
                    self._save_checkpoint(epoch, epoch_metrics, epoch_loss, is_best=False)
                
                # Log progress
                self._log_epoch_metrics(epoch, epoch_loss, epoch_metrics, gate_stats)
                
                # Early stopping
                if self._check_early_stopping(training_results['metrics_history']):
                    self.logger.log_info("Early stopping triggered")
                    break
        
        self.logger.log_info(f"\nHybrid training completed: {training_results['epochs_completed']} epochs")
        self.logger.log_info(f"Best NDCG@10: {training_results['best_metrics'].get('ndcg@10', 0):.4f}")
        
        return training_results
    
    def _train_epoch(self, epoch: int) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        """
        Train for one epoch.
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            Tuple[float, Dict[str, float], Dict[str, Any]]: (Loss, Metrics, Gate Stats)
        """
        # Get training batches
        batches = self.data_loader.get_train_batches()
        
        total_loss = 0.0
        batch_count = 0
        batch_losses = []
        gate_decisions = []
        llm_calls = 0
        total_calls = 0
        
        for batch_idx, batch in enumerate(batches):
            # Run hybrid training step
            loss, stats = self.phase3.train_step(batch)
            
            total_loss += loss
            batch_losses.append(loss)
            batch_count += 1
            
            # Track gate decisions
            if 'gate_decision' in stats:
                gate_decisions.append(stats['gate_decision'])
                if stats['gate_decision'] == 'llm':
                    llm_calls += 1
                total_calls += 1
            
            # Log batch progress
            if (batch_idx + 1) % 10 == 0:
                self.logger.log_debug(f"  Batch {batch_idx + 1}/{len(batches)}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        
        # Compute gate statistics
        gate_stats = {
            'llm_call_ratio': llm_calls / total_calls if total_calls > 0 else 0,
            'gate_threshold': self.gate.threshold,
            'decision_distribution': {
                'gnn': total_calls - llm_calls,
                'llm': llm_calls
            }
        }
        
        # Evaluate after epoch
        eval_metrics = self._evaluate()
        
        return avg_loss, eval_metrics, gate_stats
    
    def _evaluate(self) -> Dict[str, float]:
        """
        Evaluate current hybrid model.
        
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Create evaluator
        evaluator = Evaluator(
            model=self.hybrid_engine,
            dataset=self.dataset,
            config=self.config,
            logger=self.logger
        )
        
        # Run evaluation
        results = evaluator.evaluate()
        
        # Extract metrics
        metrics = {}
        if 'metrics' in results:
            for key, value in results['metrics'].items():
                if isinstance(value, (int, float)):
                    metrics[key] = value
        
        return metrics
    
    def _optimize_gating_threshold(self) -> Dict[str, Any]:
        """
        Optimize gating threshold for quality-efficiency tradeoff.
        
        Returns:
            Dict[str, Any]: Optimization results
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("OPTIMIZING GATING THRESHOLD")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("optimize_threshold"):
            # Get candidate thresholds
            thresholds = np.linspace(0.0, 1.0, 21)
            
            results = {
                'thresholds': thresholds.tolist(),
                'ndcg_scores': [],
                'llm_call_ratios': [],
                'costs': [],
                'optimal_threshold': 0.3
            }
            
            # Save original threshold
            original_threshold = self.gate.threshold
            
            # Test each threshold
            for threshold in tqdm(thresholds, desc="Testing thresholds", 
                                  disable=not TQDM_AVAILABLE):
                self.gate.threshold = threshold
                
                # Evaluate with this threshold
                metrics = self._evaluate()
                
                # Get gate stats
                gate_stats = self._get_gate_stats()
                
                results['ndcg_scores'].append(metrics.get('ndcg@10', 0))
                results['llm_call_ratios'].append(gate_stats.get('llm_call_ratio', 0))
                
                # Estimate cost
                cost = gate_stats.get('llm_call_ratio', 0) * 0.01  # Rough estimate
                results['costs'].append(cost)
            
            # Find optimal threshold (maximize quality - cost)
            scores = []
            for i in range(len(thresholds)):
                quality = results['ndcg_scores'][i]
                cost = results['costs'][i]
                score = quality - cost * 0.5  # Weighted tradeoff
                scores.append(score)
            
            optimal_idx = np.argmax(scores)
            results['optimal_threshold'] = thresholds[optimal_idx]
            
            # Restore optimal threshold
            self.gate.threshold = results['optimal_threshold']
            
            self.logger.log_info(f"Optimal threshold: {results['optimal_threshold']:.3f}")
            self.logger.log_info(f"Optimal NDCG@10: {results['ndcg_scores'][optimal_idx]:.4f}")
            self.logger.log_info(f"Optimal LLM call ratio: {results['llm_call_ratios'][optimal_idx]:.3f}")
            
            # Restore original threshold
            self.gate.threshold = original_threshold
            
            return results
    
    def _get_gate_stats(self) -> Dict[str, Any]:
        """
        Get gate statistics.
        
        Returns:
            Dict[str, Any]: Gate statistics
        """
        # Sample some data for statistics
        test_data = self.data_loader.get_test_batches()
        test_batch = next(iter(test_data)) if test_data else {}
        
        # Get gate decisions
        gate_decisions = []
        llm_calls = 0
        total_calls = 0
        
        # Simulate on test data
        for _ in range(min(100, len(test_data))):
            # Sample a batch
            batch = self.data_loader.get_batch()
            
            # Get gate decision
            for user, items in batch.items():
                decision = self.gate.decide_path(
                    node=user,
                    context={'batch': batch},
                    prediction=None
                )
                gate_decisions.append(decision)
                if decision.get('type') == 'llm':
                    llm_calls += 1
                total_calls += 1
        
        return {
            'llm_call_ratio': llm_calls / total_calls if total_calls > 0 else 0,
            'gate_threshold': self.gate.threshold,
            'total_decisions': total_calls,
            'llm_decisions': llm_calls,
            'gnn_decisions': total_calls - llm_calls
        }
    
    def _deploy_hybrid_model(self) -> None:
        """
        Deploy the hybrid model for inference.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("DEPLOYING HYBRID MODEL")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("deploy_model"):
            # Create deployment directory
            deploy_dir = self.output_dir / "deployment"
            deploy_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model for deployment
            deploy_path = deploy_dir / "hybrid_model.pt"
            with open(deploy_path, 'wb') as f:
                pickle.dump({
                    'hybrid_engine': self.hybrid_engine,
                    'gnn_model': self.gnn_model,
                    'gate': self.gate,
                    'config': self.config,
                    'threshold': self.gate.threshold,
                    'timestamp': datetime.now().isoformat()
                }, f)
            
            # Save in PyTorch format if available
            if TORCH_AVAILABLE:
                torch_path = deploy_dir / "hybrid_model.pth"
                torch.save({
                    'gnn_state_dict': self.gnn_model.state_dict() if hasattr(self.gnn_model, 'state_dict') else None,
                    'gate_threshold': self.gate.threshold,
                    'config': self.config
                }, torch_path)
            
            # Save configuration
            deploy_config_path = deploy_dir / "deploy_config.yaml"
            with open(deploy_config_path, 'w') as f:
                yaml.dump({
                    'model_type': 'hybrid',
                    'threshold': self.gate.threshold,
                    'gnn_model': str(self.gnn_path) if self.gnn_path else 'default',
                    'llm_model': str(self.llm_path) if self.llm_path else 'default',
                    'config': self.config
                }, f, default_flow_style=False)
            
            # Create deployment script
            deploy_script_path = deploy_dir / "deploy.py"
            with open(deploy_script_path, 'w') as f:
                f.write("""
import pickle
import sys
from pathlib import Path

# Load model
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model['hybrid_engine']

# Inference function
def infer(model, user_id, item_ids, context=None):
    return model.infer(user=user_id, candidates=item_ids, context=context)

if __name__ == "__main__":
    model_path = Path(__file__).parent / "hybrid_model.pt"
    model = load_model(model_path)
    print(f"Model loaded from: {model_path}")
    print("Deployment ready!")
""")
            
            self.logger.log_info(f"Deployment package saved to: {deploy_dir}")
            
            # Log deployment info
            self.logger.log_info(f"Model path: {deploy_path}")
            self.logger.log_info(f"Config path: {deploy_config_path}")
            self.logger.log_info(f"Deployment script: {deploy_script_path}")
    
    def _check_early_stopping(self, history: List[Dict[str, float]]) -> bool:
        """
        Check if early stopping criteria are met.
        
        Args:
            history (List[Dict[str, float]]): Training history
            
        Returns:
            bool: Whether to stop training
        """
        if len(history) < 10:
            return False
        
        # Check if NDCG@10 has plateaued
        ndcg_values = [h.get('ndcg@10', 0) for h in history[-10:]]
        
        if len(ndcg_values) < 10:
            return False
        
        # Check if improvement is less than threshold
        improvement = ndcg_values[-1] - ndcg_values[0]
        if improvement < 0.001:
            self.logger.log_info("No significant improvement in last 10 epochs")
            return True
        
        return False
    
    def _log_epoch_metrics(self, epoch: int, loss: float, metrics: Dict[str, float], gate_stats: Dict[str, Any]) -> None:
        """
        Log epoch metrics.
        
        Args:
            epoch (int): Epoch number
            loss (float): Epoch loss
            metrics (Dict[str, float]): Epoch metrics
            gate_stats (Dict[str, Any]): Gate statistics
        """
        self.logger.log_info(f"Epoch {epoch + 1} Results:")
        self.logger.log_info(f"  Loss: {loss:.4f}")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.logger.log_info(f"  {key}: {value:.4f}")
        
        self.logger.log_info(f"  Gate Threshold: {gate_stats.get('gate_threshold', 0):.3f}")
        self.logger.log_info(f"  LLM Call Ratio: {gate_stats.get('llm_call_ratio', 0):.3f}")
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], loss: float, is_best: bool = False) -> None:
        """
        Save hybrid training checkpoint.
        
        Args:
            epoch (int): Current epoch
            metrics (Dict[str, float]): Evaluation metrics
            loss (float): Training loss
            is_best (bool): Whether this is the best model
        """
        checkpoint = {
            'epoch': epoch,
            'epochs_completed': epoch + 1,
            'loss': loss,
            'metrics': metrics,
            'gate_threshold': self.gate.threshold,
            'gnn_state': self.gnn_model.state_dict() if TORCH_AVAILABLE else None,
            'gate_state': self.gate.state_dict() if TORCH_AVAILABLE else None,
            'config': self.config,
            'hybrid_engine': self.hybrid_engine,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save checkpoint
        if is_best:
            filename = "phase3_best.pt"
        else:
            filename = f"phase3_epoch_{epoch+1:03d}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        self.logger.log_debug(f"Checkpoint saved: {checkpoint_path}")
        
        # Also save as JSON for inspection
        json_path = checkpoint_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump({
                'epoch': epoch,
                'epochs_completed': epoch + 1,
                'loss': loss,
                'metrics': metrics,
                'gate_threshold': self.gate.threshold,
                'timestamp': checkpoint['timestamp']
            }, f, indent=2, default=str)
    
    def _resume_training(self) -> None:
        """
        Resume training from checkpoint.
        """
        self.logger.log_info(f"\nResuming from checkpoint: {self.resume_from}")
        
        if not self.resume_from.exists():
            self.logger.log_warning(f"Checkpoint not found: {self.resume_from}")
            return
        
        with open(self.resume_from, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Restore state
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        
        # Restore gate threshold
        if 'gate_threshold' in checkpoint:
            self.gate.threshold = checkpoint['gate_threshold']
        
        # Restore model states
        if TORCH_AVAILABLE:
            if 'gnn_state' in checkpoint and self.gnn_model:
                self.gnn_model.load_state_dict(checkpoint['gnn_state'])
            if 'gate_state' in checkpoint and self.gate:
                self.gate.load_state_dict(checkpoint['gate_state'])
        
        # Restore hybrid engine
        if 'hybrid_engine' in checkpoint:
            self.hybrid_engine = checkpoint['hybrid_engine']
        
        self.logger.log_info(f"Resuming from epoch {self.start_epoch}")
        self.logger.log_info(f"Best metrics: {checkpoint.get('metrics', {})}")
        self.logger.log_info(f"Gate threshold: {self.gate.threshold}")
    
    def _save_final_model(self) -> None:
        """
        Save final hybrid model.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("SAVING FINAL HYBRID MODEL")
        self.logger.log_info("-" * 50)
        
        # Save final model
        model_path = self.checkpoint_dir / "hybrid_final.pt"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'hybrid_engine': self.hybrid_engine,
                'gnn_model': self.gnn_model,
                'gate': self.gate,
                'config': self.config,
                'threshold': self.gate.threshold,
                'timestamp': datetime.now().isoformat()
            }, f)
        
        self.logger.log_info(f"Final hybrid model saved to: {model_path}")
        
        # Save as PyTorch model if available
        if TORCH_AVAILABLE:
            torch_path = self.checkpoint_dir / "hybrid_final.pth"
            torch.save({
                'gnn_state_dict': self.gnn_model.state_dict(),
                'gate_threshold': self.gate.threshold,
                'config': self.config
            }, torch_path)
            self.logger.log_info(f"PyTorch model saved to: {torch_path}")
    
    def _generate_summary(self, results: Dict[str, Any]) -> None:
        """
        Generate training summary.
        
        Args:
            results (Dict[str, Any]): Training results
        """
        summary_path = self.output_dir / "training_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PHASE 3 HYBRID TRAINING SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write(f"Seed: {self.seed}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Epochs completed: {results.get('epochs_completed', 0)}\n")
            f.write("\n")
            
            f.write("BEST METRICS\n")
            f.write("-" * 40 + "\n")
            best_metrics = results.get('best_metrics', {})
            for key, value in best_metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"  {key}: {value:.4f}\n")
            f.write("\n")
            
            f.write("OPTIMAL GATING\n")
            f.write("-" * 40 + "\n")
            opt_results = results.get('threshold_optimization', {})
            f.write(f"  Optimal Threshold: {opt_results.get('optimal_threshold', 0):.3f}\n")
            f.write(f"  Optimal NDCG@10: {opt_results.get('ndcg_scores', [0])[0]:.4f}\n")
            f.write(f"  Optimal LLM Ratio: {opt_results.get('llm_call_ratios', [0])[0]:.3f}\n")
            f.write("\n")
            
            f.write("FINAL METRICS\n")
            f.write("-" * 40 + "\n")
            if results.get('metrics_history'):
                final_metrics = results['metrics_history'][-1]
                for key, value in final_metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {key}: {value:.4f}\n")
            f.write("\n")
            
            f.write("GATE STATISTICS\n")
            f.write("-" * 40 + "\n")
            if results.get('gate_history'):
                final_gate = results['gate_history'][-1]
                f.write(f"  LLM Call Ratio: {final_gate.get('llm_call_ratio', 0):.3f}\n")
                f.write(f"  Gate Threshold: {final_gate.get('gate_threshold', 0):.3f}\n")
                decisions = final_gate.get('decision_distribution', {})
                f.write(f"  GNN Decisions: {decisions.get('gnn', 0)}\n")
                f.write(f"  LLM Decisions: {decisions.get('llm', 0)}\n")
            f.write("\n")
            
            f.write("OUTPUTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Output directory: {self.output_dir}\n")
            f.write(f"  Checkpoint directory: {self.checkpoint_dir}\n")
            f.write(f"  Deployment directory: {self.output_dir / 'deployment'}\n")
            f.write(f"  Log directory: {self.log_dir}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("End of Summary\n")
            f.write("=" * 80 + "\n")
        
        self.logger.log_info(f"Training summary saved to: {summary_path}")
    
    def evaluate_hybrid(self) -> Dict[str, Any]:
        """
        Evaluate the hybrid model.
        
        Returns:
            Dict[str, Any]: Evaluation results
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("EVALUATING HYBRID MODEL")
        self.logger.log_info("-" * 50)
        
        if not self.hybrid_engine:
            self.logger.log_error("Hybrid engine not initialized")
            return {}
        
        # Create evaluator
        evaluator = Evaluator(
            model=self.hybrid_engine,
            dataset=self.dataset,
            config=self.config,
            logger=self.logger
        )
        
        # Run evaluation
        results = evaluator.evaluate()
        
        # Add efficiency metrics
        efficiency_evaluator = EfficiencyEvaluator(self.config)
        efficiency_results = efficiency_evaluator.compute_llm_call_ratio(
            self.gate, 
            self.data_loader.get_test_batches()
        )
        results['efficiency'] = efficiency_results
        
        self.logger.log_info("Hybrid evaluation completed")
        
        return results


def main():
    """
    Main entry point for Phase 3 training.
    """
    parser = argparse.ArgumentParser(description="H-GRAGrecsys Phase 3 Hybrid Training Script")
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--gnn-path',
        type=str,
        default=None,
        help='Path to Phase 2 GNN model checkpoint'
    )
    
    parser.add_argument(
        '--llm-path',
        type=str,
        default=None,
        help='Path to Phase 1 LLM model checkpoint'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory to save checkpoints'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for training artifacts'
    )
    
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Checkpoint to resume from'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--no-verbose',
        action='store_true',
        help='Disable verbose output'
    )
    
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only run evaluation, skip training'
    )
    
    parser.add_argument(
        '--deploy-only',
        action='store_true',
        help='Only deploy model, skip training'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Initial gate threshold'
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = Phase3Trainer(
        config_path=args.config,
        gnn_path=args.gnn_path,
        llm_path=args.llm_path,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        seed=args.seed,
        device=args.device,
        verbose=not args.no_verbose
    )
    
    # Override threshold if provided
    if args.threshold is not None:
        trainer.gate.threshold = args.threshold
        trainer.logger.log_info(f"Threshold set to: {args.threshold}")
    
    # Run training or evaluation
    if args.eval_only:
        results = trainer.evaluate_hybrid()
        print(f"Hybrid evaluation results: {results}")
    elif args.deploy_only:
        # Load model and deploy
        trainer._load_gnn_model()
        trainer._load_llm_model()
        trainer._initialize_gate()
        trainer._initialize_hybrid_engine()
        trainer._deploy_hybrid_model()
        print(f"Model deployed to: {trainer.output_dir / 'deployment'}")
    else:
        results = trainer.train()
        
        # Print summary
        print("\n" + "=" * 40)
        print("Hybrid training completed!")
        print(f"Best NDCG@10: {results['best_metrics'].get('ndcg@10', 0):.4f}")
        print(f"Best Hit Rate: {results['best_metrics'].get('hit_rate', 0):.4f}")
        print(f"Optimal Threshold: {results['threshold_optimization'].get('optimal_threshold', 0):.3f}")
        print(f"Checkpoints saved in: {trainer.checkpoint_dir}")
        print(f"Deployment package in: {trainer.output_dir / 'deployment'}")
        print("=" * 40 + "\n")
    
    return results


if __name__ == "__main__":
    main()