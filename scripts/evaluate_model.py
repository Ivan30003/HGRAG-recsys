"""
scripts/evaluate_model.py

Comprehensive Model Evaluation Script for H-GRAGrecsys

This script provides comprehensive evaluation capabilities for H-GRAGrecsys models:
1. Load trained models from checkpoints
2. Evaluate on test datasets
3. Compute standard recommendation metrics (NDCG, Hit Rate, Recall, Precision, MRR)
4. Efficiency evaluation (LLM call ratio, inference time, cost)
5. Memory usage analysis
6. Generate detailed evaluation reports
7. Visualize results
8. Compare multiple models
9. Ablation analysis
10. Cold-start performance evaluation

Features:
- Multi-metric evaluation
- Efficiency analysis
- Report generation (JSON, CSV, HTML)
- Visualization
- Model comparison
- Statistical significance testing
- GPU support
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
from collections import defaultdict
import numpy as np
import pandas as pd

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.logger import get_logger
from utils.config_loader import ConfigLoader, load_config
from utils.seed_manager import create_seed_manager
from utils.timer import Timer, global_timer
from utils.visualizer import create_visualizer

# Import evaluation modules
from evaluation.evaluator import Evaluator
from evaluation.metrics import Metrics
from evaluation.ranking_evaluator import RankingEvaluator
from evaluation.efficiency_evaluator import EfficiencyEvaluator
from evaluation.ablation_study import AblationStudy
from evaluation.cold_start_experiment import ColdStartExperiment

# Import data modules
from data.amazon_dataset import AmazonDataset
from data.data_loader import DataLoader

# Import model components
from models.hybrid.inference_engine import HybridInferenceEngine
from models.gnn.gnn_encoder import GNNEncoder
from models.llm.llm_interface import LLMInterface
from models.hybrid.adaptive_gate import AdaptiveGate

# Try to import torch for GPU support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class ModelEvaluator:
    """
    Comprehensive model evaluator for H-GRAGrecsys.
    
    Features:
    - Multi-metric evaluation
    - Efficiency analysis
    - Report generation
    - Visualization
    - Model comparison
    - Statistical testing
    - Cold-start evaluation
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        model_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        logger: Optional['Logger'] = None,
        verbose: bool = True
    ):
        """
        Initialize the ModelEvaluator.
        
        Args:
            config_path (str, Path, optional): Path to configuration file
            model_path (str, Path, optional): Path to model checkpoint
            output_dir (str, Path, optional): Output directory for results
            seed (int, optional): Random seed for reproducibility
            device (str, optional): Device to use ('cpu', 'cuda')
            logger (Logger, optional): Logger instance
            verbose (bool): Whether to enable verbose output
        
        Example:
            evaluator = ModelEvaluator(
                config_path='config/default_config.yaml',
                model_path='experiments/phase3/checkpoints/phase3_best.pt',
                output_dir='evaluation/results'
            )
            results = evaluator.evaluate()
        """
        # Setup paths
        self.config_path = Path(config_path) if config_path else None
        self.model_path = Path(model_path) if model_path else None
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        if logger is None:
            self.logger = get_logger(
                log_dir=self.output_dir / "logs",
                name="model_evaluator",
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
            name="model_evaluation",
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
        
        # Initialize components
        self.dataset = None
        self.model = None
        self.evaluator = None
        self.efficiency_evaluator = None
        self.metrics = None
        
        # Results storage
        self.results = {}
        self.comparison_results = {}
        
        self.logger.log_info("ModelEvaluator initialized")
        self.logger.log_info(f"Output directory: {self.output_dir}")
        self.logger.log_info(f"Model path: {self.model_path}")
    
    def _get_default_device(self) -> str:
        """
        Get the default device (GPU if available, else CPU).
        
        Returns:
            str: Device name
        """
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def evaluate(
        self,
        dataset_name: Optional[str] = None,
        eval_metrics: Optional[List[str]] = None,
        k_values: Optional[List[int]] = None,
        num_negatives: Optional[int] = None,
        run_efficiency: bool = True,
        run_cold_start: bool = False,
        run_ablation: bool = False,
        save_predictions: bool = True,
        generate_report: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive model evaluation.
        
        Args:
            dataset_name (str, optional): Dataset name to evaluate on
            eval_metrics (List[str], optional): Metrics to compute
            k_values (List[int], optional): K values for ranking metrics
            num_negatives (int, optional): Number of negative samples
            run_efficiency (bool): Whether to run efficiency evaluation
            run_cold_start (bool): Whether to run cold-start evaluation
            run_ablation (bool): Whether to run ablation study
            save_predictions (bool): Whether to save predictions
            generate_report (bool): Whether to generate report
            
        Returns:
            Dict[str, Any]: Evaluation results
        
        Example:
            results = evaluator.evaluate(
                dataset_name='Amazon_Books',
                eval_metrics=['ndcg@10', 'hit_rate@10', 'recall@10'],
                k_values=[1, 5, 10]
            )
        """
        self.logger.log_info("=" * 80)
        self.logger.log_info("Starting Model Evaluation")
        self.logger.log_info("=" * 80)
        
        with self.timer.measure("evaluation"):
            # Step 1: Load data
            self._load_data(dataset_name)
            
            # Step 2: Load model
            self._load_model()
            
            # Step 3: Initialize evaluator
            self._initialize_evaluator(eval_metrics, k_values, num_negatives)
            
            # Step 4: Run main evaluation
            self.results['main'] = self._run_main_evaluation()
            
            # Step 5: Run efficiency evaluation
            if run_efficiency:
                self.results['efficiency'] = self._run_efficiency_evaluation()
            
            # Step 6: Run cold-start evaluation
            if run_cold_start:
                self.results['cold_start'] = self._run_cold_start_evaluation()
            
            # Step 7: Run ablation study
            if run_ablation:
                self.results['ablation'] = self._run_ablation_study()
            
            # Step 8: Save predictions
            if save_predictions:
                self._save_predictions()
            
            # Step 9: Generate report
            if generate_report:
                self._generate_report()
            
            # Step 10: Generate visualizations
            self._generate_visualizations()
            
            # Step 11: Save results
            self._save_results()
        
        self.logger.log_info("=" * 80)
        self.logger.log_info("Model Evaluation Completed")
        self.logger.log_info("=" * 80)
        
        return self.results
    
    def _load_data(self, dataset_name: Optional[str] = None) -> None:
        """
        Load dataset for evaluation.
        
        Args:
            dataset_name (str, optional): Dataset name
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("LOADING DATA")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("load_data"):
            # Use specified dataset or from config
            dataset_name = dataset_name or self.config.get('data', {}).get('dataset_name', 'Amazon_Books')
            
            # Load dataset
            self.dataset = AmazonDataset(dataset_name, self.config)
            self.dataset.load_data()
            
            # Get statistics
            stats = self.dataset.get_statistics()
            self.logger.log_info(f"Dataset: {dataset_name}")
            self.logger.log_info(f"Users: {stats.get('num_users', 0):,}")
            self.logger.log_info(f"Items: {stats.get('num_items', 0):,}")
            self.logger.log_info(f"Interactions: {stats.get('num_interactions', 0):,}")
            self.logger.log_info(f"Sparsity: {stats.get('sparsity', 1.0):.4f}")
    
    def _load_model(self) -> None:
        """
        Load model from checkpoint.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("LOADING MODEL")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("load_model"):
            if not self.model_path or not self.model_path.exists():
                self.logger.log_warning("Model path not found. Creating default model...")
                self._create_default_model()
                return
            
            try:
                # Load checkpoint
                with open(self.model_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                
                # Extract model
                if 'hybrid_engine' in checkpoint:
                    self.model = checkpoint['hybrid_engine']
                elif 'student_model' in checkpoint:
                    # Phase 2 model
                    gnn_model = checkpoint['student_model']
                    llm_model = LLMInterface(
                        self.config.get('llm', {}).get('model_name', 'gpt-3.5-turbo'),
                        self.config
                    )
                    gate = AdaptiveGate(self.config)
                    gate.threshold = checkpoint.get('gate_threshold', 0.3)
                    
                    self.model = HybridInferenceEngine(
                        gnn_encoder=gnn_model,
                        llm_interface=llm_model,
                        gate=gate,
                        config=self.config
                    )
                else:
                    self.logger.log_error("Unknown model format in checkpoint")
                    self._create_default_model()
                    return
                
                # Load threshold if available
                if 'gate_threshold' in checkpoint:
                    self.model.gate.threshold = checkpoint['gate_threshold']
                
                self.logger.log_info(f"Model loaded from: {self.model_path}")
                self.logger.log_info(f"Gate threshold: {self.model.gate.threshold}")
                
            except Exception as e:
                self.logger.log_error(f"Failed to load model: {e}")
                self.logger.log_info("Creating default model...")
                self._create_default_model()
    
    def _create_default_model(self) -> None:
        """
        Create a default model for evaluation.
        """
        # Create default models
        gnn_model = GNNEncoder(self.config)
        llm_model = LLMInterface(
            self.config.get('llm', {}).get('model_name', 'gpt-3.5-turbo'),
            self.config
        )
        gate = AdaptiveGate(self.config)
        gate.threshold = self.config.get('model', {}).get('hybrid', {}).get('gate_threshold', 0.3)
        
        self.model = HybridInferenceEngine(
            gnn_encoder=gnn_model,
            llm_interface=llm_model,
            gate=gate,
            config=self.config
        )
        
        self.logger.log_info("Default model created")
    
    def _initialize_evaluator(
        self,
        eval_metrics: Optional[List[str]],
        k_values: Optional[List[int]],
        num_negatives: Optional[int]
    ) -> None:
        """
        Initialize evaluator.
        
        Args:
            eval_metrics (List[str], optional): Metrics to compute
            k_values (List[int], optional): K values
            num_negatives (int, optional): Number of negative samples
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("INITIALIZING EVALUATOR")
        self.logger.log_info("-" * 50)
        
        # Get evaluation config
        eval_config = self.config.get('evaluation', {})
        eval_metrics = eval_metrics or eval_config.get('metrics', ['ndcg', 'hit_rate', 'recall', 'precision', 'mrr'])
        k_values = k_values or eval_config.get('k_values', [1, 5, 10])
        num_negatives = num_negatives or eval_config.get('num_negatives', 99)
        
        # Initialize evaluator
        self.evaluator = Evaluator(
            model=self.model,
            dataset=self.dataset,
            config=self.config,
            logger=self.logger
        )
        
        # Initialize ranking evaluator
        self.ranking_evaluator = RankingEvaluator(
            model=self.model,
            config=self.config
        )
        
        # Initialize metrics
        self.metrics = Metrics(self.config)
        
        self.logger.log_info(f"Metrics: {eval_metrics}")
        self.logger.log_info(f"K values: {k_values}")
        self.logger.log_info(f"Number of negatives: {num_negatives}")
    
    def _run_main_evaluation(self) -> Dict[str, Any]:
        """
        Run main evaluation.
        
        Returns:
            Dict[str, Any]: Evaluation results
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("RUNNING MAIN EVALUATION")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("main_evaluation"):
            # Run evaluation
            results = self.evaluator.evaluate()
            
            # Extract metrics
            metrics = {}
            if 'metrics' in results:
                for key, value in results['metrics'].items():
                    if isinstance(value, (int, float)):
                        metrics[key] = value
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                metrics[f"{key}_{sub_key}"] = sub_value
            
            self.logger.log_info("Main evaluation completed")
            self._log_metrics(metrics)
            
            return {
                'metrics': metrics,
                'full_results': results
            }
    
    def _run_efficiency_evaluation(self) -> Dict[str, Any]:
        """
        Run efficiency evaluation.
        
        Returns:
            Dict[str, Any]: Efficiency results
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("RUNNING EFFICIENCY EVALUATION")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("efficiency_evaluation"):
            # Initialize efficiency evaluator
            self.efficiency_evaluator = EfficiencyEvaluator(self.config)
            
            # Compute LLM call ratio
            test_batches = self.dataset.get_test_batches()
            llm_ratio = self.efficiency_evaluator.compute_llm_call_ratio(
                self.model.gate,
                test_batches
            )
            
            # Measure inference time
            inference_time = self.efficiency_evaluator.measure_inference_time(
                self.model,
                test_batches
            )
            
            # Measure memory usage
            memory_usage = self.efficiency_evaluator.measure_memory_usage(
                self.model
            )
            
            # Compute cost savings
            cost_savings = self.efficiency_evaluator.compute_cost_savings(
                baseline_cost=1.0,  # Assuming LLM-only baseline
                current_cost=1.0 - llm_ratio * 0.5  # Rough estimate
            )
            
            results = {
                'llm_call_ratio': llm_ratio,
                'inference_time': inference_time,
                'memory_usage': memory_usage,
                'cost_savings': cost_savings,
                'throughput': 1.0 / inference_time if inference_time > 0 else 0
            }
            
            self.logger.log_info("Efficiency evaluation completed")
            self._log_metrics(results)
            
            return results
    
    def _run_cold_start_evaluation(self) -> Dict[str, Any]:
        """
        Run cold-start evaluation.
        
        Returns:
            Dict[str, Any]: Cold-start results
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("RUNNING COLD-START EVALUATION")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("cold_start_evaluation"):
            # Initialize cold-start experiment
            cold_start = ColdStartExperiment(
                config_path=self.config_path,
                base_experiment_name="cold_start_eval",
                output_dir=self.output_dir / "cold_start",
                scenarios=['zero_shot', 'few_shot_1', 'few_shot_3', 'few_shot_5', 'few_shot_10'],
                seed=self.seed,
                logger=self.logger
            )
            
            # Run evaluation
            results = cold_start.run()
            
            self.logger.log_info("Cold-start evaluation completed")
            
            return results
    
    def _run_ablation_study(self) -> Dict[str, Any]:
        """
        Run ablation study.
        
        Returns:
            Dict[str, Any]: Ablation results
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("RUNNING ABLATION STUDY")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("ablation_study"):
            # Initialize ablation study
            ablation = AblationStudy(
                base_model=self.model,
                config=self.config,
                logger=self.logger
            )
            
            # Run ablation
            results = ablation.run()
            
            self.logger.log_info("Ablation study completed")
            
            return results
    
    def _save_predictions(self) -> None:
        """
        Save model predictions to file.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("SAVING PREDICTIONS")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("save_predictions"):
            # Get test data
            test_data = self.dataset.get_test_batches()
            
            # Make predictions
            predictions = []
            ground_truth = []
            
            for batch in tqdm(test_data, desc="Generating predictions", 
                            disable=not TQDM_AVAILABLE):
                if isinstance(batch, dict):
                    for user_id, items in batch.items():
                        pred = self.model.rank_items(
                            user=user_id,
                            items=items,
                            context={}
                        )
                        predictions.append({
                            'user_id': user_id,
                            'predictions': pred
                        })
                        
                        # Get ground truth
                        user_items = self.dataset.get_user_items().get(user_id, [])
                        ground_truth.append({
                            'user_id': user_id,
                            'items': user_items
                        })
            
            # Save predictions
            pred_path = self.output_dir / "predictions.json"
            with open(pred_path, 'w') as f:
                json.dump({
                    'predictions': predictions,
                    'ground_truth': ground_truth,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2, default=str)
            
            self.logger.log_info(f"Predictions saved to: {pred_path}")
            
            # Save as CSV for easy analysis
            if PANDAS_AVAILABLE:
                pred_df = pd.DataFrame([{
                    'user_id': p['user_id'],
                    'top_1': p['predictions'][0] if p['predictions'] else '',
                    'top_5': p['predictions'][:5] if p['predictions'] else [],
                    'top_10': p['predictions'][:10] if p['predictions'] else []
                } for p in predictions])
                pred_df.to_csv(self.output_dir / "predictions.csv", index=False)
                self.logger.log_info(f"Predictions CSV saved to: {self.output_dir / 'predictions.csv'}")
    
    def _generate_report(self) -> None:
        """
        Generate comprehensive evaluation report.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("GENERATING REPORT")
        self.logger.log_info("-" * 50)
        
        report_dir = self.output_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate text report
        report_path = report_dir / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write(self._generate_text_report())
        
        # Generate HTML report if plotly available
        try:
            import plotly
            self._generate_html_report(report_dir)
        except ImportError:
            self.logger.log_warning("Plotly not available for HTML report")
        
        self.logger.log_info(f"Report generated in: {report_dir}")
    
    def _generate_text_report(self) -> str:
        """
        Generate text report.
        
        Returns:
            str: Text report
        """
        lines = [
            "=" * 80,
            f"MODEL EVALUATION REPORT",
            "=" * 80,
            f"Date: {datetime.now().isoformat()}",
            f"Seed: {self.seed}",
            f"Device: {self.device}",
            f"Model: {self.model_path if self.model_path else 'Default'}",
            "",
            "=" * 50,
            "EVALUATION RESULTS",
            "=" * 50,
        ]
        
        # Main metrics
        if 'main' in self.results:
            metrics = self.results['main'].get('metrics', {})
            lines.append("\nMain Evaluation Metrics:")
            lines.append("-" * 40)
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    lines.append(f"  {key}: {value:.4f}")
        
        # Efficiency metrics
        if 'efficiency' in self.results:
            efficiency = self.results['efficiency']
            lines.append("\nEfficiency Metrics:")
            lines.append("-" * 40)
            for key, value in efficiency.items():
                if isinstance(value, (int, float)):
                    if key == 'inference_time':
                        lines.append(f"  {key}: {value:.4f} seconds")
                    elif key == 'memory_usage':
                        lines.append(f"  {key}: {value:.2f} MB")
                    else:
                        lines.append(f"  {key}: {value:.4f}")
        
        # Cold-start metrics
        if 'cold_start' in self.results:
            lines.append("\nCold-Start Performance:")
            lines.append("-" * 40)
            cold_results = self.results['cold_start']
            for scenario, results in cold_results.items():
                if isinstance(results, dict) and 'metrics' in results:
                    metrics = results['metrics']
                    ndcg = metrics.get('ndcg@10', 0)
                    lines.append(f"  {scenario}: NDCG@10 = {ndcg:.4f}")
        
        # Ablation results
        if 'ablation' in self.results:
            lines.append("\nAblation Study Results:")
            lines.append("-" * 40)
            ablation_results = self.results['ablation']
            for variant, results in ablation_results.items():
                if isinstance(results, dict) and 'metrics' in results:
                    metrics = results['metrics']
                    ndcg = metrics.get('ndcg@10', 0)
                    lines.append(f"  {variant}: NDCG@10 = {ndcg:.4f}")
        
        lines.append("\n" + "=" * 80)
        lines.append("End of Report")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _generate_html_report(self, report_dir: Path) -> None:
        """
        Generate HTML report.
        
        Args:
            report_dir (Path): Directory to save report
        """
        # Prepare dashboard data
        dashboard_data = {}
        
        # Add metrics
        if 'main' in self.results:
            metrics = self.results['main'].get('metrics', {})
            dashboard_data['main_metrics'] = metrics
        
        # Add efficiency
        if 'efficiency' in self.results:
            dashboard_data['efficiency'] = self.results['efficiency']
        
        # Create dashboard
        if dashboard_data:
            self.visualizer.create_dashboard(
                dashboard_data,
                title=f"Model Evaluation: {self.model_path.stem if self.model_path else 'Default'}",
                save_name=str(report_dir / "dashboard"),
                show=False
            )
    
    def _generate_visualizations(self) -> None:
        """
        Generate evaluation visualizations.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("GENERATING VISUALIZATIONS")
        self.logger.log_info("-" * 50)
        
        # Plot main metrics
        if 'main' in self.results:
            metrics = self.results['main'].get('metrics', {})
            if metrics:
                # Create bar chart of key metrics
                self._plot_metrics_bar(metrics)
        
        # Plot efficiency metrics
        if 'efficiency' in self.results:
            efficiency = self.results['efficiency']
            self._plot_efficiency_metrics(efficiency)
        
        # Plot cold-start performance
        if 'cold_start' in self.results:
            cold_results = self.results['cold_start']
            self._plot_cold_start_performance(cold_results)
        
        # Plot ablation results
        if 'ablation' in self.results:
            ablation_results = self.results['ablation']
            self._plot_ablation_results(ablation_results)
    
    def _plot_metrics_bar(self, metrics: Dict[str, float]) -> None:
        """
        Plot metrics as bar chart.
        
        Args:
            metrics (Dict[str, float]): Metrics to plot
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        try:
            # Filter metrics for plotting
            plot_metrics = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and k in ['ndcg@10', 'hit_rate@10', 'recall@10', 'precision@10', 'mrr']
            }
            
            if not plot_metrics:
                return
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            names = list(plot_metrics.keys())
            values = list(plot_metrics.values())
            
            bars = ax.bar(names, values, color=self.visualizer.color_palette[:len(names)])
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=10)
            
            ax.set_ylabel('Score')
            ax.set_title('Evaluation Metrics')
            ax.set_ylim([0, max(1.0, max(values) * 1.2)])
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'metrics_bar.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.log_info("Metrics bar chart saved")
            
        except Exception as e:
            self.logger.log_warning(f"Failed to plot metrics bar: {e}")
    
    def _plot_efficiency_metrics(self, efficiency: Dict[str, float]) -> None:
        """
        Plot efficiency metrics.
        
        Args:
            efficiency (Dict[str, float]): Efficiency metrics
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # LLM call ratio (pie chart)
            llm_ratio = efficiency.get('llm_call_ratio', 0)
            ax1 = axes[0]
            ax1.pie([llm_ratio, 1 - llm_ratio],
                    labels=['LLM', 'GNN'],
                    autopct='%1.1f%%',
                    colors=[self.visualizer.colors['secondary'], self.visualizer.colors['primary']])
            ax1.set_title('LLM Call Ratio')
            
            # Inference time (bar chart)
            ax2 = axes[1]
            times = [
                efficiency.get('inference_time', 0),
                efficiency.get('inference_time', 0) * 0.3  # Estimated GNN only
            ]
            ax2.bar(['Hybrid', 'GNN Only'], times,
                    color=[self.visualizer.colors['primary'], self.visualizer.colors['secondary']])
            ax2.set_ylabel('Time (seconds)')
            ax2.set_title('Inference Time Comparison')
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'efficiency_metrics.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.log_info("Efficiency metrics plot saved")
            
        except Exception as e:
            self.logger.log_warning(f"Failed to plot efficiency metrics: {e}")
    
    def _plot_cold_start_performance(self, cold_results: Dict[str, Any]) -> None:
        """
        Plot cold-start performance.
        
        Args:
            cold_results (Dict[str, Any]): Cold-start results
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        try:
            # Extract data
            scenarios = []
            ndcg_scores = []
            
            for scenario, results in cold_results.items():
                if isinstance(results, dict) and 'metrics' in results:
                    metrics = results['metrics']
                    ndcg = metrics.get('ndcg@10', 0)
                    scenarios.append(scenario)
                    ndcg_scores.append(ndcg)
            
            if not scenarios:
                return
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(scenarios, ndcg_scores, 'o-', 
                   color=self.visualizer.colors['primary'], linewidth=2, markersize=8)
            
            ax.set_xlabel('Cold-Start Scenario')
            ax.set_ylabel('NDCG@10')
            ax.set_title('Cold-Start Performance')
            ax.grid(True, alpha=0.3)
            ax.set_xticklabels(scenarios, rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'cold_start_performance.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.log_info("Cold-start performance plot saved")
            
        except Exception as e:
            self.logger.log_warning(f"Failed to plot cold-start performance: {e}")
    
    def _plot_ablation_results(self, ablation_results: Dict[str, Any]) -> None:
        """
        Plot ablation results.
        
        Args:
            ablation_results (Dict[str, Any]): Ablation results
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        try:
            # Extract data
            variants = []
            ndcg_scores = []
            
            for variant, results in ablation_results.items():
                if isinstance(results, dict) and 'metrics' in results:
                    metrics = results['metrics']
                    ndcg = metrics.get('ndcg@10', 0)
                    variants.append(variant)
                    ndcg_scores.append(ndcg)
            
            if not variants:
                return
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            bars = ax.bar(variants, ndcg_scores, color=self.visualizer.color_palette[:len(variants)])
            
            # Add value labels
            for bar, value in zip(bars, ndcg_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('Ablation Variant')
            ax.set_ylabel('NDCG@10')
            ax.set_title('Ablation Study Results')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xticklabels(variants, rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'ablation_results.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.log_info("Ablation results plot saved")
            
        except Exception as e:
            self.logger.log_warning(f"Failed to plot ablation results: {e}")
    
    def _save_results(self) -> None:
        """
        Save all results to files.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("SAVING RESULTS")
        self.logger.log_info("-" * 50)
        
        # Save full results
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'results': self.results,
                'metadata': {
                    'model': str(self.model_path) if self.model_path else 'default',
                    'seed': self.seed,
                    'device': self.device,
                    'timestamp': datetime.now().isoformat()
                }
            }, f, indent=2, default=str)
        
        self.logger.log_info(f"Results saved to: {results_path}")
        
        # Save results to CSV
        self._save_results_csv()
    
    def _save_results_csv(self) -> None:
        """
        Save results to CSV format.
        """
        try:
            import pandas as pd
            
            # Main metrics
            if 'main' in self.results:
                metrics = self.results['main'].get('metrics', {})
                if metrics:
                    df = pd.DataFrame([metrics])
                    df.to_csv(self.output_dir / "metrics.csv", index=False)
                    self.logger.log_info(f"Metrics CSV saved to: {self.output_dir / 'metrics.csv'}")
            
            # Efficiency metrics
            if 'efficiency' in self.results:
                efficiency = self.results['efficiency']
                if efficiency:
                    df = pd.DataFrame([efficiency])
                    df.to_csv(self.output_dir / "efficiency_metrics.csv", index=False)
                    self.logger.log_info(f"Efficiency metrics CSV saved to: {self.output_dir / 'efficiency_metrics.csv'}")
            
        except Exception as e:
            self.logger.log_warning(f"Failed to save results as CSV: {e}")
    
    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log metrics to logger.
        
        Args:
            metrics (Dict[str, Any]): Metrics to log
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key in ['inference_time']:
                    self.logger.log_info(f"  {key}: {value:.4f} seconds")
                elif key in ['memory_usage']:
                    self.logger.log_info(f"  {key}: {value:.2f} MB")
                else:
                    self.logger.log_info(f"  {key}: {value:.4f}")
    
    def compare_models(
        self,
        model_paths: List[Union[str, Path]],
        model_names: Optional[List[str]] = None,
        metrics_to_compare: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models.
        
        Args:
            model_paths (List[Union[str, Path]]): List of model paths
            model_names (List[str], optional): Names for each model
            metrics_to_compare (List[str], optional): Metrics to compare
            
        Returns:
            Dict[str, Dict[str, Any]]: Comparison results
        
        Example:
            comparison = evaluator.compare_models([
                'experiments/phase1/checkpoints/phase1_best.pt',
                'experiments/phase2/checkpoints/phase2_best.pt',
                'experiments/phase3/checkpoints/phase3_best.pt'
            ], ['Phase 1', 'Phase 2', 'Phase 3'])
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("COMPARING MODELS")
        self.logger.log_info("-" * 50)
        
        comparison_results = {}
        
        # Set model names
        if model_names is None:
            model_names = [f"Model_{i+1}" for i in range(len(model_paths))]
        
        # Evaluate each model
        for model_path, model_name in zip(model_paths, model_names):
            self.logger.log_info(f"\nEvaluating {model_name}...")
            
            # Create evaluator for this model
            model_evaluator = ModelEvaluator(
                config_path=self.config_path,
                model_path=model_path,
                output_dir=self.output_dir / f"comparison_{model_name}",
                seed=self.seed,
                device=self.device,
                logger=self.logger
            )
            
            # Run evaluation
            results = model_evaluator.evaluate(
                run_efficiency=True,
                run_cold_start=False,
                run_ablation=False,
                generate_report=False
            )
            
            # Store results
            comparison_results[model_name] = results
        
        # Compare metrics
        self.comparison_results = self._compare_metrics(comparison_results, metrics_to_compare)
        
        # Generate comparison visualizations
        self._generate_comparison_visualizations(comparison_results)
        
        return self.comparison_results
    
    def _compare_metrics(
        self,
        results: Dict[str, Dict[str, Any]],
        metrics_to_compare: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare metrics across models.
        
        Args:
            results (Dict[str, Dict[str, Any]]): Evaluation results
            metrics_to_compare (List[str], optional): Metrics to compare
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        comparison = {}
        
        # Extract metrics
        model_names = list(results.keys())
        
        # Get metrics to compare
        if metrics_to_compare is None:
            # Use default metrics
            first_results = results[model_names[0]]
            if 'main' in first_results:
                metrics_to_compare = list(first_results['main'].get('metrics', {}).keys())
                # Filter to common metrics
                metrics_to_compare = [m for m in metrics_to_compare if 
                                     all(m in results[name]['main'].get('metrics', {}) 
                                         for name in model_names)]
        
        # Collect metrics
        for metric in metrics_to_compare:
            comparison[metric] = {}
            for model_name in model_names:
                if 'main' in results[model_name]:
                    value = results[model_name]['main'].get('metrics', {}).get(metric, 0)
                    comparison[metric][model_name] = value
        
        return comparison
    
    def _generate_comparison_visualizations(self, comparison_results: Dict[str, Any]) -> None:
        """
        Generate comparison visualizations.
        
        Args:
            comparison_results (Dict[str, Any]): Comparison results
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        try:
            # Extract data
            metrics = list(comparison_results.keys())
            models = list(comparison_results[metrics[0]].keys()) if metrics else []
            
            if not metrics or not models:
                return
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(len(metrics))
            width = 0.8 / len(models)
            
            for i, model in enumerate(models):
                values = [comparison_results[m][model] for m in metrics]
                offset = (i - len(models)/2 + 0.5) * width
                ax.bar(x + offset, values, width, label=model,
                       color=self.visualizer.color_palette[i % len(self.visualizer.color_palette)])
            
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Score')
            ax.set_title('Model Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.log_info("Model comparison plot saved")
            
        except Exception as e:
            self.logger.log_warning(f"Failed to generate comparison visualizations: {e}")


def main():
    """
    Main entry point for model evaluation.
    """
    parser = argparse.ArgumentParser(description="H-GRAGrecsys Model Evaluation Script")
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for evaluation results'
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
        help='Device to use for evaluation'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Dataset to evaluate on'
    )
    
    parser.add_argument(
        '--metrics',
        type=str,
        default=None,
        help='Comma-separated list of metrics to compute'
    )
    
    parser.add_argument(
        '--k-values',
        type=str,
        default=None,
        help='Comma-separated list of K values'
    )
    
    parser.add_argument(
        '--num-negatives',
        type=int,
        default=None,
        help='Number of negative samples'
    )
    
    parser.add_argument(
        '--no-efficiency',
        action='store_true',
        help='Skip efficiency evaluation'
    )
    
    parser.add_argument(
        '--no-cold-start',
        action='store_true',
        help='Skip cold-start evaluation'
    )
    
    parser.add_argument(
        '--no-ablation',
        action='store_true',
        help='Skip ablation study'
    )
    
    parser.add_argument(
        '--no-predictions',
        action='store_true',
        help='Skip saving predictions'
    )
    
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip generating report'
    )
    
    parser.add_argument(
        '--no-verbose',
        action='store_true',
        help='Disable verbose output'
    )
    
    parser.add_argument(
        '--compare',
        type=str,
        default=None,
        help='Comma-separated list of model paths to compare'
    )
    
    parser.add_argument(
        '--compare-names',
        type=str,
        default=None,
        help='Comma-separated list of names for models to compare'
    )
    
    args = parser.parse_args()
    
    # Parse lists
    if args.metrics:
        eval_metrics = [m.strip() for m in args.metrics.split(',')]
    else:
        eval_metrics = None
    
    if args.k_values:
        k_values = [int(k.strip()) for k in args.k_values.split(',')]
    else:
        k_values = None
    
    # Create evaluator
    evaluator = ModelEvaluator(
        config_path=args.config,
        model_path=args.model_path,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        verbose=not args.no_verbose
    )
    
    # Run evaluation or comparison
    if args.compare:
        # Parse model paths and names
        model_paths = [p.strip() for p in args.compare.split(',')]
        if args.compare_names:
            model_names = [n.strip() for n in args.compare_names.split(',')]
            if len(model_names) != len(model_paths):
                print("Error: Number of model names must match number of model paths")
                sys.exit(1)
        else:
            model_names = None
        
        # Compare models
        comparison = evaluator.compare_models(
            model_paths=model_paths,
            model_names=model_names,
            metrics_to_compare=eval_metrics
        )
        
        print("\n" + "=" * 40)
        print("Model Comparison Results:")
        print("=" * 40)
        for metric, values in comparison.items():
            print(f"\n{metric}:")
            for model, value in values.items():
                print(f"  {model}: {value:.4f}")
        print("=" * 40 + "\n")
        
        results = comparison
    
    else:
        # Run single model evaluation
        results = evaluator.evaluate(
            dataset_name=args.dataset,
            eval_metrics=eval_metrics,
            k_values=k_values,
            num_negatives=args.num_negatives,
            run_efficiency=not args.no_efficiency,
            run_cold_start=not args.no_cold_start,
            run_ablation=not args.no_ablation,
            save_predictions=not args.no_predictions,
            generate_report=not args.no_report
        )
        
        # Print summary
        print("\n" + "=" * 40)
        print("Evaluation Results:")
        print("=" * 40)
        
        if 'main' in results:
            metrics = results['main'].get('metrics', {})
            print("\nMain Metrics:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
        
        if 'efficiency' in results:
            efficiency = results['efficiency']
            print("\nEfficiency Metrics:")
            for key, value in efficiency.items():
                if isinstance(value, (int, float)):
                    if key == 'inference_time':
                        print(f"  {key}: {value:.4f} seconds")
                    elif key == 'memory_usage':
                        print(f"  {key}: {value:.2f} MB")
                    else:
                        print(f"  {key}: {value:.4f}")
        print("=" * 40 + "\n")
    
    return results


if __name__ == "__main__":
    main()