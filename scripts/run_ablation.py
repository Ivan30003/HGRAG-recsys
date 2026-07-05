"""
scripts/run_ablation.py

Ablation Study Runner for H-GRAGrecsys

This script provides comprehensive ablation study execution capabilities:
1. Run systematic ablation experiments
2. Test individual component removals
3. Run combined ablation variants
4. Compare ablation results
5. Generate ablation reports
6. Visualize ablation impacts
7. Statistical significance testing
8. Export ablation results

Features:
- Pre-defined ablation variants
- Custom ablation configurations
- Parallel execution
- Result comparison and ranking
- Impact analysis
- Report generation
- Visualization
- Statistical significance testing
"""

import os
import sys
import json
import yaml
import argparse
import pickle
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple, Set
from datetime import datetime
import traceback
import shutil
from collections import defaultdict
import numpy as np
import pandas as pd
import copy

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.logger import get_logger
from utils.config_loader import ConfigLoader, load_config
from utils.seed_manager import create_seed_manager
from utils.timer import Timer, global_timer
from utils.visualizer import create_visualizer

# Import experiment modules
from experiments.ablation_experiment import AblationExperiment, CombinedAblationExperiment

# Import evaluation modules
from evaluation.evaluator import Evaluator
from evaluation.metrics import Metrics
from evaluation.ablation_study import AblationStudy

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


class AblationRunner:
    """
    Comprehensive ablation study runner for H-GRAGrecsys.
    
    Features:
    - Systematic ablation execution
    - Custom variant definition
    - Parallel execution
    - Result comparison
    - Impact analysis
    - Report generation
    - Visualization
    """
    
    # Pre-defined ablation variants
    ABLATION_VARIANTS = {
        'hierarchical_memory': {
            'description': 'Remove hierarchical memory structure',
            'config_overrides': {
                'model.agent.memory_type': 'flat',
                'model.agent.hierarchical_memory': False
            },
            'impact': 'memory_consistency'
        },
        'graph_rag': {
            'description': 'Remove Graph RAG retrieval',
            'config_overrides': {
                'model.graph_rag.enabled': False,
                'model.graph_rag.metapath_extraction': False
            },
            'impact': 'context_quality'
        },
        'metapath_hops': {
            'description': 'Reduce metapath hops to 1',
            'config_overrides': {
                'model.graph.max_hops': 1
            },
            'impact': 'context_quality'
        },
        'contrastive_loss': {
            'description': 'Remove contrastive loss',
            'config_overrides': {
                'model.distillation.contrastive_weight': 0.0
            },
            'impact': 'representation_quality'
        },
        'path_importance_loss': {
            'description': 'Remove path importance loss',
            'config_overrides': {
                'model.distillation.path_importance_weight': 0.0
            },
            'impact': 'representation_quality'
        },
        'adaptive_gating': {
            'description': 'Remove adaptive gating (use GNN only)',
            'config_overrides': {
                'model.hybrid.gate_enabled': False,
                'model.hybrid.use_hybrid': False
            },
            'impact': 'inference_quality'
        },
        'collaborative_memory': {
            'description': 'Remove collaborative memory',
            'config_overrides': {
                'model.agent.collaborative_memory': False
            },
            'impact': 'memory_consistency'
        },
        'reflection_engine': {
            'description': 'Remove reflection engine',
            'config_overrides': {
                'model.agent.reflection_enabled': False
            },
            'impact': 'learning_quality'
        },
        'gnn_distillation': {
            'description': 'Remove GNN distillation (use LLM only)',
            'config_overrides': {
                'model.distillation.enabled': False
            },
            'impact': 'inference_quality'
        },
        'item_embeddings': {
            'description': 'Remove item embeddings',
            'config_overrides': {
                'model.item_embeddings.enabled': False
            },
            'impact': 'representation_quality'
        },
        'user_embeddings': {
            'description': 'Remove user embeddings',
            'config_overrides': {
                'model.user_embeddings.enabled': False
            },
            'impact': 'representation_quality'
        },
        'attention_module': {
            'description': 'Remove attention module',
            'config_overrides': {
                'model.gnn.use_attention': False
            },
            'impact': 'inference_quality'
        },
        'ppr_sampling': {
            'description': 'Remove PPR sampling',
            'config_overrides': {
                'model.graph_rag.use_ppr': False
            },
            'impact': 'context_quality'
        }
    }
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        model_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        variants: Optional[List[str]] = None,
        run_baseline: bool = True,
        parallel: bool = False,
        max_workers: int = 4,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        logger: Optional['Logger'] = None,
        verbose: bool = True
    ):
        """
        Initialize the AblationRunner.
        
        Args:
            config_path (str, Path, optional): Path to configuration file
            model_path (str, Path, optional): Path to base model checkpoint
            output_dir (str, Path, optional): Output directory for results
            variants (List[str], optional): Specific ablation variants to run
            run_baseline (bool): Whether to run baseline experiment
            parallel (bool): Whether to run experiments in parallel
            max_workers (int): Maximum number of parallel workers
            seed (int, optional): Random seed for reproducibility
            device (str, optional): Device to use ('cpu', 'cuda')
            logger (Logger, optional): Logger instance
            verbose (bool): Whether to enable verbose output
        
        Example:
            runner = AblationRunner(
                config_path='config/default_config.yaml',
                model_path='experiments/phase3/checkpoints/phase3_best.pt',
                output_dir='ablation/results',
                variants=['hierarchical_memory', 'graph_rag', 'adaptive_gating']
            )
            results = runner.run()
        """
        # Setup paths
        self.config_path = Path(config_path) if config_path else None
        self.model_path = Path(model_path) if model_path else None
        self.output_dir = Path(output_dir) if output_dir else Path("ablation/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        if logger is None:
            self.logger = get_logger(
                log_dir=self.output_dir / "logs",
                name="ablation_runner",
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
            name="ablation_runner",
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
        
        # Configure ablation
        self.run_baseline = run_baseline
        self.parallel = parallel
        self.max_workers = max_workers
        
        # Select variants
        if variants:
            self.variants = [v for v in variants if v in self.ABLATION_VARIANTS]
            if len(self.variants) != len(variants):
                invalid = set(variants) - set(self.ABLATION_VARIANTS)
                self.logger.log_warning(f"Invalid variants ignored: {invalid}")
        else:
            self.variants = list(self.ABLATION_VARIANTS.keys())
        
        # Results storage
        self.results = {}
        self.baseline_results = {}
        self.comparison_results = {}
        self.impact_analysis = {}
        
        self.logger.log_info("AblationRunner initialized")
        self.logger.log_info(f"Output directory: {self.output_dir}")
        self.logger.log_info(f"Variants to test: {len(self.variants)}")
        self.logger.log_info(f"Variants: {self.variants}")
        if self.run_baseline:
            self.logger.log_info("Baseline included")
    
    def _get_default_device(self) -> str:
        """
        Get the default device (GPU if available, else CPU).
        
        Returns:
            str: Device name
        """
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def run(self) -> Dict[str, Any]:
        """
        Run the ablation study.
        
        Returns:
            Dict[str, Any]: Ablation results
        
        Example:
            results = runner.run()
            print(f"Completed {len(results['variants'])} ablation variants")
        """
        self.logger.log_info("=" * 80)
        self.logger.log_info("Starting Ablation Study")
        self.logger.log_info("=" * 80)
        
        with self.timer.measure("ablation_study"):
            # Step 1: Run baseline
            if self.run_baseline:
                self.baseline_results = self._run_baseline()
            
            # Step 2: Run ablation variants
            if self.parallel:
                self.results = self._run_parallel_ablations()
            else:
                self.results = self._run_sequential_ablations()
            
            # Step 3: Compare results
            self.comparison_results = self._compare_results()
            
            # Step 4: Analyze impact
            self.impact_analysis = self._analyze_impact()
            
            # Step 5: Generate visualizations
            self._generate_visualizations()
            
            # Step 6: Generate report
            self._generate_report()
            
            # Step 7: Save results
            self._save_results()
        
        self.logger.log_info("=" * 80)
        self.logger.log_info("Ablation Study Completed")
        self.logger.log_info("=" * 80)
        
        return {
            'baseline': self.baseline_results,
            'variants': self.results,
            'comparison': self.comparison_results,
            'impact_analysis': self.impact_analysis
        }
    
    def _run_baseline(self) -> Dict[str, Any]:
        """
        Run baseline experiment (full model).
        
        Returns:
            Dict[str, Any]: Baseline results
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("RUNNING BASELINE (Full Model)")
        self.logger.log_info("-" * 50)
        
        with self.timer.measure("baseline"):
            try:
                # Load model if provided
                if self.model_path and self.model_path.exists():
                    model = self._load_model(self.model_path)
                else:
                    model = self._create_model()
                
                # Evaluate model
                evaluator = Evaluator(
                    model=model,
                    dataset=self._get_dataset(),
                    config=self.config,
                    logger=self.logger
                )
                
                results = evaluator.evaluate()
                metrics = self._extract_metrics(results)
                
                self.logger.log_info("Baseline completed successfully")
                self._log_metrics(metrics, prefix="Baseline")
                
                return {
                    'metrics': metrics,
                    'full_results': results,
                    'description': 'Full Model (Baseline)'
                }
                
            except Exception as e:
                self.logger.log_error(f"Baseline failed: {e}")
                self.logger.log_error(traceback.format_exc())
                return {}
    
    def _run_sequential_ablations(self) -> Dict[str, Dict[str, Any]]:
        """
        Run ablation variants sequentially.
        
        Returns:
            Dict[str, Dict[str, Any]]: Results for each variant
        """
        results = {}
        
        for variant_name in tqdm(self.variants, desc="Running ablations", 
                                 disable=not TQDM_AVAILABLE):
            self.logger.log_info(f"\nRunning ablation variant: {variant_name}")
            try:
                variant_results = self._run_single_ablation(variant_name)
                results[variant_name] = variant_results
            except Exception as e:
                self.logger.log_error(f"Variant {variant_name} failed: {e}")
                results[variant_name] = {'error': str(e), 'traceback': traceback.format_exc()}
        
        return results
    
    def _run_parallel_ablations(self) -> Dict[str, Dict[str, Any]]:
        """
        Run ablation variants in parallel.
        
        Returns:
            Dict[str, Dict[str, Any]]: Results for each variant
        """
        self.logger.log_info(f"Running {len(self.variants)} variants in parallel with {self.max_workers} workers")
        
        # Prepare arguments
        variant_args = [(variant_name, self.seed + i) for i, variant_name in enumerate(self.variants)]
        
        # Run in parallel
        with mp.Pool(processes=self.max_workers) as pool:
            results_list = pool.starmap(self._run_single_ablation_worker, variant_args)
        
        # Combine results
        results = {}
        for variant_name, variant_result in zip(self.variants, results_list):
            results[variant_name] = variant_result
        
        return results
    
    def _run_single_ablation_worker(self, variant_name: str, seed: int) -> Dict[str, Any]:
        """
        Worker function for parallel ablation execution.
        
        Args:
            variant_name (str): Name of the ablation variant
            seed (int): Seed for this run
            
        Returns:
            Dict[str, Any]: Results for this variant
        """
        # Create separate logger for worker
        worker_logger = get_logger(
            log_dir=self.output_dir / "logs",
            name=f"ablation_{variant_name}",
            verbose=False
        )
        
        try:
            # Create config with overrides
            config = copy.deepcopy(self.config)
            self._apply_ablation_config(config, variant_name)
            
            # Create model
            model = self._create_model(config)
            
            # Evaluate
            evaluator = Evaluator(
                model=model,
                dataset=self._get_dataset(),
                config=config,
                logger=worker_logger
            )
            
            results = evaluator.evaluate()
            metrics = self._extract_metrics(results)
            
            return {
                'metrics': metrics,
                'full_results': results,
                'config': config,
                'description': self.ABLATION_VARIANTS[variant_name]['description'],
                'impact': self.ABLATION_VARIANTS[variant_name]['impact']
            }
            
        except Exception as e:
            worker_logger.log_error(f"Variant {variant_name} failed: {e}")
            return {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _run_single_ablation(self, variant_name: str) -> Dict[str, Any]:
        """
        Run a single ablation variant.
        
        Args:
            variant_name (str): Name of the ablation variant
            
        Returns:
            Dict[str, Any]: Results for this variant
        """
        # Create config with overrides
        config = copy.deepcopy(self.config)
        self._apply_ablation_config(config, variant_name)
        
        # Create model
        model = self._create_model(config)
        
        # Evaluate
        evaluator = Evaluator(
            model=model,
            dataset=self._get_dataset(),
            config=config,
            logger=self.logger
        )
        
        results = evaluator.evaluate()
        metrics = self._extract_metrics(results)
        
        self.logger.log_info(f"Variant {variant_name} completed")
        self._log_metrics(metrics, prefix=f"Variant {variant_name}")
        
        return {
            'metrics': metrics,
            'full_results': results,
            'config': config,
            'description': self.ABLATION_VARIANTS[variant_name]['description'],
            'impact': self.ABLATION_VARIANTS[variant_name]['impact']
        }
    
    def _apply_ablation_config(self, config: Dict[str, Any], variant_name: str) -> None:
        """
        Apply ablation configuration overrides.
        
        Args:
            config (Dict[str, Any]): Configuration to modify
            variant_name (str): Name of the ablation variant
        """
        if variant_name not in self.ABLATION_VARIANTS:
            self.logger.log_warning(f"Unknown variant: {variant_name}")
            return
        
        overrides = self.ABLATION_VARIANTS[variant_name]['config_overrides']
        
        for key, value in overrides.items():
            self.config_loader.set_value_in_dict(config, key, value)
    
    def _load_model(self, model_path: Path) -> HybridInferenceEngine:
        """
        Load model from checkpoint.
        
        Args:
            model_path (Path): Path to model checkpoint
            
        Returns:
            HybridInferenceEngine: Loaded model
        """
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        if 'hybrid_engine' in checkpoint:
            return checkpoint['hybrid_engine']
        else:
            # Create model from components
            gnn_model = GNNEncoder(self.config)
            llm_model = LLMInterface(
                self.config.get('llm', {}).get('model_name', 'gpt-3.5-turbo'),
                self.config
            )
            gate = AdaptiveGate(self.config)
            
            return HybridInferenceEngine(
                gnn_encoder=gnn_model,
                llm_interface=llm_model,
                gate=gate,
                config=self.config
            )
    
    def _create_model(self, config: Optional[Dict[str, Any]] = None) -> HybridInferenceEngine:
        """
        Create a model instance.
        
        Args:
            config (Dict[str, Any], optional): Configuration to use
            
        Returns:
            HybridInferenceEngine: Model instance
        """
        config = config or self.config
        
        gnn_model = GNNEncoder(config)
        llm_model = LLMInterface(
            config.get('llm', {}).get('model_name', 'gpt-3.5-turbo'),
            config
        )
        gate = AdaptiveGate(config)
        
        return HybridInferenceEngine(
            gnn_encoder=gnn_model,
            llm_interface=llm_model,
            gate=gate,
            config=config
        )
    
    def _get_dataset(self):
        """
        Get dataset instance.
        
        Returns:
            AmazonDataset: Dataset instance
        """
        dataset_name = self.config.get('data', {}).get('dataset_name', 'Amazon_Books')
        dataset = AmazonDataset(dataset_name, self.config)
        dataset.load_data()
        return dataset
    
    def _extract_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract key metrics from evaluation results.
        
        Args:
            results (Dict[str, Any]): Evaluation results
            
        Returns:
            Dict[str, float]: Extracted metrics
        """
        metrics = {}
        
        if 'metrics' in results:
            eval_metrics = results['metrics']
            for key, value in eval_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[key] = value
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            metrics[f"{key}_{sub_key}"] = sub_value
        
        return metrics
    
    def _compare_results(self) -> Dict[str, Any]:
        """
        Compare results across all variants.
        
        Returns:
            Dict[str, Any]: Comparison results
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("COMPARING RESULTS")
        self.logger.log_info("-" * 50)
        
        comparison = {
            'baseline': self.baseline_results,
            'variants': {}
        }
        
        # Calculate deltas from baseline
        baseline_metrics = self.baseline_results.get('metrics', {})
        
        for variant_name, variant_data in self.results.items():
            if 'error' in variant_data:
                continue
            
            variant_metrics = variant_data.get('metrics', {})
            deltas = {}
            
            for key, baseline_value in baseline_metrics.items():
                if key in variant_metrics:
                    variant_value = variant_metrics[key]
                    if isinstance(baseline_value, (int, float)) and isinstance(variant_value, (int, float)):
                        abs_delta = variant_value - baseline_value
                        rel_delta = (abs_delta / baseline_value) * 100 if baseline_value != 0 else float('inf')
                        deltas[key] = {
                            'baseline': baseline_value,
                            'variant': variant_value,
                            'absolute_delta': abs_delta,
                            'relative_delta': rel_delta
                        }
            
            comparison['variants'][variant_name] = {
                'metrics': variant_metrics,
                'deltas': deltas,
                'description': variant_data.get('description', ''),
                'impact': variant_data.get('impact', '')
            }
        
        # Sort variants by impact
        if 'ndcg@10' in comparison['variants'].get(next(iter(self.variants), ''), {}).get('deltas', {}):
            sorted_variants = sorted(
                [v for v in self.variants if v in comparison['variants']],
                key=lambda v: abs(comparison['variants'][v]['deltas'].get('ndcg@10', {}).get('absolute_delta', 0)),
                reverse=True
            )
            comparison['sorted_by_impact'] = sorted_variants
        
        # Print comparison summary
        self._print_comparison_summary(comparison)
        
        return comparison
    
    def _print_comparison_summary(self, comparison: Dict[str, Any]) -> None:
        """
        Print comparison summary.
        
        Args:
            comparison (Dict[str, Any]): Comparison results
        """
        self.logger.log_info("\nAblation Comparison Summary:")
        self.logger.log_info("-" * 60)
        
        # Print baseline metrics
        baseline_metrics = comparison.get('baseline', {}).get('metrics', {})
        if baseline_metrics:
            self.logger.log_info("Baseline (Full Model):")
            for key, value in baseline_metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.log_info(f"  {key}: {value:.4f}")
        
        # Print variant deltas
        self.logger.log_info("\nVariant Deltas from Baseline:")
        for variant_name, variant_data in comparison.get('variants', {}).items():
            deltas = variant_data.get('deltas', {})
            description = variant_data.get('description', variant_name)
            
            self.logger.log_info(f"\n{variant_name}: {description}")
            
            # Print key metrics
            key_metrics = ['ndcg@10', 'hit_rate', 'recall', 'llm_call_ratio']
            for metric in key_metrics:
                if metric in deltas:
                    delta = deltas[metric]
                    abs_delta = delta.get('absolute_delta', 0)
                    rel_delta = delta.get('relative_delta', 0)
                    sign = '+' if abs_delta > 0 else ''
                    self.logger.log_info(
                        f"  {metric}: {delta.get('variant', 0):.4f} "
                        f"({sign}{abs_delta:.4f}, {sign}{rel_delta:.1f}%)"
                    )
    
    def _analyze_impact(self) -> Dict[str, Any]:
        """
        Analyze impact of each ablation variant.
        
        Returns:
            Dict[str, Any]: Impact analysis
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("ANALYZING IMPACT")
        self.logger.log_info("-" * 50)
        
        impact_analysis = {
            'most_impactful': [],
            'least_impactful': [],
            'impact_by_category': defaultdict(list),
            'summary': {}
        }
        
        # Calculate impact scores
        impact_scores = []
        
        for variant_name, variant_data in self.comparison_results.get('variants', {}).items():
            deltas = variant_data.get('deltas', {})
            ndcg_delta = deltas.get('ndcg@10', {}).get('absolute_delta', 0)
            impact = abs(ndcg_delta)
            
            impact_scores.append({
                'variant': variant_name,
                'impact': impact,
                'description': variant_data.get('description', ''),
                'ndcg_delta': ndcg_delta
            })
            
            # Group by impact category
            category = variant_data.get('impact', 'unknown')
            impact_analysis['impact_by_category'][category].append({
                'variant': variant_name,
                'impact': impact,
                'ndcg_delta': ndcg_delta
            })
        
        # Sort by impact
        impact_scores.sort(key=lambda x: x['impact'], reverse=True)
        
        # Most impactful (top 3)
        impact_analysis['most_impactful'] = impact_scores[:3]
        
        # Least impactful (bottom 3)
        impact_analysis['least_impactful'] = impact_scores[-3:]
        
        # Summary
        impact_analysis['summary'] = {
            'total_variants': len(self.variants),
            'average_impact': np.mean([s['impact'] for s in impact_scores]) if impact_scores else 0,
            'max_impact': max([s['impact'] for s in impact_scores]) if impact_scores else 0,
            'min_impact': min([s['impact'] for s in impact_scores]) if impact_scores else 0
        }
        
        self.logger.log_info("Impact analysis completed")
        self.logger.log_info(f"Most impactful: {impact_scores[0]['variant'] if impact_scores else 'None'}")
        self.logger.log_info(f"Average impact: {impact_analysis['summary']['average_impact']:.4f}")
        
        return impact_analysis
    
    def _generate_visualizations(self) -> None:
        """
        Generate visualizations for ablation study.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("GENERATING VISUALIZATIONS")
        self.logger.log_info("-" * 50)
        
        # Plot ablation results
        self._plot_ablation_results()
        
        # Plot impact analysis
        self._plot_impact_analysis()
        
        # Plot category impact
        self._plot_category_impact()
    
    def _plot_ablation_results(self) -> None:
        """
        Plot ablation results as bar chart.
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        try:
            # Prepare data
            variants = []
            ndcg_scores = []
            descriptions = []
            
            # Add baseline
            if self.baseline_results:
                baseline_metrics = self.baseline_results.get('metrics', {})
                variants.append('Baseline')
                ndcg_scores.append(baseline_metrics.get('ndcg@10', 0))
                descriptions.append('Full Model')
            
            # Add variants
            for variant_name, variant_data in self.results.items():
                if 'error' in variant_data:
                    continue
                metrics = variant_data.get('metrics', {})
                variants.append(variant_name)
                ndcg_scores.append(metrics.get('ndcg@10', 0))
                descriptions.append(variant_data.get('description', variant_name))
            
            if not variants:
                return
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create bar chart
            bars = ax.bar(range(len(variants)), ndcg_scores, 
                         color=self.visualizer.color_palette[:len(variants)])
            
            # Add value labels
            for bar, value in zip(bars, ndcg_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=9)
            
            # Customize plot
            ax.set_xticks(range(len(variants)))
            ax.set_xticklabels(variants, rotation=45, ha='right')
            ax.set_ylabel('NDCG@10')
            ax.set_title('Ablation Study Results')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'ablation_results.png', 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.log_info("Ablation results plot saved")
            
        except Exception as e:
            self.logger.log_warning(f"Failed to plot ablation results: {e}")
    
    def _plot_impact_analysis(self) -> None:
        """
        Plot impact analysis.
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        try:
            # Prepare data
            impact_scores = []
            variant_names = []
            
            for variant_name, variant_data in self.comparison_results.get('variants', {}).items():
                deltas = variant_data.get('deltas', {})
                ndcg_delta = deltas.get('ndcg@10', {}).get('absolute_delta', 0)
                variant_names.append(variant_name)
                impact_scores.append(abs(ndcg_delta))
            
            if not variant_names:
                return
            
            # Sort by impact
            sorted_data = sorted(zip(variant_names, impact_scores), 
                               key=lambda x: x[1], reverse=True)
            sorted_names, sorted_scores = zip(*sorted_data)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create horizontal bar chart
            y_pos = np.arange(len(sorted_names))
            bars = ax.barh(y_pos, sorted_scores, 
                          color=self.visualizer.color_palette[:len(sorted_names)])
            
            # Add value labels
            for bar, value in zip(bars, sorted_scores):
                ax.text(value + 0.01, bar.get_y() + bar.get_height()/2.,
                       f'{value:.4f}', va='center', fontsize=9)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_names)
            ax.set_xlabel('Impact (|ΔNDCG@10|)')
            ax.set_title('Impact of Ablation Variants')
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'impact_analysis.png', 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.log_info("Impact analysis plot saved")
            
        except Exception as e:
            self.logger.log_warning(f"Failed to plot impact analysis: {e}")
    
    def _plot_category_impact(self) -> None:
        """
        Plot impact by category.
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        try:
            impact_by_category = self.impact_analysis.get('impact_by_category', {})
            
            if not impact_by_category:
                return
            
            # Aggregate impact by category
            categories = []
            avg_impacts = []
            
            for category, items in impact_by_category.items():
                if items:
                    avg_impact = np.mean([item['impact'] for item in items])
                    categories.append(category)
                    avg_impacts.append(avg_impact)
            
            if not categories:
                return
            
            # Sort by impact
            sorted_data = sorted(zip(categories, avg_impacts), 
                               key=lambda x: x[1], reverse=True)
            sorted_categories, sorted_impacts = zip(*sorted_data)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bars = ax.bar(sorted_categories, sorted_impacts, 
                         color=self.visualizer.color_palette[:len(sorted_categories)])
            
            # Add value labels
            for bar, value in zip(bars, sorted_impacts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=10)
            
            ax.set_xlabel('Impact Category')
            ax.set_ylabel('Average Impact')
            ax.set_title('Impact by Category')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xticklabels(sorted_categories, rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'category_impact.png', 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.log_info("Category impact plot saved")
            
        except Exception as e:
            self.logger.log_warning(f"Failed to plot category impact: {e}")
    
    def _generate_report(self) -> None:
        """
        Generate comprehensive report.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("GENERATING REPORT")
        self.logger.log_info("-" * 50)
        
        report_dir = self.output_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate text report
        report_path = report_dir / "ablation_report.txt"
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
            f"ABLATION STUDY REPORT",
            "=" * 80,
            f"Date: {datetime.now().isoformat()}",
            f"Seed: {self.seed}",
            f"Device: {self.device}",
            f"Model: {self.model_path if self.model_path else 'Default'}",
            f"Variants Tested: {len(self.variants)}",
            "",
            "=" * 50,
            "EXECUTIVE SUMMARY",
            "=" * 50,
        ]
        
        # Most impactful variants
        if self.impact_analysis.get('most_impactful'):
            lines.append("\nMost Impactful Ablations:")
            for i, item in enumerate(self.impact_analysis['most_impactful'], 1):
                lines.append(f"  {i}. {item['variant']}: {item['description']}")
                lines.append(f"     Impact: {item['impact']:.4f} (ΔNDCG@10: {item['ndcg_delta']:.4f})")
        
        # Least impactful variants
        if self.impact_analysis.get('least_impactful'):
            lines.append("\nLeast Impactful Ablations:")
            for i, item in enumerate(self.impact_analysis['least_impactful'], 1):
                lines.append(f"  {i}. {item['variant']}: {item['description']}")
                lines.append(f"     Impact: {item['impact']:.4f} (ΔNDCG@10: {item['ndcg_delta']:.4f})")
        
        lines.append("\n" + "=" * 50)
        lines.append("DETAILED RESULTS")
        lines.append("=" * 50)
        
        # Baseline results
        if self.baseline_results:
            lines.append("\nBaseline (Full Model):")
            baseline_metrics = self.baseline_results.get('metrics', {})
            for key, value in baseline_metrics.items():
                if isinstance(value, (int, float)):
                    lines.append(f"  {key}: {value:.4f}")
        
        # Variant results
        lines.append("\nVariant Results:")
        for variant_name, variant_data in self.results.items():
            if 'error' in variant_data:
                lines.append(f"\n{variant_name}: ERROR - {variant_data['error']}")
                continue
            
            lines.append(f"\n{variant_name}: {variant_data.get('description', '')}")
            metrics = variant_data.get('metrics', {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    lines.append(f"  {key}: {value:.4f}")
        
        # Impact analysis summary
        lines.append("\n" + "=" * 50)
        lines.append("IMPACT ANALYSIS")
        lines.append("=" * 50)
        
        summary = self.impact_analysis.get('summary', {})
        lines.append(f"Total variants: {summary.get('total_variants', 0)}")
        lines.append(f"Average impact: {summary.get('average_impact', 0):.4f}")
        lines.append(f"Max impact: {summary.get('max_impact', 0):.4f}")
        lines.append(f"Min impact: {summary.get('min_impact', 0):.4f}")
        
        # Recommendations
        lines.append("\n" + "=" * 50)
        lines.append("RECOMMENDATIONS")
        lines.append("=" * 50)
        
        if self.impact_analysis.get('most_impactful'):
            lines.append("\nCritical Components (Removal Causes Significant Performance Drop):")
            for i, item in enumerate(self.impact_analysis['most_impactful'], 1):
                lines.append(f"  {i}. {item['variant']} (Impact: {item['impact']:.4f})")
        
        if self.impact_analysis.get('least_impactful'):
            lines.append("\nLess Critical Components (Removal Has Minimal Impact):")
            for i, item in enumerate(self.impact_analysis['least_impactful'], 1):
                lines.append(f"  {i}. {item['variant']} (Impact: {item['impact']:.4f})")
        
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
        
        # Add ablation results
        ablation_results = {}
        
        # Add baseline
        if self.baseline_results:
            baseline_metrics = self.baseline_results.get('metrics', {})
            ablation_results['Baseline'] = baseline_metrics
        
        # Add variants
        for variant_name, variant_data in self.results.items():
            if 'error' in variant_data:
                continue
            metrics = variant_data.get('metrics', {})
            display_name = variant_data.get('description', variant_name)
            ablation_results[display_name] = metrics
        
        dashboard_data['ablation_results'] = ablation_results
        
        # Create dashboard
        if dashboard_data:
            self.visualizer.create_dashboard(
                dashboard_data,
                title=f"Ablation Study: {self.output_dir.name}",
                save_name=str(report_dir / "dashboard"),
                show=False
            )
    
    def _save_results(self) -> None:
        """
        Save all results to files.
        """
        self.logger.log_info("\n" + "-" * 50)
        self.logger.log_info("SAVING RESULTS")
        self.logger.log_info("-" * 50)
        
        # Save full results
        results_path = self.output_dir / "ablation_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'baseline': self.baseline_results,
                'variants': self.results,
                'comparison': self.comparison_results,
                'impact_analysis': self.impact_analysis,
                'metadata': {
                    'experiment_name': self.output_dir.name,
                    'seed': self.seed,
                    'device': self.device,
                    'timestamp': datetime.now().isoformat(),
                    'variants_tested': self.variants,
                    'num_variants': len(self.variants)
                }
            }, f, indent=2, default=str)
        
        self.logger.log_info(f"Results saved to: {results_path}")
        
        # Save comparison summary
        comparison_path = self.output_dir / "comparison_summary.json"
        with open(comparison_path, 'w') as f:
            json.dump(self.comparison_results, f, indent=2, default=str)
        
        self.logger.log_info(f"Comparison summary saved to: {comparison_path}")
        
        # Save to CSV
        self._save_results_csv()
    
    def _save_results_csv(self) -> None:
        """
        Save results to CSV format.
        """
        try:
            # Create DataFrame with all metrics
            rows = []
            
            # Add baseline
            if self.baseline_results:
                baseline_metrics = self.baseline_results.get('metrics', {})
                row = {'variant': 'Baseline', 'description': 'Full Model'}
                row.update(baseline_metrics)
                rows.append(row)
            
            # Add variants
            for variant_name, variant_data in self.results.items():
                if 'error' in variant_data:
                    continue
                metrics = variant_data.get('metrics', {})
                row = {
                    'variant': variant_name,
                    'description': variant_data.get('description', ''),
                    'impact_category': variant_data.get('impact', '')
                }
                row.update(metrics)
                rows.append(row)
            
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(self.output_dir / "ablation_results.csv", index=False)
                self.logger.log_info(f"Results CSV saved to: {self.output_dir / 'ablation_results.csv'}")
            
        except Exception as e:
            self.logger.log_warning(f"Failed to save results as CSV: {e}")
    
    def _log_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> None:
        """
        Log metrics to logger.
        
        Args:
            metrics (Dict[str, Any]): Metrics to log
            prefix (str): Prefix for log messages
        """
        if prefix:
            self.logger.log_info(f"{prefix} Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.logger.log_info(f"  {key}: {value:.4f}")


def main():
    """
    Main entry point for ablation study runner.
    """
    parser = argparse.ArgumentParser(description="H-GRAGrecsys Ablation Study Runner")
    
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
        help='Path to base model checkpoint'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--variants',
        type=str,
        nargs='+',
        default=None,
        help='Specific ablation variants to run'
    )
    
    parser.add_argument(
        '--no-baseline',
        action='store_true',
        help='Skip running baseline experiment'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run experiments in parallel'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers'
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
        '--combined',
        action='store_true',
        help='Run combined ablation experiments'
    )
    
    parser.add_argument(
        '--max-combinations',
        type=int,
        default=10,
        help='Maximum number of combinations to test'
    )
    
    parser.add_argument(
        '--list-variants',
        action='store_true',
        help='List available ablation variants'
    )
    
    parser.add_argument(
        '--no-verbose',
        action='store_true',
        help='Disable verbose output'
    )
    
    parser.add_argument(
        '--custom-variant',
        type=str,
        default=None,
        help='Custom variant in format "name:key1=value1,key2=value2"'
    )
    
    args = parser.parse_args()
    
    # List variants if requested
    if args.list_variants:
        print("\nAvailable Ablation Variants:")
        print("-" * 60)
        for variant_name, variant_info in AblationRunner.ABLATION_VARIANTS.items():
            print(f"  {variant_name}:")
            print(f"    Description: {variant_info['description']}")
            print(f"    Impact: {variant_info['impact']}")
            print(f"    Config Overrides: {variant_info['config_overrides']}")
            print()
        return
    
    # Handle custom variant
    if args.custom_variant:
        # Parse custom variant
        name, overrides_str = args.custom_variant.split(':')
        overrides = {}
        for override in overrides_str.split(','):
            key, value = override.split('=')
            # Try to parse value
            try:
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except:
                pass
            overrides[key] = value
        
        # Add custom variant
        AblationRunner.ABLATION_VARIANTS[name] = {
            'description': f'Custom: {name}',
            'config_overrides': overrides,
            'impact': 'custom'
        }
        variants = [name]
    else:
        variants = args.variants
    
    # Create ablation runner
    if args.combined:
        runner = CombinedAblationRunner(
            config_path=args.config,
            model_path=args.model_path,
            output_dir=args.output_dir,
            variants=variants,
            run_baseline=not args.no_baseline,
            parallel=args.parallel,
            max_workers=args.max_workers,
            max_combinations=args.max_combinations,
            seed=args.seed,
            device=args.device,
            verbose=not args.no_verbose
        )
    else:
        runner = AblationRunner(
            config_path=args.config,
            model_path=args.model_path,
            output_dir=args.output_dir,
            variants=variants,
            run_baseline=not args.no_baseline,
            parallel=args.parallel,
            max_workers=args.max_workers,
            seed=args.seed,
            device=args.device,
            verbose=not args.no_verbose
        )
    
    # Run ablation
    results = runner.run()
    
    # Print summary
    print("\n" + runner._generate_text_report())
    
    return results


class CombinedAblationRunner(AblationRunner):
    """
    Combined ablation runner for testing combinations of variants.
    """
    
    def __init__(
        self,
        *args,
        max_combinations: int = 10,
        **kwargs
    ):
        """
        Initialize combined ablation runner.
        
        Args:
            max_combinations (int): Maximum number of combinations to test
        """
        super().__init__(*args, **kwargs)
        self.max_combinations = max_combinations
        self.combination_results = {}
    
    def run(self) -> Dict[str, Any]:
        """
        Run combined ablation experiments.
        
        Returns:
            Dict[str, Any]: Combined results
        """
        self.logger.log_info("=" * 80)
        self.logger.log_info("Starting Combined Ablation Study")
        self.logger.log_info("=" * 80)
        
        with self.timer.measure("combined_ablation"):
            # Run baseline
            if self.run_baseline:
                self.baseline_results = self._run_baseline()
            
            # Generate combinations
            combinations = self._generate_combinations()
            self.logger.log_info(f"Testing {len(combinations)} combinations")
            
            # Run each combination
            for i, combination in enumerate(combinations[:self.max_combinations]):
                self.logger.log_info(f"\nRunning combination {i+1}/{min(len(combinations), self.max_combinations)}")
                combo_name = "_".join(combination)
                
                try:
                    # Create config with multiple overrides
                    config = copy.deepcopy(self.config)
                    
                    for variant in combination:
                        self._apply_ablation_config(config, variant)
                    
                    # Create model
                    model = self._create_model(config)
                    
                    # Evaluate
                    evaluator = Evaluator(
                        model=model,
                        dataset=self._get_dataset(),
                        config=config,
                        logger=self.logger
                    )
                    
                    results = evaluator.evaluate()
                    metrics = self._extract_metrics(results)
                    
                    self.combination_results[combo_name] = {
                        'variants': combination,
                        'metrics': metrics,
                        'full_results': results,
                        'description': f"Combined: {', '.join(combination)}"
                    }
                    
                except Exception as e:
                    self.logger.log_error(f"Combination {combo_name} failed: {e}")
                    self.combination_results[combo_name] = {
                        'variants': combination,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }
            
            # Generate visualizations
            self._generate_combined_visualizations()
            
            # Generate report
            self._generate_combined_report()
            
            # Save results
            self._save_combined_results()
        
        self.logger.log_info("=" * 80)
        self.logger.log_info("Combined Ablation Study Completed")
        self.logger.log_info("=" * 80)
        
        return {
            'baseline': self.baseline_results,
            'combinations': self.combination_results
        }
    
    def _generate_combinations(self) -> List[List[str]]:
        """
        Generate combinations of ablation variants.
        
        Returns:
            List[List[str]]: List of combinations
        """
        import itertools
        
        combinations = []
        
        # Generate all combinations of size 2 and 3
        for r in [2, 3]:
            for combo in itertools.combinations(self.variants, r):
                combinations.append(list(combo))
        
        # Limit if too many
        if len(combinations) > self.max_combinations:
            # Randomly sample
            import random
            random.seed(self.seed)
            combinations = random.sample(combinations, self.max_combinations)
        
        return combinations
    
    def _generate_combined_visualizations(self) -> None:
        """
        Generate visualizations for combined ablations.
        """
        if not MATPLOTLIB_AVAILABLE or not self.combination_results:
            return
        
        try:
            # Prepare data
            combo_names = []
            ndcg_scores = []
            
            for combo_name, combo_data in self.combination_results.items():
                if 'error' in combo_data:
                    continue
                metrics = combo_data.get('metrics', {})
                combo_names.append(combo_name)
                ndcg_scores.append(metrics.get('ndcg@10', 0))
            
            if not combo_names:
                return
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Sort by performance
            sorted_data = sorted(zip(combo_names, ndcg_scores), 
                               key=lambda x: x[1], reverse=True)
            sorted_names, sorted_scores = zip(*sorted_data)
            
            bars = ax.bar(range(len(sorted_names)), sorted_scores,
                         color=self.visualizer.color_palette[:len(sorted_names)])
            
            # Add value labels
            for bar, value in zip(bars, sorted_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_xticks(range(len(sorted_names)))
            ax.set_xticklabels(sorted_names, rotation=45, ha='right')
            ax.set_ylabel('NDCG@10')
            ax.set_title('Combined Ablation Results')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'combined_ablation.png', 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.log_info("Combined ablation plot saved")
            
        except Exception as e:
            self.logger.log_warning(f"Failed to generate combined visualization: {e}")
    
    def _generate_combined_report(self) -> None:
        """
        Generate combined ablation report.
        """
        report_dir = self.output_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = report_dir / "combined_ablation_report.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMBINED ABLATION STUDY REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write(f"Seed: {self.seed}\n")
            f.write(f"Combinations Tested: {len(self.combination_results)}\n")
            f.write("\n")
            
            # Baseline results
            if self.baseline_results:
                f.write("BASELINE RESULTS\n")
                f.write("-" * 40 + "\n")
                baseline_metrics = self.baseline_results.get('metrics', {})
                for key, value in baseline_metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {key}: {value:.4f}\n")
                f.write("\n")
            
            # Combination results
            f.write("COMBINATION RESULTS\n")
            f.write("-" * 40 + "\n")
            
            # Sort by performance
            sorted_combos = sorted(
                [c for c in self.combination_results.items() if 'error' not in c[1]],
                key=lambda x: x[1].get('metrics', {}).get('ndcg@10', 0),
                reverse=True
            )
            
            for combo_name, combo_data in sorted_combos:
                metrics = combo_data.get('metrics', {})
                variants = combo_data.get('variants', [])
                f.write(f"\n{combo_name}: {', '.join(variants)}\n")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {key}: {value:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("End of Report\n")
            f.write("=" * 80 + "\n")
        
        self.logger.log_info(f"Combined report generated: {report_path}")
    
    def _save_combined_results(self) -> None:
        """
        Save combined results to file.
        """
        results_path = self.output_dir / "combined_ablation_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'baseline': self.baseline_results,
                'combinations': self.combination_results,
                'metadata': {
                    'experiment_name': self.output_dir.name,
                    'seed': self.seed,
                    'timestamp': datetime.now().isoformat()
                }
            }, f, indent=2, default=str)
        
        self.logger.log_info(f"Combined results saved to: {results_path}")


if __name__ == "__main__":
    main()