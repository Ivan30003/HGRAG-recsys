"""
utils/visualization.py

Comprehensive visualization module for H-GRAGrecsys with support for:
- Training metrics visualization (loss, accuracy, NDCG)
- Memory consistency plots
- Quality-cost tradeoff analysis
- Ablation study results
- Gate sensitivity analysis
- Cold-start performance visualization
- Graph visualization (networkx, plotly)
- Confusion matrices and heatmaps
- Performance comparison charts
- Interactive dashboards
"""

import os
import sys
import json
import math
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple, Callable
from datetime import datetime
import warnings

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.logger import get_logger
from utils.config_loader import ConfigLoader
from utils.timer import Timer, global_timer

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import cm
    from matplotlib.ticker import MaxNLocator
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib/seaborn not available. Install with: pip install matplotlib seaborn")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("plotly not available. Install with: pip install plotly")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("networkx not available. Install with: pip install networkx")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("pandas not available. Install with: pip install pandas")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("numpy not available. Install with: pip install numpy")


class Visualizer:
    """
    Comprehensive visualization class for H-GRAGrecsys.
    
    Features:
    - Training and evaluation metrics visualization
    - Quality-cost tradeoff analysis
    - Ablation study visualization
    - Gate sensitivity analysis
    - Cold-start performance visualization
    - Graph visualization
    - Interactive dashboards
    - Multiple output formats (PNG, PDF, HTML)
    - Customizable themes and styles
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        style: str = 'seaborn-v0_8-darkgrid',
        interactive: bool = False,
        dpi: int = 300,
        figsize: Tuple[int, int] = (12, 8),
        logger: Optional['Logger'] = None
    ):
        """
        Initialize the Visualizer.
        
        Args:
            config_path (str, Path, optional): Path to configuration file
            output_dir (str, Path, optional): Directory to save visualizations
            style (str): Matplotlib style to use
            interactive (bool): Whether to use interactive plotly plots
            dpi (int): DPI for saved figures
            figsize (Tuple[int, int]): Default figure size
            logger (Logger, optional): Logger instance
        
        Example:
            visualizer = Visualizer(
                config_path='config/default_config.yaml',
                output_dir='plots/experiment1',
                style='seaborn-v0_8-darkgrid'
            )
        """
        self.output_dir = Path(output_dir) if output_dir else Path("plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.style = style
        self.interactive = interactive
        self.dpi = dpi
        self.figsize = figsize
        
        # Setup logger
        if logger is None:
            self.logger = get_logger(
                log_dir="logs/visualizer",
                name="visualizer",
                verbose=True
            )
        else:
            self.logger = logger
        
        # Load configuration
        self.config = {}
        if config_path:
            loader = ConfigLoader(config_path)
            self.config = loader.load_config()
            # Update settings from config
            viz_config = self.config.get('visualization', {})
            self.output_dir = Path(viz_config.get('output_dir', self.output_dir))
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.dpi = viz_config.get('dpi', self.dpi)
            self.style = viz_config.get('style', self.style)
            self.figsize = tuple(viz_config.get('figsize', self.figsize))
        
        # Setup matplotlib
        if MATPLOTLIB_AVAILABLE:
            self._setup_matplotlib()
        
        # Setup plotly
        if PLOTLY_AVAILABLE:
            self._setup_plotly()
        
        # Color palettes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#00B4D8',
            'warning': '#F18F01',
            'danger': '#C73E1D',
            'info': '#6C7A89',
            'light': '#F0F3F5',
            'dark': '#1A1A2E',
            'blue': '#2E86AB',
            'green': '#00B4D8',
            'red': '#C73E1D',
            'orange': '#F18F01',
            'purple': '#A23B72',
            'pink': '#D64C9A',
            'teal': '#0FAB9C',
            'gray': '#6C7A89'
        }
        
        self.color_palette = [
            '#2E86AB', '#A23B72', '#00B4D8', '#F18F01', '#C73E1D',
            '#0FAB9C', '#D64C9A', '#6C7A89', '#1A1A2E', '#E8D5B7'
        ]
        
        # Timer
        self.timer = Timer(name="visualizer", auto_start=False)
        
        self.logger.log_info(f"Visualizer initialized with output dir: {self.output_dir}")
    
    def _setup_matplotlib(self) -> None:
        """Setup matplotlib with custom styles."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        try:
            # Try to use the specified style
            if self.style in plt.style.available:
                plt.style.use(self.style)
            else:
                plt.style.use('seaborn-v0_8-darkgrid')
        except:
            plt.style.use('default')
        
        # Set default parameters
        plt.rcParams['figure.figsize'] = self.figsize
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['savefig.dpi'] = self.dpi
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        
        # Set color cycle
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=self.color_palette)
    
    def _setup_plotly(self) -> None:
        """Setup plotly with custom theme."""
        if not PLOTLY_AVAILABLE:
            return
        
        pio.templates.default = 'plotly_white'
        pio.renderers.default = 'browser' if self.interactive else 'png'
    
    def plot_training_metrics(
        self,
        metrics: Dict[str, List[float]],
        title: str = "Training Metrics",
        save_name: Optional[str] = None,
        show: bool = True,
        separate_plots: bool = False,
        y_labels: Optional[Dict[str, str]] = None,
        smoothing: int = 0
    ) -> Optional[str]:
        """
        Plot training metrics over epochs.
        
        Args:
            metrics (Dict[str, List[float]]): Dictionary of metric names to values
            title (str): Plot title
            save_name (str, optional): Filename to save plot
            show (bool): Whether to display the plot
            separate_plots (bool): Whether to plot each metric separately
            y_labels (Dict[str, str]): Custom y-axis labels
            smoothing (int): Window size for smoothing (0 = no smoothing)
            
        Returns:
            Optional[str]: Path to saved file if saved
            
        Example:
            metrics = {
                'loss': [0.5, 0.4, 0.3, 0.2, 0.1],
                'ndcg@10': [0.3, 0.4, 0.5, 0.6, 0.7],
                'hit_rate': [0.2, 0.3, 0.4, 0.5, 0.6]
            }
            visualizer.plot_training_metrics(metrics, title="Phase1 Training")
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.log_error("matplotlib not available")
            return None
        
        self.timer.start()
        
        # Apply smoothing if requested
        if smoothing > 0:
            metrics = self._smooth_metrics(metrics, smoothing)
        
        if separate_plots:
            # Create separate plots for each metric
            for metric_name, values in metrics.items():
                fig, ax = self._create_figure()
                ax.plot(values, linewidth=2, color=self.colors['primary'])
                ax.set_xlabel('Epoch')
                ax.set_ylabel(y_labels.get(metric_name, metric_name) if y_labels else metric_name)
                ax.set_title(f"{title} - {metric_name}")
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                
                save_path = None
                if save_name:
                    save_path = self._save_figure(fig, f"{save_name}_{metric_name}")
                if show:
                    plt.show()
                plt.close(fig)
        else:
            # Create single plot with multiple metrics
            fig, ax = self._create_figure()
            
            for i, (metric_name, values) in enumerate(metrics.items()):
                color = self.color_palette[i % len(self.color_palette)]
                ax.plot(values, linewidth=2, color=color, label=metric_name)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.set_title(title)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            save_path = None
            if save_name:
                save_path = self._save_figure(fig, save_name)
            if show:
                plt.show()
            plt.close(fig)
        
        elapsed = self.timer.stop()
        self.logger.log_info(f"Training metrics plot created in {elapsed:.2f}s")
        return save_path
    
    def plot_memory_consistency(
        self,
        consistency_values: List[float],
        steps: Optional[List[int]] = None,
        title: str = "Memory Consistency",
        save_name: Optional[str] = None,
        show: bool = True,
        threshold: Optional[float] = None
    ) -> Optional[str]:
        """
        Plot memory consistency over time.
        
        Args:
            consistency_values (List[float]): Consistency scores
            steps (List[int], optional): Step numbers (if None, uses 0..n-1)
            title (str): Plot title
            save_name (str, optional): Filename to save plot
            show (bool): Whether to display the plot
            threshold (float, optional): Threshold line to show
            
        Returns:
            Optional[str]: Path to saved file if saved
            
        Example:
            consistency_scores = [0.95, 0.92, 0.88, 0.85, 0.90]
            visualizer.plot_memory_consistency(
                consistency_scores,
                threshold=0.85,
                title="Agent Memory Consistency"
            )
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.log_error("matplotlib not available")
            return None
        
        self.timer.start()
        
        fig, ax = self._create_figure()
        
        if steps is None:
            steps = list(range(len(consistency_values)))
        
        ax.plot(steps, consistency_values, linewidth=2, color=self.colors['primary'], marker='o')
        ax.fill_between(steps, consistency_values, alpha=0.2, color=self.colors['primary'])
        
        # Add threshold line
        if threshold is not None:
            ax.axhline(y=threshold, color=self.colors['red'], linestyle='--', 
                      linewidth=2, alpha=0.7, label=f'Threshold: {threshold}')
            ax.legend()
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Consistency Score')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        save_path = None
        if save_name:
            save_path = self._save_figure(fig, save_name)
        if show:
            plt.show()
        plt.close(fig)
        
        elapsed = self.timer.stop()
        self.logger.log_info(f"Memory consistency plot created in {elapsed:.2f}s")
        return save_path
    
    def plot_quality_cost_tradeoff(
        self,
        results: Dict[str, Dict[str, float]],
        title: str = "Quality vs Cost Tradeoff",
        save_name: Optional[str] = None,
        show: bool = True,
        x_metric: str = 'cost',
        y_metric: str = 'quality',
        annotate: bool = True
    ) -> Optional[str]:
        """
        Plot quality-cost tradeoff for different methods.
        
        Args:
            results (Dict[str, Dict[str, float]]): Method names to metrics
            title (str): Plot title
            save_name (str, optional): Filename to save plot
            show (bool): Whether to display the plot
            x_metric (str): Metric name for x-axis
            y_metric (str): Metric name for y-axis
            annotate (bool): Whether to annotate points with method names
            
        Returns:
            Optional[str]: Path to saved file if saved
            
        Example:
            results = {
                'GNN': {'quality': 0.75, 'cost': 0.1},
                'LLM': {'quality': 0.92, 'cost': 0.8},
                'Hybrid': {'quality': 0.88, 'cost': 0.3}
            }
            visualizer.plot_quality_cost_tradeoff(
                results, 
                title="Method Comparison",
                x_metric='cost',
                y_metric='quality'
            )
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.log_error("matplotlib not available")
            return None
        
        self.timer.start()
        
        fig, ax = self._create_figure()
        
        # Extract data
        methods = list(results.keys())
        x_values = [results[m].get(x_metric, 0) for m in methods]
        y_values = [results[m].get(y_metric, 0) for m in methods]
        
        # Create scatter plot
        scatter = ax.scatter(x_values, y_values, s=200, c=self.color_palette[:len(methods)], 
                            alpha=0.7, edgecolors='black', linewidth=1)
        
        # Annotate points
        if annotate:
            for i, method in enumerate(methods):
                ax.annotate(method, (x_values[i], y_values[i]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold')
        
        # Add quadrant lines (optional)
        if len(x_values) > 1:
            ax.axhline(y=np.mean(y_values), color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=np.mean(x_values), color='gray', linestyle='--', alpha=0.5)
            
            # Add quadrant labels
            ax.text(ax.get_xlim()[1]*0.95, ax.get_ylim()[1]*0.95, 'High Quality, High Cost',
                   ha='right', va='top', style='italic', color='gray', fontsize=10)
            ax.text(ax.get_xlim()[1]*0.95, ax.get_ylim()[0]*1.05, 'Low Quality, High Cost',
                   ha='right', va='bottom', style='italic', color='gray', fontsize=10)
            ax.text(ax.get_xlim()[0]*1.05, ax.get_ylim()[1]*0.95, 'High Quality, Low Cost',
                   ha='left', va='top', style='italic', color='gray', fontsize=10)
            ax.text(ax.get_xlim()[0]*1.05, ax.get_ylim()[0]*1.05, 'Low Quality, Low Cost',
                   ha='left', va='bottom', style='italic', color='gray', fontsize=10)
        
        ax.set_xlabel(x_metric.replace('_', ' ').title())
        ax.set_ylabel(y_metric.replace('_', ' ').title())
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        save_path = None
        if save_name:
            save_path = self._save_figure(fig, save_name)
        if show:
            plt.show()
        plt.close(fig)
        
        elapsed = self.timer.stop()
        self.logger.log_info(f"Quality-cost tradeoff plot created in {elapsed:.2f}s")
        return save_path
    
    def plot_ablation_results(
        self,
        results: Dict[str, Dict[str, float]],
        baseline: Optional[str] = None,
        title: str = "Ablation Study Results",
        save_name: Optional[str] = None,
        show: bool = True,
        sort_by: Optional[str] = None
    ) -> Optional[str]:
        """
        Plot ablation study results as grouped bar chart.
        
        Args:
            results (Dict[str, Dict[str, float]]): Ablation name to metrics
            baseline (str, optional): Name of baseline for comparison
            title (str): Plot title
            save_name (str, optional): Filename to save plot
            show (bool): Whether to display the plot
            sort_by (str, optional): Metric to sort by
            
        Returns:
            Optional[str]: Path to saved file if saved
            
        Example:
            results = {
                'Full Model': {'ndcg@10': 0.72, 'hit_rate': 0.65},
                'No Memory': {'ndcg@10': 0.58, 'hit_rate': 0.50},
                'No RAG': {'ndcg@10': 0.62, 'hit_rate': 0.55},
                'No Gating': {'ndcg@10': 0.68, 'hit_rate': 0.60}
            }
            visualizer.plot_ablation_results(results, baseline='Full Model')
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.log_error("matplotlib not available")
            return None
        
        self.timer.start()
        
        # Get all metrics
        metrics_names = list(results.values())[0].keys() if results else []
        
        if not metrics_names:
            self.logger.log_warning("No metrics found in results")
            return None
        
        # Sort if requested
        ablation_names = list(results.keys())
        if sort_by and sort_by in metrics_names:
            ablation_names.sort(key=lambda x: results[x].get(sort_by, 0), reverse=True)
        
        # Prepare data
        x = np.arange(len(ablation_names))
        width = 0.8 / len(metrics_names)
        
        fig, ax = self._create_figure()
        
        # Plot bars
        for i, metric_name in enumerate(metrics_names):
            values = [results[name].get(metric_name, 0) for name in ablation_names]
            offset = (i - len(metrics_names)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=metric_name, 
                   color=self.color_palette[i % len(self.color_palette)],
                   alpha=0.8)
            
            # Add value labels
            for j, v in enumerate(values):
                ax.text(x[j] + offset, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Add baseline reference line
        if baseline and baseline in ablation_names:
            baseline_idx = ablation_names.index(baseline)
            ax.axvline(x=baseline_idx + 0.5, color='red', linestyle='--', alpha=0.5)
            ax.text(baseline_idx + 0.5, ax.get_ylim()[1]*0.95, 'Baseline', 
                   ha='center', va='top', color='red', rotation=90)
        
        ax.set_xlabel('Ablation Variant')
        ax.set_ylabel('Performance')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(ablation_names, rotation=45, ha='right')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        save_path = None
        if save_name:
            save_path = self._save_figure(fig, save_name)
        if show:
            plt.show()
        plt.close(fig)
        
        elapsed = self.timer.stop()
        self.logger.log_info(f"Ablation results plot created in {elapsed:.2f}s")
        return save_path
    
    def plot_gate_sensitivity(
        self,
        thresholds: List[float],
        metrics: Dict[str, List[float]],
        title: str = "Gate Sensitivity Analysis",
        save_name: Optional[str] = None,
        show: bool = True,
        optimal_threshold: Optional[float] = None
    ) -> Optional[str]:
        """
        Plot gate sensitivity analysis.
        
        Args:
            thresholds (List[float]): Threshold values
            metrics (Dict[str, List[float]]): Metric names to values at each threshold
            title (str): Plot title
            save_name (str, optional): Filename to save plot
            show (bool): Whether to display the plot
            optimal_threshold (float, optional): Optimal threshold to highlight
            
        Returns:
            Optional[str]: Path to saved file if saved
            
        Example:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            metrics = {
                'ndcg@10': [0.65, 0.68, 0.72, 0.70, 0.68, 0.65, 0.60, 0.55, 0.50],
                'cost': [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
            }
            visualizer.plot_gate_sensitivity(thresholds, metrics, optimal_threshold=0.3)
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.log_error("matplotlib not available")
            return None
        
        self.timer.start()
        
        fig, ax1 = self._create_figure()
        
        # Primary y-axis
        for i, (metric_name, values) in enumerate(metrics.items()):
            color = self.color_palette[i % len(self.color_palette)]
            ax1.plot(thresholds, values, linewidth=2, color=color, marker='o', label=metric_name)
        
        ax1.set_xlabel('Gate Threshold')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        
        # Add optimal threshold
        if optimal_threshold is not None:
            ax1.axvline(x=optimal_threshold, color='red', linestyle='--', 
                       linewidth=2, alpha=0.7, label=f'Optimal: {optimal_threshold}')
        
        ax1.legend(loc='best')
        ax1.set_title(title)
        
        save_path = None
        if save_name:
            save_path = self._save_figure(fig, save_name)
        if show:
            plt.show()
        plt.close(fig)
        
        elapsed = self.timer.stop()
        self.logger.log_info(f"Gate sensitivity plot created in {elapsed:.2f}s")
        return save_path
    
    def plot_cold_start_performance(
        self,
        results: Dict[int, Dict[str, float]],
        title: str = "Cold Start Performance",
        save_name: Optional[str] = None,
        show: bool = True
    ) -> Optional[str]:
        """
        Plot cold-start performance by interaction count.
        
        Args:
            results (Dict[int, Dict[str, float]]): Interaction count to metrics
            title (str): Plot title
            save_name (str, optional): Filename to save plot
            show (bool): Whether to display the plot
            
        Returns:
            Optional[str]: Path to saved file if saved
            
        Example:
            results = {
                0: {'ndcg@10': 0.15, 'hit_rate': 0.10},
                5: {'ndcg@10': 0.35, 'hit_rate': 0.30},
                10: {'ndcg@10': 0.50, 'hit_rate': 0.45},
                20: {'ndcg@10': 0.65, 'hit_rate': 0.60}
            }
            visualizer.plot_cold_start_performance(results)
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.log_error("matplotlib not available")
            return None
        
        self.timer.start()
        
        # Sort results by interaction count
        sorted_interactions = sorted(results.keys())
        metrics_names = list(results[sorted_interactions[0]].keys())
        
        fig, ax = self._create_figure()
        
        # Plot each metric
        for i, metric_name in enumerate(metrics_names):
            values = [results[interactions].get(metric_name, 0) for interactions in sorted_interactions]
            color = self.color_palette[i % len(self.color_palette)]
            ax.plot(sorted_interactions, values, linewidth=2, color=color, 
                   marker='o', label=metric_name)
        
        ax.set_xlabel('Number of Interactions')
        ax.set_ylabel('Performance')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log' if len(sorted_interactions) > 10 else 'linear')
        
        save_path = None
        if save_name:
            save_path = self._save_figure(fig, save_name)
        if show:
            plt.show()
        plt.close(fig)
        
        elapsed = self.timer.stop()
        self.logger.log_info(f"Cold-start plot created in {elapsed:.2f}s")
        return save_path
    
    def plot_graph(
        self,
        graph_data: Dict[str, Any],
        title: str = "Graph Visualization",
        save_name: Optional[str] = None,
        show: bool = True,
        node_colors: Optional[Dict[str, str]] = None,
        node_sizes: Optional[Dict[str, float]] = None,
        edge_weights: bool = False,
        layout: str = 'spring'
    ) -> Optional[str]:
        """
        Plot graph visualization.
        
        Args:
            graph_data (Dict[str, Any]): Graph data with nodes and edges
            title (str): Plot title
            save_name (str, optional): Filename to save plot
            show (bool): Whether to display the plot
            node_colors (Dict[str, str]): Node colors by node ID
            node_sizes (Dict[str, float]): Node sizes by node ID
            edge_weights (bool): Whether to show edge weights
            layout (str): Graph layout algorithm ('spring', 'circular', 'kamada_kawai')
            
        Returns:
            Optional[str]: Path to saved file if saved
            
        Example:
            graph_data = {
                'nodes': ['user_1', 'item_1', 'user_2', 'item_2'],
                'edges': [('user_1', 'item_1'), ('user_1', 'item_2'), ('user_2', 'item_1')],
                'types': {'user_1': 'user', 'item_1': 'item'}
            }
            visualizer.plot_graph(graph_data, title="User-Item Graph")
        """
        if not NETWORKX_AVAILABLE:
            self.logger.log_error("networkx not available")
            return None
        
        if not MATPLOTLIB_AVAILABLE:
            self.logger.log_error("matplotlib not available")
            return None
        
        self.timer.start()
        
        # Create networkx graph
        G = nx.Graph()
        
        # Add nodes
        nodes = graph_data.get('nodes', [])
        for node in nodes:
            G.add_node(node)
        
        # Add edges
        edges = graph_data.get('edges', [])
        for edge in edges:
            if len(edge) == 2:
                G.add_edge(edge[0], edge[1])
            elif len(edge) == 3:
                G.add_edge(edge[0], edge[1], weight=edge[2])
        
        # Get node types
        node_types = graph_data.get('types', {})
        
        fig, ax = self._create_figure()
        
        # Compute layout
        if layout == 'spring':
            pos = nx.spring_layout(G, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # Prepare node colors
        if node_colors is None:
            # Color by type
            type_colors = {}
            for node, node_type in node_types.items():
                if node_type not in type_colors:
                    type_colors[node_type] = self.color_palette[len(type_colors) % len(self.color_palette)]
            
            node_color_list = [type_colors.get(node_types.get(node, 'unknown'), self.colors['gray']) 
                              for node in G.nodes()]
        else:
            node_color_list = [node_colors.get(node, self.colors['gray']) for node in G.nodes()]
        
        # Prepare node sizes
        if node_sizes is None:
            node_sizes = {node: 500 for node in G.nodes()}
        
        node_size_list = [node_sizes.get(node, 500) for node in G.nodes()]
        
        # Draw graph
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color_list, 
                               node_size=node_size_list, alpha=0.8)
        
        # Draw edges
        if edge_weights:
            edge_weights_list = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, ax=ax, width=edge_weights_list, 
                                   edge_color=self.colors['gray'], alpha=0.5)
            nx.draw_networkx_edge_labels(G, pos, ax=ax, 
                                         edge_labels={(u, v): f'{G[u][v].get("weight", 1.0):.2f}' 
                                                     for u, v in G.edges()}, font_size=8)
        else:
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color=self.colors['gray'], alpha=0.5)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')
        
        ax.set_title(title)
        ax.axis('off')
        
        # Add legend for types
        if node_types and node_colors is None:
            type_legend = []
            for node_type, color in type_colors.items():
                patch = mpatches.Patch(color=color, label=node_type)
                type_legend.append(patch)
            ax.legend(handles=type_legend, loc='best')
        
        save_path = None
        if save_name:
            save_path = self._save_figure(fig, save_name)
        if show:
            plt.show()
        plt.close(fig)
        
        elapsed = self.timer.stop()
        self.logger.log_info(f"Graph plot created in {elapsed:.2f}s")
        return save_path
    
    def plot_confusion_matrix(
        self,
        cm: Union[List[List[float]], np.ndarray],
        labels: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        save_name: Optional[str] = None,
        show: bool = True,
        normalize: bool = False
    ) -> Optional[str]:
        """
        Plot confusion matrix as heatmap.
        
        Args:
            cm (Union[List[List[float]], np.ndarray]): Confusion matrix
            labels (List[str], optional): Class labels
            title (str): Plot title
            save_name (str, optional): Filename to save plot
            show (bool): Whether to display the plot
            normalize (bool): Whether to normalize the matrix
            
        Returns:
            Optional[str]: Path to saved file if saved
            
        Example:
            cm = [[10, 2, 1], [3, 8, 2], [1, 2, 9]]
            visualizer.plot_confusion_matrix(cm, labels=['A', 'B', 'C'])
        """
        if not MATPLOTLIB_AVAILABLE or not SEABORN_AVAILABLE:
            self.logger.log_error("matplotlib or seaborn not available")
            return None
        
        self.timer.start()
        
        cm_array = np.array(cm)
        
        if normalize:
            cm_array = cm_array.astype('float') / cm_array.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        fig, ax = self._create_figure()
        
        sns.heatmap(cm_array, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Normalized' if normalize else 'Count'})
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)
        
        save_path = None
        if save_name:
            save_path = self._save_figure(fig, save_name)
        if show:
            plt.show()
        plt.close(fig)
        
        elapsed = self.timer.stop()
        self.logger.log_info(f"Confusion matrix plot created in {elapsed:.2f}s")
        return save_path
    
    def plot_efficiency_metrics(
        self,
        metrics: Dict[str, Dict[str, float]],
        title: str = "Efficiency Metrics",
        save_name: Optional[str] = None,
        show: bool = True,
        metric_type: str = 'bar'  # 'bar' or 'radar'
    ) -> Optional[str]:
        """
        Plot efficiency metrics comparison.
        
        Args:
            metrics (Dict[str, Dict[str, float]]): Method names to efficiency metrics
            title (str): Plot title
            save_name (str, optional): Filename to save plot
            show (bool): Whether to display the plot
            metric_type (str): 'bar' or 'radar'
            
        Returns:
            Optional[str]: Path to saved file if saved
            
        Example:
            metrics = {
                'GNN': {'time': 0.1, 'memory': 512, 'cost': 0.1},
                'LLM': {'time': 0.5, 'memory': 1024, 'cost': 0.8},
                'Hybrid': {'time': 0.2, 'memory': 768, 'cost': 0.3}
            }
            visualizer.plot_efficiency_metrics(metrics)
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.log_error("matplotlib not available")
            return None
        
        self.timer.start()
        
        if metric_type == 'radar':
            save_path = self._plot_radar_chart(metrics, title, save_name, show)
        else:
            save_path = self._plot_efficiency_bar(metrics, title, save_name, show)
        
        elapsed = self.timer.stop()
        self.logger.log_info(f"Efficiency metrics plot created in {elapsed:.2f}s")
        return save_path
    
    def _plot_efficiency_bar(
        self,
        metrics: Dict[str, Dict[str, float]],
        title: str,
        save_name: Optional[str],
        show: bool
    ) -> Optional[str]:
        """Helper for efficiency bar chart."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        methods = list(metrics.keys())
        metric_names = list(metrics[methods[0]].keys())
        
        x = np.arange(len(methods))
        width = 0.8 / len(metric_names)
        
        fig, ax = self._create_figure()
        
        for i, metric_name in enumerate(metric_names):
            values = [metrics[m].get(metric_name, 0) for m in methods]
            offset = (i - len(metric_names)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=metric_name,
                   color=self.color_palette[i % len(self.color_palette)])
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        save_path = None
        if save_name:
            save_path = self._save_figure(fig, save_name)
        if show:
            plt.show()
        plt.close(fig)
        
        return save_path
    
    def _plot_radar_chart(
        self,
        metrics: Dict[str, Dict[str, float]],
        title: str,
        save_name: Optional[str],
        show: bool
    ) -> Optional[str]:
        """Helper for radar chart."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        methods = list(metrics.keys())
        metric_names = list(metrics[methods[0]].keys())
        
        # Normalize metrics
        normalized_metrics = {}
        for method in methods:
            normalized_metrics[method] = {}
            for metric in metric_names:
                values = [metrics[m].get(metric, 0) for m in methods]
                max_val = max(values) if max(values) > 0 else 1
                normalized_metrics[method][metric] = metrics[method].get(metric, 0) / max_val
        
        fig, ax = self._create_figure(subplot_kw={'projection': 'polar'})
        
        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, method in enumerate(methods):
            values = [normalized_metrics[method].get(m, 0) for m in metric_names]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=method, color=self.color_palette[i % len(self.color_palette)])
            ax.fill(angles, values, alpha=0.1, color=self.color_palette[i % len(self.color_palette)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1.1)
        ax.set_title(title, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        
        save_path = None
        if save_name:
            save_path = self._save_figure(fig, save_name)
        if show:
            plt.show()
        plt.close(fig)
        
        return save_path
    
    def create_dashboard(
        self,
        data: Dict[str, Any],
        title: str = "H-GRAGrecsys Dashboard",
        save_name: Optional[str] = None,
        show: bool = True
    ) -> Optional[str]:
        """
        Create interactive dashboard using plotly.
        
        Args:
            data (Dict[str, Any]): Dashboard data
            title (str): Dashboard title
            save_name (str, optional): Filename to save dashboard (HTML)
            show (bool): Whether to display the dashboard
            
        Returns:
            Optional[str]: Path to saved file if saved
            
        Example:
            data = {
                'training_metrics': {'loss': [...], 'ndcg': [...]},
                'ablation_results': {...},
                'gate_analysis': {'thresholds': [...], 'metrics': {...}}
            }
            visualizer.create_dashboard(data, title="Experiment Dashboard")
        """
        if not PLOTLY_AVAILABLE:
            self.logger.log_error("plotly not available")
            return None
        
        self.timer.start()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Metrics', 'Ablation Results',
                           'Gate Sensitivity', 'Quality-Cost Tradeoff'),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': True}, {'secondary_y': False}]]
        )
        
        # Plot training metrics
        if 'training_metrics' in data:
            for i, (metric_name, values) in enumerate(data['training_metrics'].items()):
                fig.add_trace(
                    go.Scatter(y=values, mode='lines+markers', name=metric_name),
                    row=1, col=1
                )
        
        # Plot ablation results
        if 'ablation_results' in data:
            ablation_data = data['ablation_results']
            ablation_names = list(ablation_data.keys())
            metrics_names = list(ablation_data[ablation_names[0]].keys())
            
            for metric in metrics_names:
                values = [ablation_data[name].get(metric, 0) for name in ablation_names]
                fig.add_trace(
                    go.Bar(x=ablation_names, y=values, name=metric),
                    row=1, col=2
                )
        
        # Plot gate sensitivity
        if 'gate_analysis' in data:
            gate_data = data['gate_analysis']
            thresholds = gate_data.get('thresholds', [])
            
            for metric_name, values in gate_data.get('metrics', {}).items():
                fig.add_trace(
                    go.Scatter(x=thresholds, y=values, mode='lines+markers', name=metric_name),
                    row=2, col=1
                )
        
        # Plot quality-cost tradeoff
        if 'quality_cost' in data:
            qc_data = data['quality_cost']
            methods = list(qc_data.keys())
            quality = [qc_data[m].get('quality', 0) for m in methods]
            cost = [qc_data[m].get('cost', 0) for m in methods]
            
            fig.add_trace(
                go.Scatter(x=cost, y=quality, mode='markers+text',
                          text=methods, textposition="top center",
                          marker=dict(size=15)),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=title,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text='Epoch', row=1, col=1)
        fig.update_yaxes(title_text='Value', row=1, col=1)
        fig.update_xaxes(title_text='Method', row=1, col=2)
        fig.update_yaxes(title_text='Performance', row=1, col=2)
        fig.update_xaxes(title_text='Threshold', row=2, col=1)
        fig.update_yaxes(title_text='Value', row=2, col=1)
        fig.update_xaxes(title_text='Cost', row=2, col=2)
        fig.update_yaxes(title_text='Quality', row=2, col=2)
        
        save_path = None
        if save_name:
            save_path = self.output_dir / f"{save_name}.html"
            fig.write_html(str(save_path))
            self.logger.log_info(f"Dashboard saved to: {save_path}")
        if show:
            fig.show()
        
        elapsed = self.timer.stop()
        self.logger.log_info(f"Dashboard created in {elapsed:.2f}s")
        return save_path
    
    def _create_figure(self, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Create a matplotlib figure and axes."""
        if not MATPLOTLIB_AVAILABLE:
            return None, None
        
        fig, ax = plt.subplots(**kwargs)
        return fig, ax
    
    def _save_figure(self, fig: plt.Figure, name: str) -> str:
        """
        Save a figure to multiple formats.
        
        Args:
            fig (plt.Figure): Figure to save
            name (str): Base filename
            
        Returns:
            str: Path to saved file
        """
        if not MATPLOTLIB_AVAILABLE:
            return ""
        
        filepath = self.output_dir / name
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save in multiple formats
        formats = ['.png', '.pdf']
        for fmt in formats:
            fig.savefig(str(filepath) + fmt, dpi=self.dpi, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
        
        self.logger.log_debug(f"Figure saved to: {filepath}.png, {filepath}.pdf")
        return str(filepath) + '.png'
    
    def _smooth_metrics(
        self,
        metrics: Dict[str, List[float]],
        window: int
    ) -> Dict[str, List[float]]:
        """
        Smooth metrics using moving average.
        
        Args:
            metrics (Dict[str, List[float]]): Metrics to smooth
            window (int): Smoothing window size
            
        Returns:
            Dict[str, List[float]]: Smoothed metrics
        """
        smoothed = {}
        
        for name, values in metrics.items():
            if len(values) < window:
                smoothed[name] = values
                continue
            
            smoothed_values = []
            for i in range(len(values)):
                start = max(0, i - window // 2)
                end = min(len(values), i + window // 2 + 1)
                smoothed_values.append(np.mean(values[start:end]))
            
            smoothed[name] = smoothed_values
        
        return smoothed
    
    def __repr__(self) -> str:
        """String representation of the Visualizer."""
        return f"Visualizer(output_dir='{self.output_dir}', interactive={self.interactive})"


# Convenience functions
def create_visualizer(
    config_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    interactive: bool = False
) -> Visualizer:
    """
    Factory function to create a Visualizer instance.
    
    Args:
        config_path (str, Path, optional): Path to configuration file
        output_dir (str, Path, optional): Directory to save visualizations
        interactive (bool): Whether to use interactive plotly plots
        
    Returns:
        Visualizer: Configured Visualizer instance
    
    Example:
        visualizer = create_visualizer(
            config_path='config/default_config.yaml',
            output_dir='plots/experiment1'
        )
    """
    return Visualizer(
        config_path=config_path,
        output_dir=output_dir,
        interactive=interactive
    )


# Check if seaborn is available
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    warnings.warn("seaborn not available. Install with: pip install seaborn")


# For testing the visualizer
if __name__ == "__main__":
    import tempfile
    import time
    
    print("Testing Visualizer...")
    
    # Create test data
    test_metrics = {
        'loss': [0.5, 0.4, 0.35, 0.3, 0.25, 0.22, 0.2, 0.18],
        'ndcg@10': [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
        'hit_rate': [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
    }
    
    consistency_scores = [0.95, 0.93, 0.90, 0.88, 0.85, 0.87, 0.90, 0.92]
    
    ablation_results = {
        'Full Model': {'ndcg@10': 0.72, 'hit_rate': 0.65, 'recall': 0.60},
        'No Memory': {'ndcg@10': 0.58, 'hit_rate': 0.50, 'recall': 0.45},
        'No RAG': {'ndcg@10': 0.62, 'hit_rate': 0.55, 'recall': 0.50},
        'No Gating': {'ndcg@10': 0.68, 'hit_rate': 0.60, 'recall': 0.55},
        'No Distillation': {'ndcg@10': 0.65, 'hit_rate': 0.58, 'recall': 0.52}
    }
    
    gate_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    gate_metrics = {
        'ndcg@10': [0.65, 0.68, 0.72, 0.70, 0.68, 0.65, 0.60, 0.55, 0.50],
        'cost': [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
        'accuracy': [0.70, 0.73, 0.76, 0.75, 0.72, 0.70, 0.65, 0.60, 0.55]
    }
    
    cold_start_results = {
        0: {'ndcg@10': 0.15, 'hit_rate': 0.10},
        5: {'ndcg@10': 0.35, 'hit_rate': 0.30},
        10: {'ndcg@10': 0.50, 'hit_rate': 0.45},
        20: {'ndcg@10': 0.65, 'hit_rate': 0.60},
        50: {'ndcg@10': 0.75, 'hit_rate': 0.70},
        100: {'ndcg@10': 0.82, 'hit_rate': 0.78}
    }
    
    quality_cost_results = {
        'GNN': {'quality': 0.72, 'cost': 0.1, 'time': 0.05},
        'LLM': {'quality': 0.92, 'cost': 0.8, 'time': 0.5},
        'Hybrid': {'quality': 0.88, 'cost': 0.3, 'time': 0.15},
        'Ensemble': {'quality': 0.90, 'cost': 0.45, 'time': 0.25}
    }
    
    # Test with temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing visualizer in: {temp_dir}")
        
        visualizer = create_visualizer(
            output_dir=temp_dir,
            interactive=False
        )
        
        # Test training metrics
        visualizer.plot_training_metrics(
            test_metrics,
            title="Training Metrics Test",
            save_name="training_metrics",
            show=False
        )
        print("✓ Training metrics plot created")
        
        # Test memory consistency
        visualizer.plot_memory_consistency(
            consistency_scores,
            threshold=0.85,
            title="Memory Consistency Test",
            save_name="memory_consistency",
            show=False
        )
        print("✓ Memory consistency plot created")
        
        # Test ablation results
        visualizer.plot_ablation_results(
            ablation_results,
            baseline='Full Model',
            title="Ablation Study Test",
            save_name="ablation_results",
            show=False
        )
        print("✓ Ablation results plot created")
        
        # Test gate sensitivity
        visualizer.plot_gate_sensitivity(
            gate_thresholds,
            gate_metrics,
            optimal_threshold=0.3,
            title="Gate Sensitivity Test",
            save_name="gate_sensitivity",
            show=False
        )
        print("✓ Gate sensitivity plot created")
        
        # Test cold start performance
        visualizer.plot_cold_start_performance(
            cold_start_results,
            title="Cold Start Performance Test",
            save_name="cold_start",
            show=False
        )
        print("✓ Cold start plot created")
        
        # Test quality-cost tradeoff
        visualizer.plot_quality_cost_tradeoff(
            quality_cost_results,
            title="Quality-Cost Tradeoff Test",
            save_name="quality_cost",
            show=False
        )
        print("✓ Quality-cost tradeoff plot created")
        
        # Test efficiency metrics
        visualizer.plot_efficiency_metrics(
            quality_cost_results,
            title="Efficiency Metrics Test",
            save_name="efficiency",
            show=False
        )
        print("✓ Efficiency metrics plot created")
        
        # Test graph visualization (if networkx available)
        if NETWORKX_AVAILABLE:
            graph_data = {
                'nodes': ['user1', 'user2', 'user3', 'item1', 'item2', 'item3'],
                'edges': [('user1', 'item1'), ('user1', 'item2'), ('user2', 'item2'),
                         ('user3', 'item2'), ('user2', 'item3'), ('user3', 'item3')],
                'types': {'user1': 'user', 'user2': 'user', 'user3': 'user',
                         'item1': 'item', 'item2': 'item', 'item3': 'item'}
            }
            visualizer.plot_graph(
                graph_data,
                title="Graph Test",
                save_name="graph",
                show=False
            )
            print("✓ Graph plot created")
        
        # Test dashboard (if plotly available)
        if PLOTLY_AVAILABLE:
            dashboard_data = {
                'training_metrics': test_metrics,
                'ablation_results': ablation_results,
                'gate_analysis': {'thresholds': gate_thresholds, 'metrics': gate_metrics},
                'quality_cost': quality_cost_results
            }
            visualizer.create_dashboard(
                dashboard_data,
                title="H-GRAGrecsys Dashboard Test",
                save_name="dashboard",
                show=False
            )
            print("✓ Dashboard created")
        
        print(f"\nAll visualizations saved to: {temp_dir}")
        print(f"Files created: {list(Path(temp_dir).glob('*'))}")
    
    print("\nVisualizer tests completed successfully!")