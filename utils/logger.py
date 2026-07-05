"""
utils/logger.py

Comprehensive logging module for H-GRAGrecsys with support for:
- Multi-level logging (INFO, WARNING, ERROR, DEBUG)
- Metrics logging with step tracking
- Experiment tracking and visualization preparation
- JSON serialization for structured logs
- File and console output with rotation
"""

import os
import sys
import json
import yaml
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import numpy as np
import pandas as pd
from collections import defaultdict

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration utilities
# from utils.config_loader import ConfigLoader

class Logger:
    """
    Main logging class for H-GRAGrecsys with metrics tracking and experiment logging.
    
    Features:
    - Structured logging with JSON support
    - Metrics aggregation and tracking
    - Experiment metadata management
    - Automatic log rotation
    - Console and file output
    """
    
    def __init__(
        self,
        log_dir: str,
        name: str = "h_gragrecsys",
        config_path: Optional[str] = None,
        verbose: bool = True,
        max_bytes: int = 10485760,  # 10MB
        backup_count: int = 5
    ):
        """
        Initialize the logger.
        
        Args:
            log_dir (str): Directory to save log files
            name (str): Name of the logger instance
            config_path (str, optional): Path to config file for log settings
            verbose (bool): Whether to print to console
            max_bytes (int): Maximum size of log file before rotation
            backup_count (int): Number of backup files to keep
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.name = name
        self.verbose = verbose
        
        # Load configuration if provided
        self.config = {}
        if config_path and Path(config_path).exists():
            loader = ConfigLoader(config_path)
            self.config = loader.load_config()
            # Override with config values if present
            log_config = self.config.get('logging', {})
            max_bytes = log_config.get('max_bytes', max_bytes)
            backup_count = log_config.get('backup_count', backup_count)
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup logging
        self.logger = self._setup_logger(max_bytes, backup_count)
        
        # Initialize metrics storage
        self.metrics_history = defaultdict(list)
        self.current_metrics = {}
        self.experiment_metadata = {}
        
        # Log initialization
        self.log_info(f"Logger initialized at {self.timestamp}")
        self.log_info(f"Log directory: {self.log_dir}")
    
    def _setup_logger(self, max_bytes: int, backup_count: int) -> logging.Logger:
        """
        Set up the underlying Python logger with handlers.
        
        Args:
            max_bytes (int): Maximum size of log file before rotation
            backup_count (int): Number of backup files to keep
            
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplication
        if logger.handlers:
            logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with rotation
        log_file = self.log_dir / f"{self.name}_{self.timestamp}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Separate error log file
        error_log_file = self.log_dir / f"{self.name}_error_{self.timestamp}.log"
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
        
        # Console handler if verbose
        if self.verbose:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(simple_formatter)
            logger.addHandler(console_handler)
        
        # JSON log handler for structured logging
        json_log_file = self.log_dir / f"{self.name}_structured_{self.timestamp}.log"
        json_handler = logging.FileHandler(json_log_file, encoding='utf-8')
        json_handler.setLevel(logging.INFO)
        json_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(json_handler)
        
        return logger
    
    def log_info(self, message: str) -> None:
        """
        Log an info message.
        
        Args:
            message (str): Message to log
        """
        self.logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """
        Log a warning message.
        
        Args:
            message (str): Message to log
        """
        self.logger.warning(message)
    
    def log_error(self, message: str, exc_info: bool = False) -> None:
        """
        Log an error message.
        
        Args:
            message (str): Message to log
            exc_info (bool): Whether to include exception info
        """
        self.logger.error(message, exc_info=exc_info)
    
    def log_debug(self, message: str) -> None:
        """
        Log a debug message.
        
        Args:
            message (str): Message to log
        """
        self.logger.debug(message)
    
    def log_structured(
        self,
        event_type: str,
        data: Dict[str, Any],
        level: str = "INFO"
    ) -> None:
        """
        Log structured data in JSON format.
        
        Args:
            event_type (str): Type of event (e.g., 'training_step', 'evaluation')
            data (Dict[str, Any]): Data to log
            level (str): Log level ('INFO', 'WARNING', 'ERROR', 'DEBUG')
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'experiment_id': self.experiment_metadata.get('experiment_id', 'default'),
            'data': data
        }
        
        json_str = json.dumps(log_entry, default=str)
        
        # Log to appropriate level
        if level.upper() == 'INFO':
            self.logger.info(json_str)
        elif level.upper() == 'WARNING':
            self.logger.warning(json_str)
        elif level.upper() == 'ERROR':
            self.logger.error(json_str)
        else:
            self.logger.debug(json_str)
    
    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int, List[float]]],
        step: Optional[int] = None,
        phase: Optional[str] = None
    ) -> None:
        """
        Log training/evaluation metrics with step tracking.
        
        Args:
            metrics (Dict[str, Union[float, int, List[float]]]): Metrics to log
            step (int, optional): Current step/epoch
            phase (str, optional): Training phase ('phase1', 'phase2', 'phase3')
        
        Example:
            logger.log_metrics(
                {'loss': 0.5, 'accuracy': 0.85, 'ndcg@10': 0.42},
                step=100,
                phase='phase2'
            )
        """
        # Store in history
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.metrics_history[key].append(value)
            elif isinstance(value, list):
                self.metrics_history[key].extend(value)
            else:
                self.metrics_history[key].append(value)
        
        # Store current metrics
        self.current_metrics.update(metrics)
        
        # Prepare structured log entry
        log_data = {
            'metrics': metrics,
            'step': step,
            'phase': phase,
            'experiment_metadata': self.experiment_metadata
        }
        
        self.log_structured('metrics', log_data)
        
        # Also log as simple message for readability
        metrics_str = ', '.join([f"{k}: {v}" for k, v in metrics.items() if isinstance(v, (int, float))])
        phase_str = f" [{phase}]" if phase else ""
        step_str = f" at step {step}" if step is not None else ""
        self.log_info(f"Metrics{phase_str}{step_str}: {metrics_str}")
    
    def log_agent_memory(
        self,
        agent_id: str,
        agent_type: str,
        memory_type: str,
        memory_content: Dict[str, Any],
        consistency_score: Optional[float] = None
    ) -> None:
        """
        Log agent memory updates and consistency scores.
        
        Args:
            agent_id (str): ID of the agent
            agent_type (str): Type of agent ('user' or 'item')
            memory_type (str): Type of memory ('intrinsic', 'collaborative', 'interaction')
            memory_content (Dict[str, Any]): Memory content to log
            consistency_score (float, optional): Consistency score of the memory
        """
        log_data = {
            'agent_id': agent_id,
            'agent_type': agent_type,
            'memory_type': memory_type,
            'memory_content': memory_content,
            'consistency_score': consistency_score,
            'timestamp': datetime.now().isoformat()
        }
        
        self.log_structured('agent_memory', log_data)
        
        # Also log summary
        summary = f"Agent {agent_id} ({agent_type}) {memory_type} memory updated"
        if consistency_score is not None:
            summary += f" [consistency: {consistency_score:.4f}]"
        self.log_debug(summary)
    
    def log_training_phase(
        self,
        phase: str,
        metrics: Dict[str, Any],
        epoch: int,
        total_epochs: int
    ) -> None:
        """
        Log training phase progress.
        
        Args:
            phase (str): Training phase ('phase1', 'phase2', 'phase3')
            metrics (Dict[str, Any]): Training metrics
            epoch (int): Current epoch
            total_epochs (int): Total epochs
        """
        log_data = {
            'phase': phase,
            'epoch': epoch,
            'total_epochs': total_epochs,
            'progress': epoch / total_epochs,
            'metrics': metrics
        }
        
        self.log_structured('training_phase', log_data)
        
        # Log progress message
        progress_pct = (epoch / total_epochs) * 100
        metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))])
        self.log_info(f"Phase {phase} - Epoch {epoch}/{total_epochs} ({progress_pct:.1f}%) - {metrics_str}")
    
    def log_experiment_start(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        **kwargs
    ) -> None:
        """
        Log the start of an experiment with metadata.
        
        Args:
            experiment_name (str): Name of the experiment
            config (Dict[str, Any]): Configuration used for the experiment
            **kwargs: Additional metadata
        """
        self.experiment_metadata = {
            'experiment_name': experiment_name,
            'experiment_id': f"{experiment_name}_{self.timestamp}",
            'start_time': datetime.now().isoformat(),
            'config': config,
            **kwargs
        }
        
        self.log_structured('experiment_start', self.experiment_metadata)
        self.log_info(f"Started experiment: {experiment_name} (ID: {self.experiment_metadata['experiment_id']})")
    
    def log_experiment_end(self, final_metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Log the end of an experiment with final metrics.
        
        Args:
            final_metrics (Dict[str, Any], optional): Final experiment metrics
        """
        end_data = {
            'end_time': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - datetime.fromisoformat(
                self.experiment_metadata.get('start_time', datetime.now().isoformat())
            )).total_seconds()
        }
        
        if final_metrics:
            end_data['final_metrics'] = final_metrics
        
        self.log_structured('experiment_end', end_data)
        self.log_info(f"Experiment ended. Duration: {end_data['duration_seconds']:.2f}s")
        
        if final_metrics:
            metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in final_metrics.items() if isinstance(v, (int, float))])
            self.log_info(f"Final metrics: {metrics_str}")
    
    def log_graph_stats(
        self,
        graph_stats: Dict[str, Any],
        step: Optional[int] = None
    ) -> None:
        """
        Log graph statistics.
        
        Args:
            graph_stats (Dict[str, Any]): Graph statistics
            step (int, optional): Current step
        
        Example:
            logger.log_graph_stats({
                'num_nodes': 1000,
                'num_edges': 5000,
                'avg_degree': 10.0
            }, step=100)
        """
        log_data = {
            'graph_stats': graph_stats,
            'step': step
        }
        
        self.log_structured('graph_stats', log_data)
        
        # Log summary
        stats_str = ', '.join([f"{k}: {v}" for k, v in graph_stats.items()])
        self.log_info(f"Graph statistics: {stats_str}")
    
    def log_distillation(
        self,
        teacher_outputs: Dict[str, Any],
        student_outputs: Dict[str, Any],
        loss_components: Dict[str, float]
    ) -> None:
        """
        Log distillation training details.
        
        Args:
            teacher_outputs (Dict[str, Any]): Teacher model outputs
            student_outputs (Dict[str, Any]): Student model outputs
            loss_components (Dict[str, float]): Individual loss components
        """
        log_data = {
            'teacher_outputs': teacher_outputs,
            'student_outputs': student_outputs,
            'loss_components': loss_components,
            'total_loss': sum(loss_components.values())
        }
        
        self.log_structured('distillation', log_data)
        
        # Log loss components
        loss_str = ', '.join([f"{k}: {v:.4f}" for k, v in loss_components.items()])
        self.log_debug(f"Distillation losses: {loss_str}")
    
    def log_gating_decision(
        self,
        node_id: str,
        gate_score: float,
        decision: str,
        confidence: float,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log adaptive gating decisions.
        
        Args:
            node_id (str): Node ID
            gate_score (float): Gate score
            decision (str): Routing decision ('gnn', 'llm', 'hybrid')
            confidence (float): Confidence score
            context (Dict[str, Any], optional): Additional context
        """
        log_data = {
            'node_id': node_id,
            'gate_score': gate_score,
            'decision': decision,
            'confidence': confidence,
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.log_structured('gating_decision', log_data)
        
        # Log summary
        self.log_debug(f"Gate decision for {node_id}: {decision} (score: {gate_score:.4f}, conf: {confidence:.4f})")
    
    def log_evaluation_results(
        self,
        metrics: Dict[str, float],
        dataset_name: str,
        split: str = 'test'
    ) -> None:
        """
        Log evaluation results.
        
        Args:
            metrics (Dict[str, float]): Evaluation metrics
            dataset_name (str): Name of the dataset
            split (str): Data split ('train', 'val', 'test')
        """
        log_data = {
            'dataset': dataset_name,
            'split': split,
            'metrics': metrics
        }
        
        self.log_structured('evaluation', log_data)
        
        # Log summary
        metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.log_info(f"Evaluation results on {dataset_name} ({split}): {metrics_str}")
    
    def log_cold_start_results(
        self,
        user_id: str,
        num_interactions: int,
        metrics: Dict[str, float]
    ) -> None:
        """
        Log cold-start experiment results.
        
        Args:
            user_id (str): User ID
            num_interactions (int): Number of interactions
            metrics (Dict[str, float]): Performance metrics
        """
        log_data = {
            'user_id': user_id,
            'num_interactions': num_interactions,
            'metrics': metrics
        }
        
        self.log_structured('cold_start', log_data)
        
        # Log summary
        metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.log_info(f"Cold-start for user {user_id} ({num_interactions} interactions): {metrics_str}")
    
    def log_ablation_result(
        self,
        ablation_name: str,
        metrics: Dict[str, float],
        delta_from_baseline: Dict[str, float]
    ) -> None:
        """
        Log ablation study results.
        
        Args:
            ablation_name (str): Name of the ablation variant
            metrics (Dict[str, float]): Performance metrics
            delta_from_baseline (Dict[str, float]): Difference from baseline
        """
        log_data = {
            'ablation_name': ablation_name,
            'metrics': metrics,
            'delta_from_baseline': delta_from_baseline
        }
        
        self.log_structured('ablation', log_data)
        
        # Log summary
        metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        delta_str = ', '.join([f"{k}: {v:+.4f}" for k, v in delta_from_baseline.items()])
        self.log_info(f"Ablation {ablation_name}: {metrics_str} (Δ: {delta_str})")
    
    def get_metrics_history(
        self,
        metric_name: Optional[str] = None
    ) -> Union[List[float], Dict[str, List[float]]]:
        """
        Get stored metrics history.
        
        Args:
            metric_name (str, optional): Specific metric to retrieve
            
        Returns:
            Union[List[float], Dict[str, List[float]]]: Metrics history
        
        Example:
            # Get all metrics
            all_metrics = logger.get_metrics_history()
            # Get specific metric
            loss_history = logger.get_metrics_history('loss')
        """
        if metric_name is not None:
            return self.metrics_history.get(metric_name, [])
        return dict(self.metrics_history)
    
    def save_metrics_csv(self, filename: Optional[str] = None) -> str:
        """
        Save metrics history to CSV file.
        
        Args:
            filename (str, optional): Output filename
            
        Returns:
            str: Path to the saved file
        """
        if not self.metrics_history:
            self.log_warning("No metrics to save")
            return ""
        
        if filename is None:
            filename = f"metrics_{self.timestamp}.csv"
        
        filepath = self.log_dir / filename
        
        # Convert history to DataFrame
        max_len = max(len(v) for v in self.metrics_history.values())
        data = {}
        for key, values in self.metrics_history.items():
            # Pad with NaN if needed
            padded_values = values + [np.nan] * (max_len - len(values))
            data[key] = padded_values
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        self.log_info(f"Metrics saved to {filepath}")
        return str(filepath)
    
    def save_experiment_summary(self) -> str:
        """
        Save experiment summary to JSON file.
        
        Returns:
            str: Path to the saved file
        """
        summary = {
            'experiment_metadata': self.experiment_metadata,
            'timestamp': self.timestamp,
            'metrics_summary': {
                key: {
                    'mean': np.mean(values) if values else None,
                    'std': np.std(values) if values else None,
                    'min': np.min(values) if values else None,
                    'max': np.max(values) if values else None,
                    'count': len(values)
                }
                for key, values in self.metrics_history.items()
            },
            'current_metrics': self.current_metrics
        }
        
        filename = f"experiment_summary_{self.timestamp}.json"
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.log_info(f"Experiment summary saved to {filepath}")
        return str(filepath)
    
    def get_log_paths(self) -> Dict[str, str]:
        """
        Get paths to all log files.
        
        Returns:
            Dict[str, str]: Dictionary of log file paths
        """
        return {
            'log_dir': str(self.log_dir),
            'main_log': str(self.log_dir / f"{self.name}_{self.timestamp}.log"),
            'error_log': str(self.log_dir / f"{self.name}_error_{self.timestamp}.log"),
            'structured_log': str(self.log_dir / f"{self.name}_structured_{self.timestamp}.log"),
            'metrics_csv': str(self.log_dir / f"metrics_{self.timestamp}.csv"),
            'summary_json': str(self.log_dir / f"experiment_summary_{self.timestamp}.json")
        }
    
    def close(self) -> None:
        """
        Close the logger and save final summaries.
        """
        # Save final metrics if any
        if self.metrics_history:
            self.save_metrics_csv()
            self.save_experiment_summary()
        
        # Close handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        
        self.log_info("Logger closed")


# Convenience function for quick logger creation
def get_logger(
    log_dir: str = "logs",
    name: str = "h_gragrecsys",
    config_path: Optional[str] = None,
    verbose: bool = True
) -> Logger:
    """
    Factory function to create and return a Logger instance.
    
    Args:
        log_dir (str): Directory to save log files
        name (str): Name of the logger
        config_path (str, optional): Path to config file
        verbose (bool): Whether to print to console
        
    Returns:
        Logger: Configured logger instance
    
    Example:
        logger = get_logger(
            log_dir="experiments/phase1_logs",
            name="phase1_bootstrap",
            config_path="config/default_config.yaml"
        )
    """
    return Logger(
        log_dir=log_dir,
        name=name,
        config_path=config_path,
        verbose=verbose
    )


# For testing the logger
if __name__ == "__main__":
    # Test the logger
    import tempfile
    import time
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing logger in: {temp_dir}")
        
        # Create logger
        logger = get_logger(
            log_dir=temp_dir,
            name="test_logger",
            verbose=True
        )
        
        # Log experiment start
        logger.log_experiment_start(
            experiment_name="TestExperiment",
            config={"param1": 0.5, "param2": "test"},
            dataset="Amazon_Books"
        )
        
        # Log some metrics
        for step in range(10):
            metrics = {
                'loss': 1.0 / (step + 1),
                'accuracy': 0.5 + step * 0.05,
                'ndcg@10': 0.3 + step * 0.03
            }
            logger.log_metrics(metrics, step=step, phase="phase1")
            
            # Log some graph stats
            if step % 3 == 0:
                logger.log_graph_stats({
                    'num_nodes': 100 + step * 10,
                    'num_edges': 500 + step * 50,
                    'avg_degree': 5.0 + step * 0.5
                }, step=step)
            
            time.sleep(0.1)
        
        # Log gating decisions
        logger.log_gating_decision(
            node_id="user_123",
            gate_score=0.75,
            decision="llm",
            confidence=0.82,
            context={"num_interactions": 15}
        )
        
        # Log evaluation results
        logger.log_evaluation_results(
            metrics={'ndcg@1': 0.45, 'ndcg@5': 0.62, 'ndcg@10': 0.71},
            dataset_name="Amazon_Books_Test",
            split="test"
        )
        
        # Log experiment end
        logger.log_experiment_end(
            final_metrics={'test_accuracy': 0.92, 'test_ndcg@10': 0.71}
        )
        
        # Get log paths
        log_paths = logger.get_log_paths()
        print("\nLog files created:")
        for name, path in log_paths.items():
            if Path(path).exists():
                size = Path(path).stat().st_size
                print(f"  {name}: {path} ({size} bytes)")
        
        # Get metrics history
        history = logger.get_metrics_history()
        print(f"\nMetrics tracked: {list(history.keys())}")
        print(f"Loss samples: {len(history.get('loss', []))}")
        
        # Save metrics
        csv_path = logger.save_metrics_csv()
        print(f"\nMetrics saved to: {csv_path}")
        
        # Close logger
        logger.close()
        
        print("\nLogger test completed successfully!")