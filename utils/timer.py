"""
utils/timer.py

Comprehensive timing and performance measurement module for H-GRAGrecsys with support for:
- High-precision timing with context managers
- Function decorators for automatic timing
- Performance profiling and statistics
- Nested timing support
- Timing report generation
- Memory usage tracking
- GPU timing support
- Progress estimation and ETA calculation
"""

import os
import sys
import time
import functools
import threading
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple, Callable, Generator
from contextlib import contextmanager
from collections import defaultdict
from datetime import datetime, timedelta
import inspect
import warnings

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.logger import get_logger
from utils.config_loader import ConfigLoader

# Try to import optional libraries
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class Timer:
    """
    Comprehensive timer class for performance measurement and profiling.
    
    Features:
    - High-precision timing (nanoseconds)
    - Context manager support
    - Function decorator for automatic timing
    - Nested timer support
    - Statistical aggregation (min, max, mean, std)
    - Memory usage tracking
    - GPU timing support (PyTorch CUDA events)
    - Progress estimation with ETA
    - Automatic reporting
    - Multi-threaded timing support
    """
    
    def __init__(
        self,
        name: str = "timer",
        logger: Optional['Logger'] = None,
        track_memory: bool = True,
        track_gpu: bool = False,
        auto_start: bool = False,
        save_report: bool = True,
        report_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the Timer.
        
        Args:
            name (str): Name of the timer instance
            logger (Logger, optional): Logger instance for logging
            track_memory (bool): Whether to track memory usage
            track_gpu (bool): Whether to track GPU usage (requires PyTorch)
            auto_start (bool): Whether to start timing immediately
            save_report (bool): Whether to save timing reports
            report_dir (str, Path, optional): Directory to save reports
        
        Example:
            timer = Timer(name="training_phase1")
            timer.start()
            # ... perform operations ...
            elapsed = timer.stop()
            print(f"Elapsed time: {elapsed:.2f} seconds")
        """
        self.name = name
        self.track_memory = track_memory and PSUTIL_AVAILABLE
        self.track_gpu = track_gpu and TORCH_AVAILABLE and torch.cuda.is_available()
        self.save_report = save_report
        
        # Setup logger
        if logger is None:
            self.logger = get_logger(
                log_dir="logs/timer",
                name=f"timer_{name}",
                verbose=True
            )
        else:
            self.logger = logger
        
        # Setup report directory
        if report_dir is None:
            report_dir = Path("logs/timer_reports")
        self.report_dir = Path(report_dir)
        if self.save_report:
            self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Timing data
        self.start_time = None
        self.end_time = None
        self.elapsed_times = []
        self.lap_times = []
        self.lap_labels = []
        self.current_lap_start = None
        
        # Nested timers
        self.child_timers = {}
        self.parent_timer = None
        self.depth = 0
        
        # Memory tracking
        self.memory_usage = []
        self.peak_memory = None
        self.start_memory = None
        self.end_memory = None
        
        # GPU tracking
        self.gpu_events = {}
        self.gpu_times = []
        
        # Statistics
        self.stats = {
            'count': 0,
            'total': 0.0,
            'min': float('inf'),
            'max': 0.0,
            'mean': 0.0,
            'std': 0.0,
            'percentiles': {}
        }
        
        # Auto-start if requested
        if auto_start:
            self.start()
        
        self.logger.log_info(f"Timer '{name}' initialized")
    
    def start(self, reset: bool = False) -> float:
        """
        Start or restart the timer.
        
        Args:
            reset (bool): Whether to reset all timing data
            
        Returns:
            float: Start time in seconds since epoch
        
        Example:
            timer.start()
            # ... do work ...
            elapsed = timer.stop()
        """
        if reset:
            self.elapsed_times = []
            self.lap_times = []
            self.lap_labels = []
            self.memory_usage = []
            self.gpu_times = []
            self.stats = {
                'count': 0,
                'total': 0.0,
                'min': float('inf'),
                'max': 0.0,
                'mean': 0.0,
                'std': 0.0,
                'percentiles': {}
            }
        
        self.start_time = time.perf_counter()
        self.current_lap_start = self.start_time
        self.lap_times = [0.0]  # Lap 0 is the initial lap
        self.lap_labels = ['start']
        
        # Track memory
        if self.track_memory:
            self.start_memory = self._get_memory_usage()
            self.peak_memory = self.start_memory
            self.memory_usage = [self.start_memory]
        
        # Track GPU
        if self.track_gpu:
            self._create_gpu_event('start')
        
        self.logger.log_debug(f"Timer '{self.name}' started")
        return self.start_time
    
    def stop(self) -> float:
        """
        Stop the timer and record elapsed time.
        
        Returns:
            float: Elapsed time in seconds
        
        Example:
            elapsed = timer.stop()
        """
        if self.start_time is None:
            self.logger.log_warning("Timer not started, calling start()")
            self.start()
            return self.stop()
        
        self.end_time = time.perf_counter()
        elapsed = self.end_time - self.start_time
        
        # Add to elapsed times
        self.elapsed_times.append(elapsed)
        
        # Update stats
        self._update_stats(elapsed)
        
        # Track memory
        if self.track_memory:
            self.end_memory = self._get_memory_usage()
            if self.end_memory > self.peak_memory:
                self.peak_memory = self.end_memory
            self.memory_usage.append(self.end_memory)
        
        # Track GPU
        if self.track_gpu:
            self._create_gpu_event('stop')
            self._record_gpu_time()
        
        # Log completion
        self.logger.log_info(f"Timer '{self.name}' stopped: {self._format_time(elapsed)}")
        
        # Save report if requested
        if self.save_report and len(self.elapsed_times) % 10 == 0:
            self.save_report()
        
        return elapsed
    
    def lap(self, label: Optional[str] = None) -> float:
        """
        Record a lap time without stopping the timer.
        
        Args:
            label (str, optional): Label for this lap
            
        Returns:
            float: Lap time in seconds since last lap
        
        Example:
            for i, batch in enumerate(dataloader):
                timer.lap(f"batch_{i}")
        """
        if self.start_time is None:
            self.logger.log_warning("Timer not started, calling start()")
            self.start()
            return self.lap(label)
        
        current_time = time.perf_counter()
        
        if self.current_lap_start is None:
            lap_time = current_time - self.start_time
        else:
            lap_time = current_time - self.current_lap_start
        
        self.lap_times.append(lap_time)
        self.lap_labels.append(label or f"lap_{len(self.lap_times)-1}")
        self.current_lap_start = current_time
        
        # Track memory at lap
        if self.track_memory:
            current_memory = self._get_memory_usage()
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
            self.memory_usage.append(current_memory)
        
        self.logger.log_debug(f"Lap recorded: {label} - {self._format_time(lap_time)}")
        return lap_time
    
    def reset(self) -> None:
        """
        Reset all timing data.
        
        Example:
            timer.reset()
            timer.start()
        """
        self.start_time = None
        self.end_time = None
        self.elapsed_times = []
        self.lap_times = []
        self.lap_labels = []
        self.memory_usage = []
        self.gpu_times = []
        self.current_lap_start = None
        self.stats = {
            'count': 0,
            'total': 0.0,
            'min': float('inf'),
            'max': 0.0,
            'mean': 0.0,
            'std': 0.0,
            'percentiles': {}
        }
        self.logger.log_debug(f"Timer '{self.name}' reset")
    
    def _update_stats(self, elapsed: float) -> None:
        """
        Update timing statistics.
        
        Args:
            elapsed (float): Elapsed time in seconds
        """
        self.stats['count'] += 1
        self.stats['total'] += elapsed
        
        if elapsed < self.stats['min']:
            self.stats['min'] = elapsed
        if elapsed > self.stats['max']:
            self.stats['max'] = elapsed
        
        self.stats['mean'] = self.stats['total'] / self.stats['count']
        
        # Calculate standard deviation if we have enough samples
        if len(self.elapsed_times) > 1:
            mean = self.stats['mean']
            variance = sum((t - mean) ** 2 for t in self.elapsed_times) / len(self.elapsed_times)
            self.stats['std'] = variance ** 0.5
        
        # Calculate percentiles
        if len(self.elapsed_times) >= 10:
            sorted_times = sorted(self.elapsed_times)
            for p in [50, 90, 95, 99]:
                idx = int(len(sorted_times) * p / 100)
                self.stats['percentiles'][p] = sorted_times[min(idx, len(sorted_times)-1)]
    
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            float: Memory usage in MB
        """
        if not PSUTIL_AVAILABLE:
            return 0.0
        
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except Exception as e:
            self.logger.log_warning(f"Failed to get memory usage: {e}")
            return 0.0
    
    def _create_gpu_event(self, name: str) -> None:
        """
        Create a GPU timing event.
        
        Args:
            name (str): Name of the event
        """
        if not self.track_gpu:
            return
        
        try:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            self.gpu_events[f'{name}_start'] = start_event
            self.gpu_events[f'{name}_end'] = end_event
        except Exception as e:
            self.logger.log_warning(f"Failed to create GPU event: {e}")
    
    def _record_gpu_time(self) -> None:
        """
        Record GPU timing between start and stop events.
        """
        if not self.track_gpu or 'start_start' not in self.gpu_events:
            return
        
        try:
            start_event = self.gpu_events.get('start_start')
            end_event = self.gpu_events.get('stop_end')
            
            if start_event and end_event:
                elapsed = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
                self.gpu_times.append(elapsed)
        except Exception as e:
            self.logger.log_warning(f"Failed to record GPU time: {e}")
    
    def get_elapsed(self, format: str = 'seconds') -> Union[float, str]:
        """
        Get the current elapsed time.
        
        Args:
            format (str): Output format ('seconds', 'milliseconds', 'microseconds', 'human')
            
        Returns:
            Union[float, str]: Elapsed time in requested format
        
        Example:
            elapsed = timer.get_elapsed(format='human')  # "1h 23m 45s"
        """
        if self.start_time is None:
            return 0.0
        
        current_time = time.perf_counter()
        elapsed = current_time - self.start_time
        
        if format == 'seconds':
            return elapsed
        elif format == 'milliseconds':
            return elapsed * 1000
        elif format == 'microseconds':
            return elapsed * 1000000
        elif format == 'human':
            return self._format_time(elapsed)
        else:
            return elapsed
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get timing statistics.
        
        Returns:
            Dict[str, Any]: Timing statistics
        
        Example:
            stats = timer.get_stats()
            print(f"Average time: {stats['mean']:.3f}s")
        """
        return {
            **self.stats,
            'total_time': self.stats['total'],
            'num_measurements': self.stats['count'],
            'total_time_human': self._format_time(self.stats['total']),
            'mean_human': self._format_time(self.stats['mean']),
            'memory_peak_mb': self.peak_memory,
            'gpu_time': sum(self.gpu_times) if self.gpu_times else None
        }
    
    @contextmanager
    def measure(self, label: Optional[str] = None):
        """
        Context manager for measuring code block execution time.
        
        Args:
            label (str, optional): Label for this measurement
        
        Yields:
            Timer: Self reference
        
        Example:
            with timer.measure("data_loading"):
                data = load_data()
        """
        # Record start lap
        self.lap(f"{label}_start" if label else "start")
        
        try:
            yield self
        finally:
            # Record end lap
            self.lap(f"{label}_end" if label else "end")
    
    def measure_time(self, func: Optional[Callable] = None, label: Optional[str] = None):
        """
        Decorator for measuring function execution time.
        
        Args:
            func (Callable, optional): Function to decorate
            label (str, optional): Label for this measurement
            
        Returns:
            Callable: Decorated function
        
        Example:
            @timer.measure_time(label="training_epoch")
            def train_epoch():
                # ... training code ...
        """
        def decorator(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                # Get function name if label not provided
                func_label = label or f"{f.__name__}"
                
                # Start timing
                self.start()
                self.logger.log_debug(f"Starting timed function: {func_label}")
                
                try:
                    result = f(*args, **kwargs)
                    elapsed = self.stop()
                    self.logger.log_info(
                        f"Function '{func_label}' completed in {self._format_time(elapsed)}"
                    )
                    return result
                except Exception as e:
                    self.stop()
                    self.logger.log_error(f"Function '{func_label}' failed after {self._format_time(self.get_elapsed())}: {e}")
                    raise
            
            return wrapper
        
        if func is not None:
            return decorator(func)
        return decorator
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """
        Format time in a human-readable string.
        
        Args:
            seconds (float): Time in seconds
            
        Returns:
            str: Human-readable time string
        
        Example:
            Timer._format_time(3661) -> "1h 1m 1s"
        """
        if seconds < 60:
            return f"{seconds:.3f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f}m"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.2f}h"
        else:
            days = seconds / 86400
            return f"{days:.2f}d"
    
    def create_child_timer(self, name: str, auto_start: bool = True) -> 'Timer':
        """
        Create a child timer for nested timing.
        
        Args:
            name (str): Name of the child timer
            auto_start (bool): Whether to auto-start the child timer
            
        Returns:
            Timer: Child timer instance
        
        Example:
            with timer.measure("outer"):
                child_timer = timer.create_child_timer("inner")
                with child_timer.measure("inner_work"):
                    # ... inner work ...
        """
        if name in self.child_timers:
            self.logger.log_warning(f"Child timer '{name}' already exists, returning existing")
            return self.child_timers[name]
        
        child = Timer(
            name=f"{self.name}_{name}",
            logger=self.logger,
            track_memory=self.track_memory,
            track_gpu=self.track_gpu,
            auto_start=auto_start,
            save_report=self.save_report,
            report_dir=self.report_dir / name
        )
        child.parent_timer = self
        child.depth = self.depth + 1
        
        self.child_timers[name] = child
        return child
    
    def get_child_timer(self, name: str) -> Optional['Timer']:
        """
        Get a child timer by name.
        
        Args:
            name (str): Name of the child timer
            
        Returns:
            Optional[Timer]: Child timer or None if not found
        """
        return self.child_timers.get(name)
    
    def estimate_remaining_time(
        self,
        progress: float,
        total: Optional[float] = None,
        smooth_window: int = 10
    ) -> Tuple[float, str]:
        """
        Estimate remaining time based on progress.
        
        Args:
            progress (float): Current progress (0.0 to 1.0)
            total (float, optional): Total expected time
            smooth_window (int): Window size for smoothing
            
        Returns:
            Tuple[float, str]: (Estimated remaining seconds, human-readable string)
        
        Example:
            for i in range(100):
                elapsed, remaining = timer.estimate_remaining_time(i/100)
                print(f"ETA: {remaining}")
        """
        if self.start_time is None:
            return 0.0, "Unknown"
        
        current_time = time.perf_counter()
        elapsed = current_time - self.start_time
        
        if progress <= 0.0:
            return float('inf'), "Unknown"
        
        if total is not None and progress > 0.0:
            estimated_total = elapsed / progress
            remaining = estimated_total - elapsed
            
            # Smooth using recent measurements
            if len(self.lap_times) > smooth_window:
                recent_avg = sum(self.lap_times[-smooth_window:]) / smooth_window
                if progress > 0.1:
                    remaining = recent_avg * (1 - progress) / progress
        else:
            # Estimate based on progress
            if progress < 1.0:
                remaining = elapsed * (1.0 - progress) / progress
            else:
                remaining = 0.0
        
        remaining = max(0.0, remaining)
        return remaining, self._format_time(remaining)
    
    def log_progress(
        self,
        current: int,
        total: int,
        label: str = "",
        interval: int = 1
    ) -> None:
        """
        Log progress with ETA.
        
        Args:
            current (int): Current progress step
            total (int): Total steps
            label (str): Progress label
            interval (int): Log interval in steps
        
        Example:
            for i in range(100):
                timer.log_progress(i, 100, "Training")
                # ... do work ...
        """
        if current % interval != 0 and current != total:
            return
        
        progress = current / total if total > 0 else 0.0
        elapsed = self.get_elapsed()
        remaining, eta = self.estimate_remaining_time(progress)
        
        if current == total:
            self.logger.log_info(
                f"{label} completed in {self._format_time(elapsed)}"
            )
        else:
            self.logger.log_info(
                f"{label}: {current}/{total} ({progress*100:.1f}%) - "
                f"Elapsed: {self._format_time(elapsed)} - "
                f"ETA: {eta}"
            )
    
    def save_report(self, filename: Optional[str] = None) -> str:
        """
        Save timing report to file.
        
        Args:
            filename (str, optional): Report filename
            
        Returns:
            str: Path to saved report
        
        Example:
            report_path = timer.save_report()
        """
        if not self.save_report:
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"timer_report_{self.name}_{timestamp}.json"
        
        filepath = self.report_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'name': self.name,
            'timestamp': datetime.now().isoformat(),
            'stats': self.get_stats(),
            'lap_times': self.lap_times,
            'lap_labels': self.lap_labels,
            'elapsed_times': self.elapsed_times,
            'memory_usage_mb': self.memory_usage if self.track_memory else None,
            'gpu_times': self.gpu_times if self.track_gpu else None,
            'depth': self.depth,
            'child_timers': {
                name: timer.get_stats() for name, timer in self.child_timers.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.log_info(f"Timer report saved to: {filepath}")
        return str(filepath)
    
    def generate_summary(self) -> str:
        """
        Generate a formatted summary string.
        
        Returns:
            str: Formatted summary
        
        Example:
            print(timer.generate_summary())
        """
        stats = self.get_stats()
        
        lines = [
            f"=== Timer Summary: {self.name} ===",
            f"Measurements: {stats['num_measurements']}",
            f"Total time: {stats['total_time_human']}",
            f"Average time: {stats['mean_human']}",
            f"Min time: {self._format_time(stats['min'])}",
            f"Max time: {self._format_time(stats['max'])}",
            f"Std dev: {self._format_time(stats['std'])}",
        ]
        
        if stats.get('percentiles'):
            lines.append("Percentiles:")
            for p, value in stats['percentiles'].items():
                lines.append(f"  P{p}: {self._format_time(value)}")
        
        if self.track_memory and stats.get('memory_peak_mb'):
            lines.append(f"Peak memory: {stats['memory_peak_mb']:.1f} MB")
        
        if self.track_gpu and stats.get('gpu_time'):
            lines.append(f"GPU time: {self._format_time(stats['gpu_time'])}")
        
        if self.child_timers:
            lines.append("Child timers:")
            for name, child in self.child_timers.items():
                child_stats = child.get_stats()
                lines.append(f"  {name}: {child_stats['total_time_human']} "
                           f"(n={child_stats['num_measurements']})")
        
        lines.append("=" * 40)
        return "\n".join(lines)
    
    def __enter__(self):
        """Enter context manager."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.stop()
        if exc_type is not None:
            self.logger.log_error(f"Timer '{self.name}' exited with error: {exc_val}")
        return False
    
    def __repr__(self) -> str:
        """String representation of the Timer."""
        return f"Timer(name='{self.name}', measurements={len(self.elapsed_times)}, total={self._format_time(self.stats['total'] if self.stats else 0.0)})"


class GlobalTimer:
    """
    Global timer registry for managing multiple timers.
    
    Features:
    - Centralized timer management
    - Timer creation and retrieval
    - Global reporting
    - Combined statistics
    """
    
    _instance = None
    _timers = {}
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.logger = get_logger(
            log_dir="logs/timer",
            name="global_timer",
            verbose=True
        )
        self._timers = {}
        self.logger.log_info("GlobalTimer initialized")
    
    def create_timer(
        self,
        name: str,
        **kwargs
    ) -> Timer:
        """
        Create or get a timer.
        
        Args:
            name (str): Timer name
            **kwargs: Timer initialization arguments
            
        Returns:
            Timer: Timer instance
        """
        with self._lock:
            if name in self._timers:
                self.logger.log_warning(f"Timer '{name}' already exists, returning existing")
                return self._timers[name]
            
            timer = Timer(name=name, **kwargs)
            self._timers[name] = timer
            return timer
    
    def get_timer(self, name: str) -> Optional[Timer]:
        """
        Get a timer by name.
        
        Args:
            name (str): Timer name
            
        Returns:
            Optional[Timer]: Timer instance or None
        """
        return self._timers.get(name)
    
    def get_all_timers(self) -> Dict[str, Timer]:
        """
        Get all timers.
        
        Returns:
            Dict[str, Timer]: Dictionary of timers
        """
        return dict(self._timers)
    
    def reset_all(self) -> None:
        """
        Reset all timers.
        """
        with self._lock:
            for timer in self._timers.values():
                timer.reset()
    
    def save_all_reports(self) -> Dict[str, str]:
        """
        Save reports for all timers.
        
        Returns:
            Dict[str, str]: Dictionary mapping timer names to report paths
        """
        reports = {}
        for name, timer in self._timers.items():
            if len(timer.elapsed_times) > 0:
                reports[name] = timer.save_report()
        return reports
    
    def generate_global_summary(self) -> str:
        """
        Generate a summary of all timers.
        
        Returns:
            str: Global summary string
        """
        lines = ["=== Global Timer Summary ==="]
        
        for name, timer in self._timers.items():
            stats = timer.get_stats()
            if stats['num_measurements'] > 0:
                lines.append(
                    f"{name}: {stats['total_time_human']} "
                    f"(n={stats['num_measurements']}, "
                    f"avg={timer._format_time(stats['mean'])}"
                )
        
        lines.append("=" * 30)
        return "\n".join(lines)


# Convenience functions
def timer(
    name: Optional[str] = None,
    log_level: str = 'info'
) -> Callable:
    """
    Decorator for timing functions.
    
    Args:
        name (str, optional): Timer name
        log_level (str): Log level for timing output
        
    Returns:
        Callable: Decorator function
    
    Example:
        @timer("data_loading")
        def load_data():
            # ... loading code ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = name or f"{func.__module__}_{func.__name__}"
            timer_instance = global_timer.create_timer(timer_name)
            
            if log_level.lower() == 'info':
                timer_instance.logger.log_info(f"Starting: {timer_name}")
            
            with timer_instance:
                result = func(*args, **kwargs)
            
            return result
        return wrapper
    return decorator


# Create global timer instance
global_timer = GlobalTimer()


# For testing the timer
if __name__ == "__main__":
    import time
    import tempfile
    
    print("Testing Timer...")
    
    # Test basic timer
    timer1 = Timer(name="test_timer", auto_start=False)
    
    # Test context manager
    with timer1.measure("main_work"):
        time.sleep(0.1)
        
        # Test lap
        timer1.lap("after_sleep")
        
        # Test nested timer
        child = timer1.create_child_timer("nested")
        with child.measure("nested_work"):
            time.sleep(0.05)
        
        # Test manual lap
        timer1.lap("after_nested")
    
    # Test decorator
    @timer("decorated_function")
    def test_function():
        time.sleep(0.1)
        return "done"
    
    test_function()
    
    # Test progress logging
    timer2 = Timer(name="progress_timer")
    timer2.start()
    
    for i in range(10):
        time.sleep(0.05)
        timer2.log_progress(i+1, 10, "Processing")
    
    timer2.stop()
    
    # Get statistics
    stats = timer1.get_stats()
    print(f"\nTimer stats: {stats}")
    
    # Generate summary
    print("\n" + timer1.generate_summary())
    
    # Test global timer
    global_timer.create_timer("global_1")
    with global_timer.get_timer("global_1") as g_timer:
        time.sleep(0.05)
    
    # Save report
    with tempfile.TemporaryDirectory() as temp_dir:
        timer1.save_report()
        print(f"\nReports saved in: {temp_dir}")
    
    # Print global summary
    print("\n" + global_timer.generate_global_summary())
    
    print("\nTimer tests completed successfully!")