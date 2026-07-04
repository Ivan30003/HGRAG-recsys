"""
utils/seed_manager.py

Comprehensive seed management module for H-GRAGrecsys with support for:
- Reproducible random number generation across all libraries
- Seed management for different components (Python, NumPy, PyTorch, CUDA)
- Deterministic data splitting and sampling
- Experiment reproducibility tracking
- Seed generation for multiple experiments
- Context managers for temporary seed changes
"""

import os
import sys
import random
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple, Generator
from contextlib import contextmanager
from datetime import datetime
import json

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.logger import get_logger
from utils.config_loader import ConfigLoader

# Try to import optional libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class SeedManager:
    """
    Comprehensive seed management for reproducible research.
    
    Features:
    - Set seeds for Python, NumPy, PyTorch, TensorFlow, CUDA
    - Generate deterministic random numbers
    - Reproducible data splitting and sampling
    - Context managers for temporary seed overrides
    - Seed logging and experiment tracking
    - Support for multi-worker reproducibility
    - Deterministic operations for all libraries
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        config_path: Optional[Union[str, Path]] = None,
        log_seed_usage: bool = True,
        deterministic_algorithms: bool = True,
        use_environment_seed: bool = True,
        logger: Optional['Logger'] = None
    ):
        """
        Initialize the SeedManager.
        
        Args:
            seed (int, optional): Initial seed value. If None, will try to get from config or generate.
            config_path (str, Path, optional): Path to configuration file
            log_seed_usage (bool): Whether to log seed usage
            deterministic_algorithms (bool): Whether to use deterministic algorithms in PyTorch/CUDA
            use_environment_seed (bool): Whether to use seed from environment variable
            logger (Logger, optional): Logger instance for logging
        
        Example:
            seed_manager = SeedManager(seed=42)
            seed_manager.set_all_seeds()
            # Now all random operations are reproducible
        """
        self.log_seed_usage = log_seed_usage
        self.deterministic_algorithms = deterministic_algorithms
        
        # Setup logger
        if logger is None:
            self.logger = get_logger(
                log_dir="logs/seed_manager",
                name="seed_manager",
                verbose=True
            )
        else:
            self.logger = logger
        
        # Load configuration
        self.config = {}
        if config_path:
            loader = ConfigLoader(config_path)
            self.config = loader.load_config()
            # Get seed from config if available
            config_seed = self.config.get('seed', self.config.get('experiment', {}).get('seed'))
            if config_seed is not None and seed is None:
                seed = config_seed
        
        # Get seed from environment variable if requested
        if use_environment_seed and seed is None:
            env_seed = os.environ.get('H_GRAG_SEED')
            if env_seed is not None:
                try:
                    seed = int(env_seed)
                    self.logger.log_info(f"Using seed from environment: {seed}")
                except ValueError:
                    self.logger.log_warning(f"Invalid environment seed: {env_seed}")
        
        # Generate seed if still None
        if seed is None:
            seed = self._generate_seed()
            self.logger.log_info(f"Generated random seed: {seed}")
        
        # Store seed
        self.base_seed = seed
        self.current_seed = seed
        self.seed_history = []
        
        # Random states for different libraries
        self.random_states = {
            'python': None,
            'numpy': None,
            'torch': None,
            'tensorflow': None,
        }
        
        # Track seeded operations
        self.seeded_operations = []
        
        # Initialize random number generators
        self._initialize_rngs(seed)
        
        # Log initialization
        self.logger.log_info(f"SeedManager initialized with base seed: {seed}")
        self.logger.log_info(f"Deterministic algorithms: {deterministic_algorithms}")
    
    def _generate_seed(self) -> int:
        """
        Generate a seed using system time and entropy.
        
        Returns:
            int: Generated seed
        """
        # Use current time with microsecond precision
        t = int(time.time() * 1000000)
        
        # Add some entropy from os.urandom if available
        try:
            entropy = int.from_bytes(os.urandom(4), 'big')
        except:
            entropy = 0
        
        # Combine and hash to get a seed in the valid range
        seed = (t ^ entropy) & 0xFFFFFFFF
        return seed
    
    def _initialize_rngs(self, seed: int) -> None:
        """
        Initialize random number generators for different libraries.
        
        Args:
            seed (int): Seed value to initialize with
        """
        # Python's random
        random.seed(seed)
        self.random_states['python'] = random.getstate()
        
        # NumPy
        if NUMPY_AVAILABLE:
            np.random.seed(seed)
            self.random_states['numpy'] = np.random.get_state()
        
        # PyTorch
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                if self.deterministic_algorithms:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    torch.use_deterministic_algorithms(True)
                self.random_states['torch'] = torch.get_rng_state()
        
        # TensorFlow
        if TENSORFLOW_AVAILABLE:
            tf.random.set_seed(seed)
            self.random_states['tensorflow'] = tf.random.get_global_generator().state
        
        self.logger.log_debug(f"Initialized all RNGs with seed: {seed}")
    
    def set_all_seeds(
        self,
        seed: Optional[int] = None,
        save_state: bool = True
    ) -> int:
        """
        Set seeds for all libraries to ensure reproducibility.
        
        Args:
            seed (int, optional): Seed to use. If None, uses current seed.
            save_state (bool): Whether to save the state for later restoration
            
        Returns:
            int: The seed that was set
        
        Example:
            # Set all seeds to 42
            seed = seed_manager.set_all_seeds(42)
            
            # Use current seed
            seed = seed_manager.set_all_seeds()
        """
        if seed is None:
            seed = self.current_seed
        
        # Save current state if requested
        if save_state:
            self._save_current_state()
        
        # Initialize RNGs with new seed
        self._initialize_rngs(seed)
        self.current_seed = seed
        
        # Log the operation
        self.seeded_operations.append({
            'operation': 'set_all_seeds',
            'seed': seed,
            'timestamp': datetime.now().isoformat()
        })
        
        self.logger.log_info(f"Set all seeds to: {seed}")
        
        if self.deterministic_algorithms and TORCH_AVAILABLE:
            if torch.cuda.is_available():
                self.logger.log_debug("CUDA deterministic algorithms enabled")
        
        return seed
    
    def set_seed(
        self,
        library: str,
        seed: Optional[int] = None,
        save_state: bool = True
    ) -> int:
        """
        Set seed for a specific library.
        
        Args:
            library (str): Library to set seed for ('python', 'numpy', 'torch', 'tensorflow')
            seed (int, optional): Seed to use. If None, uses current seed.
            save_state (bool): Whether to save the state for later restoration
            
        Returns:
            int: The seed that was set
        
        Example:
            seed = seed_manager.set_seed('torch', 42)
            seed = seed_manager.set_seed('numpy', 123)
        """
        if seed is None:
            seed = self.current_seed
        
        # Save current state if requested
        if save_state:
            self._save_current_state(library=library)
        
        # Set seed for specific library
        if library == 'python':
            random.seed(seed)
            self.random_states['python'] = random.getstate()
        
        elif library == 'numpy' and NUMPY_AVAILABLE:
            np.random.seed(seed)
            self.random_states['numpy'] = np.random.get_state()
        
        elif library == 'torch' and TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                if self.deterministic_algorithms:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
            self.random_states['torch'] = torch.get_rng_state()
        
        elif library == 'tensorflow' and TENSORFLOW_AVAILABLE:
            tf.random.set_seed(seed)
            self.random_states['tensorflow'] = tf.random.get_global_generator().state
        
        else:
            self.logger.log_warning(f"Unsupported library: {library}")
            return seed
        
        # Log the operation
        self.seeded_operations.append({
            'operation': f'set_seed_{library}',
            'seed': seed,
            'timestamp': datetime.now().isoformat()
        })
        
        self.logger.log_debug(f"Set seed for {library} to: {seed}")
        return seed
    
    def _save_current_state(self, library: Optional[str] = None) -> None:
        """
        Save current random state for restoration.
        
        Args:
            library (str, optional): Specific library to save state for
        """
        if library is None or library == 'python':
            self.random_states['python'] = random.getstate()
        
        if (library is None or library == 'numpy') and NUMPY_AVAILABLE:
            self.random_states['numpy'] = np.random.get_state()
        
        if (library is None or library == 'torch') and TORCH_AVAILABLE:
            self.random_states['torch'] = torch.get_rng_state()
            if torch.cuda.is_available():
                self.random_states['torch_cuda'] = torch.cuda.get_rng_state_all()
        
        if (library is None or library == 'tensorflow') and TENSORFLOW_AVAILABLE:
            self.random_states['tensorflow'] = tf.random.get_global_generator().state
    
    def restore_state(self, library: Optional[str] = None) -> None:
        """
        Restore previously saved random state.
        
        Args:
            library (str, optional): Specific library to restore state for
        """
        if library is None or library == 'python':
            if self.random_states['python']:
                random.setstate(self.random_states['python'])
        
        if (library is None or library == 'numpy') and NUMPY_AVAILABLE:
            if self.random_states['numpy'] is not None:
                np.random.set_state(self.random_states['numpy'])
        
        if (library is None or library == 'torch') and TORCH_AVAILABLE:
            if self.random_states['torch'] is not None:
                torch.set_rng_state(self.random_states['torch'])
                if torch.cuda.is_available() and 'torch_cuda' in self.random_states:
                    torch.cuda.set_rng_state_all(self.random_states['torch_cuda'])
        
        if (library is None or library == 'tensorflow') and TENSORFLOW_AVAILABLE:
            if self.random_states['tensorflow'] is not None:
                tf.random.get_global_generator().state = self.random_states['tensorflow']
        
        self.logger.log_debug(f"Restored state for {library if library else 'all libraries'}")
    
    @contextmanager
    def temporary_seed(self, seed: int) -> Generator[None, None, None]:
        """
        Context manager for temporarily using a different seed.
        
        Args:
            seed (int): Seed to use temporarily
        
        Yields:
            None: Context manager
        
        Example:
            with seed_manager.temporary_seed(123):
                # Operations here use seed 123
                random_value = random.random()
            # Back to previous seed after context
        """
        # Save current state
        self._save_current_state()
        old_seed = self.current_seed
        
        # Set new seed
        self.set_all_seeds(seed, save_state=False)
        
        try:
            yield
        finally:
            # Restore old state
            self.current_seed = old_seed
            self.set_all_seeds(old_seed, save_state=False)
    
    @contextmanager
    def deterministic_section(self, section_name: str, seed: Optional[int] = None) -> Generator[None, None, None]:
        """
        Context manager for a deterministic section of code.
        
        Args:
            section_name (str): Name of the section for logging
            seed (int, optional): Seed to use for this section
        
        Yields:
            None: Context manager
        
        Example:
            with seed_manager.deterministic_section('data_split', seed=42):
                # This data split will be reproducible
                train_data, test_data = split_data()
        """
        if seed is None:
            seed = self._generate_section_seed(section_name)
        
        # Save current state
        self._save_current_state()
        old_seed = self.current_seed
        
        # Log section start
        self.logger.log_debug(f"Starting deterministic section: {section_name} (seed={seed})")
        
        # Set new seed
        self.set_all_seeds(seed, save_state=False)
        
        try:
            yield
        finally:
            # Restore old state
            self.current_seed = old_seed
            self.set_all_seeds(old_seed, save_state=False)
            self.logger.log_debug(f"Completed deterministic section: {section_name}")
    
    def _generate_section_seed(self, section_name: str) -> int:
        """
        Generate a deterministic seed for a section name.
        
        Args:
            section_name (str): Name of the section
            
        Returns:
            int: Generated seed
        """
        # Hash the section name with base seed
        combined = f"{self.base_seed}_{section_name}"
        hash_value = int(hashlib.sha256(combined.encode()).hexdigest(), 16)
        return hash_value & 0xFFFFFFFF
    
    def random_int(
        self,
        min_val: int = 0,
        max_val: int = 10**6,
        deterministic: bool = True,
        operation_name: Optional[str] = None
    ) -> int:
        """
        Generate a random integer with deterministic seed tracking.
        
        Args:
            min_val (int): Minimum value (inclusive)
            max_val (int): Maximum value (exclusive)
            deterministic (bool): Whether to make operation deterministic
            operation_name (str, optional): Name of operation for logging
        
        Returns:
            int: Random integer
        
        Example:
            idx = seed_manager.random_int(0, len(dataset))
            user_id = seed_manager.random_int(1000, 2000)
        """
        # Save state for deterministic operation
        if deterministic and operation_name:
            self._save_current_state()
            # Generate deterministic seed from operation
            section_seed = self._generate_section_seed(operation_name)
            self.set_all_seeds(section_seed, save_state=False)
        
        # Generate random integer
        if NUMPY_AVAILABLE:
            value = np.random.randint(min_val, max_val)
        else:
            value = random.randint(min_val, max_val)
        
        # Restore state if needed
        if deterministic and operation_name:
            self.restore_state()
        
        # Log operation
        if self.log_seed_usage and operation_name:
            self.seeded_operations.append({
                'operation': f'random_int_{operation_name}',
                'min': min_val,
                'max': max_val,
                'value': value,
                'timestamp': datetime.now().isoformat()
            })
        
        return int(value)
    
    def random_float(
        self,
        min_val: float = 0.0,
        max_val: float = 1.0,
        deterministic: bool = True,
        operation_name: Optional[str] = None
    ) -> float:
        """
        Generate a random float with deterministic seed tracking.
        
        Args:
            min_val (float): Minimum value
            max_val (float): Maximum value
            deterministic (bool): Whether to make operation deterministic
            operation_name (str, optional): Name of operation for logging
        
        Returns:
            float: Random float
        
        Example:
            ratio = seed_manager.random_float(0.0, 1.0, operation_name='train_split')
        """
        if deterministic and operation_name:
            self._save_current_state()
            section_seed = self._generate_section_seed(operation_name)
            self.set_all_seeds(section_seed, save_state=False)
        
        # Generate random float
        if NUMPY_AVAILABLE:
            value = np.random.uniform(min_val, max_val)
        else:
            value = random.uniform(min_val, max_val)
        
        if deterministic and operation_name:
            self.restore_state()
        
        if self.log_seed_usage and operation_name:
            self.seeded_operations.append({
                'operation': f'random_float_{operation_name}',
                'min': min_val,
                'max': max_val,
                'value': value,
                'timestamp': datetime.now().isoformat()
            })
        
        return float(value)
    
    def random_choice(
        self,
        population: List[Any],
        size: int = 1,
        replace: bool = False,
        deterministic: bool = True,
        operation_name: Optional[str] = None
    ) -> Any:
        """
        Randomly choose elements from a population.
        
        Args:
            population (List[Any]): Population to choose from
            size (int): Number of elements to choose
            replace (bool): Whether to allow replacement
            deterministic (bool): Whether to make operation deterministic
            operation_name (str, optional): Name of operation for logging
        
        Returns:
            Any: Chosen element(s)
        
        Example:
            # Choose 10 items without replacement
            selected = seed_manager.random_choice(items, size=10)
            # Choose 1 item
            item = seed_manager.random_choice(items)
        """
        if deterministic and operation_name:
            self._save_current_state()
            section_seed = self._generate_section_seed(operation_name)
            self.set_all_seeds(section_seed, save_state=False)
        
        # Generate random choice
        if NUMPY_AVAILABLE:
            choices = np.random.choice(population, size=size, replace=replace)
            if size == 1:
                value = choices[0]
            else:
                value = choices.tolist()
        else:
            if size == 1:
                value = random.choice(population)
            else:
                value = random.choices(population, k=size) if replace else random.sample(population, size)
        
        if deterministic and operation_name:
            self.restore_state()
        
        if self.log_seed_usage and operation_name:
            self.seeded_operations.append({
                'operation': f'random_choice_{operation_name}',
                'size': size,
                'replace': replace,
                'value': value,
                'timestamp': datetime.now().isoformat()
            })
        
        return value
    
    def shuffle(
        self,
        data: List[Any],
        deterministic: bool = True,
        operation_name: Optional[str] = None
    ) -> List[Any]:
        """
        Shuffle a list deterministically.
        
        Args:
            data (List[Any]): Data to shuffle
            deterministic (bool): Whether to make operation deterministic
            operation_name (str, optional): Name of operation for logging
        
        Returns:
            List[Any]: Shuffled data
        
        Example:
            shuffled_data = seed_manager.shuffle(data, operation_name='train_shuffle')
        """
        # Make a copy
        shuffled = list(data)
        
        if deterministic and operation_name:
            self._save_current_state()
            section_seed = self._generate_section_seed(operation_name)
            self.set_all_seeds(section_seed, save_state=False)
        
        # Shuffle
        if NUMPY_AVAILABLE:
            np.random.shuffle(shuffled)
        else:
            random.shuffle(shuffled)
        
        if deterministic and operation_name:
            self.restore_state()
        
        if self.log_seed_usage and operation_name:
            self.seeded_operations.append({
                'operation': f'shuffle_{operation_name}',
                'size': len(data),
                'timestamp': datetime.now().isoformat()
            })
        
        return shuffled
    
    def split_indices(
        self,
        total_size: int,
        ratios: Union[List[float], Dict[str, float]],
        shuffle: bool = True,
        deterministic: bool = True,
        operation_name: Optional[str] = None
    ) -> Dict[str, List[int]]:
        """
        Split indices into multiple sets with specified ratios.
        
        Args:
            total_size (int): Total number of indices
            ratios (Union[List[float], Dict[str, float]]): Split ratios
            shuffle (bool): Whether to shuffle indices before splitting
            deterministic (bool): Whether to make operation deterministic
            operation_name (str, optional): Name of operation for logging
        
        Returns:
            Dict[str, List[int]]: Dictionary of index sets
        
        Example:
            splits = seed_manager.split_indices(
                1000,
                {'train': 0.7, 'val': 0.15, 'test': 0.15},
                operation_name='dataset_split'
            )
        """
        # Convert ratios to dict if list
        if isinstance(ratios, list):
            names = ['split'] * len(ratios)
            ratios = {f'split_{i}': r for i, r in enumerate(ratios)}
        
        # Normalize ratios
        total_ratio = sum(ratios.values())
        normalized_ratios = {k: v / total_ratio for k, v in ratios.items()}
        
        # Create indices
        indices = list(range(total_size))
        
        # Shuffle if requested
        if shuffle:
            if deterministic and operation_name:
                self._save_current_state()
                section_seed = self._generate_section_seed(f"{operation_name}_shuffle")
                self.set_all_seeds(section_seed, save_state=False)
            
            if NUMPY_AVAILABLE:
                np.random.shuffle(indices)
            else:
                random.shuffle(indices)
            
            if deterministic and operation_name:
                self.restore_state()
        
        # Split into sets
        splits = {}
        start_idx = 0
        
        for name, ratio in normalized_ratios.items():
            size = int(ratio * total_size)
            end_idx = start_idx + size
            
            # Handle last split to ensure all indices are used
            if name == list(normalized_ratios.keys())[-1]:
                end_idx = total_size
            
            splits[name] = indices[start_idx:end_idx]
            start_idx = end_idx
        
        # Log operation
        if self.log_seed_usage and operation_name:
            self.seeded_operations.append({
                'operation': f'split_indices_{operation_name}',
                'total_size': total_size,
                'ratios': ratios,
                'shuffle': shuffle,
                'splits': {k: len(v) for k, v in splits.items()},
                'timestamp': datetime.now().isoformat()
            })
        
        return splits
    
    def sample_from_dataset(
        self,
        dataset: List[Any],
        sample_size: int,
        method: str = 'random',
        deterministic: bool = True,
        operation_name: Optional[str] = None
    ) -> List[Any]:
        """
        Sample from a dataset using various sampling methods.
        
        Args:
            dataset (List[Any]): Dataset to sample from
            sample_size (int): Number of samples to draw
            method (str): Sampling method ('random', 'stratified', 'uniform')
            deterministic (bool): Whether to make operation deterministic
            operation_name (str, optional): Name of operation for logging
        
        Returns:
            List[Any]: Sampled items
        
        Example:
            # Random sampling
            samples = seed_manager.sample_from_dataset(
                dataset, 100, method='random', operation_name='random_sample'
            )
        """
        if sample_size >= len(dataset):
            return dataset
        
        if deterministic and operation_name:
            self._save_current_state()
            section_seed = self._generate_section_seed(operation_name)
            self.set_all_seeds(section_seed, save_state=False)
        
        # Apply sampling method
        if method == 'random':
            # Simple random sampling
            if NUMPY_AVAILABLE:
                indices = np.random.choice(len(dataset), sample_size, replace=False)
                samples = [dataset[i] for i in indices]
            else:
                samples = random.sample(dataset, sample_size)
        
        elif method == 'stratified':
            # Stratified sampling (requires labels)
            # Assume dataset items are tuples (data, label)
            try:
                labels = [item[1] for item in dataset]
                unique_labels = list(set(labels))
                samples = []
                samples_per_label = sample_size // len(unique_labels)
                
                for label in unique_labels:
                    label_indices = [i for i, l in enumerate(labels) if l == label]
                    if len(label_indices) > samples_per_label:
                        selected_indices = self.random_choice(
                            label_indices,
                            size=samples_per_label,
                            deterministic=False,
                            operation_name=f"{operation_name}_{label}"
                        )
                        samples.extend([dataset[i] for i in selected_indices])
                    else:
                        samples.extend([dataset[i] for i in label_indices])
                
                # If we need more samples, add from remaining
                if len(samples) < sample_size:
                    remaining = sample_size - len(samples)
                    used_indices = set([dataset.index(s) for s in samples])
                    available = [i for i in range(len(dataset)) if i not in used_indices]
                    if available:
                        extra = self.random_choice(
                            available,
                            size=min(remaining, len(available)),
                            deterministic=False,
                            operation_name=f"{operation_name}_extra"
                        )
                        samples.extend([dataset[i] for i in extra])
            except (IndexError, AttributeError):
                # Fallback to random sampling
                self.logger.log_warning("Stratified sampling failed, falling back to random sampling")
                samples = self.sample_from_dataset(dataset, sample_size, 'random', False)
        
        elif method == 'uniform':
            # Uniform sampling (evenly spaced)
            step = len(dataset) / sample_size
            indices = [int(i * step) for i in range(sample_size)]
            samples = [dataset[i] for i in indices]
        
        else:
            raise ValueError(f"Unsupported sampling method: {method}")
        
        if deterministic and operation_name:
            self.restore_state()
        
        return samples
    
    def get_next_seed(self) -> int:
        """
        Get the next sequential seed for experiments.
        
        Returns:
            int: Next seed
        
        Example:
            for i in range(10):
                seed = seed_manager.get_next_seed()
                run_experiment(seed)
        """
        # Use base seed and increment
        current_seed = self.current_seed
        next_seed = current_seed + 1
        
        # Ensure seed is within valid range
        if next_seed > 2**31 - 1:
            next_seed = self.base_seed
        
        self.current_seed = next_seed
        self.set_all_seeds(next_seed, save_state=False)
        
        return next_seed
    
    def get_seed_for_experiment(self, experiment_id: Union[str, int]) -> int:
        """
        Get a deterministic seed for a specific experiment.
        
        Args:
            experiment_id (Union[str, int]): Experiment identifier
            
        Returns:
            int: Seed for the experiment
        
        Example:
            seed = seed_manager.get_seed_for_experiment('exp_001')
            # This will always return the same seed for 'exp_001'
        """
        if isinstance(experiment_id, int):
            seed = (self.base_seed + experiment_id) & 0xFFFFFFFF
        else:
            # Generate seed from string
            combined = f"{self.base_seed}_{experiment_id}"
            seed = int(hashlib.md5(combined.encode()).hexdigest(), 16) & 0xFFFFFFFF
        
        return seed
    
    def save_seed_state(self, path: Union[str, Path]) -> None:
        """
        Save the current seed state to a file.
        
        Args:
            path (str, Path): Path to save seed state
        
        Example:
            seed_manager.save_seed_state('seeds/experiment_seed_state.json')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'base_seed': self.base_seed,
            'current_seed': self.current_seed,
            'timestamp': datetime.now().isoformat(),
            'seeded_operations_count': len(self.seeded_operations),
            'random_states': {
                'python': str(self.random_states['python']) if self.random_states['python'] else None,
                'numpy': str(self.random_states['numpy']) if self.random_states['numpy'] else None,
                'torch': str(self.random_states['torch']) if self.random_states['torch'] else None,
                'tensorflow': str(self.random_states['tensorflow']) if self.random_states['tensorflow'] else None,
            },
            'config': self.config
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        self.logger.log_info(f"Seed state saved to: {path}")
    
    def load_seed_state(self, path: Union[str, Path]) -> None:
        """
        Load seed state from a file.
        
        Args:
            path (str, Path): Path to seed state file
        
        Example:
            seed_manager.load_seed_state('seeds/experiment_seed_state.json')
        """
        path = Path(path)
        
        if not path.exists():
            self.logger.log_warning(f"Seed state file not found: {path}")
            return
        
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.base_seed = state['base_seed']
        self.current_seed = state['current_seed']
        
        # Restore random states if available
        if state.get('random_states'):
            if state['random_states']['python']:
                # Can't fully restore Python state from string, just reset with seed
                random.seed(self.current_seed)
            
            if state['random_states']['numpy'] and NUMPY_AVAILABLE:
                np.random.seed(self.current_seed)
            
            if state['random_states']['torch'] and TORCH_AVAILABLE:
                torch.manual_seed(self.current_seed)
            
            if state['random_states']['tensorflow'] and TENSORFLOW_AVAILABLE:
                tf.random.set_seed(self.current_seed)
        
        self.logger.log_info(f"Seed state loaded from: {path}")
    
    def get_seed_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive report of seed usage.
        
        Returns:
            Dict[str, Any]: Seed usage report
        
        Example:
            report = seed_manager.get_seed_report()
            print(f"Used {len(report['seeded_operations'])} seeded operations")
        """
        # Count operations by type
        operation_counts = {}
        for op in self.seeded_operations:
            op_type = op.get('operation', 'unknown')
            if op_type not in operation_counts:
                operation_counts[op_type] = 0
            operation_counts[op_type] += 1
        
        return {
            'base_seed': self.base_seed,
            'current_seed': self.current_seed,
            'total_seeded_operations': len(self.seeded_operations),
            'operation_counts': operation_counts,
            'deterministic_algorithms': self.deterministic_algorithms,
            'libraries_available': {
                'numpy': NUMPY_AVAILABLE,
                'torch': TORCH_AVAILABLE,
                'tensorflow': TENSORFLOW_AVAILABLE,
                'cuda': TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False
            },
            'recent_operations': self.seeded_operations[-10:] if self.seeded_operations else []
        }
    
    def __repr__(self) -> str:
        """String representation of the SeedManager."""
        return f"SeedManager(base_seed={self.base_seed}, current_seed={self.current_seed}, operations={len(self.seeded_operations)})"


# Convenience function for quick seed manager creation
def create_seed_manager(
    seed: Optional[int] = None,
    config_path: Optional[Union[str, Path]] = None,
    deterministic: bool = True
) -> SeedManager:
    """
    Factory function to create a SeedManager instance.
    
    Args:
        seed (int, optional): Initial seed value
        config_path (str, Path, optional): Path to configuration file
        deterministic (bool): Whether to use deterministic algorithms
        
    Returns:
        SeedManager: Configured SeedManager instance
    
    Example:
        seed_manager = create_seed_manager(seed=42, config_path='config/default_config.yaml')
        seed_manager.set_all_seeds()
    """
    logger = get_logger(
        log_dir="logs/seed_manager",
        name="seed_manager_factory",
        verbose=True
    )
    
    return SeedManager(
        seed=seed,
        config_path=config_path,
        deterministic_algorithms=deterministic,
        logger=logger
    )


# For testing the seed manager
if __name__ == "__main__":
    import tempfile
    
    print("Testing SeedManager...")
    
    # Test basic initialization
    seed_manager = create_seed_manager(seed=42)
    print(f"Initialized: {seed_manager}")
    
    # Test setting all seeds
    seed_manager.set_all_seeds()
    print("All seeds set")
    
    # Test random operations
    random_int = seed_manager.random_int(0, 100, operation_name='test_int')
    print(f"Random int: {random_int}")
    
    random_float = seed_manager.random_float(0.0, 1.0, operation_name='test_float')
    print(f"Random float: {random_float}")
    
    # Test shuffle
    data = list(range(20))
    shuffled = seed_manager.shuffle(data, operation_name='test_shuffle')
    print(f"Shuffled data: {shuffled[:5]}...")
    
    # Test splitting
    ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    splits = seed_manager.split_indices(100, ratios, operation_name='test_split')
    print(f"Split sizes: { {k: len(v) for k, v in splits.items()} }")
    
    # Test temporary seed context
    with seed_manager.temporary_seed(123):
        value1 = seed_manager.random_int(0, 100)
        print(f"Temporary seed value: {value1}")
    
    # Test deterministic section
    with seed_manager.deterministic_section('test_section'):
        value2 = seed_manager.random_int(0, 100)
        print(f"Section value: {value2}")
    
    # Test get_next_seed
    next_seed = seed_manager.get_next_seed()
    print(f"Next seed: {next_seed}")
    
    # Test get_seed_for_experiment
    exp_seed = seed_manager.get_seed_for_experiment('test_exp')
    print(f"Experiment seed: {exp_seed}")
    
    # Test saving state
    with tempfile.TemporaryDirectory() as temp_dir:
        state_path = Path(temp_dir) / 'seed_state.json'
        seed_manager.save_seed_state(state_path)
        print(f"Saved state to: {state_path}")
        
        # Test loading state
        new_manager = create_seed_manager()
        new_manager.load_seed_state(state_path)
        print(f"Loaded state into new manager: {new_manager}")
    
    # Test report
    report = seed_manager.get_seed_report()
    print(f"Seed report: total operations = {report['total_seeded_operations']}")
    print(f"Operation counts: {report['operation_counts']}")
    
    print("\nSeedManager tests completed successfully!")