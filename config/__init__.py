"""
config/__init__.py

Configuration package initialization for H-GRAGrecsys.

This module provides centralized configuration management for the entire project,
including loading, validation, and access to all configuration files.

Features:
- Centralized configuration loading
- Configuration validation
- Default configuration management
- Environment variable integration
- Configuration caching
- Configuration section access
- Configuration merging
- Configuration export

Example:
    from config import get_config, load_config
    
    # Load configuration
    config = load_config()
    
    # Access specific section
    model_config = config.get_model_config()
    training_config = config.get_training_config()
    
    # Access specific value
    hidden_dim = config.get_value('model.gnn.hidden_dim', default=256)
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple
from dataclasses import dataclass, field
from functools import lru_cache
import warnings

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.config_loader import ConfigLoader, load_config as loader_load_config
from utils.logger import get_logger

# Package metadata
__version__ = "1.0.0"
__author__ = "H-GRAGrecsys Team"
__description__ = "Configuration management for H-GRAGrecsys"

# Default configuration paths
DEFAULT_CONFIG_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "default_config.yaml"
DATASET_CONFIG_PATH = DEFAULT_CONFIG_DIR / "dataset_config.yaml"
MODEL_CONFIG_PATH = DEFAULT_CONFIG_DIR / "model_config.yaml"
TRAINING_CONFIG_PATH = DEFAULT_CONFIG_DIR / "training_config.yaml"

# Configuration cache
_config_cache = {}
_config_loader = None
_logger = None


def get_logger_instance():
    """Get or create logger instance for config package."""
    global _logger
    if _logger is None:
        _logger = get_logger(
            log_dir="logs/config",
            name="config_package",
            verbose=False
        )
    return _logger


@dataclass
class ConfigSection:
    """
    Configuration section container with attribute-style access.
    
    Attributes:
        name (str): Section name
        data (Dict[str, Any]): Configuration data
        parent (Optional['ConfigSection']): Parent section
    
    Example:
        section = ConfigSection('model', {'gnn': {'hidden_dim': 256}})
        print(section.gnn.hidden_dim)  # 256
    """
    name: str
    data: Dict[str, Any]
    parent: Optional['ConfigSection'] = None
    
    def __getattr__(self, name: str) -> Any:
        """
        Attribute-style access to configuration values.
        
        Args:
            name (str): Attribute name
            
        Returns:
            Any: Configuration value
            
        Raises:
            AttributeError: If attribute not found
        """
        if name in self.data:
            value = self.data[name]
            if isinstance(value, dict):
                return ConfigSection(name, value, parent=self)
            return value
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with default.
        
        Args:
            key (str): Configuration key
            default (Any): Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        return self.data.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert section to dictionary.
        
        Returns:
            Dict[str, Any]: Section data
        """
        result = {}
        for key, value in self.data.items():
            if isinstance(value, ConfigSection):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ConfigSection(name='{self.name}', keys={list(self.data.keys())})"


@dataclass
class HGRAGConfig:
    """
    Main configuration class for H-GRAGrecsys.
    
    Provides unified access to all configuration sections with attribute-style
    access and validation.
    
    Attributes:
        config_loader (ConfigLoader): Configuration loader instance
        config (Dict[str, Any]): Raw configuration dictionary
        model (ConfigSection): Model configuration
        training (ConfigSection): Training configuration
        evaluation (ConfigSection): Evaluation configuration
        data (ConfigSection): Data configuration
        logging (ConfigSection): Logging configuration
        experiment (ConfigSection): Experiment configuration
        seed (int): Random seed
        device (str): Device to use
    
    Example:
        config = HGRAGConfig.load()
        print(config.model.gnn.hidden_dim)
        print(config.training.phase1.num_epochs)
        print(config.data.dataset_name)
    """
    
    config_loader: ConfigLoader
    config: Dict[str, Any]
    
    # Lazy-loaded sections
    _model_config: Optional[ConfigSection] = None
    _training_config: Optional[ConfigSection] = None
    _evaluation_config: Optional[ConfigSection] = None
    _data_config: Optional[ConfigSection] = None
    _logging_config: Optional[ConfigSection] = None
    _experiment_config: Optional[ConfigSection] = None
    _gnn_config: Optional[ConfigSection] = None
    _hybrid_config: Optional[ConfigSection] = None
    _distillation_config: Optional[ConfigSection] = None
    
    def __post_init__(self):
        """Post-initialization setup."""
        self._logger = get_logger_instance()
    
    @property
    def model(self) -> ConfigSection:
        """Get model configuration."""
        if self._model_config is None:
            data = self.config.get('model', {})
            self._model_config = ConfigSection('model', data)
        return self._model_config
    
    @property
    def training(self) -> ConfigSection:
        """Get training configuration."""
        if self._training_config is None:
            data = self.config.get('training', {})
            self._training_config = ConfigSection('training', data)
        return self._training_config
    
    @property
    def evaluation(self) -> ConfigSection:
        """Get evaluation configuration."""
        if self._evaluation_config is None:
            data = self.config.get('evaluation', {})
            self._evaluation_config = ConfigSection('evaluation', data)
        return self._evaluation_config
    
    @property
    def data(self) -> ConfigSection:
        """Get data configuration."""
        if self._data_config is None:
            data = self.config.get('data', {})
            self._data_config = ConfigSection('data', data)
        return self._data_config
    
    @property
    def logging(self) -> ConfigSection:
        """Get logging configuration."""
        if self._logging_config is None:
            data = self.config.get('logging', {})
            self._logging_config = ConfigSection('logging', data)
        return self._logging_config
    
    @property
    def experiment(self) -> ConfigSection:
        """Get experiment configuration."""
        if self._experiment_config is None:
            data = self.config.get('experiment', {})
            self._experiment_config = ConfigSection('experiment', data)
        return self._experiment_config
    
    @property
    def seed(self) -> int:
        """Get random seed."""
        return self.config.get('seed', 42)
    
    @property
    def device(self) -> str:
        """Get device."""
        return self.config.get('device', 'cuda')
    
    def get_section(self, section_name: str) -> ConfigSection:
        """
        Get a configuration section by name.
        
        Args:
            section_name (str): Section name
            
        Returns:
            ConfigSection: Configuration section
            
        Example:
            gnn_config = config.get_section('gnn')
        """
        data = self.config.get(section_name, {})
        return ConfigSection(section_name, data)
    
    def get_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path (str): Dot-separated key path
            default (Any): Default value if key not found
            
        Returns:
            Any: Configuration value
            
        Example:
            hidden_dim = config.get_value('model.gnn.hidden_dim', 256)
        """
        return self.config_loader.get_value(key_path, default)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates (Dict[str, Any]): Updates to apply
            
        Example:
            config.update({'model.gnn.hidden_dim': 512})
        """
        self.config_loader.update_config(updates)
        self.config = self.config_loader.config
        
        # Clear cached sections
        self._model_config = None
        self._training_config = None
        self._evaluation_config = None
        self._data_config = None
        self._logging_config = None
        self._experiment_config = None
        self._gnn_config = None
        self._hybrid_config = None
        self._distillation_config = None
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate configuration.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        return self.config_loader.validate_config()
    
    def save(self, path: Union[str, Path]) -> str:
        """
        Save configuration to file.
        
        Args:
            path (str, Path): Output path
            
        Returns:
            str: Path to saved file
        """
        return self.config_loader.save_config(path=path)
    
    def export_to_yaml(self, path: Union[str, Path]) -> str:
        """
        Export configuration to YAML file.
        
        Args:
            path (str, Path): Output path
            
        Returns:
            str: Path to saved file
        """
        return self.config_loader.save_config(path=path, format='yaml')
    
    def export_to_json(self, path: Union[str, Path]) -> str:
        """
        Export configuration to JSON file.
        
        Args:
            path (str, Path): Output path
            
        Returns:
            str: Path to saved file
        """
        return self.config_loader.save_config(path=path, format='json')
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return self.config
    
    def __repr__(self) -> str:
        """String representation."""
        return f"HGRAGConfig(sections={list(self.config.keys())})"


# ============================================================================
# Public API Functions
# ============================================================================

def load_config(
    config_path: Optional[Union[str, Path]] = None,
    load_defaults: bool = True,
    merge_configs: bool = True,
    use_cache: bool = True
) -> HGRAGConfig:
    """
    Load configuration from file(s).
    
    Args:
        config_path (str, Path, optional): Path to configuration file
        load_defaults (bool): Whether to load default configuration
        merge_configs (bool): Whether to merge multiple config files
        use_cache (bool): Whether to use cached configuration
        
    Returns:
        HGRAGConfig: Configuration instance
        
    Example:
        # Load default configuration
        config = load_config()
        
        # Load custom configuration
        config = load_config('config/my_config.yaml')
        
        # Load with defaults
        config = load_config('config/my_config.yaml', load_defaults=True)
    """
    global _config_cache, _config_loader
    
    # Check cache
    cache_key = str(config_path) if config_path else 'default'
    if use_cache and cache_key in _config_cache:
        return _config_cache[cache_key]
    
    logger = get_logger_instance()
    logger.log_info(f"Loading configuration from: {config_path or 'default'}")
    
    # Determine config path
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    # Load configs to merge
    if merge_configs:
        # Load default config
        base_config_loader = ConfigLoader(
            config_path=DEFAULT_CONFIG_PATH if load_defaults else None,
            load_defaults=load_defaults
        )
        
        # Load specific config files
        if str(config_path) != str(DEFAULT_CONFIG_PATH):
            # Merge with main config
            config_loader = ConfigLoader(
                config_path=config_path,
                load_defaults=False
            )
            
            # Merge with base
            merged_config = base_config_loader._merge_configs(
                base_config_loader.config,
                config_loader.config
            )
            config_loader.config = merged_config
        else:
            config_loader = base_config_loader
        
        # Try to load additional configs
        if DATASET_CONFIG_PATH.exists():
            dataset_loader = ConfigLoader(
                config_path=DATASET_CONFIG_PATH,
                load_defaults=False
            )
            config_loader.config = config_loader._merge_configs(
                config_loader.config,
                dataset_loader.config
            )
        
        if MODEL_CONFIG_PATH.exists():
            model_loader = ConfigLoader(
                config_path=MODEL_CONFIG_PATH,
                load_defaults=False
            )
            config_loader.config = config_loader._merge_configs(
                config_loader.config,
                model_loader.config
            )
        
        if TRAINING_CONFIG_PATH.exists():
            training_loader = ConfigLoader(
                config_path=TRAINING_CONFIG_PATH,
                load_defaults=False
            )
            config_loader.config = config_loader._merge_configs(
                config_loader.config,
                training_loader.config
            )
    else:
        # Load single config
        config_loader = ConfigLoader(
            config_path=config_path,
            load_defaults=load_defaults
        )
    
    # Create HGRAGConfig instance
    config = HGRAGConfig(
        config_loader=config_loader,
        config=config_loader.config
    )
    
    # Cache if requested
    if use_cache:
        _config_cache[cache_key] = config
    
    logger.log_info("Configuration loaded successfully")
    
    return config


def get_default_config() -> HGRAGConfig:
    """
    Get default configuration.
    
    Returns:
        HGRAGConfig: Default configuration
        
    Example:
        config = get_default_config()
    """
    return load_config(DEFAULT_CONFIG_PATH, load_defaults=True)


def get_model_config() -> Dict[str, Any]:
    """
    Get model configuration section.
    
    Returns:
        Dict[str, Any]: Model configuration
        
    Example:
        model_config = get_model_config()
        print(model_config['gnn']['hidden_dim'])
    """
    config = load_config()
    return config.config.get('model', {})


def get_training_config() -> Dict[str, Any]:
    """
    Get training configuration section.
    
    Returns:
        Dict[str, Any]: Training configuration
        
    Example:
        training_config = get_training_config()
        print(training_config['phase1']['num_epochs'])
    """
    config = load_config()
    return config.config.get('training', {})


def get_evaluation_config() -> Dict[str, Any]:
    """
    Get evaluation configuration section.
    
    Returns:
        Dict[str, Any]: Evaluation configuration
        
    Example:
        eval_config = get_evaluation_config()
        print(eval_config['k_values'])
    """
    config = load_config()
    return config.config.get('evaluation', {})


def get_data_config() -> Dict[str, Any]:
    """
    Get data configuration section.
    
    Returns:
        Dict[str, Any]: Data configuration
        
    Example:
        data_config = get_data_config()
        print(data_config['dataset_name'])
    """
    config = load_config()
    return config.config.get('data', {})


def get_value(key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation.
    
    Args:
        key_path (str): Dot-separated key path
        default (Any): Default value if key not found
        
    Returns:
        Any: Configuration value
        
    Example:
        hidden_dim = get_value('model.gnn.hidden_dim', 256)
        batch_size = get_value('training.phase1.batch_size', 32)
    """
    config = load_config()
    return config.get_value(key_path, default)


def update_config(updates: Dict[str, Any]) -> None:
    """
    Update configuration with new values.
    
    Args:
        updates (Dict[str, Any]): Updates to apply
        
    Example:
        update_config({
            'model.gnn.hidden_dim': 512,
            'training.phase1.num_epochs': 100
        })
    """
    config = load_config()
    config.update(updates)


def validate_config() -> Tuple[bool, List[str]]:
    """
    Validate current configuration.
    
    Returns:
        Tuple[bool, List[str]]: (is_valid, error_messages)
        
    Example:
        is_valid, errors = validate_config()
        if not is_valid:
            print(f"Configuration errors: {errors}")
    """
    config = load_config()
    return config.validate()


def reset_config_cache() -> None:
    """
    Reset configuration cache.
    
    Example:
        reset_config_cache()
    """
    global _config_cache
    _config_cache = {}
    logger = get_logger_instance()
    logger.log_info("Configuration cache reset")


def list_config_sections() -> List[str]:
    """
    List all configuration sections.
    
    Returns:
        List[str]: Section names
        
    Example:
        sections = list_config_sections()
        print(f"Available sections: {sections}")
    """
    config = load_config()
    return list(config.config.keys())


def get_config_summary() -> Dict[str, Any]:
    """
    Get a summary of the current configuration.
    
    Returns:
        Dict[str, Any]: Configuration summary
        
    Example:
        summary = get_config_summary()
        print(f"Configuration has {summary['num_sections']} sections")
    """
    config = load_config()
    return {
        'num_sections': len(config.config),
        'sections': list(config.config.keys()),
        'seed': config.seed,
        'device': config.device,
        'version': __version__
    }


def export_config_to_file(
    path: Union[str, Path],
    format: str = 'yaml'
) -> str:
    """
    Export configuration to file.
    
    Args:
        path (str, Path): Output path
        format (str): Output format ('yaml' or 'json')
        
    Returns:
        str: Path to saved file
        
    Example:
        export_config_to_file('config/exported_config.yaml')
        export_config_to_file('config/exported_config.json', format='json')
    """
    config = load_config()
    path = Path(path)
    
    if format.lower() == 'yaml':
        return config.export_to_yaml(path)
    elif format.lower() == 'json':
        return config.export_to_json(path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def create_experiment_config(
    experiment_name: str,
    overrides: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> HGRAGConfig:
    """
    Create an experiment configuration.
    
    Args:
        experiment_name (str): Name of the experiment
        overrides (Dict[str, Any], optional): Configuration overrides
        output_dir (str, Path, optional): Output directory
        
    Returns:
        HGRAGConfig: Experiment configuration
        
    Example:
        config = create_experiment_config(
            experiment_name='ablation_study_1',
            overrides={
                'model.gnn.hidden_dim': 128,
                'training.phase1.num_epochs': 25
            },
            output_dir='experiments/ablation_1'
        )
    """
    config = load_config()
    
    # Update config with experiment name
    config.update({
        'experiment': {
            'name': experiment_name,
            'created': datetime.now().isoformat()
        }
    })
    
    # Apply overrides
    if overrides:
        config.update(overrides)
    
    # Save experiment config
    if output_dir:
        output_path = Path(output_dir) / 'experiment_config.yaml'
        config.save(output_path)
    
    return config


def load_config_from_dict(config_dict: Dict[str, Any]) -> HGRAGConfig:
    """
    Load configuration from dictionary.
    
    Args:
        config_dict (Dict[str, Any]): Configuration dictionary
        
    Returns:
        HGRAGConfig: Configuration instance
        
    Example:
        config_dict = {
            'model': {'gnn': {'hidden_dim': 256}},
            'training': {'phase1': {'num_epochs': 50}}
        }
        config = load_config_from_dict(config_dict)
    """
    # Create temporary file
    import tempfile
    import yaml
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        temp_path = f.name
    
    try:
        config = load_config(temp_path, load_defaults=False)
        return config
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================================
# Configuration Helper Functions
# ============================================================================

def get_phase_config(phase: str) -> Dict[str, Any]:
    """
    Get configuration for a specific training phase.
    
    Args:
        phase (str): Phase name ('phase1', 'phase2', 'phase3')
        
    Returns:
        Dict[str, Any]: Phase configuration
        
    Example:
        phase1_config = get_phase_config('phase1')
        print(phase1_config['num_epochs'])
    """
    training_config = get_training_config()
    return training_config.get(phase, {})


def get_gnn_config() -> Dict[str, Any]:
    """
    Get GNN configuration.
    
    Returns:
        Dict[str, Any]: GNN configuration
        
    Example:
        gnn_config = get_gnn_config()
        print(gnn_config['hidden_dim'])
    """
    model_config = get_model_config()
    return model_config.get('gnn', {})


def get_hybrid_config() -> Dict[str, Any]:
    """
    Get hybrid configuration.
    
    Returns:
        Dict[str, Any]: Hybrid configuration
        
    Example:
        hybrid_config = get_hybrid_config()
        print(hybrid_config['gate_threshold'])
    """
    model_config = get_model_config()
    return model_config.get('hybrid', {})


def get_distillation_config() -> Dict[str, Any]:
    """
    Get distillation configuration.
    
    Returns:
        Dict[str, Any]: Distillation configuration
        
    Example:
        distill_config = get_distillation_config()
        print(distill_config['alpha'])
    """
    model_config = get_model_config()
    return model_config.get('distillation', {})


def is_debug_mode() -> bool:
    """
    Check if debug mode is enabled.
    
    Returns:
        bool: Whether debug mode is enabled
        
    Example:
        if is_debug_mode():
            print("Debug mode enabled")
    """
    return get_value('debug', False)


def get_log_level() -> str:
    """
    Get logging level.
    
    Returns:
        str: Logging level
        
    Example:
        log_level = get_log_level()
        print(f"Log level: {log_level}")
    """
    return get_value('logging.level', 'INFO')


# ============================================================================
# Package initialization
# ============================================================================

# Import datetime for configuration
from datetime import datetime

# Package exports
__all__ = [
    # Main classes
    'HGRAGConfig',
    'ConfigSection',
    
    # Configuration loading functions
    'load_config',
    'get_default_config',
    'load_config_from_dict',
    
    # Section access functions
    'get_model_config',
    'get_training_config',
    'get_evaluation_config',
    'get_data_config',
    'get_phase_config',
    'get_gnn_config',
    'get_hybrid_config',
    'get_distillation_config',
    
    # Value access functions
    'get_value',
    'update_config',
    
    # Configuration management
    'validate_config',
    'reset_config_cache',
    'list_config_sections',
    'get_config_summary',
    'export_config_to_file',
    'create_experiment_config',
    
    # Utility functions
    'is_debug_mode',
    'get_log_level',
    
    # Constants
    'DEFAULT_CONFIG_PATH',
    'DATASET_CONFIG_PATH',
    'MODEL_CONFIG_PATH',
    'TRAINING_CONFIG_PATH',
    '__version__',
    '__author__',
    '__description__'
]


# ============================================================================
# Module initialization
# ============================================================================

# Load default configuration on import
_default_config = None


def _initialize():
    """Initialize the config package."""
    global _default_config
    
    # Load default config
    try:
        _default_config = load_config(use_cache=True)
        logger = get_logger_instance()
        logger.log_info("Config package initialized successfully")
    except Exception as e:
        logger = get_logger_instance()
        logger.log_error(f"Failed to initialize config package: {e}")


# Initialize on import
_initialize()


if __name__ == "__main__":
    # Test the config package
    print("Testing config package...")
    print(f"Version: {__version__}")
    
    # Load configuration
    config = load_config()
    print(f"Configuration loaded: {config}")
    
    # Access sections
    print(f"Model section: {config.model}")
    print(f"Training section: {config.training}")
    print(f"Evaluation section: {config.evaluation}")
    
    # Access values
    print(f"Seed: {config.seed}")
    print(f"Device: {config.device}")
    
    # Test value access
    hidden_dim = config.get_value('model.gnn.hidden_dim', 256)
    print(f"GNN hidden_dim: {hidden_dim}")
    
    # Test list sections
    sections = list_config_sections()
    print(f"Sections: {sections}")
    
    # Test summary
    summary = get_config_summary()
    print(f"Summary: {summary}")
    
    print("\nConfig package test completed successfully!")