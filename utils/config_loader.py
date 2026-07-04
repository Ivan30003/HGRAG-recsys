"""
utils/config_loader.py

Comprehensive configuration management module for H-GRAGrecsys with support for:
- Loading YAML configuration files with inheritance
- Environment variable interpolation
- Configuration validation against schemas
- Dynamic configuration updates
- Configuration merging and overriding
- Experiment configuration management
- Command-line argument integration
"""

import os
import sys
import json
import yaml
import copy
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from collections import defaultdict
import re

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import logger for configuration logging
from utils.logger import get_logger


class ConfigLoader:
    """
    Main configuration loader for H-GRAGrecsys with advanced features.
    
    Features:
    - YAML configuration loading with inheritance
    - Environment variable substitution
    - Configuration validation against schemas
    - Dynamic updates and overrides
    - Support for multiple config files
    - Command-line argument integration
    - Configuration saving and exporting
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        load_defaults: bool = True,
        env_prefix: str = "H_GRAG_",
        logger: Optional['Logger'] = None
    ):
        """
        Initialize the configuration loader.
        
        Args:
            config_path (str, Path, optional): Path to main configuration file
            load_defaults (bool): Whether to load default configuration
            env_prefix (str): Prefix for environment variables
            logger (Logger, optional): Logger instance for logging
        """
        self.config_path = Path(config_path) if config_path else None
        self.env_prefix = env_prefix
        self.logger = logger or get_logger(
            log_dir="logs/config",
            name="config_loader",
            verbose=False
        )
        
        # Store loaded configurations
        self.config = {}
        self.raw_config = {}
        self.config_sources = []
        self.schema = {}
        
        # Track changes
        self.modification_history = []
        
        # Load default config if requested
        if load_defaults:
            self._load_default_config()
        
        # Load main config if provided
        if self.config_path and self.config_path.exists():
            self.load_config(str(self.config_path))
        
        self.logger.log_info(f"ConfigLoader initialized with {'defaults' if load_defaults else 'no defaults'}")
    
    def _load_default_config(self) -> None:
        """
        Load default configuration from package or built-in defaults.
        """
        # Define default configuration structure
        default_config = {
            'model': {
                'agent': {
                    'memory_buffer_size': 10,
                    'embedding_dim': 1536,
                    'consistency_threshold': 0.15,
                    'learning_rate': 1e-4
                },
                'graph': {
                    'edge_update_rate': 0.1,
                    'pruning_threshold': 0.05,
                    'user_similarity_threshold': 0.7,
                    'item_similarity_threshold': 0.6,
                    'co_interaction_threshold': 3,
                    'ppr_restart_prob': 0.15,
                    'max_hops': 2
                },
                'gnn': {
                    'hidden_dim': 256,
                    'num_layers': 3,
                    'num_heads': 4,
                    'dropout': 0.1,
                    'activation': 'relu'
                },
                'distillation': {
                    'component_weights': [1.0, 1.0, 1.0],
                    'alpha': 0.5,
                    'beta': 0.3,
                    'gamma': 0.2,
                    'temperature': 0.07,
                    'num_epochs': 30
                },
                'hybrid': {
                    'gate_threshold': 0.3,
                    'staleness_lambda': 0.1,
                    'uniform_llm_rate': 0.15,
                    'num_epochs': 20
                }
            },
            'training': {
                'phase1': {
                    'num_epochs': 50,
                    'batch_size': 32,
                    'learning_rate': 1e-4,
                    'easy_threshold': 10,
                    'gradient_clip': 1.0
                },
                'phase2': {
                    'num_epochs': 30,
                    'batch_size': 64,
                    'learning_rate': 1e-4,
                    'gradient_clip': 1.0
                },
                'phase3': {
                    'num_epochs': 20,
                    'batch_size': 32,
                    'learning_rate': 5e-5,
                    'gradient_clip': 1.0
                }
            },
            'evaluation': {
                'k_values': [1, 5, 10],
                'num_negatives': 99,
                'num_repetitions': 3,
                'seed': 42,
                'save_predictions': False
            },
            'data': {
                'max_text_length': 512,
                'min_interactions': 5,
                'validation_ratio': 0.1,
                'test_ratio': 0.2,
                'batch_size': 32
            },
            'logging': {
                'log_dir': 'logs',
                'max_bytes': 10485760,
                'backup_count': 5,
                'verbose': True,
                'save_interval': 100
            },
            'experiment': {
                'name': 'h_grag_recsys',
                'description': 'Hierarchical Graph RAG for Recommender Systems',
                'seed': 42,
                'device': 'cuda'
            }
        }
        
        self.config = copy.deepcopy(default_config)
        self.raw_config = copy.deepcopy(default_config)
        self.config_sources.append('default')
        self.logger.log_debug("Loaded default configuration")
    
    def load_config(
        self,
        config_path: Union[str, Path],
        merge: bool = True,
        inherit_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path (str, Path): Path to configuration file
            merge (bool): Whether to merge with existing config
            inherit_from (str, optional): Base config to inherit from
            
        Returns:
            Dict[str, Any]: Loaded configuration dictionary
        
        Example:
            config = loader.load_config('config/model_config.yaml', merge=True)
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            error_msg = f"Configuration file not found: {config_path}"
            self.logger.log_error(error_msg)
            raise FileNotFoundError(error_msg)
        
        self.logger.log_info(f"Loading configuration from: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.logger.log_error(f"Error parsing YAML file: {e}")
            raise ValueError(f"Invalid YAML in {config_path}: {e}")
        
        # Handle inheritance if specified
        if inherit_from:
            base_config = self.load_config(inherit_from, merge=False)
            loaded_config = self._merge_configs(base_config, loaded_config)
        
        # Process environment variables
        loaded_config = self._process_env_vars(loaded_config)
        
        # Update config
        if merge:
            self.config = self._merge_configs(self.config, loaded_config)
        else:
            self.config = loaded_config
        
        # Store raw config and source
        self.raw_config = loaded_config
        self.config_sources.append(str(config_path))
        
        # Log the loaded configuration
        self.logger.log_info(f"Configuration loaded successfully from {config_path}")
        self.logger.log_debug(f"Config keys: {list(self.config.keys())}")
        
        return self.config
    
    def _process_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process environment variable placeholders in configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
            
        Returns:
            Dict[str, Any]: Configuration with environment variables substituted
        
        Example:
            # In YAML: model_path: ${MODEL_PATH:/default/path}
            config = loader._process_env_vars(config)
        """
        def _replace_env_vars(value):
            if isinstance(value, str):
                # Find all ${ENV_VAR} or ${ENV_VAR:default} patterns
                pattern = r'\$\{([^:}]+)(?::([^}]*))?\}'
                matches = re.findall(pattern, value)
                
                for env_var, default in matches:
                    env_value = os.environ.get(env_var, default if default else '')
                    # Replace all occurrences
                    if env_value:
                        full_pattern = f'${{{env_var}' + (f':{default}' if default else '') + '}'
                        value = value.replace(full_pattern, env_value)
                
                return value
            
            elif isinstance(value, dict):
                return {k: _replace_env_vars(v) for k, v in value.items()}
            
            elif isinstance(value, list):
                return [_replace_env_vars(item) for item in value]
            
            else:
                return value
        
        return _replace_env_vars(config)
    
    def _merge_configs(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any],
        deep: bool = True
    ) -> Dict[str, Any]:
        """
        Deep merge two configuration dictionaries.
        
        Args:
            base (Dict[str, Any]): Base configuration
            override (Dict[str, Any]): Override configuration
            deep (bool): Whether to perform deep merge
            
        Returns:
            Dict[str, Any]: Merged configuration
        
        Example:
            merged = loader._merge_configs(base_config, user_config)
        """
        if not deep:
            return {**base, **override}
        
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                # Recursively merge dictionaries
                result[key] = self._merge_configs(result[key], value, deep)
            else:
                # Override or add value
                result[key] = copy.deepcopy(value)
        
        return result
    
    def get_config_section(self, section: str) -> Dict[str, Any]:
        """
        Get a specific section of the configuration.
        
        Args:
            section (str): Section name (e.g., 'model', 'training', 'evaluation')
            
        Returns:
            Dict[str, Any]: Configuration section
        
        Example:
            model_config = loader.get_config_section('model')
            training_config = loader.get_config_section('training')
        """
        if section not in self.config:
            self.logger.log_warning(f"Section '{section}' not found in configuration")
            return {}
        
        return self.config[section]
    
    def get_value(
        self,
        key_path: str,
        default: Any = None,
        raise_error: bool = False
    ) -> Any:
        """
        Get a specific value using dot notation.
        
        Args:
            key_path (str): Dot-separated key path (e.g., 'model.agent.embedding_dim')
            default (Any): Default value if key not found
            raise_error (bool): Whether to raise KeyError if key not found
            
        Returns:
            Any: Configuration value
        
        Example:
            embedding_dim = loader.get_value('model.agent.embedding_dim', default=768)
            threshold = loader.get_value('model.graph.pruning_threshold')
        """
        try:
            parts = key_path.split('.')
            value = self.config
            for part in parts:
                if not isinstance(value, dict):
                    raise KeyError(f"Cannot traverse '{part}' in non-dict value")
                value = value[part]
            return value
        except (KeyError, TypeError) as e:
            if raise_error:
                self.logger.log_error(f"Key '{key_path}' not found in configuration")
                raise KeyError(f"Key '{key_path}' not found in configuration") from e
            else:
                self.logger.log_debug(f"Key '{key_path}' not found, returning default: {default}")
                return default
    
    def update_config(
        self,
        updates: Dict[str, Any],
        section: Optional[str] = None,
        merge: bool = True
    ) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates (Dict[str, Any]): Updates to apply
            section (str, optional): Specific section to update
            merge (bool): Whether to merge or replace
        
        Example:
            loader.update_config({'learning_rate': 1e-3}, section='training.phase1')
            loader.update_config({'model.gnn.hidden_dim': 512})
        """
        if section:
            updates = {section: updates}
        
        if merge:
            self.config = self._merge_configs(self.config, updates)
        else:
            self.config = updates
        
        self.modification_history.append({
            'timestamp': 'now',
            'updates': updates,
            'merge': merge
        })
        
        self.logger.log_info(f"Configuration updated with {len(updates)} items")
    
    def set_value(self, key_path: str, value: Any) -> None:
        """
        Set a specific value using dot notation.
        
        Args:
            key_path (str): Dot-separated key path
            value (Any): Value to set
        
        Example:
            loader.set_value('model.agent.embedding_dim', 1024)
            loader.set_value('training.phase1.batch_size', 64)
        """
        parts = key_path.split('.')
        current = self.config
        
        # Navigate to the parent
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set the value
        current[parts[-1]] = value
        
        # Log the change
        self.modification_history.append({
            'timestamp': 'now',
            'key': key_path,
            'value': value
        })
        
        self.logger.log_info(f"Set {key_path} = {value}")
    
    def load_multiple_configs(
        self,
        config_paths: List[Union[str, Path]],
        merge: bool = True
    ) -> Dict[str, Any]:
        """
        Load and merge multiple configuration files.
        
        Args:
            config_paths (List[str, Path]): List of configuration file paths
            merge (bool): Whether to merge with existing config
            
        Returns:
            Dict[str, Any]: Merged configuration
        
        Example:
            configs = loader.load_multiple_configs([
                'config/default_config.yaml',
                'config/dataset_config.yaml',
                'config/model_config.yaml'
            ])
        """
        result = {}
        
        for path in config_paths:
            if Path(path).exists():
                self.load_config(path, merge=merge)
            else:
                self.logger.log_warning(f"Configuration file not found: {path}")
        
        return self.config
    
    def validate_config(self, schema: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
        """
        Validate configuration against a schema.
        
        Args:
            schema (Dict[str, Any], optional): Validation schema
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        
        Example:
            schema = {
                'model': {
                    'required': ['agent', 'graph', 'gnn'],
                    'properties': {
                        'agent': {'type': 'dict'},
                        'graph': {'type': 'dict'},
                        'gnn': {'type': 'dict'}
                    }
                }
            }
            is_valid, errors = loader.validate_config(schema)
        """
        if schema is None and not self.schema:
            self.logger.log_warning("No schema provided for validation")
            return True, []
        
        schema = schema or self.schema
        errors = []
        
        def _validate_section(config_section, schema_section, path):
            # Check required keys
            if 'required' in schema_section:
                required_keys = schema_section['required']
                for req_key in required_keys:
                    if req_key not in config_section:
                        errors.append(f"Missing required key: {path}.{req_key}")
            
            # Check property types
            if 'properties' in schema_section:
                for prop_key, prop_schema in schema_section['properties'].items():
                    if prop_key in config_section:
                        # Check type
                        if 'type' in prop_schema:
                            expected_type = prop_schema['type']
                            actual_value = config_section[prop_key]
                            
                            # Map YAML types to Python types
                            type_map = {
                                'str': str,
                                'int': int,
                                'float': float,
                                'list': list,
                                'dict': dict,
                                'bool': bool,
                                'number': (int, float)
                            }
                            
                            if expected_type in type_map:
                                expected_types = type_map[expected_type]
                                if not isinstance(actual_value, expected_types):
                                    errors.append(
                                        f"Invalid type for {path}.{prop_key}: "
                                        f"expected {expected_type}, got {type(actual_value).__name__}"
                                    )
                        
                        # Check enum values
                        if 'enum' in prop_schema:
                            if config_section[prop_key] not in prop_schema['enum']:
                                errors.append(
                                    f"Invalid value for {path}.{prop_key}: "
                                    f"{config_section[prop_key]} not in {prop_schema['enum']}"
                                )
                        
                        # Recursively validate nested dictionaries
                        if (isinstance(config_section[prop_key], dict) and 
                            'properties' in prop_schema):
                            _validate_section(
                                config_section[prop_key],
                                prop_schema,
                                f"{path}.{prop_key}"
                            )
        
        # Validate root section
        _validate_section(self.config, schema, 'root')
        
        if errors:
            self.logger.log_warning(f"Configuration validation failed with {len(errors)} errors")
            for error in errors:
                self.logger.log_warning(f"  - {error}")
            return False, errors
        else:
            self.logger.log_info("Configuration validation passed")
            return True, []
    
    def save_config(
        self,
        config: Optional[Dict[str, Any]] = None,
        path: Optional[Union[str, Path]] = None,
        format: str = 'yaml',
        include_comments: bool = True
    ) -> str:
        """
        Save configuration to a file.
        
        Args:
            config (Dict[str, Any], optional): Configuration to save
            path (str, Path, optional): Output path
            format (str): Output format ('yaml' or 'json')
            include_comments (bool): Whether to include comments in YAML
            
        Returns:
            str: Path to the saved file
        
        Example:
            # Save current config
            loader.save_config(path='config/exported_config.yaml')
            
            # Save specific config
            loader.save_config(config, path='config/model_config.yaml')
        """
        config_to_save = config or self.config
        
        if path is None:
            path = Path('config') / f"exported_config_{'yaml' if format == 'yaml' else 'json'}"
        else:
            path = Path(path)
        
        # Create parent directories
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'yaml':
            with open(path, 'w', encoding='utf-8') as f:
                if include_comments:
                    # Add header comment
                    f.write("# H-GRAGrecsys Configuration\n")
                    f.write(f"# Generated: {import_datetime()}\n\n")
                
                yaml.dump(
                    config_to_save,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False
                )
        elif format.lower() == 'json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(
                    config_to_save,
                    f,
                    indent=2,
                    sort_keys=False,
                    default=str
                )
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.log_info(f"Configuration saved to: {path}")
        return str(path)
    
    def export_to_env(self, prefix: Optional[str] = None) -> Dict[str, str]:
        """
        Export configuration to environment variables.
        
        Args:
            prefix (str, optional): Prefix for environment variables
            
        Returns:
            Dict[str, str]: Map of environment variable names to values
        
        Example:
            env_vars = loader.export_to_env(prefix='H_GRAG_')
        """
        prefix = prefix or self.env_prefix
        env_vars = {}
        
        def _flatten_config(config, parent_key=''):
            items = []
            for key, value in config.items():
                new_key = f"{parent_key}_{key.upper()}" if parent_key else key.upper()
                if isinstance(value, dict):
                    items.extend(_flatten_config(value, new_key))
                else:
                    env_name = f"{prefix}{new_key}"
                    env_vars[env_name] = str(value)
                    items.append((env_name, str(value)))
            return items
        
        _flatten_config(self.config)
        self.logger.log_info(f"Exported {len(env_vars)} configuration items to environment variables")
        return env_vars
    
    def from_env(self, prefix: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Args:
            prefix (str, optional): Prefix for environment variables
            
        Returns:
            Dict[str, Any]: Configuration loaded from environment
        
        Example:
            config = loader.from_env(prefix='H_GRAG_')
        """
        prefix = prefix or self.env_prefix
        config = {}
        
        for env_name, env_value in os.environ.items():
            if env_name.startswith(prefix):
                # Remove prefix and split into keys
                key_path = env_name[len(prefix):].lower().split('_')
                
                # Parse value
                try:
                    # Try to parse as JSON
                    value = json.loads(env_value)
                except json.JSONDecodeError:
                    # Try to parse as bool
                    if env_value.lower() in ('true', 'false'):
                        value = env_value.lower() == 'true'
                    # Try to parse as number
                    elif env_value.replace('.', '', 1).isdigit():
                        value = float(env_value) if '.' in env_value else int(env_value)
                    else:
                        value = env_value
                
                # Set nested value
                current = config
                for part in key_path[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[key_path[-1]] = value
        
        self.logger.log_info(f"Loaded {len(os.environ)} environment variables, found {len(config)} config items")
        return config
    
    def get_modified_sections(self) -> List[str]:
        """
        Get sections that have been modified from default.
        
        Returns:
            List[str]: List of modified section names
        """
        modified = []
        
        def _compare_configs(default, current, path=''):
            if isinstance(default, dict) and isinstance(current, dict):
                # Check for new keys or different values
                all_keys = set(default.keys()) | set(current.keys())
                for key in all_keys:
                    new_path = f"{path}.{key}" if path else key
                    if key not in default:
                        modified.append(new_path)
                    elif key not in current:
                        modified.append(new_path)
                    else:
                        if isinstance(default[key], dict) and isinstance(current[key], dict):
                            _compare_configs(default[key], current[key], new_path)
                        elif default[key] != current[key]:
                            modified.append(new_path)
        
        _compare_configs(self.raw_config, self.config)
        return modified
    
    def create_experiment_config(
        self,
        experiment_name: str,
        overrides: Dict[str, Any] = None,
        experiment_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Create an experiment configuration with specific overrides.
        
        Args:
            experiment_name (str): Name of the experiment
            overrides (Dict[str, Any], optional): Configuration overrides
            experiment_dir (str, Path, optional): Directory to save experiment config
            
        Returns:
            Dict[str, Any]: Experiment configuration
        
        Example:
            exp_config = loader.create_experiment_config(
                experiment_name='ablation_study_1',
                overrides={
                    'model.gnn.hidden_dim': 128,
                    'training.phase1.num_epochs': 25
                },
                experiment_dir='experiments/ablation_study_1'
            )
        """
        # Deep copy current config
        experiment_config = copy.deepcopy(self.config)
        
        # Set experiment metadata
        if 'experiment' not in experiment_config:
            experiment_config['experiment'] = {}
        
        experiment_config['experiment']['name'] = experiment_name
        experiment_config['experiment']['created'] = import_datetime()
        experiment_config['experiment']['base_config'] = str(self.config_path) if self.config_path else 'default'
        
        # Apply overrides
        if overrides:
            for key, value in overrides.items():
                self.set_value_in_dict(experiment_config, key, value)
        
        # Save if directory provided
        if experiment_dir:
            experiment_dir = Path(experiment_dir)
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            config_path = experiment_dir / 'experiment_config.yaml'
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(experiment_config, f, default_flow_style=False)
            
            self.logger.log_info(f"Experiment configuration saved to: {config_path}")
        
        return experiment_config
    
    def set_value_in_dict(self, config: Dict[str, Any], key_path: str, value: Any) -> None:
        """
        Set a value in a dictionary using dot notation.
        
        Args:
            config (Dict[str, Any]): Dictionary to modify
            key_path (str): Dot-separated key path
            value (Any): Value to set
        
        Example:
            loader.set_value_in_dict(config, 'model.gnn.hidden_dim', 512)
        """
        parts = key_path.split('.')
        current = config
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    def get_diff(self, other_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get differences between this config and another config.
        
        Args:
            other_config (Dict[str, Any]): Other configuration to compare
            
        Returns:
            Dict[str, Any]: Dictionary containing differences
        
        Example:
            diff = loader.get_diff(other_config)
            # diff contains:
            # - added: keys present in other but not in this
            # - removed: keys present in this but not in other
            # - changed: keys with different values
        """
        def _get_diff(d1, d2, path=''):
            diff = {'added': {}, 'removed': {}, 'changed': {}}
            
            if not isinstance(d1, dict) or not isinstance(d2, dict):
                if d1 != d2:
                    return {'changed': {path: {'old': d1, 'new': d2}}}
                return diff
            
            # Check for keys in both
            all_keys = set(d1.keys()) | set(d2.keys())
            
            for key in all_keys:
                new_path = f"{path}.{key}" if path else key
                
                if key not in d1:
                    diff['added'][new_path] = d2[key]
                elif key not in d2:
                    diff['removed'][new_path] = d1[key]
                else:
                    if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                        sub_diff = _get_diff(d1[key], d2[key], new_path)
                        for category in sub_diff:
                            if sub_diff[category]:
                                diff[category].update(sub_diff[category])
                    elif d1[key] != d2[key]:
                        diff['changed'][new_path] = {'old': d1[key], 'new': d2[key]}
            
            return diff
        
        return _get_diff(self.config, other_config)
    
    def reset_to_default(self) -> None:
        """
        Reset configuration to default values.
        """
        self._load_default_config()
        self.modification_history.append({
            'timestamp': 'now',
            'action': 'reset_to_default'
        })
        self.logger.log_info("Configuration reset to default")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current configuration.
        
        Returns:
            Dict[str, Any]: Configuration summary
        
        Example:
            summary = loader.get_config_summary()
            # Returns: {'sections': [...], 'modified': [...], 'size': 42}
        """
        return {
            'sections': list(self.config.keys()),
            'modified_sections': self.get_modified_sections(),
            'config_sources': self.config_sources,
            'num_keys': len(self._flatten_dict(self.config)),
            'modification_count': len(self.modification_history),
            'config_path': str(self.config_path) if self.config_path else None
        }
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        """
        Flatten a nested dictionary.
        
        Args:
            d (Dict[str, Any]): Dictionary to flatten
            parent_key (str): Parent key for recursion
            
        Returns:
            Dict[str, Any]: Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def __repr__(self) -> str:
        """String representation of the ConfigLoader."""
        return f"ConfigLoader(config_path={self.config_path}, sections={list(self.config.keys())})"


# Helper function to import datetime (to avoid name conflict with datetime module)
def import_datetime():
    """Import and return current datetime as string."""
    from datetime import datetime
    return datetime.now().isoformat()


# Convenience function for quick config loading
def load_config(
    config_path: Optional[Union[str, Path]] = None,
    load_defaults: bool = True,
    env_prefix: str = "H_GRAG_",
    verbose: bool = True
) -> ConfigLoader:
    """
    Factory function to create and return a ConfigLoader instance.
    
    Args:
        config_path (str, Path, optional): Path to configuration file
        load_defaults (bool): Whether to load default configuration
        env_prefix (str): Prefix for environment variables
        verbose (bool): Whether to enable verbose logging
        
    Returns:
        ConfigLoader: Configured ConfigLoader instance
    
    Example:
        loader = load_config('config/default_config.yaml')
        model_config = loader.get_config_section('model')
        gnn_dim = loader.get_value('model.gnn.hidden_dim')
    """
    # Create logger for config loader
    logger = get_logger(
        log_dir="logs/config",
        name="config_loader",
        verbose=verbose
    )
    
    return ConfigLoader(
        config_path=config_path,
        load_defaults=load_defaults,
        env_prefix=env_prefix,
        logger=logger
    )


# Command-line argument parsing helper
def parse_config_args(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """
    Add configuration-related arguments to an ArgumentParser.
    
    Args:
        parser (argparse.ArgumentParser, optional): Existing parser to extend
        
    Returns:
        argparse.ArgumentParser: Parser with config arguments added
    
    Example:
        parser = parse_config_args()
        args = parser.parse_args()
        config = load_config(args.config, load_defaults=not args.no_defaults)
    """
    if parser is None:
        parser = argparse.ArgumentParser(description='H-GRAGrecsys Configuration')
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--no-defaults',
        action='store_true',
        help='Do not load default configuration'
    )
    
    parser.add_argument(
        '--save-config',
        type=str,
        default=None,
        help='Save configuration to specified path after loading'
    )
    
    parser.add_argument(
        '--override',
        action='append',
        nargs=2,
        metavar=('KEY', 'VALUE'),
        help='Override configuration values (e.g., --override model.gnn.hidden_dim 512)'
    )
    
    parser.add_argument(
        '--config-section',
        type=str,
        help='Display specific configuration section'
    )
    
    return parser


# For testing the config loader
if __name__ == "__main__":
    import tempfile
    
    print("Testing ConfigLoader...")
    
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
model:
  agent:
    embedding_dim: 768
    memory_buffer_size: 20
  gnn:
    hidden_dim: 512
    num_layers: 4

training:
  phase1:
    num_epochs: 25
    batch_size: 16

evaluation:
  k_values: [1, 5, 10, 20]
  num_negatives: 50
""")
        temp_config_path = f.name
    
    try:
        # Test loading
        loader = load_config(temp_config_path)
        print(f"Loaded config: {loader}")
        
        # Test get_value
        embedding_dim = loader.get_value('model.agent.embedding_dim')
        print(f"Embedding dim: {embedding_dim}")
        
        # Test get_config_section
        gnn_config = loader.get_config_section('model.gnn')
        print(f"GNN config: {gnn_config}")
        
        # Test update
        loader.update_config({'model.gnn.hidden_dim': 1024})
        print(f"Updated hidden_dim: {loader.get_value('model.gnn.hidden_dim')}")
        
        # Test validation
        schema = {
            'model': {
                'required': ['agent', 'graph', 'gnn'],
                'properties': {
                    'agent': {'type': 'dict'},
                    'graph': {'type': 'dict'},
                    'gnn': {'type': 'dict'}
                }
            }
        }
        is_valid, errors = loader.validate_config(schema)
        print(f"Config valid: {is_valid}")
        
        # Test saving
        saved_path = loader.save_config(path='config/test_saved_config.yaml')
        print(f"Saved config to: {saved_path}")
        
        # Test summary
        summary = loader.get_config_summary()
        print(f"Config summary: {summary}")
        
        # Test experiment config
        exp_config = loader.create_experiment_config(
            experiment_name='test_exp',
            overrides={'model.gnn.dropout': 0.2},
            experiment_dir='experiments/test_exp'
        )
        print(f"Experiment config created with {len(exp_config)} sections")
        
        # Test flatten
        flat = loader._flatten_dict(loader.config)
        print(f"Flattened config has {len(flat)} keys")
        
        print("ConfigLoader tests completed successfully!")
        
    finally:
        # Clean up
        if Path(temp_config_path).exists():
            Path(temp_config_path).unlink()
        
        # Clean up test files
        for path in ['config/test_saved_config.yaml', 'experiments/test_exp']:
            if Path(path).exists():
                if Path(path).is_dir():
                    import shutil
                    shutil.rmtree(path)
                else:
                    Path(path).unlink()