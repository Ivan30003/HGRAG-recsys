"""
Checkpoint Manager Module for H-GRAGrecsys

This module provides comprehensive checkpoint management functionality for
all training phases. It handles saving, loading, and managing training
checkpoints with features like automatic cleanup, best checkpoint tracking,
and metadata management.

Key Responsibilities:
- Save training checkpoints with metadata
- Load checkpoints with validation
- Manage checkpoint lifecycle (create, read, update, delete)
- Track best checkpoints based on metrics
- Maintain checkpoint history and statistics
- Support resuming training from checkpoints
"""

import os
import sys
import json
import shutil
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import pickle
from datetime import datetime
import numpy as np
from collections import defaultdict

# Add project root to path if needed
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Core imports
import torch

# Utils imports
from utils.logger import Logger
from utils.timer import Timer


class CheckpointManager:
    """
    Manager for handling training checkpoints
    
    This class provides a robust interface for managing training checkpoints
    with features like automatic cleanup, metadata tracking, and best checkpoint
    selection based on validation metrics.
    """
    
    def __init__(
        self,
        save_dir: str = './checkpoints',
        max_checkpoints: int = 5,
        keep_best: bool = True,
        checkpoint_extension: str = '.pt',
        logger: Optional[Logger] = None
    ):
        """
        Initialize the checkpoint manager
        
        Args:
            save_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            keep_best: Whether to keep best checkpoints even if exceeding max
            checkpoint_extension: File extension for checkpoint files
            logger: Optional Logger instance
            
        Raises:
            ValueError: If save_dir is invalid or max_checkpoints < 1
        """
        if max_checkpoints < 1:
            raise ValueError("max_checkpoints must be at least 1")
        
        self.save_dir = Path(save_dir)
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        self.checkpoint_extension = checkpoint_extension
        
        # Initialize logger
        self.logger = logger or Logger(
            log_dir=str(self.save_dir.parent / 'logs'),
            name='checkpoint_manager'
        )
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata file
        self.metadata_file = self.save_dir / 'checkpoint_metadata.json'
        self.metadata = self._load_metadata()
        
        # Tracking
        self.checkpoints = []  # List of checkpoint info dicts
        self.best_checkpoints = []  # List of best checkpoint info dicts
        self._update_checkpoint_lists()
        
        self.logger.log_info(f"CheckpointManager initialized with save_dir: {save_dir}")
        self.logger.log_info(f"Max checkpoints: {max_checkpoints}, Keep best: {keep_best}")
    
    def save_checkpoint(
        self,
        state: Dict[str, Any],
        epoch: int,
        step: int = 0,
        name: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        optimizer_metric: Optional[str] = None
    ) -> str:
        """
        Save a training checkpoint
        
        Args:
            state: State dictionary containing model, optimizer, etc.
            epoch: Current epoch number
            step: Current step number
            name: Optional custom name for the checkpoint
            metrics: Dictionary of metrics (loss, accuracy, etc.)
            metadata: Additional metadata to store with checkpoint
            is_best: Whether this checkpoint is the best performing
            optimizer_metric: Metric to optimize (e.g., 'val_loss')
            
        Returns:
            str: Path to the saved checkpoint
            
        Raises:
            RuntimeError: If checkpoint saving fails
        """
        self.logger.log_info(f"Saving checkpoint (epoch {epoch}, step {step})...")
        
        timer = Timer()
        timer.start()
        
        try:
            # Generate checkpoint filename
            if name:
                filename = f"{name}{self.checkpoint_extension}"
            else:
                timestamp = Timer.get_current_timestamp()
                filename = f"checkpoint_epoch{epoch}_step{step}_{timestamp}{self.checkpoint_extension}"
            
            checkpoint_path = self.save_dir / filename
            
            # Prepare checkpoint data
            checkpoint_data = {
                'state': state,
                'epoch': epoch,
                'step': step,
                'timestamp': Timer.get_current_timestamp(),
                'metrics': metrics or {},
                'metadata': metadata or {},
                'is_best': is_best,
                'optimizer_metric': optimizer_metric
            }
            
            # Add hash for integrity checking
            checkpoint_data['hash'] = self._compute_checkpoint_hash(checkpoint_data)
            
            # Save checkpoint
            torch.save(checkpoint_data, checkpoint_path)
            
            # Update metadata
            checkpoint_info = {
                'path': str(checkpoint_path),
                'filename': filename,
                'epoch': epoch,
                'step': step,
                'timestamp': checkpoint_data['timestamp'],
                'metrics': checkpoint_data['metrics'],
                'metadata': checkpoint_data['metadata'],
                'is_best': is_best,
                'size_mb': checkpoint_path.stat().st_size / (1024 * 1024),
                'hash': checkpoint_data['hash']
            }
            
            self.metadata['checkpoints'].append(checkpoint_info)
            self._save_metadata()
            
            # Update checkpoint lists
            self._update_checkpoint_lists()
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            timer.stop()
            self.logger.log_info(
                f"Checkpoint saved to {checkpoint_path} "
                f"in {timer.get_elapsed_time():.2f} seconds"
            )
            
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.log_error(f"Failed to save checkpoint: {e}")
            raise RuntimeError(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(
        self,
        checkpoint_name: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False,
        optimize_metric: Optional[str] = None,
        verify_hash: bool = True
    ) -> Dict[str, Any]:
        """
        Load a checkpoint from disk
        
        Args:
            checkpoint_name: Name of the checkpoint to load (without extension)
            checkpoint_path: Direct path to checkpoint file
            load_best: Whether to load the best checkpoint
            optimize_metric: Metric to use for selecting best checkpoint
            verify_hash: Whether to verify checkpoint integrity
            
        Returns:
            Dict[str, Any]: Loaded checkpoint data
            
        Raises:
            FileNotFoundError: If checkpoint not found
            RuntimeError: If checkpoint loading fails or hash verification fails
        """
        self.logger.log_info("Loading checkpoint...")
        
        timer = Timer()
        timer.start()
        
        try:
            # Determine which checkpoint to load
            if load_best:
                checkpoint_path = self.get_best_checkpoint_path(optimize_metric)
                if checkpoint_path is None:
                    raise FileNotFoundError("No best checkpoint found")
            elif checkpoint_path:
                checkpoint_path = Path(checkpoint_path)
            elif checkpoint_name:
                # Try with and without extension
                path_with_ext = self.save_dir / f"{checkpoint_name}{self.checkpoint_extension}"
                path_without_ext = self.save_dir / checkpoint_name
                
                if path_with_ext.exists():
                    checkpoint_path = path_with_ext
                elif path_without_ext.exists():
                    checkpoint_path = path_without_ext
                else:
                    # Check if it's in the metadata
                    checkpoint_info = self._find_checkpoint_by_name(checkpoint_name)
                    if checkpoint_info:
                        checkpoint_path = Path(checkpoint_info['path'])
                    else:
                        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")
            else:
                # Load latest checkpoint
                latest = self.get_latest_checkpoint_path()
                if latest:
                    checkpoint_path = Path(latest)
                else:
                    raise FileNotFoundError("No checkpoints found")
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # Verify hash
            if verify_hash and 'hash' in checkpoint_data:
                computed_hash = self._compute_checkpoint_hash(checkpoint_data)
                if computed_hash != checkpoint_data['hash']:
                    self.logger.log_warning("Checkpoint hash verification failed!")
                    if not self._should_ignore_hash_failure(checkpoint_data):
                        raise RuntimeError("Checkpoint integrity check failed")
                else:
                    self.logger.log_info("Checkpoint hash verified successfully")
            
            timer.stop()
            self.logger.log_info(
                f"Checkpoint loaded from {checkpoint_path} "
                f"in {timer.get_elapsed_time():.2f} seconds"
            )
            
            return checkpoint_data
            
        except Exception as e:
            self.logger.log_error(f"Failed to load checkpoint: {e}")
            raise
    
    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest checkpoint information
        
        Returns:
            Optional[Dict[str, Any]]: Latest checkpoint info or None if none exist
        """
        if not self.metadata['checkpoints']:
            return None
        
        latest = self._get_latest_checkpoint_info()
        return latest
    
    def get_latest_checkpoint_path(self) -> Optional[str]:
        """
        Get the path to the latest checkpoint
        
        Returns:
            Optional[str]: Path to the latest checkpoint or None if none exist
        """
        latest = self.get_latest_checkpoint()
        if latest:
            return latest['path']
        return None
    
    def get_best_checkpoint(
        self,
        optimize_metric: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best checkpoint information
        
        Args:
            optimize_metric: Metric to use for selecting best checkpoint.
                           If None, uses the first metric found.
        
        Returns:
            Optional[Dict[str, Any]]: Best checkpoint info or None if none exist
        """
        if not self.metadata['checkpoints']:
            return None
        
        best_info = self._find_best_checkpoint(optimize_metric)
        return best_info
    
    def get_best_checkpoint_path(
        self,
        optimize_metric: Optional[str] = None
    ) -> Optional[str]:
        """
        Get the path to the best checkpoint
        
        Args:
            optimize_metric: Metric to use for selecting best checkpoint
        
        Returns:
            Optional[str]: Path to the best checkpoint or None if none exist
        """
        best = self.get_best_checkpoint(optimize_metric)
        if best:
            return best['path']
        return None
    
    def delete_checkpoint(
        self,
        checkpoint_name: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        delete_best: bool = False
    ) -> bool:
        """
        Delete a checkpoint
        
        Args:
            checkpoint_name: Name of the checkpoint to delete
            checkpoint_path: Direct path to checkpoint file
            delete_best: Whether to allow deleting best checkpoints
            
        Returns:
            bool: True if checkpoint was deleted, False otherwise
            
        Raises:
            ValueError: If neither checkpoint_name nor checkpoint_path is provided
        """
        if not checkpoint_name and not checkpoint_path:
            raise ValueError("Either checkpoint_name or checkpoint_path must be provided")
        
        self.logger.log_info(f"Deleting checkpoint...")
        
        try:
            # Find checkpoint path
            if checkpoint_path:
                path_to_delete = Path(checkpoint_path)
            else:
                checkpoint_info = self._find_checkpoint_by_name(checkpoint_name)
                if not checkpoint_info:
                    self.logger.log_warning(f"Checkpoint not found: {checkpoint_name}")
                    return False
                path_to_delete = Path(checkpoint_info['path'])
            
            # Check if it's a best checkpoint
            if not delete_best:
                is_best = self._is_checkpoint_best(path_to_delete)
                if is_best:
                    self.logger.log_warning(
                        f"Cannot delete best checkpoint: {path_to_delete}. "
                        "Use delete_best=True to force deletion."
                    )
                    return False
            
            # Delete file
            if path_to_delete.exists():
                path_to_delete.unlink()
                self.logger.log_info(f"Deleted checkpoint: {path_to_delete}")
            else:
                self.logger.log_warning(f"Checkpoint file not found: {path_to_delete}")
                return False
            
            # Remove from metadata
            self.metadata['checkpoints'] = [
                c for c in self.metadata['checkpoints']
                if Path(c['path']) != path_to_delete
            ]
            self._save_metadata()
            
            # Update checkpoint lists
            self._update_checkpoint_lists()
            
            return True
            
        except Exception as e:
            self.logger.log_error(f"Failed to delete checkpoint: {e}")
            return False
    
    def cleanup(self) -> int:
        """
        Clean up all checkpoints and metadata
        
        Returns:
            int: Number of files deleted
        
        Raises:
            RuntimeError: If cleanup fails
        """
        self.logger.log_info("Cleaning up checkpoints...")
        
        try:
            # Delete checkpoint files
            deleted_count = 0
            for checkpoint_info in self.metadata['checkpoints']:
                path = Path(checkpoint_info['path'])
                if path.exists():
                    path.unlink()
                    deleted_count += 1
            
            # Delete metadata file
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            
            # Reset metadata
            self.metadata = self._create_empty_metadata()
            self._save_metadata()
            
            # Reset checkpoint lists
            self.checkpoints = []
            self.best_checkpoints = []
            
            self.logger.log_info(f"Cleaned up {deleted_count} checkpoint files")
            
            return deleted_count
            
        except Exception as e:
            self.logger.log_error(f"Failed to cleanup checkpoints: {e}")
            raise RuntimeError(f"Failed to cleanup checkpoints: {e}")
    
    def get_checkpoint_info(
        self,
        checkpoint_name: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get information about a checkpoint
        
        Args:
            checkpoint_name: Name of the checkpoint
            checkpoint_path: Direct path to checkpoint file
            
        Returns:
            Optional[Dict[str, Any]]: Checkpoint information or None if not found
        """
        if checkpoint_name:
            return self._find_checkpoint_by_name(checkpoint_name)
        elif checkpoint_path:
            return self._find_checkpoint_by_path(checkpoint_path)
        else:
            return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all checkpoints with their information
        
        Returns:
            List[Dict[str, Any]]: List of checkpoint information dictionaries
        """
        return sorted(
            self.metadata['checkpoints'],
            key=lambda x: x['timestamp'],
            reverse=True
        )
    
    def get_checkpoint_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about checkpoints
        
        Returns:
            Dict[str, Any]: Checkpoint statistics
        """
        stats = {
            'total_checkpoints': len(self.metadata['checkpoints']),
            'total_size_mb': 0.0,
            'best_checkpoints': len(self.best_checkpoints),
            'oldest_checkpoint': None,
            'newest_checkpoint': None,
            'avg_epoch': 0,
            'min_epoch': float('inf'),
            'max_epoch': 0,
            'metrics_available': defaultdict(list)
        }
        
        if not self.metadata['checkpoints']:
            return stats
        
        checkpoints = self.metadata['checkpoints']
        
        # Compute statistics
        total_size = 0.0
        total_epoch = 0
        
        for checkpoint in checkpoints:
            size = checkpoint.get('size_mb', 0)
            total_size += size
            
            epoch = checkpoint.get('epoch', 0)
            total_epoch += epoch
            
            if epoch < stats['min_epoch']:
                stats['min_epoch'] = epoch
            if epoch > stats['max_epoch']:
                stats['max_epoch'] = epoch
            
            # Collect available metrics
            for metric_name, metric_value in checkpoint.get('metrics', {}).items():
                stats['metrics_available'][metric_name].append(metric_value)
        
        stats['total_size_mb'] = total_size
        stats['avg_epoch'] = total_epoch / len(checkpoints) if checkpoints else 0
        stats['oldest_checkpoint'] = checkpoints[-1] if checkpoints else None
        stats['newest_checkpoint'] = checkpoints[0] if checkpoints else None
        
        # Compute metric statistics
        for metric_name, values in stats['metrics_available'].items():
            if values:
                stats['metrics_available'][metric_name] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'count': len(values)
                }
        
        return stats
    
    def export_metadata(self, export_path: str) -> str:
        """
        Export checkpoint metadata to a file
        
        Args:
            export_path: Path to export metadata
            
        Returns:
            str: Path where metadata was exported
        """
        self.logger.log_info(f"Exporting metadata to {export_path}...")
        
        try:
            with open(export_path, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
            
            self.logger.log_info(f"Metadata exported to {export_path}")
            return export_path
            
        except Exception as e:
            self.logger.log_error(f"Failed to export metadata: {e}")
            raise RuntimeError(f"Failed to export metadata: {e}")
    
    def import_metadata(self, import_path: str) -> Dict[str, Any]:
        """
        Import checkpoint metadata from a file
        
        Args:
            import_path: Path to import metadata from
            
        Returns:
            Dict[str, Any]: Imported metadata
            
        Raises:
            FileNotFoundError: If import_path doesn't exist
            RuntimeError: If metadata import fails
        """
        if not os.path.exists(import_path):
            raise FileNotFoundError(f"Metadata file not found: {import_path}")
        
        self.logger.log_info(f"Importing metadata from {import_path}...")
        
        try:
            with open(import_path, 'r') as f:
                metadata = json.load(f)
            
            self.metadata = metadata
            self._save_metadata()
            self._update_checkpoint_lists()
            
            self.logger.log_info(f"Metadata imported from {import_path}")
            return metadata
            
        except Exception as e:
            self.logger.log_error(f"Failed to import metadata: {e}")
            raise RuntimeError(f"Failed to import metadata: {e}")
    
    # Private methods
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file or create empty"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                self.logger.log_warning("Failed to load metadata, creating new")
                return self._create_empty_metadata()
        else:
            return self._create_empty_metadata()
    
    def _save_metadata(self) -> None:
        """Save metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            self.logger.log_error(f"Failed to save metadata: {e}")
            raise
    
    def _create_empty_metadata(self) -> Dict[str, Any]:
        """Create empty metadata structure"""
        return {
            'version': '1.0',
            'created_at': Timer.get_current_timestamp(),
            'last_updated': Timer.get_current_timestamp(),
            'save_dir': str(self.save_dir),
            'max_checkpoints': self.max_checkpoints,
            'keep_best': self.keep_best,
            'checkpoints': [],
            'best_checkpoints': [],
            'statistics': {
                'total_saved': 0,
                'total_size_mb': 0,
                'avg_size_mb': 0
            }
        }
    
    def _update_checkpoint_lists(self) -> None:
        """Update checkpoint lists from metadata"""
        self.checkpoints = self.metadata.get('checkpoints', [])
        self.best_checkpoints = [
            c for c in self.checkpoints
            if c.get('is_best', False)
        ]
    
    def _get_latest_checkpoint_info(self) -> Optional[Dict[str, Any]]:
        """Get the latest checkpoint info from metadata"""
        if not self.metadata['checkpoints']:
            return None
        
        # Sort by timestamp (newest first)
        sorted_checkpoints = sorted(
            self.metadata['checkpoints'],
            key=lambda x: x['timestamp'],
            reverse=True
        )
        return sorted_checkpoints[0]
    
    def _find_checkpoint_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find checkpoint info by name"""
        for checkpoint in self.metadata['checkpoints']:
            if checkpoint['filename'].startswith(name):
                return checkpoint
        return None
    
    def _find_checkpoint_by_path(self, path: str) -> Optional[Dict[str, Any]]:
        """Find checkpoint info by path"""
        path = str(Path(path))
        for checkpoint in self.metadata['checkpoints']:
            if checkpoint['path'] == path:
                return checkpoint
        return None
    
    def _find_best_checkpoint(
        self,
        optimize_metric: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Find the best checkpoint based on metrics"""
        if not self.metadata['checkpoints']:
            return None
        
        # Get checkpoints with metrics
        checkpoints_with_metrics = []
        for checkpoint in self.metadata['checkpoints']:
            if checkpoint.get('metrics'):
                checkpoints_with_metrics.append(checkpoint)
        
        if not checkpoints_with_metrics:
            return None
        
        # Determine which metric to optimize
        if optimize_metric:
            # Check if metric exists in any checkpoint
            has_metric = any(
                optimize_metric in c.get('metrics', {})
                for c in checkpoints_with_metrics
            )
            if not has_metric:
                self.logger.log_warning(
                    f"Metric '{optimize_metric}' not found in checkpoints. "
                    "Using first available metric."
                )
                optimize_metric = None
        
        if not optimize_metric:
            # Use the first metric from the first checkpoint
            first_metrics = checkpoints_with_metrics[0].get('metrics', {})
            if first_metrics:
                optimize_metric = list(first_metrics.keys())[0]
            else:
                return None
        
        # Find best based on metric (assuming lower is better for loss, higher for accuracy)
        # Try to determine if metric is a loss or accuracy
        is_loss = any(
            'loss' in optimize_metric.lower() or 'error' in optimize_metric.lower()
        )
        
        best_checkpoint = None
        best_value = float('inf') if is_loss else float('-inf')
        
        for checkpoint in checkpoints_with_metrics:
            metric_value = checkpoint.get('metrics', {}).get(optimize_metric)
            if metric_value is None:
                continue
            
            if is_loss:
                if metric_value < best_value:
                    best_value = metric_value
                    best_checkpoint = checkpoint
            else:
                if metric_value > best_value:
                    best_value = metric_value
                    best_checkpoint = checkpoint
        
        return best_checkpoint
    
    def _is_checkpoint_best(self, checkpoint_path: Path) -> bool:
        """Check if a checkpoint is marked as best"""
        checkpoint_info = self._find_checkpoint_by_path(str(checkpoint_path))
        return checkpoint_info and checkpoint_info.get('is_best', False)
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints based on max_checkpoints policy"""
        checkpoints = self.metadata['checkpoints']
        
        # Separate best and regular checkpoints
        regular_checkpoints = [
            c for c in checkpoints
            if not c.get('is_best', False)
        ]
        
        best_checkpoints = [
            c for c in checkpoints
            if c.get('is_best', False)
        ]
        
        # Sort regular checkpoints by timestamp (oldest first)
        regular_checkpoints.sort(key=lambda x: x['timestamp'])
        
        # Determine how many checkpoints to keep
        if self.keep_best:
            # Keep all best checkpoints, limit regular ones
            max_regular = self.max_checkpoints - len(best_checkpoints)
            if max_regular < 0:
                max_regular = 0
        else:
            # Limit both best and regular checkpoints
            max_regular = self.max_checkpoints - len(best_checkpoints)
            if max_regular < 0:
                # Remove oldest best checkpoints if there are too many
                best_to_remove = len(best_checkpoints) - self.max_checkpoints
                for i in range(best_to_remove):
                    if best_checkpoints:
                        oldest_best = best_checkpoints[0]
                        self._delete_checkpoint_file(oldest_best['path'])
                        checkpoints.remove(oldest_best)
                        best_checkpoints.pop(0)
                max_regular = 0
        
        # Remove oldest regular checkpoints
        while len(regular_checkpoints) > max_regular:
            oldest = regular_checkpoints.pop(0)
            self._delete_checkpoint_file(oldest['path'])
            checkpoints.remove(oldest)
        
        # Update metadata
        self.metadata['checkpoints'] = checkpoints
        self._save_metadata()
    
    def _delete_checkpoint_file(self, path: str) -> bool:
        """Delete a checkpoint file"""
        try:
            path_obj = Path(path)
            if path_obj.exists():
                path_obj.unlink()
                self.logger.log_info(f"Deleted old checkpoint: {path}")
                return True
            return False
        except Exception as e:
            self.logger.log_warning(f"Failed to delete checkpoint {path}: {e}")
            return False
    
    def _compute_checkpoint_hash(self, checkpoint_data: Dict[str, Any]) -> str:
        """Compute hash for checkpoint integrity verification"""
        # Create a copy without the hash field
        data_copy = checkpoint_data.copy()
        if 'hash' in data_copy:
            del data_copy['hash']
        
        # Convert to string and hash
        data_str = str(data_copy)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def _should_ignore_hash_failure(self, checkpoint_data: Dict[str, Any]) -> bool:
        """Determine if hash verification failure should be ignored"""
        # Allow loading if force_load flag is set
        if checkpoint_data.get('metadata', {}).get('force_load', False):
            return True
        
        # Allow loading if it's an older checkpoint without hash
        if 'hash' not in checkpoint_data:
            return True
        
        return False
    
    def __str__(self) -> str:
        """String representation of the checkpoint manager"""
        return (
            f"CheckpointManager(save_dir={self.save_dir}, "
            f"max_checkpoints={self.max_checkpoints}, "
            f"checkpoints={len(self.metadata['checkpoints'])})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        stats = self.get_checkpoint_statistics()
        return (
            f"CheckpointManager(\n"
            f"  save_dir={self.save_dir},\n"
            f"  max_checkpoints={self.max_checkpoints},\n"
            f"  total_checkpoints={stats['total_checkpoints']},\n"
            f"  best_checkpoints={stats['best_checkpoints']},\n"
            f"  total_size_mb={stats['total_size_mb']:.2f}\n"
            f")"
        )


class CheckpointUtils:
    """
    Utility functions for working with checkpoints
    """
    
    @staticmethod
    def merge_checkpoints(
        checkpoint_paths: List[str],
        output_path: str,
        merge_strategy: str = 'averaging'
    ) -> str:
        """
        Merge multiple checkpoints into one
        
        Args:
            checkpoint_paths: List of checkpoint file paths
            output_path: Path to save merged checkpoint
            merge_strategy: Strategy for merging ('averaging', 'voting', 'max')
            
        Returns:
            str: Path to merged checkpoint
            
        Raises:
            RuntimeError: If merging fails
        """
        if not checkpoint_paths:
            raise ValueError("No checkpoints to merge")
        
        print(f"Merging {len(checkpoint_paths)} checkpoints using {merge_strategy} strategy...")
        
        try:
            # Load all checkpoints
            checkpoints = []
            for path in checkpoint_paths:
                checkpoint = torch.load(path, map_location='cpu')
                checkpoints.append(checkpoint)
            
            # Determine merge strategy
            if merge_strategy == 'averaging':
                merged = CheckpointUtils._average_checkpoints(checkpoints)
            elif merge_strategy == 'voting':
                merged = CheckpointUtils._vote_checkpoints(checkpoints)
            elif merge_strategy == 'max':
                merged = CheckpointUtils._max_checkpoints(checkpoints)
            else:
                raise ValueError(f"Unknown merge strategy: {merge_strategy}")
            
            # Save merged checkpoint
            torch.save(merged, output_path)
            
            print(f"Merged checkpoint saved to {output_path}")
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to merge checkpoints: {e}")
    
    @staticmethod
    def _average_checkpoints(checkpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Average model parameters across checkpoints"""
        averaged = checkpoints[0].copy()
        
        # Average model state dict
        model_keys = None
        for checkpoint in checkpoints:
            if 'state' in checkpoint and 'model_state_dict' in checkpoint['state']:
                current_keys = set(checkpoint['state']['model_state_dict'].keys())
                if model_keys is None:
                    model_keys = current_keys
                else:
                    model_keys = model_keys.intersection(current_keys)
        
        if not model_keys:
            return averaged
        
        # Initialize averaged state dict
        averaged_state = {}
        for key in model_keys:
            tensors = []
            for checkpoint in checkpoints:
                if 'state' in checkpoint and 'model_state_dict' in checkpoint['state']:
                    tensor = checkpoint['state']['model_state_dict'][key]
                    tensors.append(tensor)
            if tensors:
                averaged_state[key] = torch.stack(tensors).mean(dim=0)
        
        averaged['state']['model_state_dict'] = averaged_state
        
        return averaged
    
    @staticmethod
    def _vote_checkpoints(checkpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use voting for parameters (not typically used for neural networks)"""
        # For neural networks, voting doesn't make sense for parameters
        # This is a placeholder for completeness
        return checkpoints[0]
    
    @staticmethod
    def _max_checkpoints(checkpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Take maximum parameter values across checkpoints"""
        max_checkpoint = checkpoints[0].copy()
        
        # Get common model keys
        model_keys = None
        for checkpoint in checkpoints:
            if 'state' in checkpoint and 'model_state_dict' in checkpoint['state']:
                current_keys = set(checkpoint['state']['model_state_dict'].keys())
                if model_keys is None:
                    model_keys = current_keys
                else:
                    model_keys = model_keys.intersection(current_keys)
        
        if not model_keys:
            return max_checkpoint
        
        # Take max for each parameter
        max_state = {}
        for key in model_keys:
            tensors = []
            for checkpoint in checkpoints:
                if 'state' in checkpoint and 'model_state_dict' in checkpoint['state']:
                    tensor = checkpoint['state']['model_state_dict'][key]
                    tensors.append(tensor)
            if tensors:
                max_state[key] = torch.stack(tensors).max(dim=0)[0]
        
        max_checkpoint['state']['model_state_dict'] = max_state
        
        return max_checkpoint


# Export common classes and utilities
__all__ = [
    'CheckpointManager',
    'CheckpointUtils'
]