"""
Edge Updater Module for H-GRAGrecsys

This module handles dynamic updates to edges in the heterogeneous graph,
including updating interaction weights, similarity edges, pruning low-weight
edges, and maintaining edge history for tracking changes over time.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, deque
from datetime import datetime
import heapq
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.graph.heterogeneous_graph import HeterogeneousGraph
from models.graph.relation_types import RelationType, EdgeWeightFunctions
from models.agent.user_agent import UserAgent
from models.agent.item_agent import ItemAgent
from utils.logger import Logger
from utils.config_loader import ConfigLoader


class EdgeUpdateRecord:
    """
    Records history of updates to a specific edge.
    
    Attributes:
        edge_key: Tuple of (source, target, relation_type)
        update_history: List of (timestamp, old_weight, new_weight, reason)
        total_updates: Number of updates
        last_update_time: Timestamp of last update
        update_frequency: Average frequency of updates
    """
    
    def __init__(self, source: str, target: str, relation_type: str):
        self.edge_key = (source, target, relation_type)
        self.update_history: List[Tuple[float, float, float, str]] = []
        self.total_updates = 0
        self.last_update_time = 0.0
        self.update_frequency = 0.0
    
    def add_update(self, old_weight: float, new_weight: float, reason: str) -> None:
        """
        Add a new update record.
        
        Args:
            old_weight: Previous edge weight
            new_weight: New edge weight
            reason: Reason for the update
        """
        current_time = datetime.now().timestamp()
        self.update_history.append((current_time, old_weight, new_weight, reason))
        self.total_updates += 1
        self.last_update_time = current_time
        
        # Update frequency (exponential moving average)
        if self.total_updates > 1:
            time_diff = current_time - self.update_history[-2][0]
            if time_diff > 0:
                self.update_frequency = 0.9 * self.update_frequency + 0.1 * (1.0 / time_diff)
    
    def get_update_count(self) -> int:
        """Get total number of updates."""
        return self.total_updates
    
    def get_last_update(self) -> Optional[Tuple[float, float, float, str]]:
        """Get the most recent update record."""
        return self.update_history[-1] if self.update_history else None
    
    def get_update_history(self, limit: Optional[int] = None) -> List[Tuple[float, float, float, str]]:
        """Get update history, optionally limited to last N updates."""
        if limit is not None:
            return self.update_history[-limit:]
        return self.update_history.copy()
    
    def get_average_magnitude(self) -> float:
        """Get average magnitude of weight changes."""
        if len(self.update_history) < 2:
            return 0.0
        
        magnitudes = []
        for _, old_w, new_w, _ in self.update_history[1:]:
            magnitudes.append(abs(new_w - old_w))
        
        return np.mean(magnitudes) if magnitudes else 0.0
    
    def get_weight_trend(self) -> str:
        """Determine the trend of weight changes."""
        if len(self.update_history) < 3:
            return 'stable'
        
        recent_changes = []
        for _, old_w, new_w, _ in self.update_history[-10:]:
            if new_w > old_w:
                recent_changes.append(1)
            elif new_w < old_w:
                recent_changes.append(-1)
            else:
                recent_changes.append(0)
        
        avg_change = np.mean(recent_changes) if recent_changes else 0
        if avg_change > 0.1:
            return 'increasing'
        elif avg_change < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'edge_key': self.edge_key,
            'total_updates': self.total_updates,
            'last_update_time': self.last_update_time,
            'update_frequency': self.update_frequency,
            'update_history': self.update_history,
            'average_magnitude': self.get_average_magnitude(),
            'trend': self.get_weight_trend()
        }


class EdgeUpdater:
    """
    Manages dynamic updates to edges in the heterogeneous graph.
    
    This class handles:
    - Updating interaction edges based on user feedback
    - Recomputing similarity edges when node embeddings change
    - Pruning low-weight or stale edges
    - Maintaining update history for all edges
    - Supporting dynamic graph evolution
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the edge updater with configuration.
        
        Args:
            config: Configuration dictionary containing edge update parameters
        """
        self.config = config
        self.logger = Logger.get_instance(log_dir='logs', name='edge_updater')
        
        # Extract configuration
        graph_config = config.get('model', {}).get('graph', {})
        self.edge_update_rate = graph_config.get('edge_update_rate', 0.1)
        self.pruning_threshold = graph_config.get('pruning_threshold', 0.05)
        self.similarity_update_frequency = graph_config.get('similarity_update_frequency', 10)
        self.max_updates_per_step = graph_config.get('max_updates_per_step', 1000)
        self.decay_factor = graph_config.get('decay_factor', 0.95)
        self.stale_threshold = graph_config.get('stale_threshold', 3600)  # 1 hour in seconds
        
        # Edge update history
        self.update_history: Dict[Tuple[str, str, str], EdgeUpdateRecord] = {}
        
        # Update queues
        self.pending_updates: List[Dict[str, Any]] = []
        self.update_stats = {
            'total_updates': 0,
            'interaction_updates': 0,
            'similarity_updates': 0,
            'pruned_edges': 0,
            'recomputed_edges': 0,
            'failed_updates': 0
        }
        
        self.logger.log_info(f"Initialized EdgeUpdater with threshold={self.pruning_threshold}")
    
    def update_interaction_edge(self, edge: Tuple[str, str, str], 
                               success: bool, 
                               attempts: int,
                               graph: HeterogeneousGraph) -> Tuple[bool, float]:
        """
        Update an interaction edge based on interaction success/failure.
        
        Args:
            edge: Tuple of (source, target, relation_type)
            success: Whether the interaction was successful
            attempts: Number of attempts
            graph: HeterogeneousGraph instance
            
        Returns:
            Tuple[bool, float]: (success_flag, new_weight)
        """
        source, target, relation_type = edge
        
        # Validate edge exists
        current_weight = graph.get_edge_weight(source, target, relation_type)
        if current_weight is None:
            self.logger.log_warning(f"Edge {edge} not found in graph")
            self.update_stats['failed_updates'] += 1
            return False, 0.0
        
        # Calculate new weight based on success and attempts
        if success:
            # Increase weight based on success
            increment = self.edge_update_rate * (1.0 - current_weight)
            new_weight = min(1.0, current_weight + increment)
        else:
            # Decrease weight based on failure and attempts
            penalty = self.edge_update_rate * min(1.0, attempts / 10.0)
            new_weight = max(0.0, current_weight - penalty)
        
        # Apply decay for multiple attempts
        if attempts > 1:
            decay = self.decay_factor ** (attempts - 1)
            new_weight = new_weight * decay
        
        # Update the edge
        graph.update_edge_weight(source, target, relation_type, new_weight)
        
        # Record the update
        record = self._get_or_create_record(source, target, relation_type)
        reason = f"success_{success}_attempts_{attempts}"
        record.add_update(current_weight, new_weight, reason)
        
        # Update statistics
        self.update_stats['total_updates'] += 1
        self.update_stats['interaction_updates'] += 1
        
        self.logger.log_info(
            f"Updated interaction edge {source}->{target}: {current_weight:.4f} -> {new_weight:.4f}"
        )
        
        return True, new_weight
    
    def update_similarity_edge(self, edge: Tuple[str, str, str], 
                              new_embedding: torch.Tensor,
                              graph: HeterogeneousGraph) -> Tuple[bool, float]:
        """
        Update a similarity edge based on new node embedding.
        
        Args:
            edge: Tuple of (source, target, relation_type)
            new_embedding: New embedding for one of the nodes
            graph: HeterogeneousGraph instance
            
        Returns:
            Tuple[bool, float]: (success_flag, new_weight)
        """
        source, target, relation_type = edge
        
        # Validate edge exists
        current_weight = graph.get_edge_weight(source, target, relation_type)
        if current_weight is None:
            self.logger.log_warning(f"Edge {edge} not found in graph")
            self.update_stats['failed_updates'] += 1
            return False, 0.0
        
        # Get embeddings for both nodes
        source_node = graph.nodes.get(source)
        target_node = graph.nodes.get(target)
        
        if source_node is None or target_node is None:
            self.logger.log_warning(f"Nodes for edge {edge} not found")
            return False, 0.0
        
        # Recompute similarity
        if relation_type == RelationType.SIMILAR_PREF.value:
            # User similarity based on embeddings
            if source_node.embedding is not None and target_node.embedding is not None:
                new_weight = EdgeWeightFunctions.cosine_similarity(
                    source_node.embedding, target_node.embedding
                )
            else:
                return False, 0.0
        
        elif relation_type == RelationType.CONTENT_SIM.value:
            # Item content similarity
            if source_node.embedding is not None and target_node.embedding is not None:
                new_weight = EdgeWeightFunctions.cosine_similarity(
                    source_node.embedding, target_node.embedding
                )
            else:
                return False, 0.0
        
        elif relation_type == RelationType.CO_INTER.value:
            # Co-interaction similarity
            source_items = set(source_node.features.get('interacted_items', []))
            target_items = set(target_node.features.get('interacted_items', []))
            new_weight = EdgeWeightFunctions.jaccard_similarity(source_items, target_items)
        
        else:
            self.logger.log_warning(f"Unknown relation type for similarity update: {relation_type}")
            return False, 0.0
        
        # Update edge weight
        graph.update_edge_weight(source, target, relation_type, new_weight)
        
        # Record the update
        record = self._get_or_create_record(source, target, relation_type)
        reason = f"embedding_update"
        record.add_update(current_weight, new_weight, reason)
        
        # Update statistics
        self.update_stats['total_updates'] += 1
        self.update_stats['similarity_updates'] += 1
        
        self.logger.log_info(
            f"Updated similarity edge {source}->{target}: {current_weight:.4f} -> {new_weight:.4f}"
        )
        
        return True, new_weight
    
    def prune_edges(self, graph: HeterogeneousGraph, 
                   threshold: Optional[float] = None,
                   dry_run: bool = False) -> Dict[str, Any]:
        """
        Prune edges with weight below threshold.
        
        Args:
            graph: HeterogeneousGraph instance
            threshold: Pruning threshold (uses config if None)
            dry_run: If True, only simulate pruning without removing
            
        Returns:
            Dict[str, Any]: Pruning results with statistics
        """
        threshold = threshold or self.pruning_threshold
        pruned_edges = []
        edge_stats = defaultdict(int)
        
        self.logger.log_info(f"Pruning edges with weight < {threshold}")
        
        # Collect edges to prune
        edges_to_remove = []
        for (source, target, rel_type), weight in graph.edges.items():
            if weight < threshold:
                edges_to_remove.append((source, target, rel_type))
                pruned_edges.append({
                    'source': source,
                    'target': target,
                    'relation_type': rel_type,
                    'weight': weight
                })
                edge_stats[rel_type] += 1
        
        if dry_run:
            self.logger.log_info(f"[DRY RUN] Would prune {len(edges_to_remove)} edges")
            return {
                'would_prune': len(edges_to_remove),
                'edges': pruned_edges[:10],  # First 10
                'by_type': dict(edge_stats)
            }
        
        # Remove edges
        for source, target, rel_type in edges_to_remove:
            graph.remove_edge(source, target, rel_type)
            self.update_stats['pruned_edges'] += 1
        
        self.logger.log_info(f"Pruned {len(edges_to_remove)} edges")
        
        return {
            'pruned': len(edges_to_remove),
            'edges': pruned_edges[:10],  # First 10
            'by_type': dict(edge_stats)
        }
    
    def recompute_similarity_edges(self, graph: HeterogeneousGraph,
                                  changed_nodes: List[str],
                                  relation_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Recompute similarity edges for changed nodes.
        
        Args:
            graph: HeterogeneousGraph instance
            changed_nodes: List of node IDs that have changed
            relation_types: List of relation types to recompute
            
        Returns:
            Dict[str, Any]: Recompute results with statistics
        """
        relation_types = relation_types or [
            RelationType.SIMILAR_PREF.value,
            RelationType.CONTENT_SIM.value
        ]
        
        recomputed_edges = []
        updated_edges = []
        
        self.logger.log_info(f"Recomputing similarity edges for {len(changed_nodes)} nodes")
        
        for node_id in tqdm(changed_nodes, desc="Recomputing edges"):
            if node_id not in graph.nodes:
                continue
            
            node = graph.nodes[node_id]
            node_type = node.node_type
            
            # Determine which edges to recompute
            if node_type == 'user':
                target_types = ['user']
                relation_type = RelationType.SIMILAR_PREF.value
            elif node_type == 'item':
                target_types = ['item']
                relation_type = RelationType.CONTENT_SIM.value
            else:
                continue
            
            # Recompute edges to same-type nodes
            for target_id, target_node in graph.nodes.items():
                if target_id == node_id or target_node.node_type != node_type:
                    continue
                
                # Check if edge exists
                edge = (node_id, target_id, relation_type)
                current_weight = graph.get_edge_weight(node_id, target_id, relation_type)
                
                # Recompute similarity
                if relation_type == RelationType.SIMILAR_PREF.value:
                    if node.embedding is not None and target_node.embedding is not None:
                        new_weight = EdgeWeightFunctions.cosine_similarity(
                            node.embedding, target_node.embedding
                        )
                    else:
                        continue
                elif relation_type == RelationType.CONTENT_SIM.value:
                    if node.embedding is not None and target_node.embedding is not None:
                        new_weight = EdgeWeightFunctions.cosine_similarity(
                            node.embedding, target_node.embedding
                        )
                    else:
                        continue
                else:
                    continue
                
                recomputed_edges.append({
                    'source': node_id,
                    'target': target_id,
                    'old_weight': current_weight,
                    'new_weight': new_weight
                })
                
                if current_weight is not None:
                    # Update existing edge
                    graph.update_edge_weight(node_id, target_id, relation_type, new_weight)
                    updated_edges.append((node_id, target_id, relation_type, new_weight))
                    
                    # Record update
                    record = self._get_or_create_record(node_id, target_id, relation_type)
                    record.add_update(current_weight, new_weight, 'recompute')
                    
                    self.update_stats['recomputed_edges'] += 1
                elif new_weight > self.pruning_threshold:
                    # Add new edge if weight is significant
                    graph.add_edge(node_id, target_id, relation_type, new_weight)
                    updated_edges.append((node_id, target_id, relation_type, new_weight))
                    
                    # Record update
                    record = self._get_or_create_record(node_id, target_id, relation_type)
                    record.add_update(0.0, new_weight, 'new_edge')
                    
                    self.update_stats['recomputed_edges'] += 1
        
        self.logger.log_info(f"Recomputed {len(recomputed_edges)} similarity edges")
        
        return {
            'total_recomputed': len(recomputed_edges),
            'updated_edges': len(updated_edges),
            'recomputed_edges': recomputed_edges[:10]  # First 10
        }
    
    def apply_dynamic_updates(self, graph: HeterogeneousGraph,
                             interactions: List[Dict[str, Any]],
                             update_type: str = 'all') -> Dict[str, Any]:
        """
        Apply a batch of dynamic updates from interactions.
        
        Args:
            graph: HeterogeneousGraph instance
            interactions: List of interaction dictionaries
            update_type: Type of updates to apply ('all', 'interaction', 'similarity')
            
        Returns:
            Dict[str, Any]: Update results with statistics
        """
        update_results = {
            'interaction_updates': 0,
            'similarity_updates': 0,
            'pruned_edges': 0,
            'updated_nodes': set(),
            'errors': 0
        }
        
        self.logger.log_info(f"Applying dynamic updates for {len(interactions)} interactions")
        
        # Process interactions
        for interaction in tqdm(interactions, desc="Applying updates"):
            try:
                user_id = interaction.get('user_id')
                item_id = interaction.get('item_id')
                success = interaction.get('success', True)
                attempts = interaction.get('attempts', 1)
                
                if not user_id or not item_id:
                    continue
                
                # Update interaction edge
                if update_type in ['all', 'interaction']:
                    edge = (user_id, item_id, RelationType.INTERACT.value)
                    updated, weight = self.update_interaction_edge(edge, success, attempts, graph)
                    if updated:
                        update_results['interaction_updates'] += 1
                        update_results['updated_nodes'].add(user_id)
                        update_results['updated_nodes'].add(item_id)
                
                # Check if we need to update similarity edges
                if update_type in ['all', 'similarity']:
                    # Update user-user similarities if user changed significantly
                    if success and attempts == 1:
                        # Recompute similarity edges for this user
                        if user_id in graph.nodes:
                            recompute_result = self.recompute_similarity_edges(
                                graph, [user_id], 
                                [RelationType.SIMILAR_PREF.value]
                            )
                            update_results['similarity_updates'] += recompute_result['updated_edges']
                            
                            # Update item-item similarities if item changed
                            if item_id in graph.nodes:
                                recompute_result = self.recompute_similarity_edges(
                                    graph, [item_id],
                                    [RelationType.CONTENT_SIM.value]
                                )
                                update_results['similarity_updates'] += recompute_result['updated_edges']
            
            except Exception as e:
                self.logger.log_error(f"Error applying update: {str(e)}")
                update_results['errors'] += 1
        
        # Apply edge pruning after updates
        if update_type in ['all', 'interaction']:
            prune_result = self.prune_edges(graph)
            update_results['pruned_edges'] = prune_result.get('pruned', 0)
        
        self.logger.log_info(
            f"Applied dynamic updates: {update_results['interaction_updates']} interaction, "
            f"{update_results['similarity_updates']} similarity, "
            f"{update_results['pruned_edges']} pruned"
        )
        
        return update_results
    
    def get_update_history(self, edge: Tuple[str, str, str]) -> Optional[Dict[str, Any]]:
        """
        Get update history for a specific edge.
        
        Args:
            edge: Tuple of (source, target, relation_type)
            
        Returns:
            Optional[Dict[str, Any]]: Update history or None
        """
        record = self.update_history.get(edge)
        if record:
            return record.to_dict()
        return None
    
    def get_edge_statistics(self, graph: HeterogeneousGraph) -> Dict[str, Any]:
        """
        Get comprehensive statistics about edges in the graph.
        
        Args:
            graph: HeterogeneousGraph instance
            
        Returns:
            Dict[str, Any]: Edge statistics
        """
        stats = {
            'total_edges': len(graph.edges),
            'edges_by_type': defaultdict(int),
            'weight_distribution': {
                'mean': 0.0,
                'std': 0.0,
                'min': 1.0,
                'max': 0.0,
                'quartiles': []
            },
            'prunable_edges': 0,
            'high_confidence_edges': 0,
            'active_edges': 0,
            'stale_edges': 0,
            'update_frequencies': []
        }
        
        # Collect weights
        weights = []
        for (source, target, rel_type), weight in graph.edges.items():
            stats['edges_by_type'][rel_type] += 1
            weights.append(weight)
            
            if weight < self.pruning_threshold:
                stats['prunable_edges'] += 1
            if weight > 0.7:
                stats['high_confidence_edges'] += 1
            
            # Check edge freshness
            record = self.update_history.get((source, target, rel_type))
            if record:
                current_time = datetime.now().timestamp()
                time_since_update = current_time - record.last_update_time
                if time_since_update < self.stale_threshold:
                    stats['active_edges'] += 1
                else:
                    stats['stale_edges'] += 1
                
                stats['update_frequencies'].append(record.update_frequency)
            else:
                # Edge exists but no update history (consider it active)
                stats['active_edges'] += 1
        
        # Compute weight distribution statistics
        if weights:
            stats['weight_distribution']['mean'] = np.mean(weights)
            stats['weight_distribution']['std'] = np.std(weights)
            stats['weight_distribution']['min'] = np.min(weights)
            stats['weight_distribution']['max'] = np.max(weights)
            stats['weight_distribution']['quartiles'] = np.percentile(weights, [25, 50, 75]).tolist()
        
        return stats
    
    def _get_or_create_record(self, source: str, target: str, relation_type: str) -> EdgeUpdateRecord:
        """
        Get existing EdgeUpdateRecord or create a new one.
        
        Args:
            source: Source node ID
            target: Target node ID
            relation_type: Type of relation
            
        Returns:
            EdgeUpdateRecord: Update record for the edge
        """
        edge_key = (source, target, relation_type)
        if edge_key not in self.update_history:
            self.update_history[edge_key] = EdgeUpdateRecord(source, target, relation_type)
        return self.update_history[edge_key]
    
    def cleanup_update_history(self, max_age_days: int = 30) -> int:
        """
        Clean up old update history records.
        
        Args:
            max_age_days: Maximum age of history records in days
            
        Returns:
            int: Number of records removed
        """
        current_time = datetime.now().timestamp()
        max_age_seconds = max_age_days * 24 * 3600
        removed = 0
        
        for edge_key, record in list(self.update_history.items()):
            if record.last_update_time < (current_time - max_age_seconds):
                # Remove old records
                del self.update_history[edge_key]
                removed += 1
        
        self.logger.log_info(f"Cleaned up {removed} old update history records")
        return removed
    
    def reset_statistics(self) -> None:
        """Reset update statistics."""
        self.update_stats = {
            'total_updates': 0,
            'interaction_updates': 0,
            'similarity_updates': 0,
            'pruned_edges': 0,
            'recomputed_edges': 0,
            'failed_updates': 0
        }
        self.logger.log_info("Reset update statistics")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current update statistics.
        
        Returns:
            Dict[str, Any]: Update statistics
        """
        return self.update_stats.copy()
    
    def export_update_history(self, filepath: str) -> None:
        """
        Export update history to a file.
        
        Args:
            filepath: Path to export file
        """
        import json
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'update_stats': self.update_stats,
            'history': {str(k): v.to_dict() for k, v in self.update_history.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.log_info(f"Exported update history to {filepath}")
    
    def import_update_history(self, filepath: str) -> None:
        """
        Import update history from a file.
        
        Args:
            filepath: Path to import file
        """
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for edge_key_str, record_data in data.get('history', {}).items():
            # Parse edge key from string
            edge_key = eval(edge_key_str)  # Safe since we generated it
            record = EdgeUpdateRecord(edge_key[0], edge_key[1], edge_key[2])
            record.total_updates = record_data['total_updates']
            record.last_update_time = record_data['last_update_time']
            record.update_frequency = record_data['update_frequency']
            record.update_history = record_data['update_history']
            self.update_history[edge_key] = record
        
        self.update_stats = data.get('update_stats', self.update_stats)
        self.logger.log_info(f"Imported update history from {filepath}")


# Example usage
if __name__ == "__main__":
    # Load configuration
    config_path = "config/default_config.yaml"
    config_loader = ConfigLoader(config_path)
    config = config_loader.load_config()
    
    # Create edge updater
    updater = EdgeUpdater(config)
    
    # Example: Create a simple graph
    from models.graph.heterogeneous_graph import HeterogeneousGraph
    graph = HeterogeneousGraph(config)
    
    # Add some nodes
    graph.add_node('user_1', 'user', features={'interacted_items': ['item_1', 'item_2']})
    graph.add_node('user_2', 'user', features={'interacted_items': ['item_2', 'item_3']})
    graph.add_node('item_1', 'item')
    graph.add_node('item_2', 'item')
    graph.add_node('item_3', 'item')
    
    # Add some edges
    graph.add_edge('user_1', 'item_1', RelationType.INTERACT, weight=0.8)
    graph.add_edge('user_1', 'item_2', RelationType.INTERACT, weight=0.6)
    graph.add_edge('user_2', 'item_2', RelationType.INTERACT, weight=0.7)
    graph.add_edge('user_2', 'item_3', RelationType.INTERACT, weight=0.5)
    graph.add_edge('user_1', 'user_2', RelationType.SIMILAR_PREF, weight=0.9)
    graph.add_edge('item_1', 'item_2', RelationType.CONTENT_SIM, weight=0.4)
    
    # Update interaction edge
    edge = ('user_1', 'item_1', RelationType.INTERACT.value)
    success, new_weight = updater.update_interaction_edge(edge, success=True, attempts=2, graph=graph)
    print(f"Updated interaction edge: {success}, new weight: {new_weight}")
    
    # Prune edges
    prune_result = updater.prune_edges(graph, threshold=0.3)
    print(f"Pruned {prune_result['pruned']} edges")
    
    # Get statistics
    stats = updater.get_edge_statistics(graph)
    print(f"Edge statistics: {stats['total_edges']} total edges")
    
    # Apply dynamic updates
    interactions = [
        {'user_id': 'user_1', 'item_id': 'item_2', 'success': True, 'attempts': 1},
        {'user_id': 'user_2', 'item_id': 'item_3', 'success': False, 'attempts': 3}
    ]
    update_result = updater.apply_dynamic_updates(graph, interactions)
    print(f"Update results: {update_result}")