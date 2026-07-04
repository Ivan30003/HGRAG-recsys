"""
Dynamic Graph Module for H-GRAGrecsys

This module extends the heterogeneous graph with dynamic update capabilities,
supporting temporal evolution, edge weight updates, node additions/removals,
and collaborative propagation paths for recommendation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from collections import defaultdict, deque
from datetime import datetime
import heapq
from tqdm import tqdm
import sys
import os
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.graph.heterogeneous_graph import HeterogeneousGraph, GraphNode
from models.graph.relation_types import RelationType, RelationTypeRegistry, EdgeWeightFunctions
from models.graph.edge_updater import EdgeUpdater, EdgeUpdateRecord
from models.agent.user_agent import UserAgent
from models.agent.item_agent import ItemAgent
from models.agent.memory import AgentMemory
from utils.logger import Logger
from utils.config_loader import ConfigLoader


class DynamicGraph(HeterogeneousGraph):
    """
    Dynamic heterogeneous graph with temporal evolution capabilities.
    
    This class extends HeterogeneousGraph with:
    - Time-aware edge updates with decay
    - Automatic edge pruning based on staleness
    - Collaborative propagation paths for recommendation
    - PPR (Personalized PageRank) sampling
    - Historical snapshots and rollback capabilities
    - Temporal statistics tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dynamic graph with configuration.
        
        Args:
            config: Configuration dictionary containing graph parameters
        """
        super().__init__(config)
        
        # Initialize edge updater
        self.edge_updater = EdgeUpdater(config)
        
        # Dynamic graph configuration
        graph_config = config.get('model', {}).get('graph', {})
        self.edge_update_rate = graph_config.get('edge_update_rate', 0.1)
        self.pruning_threshold = graph_config.get('pruning_threshold', 0.05)
        self.decay_factor = graph_config.get('decay_factor', 0.95)
        self.max_edges_per_node = graph_config.get('max_edges_per_node', 100)
        self.stale_threshold = graph_config.get('stale_threshold', 3600 * 24 * 7)  # 7 days
        
        # PPR configuration
        self.ppr_restart_prob = graph_config.get('ppr_restart_prob', 0.15)
        self.ppr_max_iterations = graph_config.get('ppr_max_iterations', 100)
        self.ppr_tolerance = graph_config.get('ppr_tolerance', 1e-6)
        
        # Temporal tracking
        self.temporal_stats = {
            'total_evolutions': 0,
            'nodes_added': 0,
            'nodes_removed': 0,
            'edges_added': 0,
            'edges_removed': 0,
            'edges_updated': 0,
            'pruned_edges': 0,
            'snapshots_taken': 0,
            'evolution_timeline': []
        }
        
        # Historical snapshots for rollback
        self.snapshots: Dict[str, Dict[str, Any]] = {}
        self.current_snapshot_id = 0
        self.max_snapshots = graph_config.get('max_snapshots', 10)
        
        # Collaboration propagation cache
        self.propagation_cache: Dict[Tuple[str, int], List[Tuple[str, float]]] = {}
        self.cache_size = graph_config.get('propagation_cache_size', 1000)
        
        # Register relation types
        self.relation_registry = RelationTypeRegistry(config)
        
        self.logger.log_info(f"Initialized DynamicGraph with PPR restart_prob={self.ppr_restart_prob}")
    
    def evolve_step(self, interactions: List[Dict[str, Any]], 
                   reflections: Optional[List[Dict[str, Any]]] = None,
                   update_type: str = 'all') -> Dict[str, Any]:
        """
        Evolve the graph one step based on new interactions and reflections.
        
        Args:
            interactions: List of interaction dictionaries
            reflections: List of reflection dictionaries (optional)
            update_type: Type of updates to apply ('all', 'interaction', 'similarity')
            
        Returns:
            Dict[str, Any]: Evolution statistics
        """
        self.logger.log_info(f"Evolving graph with {len(interactions)} interactions")
        
        evolution_stats = {
            'timestamp': datetime.now().isoformat(),
            'interactions_processed': len(interactions),
            'reflections_processed': len(reflections) if reflections else 0,
            'edges_added': 0,
            'edges_updated': 0,
            'edges_removed': 0,
            'nodes_added': 0,
            'pruned_edges': 0,
            'update_details': []
        }
        
        # Process interactions
        if interactions:
            # Apply dynamic updates
            update_results = self.edge_updater.apply_dynamic_updates(
                self, interactions, update_type
            )
            
            evolution_stats['edges_updated'] = update_results.get('interaction_updates', 0)
            evolution_stats['edges_added'] = update_results.get('similarity_updates', 0)
            evolution_stats['pruned_edges'] = update_results.get('pruned_edges', 0)
            
            # Update node timestamps
            updated_nodes = update_results.get('updated_nodes', set())
            for node_id in updated_nodes:
                if node_id in self.nodes:
                    self.nodes[node_id].updated_at = datetime.now().timestamp()
        
        # Process reflections
        if reflections:
            reflection_stats = self._process_reflections(reflections)
            evolution_stats.update(reflection_stats)
        
        # Apply edge decay and pruning
        self._apply_temporal_decay()
        
        # Update temporal statistics
        self.temporal_stats['total_evolutions'] += 1
        self.temporal_stats['edges_added'] += evolution_stats['edges_added']
        self.temporal_stats['edges_removed'] += evolution_stats['edges_removed']
        self.temporal_stats['edges_updated'] += evolution_stats['edges_updated']
        self.temporal_stats['pruned_edges'] += evolution_stats['pruned_edges']
        self.temporal_stats['evolution_timeline'].append(evolution_stats)
        
        # Clear propagation cache after evolution
        self.propagation_cache.clear()
        
        self.logger.log_info(f"Graph evolution complete: {evolution_stats}")
        return evolution_stats
    
    def _process_reflections(self, reflections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process reflection data to update graph.
        
        Args:
            reflections: List of reflection dictionaries
            
        Returns:
            Dict[str, Any]: Reflection processing statistics
        """
        stats = {
            'reflections_processed': len(reflections),
            'edges_added_reflection': 0,
            'edges_updated_reflection': 0
        }
        
        for reflection in reflections:
            user_id = reflection.get('user_id')
            item_id = reflection.get('item_id')
            outcome = reflection.get('outcome', 0.5)
            explanation = reflection.get('explanation', '')
            
            if not user_id or not item_id:
                continue
            
            # Update interaction edge based on reflection
            edge = (user_id, item_id, RelationType.INTERACT.value)
            current_weight = self.get_edge_weight(user_id, item_id, RelationType.INTERACT)
            
            if current_weight is not None:
                # Adjust weight based on reflection outcome
                adjustment = (outcome - 0.5) * 0.2  # Scale adjustment
                new_weight = max(0.0, min(1.0, current_weight + adjustment))
                
                self.update_edge_weight(user_id, item_id, RelationType.INTERACT, new_weight)
                stats['edges_updated_reflection'] += 1
                
                # Record update
                record = self.edge_updater._get_or_create_record(
                    user_id, item_id, RelationType.INTERACT.value
                )
                record.add_update(current_weight, new_weight, f'reflection: {explanation[:50]}')
        
        return stats
    
    def _apply_temporal_decay(self) -> None:
        """
        Apply temporal decay to all edges and prune stale ones.
        """
        current_time = datetime.now().timestamp()
        edges_to_prune = []
        
        for (source, target, rel_type), weight in list(self.edges.items()):
            # Check edge staleness
            record = self.edge_updater.update_history.get((source, target, rel_type))
            if record:
                time_since_update = current_time - record.last_update_time
                
                # Apply decay
                if time_since_update > self.stale_threshold:
                    decayed_weight = weight * (self.decay_factor ** (time_since_update / self.stale_threshold))
                    if decayed_weight < self.pruning_threshold:
                        edges_to_prune.append((source, target, rel_type))
                    else:
                        self.update_edge_weight(source, target, rel_type, decayed_weight)
                        self.temporal_stats['edges_updated'] += 1
        
        # Prune edges
        for source, target, rel_type in edges_to_prune:
            self.remove_edge(source, target, rel_type)
            self.temporal_stats['pruned_edges'] += 1
        
        if edges_to_prune:
            self.logger.log_info(f"Pruned {len(edges_to_prune)} stale edges")
    
    def update_edge_weights(self, interaction_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update edge weights based on interaction results.
        
        Args:
            interaction_results: List of dictionaries with interaction feedback
            
        Returns:
            Dict[str, Any]: Update statistics
        """
        stats = {
            'updated_edges': 0,
            'failed_updates': 0,
            'weight_changes': []
        }
        
        for result in interaction_results:
            user_id = result.get('user_id')
            item_id = result.get('item_id')
            success = result.get('success', True)
            attempts = result.get('attempts', 1)
            
            if not user_id or not item_id:
                continue
            
            edge = (user_id, item_id, RelationType.INTERACT.value)
            updated, new_weight = self.edge_updater.update_interaction_edge(
                edge, success, attempts, self
            )
            
            if updated:
                stats['updated_edges'] += 1
                stats['weight_changes'].append({
                    'user': user_id,
                    'item': item_id,
                    'new_weight': new_weight,
                    'success': success
                })
            else:
                stats['failed_updates'] += 1
        
        return stats
    
    def prune_edges(self, threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Prune edges with weight below threshold.
        
        Args:
            threshold: Pruning threshold (uses config if None)
            
        Returns:
            Dict[str, Any]: Pruning statistics
        """
        return self.edge_updater.prune_edges(self, threshold)
    
    def recompute_similarity_edges(self, changed_nodes: List[str]) -> Dict[str, Any]:
        """
        Recompute similarity edges for changed nodes.
        
        Args:
            changed_nodes: List of node IDs that have changed
            
        Returns:
            Dict[str, Any]: Recompute statistics
        """
        return self.edge_updater.recompute_similarity_edges(self, changed_nodes)
    
    def add_new_agents(self, agents: List[Union[UserAgent, ItemAgent]],
                      initial_interactions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Add new agents to the graph.
        
        Args:
            agents: List of new UserAgent or ItemAgent objects
            initial_interactions: Optional list of initial interactions
            
        Returns:
            Dict[str, Any]: Addition statistics
        """
        stats = {
            'users_added': 0,
            'items_added': 0,
            'edges_added': 0,
            'failed': 0
        }
        
        for agent in agents:
            try:
                if isinstance(agent, UserAgent):
                    # Add user node
                    self.add_node(
                        node_id=agent.agent_id,
                        node_type='user',
                        features={
                            'preference_vector': agent.get_preference_memory(),
                            'interaction_count': len(agent.get_recommendation_history()),
                            'embedding': agent.get_embedding()
                        },
                        embedding=agent.get_embedding()
                    )
                    stats['users_added'] += 1
                    
                    # Add initial interactions
                    if initial_interactions:
                        for interaction in initial_interactions:
                            if interaction.get('user_id') == agent.agent_id:
                                item_id = interaction.get('item_id')
                                if item_id in self.nodes:
                                    self.add_edge(
                                        source=agent.agent_id,
                                        target=item_id,
                                        relation_type=RelationType.INTERACT,
                                        weight=interaction.get('rating', 1.0),
                                        metadata={'initial': True}
                                    )
                                    stats['edges_added'] += 1
                    
                    # Connect to similar users
                    self._connect_new_user(agent)
                
                elif isinstance(agent, ItemAgent):
                    # Add item node
                    self.add_node(
                        node_id=agent.agent_id,
                        node_type='item',
                        features={
                            'metadata': agent.get_item_metadata(),
                            'popularity_score': agent.get_popularity_score(),
                            'embedding': agent.get_content_embedding()
                        },
                        embedding=agent.get_content_embedding()
                    )
                    stats['items_added'] += 1
                    
                    # Connect to similar items
                    self._connect_new_item(agent)
            
            except Exception as e:
                self.logger.log_error(f"Failed to add agent {agent.agent_id}: {str(e)}")
                stats['failed'] += 1
        
        self.temporal_stats['nodes_added'] += stats['users_added'] + stats['items_added']
        self.temporal_stats['edges_added'] += stats['edges_added']
        
        return stats
    
    def _connect_new_user(self, user_agent: UserAgent) -> None:
        """
        Connect a new user to similar existing users.
        
        Args:
            user_agent: New UserAgent object
        """
        user_embedding = user_agent.get_embedding()
        if user_embedding is None:
            return
        
        # Find similar users
        similar_users = []
        for node_id, node in self.nodes.items():
            if node.node_type == 'user' and node_id != user_agent.agent_id:
                if node.embedding is not None:
                    similarity = EdgeWeightFunctions.cosine_similarity(
                        user_embedding, node.embedding
                    )
                    if similarity > 0.5:
                        similar_users.append((node_id, similarity))
        
        # Sort by similarity and add top edges
        similar_users.sort(key=lambda x: x[1], reverse=True)
        for target_id, similarity in similar_users[:self.max_edges_per_node]:
            self.add_edge(
                source=user_agent.agent_id,
                target=target_id,
                relation_type=RelationType.SIMILAR_PREF,
                weight=similarity,
                metadata={'new_user': True}
            )
    
    def _connect_new_item(self, item_agent: ItemAgent) -> None:
        """
        Connect a new item to similar existing items.
        
        Args:
            item_agent: New ItemAgent object
        """
        item_embedding = item_agent.get_content_embedding()
        if item_embedding is None:
            return
        
        # Find similar items
        similar_items = []
        for node_id, node in self.nodes.items():
            if node.node_type == 'item' and node_id != item_agent.agent_id:
                if node.embedding is not None:
                    similarity = EdgeWeightFunctions.cosine_similarity(
                        item_embedding, node.embedding
                    )
                    if similarity > 0.5:
                        similar_items.append((node_id, similarity))
        
        # Sort by similarity and add top edges
        similar_items.sort(key=lambda x: x[1], reverse=True)
        for target_id, similarity in similar_items[:self.max_edges_per_node]:
            self.add_edge(
                source=item_agent.agent_id,
                target=target_id,
                relation_type=RelationType.CONTENT_SIM,
                weight=similarity,
                metadata={'new_item': True}
            )
    
    def get_collaborative_propagation_paths(self, node_id: str, 
                                           k: int = 3,
                                           max_paths: int = 10) -> List[List[Tuple[str, str, float]]]:
        """
        Get collaborative propagation paths starting from a node.
        
        Args:
            node_id: Starting node ID
            k: Number of hops to propagate
            max_paths: Maximum number of paths to return
            
        Returns:
            List[List[Tuple[str, str, float]]]: List of paths, each path is list of (node, relation, weight)
        """
        if node_id not in self.nodes:
            self.logger.log_warning(f"Node {node_id} not found")
            return []
        
        paths = []
        visited = set([node_id])
        
        # BFS to find propagation paths
        queue = deque([(node_id, [])])
        
        while queue and len(paths) < max_paths:
            current_node, current_path = queue.popleft()
            
            if len(current_path) >= k:
                paths.append(current_path)
                continue
            
            # Get neighbors with high weights
            neighbors = []
            for rel_type in self.adjacency_lists:
                for neighbor, weight in self.get_neighbors(current_node, rel_type):
                    if neighbor not in visited and weight > 0.3:
                        neighbors.append((neighbor, rel_type, weight))
            
            # Sort by weight
            neighbors.sort(key=lambda x: x[2], reverse=True)
            
            # Explore top neighbors
            for neighbor, rel_type, weight in neighbors[:3]:  # Branching factor 3
                new_path = current_path + [(current_node, rel_type, weight)]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))
        
        return paths
    
    def compute_ppr_scores(self, start_node: str, 
                          restart_prob: Optional[float] = None,
                          max_iterations: Optional[int] = None,
                          tolerance: Optional[float] = None) -> Dict[str, float]:
        """
        Compute Personalized PageRank scores from a start node.
        
        Args:
            start_node: Starting node ID
            restart_prob: Probability of restarting (default from config)
            max_iterations: Maximum iterations (default from config)
            tolerance: Convergence tolerance (default from config)
            
        Returns:
            Dict[str, float]: PPR scores for all nodes
        """
        if start_node not in self.nodes:
            self.logger.log_warning(f"Node {start_node} not found")
            return {}
        
        restart_prob = restart_prob or self.ppr_restart_prob
        max_iterations = max_iterations or self.ppr_max_iterations
        tolerance = tolerance or self.ppr_tolerance
        
        # Check cache
        cache_key = (start_node, int(restart_prob * 100))
        if cache_key in self.propagation_cache:
            cached_scores = dict(self.propagation_cache[cache_key])
            self.logger.log_info(f"Using cached PPR scores for node {start_node}")
            return cached_scores
        
        # Initialize scores
        num_nodes = len(self.nodes)
        node_ids = list(self.nodes.keys())
        node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        start_idx = node_to_idx[start_node]
        
        # Build adjacency matrix (simplified)
        scores = np.zeros(num_nodes)
        scores[start_idx] = 1.0
        
        # Build transition matrix
        transition = np.zeros((num_nodes, num_nodes))
        for (src, dst, rel_type), weight in self.edges.items():
            if src in node_to_idx and dst in node_to_idx:
                src_idx = node_to_idx[src]
                dst_idx = node_to_idx[dst]
                # Weighted transition
                transition[dst_idx, src_idx] += weight
        
        # Normalize columns
        for col in range(num_nodes):
            col_sum = transition[:, col].sum()
            if col_sum > 0:
                transition[:, col] /= col_sum
        
        # Power iteration
        for iteration in range(max_iterations):
            new_scores = (1 - restart_prob) * transition @ scores
            new_scores[start_idx] += restart_prob
            
            # Check convergence
            diff = np.linalg.norm(new_scores - scores, 1)
            scores = new_scores
            
            if diff < tolerance:
                self.logger.log_info(f"PPR converged after {iteration+1} iterations")
                break
        
        # Convert to dictionary
        ppr_scores = {node_ids[i]: scores[i] for i in range(num_nodes)}
        
        # Cache results
        if len(self.propagation_cache) >= self.cache_size:
            # Remove oldest entry
            self.propagation_cache.pop(next(iter(self.propagation_cache)))
        self.propagation_cache[cache_key] = list(ppr_scores.items())
        
        return ppr_scores
    
    def get_ppr_neighbors(self, start_node: str, 
                         k: int = 10,
                         restart_prob: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Get top k neighbors by PPR score.
        
        Args:
            start_node: Starting node ID
            k: Number of neighbors to return
            restart_prob: Probability of restarting
            
        Returns:
            List[Tuple[str, float]]: Top k (neighbor, score) pairs
        """
        ppr_scores = self.compute_ppr_scores(start_node, restart_prob)
        
        # Sort by score and return top k (excluding start node)
        sorted_nodes = sorted(
            [(nid, score) for nid, score in ppr_scores.items() if nid != start_node],
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_nodes[:k]
    
    def take_snapshot(self, snapshot_name: Optional[str] = None) -> str:
        """
        Take a snapshot of the current graph state.
        
        Args:
            snapshot_name: Name for the snapshot (auto-generated if None)
            
        Returns:
            str: Snapshot ID
        """
        if snapshot_name is None:
            self.current_snapshot_id += 1
            snapshot_name = f"snapshot_{self.current_snapshot_id}_{int(datetime.now().timestamp())}"
        
        # Store complete graph state
        snapshot = {
            'name': snapshot_name,
            'timestamp': datetime.now().isoformat(),
            'nodes': {nid: node.to_dict() for nid, node in self.nodes.items()},
            'edges': dict(self.edges),
            'adjacency_lists': dict(self.adjacency_lists),
            'reverse_adjacency': dict(self.reverse_adjacency),
            'edge_metadata': dict(self.edge_metadata),
            'graph_stats': dict(self.graph_stats),
            'temporal_stats': dict(self.temporal_stats),
            'update_history': {
                str(k): v.to_dict() for k, v in self.edge_updater.update_history.items()
            }
        }
        
        self.snapshots[snapshot_name] = snapshot
        self.temporal_stats['snapshots_taken'] += 1
        
        # Limit number of snapshots
        if len(self.snapshots) > self.max_snapshots:
            oldest = min(self.snapshots.keys())
            del self.snapshots[oldest]
        
        self.logger.log_info(f"Snapshot taken: {snapshot_name}")
        return snapshot_name
    
    def rollback_to_snapshot(self, snapshot_name: str) -> bool:
        """
        Rollback graph to a previous snapshot.
        
        Args:
            snapshot_name: Name of the snapshot to rollback to
            
        Returns:
            bool: True if rollback successful
        """
        if snapshot_name not in self.snapshots:
            self.logger.log_error(f"Snapshot {snapshot_name} not found")
            return False
        
        snapshot = self.snapshots[snapshot_name]
        
        try:
            # Restore nodes
            self.nodes.clear()
            self.node_id_to_type.clear()
            self.type_to_nodes.clear()
            
            for node_id, node_data in snapshot['nodes'].items():
                node = GraphNode.from_dict(node_data)
                self.nodes[node_id] = node
                self.node_id_to_type[node_id] = node.node_type
                self.type_to_nodes[node.node_type].append(node_id)
            
            # Restore edges
            self.edges = dict(snapshot['edges'])
            self.adjacency_lists = defaultdict(lambda: defaultdict(list), snapshot['adjacency_lists'])
            self.reverse_adjacency = defaultdict(lambda: defaultdict(list), snapshot['reverse_adjacency'])
            self.edge_metadata = dict(snapshot['edge_metadata'])
            self.graph_stats = dict(snapshot['graph_stats'])
            
            # Restore temporal stats
            self.temporal_stats = dict(snapshot['temporal_stats'])
            
            # Restore update history
            self.edge_updater.update_history.clear()
            for edge_key_str, record_data in snapshot['update_history'].items():
                edge_key = eval(edge_key_str)
                record = EdgeUpdateRecord(edge_key[0], edge_key[1], edge_key[2])
                record.total_updates = record_data['total_updates']
                record.last_update_time = record_data['last_update_time']
                record.update_frequency = record_data['update_frequency']
                record.update_history = record_data['update_history']
                self.edge_updater.update_history[edge_key] = record
            
            self.logger.log_info(f"Rolled back to snapshot: {snapshot_name}")
            return True
            
        except Exception as e:
            self.logger.log_error(f"Rollback failed: {str(e)}")
            return False
    
    def get_temporal_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive temporal statistics.
        
        Returns:
            Dict[str, Any]: Temporal statistics
        """
        stats = {
            'total_evolutions': self.temporal_stats['total_evolutions'],
            'nodes_added': self.temporal_stats['nodes_added'],
            'nodes_removed': self.temporal_stats.get('nodes_removed', 0),
            'edges_added': self.temporal_stats['edges_added'],
            'edges_removed': self.temporal_stats['edges_removed'],
            'edges_updated': self.temporal_stats['edges_updated'],
            'pruned_edges': self.temporal_stats['pruned_edges'],
            'snapshots_taken': self.temporal_stats['snapshots_taken'],
            'current_nodes': len(self.nodes),
            'current_edges': len(self.edges),
            'recent_evolution': self.temporal_stats['evolution_timeline'][-5:] if self.temporal_stats['evolution_timeline'] else [],
            'cache_size': len(self.propagation_cache),
            'edge_update_stats': self.edge_updater.get_stats()
        }
        
        return stats
    
    def get_edge_update_history(self, source: str, target: str, relation_type: str) -> Optional[Dict[str, Any]]:
        """
        Get update history for a specific edge.
        
        Args:
            source: Source node ID
            target: Target node ID
            relation_type: Type of relation
            
        Returns:
            Optional[Dict[str, Any]]: Update history or None
        """
        return self.edge_updater.get_update_history((source, target, relation_type))
    
    def clear_propagation_cache(self) -> None:
        """Clear the propagation cache."""
        self.propagation_cache.clear()
        self.logger.log_info("Cleared propagation cache")
    
    def get_evolution_timeline(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get evolution timeline.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List[Dict[str, Any]]: Timeline events
        """
        return self.temporal_stats['evolution_timeline'][-limit:]
    
    def get_dynamic_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive dynamic graph statistics.
        
        Returns:
            Dict[str, Any]: Combined static and dynamic statistics
        """
        stats = self.get_graph_statistics()
        stats['temporal'] = self.get_temporal_statistics()
        stats['edge_update_history_count'] = len(self.edge_updater.update_history)
        stats['propagation_cache_size'] = len(self.propagation_cache)
        stats['snapshot_count'] = len(self.snapshots)
        stats['decay_factor'] = self.decay_factor
        stats['pruning_threshold'] = self.pruning_threshold
        stats['ppr_restart_prob'] = self.ppr_restart_prob
        
        return stats
    
    def to_networkx_dynamic(self) -> Dict[str, Any]:
        """
        Convert to NetworkX with temporal information.
        
        Returns:
            Dict[str, Any]: NetworkX graph with temporal metadata
        """
        import networkx as nx
        
        G = nx.MultiDiGraph()
        
        # Add nodes with temporal info
        for node_id, node in self.nodes.items():
            G.add_node(
                node_id,
                node_type=node.node_type,
                features=node.features,
                embedding=node.embedding.numpy().tolist() if node.embedding is not None else None,
                created_at=node.created_at,
                updated_at=node.updated_at
            )
        
        # Add edges with temporal info
        for (src, dst, rel_type), weight in self.edges.items():
            # Get update history
            history = self.edge_updater.get_update_history((src, dst, rel_type))
            
            G.add_edge(
                src, dst,
                key=rel_type,
                weight=weight,
                relation_type=rel_type,
                update_count=history['total_updates'] if history else 0,
                last_update=history['last_update_time'] if history else 0,
                trend=history['trend'] if history else 'unknown'
            )
        
        return {
            'graph': G,
            'temporal_stats': self.get_temporal_statistics(),
            'timestamp': datetime.now().isoformat()
        }
    
    def __str__(self) -> str:
        """String representation with dynamic state."""
        base = super().__str__()
        return f"{base} (Dynamic: {self.temporal_stats['total_evolutions']} evolutions)"


# Example usage
if __name__ == "__main__":
    # Load configuration
    config_path = "config/default_config.yaml"
    config_loader = ConfigLoader(config_path)
    config = config_loader.load_config()
    
    # Create dynamic graph
    dynamic_graph = DynamicGraph(config)
    
    # Add some initial nodes
    dynamic_graph.add_node('user_1', 'user', features={'interacted_items': ['item_1', 'item_2']})
    dynamic_graph.add_node('user_2', 'user', features={'interacted_items': ['item_2', 'item_3']})
    dynamic_graph.add_node('item_1', 'item')
    dynamic_graph.add_node('item_2', 'item')
    dynamic_graph.add_node('item_3', 'item')
    
    # Add edges
    dynamic_graph.add_edge('user_1', 'item_1', RelationType.INTERACT, weight=0.8)
    dynamic_graph.add_edge('user_1', 'item_2', RelationType.INTERACT, weight=0.6)
    dynamic_graph.add_edge('user_2', 'item_2', RelationType.INTERACT, weight=0.7)
    dynamic_graph.add_edge('user_2', 'item_3', RelationType.INTERACT, weight=0.5)
    dynamic_graph.add_edge('user_1', 'user_2', RelationType.SIMILAR_PREF, weight=0.9)
    
    # Evolve the graph
    interactions = [
        {'user_id': 'user_1', 'item_id': 'item_2', 'success': True, 'attempts': 1},
        {'user_id': 'user_2', 'item_id': 'item_3', 'success': False, 'attempts': 2}
    ]
    
    evolution_stats = dynamic_graph.evolve_step(interactions)
    print(f"Evolution stats: {evolution_stats}")
    
    # Compute PPR scores
    ppr_scores = dynamic_graph.compute_ppr_scores('user_1')
    print(f"PPR scores for user_1: {dict(list(ppr_scores.items())[:3])}")
    
    # Get collaborative propagation paths
    paths = dynamic_graph.get_collaborative_propagation_paths('user_1', k=2, max_paths=5)
    print(f"Propagation paths: {paths}")
    
    # Take a snapshot
    snapshot_id = dynamic_graph.take_snapshot('test_snapshot')
    print(f"Snapshot taken: {snapshot_id}")
    
    # Get dynamic statistics
    stats = dynamic_graph.get_dynamic_statistics()
    print(f"Dynamic statistics: {stats}")