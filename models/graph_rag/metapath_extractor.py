"""
Metapath Extractor Module for H-GRAGrecsys

This module extracts and processes metapath instances from the heterogeneous graph,
supporting various path types for collaborative filtering and recommendation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from collections import defaultdict, deque
import heapq
from tqdm import tqdm
import sys
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.graph.dynamic_graph import DynamicGraph
from models.graph.relation_types import RelationType, RelationTypeRegistry
from utils.logger import Logger
from utils.config_loader import ConfigLoader


class MetapathExtractor:
    """
    Extracts and processes metapath instances from the heterogeneous graph.
    
    This class handles:
    - Extracting common metapath types (U-I-U, U-U-I, I-I-U, etc.)
    - Finding all paths between nodes
    - Verbalizing paths for LLM consumption
    - Assigning importance weights to paths
    - Getting path embeddings for downstream tasks
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the metapath extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = Logger.get_instance(log_dir='logs', name='metapath_extractor')
        
        # Extract configuration
        graph_rag_config = config.get('model', {}).get('graph_rag', {})
        self.max_path_length = graph_rag_config.get('max_metapath_length', 5)
        self.max_paths_per_type = graph_rag_config.get('max_paths_per_type', 20)
        self.min_path_weight = graph_rag_config.get('min_path_weight', 0.1)
        self.use_ppr_weighting = graph_rag_config.get('use_ppr_weighting', True)
        
        # Path type definitions
        self.path_type_definitions = self._initialize_path_types()
        
        # Relation registry
        self.relation_registry = RelationTypeRegistry(config)
        
        # Cache for extracted paths
        self.path_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.cache_size = graph_rag_config.get('path_cache_size', 500)
        
        # Statistics
        self.extraction_stats = {
            'total_extractions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'paths_extracted': 0,
            'paths_by_type': defaultdict(int),
            'avg_path_length': 0.0
        }
        
        self.logger.log_info(f"Initialized MetapathExtractor with max_length={self.max_path_length}")
    
    def _initialize_path_types(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize common metapath type definitions.
        
        Returns:
            Dict[str, Dict[str, Any]]: Path type definitions
        """
        return {
            'user_item_user': {
                'pattern': ['user', 'item', 'user'],
                'relations': [RelationType.INTERACT.value, RelationType.INTERACT.value],
                'description': 'Users who interacted with same items',
                'use_case': 'collaborative_filtering',
                'max_length': 3
            },
            'user_user_item': {
                'pattern': ['user', 'user', 'item'],
                'relations': [RelationType.SIMILAR_PREF.value, RelationType.INTERACT.value],
                'description': 'Items interacted by similar users',
                'use_case': 'collaborative_filtering',
                'max_length': 3
            },
            'item_item_user': {
                'pattern': ['item', 'item', 'user'],
                'relations': [RelationType.CONTENT_SIM.value, RelationType.INTERACT.value],
                'description': 'Users who interacted with similar items',
                'use_case': 'content_based',
                'max_length': 3
            },
            'user_item_item': {
                'pattern': ['user', 'item', 'item'],
                'relations': [RelationType.INTERACT.value, RelationType.CONTENT_SIM.value],
                'description': 'Similar items to those user interacted with',
                'use_case': 'content_based',
                'max_length': 3
            },
            'user_item_user_item': {
                'pattern': ['user', 'item', 'user', 'item'],
                'relations': [RelationType.INTERACT.value, RelationType.INTERACT.value, 
                             RelationType.INTERACT.value],
                'description': 'Items interacted by users who share items',
                'use_case': 'collaborative_filtering',
                'max_length': 4
            },
            'user_user_item_user': {
                'pattern': ['user', 'user', 'item', 'user'],
                'relations': [RelationType.SIMILAR_PREF.value, RelationType.INTERACT.value,
                             RelationType.INTERACT.value],
                'description': 'Users who interacted with items of similar users',
                'use_case': 'collaborative_filtering',
                'max_length': 4
            },
            'item_user_item_user': {
                'pattern': ['item', 'user', 'item', 'user'],
                'relations': [RelationType.INTERACT.value, RelationType.CONTENT_SIM.value,
                             RelationType.INTERACT.value],
                'description': 'Users who interacted with similar items',
                'use_case': 'hybrid',
                'max_length': 4
            },
            'user_item_user_user': {
                'pattern': ['user', 'item', 'user', 'user'],
                'relations': [RelationType.INTERACT.value, RelationType.CO_INTER.value,
                             RelationType.SIMILAR_PREF.value],
                'description': 'Users similar to those who interacted with items',
                'use_case': 'social',
                'max_length': 4
            },
            'all': {
                'pattern': ['*'],
                'relations': ['*'],
                'description': 'All path types',
                'use_case': 'general',
                'max_length': 5
            }
        }
    
    def extract_user_item_user(self, anchor_user: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Extract User-Item-User metapaths.
        
        Args:
            anchor_user: Starting user ID
            k: Number of paths to return
            
        Returns:
            List[Dict[str, Any]]: List of metapath instances
        """
        return self.extract_metapath_instances(anchor_user, 'user_item_user', k)
    
    def extract_user_user_item(self, anchor_user: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Extract User-User-Item metapaths.
        
        Args:
            anchor_user: Starting user ID
            k: Number of paths to return
            
        Returns:
            List[Dict[str, Any]]: List of metapath instances
        """
        return self.extract_metapath_instances(anchor_user, 'user_user_item', k)
    
    def extract_item_item_user(self, anchor_item: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Extract Item-Item-User metapaths.
        
        Args:
            anchor_item: Starting item ID
            k: Number of paths to return
            
        Returns:
            List[Dict[str, Any]]: List of metapath instances
        """
        return self.extract_metapath_instances(anchor_item, 'item_item_user', k)
    
    def extract_metapath_instances(self, start_node: str, 
                                   metapath_type: str,
                                   max_instances: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Extract metapath instances of a specific type.
        
        Args:
            start_node: Starting node ID
            metapath_type: Type of metapath (key from path_type_definitions)
            max_instances: Maximum instances to return
            
        Returns:
            List[Dict[str, Any]]: List of metapath instances with metadata
        """
        self.logger.log_info(f"Extracting {metapath_type} metapaths for node {start_node}")
        
        # Check cache
        cache_key = f"{start_node}_{metapath_type}"
        if cache_key in self.path_cache:
            self.logger.log_info(f"Cache hit for {cache_key}")
            self.extraction_stats['cache_hits'] += 1
            cached_paths = self.path_cache[cache_key]
            if max_instances:
                return cached_paths[:max_instances]
            return cached_paths
        
        self.extraction_stats['cache_misses'] += 1
        
        # Get path type definition
        path_def = self.path_type_definitions.get(metapath_type)
        if not path_def:
            self.logger.log_warning(f"Unknown metapath type: {metapath_type}")
            return []
        
        # Extract paths
        if metapath_type == 'all':
            paths = self._extract_all_metapaths(start_node, max_instances or self.max_paths_per_type)
        else:
            paths = self._extract_specific_metapath(
                start_node, 
                path_def['pattern'],
                path_def['relations'],
                max_instances or self.max_paths_per_type
            )
        
        # Assign importance weights
        for path in paths:
            path['importance'] = self.assign_path_importance(path)
        
        # Sort by importance
        paths.sort(key=lambda x: x['importance'], reverse=True)
        
        # Cache results
        if len(self.path_cache) < self.cache_size:
            self.path_cache[cache_key] = paths
        
        # Update statistics
        self.extraction_stats['total_extractions'] += 1
        self.extraction_stats['paths_extracted'] += len(paths)
        self.extraction_stats['paths_by_type'][metapath_type] += len(paths)
        
        # Update average path length
        if paths:
            avg_len = np.mean([len(p.get('path', [])) for p in paths])
            self.extraction_stats['avg_path_length'] = (
                (self.extraction_stats['avg_path_length'] * (self.extraction_stats['total_extractions'] - 1) + 
                 avg_len) / self.extraction_stats['total_extractions']
            )
        
        self.logger.log_info(f"Extracted {len(paths)} {metapath_type} metapaths")
        return paths
    
    def _extract_specific_metapath(self, start_node: str, 
                                   pattern: List[str],
                                   relations: List[str],
                                   max_paths: int) -> List[Dict[str, Any]]:
        """
        Extract specific metapath type from graph.
        
        Args:
            start_node: Starting node ID
            pattern: List of node types in path
            relations: List of relation types
            max_paths: Maximum paths to return
            
        Returns:
            List[Dict[str, Any]]: Extracted metapaths
        """
        paths = []
        
        # Check if node exists and matches type
        if start_node not in self.graph.nodes:
            self.logger.log_warning(f"Node {start_node} not found in graph")
            return paths
        
        start_type = self.graph.node_id_to_type.get(start_node, 'unknown')
        if pattern[0] != '*' and start_type != pattern[0]:
            self.logger.log_warning(f"Node type mismatch: expected {pattern[0]}, got {start_type}")
            return paths
        
        # BFS to find paths
        queue = deque([(start_node, [start_node], [])])
        visited = set([start_node])
        
        while queue and len(paths) < max_paths:
            current_node, node_path, rel_path = queue.popleft()
            
            # Check if path length matches pattern
            if len(node_path) == len(pattern):
                # Verify path matches pattern
                if self._matches_pattern(node_path, pattern):
                    paths.append({
                        'start_node': start_node,
                        'end_node': node_path[-1],
                        'path': node_path,
                        'relations': rel_path,
                        'path_type': '_'.join(pattern),
                        'length': len(node_path)
                    })
                continue
            
            # Get next node type from pattern
            next_idx = len(node_path)
            if next_idx >= len(pattern):
                continue
            
            next_type = pattern[next_idx]
            
            # Get neighbors matching next type
            neighbors = []
            for rel_type in RelationType.get_all_types():
                if rel_type not in relations and relations[0] != '*':
                    continue
                
                for neighbor, weight in self.graph.get_neighbors(current_node, rel_type):
                    neighbor_type = self.graph.node_id_to_type.get(neighbor, 'unknown')
                    
                    if next_type == '*' or neighbor_type == next_type:
                        if neighbor not in node_path:  # Avoid cycles
                            neighbors.append((neighbor, rel_type, weight))
            
            # Sort by weight
            neighbors.sort(key=lambda x: x[2], reverse=True)
            
            # Add to queue
            for neighbor, rel_type, weight in neighbors[:10]:  # Limit branching
                queue.append((
                    neighbor,
                    node_path + [neighbor],
                    rel_path + [rel_type]
                ))
        
        return paths
    
    def _extract_all_metapaths(self, start_node: str, max_paths: int) -> List[Dict[str, Any]]:
        """
        Extract all types of metapaths from a node.
        
        Args:
            start_node: Starting node ID
            max_paths: Maximum paths to return
            
        Returns:
            List[Dict[str, Any]]: All extracted metapaths
        """
        all_paths = []
        
        # Extract each path type
        for path_type in self.path_type_definitions.keys():
            if path_type != 'all':
                paths = self.extract_metapath_instances(start_node, path_type, max_paths // 4)
                all_paths.extend(paths)
        
        # Sort by importance
        all_paths.sort(key=lambda x: x.get('importance', 0), reverse=True)
        
        return all_paths[:max_paths]
    
    def _matches_pattern(self, node_path: List[str], pattern: List[str]) -> bool:
        """
        Check if a node path matches a pattern.
        
        Args:
            node_path: List of node IDs
            pattern: List of node types
            
        Returns:
            bool: True if path matches pattern
        """
        if len(node_path) != len(pattern):
            return False
        
        for node_id, expected_type in zip(node_path, pattern):
            actual_type = self.graph.node_id_to_type.get(node_id, 'unknown')
            if expected_type != '*' and actual_type != expected_type:
                return False
        
        return True
    
    def extract_all_paths(self, start_node: str, 
                         path_types: Optional[List[str]] = None,
                         max_len: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Extract all paths from a node with specified path types.
        
        Args:
            start_node: Starting node ID
            path_types: List of path types to extract (None for all)
            max_len: Maximum path length
            
        Returns:
            List[Dict[str, Any]]: All extracted paths
        """
        if path_types is None:
            path_types = ['all']
        
        all_paths = []
        max_len = max_len or self.max_path_length
        
        for path_type in path_types:
            if path_type == 'all':
                paths = self._extract_all_metapaths(
                    start_node, 
                    self.max_paths_per_type
                )
            else:
                paths = self.extract_metapath_instances(
                    start_node, 
                    path_type,
                    self.max_paths_per_type
                )
            
            # Filter by max length
            if max_len:
                paths = [p for p in paths if p.get('length', 0) <= max_len]
            
            all_paths.extend(paths)
        
        # Remove duplicates
        unique_paths = []
        seen = set()
        for path in all_paths:
            path_key = tuple(path.get('path', []))
            if path_key not in seen:
                seen.add(path_key)
                unique_paths.append(path)
        
        # Sort by importance
        unique_paths.sort(key=lambda x: x.get('importance', 0), reverse=True)
        
        self.logger.log_info(f"Extracted {len(unique_paths)} total paths from {start_node}")
        return unique_paths
    
    def verbalize_path(self, path: Dict[str, Any], relation_type: Optional[str] = None) -> str:
        """
        Convert a metapath to natural language description.
        
        Args:
            path: Metapath dictionary
            relation_type: Type of relation for verbalization
            
        Returns:
            str: Natural language description of the path
        """
        if not path:
            return ""
        
        node_path = path.get('path', [])
        rel_path = path.get('relations', [])
        
        if not node_path:
            return ""
        
        # Get node types
        node_types = [self.graph.node_id_to_type.get(nid, 'unknown') for nid in node_path]
        
        # Build verbalization
        verbalization = []
        for i, node_id in enumerate(node_path):
            node_type = node_types[i]
            
            # Add node description
            if node_type == 'user':
                # Try to get user name or ID
                user_name = self._get_node_label(node_id, 'user')
                verbalization.append(f"user {user_name}")
            elif node_type == 'item':
                item_name = self._get_node_label(node_id, 'item')
                verbalization.append(f"item {item_name}")
            else:
                verbalization.append(f"{node_type} {node_id}")
            
            # Add relation description
            if i < len(rel_path):
                rel_type = rel_path[i]
                rel_desc = self._get_relation_description(rel_type)
                verbalization.append(rel_desc)
        
        # Join with spaces
        return " ".join(verbalization)
    
    def _get_node_label(self, node_id: str, node_type: str) -> str:
        """
        Get a readable label for a node.
        
        Args:
            node_id: Node ID
            node_type: Type of node
            
        Returns:
            str: Readable label
        """
        node = self.graph.nodes.get(node_id)
        if not node:
            return node_id
        
        # Try to extract a name or title
        features = node.features
        if node_type == 'user':
            return features.get('name', features.get('username', node_id))
        elif node_type == 'item':
            return features.get('title', features.get('name', node_id))
        
        return node_id
    
    def _get_relation_description(self, relation_type: str) -> str:
        """
        Get a description for a relation type.
        
        Args:
            relation_type: Type of relation
            
        Returns:
            str: Description of the relation
        """
        descriptions = {
            RelationType.INTERACT.value: "interacted with",
            RelationType.SIMILAR_PREF.value: "has similar preferences to",
            RelationType.CO_INTER.value: "co-interacted with",
            RelationType.CONTENT_SIM.value: "is similar to"
        }
        return descriptions.get(relation_type, f"connected by {relation_type}")
    
    def assign_path_importance(self, path: Dict[str, Any]) -> float:
        """
        Assign importance weight to a metapath.
        
        Args:
            path: Metapath dictionary
            
        Returns:
            float: Importance weight
        """
        path_length = path.get('length', 0)
        path_type = path.get('path_type', '')
        
        # Base importance: longer paths are less important
        length_penalty = 1.0 / (1.0 + 0.5 * (path_length - 3)) if path_length > 3 else 1.0
        
        # Type-specific weights
        type_weights = {
            'user_item_user': 1.0,
            'user_user_item': 0.9,
            'item_item_user': 0.8,
            'user_item_item': 0.8,
            'user_item_user_item': 0.7,
            'user_user_item_user': 0.7,
            'item_user_item_user': 0.6,
            'user_item_user_user': 0.6
        }
        
        type_weight = type_weights.get(path_type, 0.5)
        
        # Edge weights along path
        edge_weights = []
        for rel_type in path.get('relations', []):
            # Get average weight for this relation type
            weights = [w for (src, dst, rt), w in self.graph.edges.items() if rt == rel_type]
            edge_weight = np.mean(weights) if weights else 0.5
            edge_weights.append(edge_weight)
        
        avg_edge_weight = np.mean(edge_weights) if edge_weights else 0.5
        
        # PPR-based weighting
        ppr_weight = 1.0
        if self.use_ppr_weighting and path.get('end_node'):
            ppr_scores = self.graph.compute_ppr_scores(path['start_node'])
            ppr_weight = ppr_scores.get(path['end_node'], 0.1)
            ppr_weight = max(0.1, min(1.0, ppr_weight * 2))  # Scale to [0.1, 1.0]
        
        # Combine factors
        importance = (
            0.3 * length_penalty +
            0.3 * type_weight +
            0.2 * avg_edge_weight +
            0.2 * ppr_weight
        )
        
        return min(1.0, max(0.0, importance))
    
    def get_path_embedding(self, path: Dict[str, Any]) -> torch.Tensor:
        """
        Get embedding for a metapath.
        
        Args:
            path: Metapath dictionary
            
        Returns:
            torch.Tensor: Path embedding
        """
        # Get embeddings for nodes in path
        node_embeddings = []
        for node_id in path.get('path', []):
            node = self.graph.nodes.get(node_id)
            if node and node.embedding is not None:
                node_embeddings.append(node.embedding)
            else:
                # Fallback to zero vector
                node_embeddings.append(torch.zeros(self.config.get('model', {}).get('gnn', {}).get('hidden_dim', 256)))
        
        if not node_embeddings:
            return torch.zeros(self.config.get('model', {}).get('gnn', {}).get('hidden_dim', 256))
        
        # Stack embeddings
        embeddings = torch.stack(node_embeddings)
        
        # Average pooling
        path_embedding = torch.mean(embeddings, dim=0)
        
        return path_embedding
    
    def get_path_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about path extraction.
        
        Returns:
            Dict[str, Any]: Path statistics
        """
        stats = self.extraction_stats.copy()
        stats['cache_size'] = len(self.path_cache)
        stats['cache_hit_rate'] = (
            stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
            if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0.0
        )
        stats['path_types_available'] = list(self.path_type_definitions.keys())
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the path cache."""
        self.path_cache.clear()
        self.logger.log_info("Cleared path cache")
    
    def get_metapath_patterns(self, node_type: str, target_type: str) -> List[str]:
        """
        Get relevant metapath patterns for connecting node types.
        
        Args:
            node_type: Source node type
            target_type: Target node type
            
        Returns:
            List[str]: List of metapath type names
        """
        patterns = []
        
        for path_type, path_def in self.path_type_definitions.items():
            if path_type == 'all':
                continue
            
            pattern = path_def.get('pattern', [])
            if pattern and pattern[0] == node_type and pattern[-1] == target_type:
                patterns.append(path_type)
        
        return patterns
    
    def convert_path_to_text(self, path: Dict[str, Any], 
                            include_relations: bool = True) -> str:
        """
        Convert path to human-readable text.
        
        Args:
            path: Metapath dictionary
            include_relations: Whether to include relation descriptions
            
        Returns:
            str: Text description of the path
        """
        nodes = path.get('path', [])
        relations = path.get('relations', [])
        
        if not nodes:
            return ""
        
        # Get node descriptions
        node_texts = []
        for node_id in nodes:
            node_type = self.graph.node_id_to_type.get(node_id, 'unknown')
            label = self._get_node_label(node_id, node_type)
            node_texts.append(f"{node_type} '{label}'")
        
        # Build text
        if include_relations and relations:
            result = []
            for i, node_text in enumerate(node_texts):
                result.append(node_text)
                if i < len(relations):
                    rel_desc = self._get_relation_description(relations[i])
                    result.append(rel_desc)
            return " ".join(result)
        else:
            return " -> ".join(node_texts)
    
    def get_diverse_paths(self, start_node: str, 
                         num_paths: int = 10,
                         diversity_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Get diverse set of metapaths to avoid redundancy.
        
        Args:
            start_node: Starting node ID
            num_paths: Number of paths to return
            diversity_weight: Weight for diversity vs importance
            
        Returns:
            List[Dict[str, Any]]: Diverse set of paths
        """
        # Get all paths
        all_paths = self.extract_all_paths(start_node)
        
        if len(all_paths) <= num_paths:
            return all_paths
        
        # Use MMR (Maximum Marginal Relevance) for diversity
        selected_paths = []
        remaining_paths = all_paths.copy()
        
        # Select first path (highest importance)
        first_path = max(remaining_paths, key=lambda x: x.get('importance', 0))
        selected_paths.append(first_path)
        remaining_paths.remove(first_path)
        
        # Select remaining paths
        for _ in range(num_paths - 1):
            if not remaining_paths:
                break
            
            best_score = -1
            best_path = None
            
            for path in remaining_paths:
                # Importance score
                importance = path.get('importance', 0)
                
                # Diversity score (maximize difference from selected)
                diversity = 1.0
                for selected in selected_paths:
                    similarity = self._path_similarity(path, selected)
                    diversity = min(diversity, 1.0 - similarity)
                
                # Combined score
                score = (1 - diversity_weight) * importance + diversity_weight * diversity
                
                if score > best_score:
                    best_score = score
                    best_path = path
            
            if best_path:
                selected_paths.append(best_path)
                remaining_paths.remove(best_path)
        
        return selected_paths
    
    def _path_similarity(self, path_a: Dict[str, Any], path_b: Dict[str, Any]) -> float:
        """
        Compute similarity between two paths.
        
        Args:
            path_a: First path
            path_b: Second path
            
        Returns:
            float: Similarity score
        """
        nodes_a = set(path_a.get('path', []))
        nodes_b = set(path_b.get('path', []))
        
        if not nodes_a or not nodes_b:
            return 0.0
        
        # Jaccard similarity of node sets
        intersection = len(nodes_a & nodes_b)
        union = len(nodes_a | nodes_b)
        
        return intersection / union if union > 0 else 0.0
    
    def visualize_path(self, path: Dict[str, Any], output_file: Optional[str] = None) -> None:
        """
        Visualize a metapath for debugging or analysis.
        
        Args:
            path: Metapath dictionary
            output_file: Output file path (optional)
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        
        # Create a directed graph for the path
        G = nx.DiGraph()
        nodes = path.get('path', [])
        relations = path.get('relations', [])
        
        # Add nodes
        for i, node_id in enumerate(nodes):
            node_type = self.graph.node_id_to_type.get(node_id, 'unknown')
            G.add_node(node_id, label=f"{node_type}\n{node_id[:8]}")
        
        # Add edges
        for i in range(len(nodes) - 1):
            if i < len(relations):
                G.add_edge(nodes[i], nodes[i+1], label=relations[i])
        
        # Draw the graph
        plt.figure(figsize=(12, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=2000, font_size=8, font_weight='bold')
        
        # Add edge labels
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title(f"Metapath: {path.get('path_type', 'unknown')}")
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.log_info(f"Saved path visualization to {output_file}")
        else:
            plt.show()
        
        plt.close()


# Example usage
if __name__ == "__main__":
    # Load configuration
    config_path = "config/default_config.yaml"
    config_loader = ConfigLoader(config_path)
    config = config_loader.load_config()
    
    # Create dynamic graph
    from models.graph.dynamic_graph import DynamicGraph
    graph = DynamicGraph(config)
    
    # Add some nodes and edges (sample data)
    graph.add_node('user_1', 'user', features={'name': 'Alice'})
    graph.add_node('user_2', 'user', features={'name': 'Bob'})
    graph.add_node('user_3', 'user', features={'name': 'Charlie'})
    graph.add_node('item_1', 'item', features={'title': 'Product A'})
    graph.add_node('item_2', 'item', features={'title': 'Product B'})
    graph.add_node('item_3', 'item', features={'title': 'Product C'})
    
    graph.add_edge('user_1', 'item_1', RelationType.INTERACT, weight=0.9)
    graph.add_edge('user_1', 'item_2', RelationType.INTERACT, weight=0.7)
    graph.add_edge('user_2', 'item_2', RelationType.INTERACT, weight=0.8)
    graph.add_edge('user_2', 'item_3', RelationType.INTERACT, weight=0.6)
    graph.add_edge('user_3', 'item_1', RelationType.INTERACT, weight=0.5)
    graph.add_edge('user_3', 'item_3', RelationType.INTERACT, weight=0.4)
    graph.add_edge('user_1', 'user_2', RelationType.SIMILAR_PREF, weight=0.8)
    graph.add_edge('user_1', 'user_3', RelationType.SIMILAR_PREF, weight=0.6)
    graph.add_edge('item_1', 'item_2', RelationType.CONTENT_SIM, weight=0.6)
    graph.add_edge('item_2', 'item_3', RelationType.CONTENT_SIM, weight=0.4)
    
    # Create metapath extractor
    extractor = MetapathExtractor(config)
    extractor.graph = graph  # Set graph reference
    
    # Extract different metapath types
    ui_u_paths = extractor.extract_user_item_user('user_1', k=5)
    print(f"User-Item-User paths: {len(ui_u_paths)}")
    
    uu_i_paths = extractor.extract_user_user_item('user_1', k=5)
    print(f"User-User-Item paths: {len(uu_i_paths)}")
    
    # Extract all paths
    all_paths = extractor.extract_all_paths('user_1')
    print(f"All paths: {len(all_paths)}")
    
    # Get diverse paths
    diverse_paths = extractor.get_diverse_paths('user_1', num_paths=5)
    print(f"Diverse paths: {len(diverse_paths)}")
    
    # Verbalize a path
    if all_paths:
        verbalized = extractor.verbalize_path(all_paths[0])
        print(f"Verbalized path: {verbalized}")
    
    # Get statistics
    stats = extractor.get_path_statistics()
    print(f"Path statistics: {stats}")