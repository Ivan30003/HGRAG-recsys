"""
Metapath Extractor Module
Implements multi-hop metapath extraction from heterogeneous interaction graphs.
Provides structured relational context for Graph RAG retrieval in the Hybrid-GraphRAG framework.
"""

import logging
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field
import numpy as np
import heapq

logger = logging.getLogger(__name__)


@dataclass
class Metapath:
    """
    Represents a metapath in the heterogeneous graph.
    A metapath is a sequence of nodes connected by specified edge types.
    """
    # Sequence of node IDs in the path
    nodes: List[str]
    # Sequence of edge types connecting nodes
    edge_types: List[str]
    # Type of the metapath (e.g., 'user-item-user', 'user-user-item')
    path_type: str
    # Cumulative score/weight of the path
    score: float = 1.0
    # Length of the path (number of edges)
    length: int = 0
    # Metadata about the path
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.length = len(self.edge_types)
    
    def to_description(self) -> str:
        """Generate human-readable description of the metapath."""
        if self.path_type == 'user-item-user':
            return (
                f"Users who interacted with '{self._get_item_name(1)}' "
                f"also have similar preferences to you"
            )
        elif self.path_type == 'user-user-item':
            return (
                f"User '{self._get_user_name(1)}' with similar preferences "
                f"to you interacted with '{self._get_item_name(2)}'"
            )
        elif self.path_type == 'item-item-analogy':
            return (
                f"Item '{self._get_item_name(0)}' is similar to "
                f"'{self._get_item_name(1)}' which you previously enjoyed"
            )
        else:
            return f"Path: {' -> '.join(self.nodes)}"
    
    def _get_item_name(self, position: int) -> str:
        """Get item name from path position."""
        if position < len(self.nodes):
            return self.nodes[position]
        return "unknown"
    
    def _get_user_name(self, position: int) -> str:
        """Get user name from path position."""
        if position < len(self.nodes):
            return self.nodes[position]
        return "unknown"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'nodes': self.nodes,
            'edge_types': self.edge_types,
            'path_type': self.path_type,
            'score': self.score,
            'length': self.length,
            'description': self.to_description(),
            'metadata': self.metadata
        }


@dataclass
class GraphContext:
    """
    Structured context extracted from graph for Graph RAG.
    Contains neighbors, metapaths, and statistics.
    """
    # Center node ID
    center_node: str
    # 1-hop neighbors by edge type
    neighbors_1hop: Dict[str, List[Tuple[str, float]]]
    # 2-hop neighbors by edge type
    neighbors_2hop: Dict[str, List[Tuple[str, float]]]
    # Extracted metapaths
    metapaths: List[Metapath]
    # Similar users with their preferences
    similar_users: List[Dict]
    # Item relationships
    item_relations: List[Dict]
    # Graph statistics
    statistics: Dict
    # Influential paths (for reflection)
    influential_paths: List[Dict]
    # Popular items in neighborhood
    popular_in_neighborhood: List[str]
    # Trending items
    trending_items: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'center_node': self.center_node,
            'neighbors_1hop': {
                k: [(n, w) for n, w in v] 
                for k, v in self.neighbors_1hop.items()
            },
            'neighbors_2hop': {
                k: [(n, w) for n, w in v] 
                for k, v in self.neighbors_2hop.items()
            },
            'metapaths': [mp.to_dict() for mp in self.metapaths],
            'similar_users': self.similar_users,
            'item_relations': self.item_relations,
            'statistics': self.statistics,
            'influential_paths': self.influential_paths,
            'popular_in_neighborhood': self.popular_in_neighborhood,
            'trending_items': self.trending_items
        }


class MetapathExtractor:
    """
    Extracts multi-hop metapaths from heterogeneous interaction graphs.
    
    Supports three metapath types:
    1. User-Item-User: u -> i -> u' 
       (users who interacted with same items)
    2. User-User-Item: u -> u' -> i
       (items that similar users interacted with)
    3. Item-Item-Analogy: i -> i' -> u_history
       (items similar to ones the user has interacted with)
    
    Used by the Graph RAG mechanism to provide structured relational
    context for LLM-based agent reasoning.
    """
    
    # Metapath type definitions
    METAPATH_TYPES = {
        'user-item-user': {
            'pattern': ['user', 'item', 'user'],
            'edge_types': ['interact', 'interact'],
            'description': 'Users who interacted with the same items'
        },
        'user-user-item': {
            'pattern': ['user', 'user', 'item'],
            'edge_types': ['similar_pref', 'interact'],
            'description': 'Items that similar users interacted with'
        },
        'item-item-analogy': {
            'pattern': ['item', 'item', 'user_history'],
            'edge_types': ['content_sim', 'interact'],
            'description': 'Items similar to ones the user enjoyed'
        }
    }
    
    def __init__(self,
                 max_paths_per_type: int = 5,
                 min_path_score: float = 0.1,
                 use_edge_weights: bool = True,
                 diversity_weight: float = 0.2):
        """
        Initialize metapath extractor.
        
        Args:
            max_paths_per_type: Maximum paths to extract per metapath type
            min_path_score: Minimum cumulative score for path inclusion
            use_edge_weights: Whether to use edge weights in scoring
            diversity_weight: Weight for path diversity (0 = no diversity, 1 = max diversity)
        """
        self.max_paths_per_type = max_paths_per_type
        self.min_path_score = min_path_score
        self.use_edge_weights = use_edge_weights
        self.diversity_weight = diversity_weight
        
        # Cache for frequently accessed paths
        self._path_cache: Dict[str, List[Metapath]] = {}
        self._cache_ttl = 100  # Number of extractions before cache refresh
        self._cache_counter = 0
        
        # Statistics
        self.stats = {
            'total_extractions': 0,
            'avg_paths_found': 0.0,
            'avg_extraction_time': 0.0,
            'cache_hits': 0
        }
    
    def extract_context(self,
                        user_id: str,
                        candidate_ids: List[str],
                        graph: Any,  # HeterogeneousGraph instance
                        max_hops: int = 2,
                        top_k: int = 15
                        ) -> Dict:
        """
        Extract comprehensive graph context for a user-candidate pair.
        
        This is the main entry point for Graph RAG retrieval.
        
        Args:
            user_id: Center user agent ID
            candidate_ids: List of candidate item IDs to evaluate
            graph: HeterogeneousGraph instance
            max_hops: Maximum number of hops for neighbor extraction
            top_k: Maximum total paths to include in context
        
        Returns:
            Dictionary with structured graph context
        """
        import time
        start_time = time.time()
        
        self.stats['total_extractions'] += 1
        
        # Check cache
        cache_key = f"{user_id}_{'_'.join(sorted(candidate_ids[:5]))}"
        if cache_key in self._path_cache and self._cache_counter < self._cache_ttl:
            self.stats['cache_hits'] += 1
            return self._path_cache[cache_key]
        
        # Extract neighbors
        neighbors_1hop = self._extract_neighbors(graph, user_id, hops=1)
        neighbors_2hop = self._extract_neighbors(graph, user_id, hops=2)
        
        # Extract metapaths
        metapaths = self._extract_all_metapaths(
            user_id, candidate_ids, graph, max_hops
        )
        
        # Get similar users
        similar_users = self._extract_similar_users(
            user_id, graph, top_k=5
        )
        
        # Get item relationships
        item_relations = self._extract_item_relations(
            candidate_ids, graph
        )
        
        # Compute statistics
        statistics = self._compute_statistics(
            user_id, candidate_ids, graph, neighbors_1hop, neighbors_2hop
        )
        
        # Identify influential paths
        influential_paths = self._identify_influential_paths(
            metapaths, user_id, candidate_ids
        )
        
        # Find popular items in neighborhood
        popular_in_neighborhood = self._find_popular_in_neighborhood(
            graph, neighbors_1hop, neighbors_2hop, candidate_ids
        )
        
        # Build context object
        context = GraphContext(
            center_node=user_id,
            neighbors_1hop=neighbors_1hop,
            neighbors_2hop=neighbors_2hop,
            metapaths=metapaths[:top_k],
            similar_users=similar_users,
            item_relations=item_relations,
            statistics=statistics,
            influential_paths=influential_paths,
            popular_in_neighborhood=popular_in_neighborhood,
            trending_items=self._find_trending_items(graph, neighbors_1hop)
        )
        
        # Update cache
        self._cache_counter += 1
        if self._cache_counter >= self._cache_ttl:
            self._path_cache.clear()
            self._cache_counter = 0
        self._path_cache[cache_key] = context.to_dict()
        
        # Update statistics
        elapsed = time.time() - start_time
        self.stats['avg_extraction_time'] = (
            self.stats['avg_extraction_time'] * (self.stats['total_extractions'] - 1) + elapsed
        ) / self.stats['total_extractions']
        self.stats['avg_paths_found'] = (
            self.stats['avg_paths_found'] * (self.stats['total_extractions'] - 1) + len(metapaths)
        ) / self.stats['total_extractions']
        
        return context.to_dict()
    
    def _extract_all_metapaths(self,
                                user_id: str,
                                candidate_ids: List[str],
                                graph: Any,
                                max_hops: int = 2
                                ) -> List[Metapath]:
        """
        Extract all types of metapaths.
        
        Args:
            user_id: Center user ID
            candidate_ids: Candidate item IDs
            graph: HeterogeneousGraph instance
            max_hops: Maximum hops
        
        Returns:
            List of Metapath objects
        """
        all_paths = []
        
        # 1. User-Item-User paths
        uiu_paths = self._extract_user_item_user_paths(
            user_id, graph, max_hops
        )
        all_paths.extend(uiu_paths)
        
        # 2. User-User-Item paths
        for candidate_id in candidate_ids:
            uui_paths = self._extract_user_user_item_paths(
                user_id, candidate_id, graph
            )
            all_paths.extend(uui_paths)
        
        # 3. Item-Item-Analogy paths
        iia_paths = self._extract_item_item_analogy_paths(
            user_id, candidate_ids, graph
        )
        all_paths.extend(iia_paths)
        
        # Sort by score and apply diversity
        all_paths = self._rank_and_diversify(all_paths)
        
        return all_paths
    
    def _extract_user_item_user_paths(self,
                                        user_id: str,
                                        graph: Any,
                                        max_hops: int = 2
                                        ) -> List[Metapath]:
        """
        Extract User-Item-User metapaths.
        Finds users who interacted with the same items as the center user.
        
        Args:
            user_id: Center user ID
            graph: HeterogeneousGraph instance
            max_hops: Maximum hops
        
        Returns:
            List of Metapath objects
        """
        paths = []
        
        # Get items the user interacted with
        user_items = graph.adjacency.get('interact', {}).get(user_id, set())
        
        if not user_items:
            return paths
        
        # For each item, find other users who interacted with it
        for item_id in list(user_items)[:10]:  # Limit to top 10 items
            # Get other users who interacted with this item
            item_users = graph.adjacency.get('interact', {}).get(item_id, set())
            
            for other_user in item_users:
                if other_user == user_id:
                    continue
                
                # Compute path score
                score = 1.0
                if self.use_edge_weights:
                    w1 = graph.get_edge_weight(user_id, item_id, 'interact')
                    w2 = graph.get_edge_weight(item_id, other_user, 'interact')
                    score = w1 * w2
                
                if score < self.min_path_score:
                    continue
                
                # Get user similarity
                user_sim = graph.get_edge_weight(user_id, other_user, 'similar_pref')
                
                path = Metapath(
                    nodes=[user_id, item_id, other_user],
                    edge_types=['interact', 'interact'],
                    path_type='user-item-user',
                    score=score,
                    metadata={
                        'shared_item': item_id,
                        'similar_user': other_user,
                        'user_similarity': user_sim,
                        'shared_interactions': self._count_shared_interactions(
                            user_id, other_user, graph
                        )
                    }
                )
                paths.append(path)
        
        return paths[:self.max_paths_per_type]
    
    def _extract_user_user_item_paths(self,
                                        user_id: str,
                                        candidate_id: str,
                                        graph: Any
                                        ) -> List[Metapath]:
        """
        Extract User-User-Item metapaths.
        Finds items that similar users have interacted with.
        
        Args:
            user_id: Center user ID
            candidate_id: Candidate item ID
            graph: HeterogeneousGraph instance
        
        Returns:
            List of Metapath objects
        """
        paths = []
        
        # Get similar users
        similar_users = graph.adjacency.get('similar_pref', {}).get(user_id, set())
        
        if not similar_users:
            return paths
        
        # For each similar user, check if they interacted with candidate
        for sim_user in list(similar_users)[:10]:
            # Check if this similar user interacted with the candidate
            sim_user_items = graph.adjacency.get('interact', {}).get(sim_user, set())
            
            if candidate_id in sim_user_items:
                # Compute path score
                score = 1.0
                if self.use_edge_weights:
                    w1 = graph.get_edge_weight(user_id, sim_user, 'similar_pref')
                    w2 = graph.get_edge_weight(sim_user, candidate_id, 'interact')
                    score = w1 * w2
                
                if score < self.min_path_score:
                    continue
                
                path = Metapath(
                    nodes=[user_id, sim_user, candidate_id],
                    edge_types=['similar_pref', 'interact'],
                    path_type='user-user-item',
                    score=score,
                    metadata={
                        'similar_user': sim_user,
                        'candidate_item': candidate_id,
                        'user_similarity': graph.get_edge_weight(
                            user_id, sim_user, 'similar_pref'
                        ),
                        'interaction_strength': graph.get_edge_weight(
                            sim_user, candidate_id, 'interact'
                        )
                    }
                )
                paths.append(path)
        
        return paths[:self.max_paths_per_type]
    
    def _extract_item_item_analogy_paths(self,
                                           user_id: str,
                                           candidate_ids: List[str],
                                           graph: Any
                                           ) -> List[Metapath]:
        """
        Extract Item-Item-Analogy metapaths.
        Finds candidate items similar to items the user has enjoyed.
        
        Args:
            user_id: Center user ID
            candidate_ids: Candidate item IDs
            graph: HeterogeneousGraph instance
        
        Returns:
            List of Metapath objects
        """
        paths = []
        
        # Get items the user has interacted with
        user_items = graph.adjacency.get('interact', {}).get(user_id, set())
        
        if not user_items:
            return paths
        
        # For each candidate, find similar items the user has interacted with
        for candidate_id in candidate_ids:
            # Get items similar to the candidate
            similar_items = graph.adjacency.get('content_sim', {}).get(candidate_id, set())
            
            # Find intersection with user's items
            common_items = similar_items & user_items
            
            for common_item in common_items:
                # Compute path score
                score = 1.0
                if self.use_edge_weights:
                    w1 = graph.get_edge_weight(candidate_id, common_item, 'content_sim')
                    w2 = graph.get_edge_weight(user_id, common_item, 'interact')
                    score = w1 * w2
                
                if score < self.min_path_score:
                    continue
                
                path = Metapath(
                    nodes=[candidate_id, common_item, user_id],
                    edge_types=['content_sim', 'interact'],
                    path_type='item-item-analogy',
                    score=score,
                    metadata={
                        'candidate_item': candidate_id,
                        'similar_to': common_item,
                        'similarity_score': graph.get_edge_weight(
                            candidate_id, common_item, 'content_sim'
                        ),
                        'user_interaction_strength': graph.get_edge_weight(
                            user_id, common_item, 'interact'
                        )
                    }
                )
                paths.append(path)
        
        return paths[:self.max_paths_per_type]
    
    def _extract_neighbors(self,
                            graph: Any,
                            center_id: str,
                            hops: int = 1
                            ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Extract neighbors at specified hop distance.
        
        Args:
            graph: HeterogeneousGraph instance
            center_id: Center node ID
            hops: Number of hops
        
        Returns:
            Dictionary mapping edge_type -> list of (neighbor_id, weight)
        """
        all_neighbors = graph.get_neighbors(
            center_id, 
            edge_types=None,
            max_hops=hops
        )
        
        result = {}
        for hop, nodes in all_neighbors.items():
            if hop > hops:
                break
            
            # Get edge types for each neighbor
            edge_type_neighbors = defaultdict(list)
            for node in nodes:
                for edge_type in ['interact', 'similar_pref', 'co_interact', 'content_sim']:
                    weight = graph.get_edge_weight(center_id, node, edge_type)
                    if weight > 0:
                        edge_type_neighbors[edge_type].append((node, weight))
            
            for edge_type, neighbors in edge_type_neighbors.items():
                key = f"{edge_type}_{hop}hop"
                if key not in result:
                    result[key] = []
                result[key].extend(neighbors)
        
        return result
    
    def _extract_similar_users(self,
                                user_id: str,
                                graph: Any,
                                top_k: int = 5
                                ) -> List[Dict]:
        """
        Extract similar users and their preferences.
        
        Args:
            user_id: Center user ID
            graph: HeterogeneousGraph instance
            top_k: Maximum number of similar users
        
        Returns:
            List of similar user dictionaries
        """
        similar = []
        
        # Get users with similar preferences
        similar_users = graph.adjacency.get('similar_pref', {}).get(user_id, set())
        
        # Sort by similarity weight
        user_scores = []
        for su_id in similar_users:
            sim = graph.get_edge_weight(user_id, su_id, 'similar_pref')
            user_scores.append((su_id, sim))
        
        user_scores.sort(key=lambda x: x[1], reverse=True)
        
        for su_id, sim in user_scores[:top_k]:
            # Get this user's interacted items
            interacted = graph.adjacency.get('interact', {}).get(su_id, set())
            
            # Get shared items
            user_items = graph.adjacency.get('interact', {}).get(user_id, set())
            shared = list(interacted & user_items)[:5]
            
            similar.append({
                'user_id': su_id,
                'similarity': sim,
                'shared_items': shared,
                'interaction_count': len(interacted),
                'preferences': f"User with {sim:.0%} preference similarity"
            })
        
        return similar
    
    def _extract_item_relations(self,
                                 candidate_ids: List[str],
                                 graph: Any
                                 ) -> List[Dict]:
        """
        Extract relationships between candidate items and others.
        
        Args:
            candidate_ids: Candidate item IDs
            graph: HeterogeneousGraph instance
        
        Returns:
            List of item relation dictionaries
        """
        relations = []
        
        for item_id in candidate_ids[:10]:
            # Content-similar items
            similar_items = graph.adjacency.get('content_sim', {}).get(item_id, set())
            for sim_item in list(similar_items)[:3]:
                sim_score = graph.get_edge_weight(item_id, sim_item, 'content_sim')
                relations.append({
                    'source_item': item_id,
                    'target_item': sim_item,
                    'relation_type': 'content_sim',
                    'strength': sim_score
                })
            
            # Co-interacted items
            co_items = graph.adjacency.get('co_interact', {}).get(item_id, set())
            for co_item in list(co_items)[:3]:
                co_users = len(
                    graph.adjacency.get('interact', {}).get(item_id, set()) &
                    graph.adjacency.get('interact', {}).get(co_item, set())
                )
                relations.append({
                    'source_item': item_id,
                    'target_item': co_item,
                    'relation_type': 'co_interact',
                    'co_interacting_users': co_users
                })
        
        return relations[:15]
    
    def _compute_statistics(self,
                             user_id: str,
                             candidate_ids: List[str],
                             graph: Any,
                             neighbors_1hop: Dict,
                             neighbors_2hop: Dict
                             ) -> Dict:
        """
        Compute graph statistics for context.
        
        Args:
            user_id: Center user ID
            candidate_ids: Candidate item IDs
            graph: HeterogeneousGraph instance
            neighbors_1hop: 1-hop neighbors
            neighbors_2hop: 2-hop neighbors
        
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        # Number of similar users
        similar_users = graph.adjacency.get('similar_pref', {}).get(user_id, set())
        stats['num_similar_users'] = len(similar_users)
        
        # Number of shared items with similar users
        user_items = graph.adjacency.get('interact', {}).get(user_id, set())
        shared_items = set()
        for su in similar_users:
            su_items = graph.adjacency.get('interact', {}).get(su, set())
            shared_items.update(user_items & su_items)
        stats['num_shared_items'] = len(shared_items)
        
        # Candidate popularity percentiles
        for item_id in candidate_ids[:5]:
            item_interactions = len(
                graph.adjacency.get('interact', {}).get(item_id, set())
            )
            # Simplified percentile
            stats[f'popularity_{item_id}'] = min(100, item_interactions * 5)
        
        # Graph density around user
        total_neighbors = sum(
            len(v) for v in neighbors_1hop.values()
        )
        stats['local_density'] = min(1.0, total_neighbors / 50)
        
        return stats
    
    def _identify_influential_paths(self,
                                      metapaths: List[Metapath],
                                      user_id: str,
                                      candidate_ids: List[str]
                                      ) -> List[Dict]:
        """
        Identify the most influential metapaths for decision making.
        
        Args:
            metapaths: Extracted metapaths
            user_id: Center user ID
            candidate_ids: Candidate item IDs
        
        Returns:
            List of influential path dictionaries
        """
        # Group paths by type and score
        paths_by_type = defaultdict(list)
        for path in metapaths:
            paths_by_type[path.path_type].append(path)
        
        influential = []
        
        # Select top paths from each type
        for path_type, paths in paths_by_type.items():
            # Get the highest scoring path
            if paths:
                top_path = max(paths, key=lambda p: p.score)
                influential.append({
                    'type': path_type,
                    'description': top_path.to_description(),
                    'score': top_path.score,
                    'nodes': top_path.nodes,
                    'metadata': top_path.metadata
                })
        
        # Sort by score
        influential.sort(key=lambda x: x['score'], reverse=True)
        
        return influential
    
    def _find_popular_in_neighborhood(self,
                                        graph: Any,
                                        neighbors_1hop: Dict,
                                        neighbors_2hop: Dict,
                                        exclude_ids: List[str]
                                        ) -> List[str]:
        """
        Find popular items in the user's graph neighborhood.
        
        Args:
            graph: HeterogeneousGraph instance
            neighbors_1hop: 1-hop neighbors
            neighbors_2hop: 2-hop neighbors
            exclude_ids: IDs to exclude (candidates already being considered)
        
        Returns:
            List of popular item IDs
        """
        item_counts = defaultdict(int)
        exclude_set = set(exclude_ids)
        
        # Count item appearances in neighborhood
        for edge_type, neighbors in {**neighbors_1hop, **neighbors_2hop}.items():
            for neighbor_id, weight in neighbors:
                if neighbor_id not in exclude_set:
                    # Check if it's an item
                    item_neighbors = graph.adjacency.get('interact', {}).get(neighbor_id, set())
                    for item_id in item_neighbors:
                        if item_id not in exclude_set:
                            item_counts[item_id] += 1
        
        # Sort by count
        popular = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [item_id for item_id, _ in popular[:10]]
    
    def _find_trending_items(self,
                               graph: Any,
                               neighbors_1hop: Dict
                               ) -> List[str]:
        """
        Find trending items in the neighborhood.
        Simplified: items with recent high interaction velocity.
        
        Args:
            graph: HeterogeneousGraph instance
            neighbors_1hop: 1-hop neighbors
        
        Returns:
            List of trending item IDs
        """
        # Simplified: items that many neighbors recently interacted with
        trending = []
        
        for edge_type, neighbors in neighbors_1hop.items():
            if 'interact' in edge_type:
                for item_id, weight in neighbors:
                    if weight > 0.8:  # High weight = recent/strong interaction
                        trending.append(item_id)
        
        return list(set(trending))[:5]
    
    def _rank_and_diversify(self, paths: List[Metapath]) -> List[Metapath]:
        """
        Rank paths by score while ensuring type diversity.
        
        Args:
            paths: List of Metapath objects
        
        Returns:
            Ranked and diversified list
        """
        if not paths:
            return []
        
        # Group by type
        paths_by_type = defaultdict(list)
        for path in paths:
            paths_by_type[path.path_type].append(path)
        
        # Sort each type group by score
        for path_type in paths_by_type:
            paths_by_type[path_type].sort(key=lambda p: p.score, reverse=True)
        
        # Interleave paths from different types
        ranked = []
        type_iterators = {
            ptype: iter(paths) 
            for ptype, paths in paths_by_type.items()
        }
        
        while any(type_iterators.values()):
            for ptype in ['user-item-user', 'user-user-item', 'item-item-analogy']:
                iterator = type_iterators.get(ptype)
                if iterator:
                    try:
                        ranked.append(next(iterator))
                    except StopIteration:
                        type_iterators[ptype] = None
        
        return ranked
    
    def _count_shared_interactions(self,
                                     user1: str,
                                     user2: str,
                                     graph: Any) -> int:
        """
        Count shared interactions between two users.
        
        Args:
            user1: First user ID
            user2: Second user ID
            graph: HeterogeneousGraph instance
        
        Returns:
            Number of shared interactions
        """
        items1 = graph.adjacency.get('interact', {}).get(user1, set())
        items2 = graph.adjacency.get('interact', {}).get(user2, set())
        return len(items1 & items2)
    
    def extract_paths_for_reflection(self,
                                      user_id: str,
                                      wrong_item_id: str,
                                      correct_item_id: str,
                                      graph: Any
                                      ) -> List[Dict]:
        """
        Extract paths specifically for reflection analysis.
        Focuses on paths that explain why one item was preferred over another.
        
        Args:
            user_id: User agent ID
            wrong_item_id: Incorrectly chosen item
            correct_item_id: Actually preferred item
            graph: HeterogeneousGraph instance
        
        Returns:
            List of reflection-relevant path dictionaries
        """
        paths = []
        
        # Paths for the wrong item
        wrong_paths = self._extract_user_user_item_paths(
            user_id, wrong_item_id, graph
        )
        
        # Paths for the correct item
        correct_paths = self._extract_user_user_item_paths(
            user_id, correct_item_id, graph
        )
        
        # Compare path scores to understand the mistake
        wrong_scores = [p.score for p in wrong_paths]
        correct_scores = [p.score for p in correct_paths]
        
        avg_wrong = np.mean(wrong_scores) if wrong_scores else 0.0
        avg_correct = np.mean(correct_scores) if correct_scores else 0.0
        
        paths.append({
            'type': 'reflection_analysis',
            'wrong_item_paths_count': len(wrong_paths),
            'correct_item_paths_count': len(correct_paths),
            'avg_wrong_score': avg_wrong,
            'avg_correct_score': avg_correct,
            'score_difference': avg_correct - avg_wrong,
            'analysis': (
                f"The correct item had {'stronger' if avg_correct > avg_wrong else 'weaker'} "
                f"collaborative signals (diff: {avg_correct - avg_wrong:.3f})"
            ),
            'suggested_focus': (
                'user-user-item' if avg_wrong > avg_correct else 'item-item-analogy'
            )
        })
        
        return paths
    
    def get_extraction_statistics(self) -> Dict:
        """Get metapath extraction statistics."""
        return dict(self.stats)
    
    def clear_cache(self):
        """Clear the path cache."""
        self._path_cache.clear()
        self._cache_counter = 0
        logger.debug("Metapath cache cleared")