"""
Graph RAG Retriever Module for H-GRAGrecsys

This module implements the retrieval component of the Graph RAG system,
extracting relevant subgraphs, metapaths, and context from the heterogeneous
graph for recommendation and explanation generation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from collections import defaultdict, deque
import heapq
from tqdm import tqdm
import sys
import os
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.graph.dynamic_graph import DynamicGraph
from models.graph.relation_types import RelationType, EdgeWeightFunctions
from models.graph_rag.metapath_extractor import MetapathExtractor
from models.graph_rag.context_constructor import ContextConstructor
from models.graph_rag.ppr_sampler import PPRSampler
from models.agent.user_agent import UserAgent
from models.agent.item_agent import ItemAgent
from utils.logger import Logger
from utils.config_loader import ConfigLoader


class GraphRAGRetriever:
    """
    Graph RAG Retriever for extracting relevant context from heterogeneous graph.
    
    This class handles:
    - Retrieving context subgraphs for query nodes
    - Extracting metapath instances
    - Ranking nodes based on relevance
    - Building context for LLM reasoning
    - Supporting different retrieval strategies
    """
    
    def __init__(self, graph: DynamicGraph, config: Dict[str, Any]):
        """
        Initialize the Graph RAG retriever.
        
        Args:
            graph: DynamicGraph instance
            config: Configuration dictionary
        """
        self.graph = graph
        self.config = config
        self.logger = Logger.get_instance(log_dir='logs', name='graph_rag_retriever')
        
        # Extract configuration
        graph_rag_config = config.get('model', {}).get('graph_rag', {})
        self.max_hop_count = graph_rag_config.get('max_hop_count', 3)
        self.max_neighbors_per_hop = graph_rag_config.get('max_neighbors_per_hop', 50)
        self.min_edge_weight = graph_rag_config.get('min_edge_weight', 0.05)
        self.context_size = graph_rag_config.get('context_size', 10)
        self.retrieval_strategy = graph_rag_config.get('retrieval_strategy', 'hybrid')
        self.use_ppr = graph_rag_config.get('use_ppr', True)
        self.ppr_restart_prob = graph_rag_config.get('ppr_restart_prob', 0.15)
        
        # Initialize components
        self.metapath_extractor = MetapathExtractor(config)
        self.context_constructor = ContextConstructor(config)
        self.ppr_sampler = PPRSampler(graph, config)
        
        # Cache for retrieved contexts
        self.context_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_size = graph_rag_config.get('retrieval_cache_size', 100)
        
        # Statistics
        self.retrieval_stats = {
            'total_retrievals': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_retrieval_time': 0.0,
            'avg_context_nodes': 0,
            'avg_context_edges': 0,
            'retrieval_times': []
        }
        
        self.logger.log_info(f"Initialized GraphRAGRetriever with strategy={self.retrieval_strategy}")
    
    def retrieve_context(self, query_node: str, 
                        hop_count: Optional[int] = None,
                        max_nodes: Optional[int] = None,
                        relation_types: Optional[List[str]] = None,
                        use_cache: bool = True) -> Dict[str, Any]:
        """
        Retrieve comprehensive context for a query node.
        
        Args:
            query_node: ID of the query node
            hop_count: Number of hops to expand
            max_nodes: Maximum nodes in context
            relation_types: Types of relations to include
            use_cache: Whether to use cached context
            
        Returns:
            Dict[str, Any]: Context dictionary containing:
                - subgraph: Retrieved subgraph
                - metapaths: Extracted metapath instances
                - ppr_scores: PPR scores for ranking
                - nodes: List of nodes in context
                - edges: List of edges in context
                - statistics: Context statistics
        """
        self.logger.log_info(f"Retrieving context for node: {query_node}")
        
        # Check cache
        cache_key = f"{query_node}_{hop_count}_{max_nodes}_{str(relation_types)}"
        if use_cache and cache_key in self.context_cache:
            self.logger.log_info(f"Cache hit for node {query_node}")
            self.retrieval_stats['cache_hits'] += 1
            return self.context_cache[cache_key]
        
        self.retrieval_stats['cache_misses'] += 1
        start_time = torch.tensor(0.0).item()
        
        # Set parameters
        hop_count = hop_count or self.max_hop_count
        max_nodes = max_nodes or self.max_neighbors_per_hop * hop_count
        
        # Step 1: Get subgraph
        subgraph = self.get_subgraph(query_node, hop_count, max_nodes, relation_types)
        
        # Step 2: Extract metapaths
        metapaths = self.metapath_extractor.extract_all_paths(
            query_node, 
            path_types=self._get_relevant_path_types(query_node),
            max_len=hop_count
        )
        
        # Step 3: Compute PPR scores for ranking
        ppr_scores = {}
        if self.use_ppr:
            ppr_scores = self.graph.compute_ppr_scores(
                query_node, 
                restart_prob=self.ppr_restart_prob
            )
        
        # Step 4: Rank nodes in context
        ranked_nodes = self.rank_nodes(
            query_node, 
            list(subgraph.nodes.keys()),
            context_type='retrieval'
        )
        
        # Step 5: Build context
        context = {
            'query_node': query_node,
            'subgraph': subgraph,
            'metapaths': metapaths,
            'ppr_scores': ppr_scores,
            'ranked_nodes': ranked_nodes[:self.context_size],
            'nodes': list(subgraph.nodes.keys()),
            'edges': list(subgraph.edges.keys()),
            'statistics': {
                'num_nodes': len(subgraph.nodes),
                'num_edges': len(subgraph.edges),
                'num_metapaths': len(metapaths),
                'hop_count': hop_count,
                'max_nodes': max_nodes,
                'retrieval_time': torch.tensor(0.0).item() - start_time
            }
        }
        
        # Update statistics
        self.retrieval_stats['total_retrievals'] += 1
        self.retrieval_stats['avg_context_nodes'] = (
            (self.retrieval_stats['avg_context_nodes'] * (self.retrieval_stats['total_retrievals'] - 1) + 
             len(subgraph.nodes)) / self.retrieval_stats['total_retrievals']
        )
        self.retrieval_stats['avg_context_edges'] = (
            (self.retrieval_stats['avg_context_edges'] * (self.retrieval_stats['total_retrievals'] - 1) + 
             len(subgraph.edges)) / self.retrieval_stats['total_retrievals']
        )
        self.retrieval_stats['retrieval_times'].append(context['statistics']['retrieval_time'])
        self.retrieval_stats['avg_retrieval_time'] = np.mean(self.retrieval_stats['retrieval_times'])
        
        # Cache context
        if use_cache and len(self.context_cache) < self.cache_size:
            self.context_cache[cache_key] = context
        
        self.logger.log_info(f"Retrieved context with {len(subgraph.nodes)} nodes, {len(subgraph.edges)} edges")
        return context
    
    def get_subgraph(self, start_node: str, 
                    hop_count: int = 2,
                    max_nodes: int = 100,
                    relation_types: Optional[List[str]] = None) -> DynamicGraph:
        """
        Extract a subgraph starting from a node.
        
        Args:
            start_node: Starting node ID
            hop_count: Number of hops to expand
            max_nodes: Maximum nodes in subgraph
            relation_types: Types of relations to include
            
        Returns:
            DynamicGraph: Extracted subgraph
        """
        if start_node not in self.graph.nodes:
            self.logger.log_warning(f"Node {start_node} not found in graph")
            return DynamicGraph(self.config)
        
        # Default relation types
        if relation_types is None:
            relation_types = RelationType.get_all_types()
        
        # BFS expansion
        visited = set([start_node])
        frontier = set([start_node])
        nodes_to_include = [start_node]
        
        for hop in range(hop_count):
            new_frontier = set()
            
            for node_id in frontier:
                # Get neighbors for each relation type
                for rel_type in relation_types:
                    neighbors = self.graph.get_neighbors(node_id, rel_type, max_neighbors=10)
                    
                    for neighbor_id, weight in neighbors:
                        if neighbor_id not in visited and weight > self.min_edge_weight:
                            new_frontier.add(neighbor_id)
                            nodes_to_include.append(neighbor_id)
                            
                            # Check limit
                            if len(nodes_to_include) >= max_nodes:
                                break
                    
                    if len(nodes_to_include) >= max_nodes:
                        break
                
                if len(nodes_to_include) >= max_nodes:
                    break
            
            visited.update(new_frontier)
            frontier = new_frontier
            
            if len(nodes_to_include) >= max_nodes:
                break
        
        # Extract subgraph
        subgraph = self.graph.get_subgraph(nodes_to_include)
        
        self.logger.log_info(f"Extracted subgraph with {len(subgraph.nodes)} nodes, {len(subgraph.edges)} edges")
        return subgraph
    
    def rank_nodes(self, query_node: str, 
                  candidate_nodes: List[str],
                  context_type: str = 'retrieval') -> List[Tuple[str, float]]:
        """
        Rank candidate nodes by relevance to the query node.
        
        Args:
            query_node: Query node ID
            candidate_nodes: List of candidate node IDs
            context_type: Type of context for ranking
            
        Returns:
            List[Tuple[str, float]]: Ranked list of (node_id, score) pairs
        """
        if query_node not in self.graph.nodes:
            self.logger.log_warning(f"Query node {query_node} not found")
            return [(node, 0.0) for node in candidate_nodes]
        
        scores = {}
        query_node_type = self.graph.node_id_to_type.get(query_node, 'unknown')
        
        for candidate in candidate_nodes:
            if candidate not in self.graph.nodes:
                scores[candidate] = 0.0
                continue
            
            candidate_node_type = self.graph.node_id_to_type.get(candidate, 'unknown')
            
            # Different ranking strategies based on node types
            if context_type == 'retrieval':
                score = self._compute_retrieval_score(query_node, candidate)
            elif context_type == 'recommendation':
                score = self._compute_recommendation_score(query_node, candidate)
            else:
                # Default: combined score
                score = self._compute_combined_score(query_node, candidate)
            
            # Apply type-specific boost
            if query_node_type == 'user' and candidate_node_type == 'item':
                score *= 1.2  # Boost item scores for user queries
            elif query_node_type == 'item' and candidate_node_type == 'user':
                score *= 1.1  # Boost user scores for item queries
            
            scores[candidate] = score
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def _compute_retrieval_score(self, query_node: str, candidate: str) -> float:
        """
        Compute retrieval score using multiple factors.
        
        Args:
            query_node: Query node ID
            candidate: Candidate node ID
            
        Returns:
            float: Retrieval score
        """
        score = 0.0
        
        # Factor 1: Direct connection weight
        direct_weight = 0.0
        for rel_type in RelationType.get_all_types():
            weight = self.graph.get_edge_weight(query_node, candidate, rel_type)
            if weight is not None:
                direct_weight = max(direct_weight, weight)
        
        # Factor 2: PPR score
        ppr_score = 0.0
        if self.use_ppr:
            ppr_scores = self.graph.compute_ppr_scores(query_node)
            ppr_score = ppr_scores.get(candidate, 0.0)
        
        # Factor 3: Common neighbor overlap
        common_neighbors = self._compute_common_neighbor_score(query_node, candidate)
        
        # Factor 4: Node importance (degree centrality)
        degree_score = self._compute_node_importance(candidate)
        
        # Combine factors with weights
        score = (
            0.3 * direct_weight +
            0.4 * ppr_score +
            0.2 * common_neighbors +
            0.1 * degree_score
        )
        
        return min(1.0, score)
    
    def _compute_recommendation_score(self, user_node: str, item_node: str) -> float:
        """
        Compute recommendation score for user-item pair.
        
        Args:
            user_node: User node ID
            item_node: Item node ID
            
        Returns:
            float: Recommendation score
        """
        # Check direct interaction
        interaction_weight = self.graph.get_edge_weight(
            user_node, item_node, RelationType.INTERACT
        )
        
        if interaction_weight is not None:
            return interaction_weight
        
        # Collaborative filtering via similar users
        similar_users = self.graph.get_neighbors(user_node, RelationType.SIMILAR_PREF)
        collaborative_score = 0.0
        
        for similar_user, sim_weight in similar_users[:10]:
            if sim_weight > 0.3:
                # Check if similar user interacted with item
                interaction = self.graph.get_edge_weight(
                    similar_user, item_node, RelationType.INTERACT
                )
                if interaction is not None:
                    collaborative_score += interaction * sim_weight
        
        # Content-based via similar items
        similar_items = self.graph.get_neighbors(item_node, RelationType.CONTENT_SIM)
        content_score = 0.0
        
        for similar_item, sim_weight in similar_items[:10]:
            if sim_weight > 0.3:
                # Check if user interacted with similar item
                interaction = self.graph.get_edge_weight(
                    user_node, similar_item, RelationType.INTERACT
                )
                if interaction is not None:
                    content_score += interaction * sim_weight
        
        # Combine scores
        if collaborative_score > 0 or content_score > 0:
            total_score = 0.5 * collaborative_score + 0.5 * content_score
            return min(1.0, total_score)
        
        return 0.0
    
    def _compute_combined_score(self, query_node: str, candidate: str) -> float:
        """
        Compute combined score for general use.
        
        Args:
            query_node: Query node ID
            candidate: Candidate node ID
            
        Returns:
            float: Combined score
        """
        retrieval_score = self._compute_retrieval_score(query_node, candidate)
        
        # For user-item pairs, add recommendation score
        query_type = self.graph.node_id_to_type.get(query_node, 'unknown')
        candidate_type = self.graph.node_id_to_type.get(candidate, 'unknown')
        
        if query_type == 'user' and candidate_type == 'item':
            rec_score = self._compute_recommendation_score(query_node, candidate)
            return 0.6 * retrieval_score + 0.4 * rec_score
        elif query_type == 'item' and candidate_type == 'user':
            rec_score = self._compute_recommendation_score(candidate, query_node)
            return 0.6 * retrieval_score + 0.4 * rec_score
        
        return retrieval_score
    
    def _compute_common_neighbor_score(self, node_a: str, node_b: str) -> float:
        """
        Compute score based on common neighbors.
        
        Args:
            node_a: First node ID
            node_b: Second node ID
            
        Returns:
            float: Common neighbor score
        """
        neighbors_a = set()
        neighbors_b = set()
        
        # Get all neighbors
        for rel_type in RelationType.get_all_types():
            neighbors_a.update([n for n, _ in self.graph.get_neighbors(node_a, rel_type)])
            neighbors_b.update([n for n, _ in self.graph.get_neighbors(node_b, rel_type)])
        
        if not neighbors_a or not neighbors_b:
            return 0.0
        
        common = len(neighbors_a & neighbors_b)
        union = len(neighbors_a | neighbors_b)
        
        return common / union if union > 0 else 0.0
    
    def _compute_node_importance(self, node_id: str) -> float:
        """
        Compute node importance score.
        
        Args:
            node_id: Node ID
            
        Returns:
            float: Importance score
        """
        if node_id not in self.graph.nodes:
            return 0.0
        
        # Degree centrality
        degree = self.graph.get_connection_count(node_id)
        max_degree = max([self.graph.get_connection_count(n) for n in self.graph.nodes] or [1])
        
        degree_score = degree / max_degree if max_degree > 0 else 0.0
        
        # PPR score (use graph's PPR)
        ppr_scores = self.graph.compute_ppr_scores(node_id)
        ppr_score = ppr_scores.get(node_id, 0.0)
        
        return 0.5 * degree_score + 0.5 * ppr_score
    
    def _get_relevant_path_types(self, query_node: str) -> List[str]:
        """
        Get relevant path types for metapath extraction.
        
        Args:
            query_node: Query node ID
            
        Returns:
            List[str]: List of path type names
        """
        node_type = self.graph.node_id_to_type.get(query_node, 'unknown')
        
        if node_type == 'user':
            return [
                'user-item-user',
                'user-item-item',
                'user-user-item',
                'user-user-user'
            ]
        elif node_type == 'item':
            return [
                'item-user-item',
                'item-item-user',
                'item-user-user',
                'item-item-item'
            ]
        else:
            return ['all']
    
    def get_relevant_neighbors(self, node_id: str, 
                              relation_types: Optional[List[str]] = None,
                              top_k: int = 10) -> List[Tuple[str, str, float]]:
        """
        Get most relevant neighbors for a node.
        
        Args:
            node_id: Node ID
            relation_types: Types of relations to include
            top_k: Number of neighbors to return
            
        Returns:
            List[Tuple[str, str, float]]: List of (neighbor_id, relation_type, weight)
        """
        if node_id not in self.graph.nodes:
            self.logger.log_warning(f"Node {node_id} not found")
            return []
        
        if relation_types is None:
            relation_types = RelationType.get_all_types()
        
        # Collect all neighbors with weights
        all_neighbors = []
        for rel_type in relation_types:
            neighbors = self.graph.get_neighbors(node_id, rel_type)
            for neighbor, weight in neighbors:
                if weight > self.min_edge_weight:
                    all_neighbors.append((neighbor, rel_type, weight))
        
        # Sort by weight
        all_neighbors.sort(key=lambda x: x[2], reverse=True)
        
        # Apply PPR re-ranking
        if self.use_ppr and len(all_neighbors) > 0:
            ppr_scores = self.graph.compute_ppr_scores(node_id)
            
            # Combine original weight with PPR
            scored_neighbors = []
            for neighbor, rel_type, weight in all_neighbors:
                ppr_score = ppr_scores.get(neighbor, 0.0)
                combined_score = 0.6 * weight + 0.4 * ppr_score
                scored_neighbors.append((neighbor, rel_type, combined_score))
            
            scored_neighbors.sort(key=lambda x: x[2], reverse=True)
            all_neighbors = [(n, rt, s) for n, rt, s in scored_neighbors]
        
        return all_neighbors[:top_k]
    
    def build_context_subgraph(self, start_node: str, 
                              k: int = 3,
                              include_metapaths: bool = True) -> Dict[str, Any]:
        """
        Build a context subgraph for LLM reasoning.
        
        Args:
            start_node: Starting node ID
            k: Number of hops
            include_metapaths: Whether to include metapaths
            
        Returns:
            Dict[str, Any]: Context subgraph with metapaths
        """
        # Get subgraph
        subgraph = self.get_subgraph(start_node, hop_count=k)
        
        # Extract metapaths
        metapaths = []
        if include_metapaths:
            metapaths = self.metapath_extractor.extract_all_paths(
                start_node,
                path_types=self._get_relevant_path_types(start_node),
                max_len=k
            )
        
        # Rank nodes
        ranked_nodes = self.rank_nodes(
            start_node,
            list(subgraph.nodes.keys()),
            context_type='retrieval'
        )
        
        # Build context
        context = {
            'start_node': start_node,
            'subgraph': subgraph,
            'metapaths': metapaths,
            'ranked_nodes': ranked_nodes[:self.context_size],
            'node_types': {
                nid: self.graph.node_id_to_type.get(nid, 'unknown')
                for nid in subgraph.nodes.keys()
            },
            'statistics': {
                'num_nodes': len(subgraph.nodes),
                'num_edges': len(subgraph.edges),
                'num_metapaths': len(metapaths),
                'max_ranked_score': ranked_nodes[0][1] if ranked_nodes else 0.0
            }
        }
        
        self.logger.log_info(f"Built context subgraph with {len(subgraph.nodes)} nodes")
        return context
    
    def extract_metapath_instances(self, start_node: str, 
                                  metapath_type: str,
                                  max_instances: int = 10) -> List[Dict[str, Any]]:
        """
        Extract metapath instances of a specific type.
        
        Args:
            start_node: Starting node ID
            metapath_type: Type of metapath
            max_instances: Maximum instances to return
            
        Returns:
            List[Dict[str, Any]]: Metapath instances with metadata
        """
        return self.metapath_extractor.extract_metapath_instances(
            start_node, metapath_type, max_instances
        )
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """
        Get retrieval statistics.
        
        Returns:
            Dict[str, Any]: Retrieval statistics
        """
        stats = self.retrieval_stats.copy()
        stats['cache_size'] = len(self.context_cache)
        stats['cache_hit_rate'] = (
            stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
            if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0.0
        )
        stats['recent_retrieval_times'] = stats['retrieval_times'][-10:]
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the retrieval cache."""
        self.context_cache.clear()
        self.logger.log_info("Cleared retrieval cache")
    
    def warm_start(self, query_node: str, 
                  interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Warm start retrieval for a new node.
        
        Args:
            query_node: New node ID
            interactions: Initial interactions
            
        Returns:
            Dict[str, Any]: Context for the new node
        """
        self.logger.log_info(f"Warm starting retrieval for node {query_node}")
        
        # Add initial interactions to graph if not present
        if query_node not in self.graph.nodes:
            # Create node
            self.graph.add_node(query_node, 'user')
            
            # Add interaction edges
            for interaction in interactions:
                item_id = interaction.get('item_id')
                rating = interaction.get('rating', 1.0)
                
                if item_id in self.graph.nodes:
                    self.graph.add_edge(
                        query_node, item_id, RelationType.INTERACT, weight=rating
                    )
        
        # Retrieve context
        context = self.retrieve_context(query_node, hop_count=2)
        
        return context
    
    def compute_relevance_score(self, context: Dict[str, Any], 
                              candidate: str) -> float:
        """
        Compute relevance score between context and candidate.
        
        Args:
            context: Context dictionary
            candidate: Candidate node ID
            
        Returns:
            float: Relevance score
        """
        query_node = context['query_node']
        return self._compute_combined_score(query_node, candidate)
    
    def to_networkx_retrieval(self, context: Dict[str, Any]) -> nx.Graph:
        """
        Convert retrieval context to NetworkX graph.
        
        Args:
            context: Context dictionary
            
        Returns:
            nx.Graph: NetworkX graph with retrieval metadata
        """
        subgraph = context['subgraph']
        G = nx.Graph()
        
        # Add nodes with metadata
        for node_id, node in subgraph.nodes.items():
            G.add_node(
                node_id,
                node_type=node.node_type,
                rank=context.get('ranked_nodes', {}).get(node_id, 0.0)
            )
        
        # Add edges with metadata
        for (src, dst, rel_type), weight in subgraph.edges.items():
            G.add_edge(
                src, dst,
                relation_type=rel_type,
                weight=weight
            )
        
        return G
    
    def __str__(self) -> str:
        """String representation of the retriever."""
        return (f"GraphRAGRetriever(strategy={self.retrieval_strategy}, "
                f"hops={self.max_hop_count}, "
                f"retrievals={self.retrieval_stats['total_retrievals']})")


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
    graph.add_node('item_1', 'item', features={'title': 'Product A'})
    graph.add_node('item_2', 'item', features={'title': 'Product B'})
    graph.add_node('item_3', 'item', features={'title': 'Product C'})
    
    graph.add_edge('user_1', 'item_1', RelationType.INTERACT, weight=0.9)
    graph.add_edge('user_1', 'item_2', RelationType.INTERACT, weight=0.7)
    graph.add_edge('user_2', 'item_2', RelationType.INTERACT, weight=0.8)
    graph.add_edge('user_2', 'item_3', RelationType.INTERACT, weight=0.6)
    graph.add_edge('user_1', 'user_2', RelationType.SIMILAR_PREF, weight=0.8)
    graph.add_edge('item_1', 'item_2', RelationType.CONTENT_SIM, weight=0.6)
    
    # Create retriever
    retriever = GraphRAGRetriever(graph, config)
    
    # Retrieve context for user
    context = retriever.retrieve_context('user_1', hop_count=2)
    print(f"Retrieved context: {len(context['nodes'])} nodes, {len(context['edges'])} edges")
    
    # Get relevant neighbors
    neighbors = retriever.get_relevant_neighbors('user_1', top_k=5)
    print(f"Relevant neighbors: {neighbors}")
    
    # Get statistics
    stats = retriever.get_context_statistics()
    print(f"Retriever statistics: {stats}")