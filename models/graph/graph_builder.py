"""
Graph Builder Module for H-GRAGrecsys

This module constructs the heterogeneous graph from various data sources,
including user-item interactions, user similarities, item similarities,
and co-interaction patterns. It handles both initial graph construction
and incremental updates.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import networkx as nx
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.graph.heterogeneous_graph import HeterogeneousGraph, GraphNode
from models.graph.relation_types import RelationType, EdgeWeightFunctions
from models.agent.user_agent import UserAgent
from models.agent.item_agent import ItemAgent
from utils.logger import Logger
from utils.config_loader import ConfigLoader
from data.dataset import BaseDataset


class GraphBuilder:
    """
    Builds and maintains the heterogeneous graph from various data sources.
    
    This class handles:
    - Building initial graph from dataset
    - Adding user-item interaction edges
    - Computing and adding similarity edges
    - Adding co-interaction edges
    - Initializing new users and items
    - Maintaining graph statistics and quality metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the graph builder with configuration.
        
        Args:
            config: Configuration dictionary containing graph building parameters
        """
        self.config = config
        self.logger = Logger.get_instance(log_dir='logs', name='graph_builder')
        
        # Extract graph configuration
        graph_config = config.get('model', {}).get('graph', {})
        self.user_similarity_threshold = graph_config.get('user_similarity_threshold', 0.7)
        self.item_similarity_threshold = graph_config.get('item_similarity_threshold', 0.6)
        self.co_interaction_threshold = graph_config.get('co_interaction_threshold', 3)
        self.max_neighbors_per_node = graph_config.get('max_neighbors_per_node', 50)
        self.min_edge_weight = graph_config.get('min_edge_weight', 0.05)
        self.embedding_dim = config.get('model', {}).get('gnn', {}).get('hidden_dim', 256)
        
        # Statistics tracking
        self.building_stats = {
            'total_users': 0,
            'total_items': 0,
            'total_interactions': 0,
            'similarity_edges_added': 0,
            'co_interaction_edges_added': 0,
            'pruned_edges': 0,
            'build_time': 0.0
        }
        
        self.logger.log_info(f"Initialized GraphBuilder with similarity_threshold={self.user_similarity_threshold}")
    
    def build_graph(self, agents: Dict[str, Any], interactions: List[Dict[str, Any]]) -> HeterogeneousGraph:
        """
        Build the complete heterogeneous graph from agents and interactions.
        
        Args:
            agents: Dictionary mapping agent_id to agent object (UserAgent or ItemAgent)
            interactions: List of interaction dictionaries with keys: 'user_id', 'item_id', 'rating', 'timestamp'
            
        Returns:
            HeterogeneousGraph: The constructed graph
            
        Raises:
            ValueError: If agents or interactions are empty
        """
        if not agents:
            raise ValueError("Cannot build graph: agents dictionary is empty")
        if not interactions:
            raise ValueError("Cannot build graph: interactions list is empty")
        
        self.logger.log_info(f"Building graph with {len(agents)} agents and {len(interactions)} interactions")
        
        # Create new graph
        graph = HeterogeneousGraph(self.config)
        
        # Add all agents as nodes
        self._add_agent_nodes(graph, agents)
        
        # Add user-item interaction edges
        self._add_interaction_edges(graph, interactions, agents)
        
        # Add user similarity edges
        self._add_user_similarity_edges(graph, agents)
        
        # Add item similarity edges
        self._add_item_similarity_edges(graph, agents)
        
        # Add co-interaction edges between users
        self._add_co_interaction_edges(graph, interactions, agents)
        
        # Update graph statistics
        stats = graph.get_graph_statistics()
        self.building_stats.update({
            'total_users': len([a for a in agents.values() if a.agent_type == 'user']),
            'total_items': len([a for a in agents.values() if a.agent_type == 'item']),
            'total_interactions': len(interactions)
        })
        
        self.logger.log_info(f"Graph built successfully: {stats}")
        return graph
    
    def _add_agent_nodes(self, graph: HeterogeneousGraph, agents: Dict[str, Any]) -> None:
        """
        Add all agents as nodes to the graph.
        
        Args:
            graph: HeterogeneousGraph instance
            agents: Dictionary of agent objects
        """
        for agent_id, agent in tqdm(agents.items(), desc="Adding agent nodes"):
            # Determine node type
            if isinstance(agent, UserAgent):
                node_type = 'user'
                features = {
                    'preference_vector': agent.get_preference_memory(),
                    'interaction_count': len(agent.get_recommendation_history()),
                    'embedding': agent.get_embedding()
                }
            elif isinstance(agent, ItemAgent):
                node_type = 'item'
                features = {
                    'metadata': agent.get_item_metadata(),
                    'popularity_score': agent.get_popularity_score(),
                    'embedding': agent.get_content_embedding()
                }
            else:
                self.logger.log_warning(f"Unknown agent type for {agent_id}, skipping")
                continue
            
            # Add node to graph
            graph.add_node(
                node_id=agent_id,
                node_type=node_type,
                features=features,
                embedding=agent.get_embedding() if hasattr(agent, 'get_embedding') else None,
                metadata={'agent_type': type(agent).__name__}
            )
    
    def _add_interaction_edges(self, graph: HeterogeneousGraph, 
                               interactions: List[Dict[str, Any]], 
                               agents: Dict[str, Any]) -> None:
        """
        Add user-item interaction edges to the graph.
        
        Args:
            graph: HeterogeneousGraph instance
            interactions: List of interaction dictionaries
            agents: Dictionary of agent objects
        """
        interaction_count = 0
        for interaction in tqdm(interactions, desc="Adding interaction edges"):
            user_id = interaction.get('user_id')
            item_id = interaction.get('item_id')
            rating = interaction.get('rating', 1.0)
            
            if user_id not in agents or item_id not in agents:
                continue
            
            # Calculate edge weight based on rating and interaction strength
            weight = self._calculate_interaction_weight(rating, interaction)
            
            # Add edge
            graph.add_edge(
                source=user_id,
                target=item_id,
                relation_type=RelationType.INTERACT,
                weight=weight,
                metadata={
                    'rating': rating,
                    'timestamp': interaction.get('timestamp', 0),
                    'interaction_type': interaction.get('type', 'explicit')
                }
            )
            interaction_count += 1
        
        self.logger.log_info(f"Added {interaction_count} interaction edges")
        self.building_stats['total_interactions'] = interaction_count
    
    def _calculate_interaction_weight(self, rating: float, interaction: Dict[str, Any]) -> float:
        """
        Calculate edge weight for an interaction.
        
        Args:
            rating: Rating value (e.g., 1-5 or binary)
            interaction: Full interaction dictionary
            
        Returns:
            float: Normalized weight between 0 and 1
        """
        # Normalize rating to [0, 1]
        if isinstance(rating, (int, float)):
            # Assuming rating is 1-5 scale
            normalized_rating = max(0.0, min(1.0, (rating - 1.0) / 4.0))
        else:
            normalized_rating = 1.0
        
        # Consider recency if timestamp is available
        timestamp = interaction.get('timestamp', 0)
        if timestamp > 0:
            # Decay weight for older interactions (optional)
            # Here we keep it simple
            pass
        
        return max(0.1, normalized_rating)  # Minimum weight 0.1
    
    def _add_user_similarity_edges(self, graph: HeterogeneousGraph, 
                                   agents: Dict[str, Any]) -> None:
        """
        Add user similarity edges based on preference similarity.
        
        Args:
            graph: HeterogeneousGraph instance
            agents: Dictionary of agent objects
        """
        # Extract user agents
        user_agents = {aid: agent for aid, agent in agents.items() 
                      if isinstance(agent, UserAgent)}
        
        if len(user_agents) < 2:
            self.logger.log_info("Not enough users to compute similarity edges")
            return
        
        # Get user embeddings
        user_ids = list(user_agents.keys())
        user_embeddings = []
        valid_users = []
        
        for uid in user_ids:
            embedding = user_agents[uid].get_embedding()
            if embedding is not None:
                user_embeddings.append(embedding.numpy() if torch.is_tensor(embedding) else embedding)
                valid_users.append(uid)
        
        if len(valid_users) < 2:
            self.logger.log_warning("Not enough users with embeddings")
            return
        
        # Compute similarity matrix
        embeddings_array = np.array(user_embeddings)
        similarity_matrix = cosine_similarity(embeddings_array)
        
        # Add similarity edges
        edges_added = 0
        for i in tqdm(range(len(valid_users)), desc="Adding user similarity edges"):
            for j in range(i + 1, len(valid_users)):
                similarity = similarity_matrix[i][j]
                
                if similarity >= self.user_similarity_threshold:
                    user_a = valid_users[i]
                    user_b = valid_users[j]
                    
                    graph.add_edge(
                        source=user_a,
                        target=user_b,
                        relation_type=RelationType.SIMILAR_PREF,
                        weight=similarity,
                        metadata={
                            'similarity_computation': 'cosine',
                            'common_items': self._get_common_items(user_a, user_b, agents)
                        }
                    )
                    edges_added += 1
        
        self.logger.log_info(f"Added {edges_added} user similarity edges")
        self.building_stats['similarity_edges_added'] += edges_added
    
    def _add_item_similarity_edges(self, graph: HeterogeneousGraph, 
                                   agents: Dict[str, Any]) -> None:
        """
        Add item similarity edges based on content similarity.
        
        Args:
            graph: HeterogeneousGraph instance
            agents: Dictionary of agent objects
        """
        # Extract item agents
        item_agents = {aid: agent for aid, agent in agents.items() 
                      if isinstance(agent, ItemAgent)}
        
        if len(item_agents) < 2:
            self.logger.log_info("Not enough items to compute similarity edges")
            return
        
        # Get item embeddings
        item_ids = list(item_agents.keys())
        item_embeddings = []
        valid_items = []
        
        for iid in item_ids:
            embedding = item_agents[iid].get_content_embedding()
            if embedding is not None:
                item_embeddings.append(embedding.numpy() if torch.is_tensor(embedding) else embedding)
                valid_items.append(iid)
        
        if len(valid_items) < 2:
            self.logger.log_warning("Not enough items with embeddings")
            return
        
        # Compute similarity matrix
        embeddings_array = np.array(item_embeddings)
        similarity_matrix = cosine_similarity(embeddings_array)
        
        # Add similarity edges
        edges_added = 0
        for i in tqdm(range(len(valid_items)), desc="Adding item similarity edges"):
            for j in range(i + 1, len(valid_items)):
                similarity = similarity_matrix[i][j]
                
                if similarity >= self.item_similarity_threshold:
                    item_a = valid_items[i]
                    item_b = valid_items[j]
                    
                    graph.add_edge(
                        source=item_a,
                        target=item_b,
                        relation_type=RelationType.CONTENT_SIM,
                        weight=similarity,
                        metadata={
                            'similarity_computation': 'cosine',
                            'shared_categories': self._get_shared_categories(item_a, item_b, agents)
                        }
                    )
                    edges_added += 1
        
        self.logger.log_info(f"Added {edges_added} item similarity edges")
        self.building_stats['similarity_edges_added'] += edges_added
    
    def _add_co_interaction_edges(self, graph: HeterogeneousGraph, 
                                  interactions: List[Dict[str, Any]], 
                                  agents: Dict[str, Any]) -> None:
        """
        Add co-interaction edges between users who interacted with same items.
        
        Args:
            graph: HeterogeneousGraph instance
            interactions: List of interaction dictionaries
            agents: Dictionary of agent objects
        """
        # Build user-item interaction map
        user_items = defaultdict(set)
        item_users = defaultdict(set)
        
        for interaction in interactions:
            user_id = interaction.get('user_id')
            item_id = interaction.get('item_id')
            if user_id in agents and item_id in agents:
                user_items[user_id].add(item_id)
                item_users[item_id].add(user_id)
        
        # Compute co-interaction scores
        edges_added = 0
        user_ids = list(user_items.keys())
        
        for i in tqdm(range(len(user_ids)), desc="Adding co-interaction edges"):
            for j in range(i + 1, len(user_ids)):
                user_a = user_ids[i]
                user_b = user_ids[j]
                
                # Compute Jaccard similarity of interacted items
                items_a = user_items[user_a]
                items_b = user_items[user_b]
                intersection = len(items_a & items_b)
                
                if intersection >= self.co_interaction_threshold:
                    jaccard_sim = len(intersection) / len(items_a | items_b) if items_a | items_b else 0
                    
                    if jaccard_sim > self.min_edge_weight:
                        graph.add_edge(
                            source=user_a,
                            target=user_b,
                            relation_type=RelationType.CO_INTER,
                            weight=jaccard_sim,
                            metadata={
                                'common_items': list(items_a & items_b),
                                'intersection_count': intersection
                            }
                        )
                        edges_added += 1
        
        self.logger.log_info(f"Added {edges_added} co-interaction edges")
        self.building_stats['co_interaction_edges_added'] = edges_added
    
    def _get_common_items(self, user_a: str, user_b: str, agents: Dict[str, Any]) -> int:
        """
        Get number of common items between two users.
        
        Args:
            user_a: First user ID
            user_b: Second user ID
            agents: Dictionary of agent objects
            
        Returns:
            int: Number of common items
        """
        if user_a not in agents or user_b not in agents:
            return 0
        
        user_a_items = set(agents[user_a].get_recommendation_history())
        user_b_items = set(agents[user_b].get_recommendation_history())
        
        return len(user_a_items & user_b_items)
    
    def _get_shared_categories(self, item_a: str, item_b: str, agents: Dict[str, Any]) -> List[str]:
        """
        Get shared categories between two items.
        
        Args:
            item_a: First item ID
            item_b: Second item ID
            agents: Dictionary of agent objects
            
        Returns:
            List[str]: List of shared category names
        """
        if item_a not in agents or item_b not in agents:
            return []
        
        metadata_a = agents[item_a].get_item_metadata()
        metadata_b = agents[item_b].get_item_metadata()
        
        categories_a = set(metadata_a.get('categories', []))
        categories_b = set(metadata_b.get('categories', []))
        
        return list(categories_a & categories_b)
    
    def build_initial_graph(self, dataset: BaseDataset, 
                           num_users: Optional[int] = None,
                           num_items: Optional[int] = None) -> HeterogeneousGraph:
        """
        Build initial graph directly from dataset.
        
        Args:
            dataset: BaseDataset instance
            num_users: Number of users to include (optional)
            num_items: Number of items to include (optional)
            
        Returns:
            HeterogeneousGraph: Constructed graph
        """
        self.logger.log_info(f"Building initial graph from dataset")
        
        # Get data from dataset
        user_items = dataset.get_user_items()
        item_features = dataset.get_item_features()
        interactions = dataset.get_interactions()
        
        # Create agent objects (simplified for initial build)
        agents = {}
        
        # Create user agents
        users = list(user_items.keys())
        if num_users:
            users = users[:num_users]
        
        for user_id in users:
            # Create simplified user agent
            agent = UserAgent(user_id, self.config)
            for item_id in user_items[user_id]:
                agent.update_preference({'item_id': item_id, 'rating': 1.0})
            agents[user_id] = agent
        
        # Create item agents
        items = list(item_features.keys())
        if num_items:
            items = items[:num_items]
        
        for item_id in items:
            agent = ItemAgent(item_id, self.config)
            # Update with metadata
            if item_id in item_features:
                agent.update_collaborative_pattern([], item_features[item_id])
            agents[item_id] = agent
        
        # Build graph from agents and interactions
        graph = self.build_graph(agents, interactions)
        
        self.logger.log_info(f"Initial graph built with {len(agents)} nodes")
        return graph
    
    def initialize_new_user(self, user_agent: UserAgent, 
                           initial_interactions: List[Dict[str, Any]],
                           graph: HeterogeneousGraph) -> None:
        """
        Initialize a new user in the graph with their interactions.
        
        Args:
            user_agent: UserAgent object for the new user
            initial_interactions: List of initial interactions
            graph: Existing HeterogeneousGraph instance
        """
        self.logger.log_info(f"Initializing new user {user_agent.agent_id}")
        
        # Add user node to graph
        graph.add_node(
            node_id=user_agent.agent_id,
            node_type='user',
            features={
                'preference_vector': user_agent.get_preference_memory(),
                'interaction_count': len(initial_interactions),
                'embedding': user_agent.get_embedding()
            },
            embedding=user_agent.get_embedding()
        )
        
        # Add interaction edges
        for interaction in initial_interactions:
            item_id = interaction.get('item_id')
            if item_id not in graph.nodes:
                self.logger.log_warning(f"Item {item_id} not in graph, skipping")
                continue
            
            weight = self._calculate_interaction_weight(
                interaction.get('rating', 1.0), interaction
            )
            
            graph.add_edge(
                source=user_agent.agent_id,
                target=item_id,
                relation_type=RelationType.INTERACT,
                weight=weight,
                metadata={'initial_interaction': True}
            )
        
        # Find similar users based on interactions
        self._connect_to_similar_users(user_agent, initial_interactions, graph)
        
        self.logger.log_info(f"New user {user_agent.agent_id} initialized with {len(initial_interactions)} interactions")
    
    def _connect_to_similar_users(self, user_agent: UserAgent, 
                                  interactions: List[Dict[str, Any]],
                                  graph: HeterogeneousGraph) -> None:
        """
        Connect new user to similar users based on their interactions.
        
        Args:
            user_agent: New UserAgent object
            interactions: List of interactions
            graph: Existing HeterogeneousGraph instance
        """
        new_user_items = set(interaction.get('item_id') for interaction in interactions)
        
        if not new_user_items:
            return
        
        # Find existing users with similar items
        for node_id, node in graph.nodes.items():
            if node.node_type != 'user' or node_id == user_agent.agent_id:
                continue
            
            # Get user's items from interactions
            user_items = set(node.features.get('interacted_items', []))
            if not user_items:
                continue
            
            # Compute overlap
            common_items = len(new_user_items & user_items)
            if common_items >= 1:  # At least one common item
                jaccard = common_items / len(new_user_items | user_items)
                
                if jaccard > self.min_edge_weight:
                    graph.add_edge(
                        source=user_agent.agent_id,
                        target=node_id,
                        relation_type=RelationType.SIMILAR_PREF,
                        weight=jaccard,
                        metadata={'new_user': True}
                    )
    
    def initialize_new_item(self, item_agent: ItemAgent, 
                           initial_interactions: List[Dict[str, Any]],
                           graph: HeterogeneousGraph) -> None:
        """
        Initialize a new item in the graph with initial interactions.
        
        Args:
            item_agent: ItemAgent object for the new item
            initial_interactions: List of initial interactions
            graph: Existing HeterogeneousGraph instance
        """
        self.logger.log_info(f"Initializing new item {item_agent.agent_id}")
        
        # Add item node to graph
        graph.add_node(
            node_id=item_agent.agent_id,
            node_type='item',
            features={
                'metadata': item_agent.get_item_metadata(),
                'popularity_score': 0.0,
                'embedding': item_agent.get_content_embedding()
            },
            embedding=item_agent.get_content_embedding()
        )
        
        # Add interaction edges
        for interaction in initial_interactions:
            user_id = interaction.get('user_id')
            if user_id not in graph.nodes:
                self.logger.log_warning(f"User {user_id} not in graph, skipping")
                continue
            
            weight = self._calculate_interaction_weight(
                interaction.get('rating', 1.0), interaction
            )
            
            graph.add_edge(
                source=user_id,
                target=item_agent.agent_id,
                relation_type=RelationType.INTERACT,
                weight=weight,
                metadata={'initial_interaction': True}
            )
        
        # Find similar items
        self._connect_to_similar_items(item_agent, graph)
        
        self.logger.log_info(f"New item {item_agent.agent_id} initialized with {len(initial_interactions)} interactions")
    
    def _connect_to_similar_items(self, item_agent: ItemAgent, 
                                 graph: HeterogeneousGraph) -> None:
        """
        Connect new item to similar existing items.
        
        Args:
            item_agent: New ItemAgent object
            graph: Existing HeterogeneousGraph instance
        """
        new_item_embedding = item_agent.get_content_embedding()
        if new_item_embedding is None:
            return
        
        # Find similar items
        for node_id, node in graph.nodes.items():
            if node.node_type != 'item' or node_id == item_agent.agent_id:
                continue
            
            if node.embedding is not None:
                similarity = EdgeWeightFunctions.cosine_similarity(
                    new_item_embedding, node.embedding
                )
                
                if similarity >= self.item_similarity_threshold:
                    graph.add_edge(
                        source=item_agent.agent_id,
                        target=node_id,
                        relation_type=RelationType.CONTENT_SIM,
                        weight=similarity,
                        metadata={'new_item': True}
                    )
    
    def build_from_networkx(self, nx_graph: nx.Graph, 
                           node_type_map: Dict[str, str],
                           edge_type_map: Dict[Tuple[str, str], str]) -> HeterogeneousGraph:
        """
        Build graph from a NetworkX graph.
        
        Args:
            nx_graph: NetworkX graph object
            node_type_map: Mapping from node ID to node type
            edge_type_map: Mapping from (source, target) to relation type
            
        Returns:
            HeterogeneousGraph: Constructed graph
        """
        graph = HeterogeneousGraph(self.config)
        
        # Add nodes
        for node_id, node_data in nx_graph.nodes(data=True):
            node_type = node_type_map.get(node_id, 'unknown')
            graph.add_node(
                node_id=node_id,
                node_type=node_type,
                features=node_data.get('features', {}),
                embedding=node_data.get('embedding')
            )
        
        # Add edges
        for source, target, edge_data in nx_graph.edges(data=True):
            relation_type = edge_type_map.get((source, target), RelationType.INTERACT.value)
            weight = edge_data.get('weight', 1.0)
            
            graph.add_edge(
                source=source,
                target=target,
                relation_type=relation_type,
                weight=weight,
                metadata=edge_data.get('metadata', {})
            )
        
        self.logger.log_info(f"Built graph from NetworkX with {len(graph.nodes)} nodes")
        return graph
    
    def get_building_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the graph building process.
        
        Returns:
            Dict[str, Any]: Building statistics
        """
        return self.building_stats.copy()
    
    def validate_graph(self, graph: HeterogeneousGraph) -> Dict[str, Any]:
        """
        Validate the constructed graph for structural integrity.
        
        Args:
            graph: HeterogeneousGraph instance to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check for isolated nodes
        isolated_nodes = []
        for node_id in graph.nodes:
            if graph.get_connection_count(node_id) == 0:
                isolated_nodes.append(node_id)
        
        if isolated_nodes:
            validation_results['warnings'].append(
                f"Found {len(isolated_nodes)} isolated nodes"
            )
            validation_results['statistics']['isolated_nodes'] = isolated_nodes[:10]  # First 10
        
        # Check edge weights
        invalid_weights = []
        for (src, dst, rel_type), weight in graph.edges.items():
            if weight < 0 or weight > 1:
                invalid_weights.append((src, dst, rel_type, weight))
        
        if invalid_weights:
            validation_results['errors'].append(
                f"Found {len(invalid_weights)} edges with invalid weights"
            )
            validation_results['statistics']['invalid_weights'] = invalid_weights[:5]
        
        # Check node features
        nodes_without_embedding = []
        for node_id, node in graph.nodes.items():
            if node.embedding is None:
                nodes_without_embedding.append(node_id)
        
        if nodes_without_embedding:
            validation_results['warnings'].append(
                f"Found {len(nodes_without_embedding)} nodes without embeddings"
            )
        
        # Compute graph density
        stats = graph.get_graph_statistics()
        validation_results['statistics']['density'] = stats.get('density', 0.0)
        validation_results['statistics']['avg_degree'] = stats.get('avg_degree', 0.0)
        
        if stats.get('density', 0) < 0.001:
            validation_results['warnings'].append(
                f"Graph density is very low: {stats.get('density', 0):.6f}"
            )
        
        # Set validity flag
        if validation_results['errors']:
            validation_results['is_valid'] = False
        
        self.logger.log_info(f"Graph validation completed: {validation_results['is_valid']}")
        return validation_results
    
    def sample_subgraph(self, graph: HeterogeneousGraph, 
                       center_nodes: List[str],
                       hop_count: int = 2,
                       max_nodes: int = 1000) -> HeterogeneousGraph:
        """
        Sample a subgraph centered around specific nodes.
        
        Args:
            graph: Original HeterogeneousGraph instance
            center_nodes: List of center node IDs
            hop_count: Number of hops to expand
            max_nodes: Maximum nodes in subgraph
            
        Returns:
            HeterogeneousGraph: Sampled subgraph
        """
        # BFS expansion
        visited = set(center_nodes)
        frontier = set(center_nodes)
        
        for _ in range(hop_count):
            new_frontier = set()
            for node_id in frontier:
                # Get all neighbors
                for rel_type in graph.adjacency_lists:
                    neighbors = graph.get_neighbors(node_id, rel_type)
                    for neighbor_id, _ in neighbors:
                        if neighbor_id not in visited:
                            new_frontier.add(neighbor_id)
            
            visited.update(new_frontier)
            frontier = new_frontier
            
            if len(visited) >= max_nodes:
                break
        
        # Extract subgraph
        subgraph = graph.get_subgraph(list(visited))
        
        self.logger.log_info(f"Sampled subgraph with {len(visited)} nodes")
        return subgraph


# Example usage
if __name__ == "__main__":
    # Load configuration
    config_path = "config/default_config.yaml"
    config_loader = ConfigLoader(config_path)
    config = config_loader.load_config()
    
    # Create graph builder
    builder = GraphBuilder(config)
    
    # Example: Build from dummy data
    from data.dataset import AmazonDataset
    
    # Load dataset (simplified)
    dataset = AmazonDataset("electronics", config)
    
    # Build graph
    graph = builder.build_initial_graph(dataset, num_users=100, num_items=50)
    
    # Validate graph
    validation = builder.validate_graph(graph)
    print(f"Validation results: {validation}")
    
    # Get statistics
    stats = builder.get_building_stats()
    print(f"Building stats: {stats}")
    
    # Sample subgraph
    center_nodes = list(graph.nodes.keys())[:5]
    subgraph = builder.sample_subgraph(graph, center_nodes, hop_count=2)
    print(f"Subgraph has {len(subgraph)} nodes")