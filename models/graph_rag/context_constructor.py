"""
Context Constructor Module for H-GRAGrecsys

This module constructs comprehensive context from retrieved graph elements,
including subgraphs, metapaths, and node information, formatting them for
LLM consumption and reasoning.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from collections import defaultdict, OrderedDict
import json
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.graph.dynamic_graph import DynamicGraph
from models.graph.relation_types import RelationType, RelationTypeRegistry
from models.graph_rag.metapath_extractor import MetapathExtractor
from models.agent.user_agent import UserAgent
from models.agent.item_agent import ItemAgent
from models.llm.prompt_templates import PromptTemplates
from utils.logger import Logger
from utils.config_loader import ConfigLoader


class ContextConstructor:
    """
    Constructs context from graph elements for LLM consumption.
    
    This class handles:
    - Constructing context from subgraphs and metapaths
    - Verbalizing subgraphs and graph elements
    - Formatting context for LLM prompts
    - Adding relation descriptions and metadata
    - Ranking context elements by relevance
    - Creating different context types (recommendation, explanation, reflection)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the context constructor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = Logger.get_instance(log_dir='logs', name='context_constructor')
        
        # Extract configuration
        graph_rag_config = config.get('model', {}).get('graph_rag', {})
        self.max_context_elements = graph_rag_config.get('max_context_elements', 20)
        self.max_text_length = graph_rag_config.get('max_text_length', 512)
        self.include_node_features = graph_rag_config.get('include_node_features', True)
        self.include_relation_metadata = graph_rag_config.get('include_relation_metadata', True)
        self.include_confidence_scores = graph_rag_config.get('include_confidence_scores', True)
        self.context_format = graph_rag_config.get('context_format', 'structured')
        
        # Initialize prompt templates
        self.prompt_templates = PromptTemplates(config)
        
        # Relation registry for descriptions
        self.relation_registry = RelationTypeRegistry(config)
        
        # Cache for constructed contexts
        self.context_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_size = graph_rag_config.get('context_cache_size', 50)
        
        # Statistics
        self.construction_stats = {
            'total_constructions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_context_elements': 0,
            'avg_context_length': 0,
            'construction_times': []
        }
        
        self.logger.log_info(f"Initialized ContextConstructor with format={self.context_format}")
    
    def construct_context(self, query_node: str, 
                         subgraph: Optional[DynamicGraph] = None,
                         metapaths: Optional[List[Dict[str, Any]]] = None,
                         ppr_scores: Optional[Dict[str, float]] = None,
                         context_type: str = 'recommendation',
                         include_metadata: bool = True) -> Dict[str, Any]:
        """
        Construct comprehensive context from graph elements.
        
        Args:
            query_node: Query node ID
            subgraph: Retrieved subgraph
            metapaths: Extracted metapaths
            ppr_scores: PPR scores for ranking
            context_type: Type of context ('recommendation', 'explanation', 'reflection')
            include_metadata: Whether to include metadata
            
        Returns:
            Dict[str, Any]: Constructed context
        """
        self.logger.log_info(f"Constructing {context_type} context for node {query_node}")
        
        # Check cache
        cache_key = f"{query_node}_{context_type}_{hash(str(subgraph)) if subgraph else 'none'}"
        if cache_key in self.context_cache:
            self.logger.log_info(f"Cache hit for {cache_key}")
            self.construction_stats['cache_hits'] += 1
            return self.context_cache[cache_key]
        
        self.construction_stats['cache_misses'] += 1
        start_time = datetime.now().timestamp()
        
        # Initialize context
        context = {
            'query_node': query_node,
            'context_type': context_type,
            'timestamp': datetime.now().isoformat(),
            'elements': [],
            'metadata': {},
            'statistics': {}
        }
        
        # Get node information
        query_node_info = self._get_node_info(query_node)
        context['query_node_info'] = query_node_info
        
        # Add subgraph context
        if subgraph is not None:
            subgraph_context = self._construct_subgraph_context(subgraph, query_node, ppr_scores)
            context['subgraph'] = subgraph_context
            context['elements'].extend(subgraph_context.get('elements', []))
        
        # Add metapath context
        if metapaths is not None:
            metapath_context = self._construct_metapath_context(metapaths, query_node)
            context['metapaths'] = metapath_context
            context['elements'].extend(metapath_context.get('elements', []))
        
        # Add PPR scores context
        if ppr_scores is not None:
            ppr_context = self._construct_ppr_context(ppr_scores, query_node)
            context['ppr_scores'] = ppr_context
        
        # Rank context elements
        context['elements'] = self.rank_context_elements(
            context['elements'], 
            query_node
        )
        
        # Limit context elements
        if len(context['elements']) > self.max_context_elements:
            context['elements'] = context['elements'][:self.max_context_elements]
        
        # Format context for LLM
        context['formatted'] = self._format_context_for_llm(context, context_type)
        
        # Add metadata
        if include_metadata:
            context['metadata'] = self._construct_metadata(context, query_node)
        
        # Add statistics
        context['statistics'] = {
            'total_elements': len(context['elements']),
            'num_subgraph_nodes': len(subgraph.nodes) if subgraph else 0,
            'num_subgraph_edges': len(subgraph.edges) if subgraph else 0,
            'num_metapaths': len(metapaths) if metapaths else 0,
            'formatted_length': len(context['formatted']),
            'construction_time': datetime.now().timestamp() - start_time
        }
        
        # Cache context
        if len(self.context_cache) < self.cache_size:
            self.context_cache[cache_key] = context
        
        # Update statistics
        self.construction_stats['total_constructions'] += 1
        self.construction_stats['avg_context_elements'] = (
            (self.construction_stats['avg_context_elements'] * 
             (self.construction_stats['total_constructions'] - 1) + 
             len(context['elements'])) / self.construction_stats['total_constructions']
        )
        self.construction_stats['avg_context_length'] = (
            (self.construction_stats['avg_context_length'] * 
             (self.construction_stats['total_constructions'] - 1) + 
             len(context['formatted'])) / self.construction_stats['total_constructions']
        )
        self.construction_stats['construction_times'].append(
            context['statistics']['construction_time']
        )
        
        self.logger.log_info(f"Constructed context with {len(context['elements'])} elements")
        return context
    
    def _construct_subgraph_context(self, subgraph: DynamicGraph, 
                                   query_node: str,
                                   ppr_scores: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Construct context from subgraph.
        
        Args:
            subgraph: DynamicGraph subgraph
            query_node: Query node ID
            ppr_scores: PPR scores for ranking
            
        Returns:
            Dict[str, Any]: Subgraph context
        """
        context = {
            'nodes': [],
            'edges': [],
            'elements': [],
            'statistics': {
                'num_nodes': len(subgraph.nodes),
                'num_edges': len(subgraph.edges),
                'node_types': defaultdict(int),
                'edge_types': defaultdict(int)
            }
        }
        
        # Process nodes
        for node_id, node in subgraph.nodes.items():
            node_info = self._get_node_info(node_id)
            
            # Add PPR score if available
            if ppr_scores is not None:
                node_info['ppr_score'] = ppr_scores.get(node_id, 0.0)
            
            # Add to context
            context['nodes'].append(node_info)
            
            # Create element
            element = {
                'type': 'node',
                'node_id': node_id,
                'node_type': node.node_type,
                'features': node_info.get('features', {}),
                'importance': node_info.get('ppr_score', 0.0),
                'description': self._verbalize_node(node_id)
            }
            context['elements'].append(element)
            
            # Update statistics
            context['statistics']['node_types'][node.node_type] += 1
        
        # Process edges
        for (source, target, rel_type), weight in subgraph.edges.items():
            edge_info = {
                'source': source,
                'target': target,
                'relation_type': rel_type,
                'weight': weight
            }
            
            # Add metadata
            if self.include_relation_metadata:
                metadata = self.relation_registry.get_metadata(rel_type)
                if metadata:
                    edge_info['description'] = metadata.description
                    edge_info['is_symmetric'] = metadata.is_symmetric
            
            context['edges'].append(edge_info)
            
            # Create element
            element = {
                'type': 'edge',
                'source': source,
                'target': target,
                'relation_type': rel_type,
                'weight': weight,
                'description': self._verbalize_edge(source, target, rel_type)
            }
            context['elements'].append(element)
            
            # Update statistics
            context['statistics']['edge_types'][rel_type] += 1
        
        # Sort nodes by importance
        context['nodes'].sort(key=lambda x: x.get('ppr_score', 0.0), reverse=True)
        
        # Sort edges by weight
        context['edges'].sort(key=lambda x: x.get('weight', 0.0), reverse=True)
        
        return context
    
    def _construct_metapath_context(self, metapaths: List[Dict[str, Any]], 
                                   query_node: str) -> Dict[str, Any]:
        """
        Construct context from metapaths.
        
        Args:
            metapaths: List of metapath dictionaries
            query_node: Query node ID
            
        Returns:
            Dict[str, Any]: Metapath context
        """
        context = {
            'metapaths': [],
            'elements': [],
            'statistics': {
                'total_metapaths': len(metapaths),
                'path_types': defaultdict(int)
            }
        }
        
        for metapath in metapaths:
            # Add verbalization
            if 'verbalized' not in metapath:
                metapath['verbalized'] = self.verbalize_subgraph_element(
                    {'type': 'metapath', 'path': metapath}
                )
            
            # Add importance
            if 'importance' not in metapath:
                metapath['importance'] = metapath.get('importance', 0.5)
            
            context['metapaths'].append(metapath)
            
            # Create element
            element = {
                'type': 'metapath',
                'path': metapath.get('path', []),
                'relations': metapath.get('relations', []),
                'path_type': metapath.get('path_type', 'unknown'),
                'importance': metapath.get('importance', 0.5),
                'description': metapath.get('verbalized', '')
            }
            context['elements'].append(element)
            
            # Update statistics
            path_type = metapath.get('path_type', 'unknown')
            context['statistics']['path_types'][path_type] += 1
        
        # Sort metapaths by importance
        context['metapaths'].sort(key=lambda x: x.get('importance', 0.0), reverse=True)
        
        return context
    
    def _construct_ppr_context(self, ppr_scores: Dict[str, float], 
                              query_node: str) -> Dict[str, Any]:
        """
        Construct context from PPR scores.
        
        Args:
            ppr_scores: PPR scores dictionary
            query_node: Query node ID
            
        Returns:
            Dict[str, Any]: PPR context
        """
        # Sort nodes by PPR score
        sorted_nodes = sorted(
            [(nid, score) for nid, score in ppr_scores.items() if nid != query_node],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Take top 20
        top_nodes = sorted_nodes[:20]
        
        context = {
            'top_nodes': top_nodes,
            'num_nodes': len(ppr_scores),
            'query_node': query_node
        }
        
        # Add node types and descriptions
        for i, (node_id, score) in enumerate(top_nodes):
            node_info = self._get_node_info(node_id)
            top_nodes[i] = (node_id, score, node_info.get('node_type', 'unknown'))
        
        return context
    
    def verbalize_subgraph(self, subgraph: DynamicGraph, 
                          max_nodes: int = 20,
                          max_edges: int = 20) -> str:
        """
        Verbalize a subgraph into natural language.
        
        Args:
            subgraph: DynamicGraph subgraph
            max_nodes: Maximum nodes to include
            max_edges: Maximum edges to include
            
        Returns:
            str: Verbalized subgraph description
        """
        if not subgraph or len(subgraph.nodes) == 0:
            return "Empty subgraph"
        
        verbalization = []
        
        # Add subgraph summary
        verbalization.append(f"Subgraph with {len(subgraph.nodes)} nodes and {len(subgraph.edges)} edges.")
        
        # Add node descriptions (limited)
        node_list = list(subgraph.nodes.keys())[:max_nodes]
        if node_list:
            verbalization.append("Nodes:")
            for node_id in node_list:
                verbalization.append(f"  - {self._verbalize_node(node_id)}")
        
        # Add edge descriptions (limited)
        edge_list = list(subgraph.edges.keys())[:max_edges]
        if edge_list:
            verbalization.append("Relationships:")
            for source, target, rel_type in edge_list:
                verbalization.append(f"  - {self._verbalize_edge(source, target, rel_type)}")
        
        return "\n".join(verbalization)
    
    def verbalize_subgraph_element(self, element: Dict[str, Any]) -> str:
        """
        Verbalize a single subgraph element.
        
        Args:
            element: Element dictionary
            
        Returns:
            str: Verbalized element description
        """
        element_type = element.get('type', 'unknown')
        
        if element_type == 'node':
            node_id = element.get('node_id', '')
            return self._verbalize_node(node_id)
        
        elif element_type == 'edge':
            source = element.get('source', '')
            target = element.get('target', '')
            rel_type = element.get('relation_type', '')
            return self._verbalize_edge(source, target, rel_type)
        
        elif element_type == 'metapath':
            path = element.get('path', [])
            relations = element.get('relations', [])
            return self._verbalize_path(path, relations)
        
        else:
            return str(element)
    
    def _verbalize_node(self, node_id: str) -> str:
        """
        Verbalize a single node.
        
        Args:
            node_id: Node ID
            
        Returns:
            str: Verbalized node description
        """
        node_info = self._get_node_info(node_id)
        node_type = node_info.get('node_type', 'unknown')
        
        # Get node label
        label = node_info.get('label', node_id[:8])
        
        # Get node features summary
        feature_summary = ""
        if self.include_node_features and node_info.get('features'):
            features = node_info.get('features', {})
            if node_type == 'user':
                pref = features.get('preference_vector', [])
                if pref:
                    feature_summary = f" (preferences: {len(pref)} items)"
            elif node_type == 'item':
                metadata = features.get('metadata', {})
                title = metadata.get('title', '')
                if title:
                    feature_summary = f" ('{title}')"
        
        return f"{node_type} {label}{feature_summary}"
    
    def _verbalize_edge(self, source: str, target: str, rel_type: str) -> str:
        """
        Verbalize a single edge.
        
        Args:
            source: Source node ID
            target: Target node ID
            rel_type: Relation type
            
        Returns:
            str: Verbalized edge description
        """
        source_info = self._get_node_info(source)
        target_info = self._get_node_info(target)
        
        source_label = source_info.get('label', source[:8])
        target_label = target_info.get('label', target[:8])
        
        # Get relation description
        rel_desc = self._get_relation_description(rel_type)
        
        return f"{source_label} {rel_desc} {target_label}"
    
    def _verbalize_path(self, path: List[str], relations: List[str]) -> str:
        """
        Verbalize a metapath.
        
        Args:
            path: List of node IDs
            relations: List of relation types
            
        Returns:
            str: Verbalized path description
        """
        if not path:
            return "Empty path"
        
        verbalization = []
        for i, node_id in enumerate(path):
            node_info = self._get_node_info(node_id)
            label = node_info.get('label', node_id[:8])
            verbalization.append(label)
            
            if i < len(relations):
                rel_desc = self._get_relation_description(relations[i])
                verbalization.append(rel_desc)
        
        return " ".join(verbalization)
    
    def _get_relation_description(self, relation_type: str) -> str:
        """
        Get description for a relation type.
        
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
    
    def _get_node_info(self, node_id: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a node.
        
        Args:
            node_id: Node ID
            
        Returns:
            Dict[str, Any]: Node information
        """
        # Try to get from graph (if available)
        info = {
            'node_id': node_id,
            'node_type': 'unknown',
            'label': node_id[:8],
            'features': {},
            'metadata': {}
        }
        
        # If we have access to graph, get more info
        if hasattr(self, 'graph') and self.graph:
            node = self.graph.nodes.get(node_id)
            if node:
                info['node_type'] = node.node_type
                info['features'] = node.features
                info['metadata'] = node.metadata
                info['label'] = self._extract_node_label(node)
                if node.embedding is not None:
                    info['has_embedding'] = True
        
        return info
    
    def _extract_node_label(self, node) -> str:
        """
        Extract readable label from node.
        
        Args:
            node: GraphNode object
            
        Returns:
            str: Readable label
        """
        if node.node_type == 'user':
            return node.features.get('name', 
                   node.features.get('username', 
                   node.node_id[:8]))
        elif node.node_type == 'item':
            return node.features.get('title', 
                   node.features.get('name', 
                   node.node_id[:8]))
        return node.node_id[:8]
    
    def _format_context_for_llm(self, context: Dict[str, Any], 
                               context_type: str) -> str:
        """
        Format context for LLM consumption.
        
        Args:
            context: Context dictionary
            context_type: Type of context
            
        Returns:
            str: Formatted context string
        """
        if self.context_format == 'structured':
            return self._format_structured(context, context_type)
        elif self.context_format == 'narrative':
            return self._format_narrative(context, context_type)
        elif self.context_format == 'template':
            return self._format_template(context, context_type)
        else:
            # Default to structured
            return self._format_structured(context, context_type)
    
    def _format_structured(self, context: Dict[str, Any], 
                          context_type: str) -> str:
        """
        Format context in structured format.
        
        Args:
            context: Context dictionary
            context_type: Type of context
            
        Returns:
            str: Structured context
        """
        lines = []
        
        # Add header
        query_node = context.get('query_node', 'unknown')
        lines.append(f"Context for node: {query_node}")
        lines.append(f"Context Type: {context_type}")
        lines.append(f"Generated: {context.get('timestamp', '')}")
        lines.append("")
        
        # Add query node info
        query_info = context.get('query_node_info', {})
        lines.append(f"Query Node Type: {query_info.get('node_type', 'unknown')}")
        lines.append(f"Query Node Label: {query_info.get('label', 'unknown')}")
        lines.append("")
        
        # Add subgraph summary
        subgraph = context.get('subgraph', {})
        if subgraph:
            lines.append("Subgraph Summary:")
            lines.append(f"  Nodes: {subgraph.get('statistics', {}).get('num_nodes', 0)}")
            lines.append(f"  Edges: {subgraph.get('statistics', {}).get('num_edges', 0)}")
            
            # Node types
            node_types = subgraph.get('statistics', {}).get('node_types', {})
            if node_types:
                lines.append("  Node Types:")
                for ntype, count in node_types.items():
                    lines.append(f"    - {ntype}: {count}")
            
            # Top nodes
            top_nodes = subgraph.get('nodes', [])[:5]
            if top_nodes:
                lines.append("  Important Nodes:")
                for node in top_nodes:
                    lines.append(f"    - {node.get('node_type', '')}: {node.get('label', '')}")
            
            lines.append("")
        
        # Add metapath summary
        metapaths = context.get('metapaths', {})
        if metapaths:
            lines.append("Metapaths Summary:")
            lines.append(f"  Total: {metapaths.get('statistics', {}).get('total_metapaths', 0)}")
            
            # Path types
            path_types = metapaths.get('statistics', {}).get('path_types', {})
            if path_types:
                lines.append("  Path Types:")
                for ptype, count in path_types.items():
                    lines.append(f"    - {ptype}: {count}")
            
            # Sample metapaths
            sample_paths = metapaths.get('metapaths', [])[:3]
            if sample_paths:
                lines.append("  Sample Metapaths:")
                for path in sample_paths:
                    verbalized = path.get('verbalized', '')
                    if verbalized:
                        lines.append(f"    - {verbalized}")
            
            lines.append("")
        
        # Add PPR top nodes
        ppr = context.get('ppr_scores', {})
        if ppr:
            lines.append("Top PPR Nodes:")
            for i, (node_id, score, node_type) in enumerate(ppr.get('top_nodes', [])[:5]):
                lines.append(f"  {i+1}. {node_type} {node_id[:8]} (score: {score:.4f})")
            lines.append("")
        
        # Add formatted elements
        elements = context.get('elements', [])
        if elements:
            lines.append("Context Elements:")
            for i, element in enumerate(elements[:10]):
                description = element.get('description', '')
                if description:
                    lines.append(f"  {i+1}. {description}")
            if len(elements) > 10:
                lines.append(f"  ... and {len(elements) - 10} more elements")
        
        return "\n".join(lines)
    
    def _format_narrative(self, context: Dict[str, Any], 
                         context_type: str) -> str:
        """
        Format context in narrative format.
        
        Args:
            context: Context dictionary
            context_type: Type of context
            
        Returns:
            str: Narrative context
        """
        lines = []
        query_info = context.get('query_node_info', {})
        query_label = query_info.get('label', 'unknown')
        
        # Build narrative
        lines.append(f"Here is the context for {query_label} ({context_type} context):")
        lines.append("")
        
        # Add subgraph narrative
        subgraph = context.get('subgraph', {})
        if subgraph:
            nodes = subgraph.get('nodes', [])[:5]
            edges = subgraph.get('edges', [])[:5]
            
            if nodes:
                node_desc = ", ".join([f"{n.get('node_type', '')} {n.get('label', '')}" 
                                       for n in nodes])
                lines.append(f"The subgraph contains important entities: {node_desc}.")
            
            if edges:
                edge_desc = []
                for edge in edges:
                    source = edge.get('source', '')
                    target = edge.get('target', '')
                    rel = edge.get('relation_type', '')
                    rel_desc = self._get_relation_description(rel)
                    edge_desc.append(f"{source[:8]} {rel_desc} {target[:8]}")
                lines.append(f"Key relationships include: {'; '.join(edge_desc[:3])}.")
        
        # Add metapath narrative
        metapaths = context.get('metapaths', {})
        if metapaths:
            sample_paths = metapaths.get('metapaths', [])[:2]
            if sample_paths:
                path_desc = []
                for path in sample_paths:
                    verbalized = path.get('verbalized', '')
                    if verbalized:
                        path_desc.append(verbalized)
                if path_desc:
                    lines.append(f"Relevant patterns discovered: {'; '.join(path_desc)}.")
        
        # Add PPR narrative
        ppr = context.get('ppr_scores', {})
        if ppr:
            top_nodes = ppr.get('top_nodes', [])[:3]
            if top_nodes:
                node_desc = ", ".join([f"{nid[:8]} (score {score:.3f})" 
                                      for nid, score, _ in top_nodes])
                lines.append(f"Most relevant related nodes: {node_desc}.")
        
        return "\n".join(lines)
    
    def _format_template(self, context: Dict[str, Any], 
                        context_type: str) -> str:
        """
        Format context using prompt templates.
        
        Args:
            context: Context dictionary
            context_type: Type of context
            
        Returns:
            str: Template-formatted context
        """
        # Get appropriate template
        if context_type == 'recommendation':
            template = self.prompt_templates.get_ranking_prompt(
                context.get('query_node', ''),
                [],  # Candidates will be added later
                context['formatted'] if 'formatted' in context else context
            )
        elif context_type == 'explanation':
            template = self.prompt_templates.get_explanation_prompt(
                context.get('query_node', ''),
                '',  # Item will be added later
                ''   # Recommendation will be added later
            )
        elif context_type == 'reflection':
            template = self.prompt_templates.get_reflection_prompt(
                context.get('query_node', ''),
                '',  # Item will be added later
                {}   # Context will be added later
            )
        else:
            # Generic template
            template = self.prompt_templates.get_summarization_prompt(
                self._format_structured(context, context_type),
                self.max_text_length
            )
        
        return template
    
    def rank_context_elements(self, elements: List[Dict[str, Any]], 
                             query_node: str) -> List[Dict[str, Any]]:
        """
        Rank context elements by relevance to query.
        
        Args:
            elements: List of context elements
            query_node: Query node ID
            
        Returns:
            List[Dict[str, Any]]: Ranked elements
        """
        if not elements:
            return elements
        
        # Get query node embedding if available
        query_embedding = None
        if hasattr(self, 'graph') and self.graph:
            node = self.graph.nodes.get(query_node)
            if node and node.embedding is not None:
                query_embedding = node.embedding
        
        # Score each element
        scored_elements = []
        for element in elements:
            score = self._compute_element_relevance(element, query_node, query_embedding)
            scored_elements.append((score, element))
        
        # Sort by score
        scored_elements.sort(key=lambda x: x[0], reverse=True)
        
        return [elem for _, elem in scored_elements]
    
    def _compute_element_relevance(self, element: Dict[str, Any], 
                                  query_node: str,
                                  query_embedding: Optional[torch.Tensor] = None) -> float:
        """
        Compute relevance score for a context element.
        
        Args:
            element: Context element
            query_node: Query node ID
            query_embedding: Query node embedding
            
        Returns:
            float: Relevance score
        """
        score = 0.0
        
        # Element importance (pre-computed)
        importance = element.get('importance', 0.0)
        score += 0.3 * importance
        
        # Element type weight
        element_type = element.get('type', 'unknown')
        type_weights = {
            'node': 0.7,
            'edge': 0.6,
            'metapath': 0.8,
            'pattern': 0.5
        }
        score += 0.2 * type_weights.get(element_type, 0.3)
        
        # Element description relevance (simple keyword matching)
        description = element.get('description', '').lower()
        if description:
            # Check if description mentions query node or related concepts
            if query_node.lower() in description:
                score += 0.2
        
        # Embedding similarity (if available)
        if query_embedding is not None and element_type == 'node':
            node_id = element.get('node_id', '')
            if node_id and hasattr(self, 'graph') and self.graph:
                node = self.graph.nodes.get(node_id)
                if node and node.embedding is not None:
                    similarity = torch.cosine_similarity(
                        query_embedding.unsqueeze(0),
                        node.embedding.unsqueeze(0)
                    ).item()
                    score += 0.3 * max(0.0, similarity)
        
        return min(1.0, score)
    
    def _construct_metadata(self, context: Dict[str, Any], 
                           query_node: str) -> Dict[str, Any]:
        """
        Construct metadata for context.
        
        Args:
            context: Context dictionary
            query_node: Query node ID
            
        Returns:
            Dict[str, Any]: Metadata
        """
        metadata = {
            'query_node': query_node,
            'context_type': context.get('context_type', 'unknown'),
            'timestamp': context.get('timestamp', datetime.now().isoformat()),
            'format': self.context_format,
            'total_elements': len(context.get('elements', [])),
            'has_subgraph': 'subgraph' in context,
            'has_metapaths': 'metapaths' in context,
            'has_ppr_scores': 'ppr_scores' in context
        }
        
        # Add statistics
        stats = context.get('statistics', {})
        if stats:
            metadata['statistics'] = stats
        
        return metadata
    
    def add_relation_descriptions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add relation descriptions to context.
        
        Args:
            context: Context dictionary
            
        Returns:
            Dict[str, Any]: Context with relation descriptions
        """
        if 'subgraph' in context:
            subgraph = context['subgraph']
            for edge in subgraph.get('edges', []):
                rel_type = edge.get('relation_type', '')
                edge['description'] = self._get_relation_description(rel_type)
        
        if 'metapaths' in context:
            metapaths = context['metapaths']
            for metapath in metapaths.get('metapaths', []):
                relations = metapath.get('relations', [])
                metapath['relation_descriptions'] = [
                    self._get_relation_description(rel) for rel in relations
                ]
        
        return context
    
    def get_context_summary(self, context: Dict[str, Any]) -> str:
        """
        Get a brief summary of the context.
        
        Args:
            context: Context dictionary
            
        Returns:
            str: Context summary
        """
        lines = []
        lines.append(f"Context for {context.get('query_node', 'unknown')}")
        lines.append(f"Type: {context.get('context_type', 'unknown')}")
        lines.append(f"Elements: {len(context.get('elements', []))}")
        
        if 'subgraph' in context:
            subgraph = context['subgraph']
            lines.append(f"Subgraph: {subgraph.get('statistics', {}).get('num_nodes', 0)} nodes, "
                        f"{subgraph.get('statistics', {}).get('num_edges', 0)} edges")
        
        if 'metapaths' in context:
            metapaths = context['metapaths']
            lines.append(f"Metapaths: {metapaths.get('statistics', {}).get('total_metapaths', 0)}")
        
        return "\n".join(lines)
    
    def get_construction_statistics(self) -> Dict[str, Any]:
        """
        Get construction statistics.
        
        Returns:
            Dict[str, Any]: Construction statistics
        """
        stats = self.construction_stats.copy()
        stats['cache_size'] = len(self.context_cache)
        stats['cache_hit_rate'] = (
            stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
            if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0.0
        )
        stats['avg_construction_time'] = np.mean(stats['construction_times']) if stats['construction_times'] else 0.0
        stats['recent_construction_times'] = stats['construction_times'][-10:]
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the context cache."""
        self.context_cache.clear()
        self.logger.log_info("Cleared context cache")


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
    graph.add_node('user_1', 'user', features={'name': 'Alice', 'preference_vector': ['item_1', 'item_2']})
    graph.add_node('user_2', 'user', features={'name': 'Bob', 'preference_vector': ['item_2', 'item_3']})
    graph.add_node('item_1', 'item', features={'title': 'Product A', 'categories': ['electronics']})
    graph.add_node('item_2', 'item', features={'title': 'Product B', 'categories': ['electronics']})
    graph.add_node('item_3', 'item', features={'title': 'Product C', 'categories': ['books']})
    
    graph.add_edge('user_1', 'item_1', RelationType.INTERACT, weight=0.9)
    graph.add_edge('user_1', 'item_2', RelationType.INTERACT, weight=0.7)
    graph.add_edge('user_2', 'item_2', RelationType.INTERACT, weight=0.8)
    graph.add_edge('user_2', 'item_3', RelationType.INTERACT, weight=0.6)
    graph.add_edge('user_1', 'user_2', RelationType.SIMILAR_PREF, weight=0.8)
    graph.add_edge('item_1', 'item_2', RelationType.CONTENT_SIM, weight=0.6)
    
    # Create context constructor
    constructor = ContextConstructor(config)
    constructor.graph = graph  # Set graph reference
    
    # Extract subgraph and metapaths
    subgraph = graph.get_subgraph(['user_1', 'item_1', 'item_2', 'user_2'])
    
    # Create metapath extractor
    extractor = MetapathExtractor(config)
    extractor.graph = graph
    metapaths = extractor.extract_all_paths('user_1', max_len=3)
    
    # Compute PPR scores
    ppr_scores = graph.compute_ppr_scores('user_1')
    
    # Construct context
    context = constructor.construct_context(
        query_node='user_1',
        subgraph=subgraph,
        metapaths=metapaths[:5],
        ppr_scores=ppr_scores,
        context_type='recommendation'
    )
    
    # Print formatted context
    print("Formatted Context:")
    print("-" * 50)
    print(context['formatted'])
    print("-" * 50)
    
    # Get context summary
    summary = constructor.get_context_summary(context)
    print("\nContext Summary:")
    print(summary)
    
    # Get statistics
    stats = constructor.get_construction_statistics()
    print(f"\nConstruction Statistics: {stats}")