"""
Knowledge Distiller Module for H-GRAGrecsys

This module implements the knowledge distillation framework for transferring
knowledge from LLM teacher to GNN student. It provides:
- Knowledge extraction from teacher LLM
- Representation alignment between teacher and student
- Memory dynamics distillation for agent memories
- Path importance distillation for metapath reasoning
- Adaptive distillation with temperature scheduling
- Multi-teacher ensemble support

The knowledge distiller enables efficient transfer of complex reasoning
capabilities from LLMs to lightweight GNN models.
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import time
import math
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from sibling modules
from distillation.loss_functions import DistillationLoss, ComponentWiseLoss, PathImportanceLoss
from distillation.component_disentangler import ComponentDisentangler

# Import from GNN module
from models.gnn.gnn_encoder import GNNEncoder
from models.gnn.heterogeneous_gnn import HeterogeneousGNN

# Import from LLM module
from models.llm.llm_interface import LLMInterface
from models.llm.prompt_templates import PromptTemplates

# Import from agent module
from models.agent.user_agent import UserAgent
from models.agent.item_agent import ItemAgent
from models.agent.memory import AgentMemory, HierarchicalMemory

# Import from graph module
from models.graph.heterogeneous_graph import HeterogeneousGraph

# Import from utils
from utils.config_loader import ConfigLoader
from utils.logger import Logger
from utils.seed_manager import SeedManager
from utils.timer import Timer


@dataclass
class DistillationOutput:
    """
    Dataclass for distillation outputs.
    
    Attributes:
        teacher_outputs: Outputs from teacher model.
        student_outputs: Outputs from student model.
        losses: Loss values for each component.
        alignment_score: Score for representation alignment.
        distillation_quality: Quality metric for distillation.
        metadata: Additional metadata.
    """
    teacher_outputs: Dict[str, Any]
    student_outputs: Dict[str, Any]
    losses: Dict[str, float]
    alignment_score: float = 0.0
    distillation_quality: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeDistiller(nn.Module):
    """
    Knowledge distiller for transferring knowledge from LLM to GNN.
    
    This class orchestrates the distillation process, including knowledge
    extraction, representation alignment, and memory dynamics distillation.
    """
    
    def __init__(
        self,
        teacher_llm: Optional[LLMInterface] = None,
        student_gnn: Optional[GNNEncoder] = None,
        config: Optional[Union[str, Dict, ConfigLoader]] = None
    ):
        """
        Initialize the knowledge distiller.
        
        Args:
            teacher_llm: Optional teacher LLM interface.
            student_gnn: Optional student GNN encoder.
            config: Configuration object or path to config file.
                   Can be a string path, dict, or ConfigLoader instance.
        
        Raises:
            ValueError: If config is invalid or missing required fields.
        """
        super(KnowledgeDistiller, self).__init__()
        
        # Load configuration
        if config is None:
            self.config = {
                'model': {
                    'distillation': {
                        'temperature': 0.07,
                        'component_weights': [1.0, 1.0, 1.0],
                        'alpha': 0.5,
                        'beta': 0.3,
                        'gamma': 0.2,
                        'use_teacher_ensemble': False,
                        'distill_memory': True,
                        'align_representations': True,
                        'temperature_scheduling': True
                    }
                }
            }
        elif isinstance(config, str):
            self.config_loader = ConfigLoader(config)
            self.config = self.config_loader.load_config()
        elif isinstance(config, dict):
            self.config = config
            self.config_loader = None
        elif isinstance(config, ConfigLoader):
            self.config_loader = config
            self.config = config.load_config()
        else:
            raise ValueError(f"Invalid config type: {type(config)}")
        
        # Setup logger
        self.logger = Logger(
            log_dir=self.config.get('logging', {}).get('log_dir', './logs'),
            name='knowledge_distiller'
        )
        
        # Extract configuration
        dist_config = self.config.get('model', {}).get('distillation', {})
        
        self.temperature = dist_config.get('temperature', 0.07)
        self.component_weights = dist_config.get('component_weights', [1.0, 1.0, 1.0])
        self.alpha = dist_config.get('alpha', 0.5)
        self.beta = dist_config.get('beta', 0.3)
        self.gamma = dist_config.get('gamma', 0.2)
        self.use_teacher_ensemble = dist_config.get('use_teacher_ensemble', False)
        self.distill_memory = dist_config.get('distill_memory', True)
        self.align_representations = dist_config.get('align_representations', True)
        self.temperature_scheduling = dist_config.get('temperature_scheduling', True)
        
        # Initialize components
        self.teacher_llm = teacher_llm if teacher_llm is not None else self._create_teacher()
        self.student_gnn = student_gnn if student_gnn is not None else self._create_student()
        
        # Initialize loss functions
        self.distillation_loss = DistillationLoss(config)
        self.component_wise_loss = ComponentWiseLoss(
            component_weights=self.component_weights,
            use_contrastive=True,
            use_reconstruction=True
        )
        self.path_importance_loss = PathImportanceLoss(
            temperature=self.temperature,
            use_kl_div=True
        )
        
        # Initialize component disentangler
        self.disentangler = ComponentDisentangler(config)
        
        # Initialize prompt templates
        self.prompt_templates = PromptTemplates(config)
        
        # Distillation statistics
        self.distillation_stats = {
            'total_batches': 0,
            'avg_loss': 0.0,
            'avg_dist_loss': 0.0,
            'avg_contrastive_loss': 0.0,
            'avg_orthogonality_loss': 0.0,
            'alignment_scores': [],
            'distillation_quality': []
        }
        
        # Teacher ensemble (if enabled)
        self.teacher_ensemble = []
        if self.use_teacher_ensemble:
            self._initialize_teacher_ensemble()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to_device(self.device)
        
        self.logger.log_info(
            f"KnowledgeDistiller initialized: temperature={self.temperature}, "
            f"alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}, "
            f"distill_memory={self.distill_memory}"
        )
    
    def _create_teacher(self) -> LLMInterface:
        """Create teacher LLM interface from configuration."""
        return LLMInterface(config=self.config)
    
    def _create_student(self) -> GNNEncoder:
        """Create student GNN encoder from configuration."""
        return GNNEncoder(config=self.config)
    
    def _initialize_teacher_ensemble(self):
        """Initialize multiple teachers for ensemble distillation."""
        # Create multiple teacher instances with different configurations
        for i in range(3):  # Number of teachers in ensemble
            teacher = LLMInterface(config=self.config)
            self.teacher_ensemble.append(teacher)
        
        self.logger.log_info(f"Teacher ensemble initialized with {len(self.teacher_ensemble)} teachers")
    
    def distill_knowledge(
        self,
        batch: Dict[str, Any],
        return_outputs: bool = False
    ) -> Union[Dict[str, float], DistillationOutput]:
        """
        Distill knowledge from teacher to student for a batch.
        
        Args:
            batch: Training batch containing graph, nodes, and metadata.
            return_outputs: Whether to return detailed outputs.
        
        Returns:
            If return_outputs=False: Dict of loss values.
            If return_outputs=True: DistillationOutput object.
        
        Raises:
            ValueError: If batch is missing required fields.
        """
        # Validate batch
        if 'graph' not in batch:
            raise ValueError("Batch must contain 'graph'")
        
        # Extract batch components
        graph = batch.get('graph')
        node_features = batch.get('node_features')
        agents = batch.get('agents', {})
        interactions = batch.get('interactions', [])
        context = batch.get('context', {})
        
        # Get teacher outputs
        teacher_outputs = self._get_teacher_outputs(
            graph=graph,
            agents=agents,
            interactions=interactions,
            context=context
        )
        
        # Get student outputs
        student_outputs = self._get_student_outputs(
            graph=graph,
            node_features=node_features
        )
        
        # Compute losses
        losses = self._compute_losses(
            teacher_outputs=teacher_outputs,
            student_outputs=student_outputs,
            batch=batch
        )
        
        # Update statistics
        self._update_stats(losses)
        
        # Compute alignment score
        alignment_score = self._compute_alignment_score(
            teacher_outputs,
            student_outputs
        )
        
        # Compute distillation quality
        distillation_quality = self._compute_distillation_quality(
            losses=losses,
            alignment_score=alignment_score
        )
        
        # Store in statistics
        self.distillation_stats['alignment_scores'].append(alignment_score)
        self.distillation_stats['distillation_quality'].append(distillation_quality)
        
        # Schedule temperature if enabled
        if self.temperature_scheduling:
            self._schedule_temperature()
        
        if return_outputs:
            return DistillationOutput(
                teacher_outputs=teacher_outputs,
                student_outputs=student_outputs,
                losses=losses,
                alignment_score=alignment_score,
                distillation_quality=distillation_quality,
                metadata={
                    'batch_size': len(batch.get('node_ids', [])),
                    'temperature': self.temperature
                }
            )
        
        return losses
    
    def _get_teacher_outputs(
        self,
        graph: Optional[Any] = None,
        agents: Optional[Dict[str, Any]] = None,
        interactions: Optional[List[Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get outputs from teacher LLM.
        
        Args:
            graph: Optional graph object.
            agents: Optional agent dictionary.
            interactions: Optional interaction list.
            context: Optional context dictionary.
        
        Returns:
            Dict containing teacher outputs.
        """
        teacher_outputs = {}
        
        if self.teacher_llm is None:
            return teacher_outputs
        
        # Use teacher ensemble if enabled
        if self.use_teacher_ensemble and self.teacher_ensemble:
            return self._get_ensemble_teacher_outputs(
                graph, agents, interactions, context
            )
        
        # Get teacher embeddings
        if agents:
            # Extract agent information
            user_agents = [a for a in agents.values() if isinstance(a, UserAgent)]
            item_agents = [a for a in agents.values() if isinstance(a, ItemAgent)]
            
            # Get teacher embeddings for agents
            teacher_embeddings = []
            for agent in user_agents + item_agents:
                # Generate prompt for agent
                prompt = self._generate_agent_prompt(agent, context)
                
                # Get embedding from teacher
                embedding = self.teacher_llm.get_embedding(prompt)
                teacher_embeddings.append(embedding)
            
            if teacher_embeddings:
                teacher_outputs['embeddings'] = torch.stack(teacher_embeddings)
        
        # Get teacher predictions for interactions
        if interactions:
            teacher_predictions = []
            for interaction in interactions:
                # Generate prompt for interaction
                prompt = self._generate_interaction_prompt(interaction, context)
                
                # Get prediction from teacher
                prediction = self.teacher_llm.generate(prompt)
                
                # Convert to tensor
                # This is simplified - in practice, would parse the output
                pred_tensor = torch.tensor([float(prediction) if prediction else 0.0])
                teacher_predictions.append(pred_tensor)
            
            if teacher_predictions:
                teacher_outputs['predictions'] = torch.stack(teacher_predictions)
        
        # Get teacher memory dynamics
        if self.distill_memory and agents:
            memory_outputs = self._get_teacher_memory_outputs(agents)
            teacher_outputs['memory'] = memory_outputs
        
        return teacher_outputs
    
    def _get_ensemble_teacher_outputs(
        self,
        graph: Optional[Any] = None,
        agents: Optional[Dict[str, Any]] = None,
        interactions: Optional[List[Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get outputs from teacher ensemble.
        
        Args:
            graph: Optional graph object.
            agents: Optional agent dictionary.
            interactions: Optional interaction list.
            context: Optional context dictionary.
        
        Returns:
            Aggregated teacher outputs.
        """
        all_outputs = []
        
        for teacher in self.teacher_ensemble:
            # Temporarily set as teacher
            original_teacher = self.teacher_llm
            self.teacher_llm = teacher
            
            outputs = self._get_teacher_outputs(graph, agents, interactions, context)
            all_outputs.append(outputs)
            
            # Restore original teacher
            self.teacher_llm = original_teacher
        
        # Aggregate outputs
        aggregated_outputs = {}
        
        for key in all_outputs[0].keys():
            if key == 'embeddings':
                # Average embeddings
                embeds = [out[key] for out in all_outputs if key in out]
                if embeds:
                    aggregated_outputs[key] = torch.stack(embeds).mean(dim=0)
            elif key == 'predictions':
                # Average predictions
                preds = [out[key] for out in all_outputs if key in out]
                if preds:
                    aggregated_outputs[key] = torch.stack(preds).mean(dim=0)
            elif key == 'memory':
                # Average memory outputs
                memory_outputs = [out[key] for out in all_outputs if key in out]
                if memory_outputs:
                    aggregated_outputs[key] = self._aggregate_memory_outputs(memory_outputs)
            else:
                # Use first available
                for out in all_outputs:
                    if key in out:
                        aggregated_outputs[key] = out[key]
                        break
        
        return aggregated_outputs
    
    def _get_student_outputs(
        self,
        graph: Any,
        node_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Get outputs from student GNN.
        
        Args:
            graph: Graph object.
            node_features: Optional node features.
        
        Returns:
            Dict containing student outputs.
        """
        student_outputs = {}
        
        if self.student_gnn is None:
            return student_outputs
        
        # Forward pass through GNN
        try:
            # Encode graph
            embeddings = self.student_gnn.encode_graph(graph, node_features)
            student_outputs['embeddings'] = embeddings
            
            # Get component projections if available
            if hasattr(self.student_gnn, 'projection_heads'):
                components = {}
                for node_type, emb in embeddings.items():
                    comps = self.student_gnn.projection_heads.project_all(emb)
                    components[node_type] = comps
                student_outputs['components'] = components
            
            # Get graph-level embedding
            graph_embedding = self.student_gnn.get_graph_embedding(graph, node_features)
            student_outputs['graph_embedding'] = graph_embedding
            
        except Exception as e:
            self.logger.log_error(f"Student forward pass failed: {e}")
            student_outputs['embeddings'] = {}
        
        return student_outputs
    
    def _compute_losses(
        self,
        teacher_outputs: Dict[str, Any],
        student_outputs: Dict[str, Any],
        batch: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute distillation losses.
        
        Args:
            teacher_outputs: Teacher outputs.
            student_outputs: Student outputs.
            batch: Training batch.
        
        Returns:
            Dict of loss values.
        """
        losses = {}
        total_loss = 0.0
        
        # 1. Embedding alignment loss
        if 'embeddings' in teacher_outputs and 'embeddings' in student_outputs:
            # Convert student embeddings to tensor
            student_emb = self._flatten_embeddings(student_outputs['embeddings'])
            teacher_emb = self._flatten_embeddings(teacher_outputs['embeddings'])
            
            if student_emb is not None and teacher_emb is not None:
                # Ensure same size
                if student_emb.size(0) != teacher_emb.size(0):
                    min_size = min(student_emb.size(0), teacher_emb.size(0))
                    student_emb = student_emb[:min_size]
                    teacher_emb = teacher_emb[:min_size]
                
                align_loss = self.distillation_loss._component_loss(
                    student_emb,
                    teacher_emb
                )
                losses['alignment'] = self.alpha * align_loss.item()
                total_loss += self.alpha * align_loss
        
        # 2. Component-wise loss
        if 'components' in student_outputs and 'components' in teacher_outputs:
            # This would require matching components from teacher
            # Simplified version
            comp_loss = self.component_wise_loss(
                student_outputs['components'],
                teacher_outputs.get('components', {})
            )
            losses['component'] = self.beta * comp_loss['total'].item()
            total_loss += self.beta * comp_loss['total']
        
        # 3. Path importance loss
        if 'attention' in student_outputs and 'attention' in teacher_outputs:
            path_loss = self.path_importance_loss(
                student_outputs['attention'],
                teacher_outputs['attention']
            )
            losses['path'] = self.gamma * path_loss['total'].item()
            total_loss += self.gamma * path_loss['total']
        
        # 4. Memory dynamics loss
        if self.distill_memory and 'memory' in teacher_outputs:
            memory_loss = self._compute_memory_loss(
                teacher_outputs['memory'],
                student_outputs.get('memory', {})
            )
            if memory_loss is not None:
                losses['memory'] = 0.1 * memory_loss.item()
                total_loss += 0.1 * memory_loss
        
        # 5. Orthogonality loss
        if 'components' in student_outputs:
            for node_type, comps in student_outputs['components'].items():
                ortho_loss = self.distillation_loss.orthogonality_loss(comps)
                losses[f'orthogonality_{node_type}'] = 0.05 * ortho_loss.item()
                total_loss += 0.05 * ortho_loss
        
        # Store total loss
        losses['total'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return losses
    
    def _flatten_embeddings(
        self,
        embeddings: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Flatten dict of embeddings to single tensor.
        
        Args:
            embeddings: Dict mapping node types to embeddings or tensor.
        
        Returns:
            Flattened tensor or None.
        """
        if isinstance(embeddings, torch.Tensor):
            return embeddings
        
        if isinstance(embeddings, dict):
            all_emb = []
            for node_type, emb in embeddings.items():
                if isinstance(emb, torch.Tensor):
                    all_emb.append(emb)
            
            if all_emb:
                return torch.cat(all_emb, dim=0)
        
        return None
    
    def _generate_agent_prompt(
        self,
        agent: Any,
        context: Dict[str, Any]
    ) -> str:
        """
        Generate prompt for agent knowledge extraction.
        
        Args:
            agent: Agent object.
            context: Context dictionary.
        
        Returns:
            Prompt string.
        """
        if isinstance(agent, UserAgent):
            return self.prompt_templates.get_user_agent_prompt(agent, context)
        elif isinstance(agent, ItemAgent):
            return self.prompt_templates.get_item_agent_prompt(agent, context)
        else:
            return str(agent)
    
    def _generate_interaction_prompt(
        self,
        interaction: Any,
        context: Dict[str, Any]
    ) -> str:
        """
        Generate prompt for interaction knowledge extraction.
        
        Args:
            interaction: Interaction object.
            context: Context dictionary.
        
        Returns:
            Prompt string.
        """
        return self.prompt_templates.get_interaction_prompt(interaction, context)
    
    def _get_teacher_memory_outputs(
        self,
        agents: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get memory dynamics from teacher.
        
        Args:
            agents: Agent dictionary.
        
        Returns:
            Memory outputs.
        """
        memory_outputs = {}
        
        for agent_id, agent in agents.items():
            if hasattr(agent, 'get_intrinsic_memory'):
                memory_outputs[f'{agent_id}_intrinsic'] = agent.get_intrinsic_memory()
            if hasattr(agent, 'get_collaborative_memory'):
                memory_outputs[f'{agent_id}_collaborative'] = agent.get_collaborative_memory()
            if hasattr(agent, 'get_interaction_memory'):
                memory_outputs[f'{agent_id}_interaction'] = agent.get_interaction_memory()
        
        return memory_outputs
    
    def _compute_memory_loss(
        self,
        teacher_memory: Dict[str, Any],
        student_memory: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        """
        Compute memory dynamics loss.
        
        Args:
            teacher_memory: Teacher memory outputs.
            student_memory: Student memory outputs.
        
        Returns:
            Memory loss tensor or None.
        """
        if not teacher_memory or not student_memory:
            return None
        
        loss = 0.0
        count = 0
        
        for key, teacher_value in teacher_memory.items():
            if key in student_memory:
                student_value = student_memory[key]
                
                # Convert to tensors if needed
                if isinstance(teacher_value, (int, float)):
                    teacher_value = torch.tensor([teacher_value])
                if isinstance(student_value, (int, float)):
                    student_value = torch.tensor([student_value])
                
                # Compute loss
                if isinstance(teacher_value, torch.Tensor) and isinstance(student_value, torch.Tensor):
                    if teacher_value.shape == student_value.shape:
                        loss += F.mse_loss(student_value, teacher_value)
                        count += 1
        
        if count > 0:
            return loss / count
        
        return None
    
    def _aggregate_memory_outputs(
        self,
        memory_outputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate memory outputs from multiple teachers.
        
        Args:
            memory_outputs: List of memory output dicts.
        
        Returns:
            Aggregated memory outputs.
        """
        aggregated = {}
        
        # Collect all keys
        all_keys = set()
        for mem in memory_outputs:
            all_keys.update(mem.keys())
        
        for key in all_keys:
            values = []
            for mem in memory_outputs:
                if key in mem:
                    values.append(mem[key])
            
            if values:
                if isinstance(values[0], torch.Tensor):
                    aggregated[key] = torch.stack(values).mean(dim=0)
                elif isinstance(values[0], (int, float)):
                    aggregated[key] = np.mean(values)
                else:
                    aggregated[key] = values[0]
        
        return aggregated
    
    def _compute_alignment_score(
        self,
        teacher_outputs: Dict[str, Any],
        student_outputs: Dict[str, Any]
    ) -> float:
        """
        Compute representation alignment score.
        
        Args:
            teacher_outputs: Teacher outputs.
            student_outputs: Student outputs.
        
        Returns:
            Alignment score between 0 and 1.
        """
        if not teacher_outputs or not student_outputs:
            return 0.0
        
        # Get embeddings
        teacher_emb = self._flatten_embeddings(teacher_outputs.get('embeddings', {}))
        student_emb = self._flatten_embeddings(student_outputs.get('embeddings', {}))
        
        if teacher_emb is None or student_emb is None:
            return 0.0
        
        # Ensure same size
        if teacher_emb.size(0) != student_emb.size(0):
            min_size = min(teacher_emb.size(0), student_emb.size(0))
            teacher_emb = teacher_emb[:min_size]
            student_emb = student_emb[:min_size]
        
        # Compute cosine similarity
        teacher_norm = F.normalize(teacher_emb, p=2, dim=-1)
        student_norm = F.normalize(student_emb, p=2, dim=-1)
        
        similarity = F.cosine_similarity(teacher_norm, student_norm, dim=-1)
        alignment_score = similarity.mean().item()
        
        # Normalize to [0, 1]
        alignment_score = (alignment_score + 1) / 2
        
        return alignment_score
    
    def _compute_distillation_quality(
        self,
        losses: Dict[str, float],
        alignment_score: float
    ) -> float:
        """
        Compute overall distillation quality score.
        
        Args:
            losses: Loss values.
            alignment_score: Alignment score.
        
        Returns:
            Distillation quality score between 0 and 1.
        """
        # Lower loss is better
        total_loss = losses.get('total', 1.0)
        loss_quality = 1.0 / (1.0 + total_loss)
        
        # Combine with alignment score
        quality = 0.6 * loss_quality + 0.4 * alignment_score
        
        return min(quality, 1.0)
    
    def _schedule_temperature(self):
        """
        Schedule temperature for distillation.
        """
        # Increase temperature gradually
        self.temperature = self.temperature * 1.01
        self.temperature = min(self.temperature, 0.5)
    
    def _update_stats(self, losses: Dict[str, float]):
        """
        Update distillation statistics.
        
        Args:
            losses: Loss values.
        """
        self.distillation_stats['total_batches'] += 1
        
        # Exponential moving average
        alpha = 0.1
        
        self.distillation_stats['avg_loss'] = (
            (1 - alpha) * self.distillation_stats['avg_loss'] +
            alpha * losses.get('total', 0.0)
        )
        self.distillation_stats['avg_dist_loss'] = (
            (1 - alpha) * self.distillation_stats['avg_dist_loss'] +
            alpha * losses.get('alignment', 0.0)
        )
        self.distillation_stats['avg_contrastive_loss'] = (
            (1 - alpha) * self.distillation_stats['avg_contrastive_loss'] +
            alpha * losses.get('contrastive', 0.0)
        )
        self.distillation_stats['avg_orthogonality_loss'] = (
            (1 - alpha) * self.distillation_stats['avg_orthogonality_loss'] +
            alpha * losses.get('orthogonality', 0.0)
        )
    
    def align_representations(
        self,
        teacher_rep: torch.Tensor,
        student_rep: torch.Tensor,
        use_projection: bool = True
    ) -> torch.Tensor:
        """
        Align teacher and student representations.
        
        Args:
            teacher_rep: Teacher representation tensor.
            student_rep: Student representation tensor.
            use_projection: Whether to use projection for alignment.
        
        Returns:
            Aligned student representation.
        """
        if teacher_rep is None or student_rep is None:
            return student_rep
        
        if use_projection:
            # Learnable projection for alignment
            if not hasattr(self, 'alignment_projection'):
                self.alignment_projection = nn.Linear(
                    student_rep.size(-1),
                    teacher_rep.size(-1)
                ).to(self.device)
            
            student_aligned = self.alignment_projection(student_rep)
        else:
            # Simple alignment: match dimensions
            if student_rep.size(-1) != teacher_rep.size(-1):
                # Project to same dimension
                proj = nn.Linear(
                    student_rep.size(-1),
                    teacher_rep.size(-1)
                ).to(self.device)
                student_aligned = proj(student_rep)
            else:
                student_aligned = student_rep
        
        return student_aligned
    
    def transfer_knowledge(
        self,
        teacher_outputs: Dict[str, Any],
        student: GNNEncoder,
        batch: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Transfer knowledge from teacher to student.
        
        Args:
            teacher_outputs: Teacher outputs.
            student: Student GNN encoder.
            batch: Training batch.
        
        Returns:
            Transfer loss values.
        """
        # Get student outputs
        graph = batch.get('graph')
        node_features = batch.get('node_features')
        
        student_outputs = self._get_student_outputs(graph, node_features)
        
        # Compute losses
        losses = self._compute_losses(teacher_outputs, student_outputs, batch)
        
        return losses
    
    def distill_memory_dynamics(
        self,
        teacher_memory: AgentMemory,
        student_memory: AgentMemory,
        batch: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Distill memory dynamics from teacher to student.
        
        Args:
            teacher_memory: Teacher agent memory.
            student_memory: Student agent memory.
            batch: Training batch.
        
        Returns:
            Memory distillation losses.
        """
        # Extract memory components
        teacher_intrinsic = teacher_memory.get_intrinsic_memory()
        teacher_collaborative = teacher_memory.get_collaborative_memory()
        teacher_interaction = teacher_memory.get_interaction_memory()
        
        student_intrinsic = student_memory.get_intrinsic_memory()
        student_collaborative = student_memory.get_collaborative_memory()
        student_interaction = student_memory.get_interaction_memory()
        
        # Compute losses for each component
        losses = {}
        
        if teacher_intrinsic is not None and student_intrinsic is not None:
            losses['intrinsic'] = self.distillation_loss.reconstruction_loss(
                student_intrinsic,
                teacher_intrinsic
            ).item()
        
        if teacher_collaborative is not None and student_collaborative is not None:
            losses['collaborative'] = self.distillation_loss.reconstruction_loss(
                student_collaborative,
                teacher_collaborative
            ).item()
        
        if teacher_interaction is not None and student_interaction is not None:
            losses['interaction'] = self.distillation_loss.reconstruction_loss(
                student_interaction,
                teacher_interaction
            ).item()
        
        # Memory consistency loss
        losses['consistency'] = self.distillation_loss.memory_consistency_loss(
            {'initial': teacher_intrinsic},
            {'current': student_intrinsic}
        ).item() if teacher_intrinsic is not None else 0.0
        
        losses['total'] = sum(losses.values())
        
        return losses
    
    def get_distillation_stats(self) -> Dict[str, Any]:
        """
        Get distillation statistics.
        
        Returns:
            Dict containing distillation statistics.
        """
        return {
            'total_batches': self.distillation_stats['total_batches'],
            'avg_loss': self.distillation_stats['avg_loss'],
            'avg_dist_loss': self.distillation_stats['avg_dist_loss'],
            'avg_contrastive_loss': self.distillation_stats['avg_contrastive_loss'],
            'avg_orthogonality_loss': self.distillation_stats['avg_orthogonality_loss'],
            'avg_alignment_score': np.mean(self.distillation_stats['alignment_scores']) if self.distillation_stats['alignment_scores'] else 0.0,
            'avg_distillation_quality': np.mean(self.distillation_stats['distillation_quality']) if self.distillation_stats['distillation_quality'] else 0.0,
            'temperature': self.temperature,
            'use_teacher_ensemble': self.use_teacher_ensemble,
            'num_teachers': len(self.teacher_ensemble)
        }
    
    def reset_stats(self):
        """Reset distillation statistics."""
        self.distillation_stats = {
            'total_batches': 0,
            'avg_loss': 0.0,
            'avg_dist_loss': 0.0,
            'avg_contrastive_loss': 0.0,
            'avg_orthogonality_loss': 0.0,
            'alignment_scores': [],
            'distillation_quality': []
        }
        self.logger.log_info("Distillation statistics reset")
    
    def set_teacher(self, teacher: LLMInterface):
        """
        Set teacher model.
        
        Args:
            teacher: Teacher LLM interface.
        """
        self.teacher_llm = teacher
        self.logger.log_info("Teacher model updated")
    
    def set_student(self, student: GNNEncoder):
        """
        Set student model.
        
        Args:
            student: Student GNN encoder.
        """
        self.student_gnn = student
        self.logger.log_info("Student model updated")
    
    def set_temperature(self, temperature: float):
        """
        Set distillation temperature.
        
        Args:
            temperature: New temperature value.
        """
        self.temperature = temperature
        self.logger.log_info(f"Temperature set to {temperature}")
    
    def to_device(self, device: torch.device) -> 'KnowledgeDistiller':
        """
        Move all components to specified device.
        
        Args:
            device: PyTorch device.
        
        Returns:
            Self with components moved to device.
        """
        self.device = device
        
        if self.student_gnn:
            self.student_gnn.to_device(device)
        if self.teacher_llm and hasattr(self.teacher_llm, 'to_device'):
            self.teacher_llm.to_device(device)
        
        self.to(device)
        self.logger.log_info(f"KnowledgeDistiller moved to device: {device}")
        
        return self
    
    def save_distiller(self, save_path: str):
        """
        Save knowledge distiller state.
        
        Args:
            save_path: Path to save the distiller.
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            state_dict = {
                'student_state': self.student_gnn.state_dict() if self.student_gnn else None,
                'config': self.config,
                'temperature': self.temperature,
                'distillation_stats': self.distillation_stats,
                'version': __version__
            }
            
            torch.save(state_dict, save_path)
            self.logger.log_info(f"Distiller saved to {save_path}")
        
        except Exception as e:
            self.logger.log_error(f"Failed to save distiller: {e}")
            raise
    
    def load_distiller(self, load_path: str):
        """
        Load knowledge distiller state.
        
        Args:
            load_path: Path to load the distiller from.
        
        Raises:
            FileNotFoundError: If checkpoint not found.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Distiller checkpoint not found: {load_path}")
        
        try:
            checkpoint = torch.load(load_path, map_location=self.device)
            
            if checkpoint.get('student_state') is not None and self.student_gnn:
                self.student_gnn.load_state_dict(checkpoint['student_state'])
            
            self.temperature = checkpoint.get('temperature', self.temperature)
            self.distillation_stats = checkpoint.get('distillation_stats', self.distillation_stats)
            
            if 'config' in checkpoint:
                self.config = checkpoint['config']
            
            self.logger.log_info(f"Distiller loaded from {load_path}")
        
        except Exception as e:
            self.logger.log_error(f"Failed to load distiller: {e}")
            raise
    
    def forward(
        self,
        batch: Dict[str, Any],
        return_outputs: bool = False
    ) -> Union[Dict[str, float], DistillationOutput]:
        """
        Forward pass for knowledge distillation.
        
        Args:
            batch: Training batch.
            return_outputs: Whether to return detailed outputs.
        
        Returns:
            Loss values or DistillationOutput object.
        """
        return self.distill_knowledge(batch, return_outputs)


# Module level variables and exports
__all__ = [
    'DistillationOutput',
    'KnowledgeDistiller',
    '__doc__'
]

# Version information
__version__ = '1.0.0'
__author__ = 'H-GRAGrecsys Team'


def create_knowledge_distiller(
    teacher_llm: Optional[LLMInterface] = None,
    student_gnn: Optional[GNNEncoder] = None,
    config_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> KnowledgeDistiller:
    """
    Factory function to create a KnowledgeDistiller instance.
    
    Args:
        teacher_llm: Optional teacher LLM interface.
        student_gnn: Optional student GNN encoder.
        config_path: Optional path to configuration file.
        device: Optional device to move distiller to.
    
    Returns:
        Initialized KnowledgeDistiller instance.
    
    Example:
        >>> distiller = create_knowledge_distiller(
        ...     teacher_llm=llm_teacher,
        ...     student_gnn=gnn_student,
        ...     config_path='config/default_config.yaml'
        ... )
        >>> losses = distiller.distill_knowledge(batch)
    """
    distiller = KnowledgeDistiller(
        teacher_llm=teacher_llm,
        student_gnn=student_gnn,
        config=config_path
    )
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return distiller.to_device(device)


def create_distillation_output(
    teacher_outputs: Dict[str, Any],
    student_outputs: Dict[str, Any],
    losses: Dict[str, float],
    alignment_score: float = 0.0,
    distillation_quality: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None
) -> DistillationOutput:
    """
    Factory function to create a DistillationOutput object.
    
    Args:
        teacher_outputs: Teacher outputs.
        student_outputs: Student outputs.
        losses: Loss values.
        alignment_score: Alignment score.
        distillation_quality: Distillation quality.
        metadata: Optional metadata.
    
    Returns:
        DistillationOutput object.
    
    Example:
        >>> output = create_distillation_output(
        ...     teacher_outputs=teacher_out,
        ...     student_outputs=student_out,
        ...     losses={'total': 0.5},
        ...     alignment_score=0.8
        ... )
    """
    return DistillationOutput(
        teacher_outputs=teacher_outputs,
        student_outputs=student_outputs,
        losses=losses,
        alignment_score=alignment_score,
        distillation_quality=distillation_quality,
        metadata=metadata or {}
    )