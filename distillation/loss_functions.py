"""
Loss Functions Module for H-GRAGrecsys Distillation

This module implements various loss functions for knowledge distillation
in H-GRAGrecsys, including:
- Component-wise distillation loss for disentangled representations
- Path importance loss for metapath-based reasoning
- Contrastive loss for representation alignment
- Reconstruction loss for memory dynamics
- Orthogonality loss for disentanglement
- Combined distillation loss with adaptive weighting

The loss functions support both teacher-student distillation and
self-supervised learning objectives.
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from utils
from utils.config_loader import ConfigLoader
from utils.logger import Logger
from utils.seed_manager import SeedManager


class DistillationLoss(nn.Module):
    """
    Main distillation loss class combining multiple loss functions.
    
    This class provides a comprehensive set of loss functions for knowledge
    distillation, including component-wise, contrastive, and reconstruction losses.
    """
    
    def __init__(self, config: Optional[Union[str, Dict, ConfigLoader]] = None):
        """
        Initialize distillation loss functions.
        
        Args:
            config: Configuration object or path to config file.
                   Can be a string path, dict, or ConfigLoader instance.
        
        Raises:
            ValueError: If config is invalid or missing required fields.
        """
        super(DistillationLoss, self).__init__()
        
        # Load configuration
        if config is None:
            self.config = {
                'model': {
                    'distillation': {
                        'component_weights': [1.0, 1.0, 1.0],
                        'temperature': 0.07,
                        'alpha': 0.5,
                        'beta': 0.3,
                        'gamma': 0.2
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
            name='distillation_loss'
        )
        
        # Extract configuration
        dist_config = self.config.get('model', {}).get('distillation', {})
        
        self.component_weights = dist_config.get('component_weights', [1.0, 1.0, 1.0])
        self.temperature = dist_config.get('temperature', 0.07)
        self.alpha = dist_config.get('alpha', 0.5)
        self.beta = dist_config.get('beta', 0.3)
        self.gamma = dist_config.get('gamma', 0.2)
        
        # Smooth L1 loss for robust regression
        self.smooth_l1 = nn.SmoothL1Loss()
        
        # KL Divergence for distillation
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
        # Cross entropy for classification
        self.cross_entropy = nn.CrossEntropyLoss()
        
        self.logger.log_info(
            f"DistillationLoss initialized: temperature={self.temperature}, "
            f"alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}"
        )
    
    def component_wise_loss(
        self,
        student: Dict[str, torch.Tensor],
        teacher: Dict[str, torch.Tensor],
        component_type: str = 'all'
    ) -> torch.Tensor:
        """
        Compute component-wise distillation loss.
        
        Args:
            student: Student component embeddings.
            teacher: Teacher component embeddings.
            component_type: Component type ('intrinsic', 'collaborative', 
                           'interaction', or 'all').
        
        Returns:
            Component-wise loss tensor.
        
        Raises:
            ValueError: If component_type is invalid.
        """
        valid_components = ['intrinsic', 'collaborative', 'interaction', 'all']
        if component_type not in valid_components:
            raise ValueError(
                f"Invalid component_type: {component_type}. "
                f"Must be one of: {valid_components}"
            )
        
        if component_type == 'all':
            # Compute loss for all components
            total_loss = 0.0
            for comp in ['intrinsic', 'collaborative', 'interaction']:
                if comp in student and comp in teacher:
                    loss = self._component_loss(student[comp], teacher[comp])
                    weight = self.component_weights[
                        ['intrinsic', 'collaborative', 'interaction'].index(comp)
                    ]
                    total_loss += weight * loss
            
            return total_loss / 3.0
        
        else:
            # Compute loss for specific component
            if component_type not in student or component_type not in teacher:
                return torch.tensor(0.0, device=student[component_type].device)
            
            return self._component_loss(student[component_type], teacher[component_type])
    
    def _component_loss(
        self,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss between student and teacher embeddings.
        
        Args:
            student_emb: Student embeddings.
            teacher_emb: Teacher embeddings.
        
        Returns:
            Loss tensor.
        """
        # Normalize embeddings
        student_norm = F.normalize(student_emb, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_emb, p=2, dim=-1)
        
        # MSE loss
        mse_loss = F.mse_loss(student_norm, teacher_norm)
        
        # Cosine similarity loss
        cos_sim = F.cosine_similarity(student_norm, teacher_norm, dim=-1)
        cos_loss = (1 - cos_sim).mean()
        
        # Combined loss
        loss = 0.5 * mse_loss + 0.5 * cos_loss
        
        return loss
    
    def path_importance_loss(
        self,
        student_attn: torch.Tensor,
        teacher_attn: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute path importance distillation loss.
        
        Args:
            student_attn: Student attention weights (batch_size, num_paths).
            teacher_attn: Teacher attention weights (batch_size, num_paths).
            temperature: Optional temperature for softmax. If None, uses default.
        
        Returns:
            Path importance loss tensor.
        
        Raises:
            ValueError: If attention tensors have different shapes.
        """
        if student_attn.shape != teacher_attn.shape:
            raise ValueError(
                f"Attention tensors must have same shape: "
                f"student {student_attn.shape} vs teacher {teacher_attn.shape}"
            )
        
        temp = temperature if temperature is not None else self.temperature
        
        # Apply temperature scaling
        student_soft = F.softmax(student_attn / temp, dim=-1)
        teacher_soft = F.softmax(teacher_attn / temp, dim=-1)
        
        # KL divergence
        kl_loss = self.kl_div(
            torch.log(student_soft + 1e-9),
            teacher_soft
        )
        
        # Cross entropy
        ce_loss = -torch.sum(teacher_soft * torch.log(student_soft + 1e-9)) / student_attn.size(0)
        
        # Combined loss
        loss = 0.5 * kl_loss + 0.5 * ce_loss
        
        return loss
    
    def contrastive_loss(
        self,
        embeddings: Dict[str, torch.Tensor],
        temperature: Optional[float] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss for representation alignment.
        
        Args:
            embeddings: Dict mapping node types to embeddings.
            temperature: Optional temperature. If None, uses default.
            labels: Optional labels for supervised contrastive learning.
        
        Returns:
            Contrastive loss tensor.
        """
        temp = temperature if temperature is not None else self.temperature
        
        # Concatenate all embeddings
        all_embeddings = []
        for node_type, emb in embeddings.items():
            all_embeddings.append(emb)
        
        if not all_embeddings:
            return torch.tensor(0.0)
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Normalize embeddings
        all_embeddings = F.normalize(all_embeddings, p=2, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(all_embeddings, all_embeddings.T) / temp
        
        # If labels provided, use supervised contrastive loss
        if labels is not None:
            return self._supervised_contrastive_loss(sim_matrix, labels)
        
        # Unsupervised contrastive loss
        # Positive pairs: same node type
        # Negative pairs: different node types
        batch_size = all_embeddings.size(0)
        
        # Create mask for positive pairs (same node type)
        # This is simplified - in practice, would use node type information
        pos_mask = torch.eye(batch_size, device=all_embeddings.device)
        
        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        
        # Numerator: sum of exp(sim) for positive pairs
        pos_sum = torch.sum(exp_sim * pos_mask, dim=1)
        
        # Denominator: sum of exp(sim) for all pairs
        neg_sum = torch.sum(exp_sim, dim=1) - exp_sim.diag()
        denom = pos_sum + neg_sum
        
        # Loss
        loss = -torch.log(pos_sum / (denom + 1e-9)).mean()
        
        return loss
    
    def _supervised_contrastive_loss(
        self,
        sim_matrix: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            sim_matrix: Similarity matrix.
            labels: Label tensor.
        
        Returns:
            Supervised contrastive loss.
        """
        batch_size = sim_matrix.size(0)
        
        # Create mask for positive pairs (same labels)
        pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        pos_mask = pos_mask.float()
        
        # Remove self-pairs
        pos_mask = pos_mask - torch.eye(batch_size, device=sim_matrix.device)
        
        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        
        # Numerator: sum of exp(sim) for positive pairs
        pos_sum = torch.sum(exp_sim * pos_mask, dim=1)
        
        # Denominator: sum of exp(sim) for all pairs
        denom = torch.sum(exp_sim, dim=1) - exp_sim.diag()
        
        # Loss
        loss = -torch.log(pos_sum / (denom + 1e-9)).mean()
        
        return loss
    
    def reconstruction_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        component_type: str = 'all'
    ) -> torch.Tensor:
        """
        Compute reconstruction loss for memory dynamics.
        
        Args:
            predicted: Predicted tensor.
            target: Target tensor.
            component_type: Component type for loss weighting.
        
        Returns:
            Reconstruction loss tensor.
        """
        # MSE loss
        mse_loss = F.mse_loss(predicted, target)
        
        # L1 loss
        l1_loss = F.l1_loss(predicted, target)
        
        # Smooth L1 loss
        smooth_loss = self.smooth_l1(predicted, target)
        
        # Combined loss
        loss = 0.4 * mse_loss + 0.3 * l1_loss + 0.3 * smooth_loss
        
        return loss
    
    def orthogonality_loss(
        self,
        components: Dict[str, torch.Tensor],
        enforce_orthogonal: bool = True
    ) -> torch.Tensor:
        """
        Compute orthogonality loss for disentangled representations.
        
        Args:
            components: Dict mapping component types to tensors.
            enforce_orthogonal: Whether to enforce orthogonality.
        
        Returns:
            Orthogonality loss tensor.
        
        Raises:
            ValueError: If components dict is missing required keys.
        """
        required_keys = ['intrinsic', 'collaborative', 'interaction']
        for key in required_keys:
            if key not in components:
                raise ValueError(f"Missing component: {key}")
        
        # Extract components
        intrinsic = components['intrinsic']
        collaborative = components['collaborative']
        interaction = components['interaction']
        
        # Normalize components
        intrinsic_norm = F.normalize(intrinsic, p=2, dim=-1)
        collaborative_norm = F.normalize(collaborative, p=2, dim=-1)
        interaction_norm = F.normalize(interaction, p=2, dim=-1)
        
        # Compute pairwise dot products
        loss = 0.0
        
        pairs = [
            (intrinsic_norm, collaborative_norm),
            (intrinsic_norm, interaction_norm),
            (collaborative_norm, interaction_norm)
        ]
        
        for comp1, comp2 in pairs:
            dot_product = (comp1 * comp2).sum(dim=-1)
            loss += (dot_product ** 2).mean()
        
        # Average over pairs
        loss = loss / 3.0
        
        return loss
    
    def compute_total_loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute total distillation loss.
        
        Args:
            teacher_logits: Teacher model logits.
            student_logits: Student model logits.
            labels: Optional ground truth labels.
            temperature: Optional temperature. If None, uses default.
        
        Returns:
            Total loss tensor.
        """
        temp = temperature if temperature is not None else self.temperature
        
        # Apply temperature scaling
        teacher_soft = F.softmax(teacher_logits / temp, dim=-1)
        student_soft = F.log_softmax(student_logits / temp, dim=-1)
        
        # Distillation loss (KL divergence)
        dist_loss = self.kl_div(student_soft, teacher_soft) * (temp ** 2)
        
        # Student loss (if labels provided)
        student_loss = 0.0
        if labels is not None:
            student_loss = self.cross_entropy(student_logits, labels)
        
        # Combine losses
        total_loss = self.alpha * dist_loss + (1 - self.alpha) * student_loss
        
        return total_loss
    
    def memory_consistency_loss(
        self,
        initial_memory: Dict[str, torch.Tensor],
        current_memory: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute memory consistency loss.
        
        Args:
            initial_memory: Initial memory state.
            current_memory: Current memory state.
        
        Returns:
            Memory consistency loss tensor.
        """
        loss = 0.0
        count = 0
        
        for key in initial_memory.keys():
            if key in current_memory:
                # Compute consistency loss
                init = initial_memory[key]
                curr = current_memory[key]
                
                # MSE loss
                mse = F.mse_loss(init, curr)
                loss += mse
                count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss
    
    def knowledge_transfer_loss(
        self,
        teacher_outputs: Dict[str, torch.Tensor],
        student_outputs: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Compute knowledge transfer loss from teacher to student.
        
        Args:
            teacher_outputs: Teacher output dictionary.
            student_outputs: Student output dictionary.
            weights: Optional weights for each output type.
        
        Returns:
            Knowledge transfer loss tensor.
        """
        if weights is None:
            weights = {
                'logits': 0.5,
                'embeddings': 0.3,
                'attention': 0.2
            }
        
        total_loss = 0.0
        
        for key, weight in weights.items():
            if key in teacher_outputs and key in student_outputs:
                if key == 'logits':
                    loss = self.compute_total_loss(
                        teacher_outputs[key],
                        student_outputs[key]
                    )
                elif key == 'embeddings':
                    loss = self._component_loss(
                        student_outputs[key],
                        teacher_outputs[key]
                    )
                elif key == 'attention':
                    loss = self.path_importance_loss(
                        student_outputs[key],
                        teacher_outputs[key]
                    )
                else:
                    loss = F.mse_loss(student_outputs[key], teacher_outputs[key])
                
                total_loss += weight * loss
        
        return total_loss
    
    def get_loss_components(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get individual loss components for monitoring.
        
        Args:
            teacher_logits: Teacher model logits.
            student_logits: Student model logits.
            labels: Optional ground truth labels.
        
        Returns:
            Dict mapping loss names to loss values.
        """
        temp = self.temperature
        
        # Distillation loss
        teacher_soft = F.softmax(teacher_logits / temp, dim=-1)
        student_soft = F.log_softmax(student_logits / temp, dim=-1)
        dist_loss = self.kl_div(student_soft, teacher_soft) * (temp ** 2)
        
        # Student loss
        student_loss = 0.0
        if labels is not None:
            student_loss = self.cross_entropy(student_logits, labels)
        
        # Combined loss
        total_loss = self.alpha * dist_loss + (1 - self.alpha) * student_loss
        
        return {
            'distillation_loss': dist_loss,
            'student_loss': student_loss,
            'total_loss': total_loss,
            'temperature': torch.tensor(temp)
        }


class ComponentWiseLoss(nn.Module):
    """
    Component-wise loss for disentangled representation learning.
    
    This class implements loss functions for each component (intrinsic,
    collaborative, interaction) with separate weighting.
    """
    
    def __init__(
        self,
        component_weights: Optional[List[float]] = None,
        use_contrastive: bool = True,
        use_reconstruction: bool = True
    ):
        """
        Initialize component-wise loss.
        
        Args:
            component_weights: Weights for each component [intrinsic, collaborative, interaction].
            use_contrastive: Whether to use contrastive loss.
            use_reconstruction: Whether to use reconstruction loss.
        """
        super(ComponentWiseLoss, self).__init__()
        
        self.component_weights = component_weights or [1.0, 1.0, 1.0]
        self.use_contrastive = use_contrastive
        self.use_reconstruction = use_reconstruction
        
        self.distillation_loss = DistillationLoss()
        
        self.logger = Logger(
            log_dir='./logs',
            name='component_wise_loss'
        )
    
    def forward(
        self,
        student_components: Dict[str, torch.Tensor],
        teacher_components: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for component-wise loss.
        
        Args:
            student_components: Student component embeddings.
            teacher_components: Teacher component embeddings.
            labels: Optional labels for supervised learning.
        
        Returns:
            Dict mapping component names to losses.
        """
        losses = {}
        total_loss = 0.0
        
        # Compute loss for each component
        for idx, component in enumerate(['intrinsic', 'collaborative', 'interaction']):
            if component in student_components and component in teacher_components:
                loss = self.distillation_loss.component_wise_loss(
                    student={component: student_components[component]},
                    teacher={component: teacher_components[component]},
                    component_type=component
                )
                
                weight = self.component_weights[idx]
                weighted_loss = weight * loss
                
                losses[component] = weighted_loss
                total_loss += weighted_loss
        
        # Add contrastive loss if enabled
        if self.use_contrastive and labels is not None:
            contrastive_loss = self.distillation_loss.contrastive_loss(
                student_components,
                labels=labels
            )
            losses['contrastive'] = contrastive_loss
            total_loss += contrastive_loss
        
        # Add reconstruction loss if enabled
        if self.use_reconstruction:
            for component in ['intrinsic', 'collaborative', 'interaction']:
                if component in student_components and component in teacher_components:
                    recon_loss = self.distillation_loss.reconstruction_loss(
                        student_components[component],
                        teacher_components[component]
                    )
                    losses[f'reconstruction_{component}'] = recon_loss
                    total_loss += 0.1 * recon_loss
        
        losses['total'] = total_loss
        
        return losses


class PathImportanceLoss(nn.Module):
    """
    Path importance loss for metapath-based distillation.
    
    This class implements loss functions for distilling path importance
    information from teacher to student.
    """
    
    def __init__(self, temperature: float = 0.07, use_kl_div: bool = True):
        """
        Initialize path importance loss.
        
        Args:
            temperature: Temperature for softmax.
            use_kl_div: Whether to use KL divergence loss.
        """
        super(PathImportanceLoss, self).__init__()
        
        self.temperature = temperature
        self.use_kl_div = use_kl_div
        
        self.distillation_loss = DistillationLoss()
        
        self.logger = Logger(
            log_dir='./logs',
            name='path_importance_loss'
        )
    
    def forward(
        self,
        student_path_attn: torch.Tensor,
        teacher_path_attn: torch.Tensor,
        path_types: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for path importance loss.
        
        Args:
            student_path_attn: Student path attention weights.
            teacher_path_attn: Teacher path attention weights.
            path_types: Optional list of path type names.
        
        Returns:
            Dict mapping loss types to values.
        """
        losses = {}
        
        # Path importance loss
        path_loss = self.distillation_loss.path_importance_loss(
            student_path_attn,
            teacher_path_attn,
            temperature=self.temperature
        )
        losses['path_importance'] = path_loss
        
        # Normalization loss
        student_norm = F.normalize(student_path_attn, p=1, dim=-1)
        teacher_norm = F.normalize(teacher_path_attn, p=1, dim=-1)
        
        l1_loss = F.l1_loss(student_norm, teacher_norm)
        losses['l1_norm'] = l1_loss
        
        # Distribution loss
        if self.use_kl_div:
            student_soft = F.softmax(student_path_attn / self.temperature, dim=-1)
            teacher_soft = F.softmax(teacher_path_attn / self.temperature, dim=-1)
            
            kl_loss = F.kl_div(
                torch.log(student_soft + 1e-9),
                teacher_soft,
                reduction='batchmean'
            )
            losses['kl_div'] = kl_loss
        
        # Combined loss
        total_loss = 0.5 * losses['path_importance'] + 0.3 * losses.get('l1_norm', 0) + 0.2 * losses.get('kl_div', 0)
        losses['total'] = total_loss
        
        return losses


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for representation alignment.
    
    This class implements both supervised and unsupervised contrastive
    learning objectives.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        use_supervised: bool = False,
        margin: float = 0.1
    ):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature for similarity scaling.
            use_supervised: Whether to use supervised contrastive loss.
            margin: Margin for triplet loss.
        """
        super(ContrastiveLoss, self).__init__()
        
        self.temperature = temperature
        self.use_supervised = use_supervised
        self.margin = margin
        
        self.distillation_loss = DistillationLoss()
        
        self.logger = Logger(
            log_dir='./logs',
            name='contrastive_loss'
        )
    
    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for contrastive loss.
        
        Args:
            embeddings: Dict mapping node types to embeddings.
            labels: Optional labels for supervised contrastive loss.
            mask: Optional mask for pairwise similarity.
        
        Returns:
            Dict mapping loss types to values.
        """
        losses = {}
        
        # Contrastive loss
        if self.use_supervised and labels is not None:
            contrastive_loss = self.distillation_loss.contrastive_loss(
                embeddings,
                labels=labels
            )
        else:
            contrastive_loss = self.distillation_loss.contrastive_loss(
                embeddings
            )
        
        losses['contrastive'] = contrastive_loss
        
        # Triplet loss (optional)
        if mask is not None:
            triplet_loss = self._triplet_loss(embeddings, mask)
            losses['triplet'] = triplet_loss
        
        # Combined loss
        total_loss = losses['contrastive']
        if 'triplet' in losses:
            total_loss += 0.5 * losses['triplet']
        
        losses['total'] = total_loss
        
        return losses
    
    def _triplet_loss(
        self,
        embeddings: Dict[str, torch.Tensor],
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            embeddings: Dict mapping node types to embeddings.
            mask: Triplet mask (batch_size, 3).
        
        Returns:
            Triplet loss tensor.
        """
        # Concatenate all embeddings
        all_embeddings = []
        for node_type, emb in embeddings.items():
            all_embeddings.append(emb)
        
        if not all_embeddings:
            return torch.tensor(0.0)
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Compute pairwise distances
        distances = torch.cdist(all_embeddings, all_embeddings, p=2)
        
        # Extract anchor-positive-negative triplets
        anchor_idx = mask[:, 0]
        positive_idx = mask[:, 1]
        negative_idx = mask[:, 2]
        
        # Positive distances
        pos_dist = distances[anchor_idx, positive_idx]
        
        # Negative distances
        neg_dist = distances[anchor_idx, negative_idx]
        
        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        loss = loss.mean()
        
        return loss


# Module level variables and exports
__all__ = [
    'DistillationLoss',
    'ComponentWiseLoss',
    'PathImportanceLoss',
    'ContrastiveLoss',
    '__doc__'
]

# Version information
__version__ = '1.0.0'
__author__ = 'H-GRAGrecsys Team'


def create_distillation_loss(
    config_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> DistillationLoss:
    """
    Factory function to create a DistillationLoss instance.
    
    Args:
        config_path: Optional path to configuration file.
        device: Optional device to move loss to.
    
    Returns:
        Initialized DistillationLoss instance.
    
    Example:
        >>> loss_fn = create_distillation_loss('config/default_config.yaml')
        >>> loss = loss_fn.compute_total_loss(teacher_logits, student_logits)
    """
    loss_fn = DistillationLoss(config_path)
    
    if device is not None:
        loss_fn.to(device)
    
    return loss_fn


def create_component_wise_loss(
    component_weights: Optional[List[float]] = None,
    use_contrastive: bool = True,
    use_reconstruction: bool = True,
    device: Optional[torch.device] = None
) -> ComponentWiseLoss:
    """
    Factory function to create a ComponentWiseLoss instance.
    
    Args:
        component_weights: Weights for each component.
        use_contrastive: Whether to use contrastive loss.
        use_reconstruction: Whether to use reconstruction loss.
        device: Optional device to move loss to.
    
    Returns:
        Initialized ComponentWiseLoss instance.
    
    Example:
        >>> loss_fn = create_component_wise_loss(
        ...     component_weights=[1.0, 0.5, 0.5]
        ... )
    """
    loss_fn = ComponentWiseLoss(
        component_weights=component_weights,
        use_contrastive=use_contrastive,
        use_reconstruction=use_reconstruction
    )
    
    if device is not None:
        loss_fn.to(device)
    
    return loss_fn


def create_path_importance_loss(
    temperature: float = 0.07,
    use_kl_div: bool = True,
    device: Optional[torch.device] = None
) -> PathImportanceLoss:
    """
    Factory function to create a PathImportanceLoss instance.
    
    Args:
        temperature: Temperature for softmax.
        use_kl_div: Whether to use KL divergence.
        device: Optional device to move loss to.
    
    Returns:
        Initialized PathImportanceLoss instance.
    
    Example:
        >>> loss_fn = create_path_importance_loss(
        ...     temperature=0.1,
        ...     use_kl_div=True
        ... )
    """
    loss_fn = PathImportanceLoss(
        temperature=temperature,
        use_kl_div=use_kl_div
    )
    
    if device is not None:
        loss_fn.to(device)
    
    return loss_fn