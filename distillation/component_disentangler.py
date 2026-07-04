"""
Component Disentangler Module for H-GRAGrecsys

This module implements component disentanglement for representation learning
in H-GRAGrecsys. It provides:
- Disentanglement of representations into intrinsic, collaborative, and interaction components
- Orthogonality enforcement for disentangled representations
- Component-specific reconstruction and regularization
- Adversarial disentanglement with gradient reversal
- Mutual information minimization
- Component importance weighting

The disentangler enables learning of interpretable and disentangled
representations for recommendation systems.
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
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


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient reversal layer for adversarial disentanglement.
    
    This layer reverses the gradient during backpropagation, enabling
    adversarial training for disentanglement.
    """
    
    @staticmethod
    def forward(ctx, x, alpha):
        """
        Forward pass: identity function.
        
        Args:
            ctx: Context for backward pass.
            x: Input tensor.
            alpha: Gradient reversal strength.
        
        Returns:
            Input tensor unchanged.
        """
        ctx.alpha = alpha
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: reverse gradients.
        
        Args:
            ctx: Context with alpha.
            grad_output: Gradient from downstream.
        
        Returns:
            Reversed gradient and None for alpha.
        """
        return -ctx.alpha * grad_output, None


class ComponentDisentangler(nn.Module):
    """
    Component disentangler for disentangled representation learning.
    
    This class implements disentanglement of representations into three
    components: intrinsic, collaborative, and interaction. It enforces
    orthogonality and provides various regularization techniques.
    """
    
    def __init__(self, config: Optional[Union[str, Dict, ConfigLoader]] = None):
        """
        Initialize the component disentangler.
        
        Args:
            config: Configuration object or path to config file.
                   Can be a string path, dict, or ConfigLoader instance.
        
        Raises:
            ValueError: If config is invalid or missing required fields.
        """
        super(ComponentDisentangler, self).__init__()
        
        # Load configuration
        if config is None:
            self.config = {
                'model': {
                    'distillation': {
                        'component_weights': [1.0, 1.0, 1.0],
                        'disentangle_lambda': 0.1,
                        'orthogonality_lambda': 0.2,
                        'reconstruction_lambda': 0.3,
                        'adversarial_lambda': 0.1,
                        'mutual_info_lambda': 0.1
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
            name='component_disentangler'
        )
        
        # Extract configuration
        dist_config = self.config.get('model', {}).get('distillation', {})
        
        self.component_weights = dist_config.get('component_weights', [1.0, 1.0, 1.0])
        self.disentangle_lambda = dist_config.get('disentangle_lambda', 0.1)
        self.orthogonality_lambda = dist_config.get('orthogonality_lambda', 0.2)
        self.reconstruction_lambda = dist_config.get('reconstruction_lambda', 0.3)
        self.adversarial_lambda = dist_config.get('adversarial_lambda', 0.1)
        self.mutual_info_lambda = dist_config.get('mutual_info_lambda', 0.1)
        
        # Component dimensions
        self.component_dim = dist_config.get('component_dim', 128)
        self.num_components = 3  # intrinsic, collaborative, interaction
        
        # Initialize projection networks for each component
        self._initialize_networks()
        
        # Initialize adversarial discriminator
        self._initialize_adversarial_network()
        
        # Mutual information estimator
        self._initialize_mutual_info_network()
        
        self.logger.log_info(
            f"ComponentDisentangler initialized: "
            f"component_dim={self.component_dim}, "
            f"disentangle_lambda={self.disentangle_lambda}, "
            f"orthogonality_lambda={self.orthogonality_lambda}"
        )
    
    def _initialize_networks(self):
        """Initialize disentanglement networks for each component."""
        # Projection networks
        self.component_projections = nn.ModuleDict({
            'intrinsic': nn.Sequential(
                nn.Linear(self.component_dim * 3, self.component_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.component_dim * 2, self.component_dim)
            ),
            'collaborative': nn.Sequential(
                nn.Linear(self.component_dim * 3, self.component_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.component_dim * 2, self.component_dim)
            ),
            'interaction': nn.Sequential(
                nn.Linear(self.component_dim * 3, self.component_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.component_dim * 2, self.component_dim)
            )
        })
        
        # Reconstruction networks
        self.reconstruction_networks = nn.ModuleDict({
            'intrinsic': nn.Linear(self.component_dim, self.component_dim * 3),
            'collaborative': nn.Linear(self.component_dim, self.component_dim * 3),
            'interaction': nn.Linear(self.component_dim, self.component_dim * 3)
        })
        
        # Component importance networks
        self.importance_networks = nn.ModuleDict({
            'intrinsic': nn.Linear(self.component_dim, 1),
            'collaborative': nn.Linear(self.component_dim, 1),
            'interaction': nn.Linear(self.component_dim, 1)
        })
    
    def _initialize_adversarial_network(self):
        """Initialize adversarial discriminator for disentanglement."""
        self.adversarial_discriminator = nn.Sequential(
            nn.Linear(self.component_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_components)  # Classify component type
        )
        
        # Gradient reversal strength (learnable)
        self.grad_reverse_alpha = nn.Parameter(torch.tensor(0.1))
    
    def _initialize_mutual_info_network(self):
        """Initialize mutual information estimation network."""
        self.mutual_info_net = nn.Sequential(
            nn.Linear(self.component_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def disentangle_representations(
        self,
        representation: torch.Tensor,
        return_components: bool = True
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """
        Disentangle a representation into component representations.
        
        Args:
            representation: Input representation tensor (batch_size, input_dim).
            return_components: Whether to return disentangled components.
        
        Returns:
            If return_components=True: Dict of component representations.
            If return_components=False: Tuple of (components, losses).
        
        Raises:
            ValueError: If representation dimension is invalid.
        """
        batch_size = representation.size(0)
        input_dim = representation.size(1)
        
        # Check input dimension
        if input_dim != self.component_dim * 3 and input_dim != self.component_dim:
            # Project to correct dimension if needed
            if not hasattr(self, 'input_projection'):
                self.input_projection = nn.Linear(input_dim, self.component_dim * 3).to(representation.device)
            representation = self.input_projection(representation)
        
        # Split into components
        if representation.size(1) == self.component_dim * 3:
            # Already concatenated components
            intrinsic_in = representation[:, :self.component_dim]
            collaborative_in = representation[:, self.component_dim:2*self.component_dim]
            interaction_in = representation[:, 2*self.component_dim:]
        else:
            # Single representation - project to components
            combined = representation
            if combined.size(1) != self.component_dim * 3:
                if not hasattr(self, 'combined_projection'):
                    self.combined_projection = nn.Linear(combined.size(1), self.component_dim * 3).to(combined.device)
                combined = self.combined_projection(combined)
            
            intrinsic_in = combined[:, :self.component_dim]
            collaborative_in = combined[:, self.component_dim:2*self.component_dim]
            interaction_in = combined[:, 2*self.component_dim:]
        
        # Pass through component projections
        intrinsic = self.component_projections['intrinsic'](
            torch.cat([intrinsic_in, collaborative_in, interaction_in], dim=1)
        )
        collaborative = self.component_projections['collaborative'](
            torch.cat([intrinsic_in, collaborative_in, interaction_in], dim=1)
        )
        interaction = self.component_projections['interaction'](
            torch.cat([intrinsic_in, collaborative_in, interaction_in], dim=1)
        )
        
        components = {
            'intrinsic': intrinsic,
            'collaborative': collaborative,
            'interaction': interaction
        }
        
        if return_components:
            return components
        
        # Compute disentanglement losses
        losses = self._compute_disentanglement_losses(components, intrinsic_in, collaborative_in, interaction_in)
        
        return components, losses
    
    def _compute_disentanglement_losses(
        self,
        components: Dict[str, torch.Tensor],
        intrinsic_in: torch.Tensor,
        collaborative_in: torch.Tensor,
        interaction_in: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute disentanglement losses.
        
        Args:
            components: Dict of component representations.
            intrinsic_in: Input intrinsic component.
            collaborative_in: Input collaborative component.
            interaction_in: Input interaction component.
        
        Returns:
            Dict of loss values.
        """
        losses = {}
        total_loss = 0.0
        
        # 1. Orthogonality loss
        ortho_loss = self.enforce_orthogonality(components)
        losses['orthogonality'] = self.orthogonality_lambda * ortho_loss
        total_loss += losses['orthogonality']
        
        # 2. Reconstruction loss
        recon_loss = self._compute_reconstruction_loss(components, intrinsic_in, collaborative_in, interaction_in)
        losses['reconstruction'] = self.reconstruction_lambda * recon_loss
        total_loss += losses['reconstruction']
        
        # 3. Adversarial loss
        adv_loss = self._compute_adversarial_loss(components)
        losses['adversarial'] = self.adversarial_lambda * adv_loss
        total_loss += losses['adversarial']
        
        # 4. Mutual information loss
        mi_loss = self._compute_mutual_info_loss(components)
        losses['mutual_info'] = self.mutual_info_lambda * mi_loss
        total_loss += losses['mutual_info']
        
        # 5. Component importance regularization
        importance_loss = self._compute_importance_loss(components)
        losses['importance'] = 0.05 * importance_loss
        total_loss += losses['importance']
        
        losses['total'] = total_loss
        
        return losses
    
    def _compute_reconstruction_loss(
        self,
        components: Dict[str, torch.Tensor],
        intrinsic_in: torch.Tensor,
        collaborative_in: torch.Tensor,
        interaction_in: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss for components.
        
        Args:
            components: Dict of component representations.
            intrinsic_in: Input intrinsic component.
            collaborative_in: Input collaborative component.
            interaction_in: Input interaction component.
        
        Returns:
            Reconstruction loss tensor.
        """
        loss = 0.0
        count = 0
        
        # Reconstruct each component
        for comp_type, comp_rep in components.items():
            if comp_type in self.reconstruction_networks:
                # Reconstruct from component
                reconstructed = self.reconstruction_networks[comp_type](comp_rep)
                
                # Get target based on component type
                if comp_type == 'intrinsic':
                    target = torch.cat([intrinsic_in, collaborative_in, interaction_in], dim=1)
                elif comp_type == 'collaborative':
                    target = torch.cat([intrinsic_in, collaborative_in, interaction_in], dim=1)
                else:  # interaction
                    target = torch.cat([intrinsic_in, collaborative_in, interaction_in], dim=1)
                
                # Ensure same shape
                if reconstructed.size(1) != target.size(1):
                    # Project to same dimension
                    if not hasattr(self, f'recon_proj_{comp_type}'):
                        setattr(self, f'recon_proj_{comp_type}', 
                               nn.Linear(reconstructed.size(1), target.size(1)).to(reconstructed.device))
                    proj = getattr(self, f'recon_proj_{comp_type}')
                    reconstructed = proj(reconstructed)
                
                # Compute loss
                recon_loss = F.mse_loss(reconstructed, target)
                loss += recon_loss
                count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss
    
    def _compute_adversarial_loss(
        self,
        components: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute adversarial loss for disentanglement.
        
        Args:
            components: Dict of component representations.
        
        Returns:
            Adversarial loss tensor.
        """
        # Collect all components
        all_components = []
        component_labels = []
        
        for idx, (comp_type, comp_rep) in enumerate(components.items()):
            all_components.append(comp_rep)
            component_labels.extend([idx] * comp_rep.size(0))
        
        all_components = torch.cat(all_components, dim=0)
        component_labels = torch.tensor(component_labels, device=all_components.device)
        
        # Apply gradient reversal
        reversed_components = GradientReversalLayer.apply(
            all_components,
            self.grad_reverse_alpha.item()
        )
        
        # Discriminator predictions
        predictions = self.adversarial_discriminator(reversed_components)
        
        # Adversarial loss (binary cross-entropy)
        adv_loss = F.cross_entropy(predictions, component_labels)
        
        return adv_loss
    
    def _compute_mutual_info_loss(
        self,
        components: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute mutual information loss between components.
        
        Args:
            components: Dict of component representations.
        
        Returns:
            Mutual information loss tensor.
        """
        loss = 0.0
        count = 0
        
        # Compute pairwise mutual information
        comp_types = list(components.keys())
        
        for i in range(len(comp_types)):
            for j in range(i + 1, len(comp_types)):
                comp1 = components[comp_types[i]]
                comp2 = components[comp_types[j]]
                
                # Concatenate components
                combined = torch.cat([comp1, comp2], dim=1)
                
                # Compute mutual information estimate
                mi_score = self.mutual_info_net(combined)
                
                # Minimize mutual information (push to 0)
                mi_loss = mi_score.mean()
                loss += mi_loss
                count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss
    
    def _compute_importance_loss(
        self,
        components: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute component importance regularization loss.
        
        Args:
            components: Dict of component representations.
        
        Returns:
            Importance loss tensor.
        """
        loss = 0.0
        count = 0
        
        for comp_type, comp_rep in components.items():
            if comp_type in self.importance_networks:
                # Compute importance score
                importance = self.importance_networks[comp_type](comp_rep)
                
                # Encourage sparsity
                sparsity_loss = torch.norm(importance, p=1) / importance.numel()
                loss += sparsity_loss
                count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss
    
    def enforce_orthogonality(
        self,
        components: Dict[str, torch.Tensor],
        eps: float = 1e-9
    ) -> torch.Tensor:
        """
        Enforce orthogonality between component representations.
        
        Args:
            components: Dict of component representations.
            eps: Small value for numerical stability.
        
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
    
    def separate_components(
        self,
        embedding: torch.Tensor,
        component_type: str
    ) -> torch.Tensor:
        """
        Separate embedding into a specific component.
        
        Args:
            embedding: Input embedding tensor.
            component_type: Type of component to extract.
        
        Returns:
            Component representation tensor.
        
        Raises:
            ValueError: If component_type is invalid.
        """
        valid_components = ['intrinsic', 'collaborative', 'interaction']
        if component_type not in valid_components:
            raise ValueError(
                f"Invalid component_type: {component_type}. "
                f"Must be one of: {valid_components}"
            )
        
        # Disentangle components
        components = self.disentangle_representations(embedding, return_components=True)
        
        return components[component_type]
    
    def compute_disentanglement_loss(
        self,
        representation: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute disentanglement loss for a representation.
        
        Args:
            representation: Input representation tensor.
        
        Returns:
            Dict of loss values.
        """
        # Disentangle with loss computation
        components, losses = self.disentangle_representations(
            representation,
            return_components=False
        )
        
        return losses
    
    def get_component_importance(
        self,
        components: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Get importance scores for components.
        
        Args:
            components: Dict of component representations.
        
        Returns:
            Dict of importance scores.
        """
        importance_scores = {}
        
        for comp_type, comp_rep in components.items():
            if comp_type in self.importance_networks:
                importance = self.importance_networks[comp_type](comp_rep)
                importance_scores[comp_type] = torch.sigmoid(importance)
            else:
                importance_scores[comp_type] = torch.ones_like(comp_rep[:, :1])
        
        return importance_scores
    
    def mix_components(
        self,
        components1: Dict[str, torch.Tensor],
        components2: Dict[str, torch.Tensor],
        mix_ratio: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Mix components from two different representations.
        
        Args:
            components1: First set of components.
            components2: Second set of components.
            mix_ratio: Mixing ratio between 0 and 1.
        
        Returns:
            Mixed components.
        """
        mixed_components = {}
        
        for comp_type in components1.keys():
            if comp_type in components2:
                comp1 = components1[comp_type]
                comp2 = components2[comp_type]
                
                # Ensure same shape
                if comp1.size(1) != comp2.size(1):
                    if comp1.size(1) > comp2.size(1):
                        comp2 = F.pad(comp2, (0, comp1.size(1) - comp2.size(1)))
                    else:
                        comp1 = F.pad(comp1, (0, comp2.size(1) - comp1.size(1)))
                
                mixed = mix_ratio * comp1 + (1 - mix_ratio) * comp2
                mixed_components[comp_type] = mixed
            else:
                mixed_components[comp_type] = components1[comp_type]
        
        return mixed_components
    
    def combine_components(
        self,
        components: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Combine components into a single representation.
        
        Args:
            components: Dict of component representations.
            weights: Optional weights for each component.
        
        Returns:
            Combined representation tensor.
        """
        if weights is None:
            weights = {
                'intrinsic': 0.4,
                'collaborative': 0.3,
                'interaction': 0.3
            }
        
        # Ensure all components exist
        combined = []
        for comp_type, weight in weights.items():
            if comp_type in components:
                comp_rep = components[comp_type]
                combined.append(weight * comp_rep)
        
        if not combined:
            raise ValueError("No components to combine")
        
        return torch.cat(combined, dim=1)
    
    def set_grad_reverse_alpha(self, alpha: float):
        """
        Set gradient reversal strength.
        
        Args:
            alpha: Gradient reversal strength.
        """
        self.grad_reverse_alpha.data.fill_(alpha)
        self.logger.log_info(f"Gradient reversal alpha set to {alpha}")
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        for network in self.component_projections.values():
            if hasattr(network, 'reset_parameters'):
                for layer in network:
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
        
        for network in self.reconstruction_networks.values():
            if hasattr(network, 'reset_parameters'):
                network.reset_parameters()
        
        for network in self.importance_networks.values():
            if hasattr(network, 'reset_parameters'):
                network.reset_parameters()
        
        for layer in self.adversarial_discriminator:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        for layer in self.mutual_info_net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        self.grad_reverse_alpha.data.fill_(0.1)
    
    def get_parameters(self) -> Dict[str, int]:
        """
        Get parameter statistics.
        
        Returns:
            Dict with parameter counts for each component.
        """
        params = {
            'total': 0
        }
        
        for name, network in self.component_projections.items():
            params[f'projection_{name}'] = sum(p.numel() for p in network.parameters())
            params['total'] += params[f'projection_{name}']
        
        for name, network in self.reconstruction_networks.items():
            params[f'reconstruction_{name}'] = sum(p.numel() for p in network.parameters())
            params['total'] += params[f'reconstruction_{name}']
        
        for name, network in self.importance_networks.items():
            params[f'importance_{name}'] = sum(p.numel() for p in network.parameters())
            params['total'] += params[f'importance_{name}']
        
        params['adversarial'] = sum(p.numel() for p in self.adversarial_discriminator.parameters())
        params['total'] += params['adversarial']
        
        params['mutual_info'] = sum(p.numel() for p in self.mutual_info_net.parameters())
        params['total'] += params['mutual_info']
        
        return params
    
    def to_device(self, device: torch.device) -> 'ComponentDisentangler':
        """
        Move all components to specified device.
        
        Args:
            device: PyTorch device.
        
        Returns:
            Self with components moved to device.
        """
        self.to(device)
        self.logger.log_info(f"ComponentDisentangler moved to device: {device}")
        
        return self
    
    def forward(
        self,
        representation: torch.Tensor,
        return_components: bool = True
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """
        Forward pass for component disentanglement.
        
        Args:
            representation: Input representation tensor.
            return_components: Whether to return components or losses.
        
        Returns:
            If return_components=True: Dict of component representations.
            If return_components=False: Tuple of (components, losses).
        """
        return self.disentangle_representations(representation, return_components)


# Module level variables and exports
__all__ = [
    'GradientReversalLayer',
    'ComponentDisentangler',
    '__doc__'
]

# Version information
__version__ = '1.0.0'
__author__ = 'H-GRAGrecsys Team'


def create_component_disentangler(
    config_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> ComponentDisentangler:
    """
    Factory function to create a ComponentDisentangler instance.
    
    Args:
        config_path: Optional path to configuration file.
        device: Optional device to move disentangler to.
    
    Returns:
        Initialized ComponentDisentangler instance.
    
    Example:
        >>> disentangler = create_component_disentangler(
        ...     config_path='config/default_config.yaml'
        ... )
        >>> components = disentangler.disentangle_representations(embedding)
    """
    disentangler = ComponentDisentangler(config_path)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return disentangler.to_device(device)


def create_gradient_reversal_layer(
    alpha: float = 0.1
) -> torch.autograd.Function:
    """
    Factory function to create a GradientReversalLayer.
    
    Args:
        alpha: Gradient reversal strength.
    
    Returns:
        GradientReversalLayer function.
    
    Example:
        >>> grl = create_gradient_reversal_layer(alpha=0.2)
        >>> output = grl.apply(input, alpha)
    """
    return GradientReversalLayer