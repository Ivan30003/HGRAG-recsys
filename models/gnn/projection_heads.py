"""
Projection Heads Module for H-GRAGrecsys

This module implements projection heads for disentangled representation learning
in heterogeneous graphs. It provides components for projecting node embeddings
into different semantic spaces (intrinsic, collaborative, interaction) and
enforces orthogonality between these components.

The projection heads support:
- Component-specific projections with separate learnable parameters
- Orthogonality constraints for disentangled representations
- Multi-head projection for enhanced expressiveness
- Gradient reversal for adversarial disentanglement
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Union, Any
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


class ProjectionHead(nn.Module):
    """
    Single projection head for mapping features to a specific semantic space.
    
    This class implements a flexible projection head with multiple layers,
    activation functions, and regularization options.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = 'relu',
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        use_residual: bool = False
    ):
        """
        Initialize a projection head.
        
        Args:
            input_dim: Input feature dimension.
            output_dim: Output feature dimension.
            hidden_dims: List of hidden layer dimensions. If None, uses single layer.
            activation: Activation function ('relu', 'leaky_relu', 'elu', 'gelu', 'tanh').
            dropout: Dropout rate.
            use_layer_norm: Whether to use layer normalization.
            use_residual: Whether to use residual connections.
        
        Raises:
            ValueError: If activation function is unsupported.
        """
        super(ProjectionHead, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_type = activation
        self.dropout_rate = dropout
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # Determine layer dimensions
        if hidden_dims is None:
            # Single layer projection
            dims = [input_dim, output_dim]
        else:
            dims = [input_dim] + hidden_dims + [output_dim]
        
        # Build sequential layers
        for i in range(len(dims) - 1):
            # Linear layer
            linear = nn.Linear(dims[i], dims[i + 1])
            self.layers.append(linear)
            
            # Activation (except for last layer)
            if i < len(dims) - 2:
                if activation == 'relu':
                    act = nn.ReLU()
                elif activation == 'leaky_relu':
                    act = nn.LeakyReLU(0.2)
                elif activation == 'elu':
                    act = nn.ELU()
                elif activation == 'gelu':
                    act = nn.GELU()
                elif activation == 'tanh':
                    act = nn.Tanh()
                else:
                    raise ValueError(f"Unsupported activation: {activation}")
                
                self.layers.append(act)
                
                # Dropout
                if dropout > 0:
                    self.layers.append(nn.Dropout(dropout))
                
                # Layer normalization
                if use_layer_norm:
                    self.layers.append(nn.LayerNorm(dims[i + 1]))
        
        # Residual projection if needed
        if use_residual and input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the projection head.
        
        Args:
            x: Input tensor (batch_size, input_dim).
        
        Returns:
            Projected tensor (batch_size, output_dim).
        """
        # Store input for residual
        residual = x
        
        # Pass through layers
        for layer in self.layers:
            x = layer(x)
        
        # Apply residual connection
        if self.use_residual and self.residual_proj is not None:
            x = x + self.residual_proj(residual)
        elif self.use_residual:
            x = x + residual
        
        return x
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        if self.residual_proj is not None:
            self.residual_proj.reset_parameters()


class ComponentProjectionHeads(nn.Module):
    """
    Multi-head projection for disentangled representation learning.
    
    This class maintains separate projection heads for intrinsic, collaborative,
    and interaction components, enabling disentangled representation learning
    with orthogonality constraints.
    """
    
    def __init__(
        self,
        input_dim: int,
        config: Optional[Union[str, Dict, ConfigLoader]] = None,
        output_dim: Optional[int] = None
    ):
        """
        Initialize component projection heads.
        
        Args:
            input_dim: Input feature dimension.
            config: Configuration object or path to config file.
                   Can be a string path, dict, or ConfigLoader instance.
            output_dim: Output dimension for each component.
                       If None, uses value from config.
        
        Raises:
            ValueError: If config is invalid or missing required fields.
        """
        super(ComponentProjectionHeads, self).__init__()
        
        # Load configuration
        if config is None:
            # Default configuration
            self.config = {
                'model': {
                    'gnn': {
                        'projection_dim': 128,
                        'dropout': 0.1,
                        'activation': 'relu',
                        'use_layer_norm': True,
                        'use_residual': False,
                        'component_weights': [1.0, 1.0, 1.0]
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
            name='projection_heads'
        )
        
        # Extract configuration
        gnn_config = self.config.get('model', {}).get('gnn', {})
        self.input_dim = input_dim
        self.output_dim = output_dim or gnn_config.get('projection_dim', 128)
        self.dropout = gnn_config.get('dropout', 0.1)
        self.activation = gnn_config.get('activation', 'relu')
        self.use_layer_norm = gnn_config.get('use_layer_norm', True)
        self.use_residual = gnn_config.get('use_residual', False)
        
        # Component weights for loss computation
        self.component_weights = gnn_config.get('component_weights', [1.0, 1.0, 1.0])
        
        # Hidden dimensions for projection heads
        self.hidden_dims = gnn_config.get('projection_hidden_dims', [256, 256])
        
        # Initialize projection heads
        self._initialize_heads()
        
        # Orthogonality projection matrix (learnable)
        self.orthogonality_proj = nn.Parameter(
            torch.randn(self.output_dim, self.output_dim) * 0.01
        )
        
        self.logger.log_info(
            f"ComponentProjectionHeads initialized: "
            f"input_dim={self.input_dim}, output_dim={self.output_dim}"
        )
    
    def _initialize_heads(self):
        """
        Initialize separate projection heads for each component.
        """
        # Intrinsic component projection (user/item intrinsic features)
        self.intrinsic_head = ProjectionHead(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            dropout=self.dropout,
            use_layer_norm=self.use_layer_norm,
            use_residual=self.use_residual
        )
        
        # Collaborative component projection (social/collaborative signals)
        self.collaborative_head = ProjectionHead(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            dropout=self.dropout,
            use_layer_norm=self.use_layer_norm,
            use_residual=self.use_residual
        )
        
        # Interaction component projection (user-item interaction patterns)
        self.interaction_head = ProjectionHead(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            dropout=self.dropout,
            use_layer_norm=self.use_layer_norm,
            use_residual=self.use_residual
        )
    
    def forward(
        self,
        x: torch.Tensor,
        component_type: str,
        return_all: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through component projection heads.
        
        Args:
            x: Input tensor (batch_size, input_dim).
            component_type: Type of component ('intrinsic', 'collaborative', 'interaction').
            return_all: If True, returns all component projections.
        
        Returns:
            If return_all=False: Projected tensor for specified component.
            If return_all=True: Dict with all component projections.
        
        Raises:
            ValueError: If component_type is invalid.
        """
        valid_components = ['intrinsic', 'collaborative', 'interaction']
        if component_type not in valid_components:
            raise ValueError(
                f"Invalid component_type: {component_type}. "
                f"Must be one of: {valid_components}"
            )
        
        if component_type == 'intrinsic':
            output = self.intrinsic_head(x)
        elif component_type == 'collaborative':
            output = self.collaborative_head(x)
        elif component_type == 'interaction':
            output = self.interaction_head(x)
        
        if return_all:
            return {
                'intrinsic': self.intrinsic_head(x),
                'collaborative': self.collaborative_head(x),
                'interaction': self.interaction_head(x)
            }
        
        return output
    
    def get_component_projection(
        self,
        x: torch.Tensor,
        component_type: str
    ) -> torch.Tensor:
        """
        Get projection for a specific component.
        
        Args:
            x: Input tensor.
            component_type: Component type.
        
        Returns:
            Projected tensor.
        """
        return self.forward(x, component_type, return_all=False)
    
    def project_intrinsic(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project to intrinsic component space.
        
        Args:
            x: Input tensor.
        
        Returns:
            Intrinsic component projection.
        """
        return self.intrinsic_head(x)
    
    def project_collaborative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project to collaborative component space.
        
        Args:
            x: Input tensor.
        
        Returns:
            Collaborative component projection.
        """
        return self.collaborative_head(x)
    
    def project_interaction(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project to interaction component space.
        
        Args:
            x: Input tensor.
        
        Returns:
            Interaction component projection.
        """
        return self.interaction_head(x)
    
    def project_all(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Project to all component spaces.
        
        Args:
            x: Input tensor.
        
        Returns:
            Dict with all component projections.
        """
        return self.forward(x, 'intrinsic', return_all=True)
    
    def compute_orthogonality_loss(
        self,
        components: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute orthogonality loss between components.
        
        Args:
            components: Dict of component projections.
        
        Returns:
            Orthogonality loss value.
        
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
        
        # Compute pairwise dot products
        loss = 0.0
        pairs = [
            (intrinsic, collaborative),
            (intrinsic, interaction),
            (collaborative, interaction)
        ]
        
        for comp1, comp2 in pairs:
            # Normalize
            comp1_norm = F.normalize(comp1, p=2, dim=-1)
            comp2_norm = F.normalize(comp2, p=2, dim=-1)
            
            # Dot product (should be close to 0)
            dot_product = (comp1_norm * comp2_norm).sum(dim=-1)
            loss += (dot_product ** 2).mean()
        
        # Apply learnable orthogonality projection
        for comp in [intrinsic, collaborative, interaction]:
            projected = torch.mm(comp, self.orthogonality_proj)
            loss += (comp - projected).pow(2).mean()
        
        return loss / 3.0  # Average over pairs
    
    def disentangle_representations(
        self,
        x: torch.Tensor,
        return_loss: bool = False
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        """
        Disentangle input representations into component spaces.
        
        Args:
            x: Input tensor.
            return_loss: Whether to return orthogonality loss.
        
        Returns:
            Dict with disentangled components, and optionally loss.
        """
        # Project to all components
        components = self.project_all(x)
        
        if return_loss:
            loss = self.compute_orthogonality_loss(components)
            return components, loss
        
        return components
    
    def get_projection_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get the weight matrices of all projection heads.
        
        Returns:
            Dict mapping component names to weight matrices.
        """
        weights = {}
        
        for name, head in [
            ('intrinsic', self.intrinsic_head),
            ('collaborative', self.collaborative_head),
            ('interaction', self.interaction_head)
        ]:
            # Get first layer weights
            for layer in head.layers:
                if isinstance(layer, nn.Linear):
                    weights[name] = layer.weight
                    break
        
        return weights
    
    def compute_similarity(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        component_type: str = 'all'
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute similarity between two tensors in component space.
        
        Args:
            x1: First tensor.
            x2: Second tensor.
            component_type: Component type or 'all'.
        
        Returns:
            Similarity score(s).
        """
        if component_type == 'all':
            components1 = self.project_all(x1)
            components2 = self.project_all(x2)
            
            similarities = {}
            for comp in ['intrinsic', 'collaborative', 'interaction']:
                # Cosine similarity
                sim = F.cosine_similarity(
                    components1[comp],
                    components2[comp],
                    dim=-1
                )
                similarities[comp] = sim.mean()
            
            return similarities
        else:
            proj1 = self.forward(x1, component_type)
            proj2 = self.forward(x2, component_type)
            
            return F.cosine_similarity(proj1, proj2, dim=-1).mean()
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        self.intrinsic_head.reset_parameters()
        self.collaborative_head.reset_parameters()
        self.interaction_head.reset_parameters()
        
        # Reset orthogonality projection
        nn.init.normal_(self.orthogonality_proj, std=0.01)
    
    def get_parameters(self) -> Dict[str, int]:
        """
        Get parameter statistics.
        
        Returns:
            Dict with parameter counts for each component.
        """
        params = {
            'intrinsic': sum(p.numel() for p in self.intrinsic_head.parameters()),
            'collaborative': sum(p.numel() for p in self.collaborative_head.parameters()),
            'interaction': sum(p.numel() for p in self.interaction_head.parameters()),
            'orthogonality_proj': self.orthogonality_proj.numel(),
            'total': sum(p.numel() for p in self.parameters())
        }
        
        return params
    
    def to_device(self, device: torch.device) -> 'ComponentProjectionHeads':
        """
        Move model to specified device.
        
        Args:
            device: PyTorch device.
        
        Returns:
            Self with model moved to device.
        """
        self.to(device)
        return self


class MultiHeadProjection(nn.Module):
    """
    Multi-head projection head with multiple independent projections.
    
    This class implements multi-head projection with independent heads
    that can be aggregated using various strategies.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 4,
        aggregation: str = 'concat'
    ):
        """
        Initialize multi-head projection.
        
        Args:
            input_dim: Input feature dimension.
            output_dim: Output feature dimension per head.
            num_heads: Number of heads.
            aggregation: Aggregation method ('concat', 'mean', 'max', 'sum').
        
        Raises:
            ValueError: If aggregation method is invalid.
        """
        super(MultiHeadProjection, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.aggregation = aggregation
        
        valid_aggregations = ['concat', 'mean', 'max', 'sum']
        if aggregation not in valid_aggregations:
            raise ValueError(
                f"Invalid aggregation: {aggregation}. "
                f"Must be one of: {valid_aggregations}"
            )
        
        # Create projection heads
        self.heads = nn.ModuleList([
            ProjectionHead(input_dim, output_dim)
            for _ in range(num_heads)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-head projection.
        
        Args:
            x: Input tensor.
        
        Returns:
            Projected tensor.
        """
        # Project through each head
        head_outputs = [head(x) for head in self.heads]
        
        # Aggregate
        if self.aggregation == 'concat':
            return torch.cat(head_outputs, dim=-1)
        elif self.aggregation == 'mean':
            return torch.stack(head_outputs, dim=0).mean(dim=0)
        elif self.aggregation == 'max':
            return torch.stack(head_outputs, dim=0).max(dim=0)[0]
        elif self.aggregation == 'sum':
            return torch.stack(head_outputs, dim=0).sum(dim=0)
    
    def get_head_outputs(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get outputs from all heads.
        
        Args:
            x: Input tensor.
        
        Returns:
            List of head outputs.
        """
        return [head(x) for head in self.heads]
    
    def reset_parameters(self):
        """Reset all head parameters."""
        for head in self.heads:
            head.reset_parameters()


# Module level variables and exports
__all__ = [
    'ProjectionHead',
    'ComponentProjectionHeads',
    'MultiHeadProjection',
    '__doc__'
]

# Version information
__version__ = '1.0.0'
__author__ = 'H-GRAGrecsys Team'


def create_projection_heads(
    input_dim: int,
    config_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> ComponentProjectionHeads:
    """
    Factory function to create ComponentProjectionHeads instance.
    
    Args:
        input_dim: Input feature dimension.
        config_path: Optional path to configuration file.
        device: Optional device to move model to. Defaults to CUDA if available.
    
    Returns:
        Initialized ComponentProjectionHeads instance.
    
    Example:
        >>> projection = create_projection_heads(
        ...     input_dim=768,
        ...     config_path='config/default_config.yaml'
        ... )
        >>> projection.to_device(torch.device('cuda'))
    """
    if config_path is None:
        config = None
    else:
        config = config_path
    
    projection = ComponentProjectionHeads(
        input_dim=input_dim,
        config=config
    )
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return projection.to_device(device)


def create_multi_head_projection(
    input_dim: int,
    output_dim: int,
    num_heads: int = 4,
    aggregation: str = 'concat',
    device: Optional[torch.device] = None
) -> MultiHeadProjection:
    """
    Factory function to create MultiHeadProjection instance.
    
    Args:
        input_dim: Input feature dimension.
        output_dim: Output dimension per head.
        num_heads: Number of heads.
        aggregation: Aggregation method.
        device: Optional device.
    
    Returns:
        Initialized MultiHeadProjection instance.
    
    Example:
        >>> projection = create_multi_head_projection(
        ...     input_dim=768,
        ...     output_dim=128,
        ...     num_heads=4
        ... )
    """
    projection = MultiHeadProjection(
        input_dim=input_dim,
        output_dim=output_dim,
        num_heads=num_heads,
        aggregation=aggregation
    )
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return projection.to(device)