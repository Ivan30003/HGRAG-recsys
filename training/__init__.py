"""
Training Module for H-GRAGrecsys

This module implements the three-phase training pipeline:
- Phase 1: Bootstrap - Initializes agents with collaborative reflections
- Phase 2: Distillation - Distills knowledge from LLM teachers to GNN student
- Phase 3: Hybrid - Trains adaptive gating between GNN and LLM paths

The module provides base trainer functionality and checkpoint management
for all training phases.
"""

import os
import sys
from typing import Optional, Dict, Any, Union, List, Tuple
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Core training components
from .phase1_bootstrap import Phase1Bootstrap
from .phase2_distillation import Phase2Distillation
from .phase3_hybrid import Phase3Hybrid
from .trainer_base import BaseTrainer
from .checkpoint_manager import CheckpointManager

# Data and model imports needed for training
from data.dataset import BaseDataset, AmazonDataset
from data.data_loader import DataLoader, InteractionDataLoader
from models.agent.user_agent import UserAgent
from models.agent.item_agent import ItemAgent
from models.graph.heterogeneous_graph import HeterogeneousGraph
from models.graph.graph_builder import GraphBuilder
from models.graph_rag.retriever import GraphRAGRetriever
from models.llm.llm_interface import LLMInterface
from models.llm.reflection_engine import ReflectionEngine
from models.gnn.heterogeneous_gnn import HeterogeneousGNN
from models.gnn.projection_heads import ComponentProjectionHeads
from models.hybrid.adaptive_gate import AdaptiveGate
from models.hybrid.router import Router
from models.hybrid.inference_engine import HybridInferenceEngine
from distillation.distillation_trainer import DistillationTrainer
from distillation.knowledge_distiller import KnowledgeDistiller
from distillation.component_disentangler import ComponentDisentangler

# Utility imports
from utils.logger import Logger
from utils.config_loader import ConfigLoader
from utils.seed_manager import SeedManager
from utils.timer import Timer

# Configuration imports
from config.default_config import DEFAULT_CONFIG
from config.training_config import TRAINING_CONFIG


__all__ = [
    # Core training classes
    'Phase1Bootstrap',
    'Phase2Distillation', 
    'Phase3Hybrid',
    'BaseTrainer',
    'CheckpointManager',
    
    # Training utility functions
    'create_trainer',
    'load_training_config',
    'setup_training_environment',
    'get_training_phase',
    'validate_training_phase',
    'TrainingPhase',
    
    # Constants
    'DEFAULT_TRAINING_CONFIG',
    'TRAINING_PHASES',
    'PHASE1_CONFIG',
    'PHASE2_CONFIG',
    'PHASE3_CONFIG'
]


# Training phase constants
class TrainingPhase:
    """Enumeration of training phases"""
    PHASE1 = "phase1_bootstrap"
    PHASE2 = "phase2_distillation"
    PHASE3 = "phase3_hybrid"
    
    @classmethod
    def all_phases(cls) -> List[str]:
        return [cls.PHASE1, cls.PHASE2, cls.PHASE3]
    
    @classmethod
    def validate(cls, phase: str) -> bool:
        return phase in cls.all_phases()


# Default configuration values
DEFAULT_TRAINING_CONFIG = {
    'phase1': {
        'num_epochs': 50,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'easy_threshold': 10,
        'reflection_batch_size': 16,
        'save_interval': 5
    },
    'phase2': {
        'num_epochs': 30,
        'batch_size': 64,
        'learning_rate': 1e-4,
        'distillation_temperature': 2.0,
        'component_weights': [1.0, 1.0, 1.0],
        'save_interval': 5
    },
    'phase3': {
        'num_epochs': 20,
        'batch_size': 32,
        'learning_rate': 5e-5,
        'gate_threshold': 0.3,
        'validate_interval': 2,
        'save_interval': 3
    },
    'common': {
        'device': 'cuda',
        'seed': 42,
        'log_interval': 10,
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs'
    }
}

TRAINING_PHASES = TrainingPhase.all_phases()

# Phase-specific configuration references
PHASE1_CONFIG = DEFAULT_TRAINING_CONFIG['phase1']
PHASE2_CONFIG = DEFAULT_TRAINING_CONFIG['phase2']
PHASE3_CONFIG = DEFAULT_TRAINING_CONFIG['phase3']


def load_training_config(
    config_path: Optional[str] = None,
    phase: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load training configuration from file or use defaults
    
    Args:
        config_path: Optional path to configuration YAML file
        phase: Optional training phase to load specific config for
        
    Returns:
        Dict[str, Any]: Training configuration dictionary
        
    Raises:
        FileNotFoundError: If config_path doesn't exist
        ValueError: If phase is invalid
    """
    if phase and not TrainingPhase.validate(phase):
        raise ValueError(f"Invalid training phase: {phase}. Must be one of {TrainingPhase.all_phases()}")
    
    if config_path and os.path.exists(config_path):
        config_loader = ConfigLoader(config_path)
        config = config_loader.load_config()
    else:
        # Use default configuration
        config = DEFAULT_TRAINING_CONFIG.copy()
        
        # Try to load from training_config.py if available
        try:
            from config.training_config import TRAINING_CONFIG as CONFIG
            config = CONFIG.copy()
        except ImportError:
            # Fall back to default
            pass
    
    # If phase specified, return only that phase's config
    if phase:
        if phase in config:
            return config[phase]
        else:
            # Try to find phase in common or return phase-specific default
            if phase == TrainingPhase.PHASE1:
                return PHASE1_CONFIG.copy()
            elif phase == TrainingPhase.PHASE2:
                return PHASE2_CONFIG.copy()
            elif phase == TrainingPhase.PHASE3:
                return PHASE3_CONFIG.copy()
    
    return config


def setup_training_environment(
    config: Dict[str, Any],
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Set up training environment including seeds, device, and logging
    
    Args:
        config: Training configuration dictionary
        seed: Optional seed value (overrides config)
        
    Returns:
        Dict[str, Any]: Updated configuration with environment settings
        
    Raises:
        RuntimeError: If CUDA is requested but not available
    """
    # Set seed
    seed_value = seed or config.get('common', {}).get('seed', 42)
    SeedManager.set_seed(seed_value)
    
    # Setup device
    device = config.get('common', {}).get('device', 'cuda')
    if device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = 'cpu'
            config['common']['device'] = 'cpu'
    
    # Setup logging
    log_dir = config.get('common', {}).get('log_dir', './logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Add environment info to config
    config['environment'] = {
        'seed': seed_value,
        'device': device,
        'log_dir': log_dir,
        'timestamp': Timer.get_current_timestamp()
    }
    
    return config


def create_trainer(
    phase: str,
    config: Dict[str, Any],
    dataset: Optional[BaseDataset] = None,
    llm_interface: Optional[LLMInterface] = None,
    gnn_model: Optional[HeterogeneousGNN] = None,
    teacher_path: Optional[str] = None,
    student_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None
) -> Union[Phase1Bootstrap, Phase2Distillation, Phase3Hybrid]:
    """
    Factory function to create appropriate trainer for given phase
    
    Args:
        phase: Training phase ('phase1_bootstrap', 'phase2_distillation', or 'phase3_hybrid')
        config: Configuration dictionary
        dataset: Dataset instance (required for phase1)
        llm_interface: LLM interface (required for phase1)
        gnn_model: GNN model (required for phase2/phase3)
        teacher_path: Path to teacher model (required for phase2)
        student_path: Path to student model (optional for phase2/phase3)
        checkpoint_dir: Directory for checkpoints (optional)
        
    Returns:
        Trainer instance for the specified phase
        
    Raises:
        ValueError: If phase is invalid or required parameters are missing
    """
    if not TrainingPhase.validate(phase):
        raise ValueError(f"Invalid phase: {phase}. Must be one of {TrainingPhase.all_phases()}")
    
    # Setup checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = config.get('common', {}).get('checkpoint_dir', './checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup logger
    logger = Logger(
        log_dir=config.get('common', {}).get('log_dir', './logs'),
        name=f"trainer_{phase}"
    )
    
    if phase == TrainingPhase.PHASE1:
        # Phase 1 requires dataset and LLM
        if dataset is None:
            raise ValueError("Dataset is required for Phase 1 bootstrap training")
        if llm_interface is None:
            raise ValueError("LLM interface is required for Phase 1 bootstrap training")
        
        return Phase1Bootstrap(
            dataset=dataset,
            llm=llm_interface,
            config=config.get(phase, {})
        )
    
    elif phase == TrainingPhase.PHASE2:
        # Phase 2 requires GNN model and teacher model
        if gnn_model is None:
            raise ValueError("GNN model is required for Phase 2 distillation")
        if teacher_path is None:
            raise ValueError("Teacher model path is required for Phase 2 distillation")
        
        # Load teacher model
        teacher = load_teacher_model(teacher_path, config)
        
        # Create knowledge distiller
        distiller = KnowledgeDistiller(
            teacher_llm=teacher,
            student_gnn=gnn_model,
            config=config.get(phase, {})
        )
        
        return Phase2Distillation(
            teachers=teacher,
            student_graph=gnn_model,
            config=config.get(phase, {})
        )
    
    elif phase == TrainingPhase.PHASE3:
        # Phase 3 requires GNN model and LLM
        if gnn_model is None:
            raise ValueError("GNN model is required for Phase 3 hybrid training")
        if llm_interface is None:
            raise ValueError("LLM interface is required for Phase 3 hybrid training")
        
        # Create adaptive gate and router
        gate = AdaptiveGate(config=config.get(phase, {}))
        router = Router(gate=gate, threshold=config.get(phase, {}).get('gate_threshold', 0.3))
        
        # Create hybrid inference engine
        inference_engine = HybridInferenceEngine(
            gnn_encoder=gnn_model,
            llm_interface=llm_interface,
            router=router,
            config=config.get(phase, {})
        )
        
        return Phase3Hybrid(
            gnn_model=gnn_model,
            llm_model=llm_interface,
            gate=gate,
            config=config.get(phase, {})
        )
    
    else:
        raise ValueError(f"Unsupported phase: {phase}")


def get_training_phase(phase_name: str) -> str:
    """
    Get standardized training phase name
    
    Args:
        phase_name: Phase name or alias
        
    Returns:
        str: Standardized phase name
        
    Raises:
        ValueError: If phase_name is invalid
    """
    phase_map = {
        'phase1': TrainingPhase.PHASE1,
        'phase1_bootstrap': TrainingPhase.PHASE1,
        'bootstrap': TrainingPhase.PHASE1,
        'phase2': TrainingPhase.PHASE2,
        'phase2_distillation': TrainingPhase.PHASE2,
        'distillation': TrainingPhase.PHASE2,
        'phase3': TrainingPhase.PHASE3,
        'phase3_hybrid': TrainingPhase.PHASE3,
        'hybrid': TrainingPhase.PHASE3
    }
    
    phase_name_lower = phase_name.lower()
    if phase_name_lower in phase_map:
        return phase_map[phase_name_lower]
    else:
        raise ValueError(f"Invalid phase name: {phase_name}. Valid names: {list(phase_map.keys())}")


def validate_training_phase(phase: str) -> bool:
    """
    Validate if the given phase is a valid training phase
    
    Args:
        phase: Phase name to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        get_training_phase(phase)
        return True
    except ValueError:
        return False


def load_teacher_model(
    teacher_path: str,
    config: Dict[str, Any]
) -> LLMInterface:
    """
    Load teacher model from checkpoint
    
    Args:
        teacher_path: Path to teacher model checkpoint
        config: Configuration dictionary
        
    Returns:
        LLMInterface: Loaded teacher model
        
    Raises:
        FileNotFoundError: If teacher_path doesn't exist
        RuntimeError: If model loading fails
    """
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f"Teacher model not found at: {teacher_path}")
    
    try:
        # Load teacher model
        # This is a placeholder - actual loading depends on model format
        teacher = LLMInterface(
            model_name=config.get('llm', {}).get('teacher_model', 'gpt-4'),
            config=config
        )
        
        # Load checkpoint if it's a checkpoint file
        if teacher_path.endswith('.pt') or teacher_path.endswith('.pth'):
            import torch
            checkpoint = torch.load(teacher_path, map_location='cpu')
            # Apply checkpoint to model
            # Actual implementation depends on model architecture
        
        return teacher
        
    except Exception as e:
        raise RuntimeError(f"Failed to load teacher model: {e}")


def load_student_model(
    student_path: str,
    config: Dict[str, Any]
) -> HeterogeneousGNN:
    """
    Load student GNN model from checkpoint
    
    Args:
        student_path: Path to student model checkpoint
        config: Configuration dictionary
        
    Returns:
        HeterogeneousGNN: Loaded student model
        
    Raises:
        FileNotFoundError: If student_path doesn't exist
        RuntimeError: If model loading fails
    """
    if not os.path.exists(student_path):
        raise FileNotFoundError(f"Student model not found at: {student_path}")
    
    try:
        # Load student model
        gnn_config = config.get('gnn', {})
        student = HeterogeneousGNN(config=gnn_config)
        
        # Load checkpoint if it exists
        if student_path.endswith('.pt') or student_path.endswith('.pth'):
            import torch
            checkpoint = torch.load(student_path, map_location='cpu')
            student.load_state_dict(checkpoint.get('model_state_dict', {}))
        
        return student
        
    except Exception as e:
        raise RuntimeError(f"Failed to load student model: {e}")


def get_training_phases_sequence() -> List[str]:
    """
    Get the complete sequence of training phases
    
    Returns:
        List[str]: List of training phases in order
    """
    return TrainingPhase.all_phases()


# Module initialization
def _initialize_module():
    """Initialize the training module"""
    # Create necessary directories
    base_dirs = ['./checkpoints', './logs', './models/saved']
    for directory in base_dirs:
        os.makedirs(directory, exist_ok=True)
    
    # Set up logging
    logger = Logger(log_dir='./logs', name='training_module')
    logger.log_info("Training module initialized successfully")
    logger.log_info(f"Available phases: {TrainingPhase.all_phases()}")
    
    return logger

# Auto-initialize when module is imported
module_logger = _initialize_module()