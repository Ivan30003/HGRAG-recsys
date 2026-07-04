"""
__init__.py - Data utilities package initialization for H-GRAGrecsys

This module exports all data utility functions and classes for easy import.
"""

from data.utils.text_processor import (
    TextCleaner,
    TextTokenizer,
    KeywordExtractor,
    TextSummarizer,
    TextEmbedder,
    TextProcessor,
    TextProcessingResult
)

from data.utils.sampling import (
    SubsetSampler,
    NegativeSampler,
    BalancedSampler,
    HardNegativeMiner,
    PyTorchSampler,
    create_subsets_for_experiments,
    create_negative_samples_for_evaluation,
    sample_cold_start_users
)

# Package metadata
__version__ = '1.0.0'
__author__ = 'H-GRAGrecsys Team'

# Define what gets imported with "from data.utils import *"
__all__ = [
    # Text processing
    'TextCleaner',
    'TextTokenizer',
    'KeywordExtractor',
    'TextSummarizer',
    'TextEmbedder',
    'TextProcessor',
    'TextProcessingResult',
    
    # Sampling
    'SubsetSampler',
    'NegativeSampler',
    'BalancedSampler',
    'HardNegativeMiner',
    'PyTorchSampler',
    'create_subsets_for_experiments',
    'create_negative_samples_for_evaluation',
    'sample_cold_start_users',
]

# Module-level logger
import logging
logger = logging.getLogger(__name__)

def get_text_processor(config: dict) -> TextProcessor:
    """
    Convenience function to get a configured TextProcessor instance.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        TextProcessor instance
    """
    return TextProcessor(config)

def get_subset_sampler(config: dict) -> SubsetSampler:
    """
    Convenience function to get a configured SubsetSampler instance.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        SubsetSampler instance
    """
    return SubsetSampler(config)

def get_negative_sampler(dataset, config: dict) -> NegativeSampler:
    """
    Convenience function to get a configured NegativeSampler instance.
    
    Args:
        dataset: Dataset instance
        config: Configuration dictionary
    
    Returns:
        NegativeSampler instance
    """
    return NegativeSampler(dataset, config)

def get_balanced_sampler(config: dict) -> BalancedSampler:
    """
    Convenience function to get a configured BalancedSampler instance.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        BalancedSampler instance
    """
    return BalancedSampler(config)

def get_hard_negative_miner(config: dict) -> HardNegativeMiner:
    """
    Convenience function to get a configured HardNegativeMiner instance.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        HardNegativeMiner instance
    """
    return HardNegativeMiner(config)

# Module initialization
logger.info(f"Data utilities package initialized (version {__version__})")

# Check for required dependencies
def check_dependencies() -> dict:
    """
    Check if all required dependencies are available.
    
    Returns:
        Dictionary with dependency status
    """
    dependencies = {
        'nltk': False,
        'sklearn': False,
        'numpy': False,
        'torch': False,
        'transformers': False,
        'spacy': False
    }
    
    try:
        import nltk
        dependencies['nltk'] = True
    except ImportError:
        pass
    
    try:
        import sklearn
        dependencies['sklearn'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        pass
    
    try:
        import torch
        dependencies['torch'] = True
    except ImportError:
        pass
    
    try:
        import transformers
        dependencies['transformers'] = True
    except ImportError:
        pass
    
    try:
        import spacy
        dependencies['spacy'] = True
    except ImportError:
        pass
    
    return dependencies

# Print dependency status on import
if __name__ != '__main__':
    deps = check_dependencies()
    missing = [dep for dep, available in deps.items() if not available]
    if missing:
        logger.warning(f"Missing optional dependencies: {', '.join(missing)}")
        logger.warning("Some features may not be available")
    else:
        logger.info("All dependencies available")

# Example usage demonstration
def demo():
    """
    Demonstrate usage of data utilities.
    
    This function shows how to use the exported classes and functions.
    """
    print("=" * 60)
    print("Data Utilities Demo")
    print("=" * 60)
    
    # Example configuration
    config = {
        'embedding_dim': 384,
        'embedding_method': 'sentence-transformers',
        'sentence_transformer_model': 'all-MiniLM-L6-v2',
        'max_keywords': 10,
        'max_summary_length': 50,
        'custom_stopwords': ['amazon', 'product'],
        'use_spacy': False,
        'use_transformer_summarization': False,
        'evaluation': {'seed': 42}
    }
    
    # 1. Test TextProcessor
    print("\n1. Testing TextProcessor...")
    processor = get_text_processor(config)
    text = "This is a sample product description. It has multiple sentences and should be summarized."
    result = processor.process_text(text)
    print(f"  Cleaned: {result.cleaned}")
    print(f"  Summary: {result.summary}")
    print(f"  Keywords: {result.keywords}")
    print(f"  Word count: {result.word_count}")
    
    # 2. Test SubsetSampler
    print("\n2. Testing SubsetSampler...")
    sampler = get_subset_sampler(config)
    print(f"  SubsetSampler initialized with seed {config['evaluation']['seed']}")
    
    # 3. Test NegativeSampler
    print("\n3. Testing NegativeSampler...")
    print("  NegativeSampler requires a dataset instance")
    print("  Use: sampler = get_negative_sampler(dataset, config)")
    
    # 4. Test BalancedSampler
    print("\n4. Testing BalancedSampler...")
    balanced_sampler = get_balanced_sampler(config)
    print(f"  BalancedSampler initialized")
    
    # 5. Test HardNegativeMiner
    print("\n5. Testing HardNegativeMiner...")
    miner = get_hard_negative_miner(config)
    print(f"  HardNegativeMiner initialized with warmup_steps={config.get('hard_mining_warmup', 1000)}")
    
    # 6. Check dependencies
    print("\n6. Checking dependencies...")
    deps = check_dependencies()
    for dep, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}")
    
    print("\n" + "=" * 60)
    print("Demo complete")
    print("=" * 60)

if __name__ == "__main__":
    demo()