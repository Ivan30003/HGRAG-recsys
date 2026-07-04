# H-GRAGrecsys: Hybrid Graph Retrieval-Augmented Language Agents for Collaborative Recommendation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Paper Abstract

Recent advances in large language model (LLM) agents have shown promise for autonomous decision-making in recommender systems. However, existing approaches suffer from two fundamental limitations: flat agent memories that conflate different information modalities and prohibitive computational costs that prevent scaling beyond a few hundred users. We propose H-GRAGrecsys, a hybrid system that integrates hierarchical agent memory structures, graph-based retrieval augmented generation (Graph RAG), and knowledge distillation for scalable deployment.

Our approach extends agent-based collaborative filtering by structuring agent memories into intrinsic, collaborative, and interaction tiers that disentangle different information types, performing multi-hop retrieval over a dynamically constructed heterogeneous interaction graph, and distilling LLM-generated memory dynamics into efficient graph neural encoders with adaptive gating.

Experiments on Amazon review datasets demonstrate that H-GRAGrecsys achieves comparable recommendation quality to full LLM-based agents while reducing computational cost by **85%**, and improves NDCG@10 by **12.7%** over flat-memory agent baselines.

## Architecture Overview

The system consists of three sequential phases:

1. **Phase 1 - Bootstrap**: Initialize user/item agents with hierarchical memory and run collaborative reflection using LLM
2. **Phase 2 - Distillation**: Distill LLM-generated memory dynamics into efficient GNN encoders
3. **Phase 3 - Hybrid**: Deploy hybrid inference with adaptive gating between GNN and LLM paths

## Installation

### Prerequisites

Python 3.9 or higher
CUDA 11.8+ (for GPU training)
16GB+ RAM (32GB recommended)
50GB+ free disk space (for datasets)

### Setup

**Clone the repository:**

```
git clone https://github.com/yourusername/h-gragrecsys.git
cd h-gragrecsys
```

**Setup venv**
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Experiments**
bash```
# 1. Data Preparation
python scripts/preprocess_data.py --dataset "Amazon_CDs_Vinyl"

# 2. Phase 1: Bootstrap
python scripts/train_phase1.py

# 3. Phase 2: Distillation
python scripts/train_phase2.py --teacher-path experiments/phase1/checkpoints/phase1_best.pt

# 4. Phase 3: Hybrid
python scripts/train_phase3.py --gnn-path experiments/phase2/checkpoints/distilled_model.pt --llm-path experiments/phase1/checkpoints/phase1_best.pt

# 5. Evaluation
python scripts/evaluate_model.py --model-path experiments/phase3/checkpoints/phase3_best.pt

# 6. Ablation Study
python scripts/run_ablation.py --model-path experiments/phase3/checkpoints/phase3_best.pt


# 7. Run efficiency experiment
python experiments/efficiency_experiment.py --config config/default_config.yaml \
    --gnn-path experiments/phase2/checkpoints/distilled_model.pt \
    --llm-path experiments/phase1/checkpoints/phase1_best.pt

# Custom batch sizes
python experiments/efficiency_experiment.py --config config/default_config.yaml \
    --batch-sizes "1,4,8,16,32,64,128"

# Custom LLM call ratios
python experiments/efficiency_experiment.py --config config/default_config.yaml \
    --llm-ratios "0.0,0.25,0.5,0.75,1.0"

# Increase request count for better statistics
python experiments/efficiency_experiment.py --config config/default_config.yaml \
    --num-requests 10000 --warmup-requests 200

# Disable GPU/memory tracking
python experiments/efficiency_experiment.py --config config/default_config.yaml \
    --no-gpu --no-memory