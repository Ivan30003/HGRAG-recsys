#!/bin/bash

# Hybrid-GraphRAG Experiment Runner
# =====================================

set -e  # Exit on error

# Configuration
CONFIG_FILE="experiment_launch_confg.yaml"
LOG_DIR="./logs"
RESULTS_DIR="./results"
PYTHON="python3"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Hybrid-GraphRAG Experiment Pipeline   ${NC}"
echo -e "${BLUE}========================================${NC}"

# Create directories
mkdir -p ${LOG_DIR}
mkdir -p ${RESULTS_DIR}

# Check for OpenAI API key
if [ -z "${OPENAI_API_KEY}" ]; then
    echo -e "${YELLOW}Warning: OPENAI_API_KEY not set. LLM calls will fail.${NC}"
    echo -e "${YELLOW}Set it with: export OPENAI_API_KEY='your-key-here'${NC}"
fi

# Function to run a phase with timing
run_phase() {
    local phase_name=$1
    local script=$2
    local log_file="${LOG_DIR}/${phase_name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo -e "\n${GREEN}[$(date '+%H:%M:%S')] Starting ${phase_name}...${NC}"
    
    start_time=$(date +%s)
    
    ${PYTHON} ${script} --config ${CONFIG_FILE} 2>&1 | tee ${log_file}
    
    exit_code=${PIPESTATUS[0]}
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    if [ ${exit_code} -eq 0 ]; then
        echo -e "${GREEN}[$(date '+%H:%M:%S')] ${phase_name} completed successfully in ${duration}s${NC}"
    else
        echo -e "${RED}[$(date '+%H:%M:%S')] ${phase_name} failed with exit code ${exit_code}${NC}"
        exit ${exit_code}
    fi
}

# ============================================
# Phase 1: Bootstrap
# ============================================
run_phase "Phase 1: Bootstrap" "bootstrap_phase.py"

# ============================================
# Phase 2: GNN Training (Distillation)
# ============================================
run_phase "Phase 2: GNN Distillation" "gnn_train_phase.py"

# ============================================
# Phase 3: Hybrid Inference
# ============================================
run_phase "Phase 3: Hybrid Inference" "hybrid_inference_phase.py"

# ============================================
# Generate Summary Report
# ============================================
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  Generating Summary Report              ${NC}"
echo -e "${BLUE}========================================${NC}"

REPORT_FILE="${RESULTS_DIR}/experiment_report_$(date +%Y%m%d_%H%M%S).txt"

cat > ${REPORT_FILE} << EOF
========================================
Hybrid-GraphRAG Experiment Report
========================================
Date: $(date)
Configuration: ${CONFIG_FILE}

Phase 1 (Bootstrap):
  Log: ${LOG_DIR}/Phase_1_*

Phase 2 (Distillation):
  Log: ${LOG_DIR}/Phase_2_*

Phase 3 (Inference):
  Log: ${LOG_DIR}/Phase_3_*

Results Directory: ${RESULTS_DIR}
========================================
EOF

echo -e "${GREEN}Report saved to: ${REPORT_FILE}${NC}"
echo -e "\n${GREEN}Experiment pipeline completed successfully!${NC}"