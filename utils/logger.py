# In phase1_bootstrap.py
from utils.logger import get_logger

class Phase1Bootstrap:
    def __init__(self, config):
        self.logger = get_logger(
            log_dir="logs/phase1",
            name="phase1_bootstrap",
            config_path="config/default_config.yaml"
        )
        
        # Log start
        self.logger.log_experiment_start(
            experiment_name="Phase1_Bootstrap",
            config=config,
            dataset="Amazon_Books"
        )
    
    def bootstrap_agents(self):
        self.logger.log_info("Starting agent bootstrapping...")
        # ... implementation ...
        self.logger.log_metrics({
            'agents_initialized': len(agents),
            'memory_size': total_memories
        }, step=0, phase='phase1')

# In adaptive_gate.py
from utils.logger import get_logger

class AdaptiveGate:
    def __init__(self, config):
        self.logger = get_logger(
            log_dir="logs/hybrid",
            name="adaptive_gate"
        )
    
    def decide_path(self, gate_score, threshold):
        decision = "llm" if gate_score > threshold else "gnn"
        self.logger.log_gating_decision(
            node_id="current_node",
            gate_score=gate_score,
            decision=decision,
            confidence=gate_score,
            context={"threshold": threshold}
        )
        return decision