"""
Mixture-of-Experts (MoE) blueprint templates.

Provides routing and expert components for building sparse
mixture-of-experts architectures.
"""

from esper.blueprints.metadata import BlueprintCategory


BLUEPRINTS = [
    {
        "metadata": {
            "blueprint_id": "BP-ROUTER-TOP2",
            "name": "Top-2 Router",
            "version": "1.0.0",
            "category": BlueprintCategory.MIXTURE_OF_EXPERTS,
            "description": "Top-2 routing with 256-way choice and noisy gating",
            "param_delta": 196608,  # 768 * 256 parameters
            "flop_delta": 393216,
            "memory_footprint_kb": 768,
            "expected_latency_ms": 0.5,
            "past_accuracy_gain_estimate": 0.025,  # MoE provides good gains
            "stability_improvement_estimate": 0.005,
            "speed_improvement_estimate": 0.15,  # Sparse computation
            "risk_score": 0.3,  # Routing can be unstable
            "is_safe_action": True,
            "requires_capability": [],
            "compatible_layers": ["Linear"],
            "incompatible_with": [],
            "mergeable": False,
            "warmup_steps": 500,  # Needs warmup for balanced routing
            "peak_benefit_window": 25000,
        },
        "architecture": {
            "module_type": "torch.nn.Module",
            "config": {
                "input_dim": 768,
                "num_experts": 256,
                "top_k": 2,
                "noise_std": 0.1,
                "dropout": 0.1,
            },
            "init_params": {},
            "forward_logic": """
class TopKRouter(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2, noise_std=0.1, dropout=0.1):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k
        self.noise_std = noise_std
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        logits = self.gate(x)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        return top_k_gates, top_k_indices
""",
        },
        "validation": {
            "min_experts": 8,
            "max_experts": 512,
            "max_top_k": 4,
        },
    },
    {
        "metadata": {
            "blueprint_id": "BP-EXPERT-MLP-1D",
            "name": "Plain Expert MLP",
            "version": "1.0.0",
            "category": BlueprintCategory.MIXTURE_OF_EXPERTS,
            "description": "Individual expert MLP with GELU activation",
            "param_delta": 2359296,  # Single expert parameters
            "flop_delta": 4718592,
            "memory_footprint_kb": 9216,
            "expected_latency_ms": 1.0,
            "past_accuracy_gain_estimate": 0.02,
            "stability_improvement_estimate": 0.01,
            "speed_improvement_estimate": 0.0,  # Neutral when routed
            "risk_score": 0.1,
            "is_safe_action": True,
            "requires_capability": [],
            "compatible_layers": ["Linear"],
            "incompatible_with": ["BP-EXPERT-MLP-SWIGLU"],
            "mergeable": False,
            "warmup_steps": 100,
            "peak_benefit_window": 20000,
        },
        "architecture": {
            "module_type": "torch.nn.Sequential",
            "config": {
                "layers": [
                    {"type": "Linear", "in_features": 768, "out_features": 3072},
                    {"type": "GELU"},
                    {"type": "Dropout", "p": 0.1},
                    {"type": "Linear", "in_features": 3072, "out_features": 768},
                ]
            },
            "init_params": {},
        },
        "validation": {
            "min_hidden_dim": 256,
            "max_hidden_dim": 8192,
        },
    },
    {
        "metadata": {
            "blueprint_id": "BP-EXPERT-MLP-SWIGLU",
            "name": "SwiGLU Expert",
            "version": "1.0.0",
            "category": BlueprintCategory.MIXTURE_OF_EXPERTS,
            "description": "Expert MLP with SwiGLU activation for better performance",
            "param_delta": 3145728,
            "flop_delta": 6291456,
            "memory_footprint_kb": 12288,
            "expected_latency_ms": 1.5,
            "past_accuracy_gain_estimate": 0.025,  # Better than GELU expert
            "stability_improvement_estimate": 0.012,
            "speed_improvement_estimate": -0.05,
            "risk_score": 0.15,
            "is_safe_action": True,
            "requires_capability": [],
            "compatible_layers": ["Linear"],
            "incompatible_with": ["BP-EXPERT-MLP-1D"],
            "mergeable": False,
            "warmup_steps": 150,
            "peak_benefit_window": 25000,
        },
        "architecture": {
            "module_type": "torch.nn.Module",
            "config": {
                "hidden_dim": 768,
                "intermediate_dim": 3072,
                "dropout": 0.1,
            },
            "init_params": {},
            "forward_logic": """
class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim)
        self.w2 = nn.Linear(hidden_dim, intermediate_dim)
        self.w3 = nn.Linear(intermediate_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))
""",
        },
        "validation": {
            "min_hidden_dim": 256,
            "max_hidden_dim": 8192,
        },
    },
    {
        "metadata": {
            "blueprint_id": "BP-CAP-FUSE-HFFN",
            "name": "Capacity Factor Fusion Block",
            "version": "1.0.0",
            "category": BlueprintCategory.MIXTURE_OF_EXPERTS,
            "description": "Micro-batch reordering for reduced routing overhead",
            "param_delta": 0,  # No new parameters
            "flop_delta": 1000,  # Minimal overhead
            "memory_footprint_kb": 256,
            "expected_latency_ms": 0.2,
            "past_accuracy_gain_estimate": 0.003,
            "stability_improvement_estimate": 0.015,  # Improves routing stability
            "speed_improvement_estimate": 0.1,  # Reduces overhead
            "risk_score": 0.2,
            "is_safe_action": True,
            "requires_capability": [],
            "compatible_layers": ["Linear"],
            "incompatible_with": [],
            "mergeable": True,
            "warmup_steps": 0,
            "peak_benefit_window": 50000,
        },
        "architecture": {
            "module_type": "torch.nn.Module",
            "config": {
                "capacity_factor": 1.25,
                "min_capacity": 4,
                "drop_tokens": False,
            },
            "init_params": {},
            "forward_logic": """
class CapacityFusion(nn.Module):
    def __init__(self, capacity_factor=1.25, min_capacity=4, drop_tokens=False):
        super().__init__()
        self.capacity_factor = capacity_factor
        self.min_capacity = min_capacity
        self.drop_tokens = drop_tokens
    
    def forward(self, x, expert_indices):
        # Reorder tokens by expert assignment for efficiency
        batch_size, seq_len = x.shape[:2]
        capacity = max(int(seq_len * self.capacity_factor), self.min_capacity)
        
        # Implementation would handle token-to-expert assignment
        # This is a simplified placeholder
        return x
""",
        },
        "validation": {
            "min_capacity_factor": 1.0,
            "max_capacity_factor": 2.0,
        },
    },
    {
        "metadata": {
            "blueprint_id": "BP-ROUTER-MOE-AUXLOSS",
            "name": "Router with Auxiliary Loss",
            "version": "1.0.0",
            "category": BlueprintCategory.MIXTURE_OF_EXPERTS,
            "description": "Router with pre-wired auxiliary load-balancing loss",
            "param_delta": 196608,
            "flop_delta": 400000,
            "memory_footprint_kb": 800,
            "expected_latency_ms": 0.6,
            "past_accuracy_gain_estimate": 0.028,  # Better balanced routing
            "stability_improvement_estimate": 0.02,
            "speed_improvement_estimate": 0.12,
            "risk_score": 0.25,
            "is_safe_action": True,
            "requires_capability": [],
            "compatible_layers": ["Linear"],
            "incompatible_with": ["BP-ROUTER-TOP2"],
            "mergeable": False,
            "warmup_steps": 300,
            "peak_benefit_window": 30000,
        },
        "architecture": {
            "module_type": "torch.nn.Module",
            "config": {
                "input_dim": 768,
                "num_experts": 256,
                "top_k": 2,
                "aux_loss_weight": 0.01,
                "noise_std": 0.1,
            },
            "init_params": {},
            "forward_logic": """
class RouterWithAuxLoss(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2, aux_loss_weight=0.01, noise_std=0.1):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.noise_std = noise_std
    
    def forward(self, x):
        logits = self.gate(x)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        
        # Compute routing
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        # Compute auxiliary loss for load balancing
        gates = F.softmax(logits, dim=-1)
        expert_counts = gates.mean(dim=0)
        aux_loss = expert_counts.var() * self.aux_loss_weight
        
        return top_k_gates, top_k_indices, aux_loss
""",
        },
        "validation": {
            "min_experts": 8,
            "max_experts": 512,
        },
    },
]