"""
Transformer architecture blueprint templates.

Provides attention mechanisms and feed-forward network blueprints
for morphogenetic adaptation of transformer models.
"""

from esper.blueprints.metadata import BlueprintCategory

# Blueprint definitions for transformer components
BLUEPRINTS = [
    {
        "metadata": {
            "blueprint_id": "BP-ATTN-STD",
            "name": "Standard Multi-Head Self-Attention",
            "version": "1.0.0",
            "category": BlueprintCategory.TRANSFORMER,
            "description": "Standard multi-head self-attention with 12-16 heads and rotary positional encoding",
            "param_delta": 1048576,  # ~1M parameters for typical config
            "flop_delta": 2097152,  # ~2M FLOPs
            "memory_footprint_kb": 4096,
            "expected_latency_ms": 2.5,
            "past_accuracy_gain_estimate": 0.015,  # 1.5% improvement
            "stability_improvement_estimate": 0.02,
            "speed_improvement_estimate": -0.1,  # Slight slowdown
            "risk_score": 0.2,
            "is_safe_action": True,
            "requires_capability": [],
            "compatible_layers": ["Linear", "MultiheadAttention"],
            "incompatible_with": ["BP-ATTN-SPLIT-QK"],
            "mergeable": True,
            "warmup_steps": 100,
            "peak_benefit_window": 10000,
        },
        "architecture": {
            "module_type": "torch.nn.MultiheadAttention",
            "config": {
                "embed_dim": 768,
                "num_heads": 12,
                "dropout": 0.1,
                "batch_first": True,
            },
            "init_params": {
                "bias": True,
                "add_bias_kv": False,
                "add_zero_attn": False,
            },
        },
        "validation": {
            "min_embed_dim": 256,
            "max_embed_dim": 2048,
            "divisible_by_heads": True,
        },
    },
    {
        "metadata": {
            "blueprint_id": "BP-ATTN-SPLIT-QK",
            "name": "Split Q/KV Attention Block",
            "version": "1.0.0",
            "category": BlueprintCategory.TRANSFORMER,
            "description": "Attention with separate Q and K/V capacity for efficiency",
            "param_delta": 786432,  # ~768K parameters
            "flop_delta": 1572864,
            "memory_footprint_kb": 3072,
            "expected_latency_ms": 2.0,
            "past_accuracy_gain_estimate": 0.012,
            "stability_improvement_estimate": 0.015,
            "speed_improvement_estimate": 0.05,  # 5% speedup
            "risk_score": 0.25,
            "is_safe_action": True,
            "requires_capability": [],
            "compatible_layers": ["Linear", "MultiheadAttention"],
            "incompatible_with": ["BP-ATTN-STD"],
            "mergeable": True,
            "warmup_steps": 150,
            "peak_benefit_window": 8000,
        },
        "architecture": {
            "module_type": "torch.nn.MultiheadAttention",
            "config": {
                "embed_dim": 768,
                "num_heads": 12,
                "dropout": 0.1,
                "batch_first": True,
                "kdim": 384,  # Reduced K/V dimension
                "vdim": 384,
            },
            "init_params": {
                "bias": True,
            },
        },
        "validation": {
            "min_embed_dim": 256,
            "max_embed_dim": 2048,
        },
    },
    {
        "metadata": {
            "blueprint_id": "BP-MLP-GELU-2048",
            "name": "Feed-Forward Network with GELU",
            "version": "1.0.0",
            "category": BlueprintCategory.TRANSFORMER,
            "description": "Standard FFN with 4x expansion and GELU activation",
            "param_delta": 3145728,  # ~3M parameters
            "flop_delta": 6291456,
            "memory_footprint_kb": 12288,
            "expected_latency_ms": 1.5,
            "past_accuracy_gain_estimate": 0.018,
            "stability_improvement_estimate": 0.01,
            "speed_improvement_estimate": -0.05,
            "risk_score": 0.1,
            "is_safe_action": True,
            "requires_capability": [],
            "compatible_layers": ["Linear"],
            "incompatible_with": ["BP-MLP-SWIGLU-4096"],
            "mergeable": False,
            "warmup_steps": 50,
            "peak_benefit_window": 15000,
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
            "blueprint_id": "BP-MLP-SWIGLU-4096",
            "name": "SwiGLU Feed-Forward Network",
            "version": "1.0.0",
            "category": BlueprintCategory.TRANSFORMER,
            "description": "Modern FFN with SwiGLU activation for better scaling",
            "param_delta": 4194304,  # ~4M parameters
            "flop_delta": 8388608,
            "memory_footprint_kb": 16384,
            "expected_latency_ms": 2.0,
            "past_accuracy_gain_estimate": 0.022,  # Better than GELU
            "stability_improvement_estimate": 0.012,
            "speed_improvement_estimate": -0.08,
            "risk_score": 0.15,
            "is_safe_action": True,
            "requires_capability": [],
            "compatible_layers": ["Linear"],
            "incompatible_with": ["BP-MLP-GELU-2048"],
            "mergeable": False,
            "warmup_steps": 100,
            "peak_benefit_window": 20000,
        },
        "architecture": {
            "module_type": "torch.nn.Module",  # Custom SwiGLU implementation
            "config": {
                "hidden_dim": 768,
                "intermediate_dim": 4096,
                "dropout": 0.1,
            },
            "init_params": {},
            "forward_logic": """
class SwiGLU(nn.Module):
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
            "blueprint_id": "BP-LN-RMS",
            "name": "RMSNorm Layer",
            "version": "1.0.0",
            "category": BlueprintCategory.TRANSFORMER,
            "description": "RMS normalization (weights only, no bias) for GPT-NeoX style",
            "param_delta": 768,  # Just weight parameters
            "flop_delta": 1536,
            "memory_footprint_kb": 8,
            "expected_latency_ms": 0.1,
            "past_accuracy_gain_estimate": 0.005,
            "stability_improvement_estimate": 0.008,
            "speed_improvement_estimate": 0.02,  # Faster than LayerNorm
            "risk_score": 0.05,
            "is_safe_action": True,
            "requires_capability": [],
            "compatible_layers": ["LayerNorm"],
            "incompatible_with": [],
            "mergeable": True,
            "warmup_steps": 10,
            "peak_benefit_window": 50000,
        },
        "architecture": {
            "module_type": "torch.nn.Module",  # Custom RMSNorm
            "config": {
                "normalized_shape": 768,
                "eps": 1e-5,
            },
            "init_params": {},
            "forward_logic": """
class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps
    
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight
""",
        },
        "validation": {
            "min_dim": 64,
            "max_dim": 8192,
        },
    },
]
