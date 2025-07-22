"""
Efficiency and compression blueprint templates.

Provides parameter-efficient adaptation methods like LoRA and IA3
for memory-constrained morphogenetic evolution.
"""

from esper.blueprints.metadata import BlueprintCategory

BLUEPRINTS = [
    {
        "metadata": {
            "blueprint_id": "BP-PROJ-LoRA-64",
            "name": "Low-Rank Adapter (LoRA)",
            "version": "1.0.0",
            "category": BlueprintCategory.EFFICIENCY,
            "description": "Low-rank adaptation with rank-64 matrices for efficient capacity injection",
            "param_delta": 98304,  # 768 * 64 * 2 parameters
            "flop_delta": 196608,
            "memory_footprint_kb": 384,
            "expected_latency_ms": 0.3,
            "past_accuracy_gain_estimate": 0.012,  # Good gains for low cost
            "stability_improvement_estimate": 0.01,
            "speed_improvement_estimate": -0.02,  # Minimal slowdown
            "risk_score": 0.1,
            "is_safe_action": True,
            "requires_capability": [],
            "compatible_layers": ["Linear"],
            "incompatible_with": [],
            "mergeable": True,  # Can stack multiple LoRAs
            "warmup_steps": 50,
            "peak_benefit_window": 10000,
        },
        "architecture": {
            "module_type": "torch.nn.Module",
            "config": {
                "in_features": 768,
                "out_features": 768,
                "rank": 64,
                "alpha": 16.0,
                "dropout": 0.1,
            },
            "init_params": {},
            "forward_logic": """
class LoRAAdapter(nn.Module):
    def __init__(self, in_features, out_features, rank=64, alpha=16.0, dropout=0.1):
        super().__init__()
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        
        # Initialize A with random, B with zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x, base_output=None):
        lora_output = self.lora_B(self.dropout(self.lora_A(x))) * self.scale
        if base_output is not None:
            return base_output + lora_output
        return lora_output
""",
        },
        "validation": {
            "min_rank": 1,
            "max_rank": 256,
            "rank_ratio": 0.25,  # rank should be < 25% of min(in, out)
        },
    },
    {
        "metadata": {
            "blueprint_id": "BP-PROJ-IA3",
            "name": "IA³ Gain Vector",
            "version": "1.0.0",
            "category": BlueprintCategory.EFFICIENCY,
            "description": "Per-channel multiplicative gain vector with near-zero parameters",
            "param_delta": 768,  # Just one vector
            "flop_delta": 768,
            "memory_footprint_kb": 4,
            "expected_latency_ms": 0.05,
            "past_accuracy_gain_estimate": 0.008,
            "stability_improvement_estimate": 0.005,
            "speed_improvement_estimate": 0.01,  # Very fast
            "risk_score": 0.05,
            "is_safe_action": True,
            "requires_capability": [],
            "compatible_layers": ["Linear"],
            "incompatible_with": [],
            "mergeable": True,
            "warmup_steps": 20,
            "peak_benefit_window": 5000,
        },
        "architecture": {
            "module_type": "torch.nn.Module",
            "config": {
                "num_features": 768,
                "init_scale": 1.0,
            },
            "init_params": {},
            "forward_logic": """
class IA3Adapter(nn.Module):
    def __init__(self, num_features, init_scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_features) * init_scale)
    
    def forward(self, x, base_output=None):
        if base_output is not None:
            return base_output * self.scale
        return x * self.scale
""",
        },
        "validation": {
            "min_features": 64,
            "max_features": 8192,
        },
    },
    {
        "metadata": {
            "blueprint_id": "BP-KV-CACHE-8BIT",
            "name": "8-bit Quantized KV Cache",
            "version": "1.0.0",
            "category": BlueprintCategory.EFFICIENCY,
            "description": "8-bit quantization for key-value cache with dequant-on-the-fly",
            "param_delta": 0,  # No new parameters
            "flop_delta": 1024,  # Quantization overhead
            "memory_footprint_kb": -3072,  # Saves memory!
            "expected_latency_ms": 0.2,
            "past_accuracy_gain_estimate": -0.002,  # Slight accuracy loss
            "stability_improvement_estimate": 0.0,
            "speed_improvement_estimate": 0.05,  # Memory bandwidth improvement
            "risk_score": 0.2,
            "is_safe_action": True,
            "requires_capability": [],
            "compatible_layers": ["MultiheadAttention"],
            "incompatible_with": [],
            "mergeable": False,
            "warmup_steps": 0,
            "peak_benefit_window": 100000,
        },
        "architecture": {
            "module_type": "torch.nn.Module",
            "config": {
                "num_bits": 8,
                "symmetric": True,
                "per_channel": True,
            },
            "init_params": {},
            "forward_logic": """
class QuantizedKVCache(nn.Module):
    def __init__(self, num_bits=8, symmetric=True, per_channel=True):
        super().__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.register_buffer('scale', None)
        self.register_buffer('zero_point', None)
    
    def quantize(self, x):
        if self.per_channel:
            scale = x.abs().max(dim=-1, keepdim=True)[0] / 127.0
        else:
            scale = x.abs().max() / 127.0
        
        x_int8 = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
        return x_int8, scale
    
    def dequantize(self, x_int8, scale):
        return x_int8.float() * scale
    
    def forward(self, k, v):
        k_int8, k_scale = self.quantize(k)
        v_int8, v_scale = self.quantize(v)
        
        # Store quantized values (would be cached in practice)
        # Dequantize on access
        k_deq = self.dequantize(k_int8, k_scale)
        v_deq = self.dequantize(v_int8, v_scale)
        
        return k_deq, v_deq
""",
        },
        "validation": {
            "min_bits": 4,
            "max_bits": 8,
        },
    },
]
