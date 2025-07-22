"""
Diagnostic and safety blueprint templates.

Provides monitoring, debugging, and safety mechanisms for
morphogenetic adaptations.
"""

from esper.blueprints.metadata import BlueprintCategory


BLUEPRINTS = [
    {
        "metadata": {
            "blueprint_id": "BP-MON-ACT-STATS",
            "name": "Activation Statistics Monitor",
            "version": "1.0.0",
            "category": BlueprintCategory.DIAGNOSTICS,
            "description": "No-grad monitor that records activation statistics per step",
            "param_delta": 0,  # No trainable parameters
            "flop_delta": 256,  # Minimal computation
            "memory_footprint_kb": 64,
            "expected_latency_ms": 0.1,
            "past_accuracy_gain_estimate": 0.0,  # Pure diagnostic
            "stability_improvement_estimate": 0.005,  # Helps detect issues
            "speed_improvement_estimate": -0.01,  # Minimal overhead
            "risk_score": 0.0,  # Completely safe
            "is_safe_action": True,
            "requires_capability": [],
            "compatible_layers": ["Linear", "Conv2d", "MultiheadAttention"],
            "incompatible_with": [],
            "mergeable": True,
            "warmup_steps": 0,
            "peak_benefit_window": 100000,
        },
        "architecture": {
            "module_type": "torch.nn.Module",
            "config": {
                "track_mean": True,
                "track_std": True,
                "track_histogram": False,
                "window_size": 100,
            },
            "init_params": {},
            "forward_logic": """
class ActivationMonitor(nn.Module):
    def __init__(self, track_mean=True, track_std=True, track_histogram=False, window_size=100):
        super().__init__()
        self.track_mean = track_mean
        self.track_std = track_std
        self.track_histogram = track_histogram
        self.window_size = window_size
        
        # Buffers for statistics
        self.register_buffer('step_count', torch.tensor(0))
        if track_mean:
            self.register_buffer('running_mean', torch.tensor(0.0))
        if track_std:
            self.register_buffer('running_std', torch.tensor(0.0))
    
    def forward(self, x):
        with torch.no_grad():
            # Compute statistics
            if self.track_mean:
                mean = x.mean()
                if self.step_count > 0:
                    self.running_mean = (self.running_mean * self.step_count + mean) / (self.step_count + 1)
                else:
                    self.running_mean = mean
            
            if self.track_std:
                std = x.std()
                if self.step_count > 0:
                    self.running_std = (self.running_std * self.step_count + std) / (self.step_count + 1)
                else:
                    self.running_std = std
            
            self.step_count += 1
            
            # Log if anomaly detected
            if self.track_mean and abs(mean) > 100:
                print(f"Warning: Large activation mean detected: {mean}")
            if self.track_std and std < 0.01:
                print(f"Warning: Low activation variance detected: {std}")
        
        return x  # Pass through unchanged
""",
        },
        "validation": {
            "max_window_size": 10000,
        },
    },
    {
        "metadata": {
            "blueprint_id": "BP-CLAMP-GRAD-NORM",
            "name": "Gradient Clamp Wrapper",
            "version": "1.0.0",
            "category": BlueprintCategory.DIAGNOSTICS,
            "description": "Safety wrapper that clips gradients to prevent explosion",
            "param_delta": 0,
            "flop_delta": 100,
            "memory_footprint_kb": 4,
            "expected_latency_ms": 0.05,
            "past_accuracy_gain_estimate": 0.0,
            "stability_improvement_estimate": 0.03,  # Prevents instability
            "speed_improvement_estimate": 0.0,
            "risk_score": 0.0,
            "is_safe_action": True,
            "requires_capability": [],
            "compatible_layers": ["Linear", "Conv2d", "MultiheadAttention"],
            "incompatible_with": [],
            "mergeable": True,
            "warmup_steps": 0,
            "peak_benefit_window": 100000,
        },
        "architecture": {
            "module_type": "torch.nn.Module",
            "config": {
                "max_norm": 1.0,
                "norm_type": 2.0,
                "error_if_nonfinite": True,
            },
            "init_params": {},
            "forward_logic": """
class GradientClampWrapper(nn.Module):
    def __init__(self, module, max_norm=1.0, norm_type=2.0, error_if_nonfinite=True):
        super().__init__()
        self.module = module
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.error_if_nonfinite = error_if_nonfinite
        
        # Register gradient clipping hook
        for param in self.module.parameters():
            param.register_hook(self._clip_grad_hook)
    
    def _clip_grad_hook(self, grad):
        if grad is not None:
            # Check for non-finite values
            if self.error_if_nonfinite and not grad.isfinite().all():
                raise RuntimeError("Non-finite gradient detected")
            
            # Clip gradient norm
            grad_norm = grad.norm(self.norm_type)
            if grad_norm > self.max_norm:
                grad = grad * (self.max_norm / grad_norm)
        
        return grad
    
    def forward(self, x):
        return self.module(x)
""",
        },
        "validation": {
            "min_max_norm": 0.1,
            "max_max_norm": 100.0,
        },
    },
]