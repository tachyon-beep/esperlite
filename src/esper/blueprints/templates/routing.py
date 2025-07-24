"""
Routing and scalability blueprint templates.

Provides distributed computing and load balancing components
for multi-GPU morphogenetic training.
"""

from esper.blueprints.metadata import BlueprintCategory

BLUEPRINTS = [
    {
        "metadata": {
            "blueprint_id": "BP-ALL-REDUCE-SHARD",
            "name": "Fused All-Reduce Shard Kernel",
            "version": "1.0.0",
            "category": BlueprintCategory.ROUTING,
            "description": "Optimized all-reduce for cross-GPU expert output aggregation",
            "param_delta": 0,  # No parameters
            "flop_delta": 2048,
            "memory_footprint_kb": 128,
            "expected_latency_ms": 1.0,  # Network dependent
            "past_accuracy_gain_estimate": 0.0,  # No accuracy impact
            "stability_improvement_estimate": 0.01,
            "speed_improvement_estimate": 0.2,  # Faster than naive reduce
            "risk_score": 0.15,
            "is_safe_action": True,
            "requires_capability": ["multi_gpu", "nccl"],
            "compatible_layers": ["Linear"],
            "incompatible_with": [],
            "mergeable": True,
            "warmup_steps": 0,
            "peak_benefit_window": 100000,
        },
        "architecture": {
            "module_type": "torch.nn.Module",
            "config": {
                "reduce_op": "sum",
                "async_op": True,
                "bucket_cap_mb": 25,
            },
            "init_params": {},
            "forward_logic": """
class FusedAllReduceShard(nn.Module):
    def __init__(self, reduce_op='sum', async_op=True, bucket_cap_mb=25):
        super().__init__()
        self.reduce_op = reduce_op
        self.async_op = async_op
        self.bucket_cap_mb = bucket_cap_mb
    
    def forward(self, x):
        if torch.distributed.is_initialized():
            # Efficient bucketed all-reduce
            handle = torch.distributed.all_reduce(
                x, op=getattr(torch.distributed.ReduceOp, self.reduce_op.upper()),
                async_op=self.async_op
            )
            if self.async_op:
                return x, handle
            else:
                return x
        return x
""",
        },
        "validation": {
            "supported_ops": ["sum", "mean", "max", "min"],
        },
    },
    {
        "metadata": {
            "blueprint_id": "BP-LOAD-BAL-EMA",
            "name": "EMA Load Balancer",
            "version": "1.0.0",
            "category": BlueprintCategory.ROUTING,
            "description": "Exponential moving average load balancer for GPU utilization",
            "param_delta": 1024,  # Small state tracking
            "flop_delta": 512,
            "memory_footprint_kb": 8,
            "expected_latency_ms": 0.1,
            "past_accuracy_gain_estimate": 0.002,
            "stability_improvement_estimate": 0.02,  # Better stability
            "speed_improvement_estimate": 0.15,  # Better GPU utilization
            "risk_score": 0.1,
            "is_safe_action": True,
            "requires_capability": [],
            "compatible_layers": ["Linear"],
            "incompatible_with": [],
            "mergeable": True,
            "warmup_steps": 100,
            "peak_benefit_window": 50000,
        },
        "architecture": {
            "module_type": "torch.nn.Module",
            "config": {
                "num_experts": 256,
                "ema_decay": 0.99,
                "balance_loss_weight": 0.01,
            },
            "init_params": {},
            "forward_logic": """
class EMALoadBalancer(nn.Module):
    def __init__(self, num_experts, ema_decay=0.99, balance_loss_weight=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.ema_decay = ema_decay
        self.balance_loss_weight = balance_loss_weight
        
        # Track expert usage with EMA
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('update_count', torch.tensor(0))
    
    def update_usage(self, expert_indices):
        # Count expert usage in batch
        usage = torch.zeros_like(self.expert_usage)
        indices, counts = expert_indices.unique(return_counts=True)
        usage[indices] = counts.float()
        
        # Update EMA
        if self.update_count > 0:
            self.expert_usage = self.ema_decay * self.expert_usage + (1 - self.ema_decay) * usage
        else:
            self.expert_usage = usage
        
        self.update_count += 1
    
    def get_balance_loss(self):
        # Compute load balancing loss
        target_usage = self.expert_usage.sum() / self.num_experts
        variance = ((self.expert_usage - target_usage) ** 2).mean()
        return variance * self.balance_loss_weight
    
    def forward(self, logits, expert_indices):
        self.update_usage(expert_indices)
        
        # Adjust logits based on usage to encourage balance
        usage_penalty = (self.expert_usage / self.expert_usage.mean()).log()
        adjusted_logits = logits - usage_penalty.unsqueeze(0)
        
        return adjusted_logits
""",
        },
        "validation": {
            "min_experts": 2,
            "max_experts": 1024,
        },
    },
    {
        "metadata": {
            "blueprint_id": "BP-RING-ALLREDUCE",
            "name": "Ring All-Reduce Communication",
            "version": "1.0.0",
            "category": BlueprintCategory.ROUTING,
            "description": "Efficient ring-based all-reduce for distributed gradient aggregation",
            "param_delta": 0,
            "flop_delta": 4096,
            "memory_footprint_kb": 256,
            "expected_latency_ms": 2.0,
            "past_accuracy_gain_estimate": 0.0,
            "stability_improvement_estimate": 0.02,
            "speed_improvement_estimate": 0.25,
            "risk_score": 0.1,
            "is_safe_action": True,
            "requires_capability": ["multi_gpu", "ring_comm"],
            "compatible_layers": ["Linear", "Attention", "MLP"],
            "incompatible_with": [],
            "mergeable": False,
            "warmup_steps": 0,
            "peak_benefit_window": 100000,
        },
        "architecture": {
            "module_type": "torch.nn.Module",
            "config": {
                "world_size": 8,
                "chunk_size": 1048576,
            },
            "init_params": {},
            "forward_logic": """
class RingAllReduce(nn.Module):
    def __init__(self, world_size=8, chunk_size=1048576):
        super().__init__()
        self.world_size = world_size
        self.chunk_size = chunk_size
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
    def ring_allreduce(self, tensor):
        if not torch.distributed.is_initialized():
            return tensor
            
        # Flatten tensor for communication
        original_shape = tensor.shape
        tensor = tensor.flatten()
        
        # Split into chunks for ring communication
        chunks = tensor.split(self.chunk_size)
        result_chunks = []
        
        for chunk in chunks:
            # Ring reduce: each rank sends to next, receives from previous
            send_rank = (self.rank + 1) % self.world_size
            recv_rank = (self.rank - 1) % self.world_size
            
            # Accumulate through ring
            accumulated = chunk.clone()
            for step in range(self.world_size - 1):
                torch.distributed.send(accumulated, send_rank)
                received = torch.zeros_like(accumulated)
                torch.distributed.recv(received, recv_rank)
                accumulated += received
            
            # Broadcast final result through ring
            for step in range(self.world_size - 1):
                torch.distributed.send(accumulated, send_rank)
                torch.distributed.recv(accumulated, recv_rank)
            
            result_chunks.append(accumulated / self.world_size)
        
        # Reconstruct tensor
        result = torch.cat(result_chunks).reshape(original_shape)
        return result
    
    def forward(self, x):
        if hasattr(x, 'grad'):
            x.grad = self.ring_allreduce(x.grad)
        return x
""",
        },
        "validation": {
            "min_world_size": 2,
            "max_world_size": 128,
        },
    },
]
