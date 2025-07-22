"""
KasminaAttentionLayer for Multi-Head Attention with morphogenetic capabilities.

This module provides a specialized KasminaLayer for PyTorch's MultiheadAttention
that preserves attention semantics while enabling dynamic kernel loading.
"""

import logging
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from esper.execution.kasmina_layer import KasminaLayer

logger = logging.getLogger(__name__)


class KasminaAttentionLayer(KasminaLayer):
    """
    KasminaLayer specialized for Multi-Head Attention mechanisms.

    This layer preserves the attention mechanism's behavior while adding
    morphogenetic capabilities for Q, K, V, and output projections.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        num_seeds: int = 4,
        cache_size_mb: int = 128,
        telemetry_enabled: bool = True,
        layer_name: str = "attention_layer",
    ):
        """
        Initialize KasminaAttentionLayer.

        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of parallel attention heads
            dropout: Dropout probability on attention weights
            bias: Whether to add bias to input/output projections
            add_bias_kv: Add bias to key and value sequences
            add_zero_attn: Add a new batch of zeros to key and value sequences
            kdim: Total number of features for keys (default: embed_dim)
            vdim: Total number of features for values (default: embed_dim)
            batch_first: If True, batch dimension is first
            num_seeds: Number of morphogenetic seeds
            cache_size_mb: Kernel cache size in MB
            telemetry_enabled: Whether to enable telemetry
            layer_name: Name of the layer for logging
        """
        # Initialize base KasminaLayer with combined Q+K+V projection size
        qkv_size = embed_dim * 3  # Q, K, V projections combined
        super().__init__(
            input_size=embed_dim,
            output_size=qkv_size,  # We'll handle output projection separately
            num_seeds=num_seeds,
            cache_size_mb=cache_size_mb,
            telemetry_enabled=telemetry_enabled,
            layer_name=layer_name,
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(
                f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
            )

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        # Create separate projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Dropout layer
        self.attn_dropout = nn.Dropout(dropout)

        # Additional attention parameters
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self._reset_parameters()

        logger.info(
            f"Created KasminaAttentionLayer: embed_dim={embed_dim}, "
            f"num_heads={num_heads}, head_dim={self.head_dim}"
        )

    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def copy_weights_from_attention(
        self, original_layer: nn.MultiheadAttention
    ) -> None:
        """
        Copy weights from original MultiheadAttention layer.

        Args:
            original_layer: The original MultiheadAttention layer
        """
        with torch.no_grad():
            # Copy Q, K, V projection weights
            if (
                hasattr(original_layer, "in_proj_weight")
                and original_layer.in_proj_weight is not None
            ):
                # Combined QKV projection
                embed_dim = self.embed_dim
                self.q_proj.weight.copy_(original_layer.in_proj_weight[:embed_dim])
                self.k_proj.weight.copy_(
                    original_layer.in_proj_weight[embed_dim : 2 * embed_dim]
                )
                self.v_proj.weight.copy_(original_layer.in_proj_weight[2 * embed_dim :])
            else:
                # Separate Q, K, V projections
                if hasattr(original_layer, "q_proj_weight"):
                    self.q_proj.weight.copy_(original_layer.q_proj_weight)
                if hasattr(original_layer, "k_proj_weight"):
                    self.k_proj.weight.copy_(original_layer.k_proj_weight)
                if hasattr(original_layer, "v_proj_weight"):
                    self.v_proj.weight.copy_(original_layer.v_proj_weight)

            # Copy biases if they exist
            if (
                hasattr(original_layer, "in_proj_bias")
                and original_layer.in_proj_bias is not None
            ):
                embed_dim = self.embed_dim
                self.q_proj.bias.copy_(original_layer.in_proj_bias[:embed_dim])
                self.k_proj.bias.copy_(
                    original_layer.in_proj_bias[embed_dim : 2 * embed_dim]
                )
                self.v_proj.bias.copy_(original_layer.in_proj_bias[2 * embed_dim :])

            # Copy output projection
            if hasattr(original_layer, "out_proj"):
                self.out_proj.weight.copy_(original_layer.out_proj.weight)
                if original_layer.out_proj.bias is not None:
                    self.out_proj.bias.copy_(original_layer.out_proj.bias)

            # Copy bias_k and bias_v if they exist
            if hasattr(original_layer, "bias_k") and original_layer.bias_k is not None:
                if self.bias_k is not None:
                    self.bias_k.copy_(original_layer.bias_k)
            if hasattr(original_layer, "bias_v") and original_layer.bias_v is not None:
                if self.bias_v is not None:
                    self.bias_v.copy_(original_layer.bias_v)

        logger.info(f"Copied weights from MultiheadAttention to {self.layer_name}")

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through attention mechanism.

        Args:
            query: Query embeddings
            key: Key embeddings (default: same as query)
            value: Value embeddings (default: same as query)
            key_padding_mask: Mask for padding tokens
            need_weights: Whether to return attention weights
            attn_mask: Attention mask
            average_attn_weights: Whether to average attention weights
            is_causal: Whether to apply causal mask

        Returns:
            Tuple of (output, attention_weights)
        """
        # Default key and value to query for self-attention
        if key is None:
            key = query
        if value is None:
            value = query

        # Handle batch_first dimension ordering
        if self.batch_first:
            query, key, value = [x.transpose(0, 1) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]

        # Apply Q, K, V projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Add bias_k and bias_v if specified
        if self.bias_k is not None:
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
        if self.bias_v is not None:
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])

        # Handle zero attention
        if self.add_zero_attn:
            zero_attn_shape = (1, bsz, embed_dim)
            k = torch.cat(
                [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=0
            )
            v = torch.cat(
                [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=0
            )
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        # Reshape for multi-head attention
        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # Update source length after adding bias_k/bias_v and zero_attn
        src_len = k.shape[1]

        # Scaled dot-product attention
        attn_output, attn_weights = self._scaled_dot_product_attention(
            q, k, v, attn_mask, key_padding_mask, is_causal
        )

        # Reshape back and apply output projection
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        )
        attn_output = self.out_proj(attn_output)

        # Handle batch_first dimension ordering
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        # Average attention weights if requested
        if need_weights and average_attn_weights:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.mean(dim=1)
        elif need_weights:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)

        return attn_output, attn_weights if need_weights else None

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.

        Args:
            q: Query tensor [batch*heads, tgt_len, head_dim]
            k: Key tensor [batch*heads, src_len, head_dim]
            v: Value tensor [batch*heads, src_len, head_dim]
            attn_mask: Attention mask
            key_padding_mask: Key padding mask
            is_causal: Whether to apply causal mask

        Returns:
            Tuple of (output, attention_weights)
        """
        bsz_heads, tgt_len, head_dim = q.shape
        bsz = bsz_heads // self.num_heads
        src_len = k.shape[1]

        # Compute attention scores
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / (head_dim**0.5)

        # Apply attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).expand(bsz, -1, -1)
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_mask = attn_mask.contiguous().view(
                bsz * self.num_heads, tgt_len, src_len
            )
            attn_weights += attn_mask

        # Apply key padding mask
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len)
            key_padding_mask = key_padding_mask.expand(-1, self.num_heads, tgt_len, -1)
            key_padding_mask = key_padding_mask.contiguous().view(
                bsz * self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))

        # Apply causal mask
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(tgt_len, src_len, device=q.device), diagonal=1
            ).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.bmm(attn_weights, v)

        return attn_output, attn_weights
