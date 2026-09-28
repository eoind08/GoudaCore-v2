import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    max_seq_len: int = 4096
    n_layer: int = 20
    n_embd: int = 256
    n_head: int = 4
    n_kv_head: int = 2
    mlp_hidden: int = 768
    rope_base: float = 10000.0
    rms_eps: float = 1e-6
    qk_norm: bool = True
    z_loss: float = 0.0

    def __post_init__(self):
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_head % self.n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head")

    @property
    def head_dim(self):
        return self.n_embd // self.n_head


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


def precompute_rope(head_dim, max_seq_len, base=10000.0):
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")

    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)

    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    dtype = x.dtype
    x = x.float()

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    out = torch.stack(
        (
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ),
        dim=-1,
    ).flatten(-2)

    return out.to(dtype)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.n_rep = self.n_head // self.n_kv_head

        self.q_proj = nn.Linear(
            config.n_embd,
            self.n_head * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.n_embd,
            self.n_kv_head * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.n_embd,
            self.n_kv_head * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.n_head * self.head_dim,
            config.n_embd,
            bias=False,
        )

        self.q_norm = (
            RMSNorm(self.head_dim, config.rms_eps)
            if config.qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            RMSNorm(self.head_dim, config.rms_eps)
            if config.qk_norm
            else nn.Identity()
        )

    def forward(self, x, cos, sin):
        B, T, _ = x.shape

        q = self.q_proj(x).view(
            B, T, self.n_head, self.head_dim
        ).transpose(1, 2)

        k = self.k_proj(x).view(
            B, T, self.n_kv_head, self.head_dim
        ).transpose(1, 2)

        v = self.v_proj(x).view(
            B, T, self.n_kv_head, self.head_dim
        ).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.n_rep != 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, -1)

        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.gate_proj = nn.Linear(
            config.n_embd,
            config.mlp_hidden,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.n_embd,
            config.mlp_hidden,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.mlp_hidden,
            config.n_embd,
            bias=False,
        )

    def forward(self, x):
        return self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn_norm = RMSNorm(
            config.n_embd,
            config.rms_eps,
        )
        self.attn = Attention(config)

        self.mlp_norm = RMSNorm(
            config.n_embd,
            config.rms_eps,
        )
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(
            self.attn_norm(x),
            cos,
            sin,
        )
        x = x + self.mlp(
            self.mlp_norm(x)
        )
        return x


class Gouda(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.tok_emb = nn.Embedding(
            config.vocab_size,
            config.n_embd,
        )

        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)]
        )

        self.norm = RMSNorm(
            config.n_embd,
            config.rms_eps,
        )

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False,
        )

        self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope(
            config.head_dim,
            config.max_seq_len,
            config.rope_base,
        )

        self.register_buffer(
            "rope_cos",
            cos,
            persistent=False,
        )
        self.register_buffer(
            "rope_sin",
            sin,
            persistent=False,
        )

        self.apply(self._init_weights)

        for block in self.blocks:
            nn.init.normal_(
                block.attn.o_proj.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * config.n_layer),
            )
            nn.init.normal_(
                block.mlp.down_proj.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * config.n_layer),
            )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape

        if T > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {T} exceeds "
                f"max_seq_len={self.config.max_seq_len}"
            )

        x = self.tok_emb(input_ids)

        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.norm(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits

        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = targets.view(-1)

        loss = F.cross_entropy(
            flat_logits,
            flat_targets,
        )

        if self.config.z_loss:
            log_z = torch.logsumexp(
                flat_logits.float(),
                dim=-1,
            )
            loss = loss + self.config.z_loss * log_z.square().mean()

        return logits, loss

    def num_parameters(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())

        if non_embedding:
            n -= self.tok_emb.weight.numel()

        return n

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
    ):
        self.eval()

        for _ in range(max_new_tokens):
            x = input_ids[:, -self.config.max_seq_len:]
            logits = self(x)
            logits = logits[:, -1, :]

            if temperature == 0:
                next_token = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True,
                )
            else:
                logits = logits / temperature

                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    v, _ = torch.topk(logits, k)
                    logits[logits < v[:, [-1]]] = -float("inf")

                probs = F.softmax(logits, dim=-1)

                next_token = torch.multinomial(
                    probs,
                    num_samples=1,
                )

            input_ids = torch.cat(
                (input_ids, next_token),
                dim=1,
            )

        return input_ids