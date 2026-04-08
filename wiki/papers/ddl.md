---
type: paper
source: raw/ddl.pdf
title: "Deep Delta Learning"
authors: [Yifan Zhang, Yifeng Liu, Mengdi Wang, Quanquan Gu]
year: 2026
tags: [architecture-design, transformer, language-modeling, residual-learning]
ingested: 2026-04-08
---

# Deep Delta Learning

## One-line summary

DDL generalizes the Transformer residual connection from a fixed identity shortcut to a learnable rank-1 perturbation of the identity, enabling the network to selectively erase and rewrite feature components layer by layer.

## Key contributions

1. Proposes the Delta Operator A(X) = I - β(X)k(X)k(X)^T, a rank-1 perturbation of the identity that generalizes the residual shortcut
2. Provides spectral analysis showing β continuously interpolates between identity (β=0), orthogonal projection (β=1), and Householder reflection (β=2)
3. Unifies the residual update as a synchronized rank-1 delta write: the Delta Rule (Widrow-Hoff) applied to network depth
4. Introduces an expanded-state regime (d_v > 1) that treats the residual stream as a matrix-valued memory, decoupling memory capacity from compute width
5. Shows consistent improvements on language modeling at 124M and 353M scales as a drop-in replacement for residual additions

## Core ideas

Standard residual connections treat each layer as a forward Euler step for the ODE X' = F(X). This imposes a strictly additive, translational bias — information accumulates without a mechanism to selectively remove interfering features. DDL replaces the identity shortcut with A(X) = I - β(X)k(X)k(X)^T, where k(X) is a learned unit direction and β(X) is a learned scalar gate in (0, 2).

The full update is X_{l+1} = A(X_l)X_l + β(X_l)k(X_l)v(X_l)^T, which simplifies to a rank-1 delta: X_{l+1} = X_l + β(X_l)k(X_l)(v(X_l)^T - k(X_l)^T X_l). The term k(X)^T X reads the current projection along direction k; β scales both the removal of that component and the injection of a new one specified by v(X)^T. This is algebraically equivalent to the DeltaNet recurrence — DDL is the depth-wise isomorphism of DeltaNet's time-wise update.

The spectral analysis (Theorem 3.1) is clean: A has eigenvalue 1 with multiplicity d-1 (the subspace orthogonal to k is untouched) and eigenvalue 1-β along k. This means β=0 recovers standard residuals (identity), β=1 makes A a rank-(d-1) projector that erases the k-component before writing new content, and β=2 gives a full Householder reflection. The gate lets the network learn when to preserve, selectively forget, or flip feature directions.

For d_v > 1, the hidden state becomes a matrix X in R^{d x d_v}, treating it as a short-term memory of d_v slots per feature. The Delta Operator broadcasts across all value columns, applying the same geometric transformation synchronously. A compress-process-expand protocol interfaces this expanded state with standard Transformer sublayers.

## Methods

- **Backbone:** Llama-style (pre-norm RMSNorm, RoPE, SwiGLU MLP), only residual connection changed
- **Two parameterizations:** k-Map (backbone output determines direction k, linear projection gives value v) and v-Map (backbone output determines value v, auxiliary branch gives direction k)
- **β gate:** Computed via lightweight linear layer on normalized context, parameterized via 2*sigmoid(logit) to lie in (0, 2)
- **Normalization:** Precision-friendly RMS normalization with fixed scaling factor 1/sqrt(d) instead of explicit L2 norm
- **Training:** FineWeb-Edu 100B tokens, 124M and 353M scales, 4x H200 GPUs, AdamW, cosine LR schedule, muP initialization
- **Expanded state (d_v=4):** Embedding repeated across value channels, compress-process-expand protocol with depthwise causal convolution and learned read vector

## Results

- Valid loss (124M): 2.835 (d_v=4) vs 2.854 baseline
- Valid loss (353M): 2.593 (d_v=4) vs 2.605 baseline
- Downstream benchmarks (1-shot avg, 353M): 54.83 (d_v=4) vs 53.96 baseline
- DDL-EC variant (embedding convolution) reaches 49.47 avg at 124M vs 48.56 baseline
- d_v=1 also improves over baseline but d_v=4 gives strongest gains
- Adding channel convolution (CC) and embedding convolution (EC) provides further improvements

## Limitations noted by authors

- Gains are consistent but modest in absolute terms (~0.5-1% average benchmark improvement)
- Not evaluated at large scale (billion-parameter) or on non-language tasks
- The expanded state d_v > 1 increases memory footprint 4x without increasing attention FLOPs

## Connections

- Relates to [[residual-connections]] as a principled generalization of the identity shortcut
- Structural isomorphism with DeltaNet (Schlag et al., 2021) — same operator applied over depth instead of time
- Connects to Neural ODEs: standard ResNet is Euler for X' = F(X); DDL is Euler for X' = k(X)(v(X)^T - k(X)^T X) with adaptive step size β(X)
- Related to Hyper-Connections (Zhu et al., 2025) and mHC (Xie et al., 2025) which also modify the residual connection

## Open questions

1. How does DDL scale to billion-parameter models and long training runs?
2. Can the expanded-state regime (d_v > 1) be combined with mixture-of-experts or other capacity-scaling approaches?
3. Does DDL benefit non-language modalities (vision, multimodal)?

## PyTorch implementation sketch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaResidualBlock(nn.Module):
    """
    Drop-in replacement for Transformer residual addition.
    X_{l+1} = X_l + beta * k * (v^T - k^T @ X_l)
    where k is unit direction, beta in (0,2), v is value vector.
    """
    def __init__(self, d_model: int, d_v: int = 1):
        super().__init__()
        self.d_model = d_model
        self.d_v = d_v
        # Gate branch: outputs a single logit per token
        self.beta_linear = nn.Linear(d_model, 1)
        # Value branch: produces content to write
        self.v_proj = nn.Linear(d_model, d_v)
        # k scale for precision-friendly normalization
        self.k_scale = 1.0 / (d_model ** 0.5)

    def forward(self, x, fn_output):
        """
        x: (B, T, d_model) or (B, T, d_model, d_v) if expanded
        fn_output: (B, T, d_model) output of attention/MLP sublayer
        """
        B, T, d = x.shape[:3]
        ctx = x  # could add RMSNorm here

        # --- Direction k from sublayer output (k-Map variant) ---
        k_raw = fn_output                              # (B, T, d)
        k = F.normalize(k_raw, dim=-1)                 # unit direction

        # --- Gate beta in (0, 2) ---
        beta_logit = self.beta_linear(ctx.mean(dim=-1, keepdim=True)
                                      if ctx.dim() == 4 else ctx)
        beta = 2.0 * torch.sigmoid(beta_logit)         # (B, T, 1)

        # --- Value v ---
        x_in = ctx.mean(dim=-1) if ctx.dim() == 4 else ctx
        v = self.v_proj(x_in)                           # (B, T, d_v)

        # --- Delta update ---
        # k^T X: read current projection along k
        if self.d_v == 1:
            # Vector state: x is (B, T, d)
            k_dot_x = (k * x).sum(dim=-1, keepdim=True)  # (B, T, 1)
            correction = v - k_dot_x                       # (B, T, 1)
            delta = beta * k * correction                   # (B, T, d)
            return x + delta
        else:
            # Matrix state: x is (B, T, d, d_v)
            k_dot_X = torch.einsum('btd,btdv->btv', k, x)  # (B, T, d_v)
            correction = v - k_dot_X                         # (B, T, d_v)
            delta = beta * torch.einsum('btd,btv->btdv', k, correction)
            return x + delta


# --- Usage in Transformer block ---
# Standard:  x = x + attn(RMSNorm(x))
# With DDL:  x = delta_block(x, attn(RMSNorm(x)))
```
