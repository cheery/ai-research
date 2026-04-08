---
type: concept
tags: [discrete-diffusion, text-generation]
papers: [ebm]
created: 2026-04-08
updated: 2026-04-08
---

# Discrete Diffusion Models

## Definition

Discrete diffusion models generate sequences (text, discrete data) by starting from noise (typically a fully masked sequence) and iteratively denoising toward a clean sample, operating over discrete token spaces rather than continuous ones.

## Why it matters

Unlike autoregressive models, discrete diffusion enables parallel token generation (no fixed left-to-right order), bidirectional controllable generation, and potential sampling acceleration. However, they currently underperform AR models, and closing this gap is an active research frontier.

## How it works

The framework extends continuous diffusion to discrete spaces. A forward process gradually corrupts clean data x_0 by transitioning tokens toward a reference distribution (typically a mask/absorbing state). A backward model learns to reverse this process.

**Forward process:** At each step, tokens transition to the mask state with increasing probability: q(x_t|x_0) = alpha_t * x_0 + (1 - alpha_t) * mask. The posterior q(x_s|x_t, x_0) is tractable in closed form.

**Backward model:** Learns p_theta(x_0|x_t) to approximate q(x_0|x_t). In practice, this is parameterized as independent per-token predictions: p_theta(x_0|x_t) = Prod_i p_theta(x_0^i|x_t), enabling parallel decoding but ignoring inter-token dependencies.

**Key limitation:** The per-token factorization creates a mismatch between training and sampling distributions. The model cannot match the true joint posterior q(x_0|x_t), causing accumulated decoding errors that worsen with fewer sampling steps.

## Variants and extensions

- **D3PM** (Austin et al., 2021): Original discrete diffusion framework with various transition matrices (uniform, absorbing). Extended continuous-time formulation by Campbell et al. (2022).
- **SEDD** (Lou et al., 2024): Score entropy formulation for discrete diffusion, estimating ratios of the data distribution.
- **MDLM** (Sahoo et al., 2024): Simplified masked diffusion with continuous-time training. Strong baseline at GPT2 scale.
- **MD4** (Shi et al., 2024): Generalized masked diffusion with simplified objectives.
- **EDLM** ([[ebm]]): Adds a sequence-level energy-based correction on top of pretrained diffusion models. Residual EBM formulation p_EDLM = p_diffusion * exp(-E). First to match AR perplexity.

## Key papers

- [[ebm]] — identifies the factorization mismatch, proposes residual EBM correction, matches AR perplexity

## Current state

Discrete diffusion has made rapid progress (D3PM → SEDD → MDLM → EDLM) but still trails AR models in practice. [[ebm]] shows the gap can be closed at small scale (GPT2-small) via energy-based corrections, but scaling to large models remains untested. The parallel generation capability and controllability advantages have not yet been fully exploited in practical applications.
