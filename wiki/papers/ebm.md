---
type: paper
source: raw/EBM.pdf
title: "Energy-based Diffusion Language Models for Text Generation"
authors: [Minkai Xu, Tomas Geffner, Karsten Kreis, Weili Nie, Yilun Xu, Jure Leskovec, Stefano Ermon, Arash Vahdat]
year: 2025
tags: [discrete-diffusion, text-generation, energy-based-models, language-modeling]
ingested: 2026-04-08
---

# Energy-based Diffusion Language Models for Text Generation

## One-line summary

EDLM fixes the token-independent factorization in discrete diffusion models by adding a sequence-level energy-based correction, enabling parallel text generation that matches autoregressive perplexity.

## Key contributions

1. Identifies the training-sampling distribution mismatch in discrete diffusion: factorized per-token denoising ignores sequence-level correlations, causing accumulated decoding errors
2. Proposes residual EBM formulation: p_EDLM(x_0|x_t) = p_diffusion(x_0|x_t) * exp(-E(x_0,x_t)) / Z, where E captures inter-token dependencies
3. Shows the energy can be obtained for free from pretrained AR models (no training), or by NCE fine-tuning of bidirectional transformers
4. Introduces efficient parallel generation via self-normalized importance sampling with a configurable window
5. First discrete diffusion model to match AR perplexity on OpenWebText

## Core ideas

Discrete diffusion models (D3PM, MDLM, SEDD) denoise sequences by predicting x_0 given x_t, but factorize the prediction independently per token: p_theta(x_0|x_t) = Prod_i p_theta(x_0^i|x_t). This is efficient for parallel decoding but ignores dependencies between tokens — the model cannot match the true posterior q(x_0|x_t), leading to errors that accumulate across denoising steps. The problem worsens with fewer steps.

EDLM adds a residual energy function on top of the pretrained diffusion model. The energy E(x_0, x_t) operates at the full sequence level and captures token correlations. The formulation is EBM * diffusion — no need to retrain the base model.

For the AR-based energy (EDLM-AR), the key insight is that in masked diffusion, unmasked tokens are identical between x_t and x_0. So p_AR(x_0|x_t) = p_AR(x_0) / p_AR(x_bar_0), where x_bar_0 are the unmasked tokens. Since p_AR(x_0|x_t) ∝ p_AR(x_0), we can use the AR log-likelihood as energy without computing the intractable partition function. The carry-over variant (EDLM-coAR) makes this self-normalizing by copying unmasked tokens directly, enabling exact ELBO computation.

For NCE training (EDLM-NCE), the objective is a binary classification: discriminate real text (positive) from diffusion model output (negative). The energy function is a scalar output from a fine-tuned bidirectional transformer. The log-odds reduce to -E_phi, making the NCE loss simple.

Sampling uses self-normalized importance sampling: draw K candidates from p_diffusion, evaluate energies, resample by energy weights. An importance sampling window w in [0,1] controls which timesteps use IS — early steps (high noise) benefit most, so a short window (w=0.2) captures most of the gain efficiently.

## Methods

- **Base diffusion model:** MDLM (Sahoo et al., 2024), kept frozen
- **Architecture:** 12-layer transformer, hidden dim 768, 12 heads (GPT2-small scale)
- **AR energy:** Pretrained AR model, log-likelihood used directly as energy
- **NCE energy:** Fine-tune from MDLM, mean-pooled last token embeddings projected to scalar
- **Training:** 1M steps on OpenWebText (GPT2 tokenizer, seq len 1024), AdamW, lr 3e-4
- **NCE fine-tuning:** Converges in ~400K steps, 4 GPUs
- **Importance sampling:** k samples per step, window w=0.2 for speed, w=1.0 for quality
- **Perplexity estimation:** Rao-Blackwellized ELBO bounds (Theorem 1), with leave-one-out partition function estimation

## Results

- **Text8 BPC:** 1.24 (vs 1.37 MD4, 1.40 MDLM, 1.23 AR)
- **OpenWebText PPL:** EDLM-coAR 17.58 ≈ AR 17.56 (vs 23.83 MDLM, 24.56 SEDD)
- **Gen PPL (GPT2, 1024 steps):** EDLM-AR 25.5 vs MDLM 42.6 — ~40% improvement
- **Sampling speedup:** ~1.3x over MDLM at same quality (w=0.2, k=2)
- EDLM-NCE and EDLM-AR perform similarly; coAR enables exact perplexity computation
- Early-stage IS (w=0.2) captures most quality gains; longer windows give diminishing returns

## Limitations noted by authors

- Scale limited to GPT2-small (124M) — not tested at larger scales
- Importance sampling batch size bounded by GPU memory (max 16 candidates)
- Generated text quality (Appendix D samples) shows noticeable incoherence vs AR models
- ESS analysis suggests NCE energy may be better for IS than AR energy, but difference not clearly reflected in final metrics

## Connections

- Builds on [[discrete-diffusion]] models (D3PM, SEDD, MDLM, MD4)
- Connects energy-based models (LeCun et al., 2006) with diffusion models
- The AR energy formulation provides a novel way to do parallel sampling from AR models using diffusion as proposal
- Related to Deng et al. (2020) residual EBMs for text, but applied to diffusion rather than autoregressive models

## Open questions

1. Can EDLM scale to billion-parameter models and remain competitive with large AR models?
2. Is there a way to avoid the importance sampling overhead entirely (e.g., distillation)?
3. Can the residual EBM idea transfer to continuous diffusion models for images/audio?

## PyTorch implementation sketch

```python
import torch
import torch.nn.functional as F


def edlm_denoising_step(
    diffusion_model,     # pretrained MDLM: predicts p_theta(x_0 | x_t) per token
    energy_fn,           # E(x_0, x_t) -> scalar energy
    x_t,                 # (B, L) current diffused tokens (with mask token = vocab_size)
    t,                   # (B,) current timestep
    alpha_s, alpha_t,    # noise schedule values for s < t
    num_is_samples: int = 4,  # importance sampling candidates
):
    """
    One EDLM denoising step via importance sampling.

    1. Draw K x_0 candidates from diffusion model (parallel).
    2. Score each with energy function.
    3. Resample by importance weights.
    4. Apply posterior q(x_s | x_t, x_0) to get x_{t-1}.
    """
    B, L = x_t.shape
    mask_token = diffusion_model.vocab_size  # absorbing state index

    # --- 1. Sample K x_0 candidates from diffusion model ---
    x0_candidates = []
    for _ in range(num_is_samples):
        logits = diffusion_model(x_t, t)              # (B, L, V)
        x0_i = torch.distributions.Categorical(logits=logits).sample()
        x0_candidates.append(x0_i)
    x0_stack = torch.stack(x0_candidates, dim=1)       # (B, K, L)

    # Carry-over: unmasked tokens stay the same
    mask = (x_t == mask_token).unsqueeze(1)             # (B, 1, L)
    x0_stack = torch.where(mask, x0_stack, x_t.unsqueeze(1))

    # --- 2. Compute energies for each candidate ---
    energies = []
    for k in range(num_is_samples):
        E_k = energy_fn(x0_stack[:, k], x_t, t)        # (B,)
        energies.append(E_k)
    energies = torch.stack(energies, dim=1)              # (B, K)

    # --- 3. Importance sampling: resample by energy weights ---
    # p_EDLM ∝ p_diffusion * exp(-E) => weights ∝ exp(-E)
    log_weights = -energies                              # (B, K)
    weights = F.softmax(log_weights, dim=1)              # (B, K)
    indices = torch.multinomial(weights, 1).squeeze(1)   # (B,)

    # Gather selected x_0
    x0_selected = x0_stack[torch.arange(B), indices]    # (B, L)

    # --- 4. Apply posterior q(x_s | x_t, x_0) ---
    # For masked diffusion: masked positions in x_t get resampled
    # between x_0 and mask based on alpha schedule
    keep_mask = (x_t != mask_token)
    prob_unmask = (alpha_s - alpha_t) / (1 - alpha_t)

    rand = torch.rand_like(x_t.float())
    new_tokens = torch.where(rand < prob_unmask, x0_selected,
                             torch.full_like(x_t, mask_token))
    x_s = torch.where(keep_mask, x_t, new_tokens)

    return x_s


# --- AR energy function (no training needed) ---
def ar_energy_fn(x_0, x_t, ar_model):
    """
    Energy = -log p_AR(x_0) (up to constants that cancel in IS).
    Uses pretrained AR model as sequence-level scorer.
    """
    log_prob = ar_model.log_prob(x_0)    # (B,)
    return -log_prob                        # lower energy = better sample
```
