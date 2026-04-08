---
type: paper
source: raw/DGPO.pdf
title: "Reinforcing Diffusion Models by Direct Group Preference Optimization"
authors: [Yihong Luo, Tianyang Hu, Jing Tang]
year: 2025
tags: [reinforcement-learning, diffusion-models, preference-optimization, image-generation]
ingested: 2026-04-08
---

# Reinforcing Diffusion Models by Direct Group Preference Optimization

## One-line summary

DGPO adapts group-level preference optimization to diffusion models by dropping the policy-gradient framework, enabling 20x faster training with ODE samplers.

## Key contributions

1. Identifies that GRPO's effectiveness comes from group-level relative preference information, not the policy-gradient formulation
2. Proposes DGPO: a direct group preference objective using Bradley-Terry loss with advantage-based weighting that cancels the intractable partition function
3. Achieves ~20x faster training than Flow-GRPO while reaching 97% on GenEval (up from 63% baseline)
4. Introduces a timestep clip strategy to prevent overfitting to artifacts from few-step generation

## Core ideas

DGPO addresses the mismatch between [[grpo]] and diffusion models. GRPO requires a stochastic policy for exploration, which forced prior work (Flow-GRPO) to use inefficient SDE-based sampling. DGPO argues that the real power of GRPO lies in group-level relative preference information, not in the policy gradient itself.

The method generates a group of G samples per prompt using efficient ODE samplers, scores them with a reward model, and computes normalized advantages (mean-zero, unit-variance within the group). Samples with positive advantage form the positive group; those with negative or zero advantage form the negative group. The key weighting trick uses absolute advantages as weights — since advantages are zero-mean, the total weight of the positive and negative groups is equal. This causes the intractable partition function Z(c) to cancel out in the Bradley-Terry loss, yielding a tractable training objective.

The final loss resembles Diffusion [[dpo]] but with group-wise advantages instead of pairwise comparisons. Training uses single-timestep denoising rather than full trajectories, and shared noise across samples in the same group reduces variance.

## Methods

- **Base model:** SD3.5-M
- **Sampling:** Flow-DPM-Solver with 10 steps (ODE-based) for rollout
- **Group size:** 24 samples per prompt
- **Fine-tuning:** LoRA with rank 32
- **beta:** 100 (KL regularization strength)
- **Timestep clip:** Training only on t in [t_min, T] to avoid overfitting to few-step artifacts
- **EMA:** theta-minus updated by identity for first 200 steps, then EMA with decay 0.3
- **Text dropout:** 5% probability during training
- **Resolution:** 512

Three evaluation tasks: compositional image generation (GenEval), visual text rendering (OCR accuracy), and human preference alignment (PickScore). Out-of-domain metrics: Aesthetic Score, DeQA, ImageReward, UnifiedReward on DrawBench.

## Results

- GenEval overall: 0.97 (vs 0.95 Flow-GRPO, 0.63 baseline, 0.84 GPT-4o)
- Training speed: ~20x faster than Flow-GRPO overall; ~30x faster on GenEval convergence
- Out-of-domain metrics maintained or improved across all four metrics
- OCR accuracy: 0.96 (vs 0.92 Flow-GRPO)
- PickScore: 23.89 (vs 23.31 Flow-GRPO)
- Ablation: ODE rollout outperforms SDE rollout, confirming the policy-gradient requirement (not sample diversity) drove prior SDE usage
- Offline DGPO works but underperforms the online setting
- DGPO outperforms Diffusion DPO in both online and offline settings

## Limitations noted by authors

- Only evaluated on text-to-image synthesis; extension to text-to-video is future work
- Uses LoRA fine-tuning rather than full model fine-tuning

## Connections

- Extends [[dpo]] from pairwise to group-level preferences for diffusion models
- Re-interprets [[grpo]] by extracting the group preference signal without the policy-gradient framework
- Relates to [[rl-for-diffusion]] as an online RL post-training method that avoids stochastic policies
- Directly compared against Flow-GRPO (Liu et al., 2025) and Diffusion DPO (Wallace et al., 2024)

## Open questions

1. How does DGPO scale to larger base models or full fine-tuning beyond LoRA?
2. Can the group preference framework transfer to other generative modalities (video, audio, 3D)?
3. What is the tightness of the Jensen's inequality upper bound used in the loss derivation?

## PyTorch implementation sketch

```python
import torch
import torch.nn.functional as F


def dgpo_loss(
    model,          # current diffusion model f_theta
    ref_model,      # reference diffusion model f_ref (detached)
    x0_samples,     # (G, C, H, W) generated samples for one prompt
    rewards,        # (G,) reward scores
    t,              # sampled timestep, scalar
    noise,          # (G, C, H, W) shared noise across group
    alpha_t, sigma_t,  # noise schedule values at timestep t
    beta: float,    # KL regularization strength
):
    """
    Core DGPO loss for a single prompt's group of samples.

    1. Partition into positive/negative groups by advantage sign.
    2. Weight by |advantage| — this ensures Z(c) cancels.
    3. Bradley-Terry loss on weighted denoising score matching terms.
    """
    G = x0_samples.shape[0]

    # --- Compute advantages (group normalization) ---
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # --- Partition into positive and negative groups ---
    pos_mask = advantages > 0
    neg_mask = ~pos_mask
    # weights = |advantage|
    weights = advantages.abs()

    # --- Add noise to get x_t (shared noise across group) ---
    x_t = alpha_t * x0_samples + sigma_t * noise

    # --- Denoising score matching: ||f(x_t, t, c) - x_0||^2 ---
    with torch.no_grad():
        ref_pred = ref_model(x_t, t)                # (G, C, H, W)
    model_pred = model(x_t, t)                      # (G, C, H, W)

    dsm_model = (model_pred - x0_samples).pow(2).flatten(1).sum(1)    # (G,)
    dsm_ref   = (ref_pred   - x0_samples).pow(2).flatten(1).sum(1)    # (G,)

    # --- Weighted group preference loss ---
    # Positive group: maximize likelihood, Negative group: minimize
    log_ratio = dsm_ref - dsm_model  # (G,) — higher means model is better

    pos_term = (weights * log_ratio * pos_mask.float()).sum()
    neg_term = (weights * log_ratio * neg_mask.float()).sum()

    # Bradley-Terry sigmoid loss
    loss = -F.logsigmoid(beta * (pos_term - neg_term))
    return loss


# --- Training loop sketch ---
# for step in range(num_steps):
#     prompts = sample_prompts(batch_size)
#     for c in prompts:
#         # 1. Generate group using ODE sampler (e.g. 10-step DPM-Solver)
#         x0_group = ode_sample(model, c, num_samples=G, steps=10)
#         # 2. Score with reward model
#         r = reward_model(c, x0_group)
#         # 3. Sample timestep (clipped to [t_min, T])
#         t = torch.randint(t_min, T+1, (1,))
#         noise = torch.randn_like(x0_group)
#         # 4. Compute DGPO loss and update
#         loss = dgpo_loss(model, ref_model, x0_group, r, t, noise,
#                          alpha_t, sigma_t, beta)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
```
