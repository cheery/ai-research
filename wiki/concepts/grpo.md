---
type: concept
tags: [reinforcement-learning, preference-optimization]
papers: [dgpo, grpo]
created: 2026-04-08
updated: 2026-04-08
---

# Group Relative Policy Optimization (GRPO)

## Definition

GRPO is a reinforcement learning method that improves model outputs by sampling groups of responses per prompt, computing relative advantages within each group via normalization, and using policy gradient updates weighted by those advantages. It eliminates the need for a separate critic/value model.

## Why it matters

GRPO was introduced in [[grpo]] (DeepSeekMath) and later became central to DeepSeek-R1's reasoning capabilities. It replaces PPO's learned value function with simple group-level statistics, roughly halving memory usage. Its success in LLMs has motivated efforts to apply it to diffusion models.

## How it works

For each question q, GRPO samples G outputs {o_1, ..., o_G} from the old policy. A reward model scores each output, yielding rewards r = {r_1, ..., r_G}. Advantages are computed by normalizing within the group:

A_i = (r_i - mean(r)) / std(r)

These advantages weight a clipped policy gradient update (same structure as PPO):

J_GRPO = E[1/G * sum_i 1/|o_i| * sum_t min[ratio * A_i_t, clip(ratio, 1-eps, 1+eps) * A_i_t] - beta * KL(pi_theta || pi_ref)]

Key differences from PPO:
- No value model — the group mean replaces the learned baseline
- KL penalty is added directly to the loss (not to the reward), keeping advantage computation clean
- Uses an unbiased KL estimator guaranteed to be positive

GRPO supports two supervision modes:
- **Outcome supervision**: single reward at end of output, assigned to all tokens
- **Process supervision**: reward at each reasoning step, advantage for each token is sum of future step rewards. Performs better in practice.

Iterative RL (retraining the reward model periodically) provides further gains.

## Variants and extensions

- **Flow-GRPO** (Liu et al., 2025): Adapts GRPO to flow-matching diffusion models. Forces stochastic SDE-based sampling to create a stochastic policy suitable for the policy gradient framework. Training occurs over the full sampling trajectory.
- **DanceGRPO** (Xue et al., 2025): Another GRPO adaptation to visual generation with similar stochasticity requirements.
- **DGPO** ([[dgpo]]): Argues that GRPO's power comes from the group-level preference signal, not the policy gradient. Preserves group advantages but uses direct preference optimization instead, eliminating the stochastic policy requirement. Achieves ~20x faster training.

## Key papers

- [[grpo]] — introduces GRPO, demonstrates effectiveness on math reasoning, provides unified paradigm for SFT/RFT/DPO/PPO/GRPO
- [[dgpo]] — analyzes GRPO's components and re-implements the group signal without policy gradients

## Current state

GRPO works well for LLMs where the policy is naturally stochastic (probability distributions over tokens). The [[grpo]] paper finds that RL improves Maj@K (majority voting) but not Pass@K, suggesting GRPO makes the output distribution more robust rather than improving fundamental capabilities. For diffusion models, forcing stochasticity via SDE sampling introduces inefficiency. [[dgpo]] proposes that the group preference mechanism, not the policy gradient, is the essential ingredient.
