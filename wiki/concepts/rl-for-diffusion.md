---
type: concept
tags: [reinforcement-learning, diffusion-models]
papers: [dgpo]
created: 2026-04-08
updated: 2026-04-08
---

# RL Post-Training for Diffusion Models

## Definition

The application of reinforcement learning techniques to fine-tune pre-trained diffusion models, typically to align them with reward signals (human preferences, compositional accuracy, text rendering fidelity) without degrading image quality.

## Why it matters

Pre-trained diffusion models generate high-quality images but often fail at specific tasks like compositional instruction following, text rendering, or matching human aesthetic preferences. RL post-training provides a way to improve targeted capabilities using reward models as training signals.

## How it works

Three main approaches exist:

1. **Reward backpropagation**: Directly backpropagate through the generation process to maximize a differentiable reward function. Limited to differentiable rewards and can be unstable.

2. **Policy gradient methods**: Treat the diffusion model as a policy and apply REINFORCE-style updates. Recent work like Flow-GRPO uses [[grpo]] with stochastic SDE sampling to create a stochastic policy. This works but is slow — SDE sampling produces lower-quality samples per step, the stochasticity comes from generic Gaussian noise (weak learning signal), and training must run over the full trajectory.

3. **Direct preference methods**: Skip the reward model and optimize on preference data directly. Diffusion [[dpo]] uses pairwise preferences. [[dgpo]] extends this to group-level preferences, combining DPO's tractability with GRPO's fine-grained group information. The key advantage is compatibility with efficient ODE samplers.

The ODE vs SDE sampling distinction is critical. ODE solvers (DPM-Solver, Flow-DPM-Solver) produce better samples in fewer steps but are deterministic, which breaks policy gradient methods. [[dgpo]] shows that ODE rollouts produce higher rewards and faster convergence than SDE rollouts, suggesting the SDE requirement in prior work was an artifact of the policy gradient framework, not a source of useful diversity.

## Variants and extensions

- Fine-tuning on curated datasets (DALL-E 3, SDXL approaches)
- Multi-step reward maximization with backpropagation through diffusion
- Policy gradient with PPO-style clipping
- Direct preference methods ([[dpo]], [[dgpo]])

## Key papers

- [[dgpo]] — proposes DGPO, demonstrating that dropping the policy gradient in favor of direct group preferences enables ODE sampling and 20x speedup

## Current state

The field is converging on the view that policy gradient methods are suboptimal for diffusion models because they require compromising sampling efficiency. Direct preference methods like [[dpo]] and [[dgpo]] that work with deterministic samplers appear to be the more promising direction. The main open question is how these methods scale with model size and more complex reward signals.
