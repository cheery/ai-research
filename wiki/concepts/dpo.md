---
type: concept
tags: [reinforcement-learning, preference-optimization]
papers: [dgpo, grpo]
created: 2026-04-08
updated: 2026-04-08
---

# Direct Preference Optimization (DPO)

## Definition

DPO aligns generative models to human preferences by directly optimizing a likelihood-based objective on preference pairs, bypassing the need to train an explicit reward model.

## Why it matters

Traditional RLHF requires training a separate reward model and then using policy gradient methods to optimize it — a complex two-stage pipeline. DPO collapses this into a single stage by exploiting the closed-form relationship between the optimal policy and the reward function under KL-regularized RLHF.

## How it works

The KL-regularized RLHF objective has a known closed-form optimal solution:

p*(x_0|c) = p_ref(x_0|c) * exp(r(c,x_0)/beta) / Z(c)

where Z(c) is an intractable partition function. Rearranging gives an implicit reward parameterization:

r(c,x_0) = beta * log[p_theta(x_0|c) / p_ref(x_0|c)] + beta * log Z(c)

When two samples form a preference pair (x_w preferred over x_l), plugging this into the Bradley-Terry model makes Z(c) cancel:

L_DPO = -E[log sigma(beta * log(p_theta(x_w|c)/p_ref(x_w|c)) - beta * log(p_theta(x_l|c)/p_ref(x_l|c)))]

This is tractable without a reward model. However, DPO is fundamentally pairwise: it requires exactly one preferred and one dispreferred sample, which limits its ability to use fine-grained information from larger groups of samples.

## Variants and extensions

- **Diffusion DPO** (Wallace et al., 2024): Adapts DPO to diffusion models by defining the reward over diffusion paths x_{0:T}, using forward diffusion q(x_{1:T}|x_0) as an approximation to the inversion chain. Retains the pairwise restriction.
- **Diffusion KTO** (Yang et al., 2024): Uses Kahneman-Tversky Optimization loss instead of Bradley-Terry, handling non-paired preference data.
- **DGPO** ([[dgpo]]): Extends DPO from pairwise to group-level preferences, using advantage-based weighting to cancel Z(c) across groups rather than pairs.

## Key papers

- [[dgpo]] — provides the DPO derivation and extends it to group-level preferences for diffusion models
- [[grpo]] — frames DPO within a unified paradigm: DPO is an offline, pairwise method whose gradient coefficient is sigma(delta log-ratio), contrasting with [[grpo]]'s online, group-normalized approach

## Current state

DPO is well-established for LLM alignment. The [[grpo]] paper positions DPO as one point in a unified paradigm: offline data source, rule-based or preference-based reward, pairwise gradient coefficient. Compared to GRPO's online group-wise approach, DPO's offline pairwise nature limits the learning signal. For diffusion models, [[dgpo]] demonstrates that extending DPO to group-level preferences unlocks significant efficiency gains by eliminating the need for stochastic policies.
