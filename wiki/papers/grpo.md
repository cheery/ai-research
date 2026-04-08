---
type: paper
source: raw/GRPO.pdf
title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
authors: [Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo]
year: 2024
tags: [reinforcement-learning, mathematical-reasoning, llm-training, pre-training]
ingested: 2026-04-08
---

# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

## One-line summary

DeepSeekMath achieves near-GPT-4 math performance at 7B scale through large-scale math pre-training data and introduces GRPO, a PPO variant that replaces the critic model with group-normalized advantages.

## Key contributions

1. Constructs the DeepSeekMath Corpus — 120B math tokens from Common Crawl via an iterative fastText-based selection pipeline, ~7x larger than Minerva's data
2. Introduces GRPO: eliminates the value/critic model from PPO by computing advantages from group reward statistics, halving memory usage
3. Provides a unified paradigm framing SFT, RFT, [[dpo]], PPO, and [[grpo]] as points in a (data source, reward function, gradient coefficient) space
4. Demonstrates that code training prior to math training improves mathematical reasoning both with and without tool use

## Core ideas

The paper has two main thrusts. First, math pre-training: starting from DeepSeek-Coder-Base-v1.5 7B, continual training on a mixture of 56% math corpus, 20% code, 10% arXiv, 4% AlgebraicStack, and 10% natural language for 500B tokens yields a 7B model competitive with Minerva 540B on mathematical benchmarks.

Second, GRPO (Group Relative Policy Optimization): PPO requires a separate value model of comparable size to the policy, which is memory-expensive and hard to train accurately per-token when only the final token gets a reward. GRPO replaces this by sampling G outputs per question from the old policy, scoring them with a reward model, and normalizing within the group: A_i = (r_i - mean(r)) / std(r). This group-relative baseline eliminates the value model entirely.

GRPO can use either outcome supervision (single reward per output, assigned to all tokens) or process supervision (reward per reasoning step, with each token's advantage being the sum of future step rewards). Process supervision outperforms outcome supervision. Iterative RL — periodically retraining the reward model — provides further gains, especially in the first iteration.

The unified paradigm shows that all methods share the same gradient structure: E[GC(q,o,t) * grad log pi(o_t | q, o_<t)]. They differ only in data source (online vs offline), reward function (rule vs model), and gradient coefficient (how the reward signal modulates the update). GRPO's GC is the group-normalized advantage plus a KL penalty term.

## Methods

**Pre-training:**
- Base: DeepSeek-Coder-Base-v1.5 7B
- Data: 500B tokens (56% math, 20% code, 10% arXiv, 4% AlgebraicStack, 10% NL)
- Math corpus: iterative fastText classification on Common Crawl, 4 iterations, 120B tokens final
- Decontamination: 10-gram exact match filtering against benchmarks

**Instruction tuning:**
- 776K examples covering CoT, PoT, and tool-integrated reasoning in English and Chinese
- 500 steps, batch 256, constant lr 5e-5

**GRPO:**
- RL data: 144K CoT-format questions from GSM8K and MATH only
- Group size: 64 outputs per question
- Learning rate: 1e-6, KL coefficient: 0.04
- Max output length: 1024, batch size: 1024
- Single policy update per exploration stage
- Reward model initialized from DeepSeekMath-Base 7B, lr 2e-5

## Results

- MATH: 51.7% (vs 46.8% instruct, 36.2% base), approaching GPT-4's 52.9%
- GSM8K: 88.2% (vs 82.9% instruct, 64.2% base)
- Improvements extend to out-of-domain benchmarks (CMATH: 84.6% -> 88.8%)
- GRPO outperforms Online RFT, which outperforms offline RFT
- Process supervision > outcome supervision
- Iterative RL provides significant further gains
- RL improves Maj@K but not Pass@K, suggesting distribution robustification rather than capability gain

## Limitations noted by authors

- Weaker on geometry and theorem-proof tasks vs closed models (possible data selection bias)
- Limited few-shot capability compared to GPT-4 (similar performance in zero-shot and few-shot)
- RL only improves Maj@K, not Pass@K — suggests it stabilizes the distribution rather than improving fundamental capabilities
- arXiv papers were found ineffective for math, but the authors note limitations in this conclusion (not tested combined with other data, not tested at larger scale)

## Connections

- Introduces [[grpo]], later adapted for diffusion models by [[dgpo]]
- Provides unified paradigm covering [[dpo]] as an offline pairwise method
- Code training benefits parallel findings in code-for-reasoning literature

## Open questions

1. How to make RL improve Pass@K (fundamental capabilities) rather than just Maj@K?
2. Can the iterative data collection pipeline generalize to other domains beyond math?
3. What is the interplay between pre-training data quality and RL effectiveness?

## PyTorch implementation sketch

```python
import torch
import torch.nn.functional as F


def grpo_loss(
    policy_logprobs,    # (G, L) log pi_theta(o_t | q, o_<t) for each sample
    old_logprobs,       # (G, L) log pi_old(o_t | q, o_<t)
    ref_logprobs,       # (G, L) log pi_ref(o_t | q, o_<t)
    rewards,            # (G,) reward per sample
    clip_eps: float,    # PPO clipping epsilon
    beta: float,        # KL penalty coefficient
):
    """
    Core GRPO loss for a single question's group of G sampled outputs.

    1. Normalize rewards within the group to get advantages.
    2. Clipped policy gradient (same structure as PPO).
    3. KL penalty against reference model added directly to loss.
    """
    G, L = policy_logprobs.shape

    # --- Group-normalized advantages (no value model needed) ---
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    advantages = advantages.unsqueeze(1)  # (G, 1) broadcast over tokens

    # --- Importance ratio ---
    log_ratio = policy_logprobs - old_logprobs        # (G, L)
    ratio = log_ratio.exp()                            # (G, L)

    # --- Clipped surrogate objective (PPO-style) ---
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # --- KL penalty (unbiased estimator, guaranteed >= 0) ---
    # D_KL = (pi_ref / pi_theta) - log(pi_ref / pi_theta) - 1
    log_ref_ratio = ref_logprobs - policy_logprobs     # (G, L)
    kl_per_token = log_ref_ratio.exp() - log_ref_ratio - 1
    kl_loss = beta * kl_per_token.mean()

    return policy_loss + kl_loss


# --- Training loop sketch ---
# for step in range(num_steps):
#     questions = sample_questions(batch_size)
#     for q in questions:
#         # 1. Sample G outputs from old policy
#         outputs = [generate(old_policy, q) for _ in range(G)]
#         # 2. Score with reward model
#         rewards = reward_model(q, outputs)                  # (G,)
#         # 3. Gather logprobs from policy, old_policy, and ref
#         pi_logp = get_logprobs(policy, q, outputs)          # (G, L)
#         old_logp = get_logprobs(old_policy, q, outputs)     # (G, L)
#         ref_logp = get_logprobs(ref_model, q, outputs)      # (G, L)
#         # 4. Compute loss and update
#         loss = grpo_loss(pi_logp, old_logp, ref_logp,
#                          rewards, clip_eps=0.2, beta=0.04)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#     # Periodically: old_policy <- policy, optionally retrain reward model
```
