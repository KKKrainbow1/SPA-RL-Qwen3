# StepPPOTrainer Explained

`StepPPOTrainer` is the SPA-RL paper's modification to TRL's `PPOTrainer`. It's surgical — only **three** methods are overridden — but conceptually transforms PPO from "single scalar reward at the end" to "dense per-step rewards distributed along the trajectory".

## The three overrides

```python
class StepPPOTrainer(PPOTrainer):
    def step(self, queries, responses, scores, frag_masks=None, ...): ...
    def batched_forward_pass(self, model, queries, responses, ..., frag_masks=None): ...
    def compute_rewards(self, scores, logprobs, ref_logprobs, masks): ...
```

Everything else — GAE, ratio clip, value loss, KL controller — is inherited unchanged from TRL.

## What changes (in plain English)

### Original PPO (single-turn or sparse-reward multi-turn)

```
reward tensor across the response:
   [-kl, -kl, -kl, ..., -kl, (-kl + R_final)]
                              ↑
                only the last token gets the reward signal

mask:  response_mask (entire response = 1)
```

### Step-wise PPO

```
reward tensor across a multi-turn response:
   [-kl, ..., (-kl + step₀), -kl, ..., (-kl + step₁), -kl, ..., (-kl + step_N)]
              ↑                       ↑                       ↑
       step rewards injected at the last token of each assistant turn

mask:  frag_mask
       0 over user observations / system prompt
       1 over assistant turns
```

The **frag_mask** is what makes this work for multi-turn. Without it, GAE would treat the entire response (including embedded user observations) as one continuous trainable sequence.

## Override 1: `step(...)` — interface change

```python
def step(self, queries, responses, scores, frag_masks=None, response_masks=None):
    # `scores` is now a list of tensors, one per trajectory, each containing N step rewards
    # `frag_masks` is new — required for multi-turn
    ...
```

The TRL parent's signature treats `scores` as scalars. Step-wise PPO treats it as a length-N tensor per sample, where N is the number of assistant turns in the trajectory.

## Override 2: `batched_forward_pass(...)` — mask construction

```python
masks = torch.zeros_like(attention_mask)
frag_masks_pad = torch.zeros_like(attention_mask)
for idx, fm in enumerate(frag_masks_batch):
    frag_masks_pad[idx, :fm.shape[0]] = fm
masks[:, :-1] = frag_masks_pad[:, 1:]   # ← uses frag_mask instead of response_mask
```

This is the key line. By using `frag_mask` as the loss mask, the policy gradient and value loss are computed **only over assistant tokens**, never over user observations or padding.

## Override 3: `compute_rewards(...)` — reward injection

```python
for score, logprob, ref_logprob, mask in zip(scores, logprobs, ref_logprobs, masks):
    kl = self._kl_penalty(logprob, ref_logprob)
    reward = (-self.kl_ctl.value * kl).clone()    # baseline: -kl_coef × KL per token
    
    last_mask = mask.clone()
    for score_idx in reversed(range(len(score))):
        # Find the last "1" in the current frag_mask segment — that's the assistant turn's
        # final token position. Inject the step reward there.
        last_1_index = last_mask.nonzero()[-1]
        reward[last_1_index] += score[score_idx] * (len(score) - score_idx - 1)
        
        # Walk backward through the mask: skip the assistant segment we just used,
        # then skip the user observation segment that came before it.
        last_mask = last_mask[:last_1_index + 1]
        last_0_index = (last_mask == 0).nonzero()[-1]
        last_mask = last_mask[:last_0_index + 1]
    
    rewards.append(reward)
```

The loop walks **backward** through the trajectory's frag_mask to locate each assistant turn's final token, and injects that turn's PRM step reward there.

The `* (len(score) - score_idx - 1)` factor is a per-step weighting (later steps get smaller weights). This is empirically tuned in the SPA paper and not easily justified theoretically — treat it as a reward shaping detail.

## What the GAE sees

After all three overrides:

- **`values`** (per token, from critic) — unchanged from base TRL
- **`rewards`** (per token, custom) — has dense PRM signals at step boundaries instead of one scalar at the end
- **`masks`** (per token, from frag_mask) — restricts loss to assistant tokens only

GAE's reverse-iteration loop is **unchanged** from TRL:

```python
for t in reversed(range(gen_len)):
    nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
    delta = rewards[:, t] + γ * nextvalues - values[:, t]
    advantages[t] = delta + γ * λ * advantages[t + 1]
```

But because `rewards` is now dense, the advantages naturally have lower variance — credit doesn't have to propagate from the very end through dozens of zero-reward tokens.

## Why this isn't really "new GAE" — just new inputs

A subtle but important point: SPA does **not** change PPO's algorithm. It changes what gets fed into PPO. The credit-assignment improvements come entirely from:

1. The reward tensor having signal at every step boundary.
2. The mask correctly identifying which tokens are trainable.

Everything downstream (GAE, clip, KL, value head) is the standard TRL implementation.

This is also why SPA is easy to reproduce: you only need to understand how to build `rewards` and `frag_masks` correctly. The PPO machinery is whatever TRL ships.

## Comparison with newer methods

For context, here's how step-wise PPO compares to newer credit-assignment approaches:

| Method | Critic | Advantage estimator | Reward shape |
|---|---|---|---|
| **Step-wise PPO (SPA)** | Yes (TRL ValueHead) | GAE | Dense per-step injected into PPO reward tensor |
| GRPO | No | Group-relative (z-score over G samples) | Scalar per trajectory |
| RLOO | No | Leave-one-out across K samples | Scalar per trajectory |
| Process Reward Models in math RL | No | Soft-label per step | Token-level supervised |

GRPO and RLOO are simpler and avoid critic instability, but lose the ability to do per-step credit assignment that SPA's PRM enables. For long-horizon agent tasks, step-wise rewards still seem to help — the open question is whether GRPO + step-injected rewards (a hybrid) would dominate either.
