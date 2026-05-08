# KL Protection in SPA-RL

This doc explains the dual-layer KL mechanism inherited from TRL PPO and how the [`extensions/kl_monitoring/`](../extensions/kl_monitoring/) module wraps it with monitoring + auto-halt.

## The two PPO KL terms (often confused)

PPO has **two separate** KL-related defenses, with different purposes:

| Term | What it constrains | When it acts | Failure mode if removed |
|---|---|---|---|
| **KL(θ_t ‖ θ_ref) baked into reward** | distance from SFT base | every token, throughout training | θ_t drifts far from useful regions |
| **PPO ratio clip(θ_t / θ_old)** | per-step update size | each PPO inner-loop iteration | one bad batch destroys the policy |

These are **complementary**, not redundant:

- KL-vs-ref is a **slow gravity** pulling the policy toward θ_SFT over many steps. It uses an adaptive coefficient (TRL's `AdaptiveKLController`) that grows when KL exceeds target.
- Ratio-clip is a **hard speed limit** preventing any single update from changing the policy too aggressively. The clip range ε is fixed (default 0.2).

## How they appear in the code

### Layer 1: KL penalty in reward

In `step_ppotrainer.py`:

```python
kl = self._kl_penalty(logprob, ref_logprob)
non_score_reward = -self.kl_ctl.value * kl    # ← per-token negative offset
reward = non_score_reward.clone()
reward[step_end_pos] += step_score             # PRM signal added on top
```

This means **every token gets a small negative reward** proportional to how much θ_t disagrees with θ_ref at that token. The PRM step rewards must overcome this to give a positive learning signal.

### Layer 2: Ratio clip in policy loss

In `trl/trainer/ppo_trainer.py` (inherited):

```python
ratio = torch.exp(new_logprobs - old_logprobs)
loss_unclipped = -A * ratio
loss_clipped   = -A * torch.clamp(ratio, 1 - ε, 1 + ε)
loss = max(loss_unclipped, loss_clipped)
```

`ratio` measures how much the current policy assigns this token compared to the rollout-time policy. Outside `[1-ε, 1+ε]`, clipping kicks in and the gradient is effectively zero for that token.

## The "PRM drowned by KL" failure

The mechanism fails not when KL is too small, but when the **AdaptiveKLController overshoots** and `kl_coef` runs away. Numerically:

| Per-token | Healthy | Pathological |
|---|---|---|
| `KL[t]` | ~0.05 | ~0.5 |
| `kl_coef` | ~0.2 | ~3.0 |
| `kl_coef × KL[t]` | -0.01 | -1.5 |
| 1000-token total KL contribution | -10 | -1500 |
| Total step rewards (11 turns × ~0.5 × weight) | +27 | +27 |

In the pathological regime, KL pressure overwhelms PRM signal by ~50×. Advantages become uniformly negative, policy collapses back toward θ_SFT, and PRM has effectively no influence.

## Why AdaptiveKL can overshoot

The controller's update rule:

```python
proportional_error = clip(measured_kl / target - 1, -0.2, 0.2)
mult = 1 + proportional_error * n_steps / horizon
self.value *= mult
```

When `measured_kl` exceeds target, `mult > 1` and `kl_coef` increases. **But the controller has no derivative term** — it doesn't slow down as it approaches the target. With high `n_steps / horizon` ratios (small horizon), it can ratchet up `kl_coef` beyond what's needed before the next measurement comes in.

For typical SPA settings (horizon=10000, batch=32, ~200 total steps), this happens rarely — but when it does, training silently stalls. The monitoring module catches it.

## Monitoring strategy

[`extensions/kl_monitoring/early_stop_callback.py`](../extensions/kl_monitoring/early_stop_callback.py) tracks three signals:

| Signal | Threshold | Why |
|---|---|---|
| `ppo/policy/clipfrac` | > 0.5 sustained | Most updates are no-ops; gradient is wasted |
| `objective/kl_coef` | > 2.0 sustained | Controller has overshot; PRM is being silenced |
| `objective/kl` | > 1.0 sustained | Policy has actually drifted (vs just being penalized) |

**"Sustained"** = `consecutive_steps=3` by default. Single spikes are normal (they happen every time a difficult batch comes through); only sustained anomalies are halt-worthy.

When triggered, the callback:

1. Saves a checkpoint at the halt point (so you don't lose the partial training).
2. Logs the full health history.
3. Raises `KLDivergenceTooLargeError` to break out of the training loop.

## Recommended response to a halt

1. Look at `objective/kl_coef`'s trajectory. If monotonically increasing for 20+ steps before halt, **lower `init_kl_coef`** (try 0.05 → 0.02).
2. Look at `ppo/mean_scores`. If declining, **scale up step rewards** (try multiplying by 2-3× in `prm/rl_data_org.py`).
3. Look at `ppo/policy/clipfrac`. If high from step 1 onward, **lower learning rate** or **decrease `ppo_epochs`** (4 → 2).
4. Resume from the saved checkpoint with adjusted hyperparameters.

If multiple halts occur in succession with adjustments, the algorithm-level issue is likely **off-policy drift inherent to offline PPO** — consider iterative offline rollout (refresh data every 50 steps) or migrating to online GRPO/RLOO.
