# KL Monitoring & Early Stop

Real-time monitoring of PPO's KL behaviour with auto early-stop when KL penalty starts dominating the PRM signal.

## Why this exists

SPA is **offline PPO** — rollout policy is frozen at θ_SFT, training policy θ_t drifts further over each PPO step. Two failure modes:

| Failure | Symptom | Why |
|---|---|---|
| **Excessive ratio clipping** | `clipfrac > 0.5` | θ_t too far from θ_old → most updates get clipped to no-op |
| **KL signal drowns PRM** | `kl_coef × KL` > total step rewards | Adaptive controller overshoots, all gradient gets pushed back toward θ_SFT |

When either happens, training continues consuming GPU time but **stops learning**. The watchdog in this module detects both and stops the run.

## Dual-layer KL protection (the design)

PPO already has two KL-related defenses; this module wires up the third (monitoring) to make sure they're working:

1. **Layer 1: KL(θ_t ‖ θ_ref) baked into reward** — every token's reward is offset by `-kl_coef × KL[t]`. This is a *long-term* anchor against θ_SFT; the gradient flows through it via GAE → advantage → PPO loss.

2. **Layer 2: PPO ratio clip(θ_t / θ_old)** — the standard `clip(ratio, 1-ε, 1+ε)` term. This is a *short-term* limit on per-update step size.

3. **Layer 3 (this module): clipfrac + kl_coef monitoring + early stop** — when the first two start failing, halt before ruining the checkpoint.

## Files

| File | Role |
|---|---|
| `monitor.py` | Computes per-step health metrics from TRL's `stats` dict |
| `early_stop_callback.py` | Drop-in callback for the StepPPO training loop; raises `KLDivergenceTooLargeError` to halt |
| `wandb_logger.py` | Uniform W&B panel definitions (clipfrac/kl_coef/scores/returns_var) |

## Usage

```python
from extensions.kl_monitoring.early_stop_callback import KLEarlyStopCallback

callback = KLEarlyStopCallback(
    clipfrac_threshold=0.5,    # halt if more than 50% of tokens get clipped
    kl_coef_threshold=2.0,     # halt if adaptive controller pushes coef above 2.0
    consecutive_steps=3,       # require N steps in a row before halting
    save_on_halt=True,         # save checkpoint before stopping
)

# In your training loop (modify upstream's step_ppo.py train()):
for batch in self.ppo_trainer.dataloader:
    stats = self.ppo_trainer.step(...)
    callback.on_step_end(stats, batch_steps, ppo_trainer=self.ppo_trainer)
```

## What "PRM signal drowned by KL" actually looks like

Per-token reward in SPA is:

    reward[t] = -kl_coef × KL[t]   +   step_score[i] × weight[i]   (only at step end)

For a typical 1000-token response with 11 step-end positions:

| Regime | KL coef | Per-token KL | Total -kl × KL | Total +step_rewards | Ratio |
|---|---|---|---|---|---|
| Healthy | 0.2 | 0.05 | -10 | +27 | step rewards dominate |
| Borderline | 1.0 | 0.2 | -200 | +27 | KL dominates 7×, training stalls |
| Pathological | 3.0 | 0.5 | -1500 | +27 | KL dominates 55×, training reverses |

The `kl_coef_threshold=2.0` default is set just below the borderline regime — by the time it triggers, PRM has effectively been silenced for the past few steps and there's no reason to continue.

## Why not just clip kl_coef instead of stopping?

Tried in development. Symptoms:

- Clipping kl_coef without addressing the underlying drift just delays the failure — θ_t keeps drifting, KL keeps rising, eventually clipfrac saturates.
- Resetting kl_coef each batch breaks the AdaptiveKLController's PI feedback loop.
- Lowering learning rate when kl_coef spikes works marginally but adds noise.

Halting and inspecting the checkpoint is consistently the most useful action — usually you can resume from an earlier step with adjusted hyperparameters (smaller `init_kl_coef`, smaller batch, higher reward scale).

## Recommended thresholds by phase

| Phase | clipfrac | kl_coef | consecutive_steps |
|---|---|---|---|
| First 50 steps (warmup) | 0.6 | 3.0 | 5 |
| Steady state (50-300) | 0.5 | 2.0 | 3 |
| Late training (>300) | 0.4 | 1.5 | 2 |

These match SPA's typical training horizon (~200 steps total) — the project rarely benefits from training beyond 300 steps, and the thresholds reflect declining tolerance as the policy gets more committed.

## Reading the W&B dashboard

Three plots tell you everything:

1. **`objective/kl` over time** — should stay below ~1.0 throughout. Spikes are OK; sustained climbs are not.
2. **`objective/kl_coef` over time** — should oscillate around `init_kl_coef`. If it monotonically climbs, the adaptive controller is fighting drift it can't win.
3. **`ppo/policy/clipfrac` over time** — healthy is ~10-30%. Above 50% means most updates are no-ops.

If all three look normal but eval scores aren't improving, the problem isn't KL — it's somewhere else (data quality, learning rate, reward scaling). Don't blame this module.
