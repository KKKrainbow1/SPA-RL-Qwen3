"""W&B panel definitions + helper to register a unified KL/PPO dashboard."""

from __future__ import annotations

from typing import Any


PANEL_DEFINITIONS = [
    # Health metrics (the ones the early-stop callback watches)
    ("ppo/policy/clipfrac",   "last", "Health: clipfrac (halt > 0.5)"),
    ("objective/kl",          "last", "Health: KL(θ_t || θ_ref)"),
    ("objective/kl_coef",     "last", "Health: adaptive kl_coef (halt > 2.0)"),

    # Training signal
    ("ppo/mean_scores",       "last", "PRM step rewards (avg)"),
    ("ppo/returns/mean",      "last", "Returns mean"),
    ("ppo/returns/var",       "last", "Returns variance"),

    # PPO internals
    ("ppo/policy/policykl",   "last", "Per-step policy KL (early-stop signal)"),
    ("ppo/loss/policy",       "last", "PPO policy loss"),
    ("ppo/loss/value",        "last", "PPO value loss"),
    ("ppo/val/error",         "last", "Critic prediction error"),
    ("ppo/val/var_explained", "last", "Critic variance explained"),
]


def register_panels(wandb_run: Any) -> None:
    """Register all SPA-relevant panels on the active W&B run.

    Call once after `wandb.init(...)`. Idempotent.
    """
    for metric, summary, _description in PANEL_DEFINITIONS:
        wandb_run.define_metric(metric, summary=summary)


def emoji_for(metric_name: str, value: float) -> str:
    """Quick visual flag for stdout logs."""
    if metric_name == "ppo/policy/clipfrac":
        return "🟢" if value < 0.3 else "🟡" if value < 0.5 else "🔴"
    if metric_name == "objective/kl_coef":
        return "🟢" if value < 1.0 else "🟡" if value < 2.0 else "🔴"
    if metric_name == "objective/kl":
        return "🟢" if value < 0.5 else "🟡" if value < 1.0 else "🔴"
    return "•"
