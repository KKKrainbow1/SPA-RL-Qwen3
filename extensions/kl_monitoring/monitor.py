"""Pure functions that derive health metrics from TRL's `stats` dict.

Use these from a training callback or a separate logger; this module has
no side effects (no print, no W&B), so it's easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KLHealth:
    """Snapshot of KL/clipfrac/scores at one PPO step."""

    step: int
    kl: float                # objective/kl
    kl_coef: float           # objective/kl_coef
    clipfrac: float          # ppo/policy/clipfrac
    mean_score: float        # ppo/mean_scores
    returns_var: float       # ppo/returns/var (for sanity)

    @property
    def is_clipfrac_unhealthy(self) -> bool:
        return self.clipfrac > 0.5

    @property
    def is_kl_unhealthy(self) -> bool:
        return self.kl_coef > 2.0 or self.kl > 1.0

    @property
    def prm_to_kl_dominance(self) -> float:
        """Rough indicator: positive score divided by absolute KL pressure.

        > 1 means PRM is winning, < 1 means KL is winning. We compute a
        crude estimate from the mean score and kl_coef; not exact, but a
        reasonable trend indicator.
        """
        kl_pressure = abs(self.kl_coef * self.kl) + 1e-9
        return max(self.mean_score, 0.0) / kl_pressure


def extract_health(stats: dict[str, Any], step: int) -> KLHealth:
    """Pull the relevant fields out of TRL's `stats` dict.

    `stats` is whatever `PPOTrainer.step(...)` returns; the keys below are
    standard across TRL versions, but if your version differs adjust here.
    """

    def get_float(key: str, default: float = 0.0) -> float:
        value = stats.get(key, default)
        # TRL sometimes returns numpy scalars or 0-d tensors
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    return KLHealth(
        step=step,
        kl=get_float("objective/kl"),
        kl_coef=get_float("objective/kl_coef"),
        clipfrac=get_float("ppo/policy/clipfrac"),
        mean_score=get_float("ppo/mean_scores"),
        returns_var=get_float("ppo/returns/var"),
    )


def format_health_line(h: KLHealth) -> str:
    """One-line human-readable summary, suitable for stdout logging."""
    return (
        f"[step {h.step:>4}] "
        f"kl={h.kl:.3f}  "
        f"kl_coef={h.kl_coef:.3f}  "
        f"clipfrac={h.clipfrac:.3f}  "
        f"mean_score={h.mean_score:.3f}  "
        f"returns_var={h.returns_var:.4f}  "
        f"prm/kl_ratio={h.prm_to_kl_dominance:.2f}"
    )
