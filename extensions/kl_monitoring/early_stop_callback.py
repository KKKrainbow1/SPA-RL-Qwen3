"""Drop-in early-stop callback for the SPA StepPPO training loop.

Insert into upstream's `ppo/step_ppo.py::train()`:

    from extensions.kl_monitoring.early_stop_callback import KLEarlyStopCallback

    callback = KLEarlyStopCallback(
        clipfrac_threshold=0.5,
        kl_coef_threshold=2.0,
        consecutive_steps=3,
    )

    for batch_id, batch in enumerate(self.ppo_trainer.dataloader):
        stats = self.ppo_trainer.step(...)
        callback.on_step_end(stats, batch_steps, ppo_trainer=self.ppo_trainer)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from extensions.kl_monitoring.monitor import (
    KLHealth,
    extract_health,
    format_health_line,
)

logger = logging.getLogger("kl_monitoring")


class KLDivergenceTooLargeError(RuntimeError):
    """Raised when KL/clipfrac exceeds thresholds for `consecutive_steps`."""


class KLEarlyStopCallback:
    """Watches PPO stats and halts training when KL protection is failing."""

    def __init__(
        self,
        clipfrac_threshold: float = 0.5,
        kl_coef_threshold: float = 2.0,
        kl_threshold: float = 1.0,
        consecutive_steps: int = 3,
        save_on_halt: bool = True,
        save_dir: str | None = None,
    ):
        self.clipfrac_threshold = clipfrac_threshold
        self.kl_coef_threshold = kl_coef_threshold
        self.kl_threshold = kl_threshold
        self.consecutive_steps = consecutive_steps
        self.save_on_halt = save_on_halt
        self.save_dir = save_dir

        self._unhealthy_streak = 0
        self.history: list[KLHealth] = []

    def on_step_end(
        self,
        stats: dict[str, Any],
        step: int,
        ppo_trainer: Any | None = None,
    ) -> None:
        """Called after each `ppo_trainer.step(...)`. Raises on halt."""
        h = extract_health(stats, step)
        self.history.append(h)

        # Always log the line — both for stdout and for W&B downstream
        logger.info(format_health_line(h))

        unhealthy_now = self._is_unhealthy(h)
        if unhealthy_now:
            self._unhealthy_streak += 1
            logger.warning(
                f"  ⚠ unhealthy streak: {self._unhealthy_streak}/{self.consecutive_steps}"
            )
        else:
            self._unhealthy_streak = 0

        if self._unhealthy_streak >= self.consecutive_steps:
            self._halt(h, ppo_trainer)

    def _is_unhealthy(self, h: KLHealth) -> bool:
        if h.clipfrac > self.clipfrac_threshold:
            return True
        if h.kl_coef > self.kl_coef_threshold:
            return True
        if h.kl > self.kl_threshold:
            return True
        return False

    def _halt(self, h: KLHealth, ppo_trainer: Any | None) -> None:
        msg = (
            f"KL early-stop: {self._unhealthy_streak} consecutive unhealthy steps. "
            f"Last health: {format_health_line(h)}"
        )
        logger.error(msg)

        if self.save_on_halt and ppo_trainer is not None:
            save_dir = self.save_dir or "ckpt/early_stop_halt"
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            try:
                ppo_trainer.save_pretrained(os.path.join(save_dir, f"step-{h.step}-halt"))
                logger.warning(f"Saved checkpoint at halt to {save_dir}")
            except Exception as e:  # pragma: no cover
                logger.error(f"Failed to save halt checkpoint: {e}")

        raise KLDivergenceTooLargeError(msg)
