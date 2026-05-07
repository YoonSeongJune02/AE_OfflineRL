"""
Reward shaping utilities for DAE-augmented offline RL.

Provides RewardShaper that converts DAE reconstruction error into a reward penalty
with three optional refinements:

    1. Threshold filtering: only penalize errors above a learned/fixed threshold
       (so low-noise normal states don't get penalized).
    2. Running normalization: standardize errors using a running mean/std for
       stable scaling across training.
    3. Adaptive lambda warmup: linearly increase the penalty coefficient over
       the first `warmup_steps` to avoid disrupting early Q-value estimation.

The shaper is stateful (running stats, step counter) and is updated every train
step. All operations are vectorized over batch.
"""

from typing import Dict, Tuple
import torch


class RunningMeanStd:
    """Online running mean and variance using Welford's algorithm.

    Numerically stable, supports batch updates. Used for normalizing
    reconstruction errors across the training distribution.
    """

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: torch.Tensor) -> None:
        """Update statistics with a batch of new values."""
        batch_mean = float(x.mean().item())
        batch_var = float(x.var(unbiased=False).item())
        batch_count = x.numel()

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.var = m2 / tot_count
        self.count = tot_count

    @property
    def std(self) -> float:
        return float(max(self.var, 1e-8) ** 0.5)


class RewardShaper:
    """DAE-based reward shaper with threshold + normalization + adaptive lambda.

    The penalty is computed as:
        normalized_err = (recon_err - running_mean) / running_std
        clipped_err    = max(0, normalized_err - threshold_z)
        penalty        = tanh(clipped_err)            # bounded in [0, 1]
        lambda_t       = lambda_max * min(1, t / warmup_steps)
        shaped_reward  = reward - lambda_t * penalty

    Why these choices:
        - Threshold (in z-score space) filters out within-distribution noise so
          that only genuinely anomalous states contribute a penalty.
        - Running normalization keeps the penalty scale invariant to changes in
          the absolute reconstruction error during training.
        - tanh bounds the penalty to prevent extreme rare events from dominating
          the Q target.
        - Lambda warmup avoids destabilizing CQL during early training when the
          Q function is poorly calibrated.

    Args:
        lambda_max: Final penalty coefficient.
        warmup_steps: Steps over which lambda linearly ramps from 0 to lambda_max.
                      Set to 0 to disable warmup (use lambda_max from step 0).
        threshold_z: Z-score threshold below which no penalty is applied.
        use_normalization: If False, skip running normalization (use raw error).
        use_tanh: If False, skip the tanh squashing.
    """

    def __init__(
        self,
        lambda_max: float = 0.5,
        warmup_steps: int = 0,
        threshold_z: float = 1.0,
        use_normalization: bool = True,
        use_tanh: bool = True,
    ) -> None:
        self.lambda_max = lambda_max
        self.warmup_steps = warmup_steps
        self.threshold_z = threshold_z
        self.use_normalization = use_normalization
        self.use_tanh = use_tanh

        self.running = RunningMeanStd()
        self.step = 0

    def current_lambda(self) -> float:
        """Lambda at the current training step (with optional warmup)."""
        if self.warmup_steps <= 0:
            return self.lambda_max
        return self.lambda_max * min(1.0, self.step / self.warmup_steps)

    def shape(
        self,
        reward: torch.Tensor,
        recon_error: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute shaped reward and return diagnostics.

        Args:
            reward: Original reward of shape (batch, 1) or (batch,).
            recon_error: DAE reconstruction error of shape (batch, 1) or (batch,).

        Returns:
            shaped_reward: Same shape as reward.
            info: Dict with keys 'penalty_mean', 'recon_mean', 'lambda',
                  'penalty_active_frac' for logging.
        """
        # Match shapes
        if recon_error.dim() != reward.dim():
            recon_error = recon_error.view_as(reward)

        # Update running stats with current batch
        if self.use_normalization:
            self.running.update(recon_error)
            normalized = (recon_error - self.running.mean) / self.running.std
        else:
            normalized = recon_error

        # Threshold filtering (z-score above threshold_z)
        clipped = torch.clamp(normalized - self.threshold_z, min=0.0)

        # Bounded penalty
        penalty = torch.tanh(clipped) if self.use_tanh else clipped

        # Adaptive lambda
        lam = self.current_lambda()
        shaped = reward - lam * penalty
        self.step += 1

        info = {
            "lambda": lam,
            "penalty_mean": float(penalty.mean().item()),
            "recon_mean": float(recon_error.mean().item()),
            "penalty_active_frac": float((penalty > 0).float().mean().item()),
            "running_mean": self.running.mean,
            "running_std": self.running.std,
        }
        return shaped, info

    def state_dict(self) -> Dict:
        """Serialize shaper state (for checkpointing)."""
        return {
            "lambda_max": self.lambda_max,
            "warmup_steps": self.warmup_steps,
            "threshold_z": self.threshold_z,
            "use_normalization": self.use_normalization,
            "use_tanh": self.use_tanh,
            "running_mean": self.running.mean,
            "running_var": self.running.var,
            "running_count": self.running.count,
            "step": self.step,
        }

    def load_state_dict(self, state: Dict) -> None:
        """Restore shaper state from checkpoint."""
        self.lambda_max = state["lambda_max"]
        self.warmup_steps = state["warmup_steps"]
        self.threshold_z = state["threshold_z"]
        self.use_normalization = state["use_normalization"]
        self.use_tanh = state["use_tanh"]
        self.running.mean = state["running_mean"]
        self.running.var = state["running_var"]
        self.running.count = state["running_count"]
        self.step = state["step"]