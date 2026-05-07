"""
Denoising Autoencoder (DAE) for Robust Offline RL.

This module provides:
    - DAE: A denoising autoencoder model with LayerNorm and Dropout.
    - Noise injection strategies: Gaussian, Masking, and Mixed.
    - pretrain_dae: Pretraining routine with train/val split, early stopping,
      LR scheduling, gradient clipping, and L2 regularization.

The pretrained DAE is used by DDPGCQL_DAE to detect anomalous (out-of-distribution)
states via reconstruction error, which is then used as a penalty in reward shaping.
"""

from typing import Callable, Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==========================================================
# DAE Network
# ==========================================================
class DAE(nn.Module):
    """Denoising Autoencoder for state reconstruction.

    Architecture:
        Encoder: state_dim -> hidden_dim -> hidden_dim/2 -> latent_dim
        Decoder: latent_dim -> hidden_dim/2 -> hidden_dim -> state_dim

    Each linear layer is followed by LayerNorm, ReLU, and Dropout
    (except the bottleneck and output layers).

    Args:
        state_dim: Dimension of the input state (e.g., 19 for AD4RL).
        latent_dim: Dimension of the bottleneck latent representation.
        hidden_dim: Width of the largest hidden layer.
        dropout: Dropout probability applied after each ReLU.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map input state to latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent representation back to state space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode -> decode pass."""
        return self.decode(self.encode(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample reconstruction MSE.

        Always evaluated in eval mode (Dropout disabled) to ensure deterministic
        error values during reward shaping. The original training mode is restored.

        Args:
            x: Input states of shape (batch_size, state_dim).

        Returns:
            Reconstruction error of shape (batch_size, 1).
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            x_recon = self.forward(x)
            error = F.mse_loss(x_recon, x, reduction="none").mean(dim=-1, keepdim=True)
        if was_training:
            self.train()
        return error


# ==========================================================
# Noise Injection Strategies
# ==========================================================
def add_gaussian_noise(x: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    """Add zero-mean Gaussian noise. Simulates sensor measurement noise."""
    return x + torch.randn_like(x) * std


def add_masking_noise(x: torch.Tensor, mask_prob: float = 0.1) -> torch.Tensor:
    """Randomly zero out features. Simulates sensor dropouts/occlusions."""
    mask = (torch.rand_like(x) > mask_prob).float()
    return x * mask


def add_mixed_noise(
    x: torch.Tensor,
    gaussian_std: float = 0.1,
    mask_prob: float = 0.05,
) -> torch.Tensor:
    """Apply Gaussian noise then masking. Simulates realistic combined faults."""
    return add_masking_noise(add_gaussian_noise(x, std=gaussian_std), mask_prob=mask_prob)


NOISE_FNS: Dict[str, Callable] = {
    "gaussian": add_gaussian_noise,
    "masking": add_masking_noise,
    "mixed": add_mixed_noise,
}


def _apply_noise(
    states: torch.Tensor,
    noise_type: str,
    noise_std: float,
    mask_prob: float,
) -> torch.Tensor:
    """Dispatch noise function based on noise_type string."""
    if noise_type == "gaussian":
        return add_gaussian_noise(states, std=noise_std)
    if noise_type == "masking":
        return add_masking_noise(states, mask_prob=mask_prob)
    if noise_type == "mixed":
        return add_mixed_noise(states, gaussian_std=noise_std, mask_prob=mask_prob)
    raise ValueError(f"Unknown noise_type: {noise_type}. Must be one of {list(NOISE_FNS)}.")


# ==========================================================
# Pretraining
# ==========================================================
def pretrain_dae(
    dae: DAE,
    replay_buffer,
    device: torch.device,
    noise_type: str = "mixed",
    noise_std: float = 0.1,
    mask_prob: float = 0.05,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    val_split: float = 0.1,
    patience: int = 10,
    grad_clip: float = 1.0,
    log_wandb: bool = False,
) -> Tuple[DAE, Dict[str, List[float]]]:
    """Pretrain DAE on clean data with denoising objective.

    The DAE is trained to reconstruct CLEAN states from NOISY inputs.
    Train/val split, early stopping, LR scheduling, gradient clipping,
    and best-model selection are all built in.

    Args:
        dae: DAE module to train (modified in place).
        replay_buffer: Buffer with .sample(batch_size) -> (state, action, ...).
        device: Torch device (cpu or cuda).
        noise_type: One of {'gaussian', 'masking', 'mixed'}.
        noise_std: Std of Gaussian noise.
        mask_prob: Masking probability.
        epochs: Max epochs (early stopping may end earlier).
        batch_size: Mini-batch size.
        lr: Initial learning rate.
        weight_decay: L2 regularization strength.
        val_split: Fraction of buffer used for validation.
        patience: Early stopping patience (in epochs).
        grad_clip: Max gradient norm.
        log_wandb: Whether to log per-epoch metrics to WandB.

    Returns:
        (dae, history) where history has keys 'train_loss', 'val_loss', 'best_epoch'.
    """
    optimizer = torch.optim.Adam(dae.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    if noise_type not in NOISE_FNS:
        raise ValueError(f"noise_type must be one of {list(NOISE_FNS)}")

    buffer_size = getattr(replay_buffer, "size", None)
    if buffer_size is None:
        buffer_size = len(replay_buffer)
    val_size = int(buffer_size * val_split)

    print(f"\n[DAE Pretrain] noise_type={noise_type}, epochs={epochs}, batch_size={batch_size}")
    print(f"[DAE Pretrain] buffer_size={buffer_size}, val_size={val_size}, lr={lr}, wd={weight_decay}")

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "best_epoch": 0}
    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    no_improve_count = 0

    for epoch in range(epochs):
        # ---- Train ----
        dae.train()
        train_losses: List[float] = []
        n_iters = max(1, (buffer_size - val_size) // batch_size)

        for _ in range(n_iters):
            states = replay_buffer.sample(batch_size)[0].to(device)
            x_noisy = _apply_noise(states, noise_type, noise_std, mask_prob)
            x_recon = dae(x_noisy)
            loss = F.mse_loss(x_recon, states)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dae.parameters(), max_norm=grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        # ---- Validation ----
        dae.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for _ in range(max(1, val_size // batch_size)):
                states = replay_buffer.sample(batch_size)[0].to(device)
                x_noisy = _apply_noise(states, noise_type, noise_std, mask_prob)
                x_recon = dae(x_noisy)
                val_losses.append(F.mse_loss(x_recon, states).item())

        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if log_wandb:
            try:
                import wandb
                wandb.log({
                    "dae/train_loss": train_loss,
                    "dae/val_loss": val_loss,
                    "dae/lr": current_lr,
                    "dae/epoch": epoch,
                })
            except ImportError:
                pass

        # ---- Early Stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in dae.state_dict().items()}
            history["best_epoch"] = epoch
            no_improve_count = 0
        else:
            no_improve_count += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch [{epoch+1:3d}/{epochs}]  "
                f"train: {train_loss:.6f}  val: {val_loss:.6f}  lr: {current_lr:.2e}"
            )

        if no_improve_count >= patience:
            print(
                f"  [Early Stopping] no improvement for {patience} epochs. "
                f"best_epoch={history['best_epoch']+1}, best_val_loss={best_val_loss:.6f}"
            )
            break

    if best_state is not None:
        dae.load_state_dict(best_state)
    dae.eval()
    print(f"[DAE Pretrain] Done. best_val_loss={best_val_loss:.6f} at epoch {history['best_epoch']+1}\n")

    return dae, history