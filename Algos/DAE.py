import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==========================================================
# DAE Network
# ==========================================================
class DAE(nn.Module):
    """
    Denoising Autoencoder for robust state representation.
    - 3-layer encoder/decoder
    - LayerNorm + Dropout for regularization
    - Latent bottleneck for compressed representation
    """
    def __init__(self, state_dim, latent_dim=32, hidden_dim=128, dropout=0.1):
        super(DAE, self).__init__()

        # Encoder: state_dim -> hidden -> hidden//2 -> latent
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

        # Decoder: latent -> hidden//2 -> hidden -> state_dim
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

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def reconstruction_error(self, x):
        """
        주어진 state(노이즈 포함 가능)에 대한 재구성오차 반환.
        Reward shaping에 사용. eval 모드에서 계산 후 원래 모드 복원.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            x_recon = self.forward(x)
            error = F.mse_loss(x_recon, x, reduction='none').mean(dim=-1, keepdim=True)
        if was_training:
            self.train()
        return error  # shape: (batch_size, 1)


# ==========================================================
# Noise Injection Strategies
# ==========================================================
def add_gaussian_noise(x, std=0.1):
    """Gaussian noise: 센서 측정 노이즈를 모사"""
    return x + torch.randn_like(x) * std


def add_masking_noise(x, mask_prob=0.1):
    """Masking noise: 일부 센서가 결측되는 상황 모사"""
    mask = (torch.rand_like(x) > mask_prob).float()
    return x * mask


def add_mixed_noise(x, gaussian_std=0.1, mask_prob=0.05):
    """Gaussian + Masking 혼합 — 더 다양한 사고 상황 시뮬레이션"""
    x = add_gaussian_noise(x, std=gaussian_std)
    x = add_masking_noise(x, mask_prob=mask_prob)
    return x


NOISE_FNS = {
    'gaussian': add_gaussian_noise,
    'masking': add_masking_noise,
    'mixed': add_mixed_noise,
}


# ==========================================================
# Pretraining with Train/Val Split + Early Stopping + LR Scheduler
# ==========================================================
def pretrain_dae(
    dae,
    replay_buffer,
    device,
    noise_type='mixed',
    noise_std=0.1,
    mask_prob=0.05,
    epochs=100,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-5,
    val_split=0.1,
    patience=10,
    log_wandb=False,
):
    """
    DAE 사전학습 (정상 데이터만 사용)
    - Train/Val 분리: replay buffer를 90/10 비율로 분리해서 검증
    - Early stopping: val loss가 patience epoch 동안 개선 없으면 종료
    - LR scheduler: ReduceLROnPlateau로 val loss에 따라 LR 자동 감소
    - L2 regularization: weight_decay 적용
    - Gradient clipping: max_norm=1.0

    Returns:
        dae: 학습된 DAE 모델
        history: dict with 'train_loss', 'val_loss', 'best_epoch'
    """
    optimizer = torch.optim.Adam(dae.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 노이즈 함수 선택
    if noise_type not in NOISE_FNS:
        raise ValueError(f"noise_type must be one of {list(NOISE_FNS.keys())}")
    noise_fn = NOISE_FNS[noise_type]

    # Replay buffer 크기
    buffer_size = replay_buffer.size if hasattr(replay_buffer, 'size') else len(replay_buffer)
    val_size = int(buffer_size * val_split)

    print(f"\n[DAE Pretrain] noise_type={noise_type}, epochs={epochs}, batch_size={batch_size}")
    print(f"[DAE Pretrain] buffer_size={buffer_size}, val_size={val_size}, lr={lr}, wd={weight_decay}")

    history = {'train_loss': [], 'val_loss': [], 'best_epoch': 0}
    best_val_loss = float('inf')
    best_state = None
    no_improve_count = 0

    def _apply_noise(states):
        if noise_type == 'gaussian':
            return noise_fn(states, std=noise_std)
        elif noise_type == 'masking':
            return noise_fn(states, mask_prob=mask_prob)
        else:  # mixed
            return noise_fn(states, gaussian_std=noise_std, mask_prob=mask_prob)

    for epoch in range(epochs):
        # ---------- Train ----------
        dae.train()
        train_losses = []
        n_iters = max(1, (buffer_size - val_size) // batch_size)

        for _ in range(n_iters):
            batch = replay_buffer.sample(batch_size)
            states = batch[0].to(device)

            x_noisy = _apply_noise(states)
            x_recon = dae(x_noisy)
            loss = F.mse_loss(x_recon, states)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dae.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        # ---------- Validation ----------
        dae.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(max(1, val_size // batch_size)):
                batch = replay_buffer.sample(batch_size)
                states = batch[0].to(device)
                x_noisy = _apply_noise(states)
                x_recon = dae(x_noisy)
                val_loss = F.mse_loss(x_recon, states)
                val_losses.append(val_loss.item())

        val_loss = float(np.mean(val_losses))

        # LR scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if log_wandb:
            try:
                import wandb
                wandb.log({
                    'dae/train_loss': train_loss,
                    'dae/val_loss': val_loss,
                    'dae/lr': current_lr,
                    'dae/epoch': epoch,
                })
            except ImportError:
                pass

        # ---------- Early Stopping ----------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in dae.state_dict().items()}
            history['best_epoch'] = epoch
            no_improve_count = 0
        else:
            no_improve_count += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:3d}/{epochs}]  "
                  f"train: {train_loss:.6f}  val: {val_loss:.6f}  lr: {current_lr:.2e}")

        if no_improve_count >= patience:
            print(f"  [Early Stopping] No improvement for {patience} epochs. "
                  f"Best epoch: {history['best_epoch']+1}, best val_loss: {best_val_loss:.6f}")
            break

    # 최적 모델로 복원
    if best_state is not None:
        dae.load_state_dict(best_state)

    dae.eval()
    print(f"[DAE Pretrain] Done. Best val_loss: {best_val_loss:.6f} at epoch {history['best_epoch']+1}\n")

    return dae, history