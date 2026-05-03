import torch
import torch.nn as nn
import torch.nn.functional as F


class DAE(nn.Module):
    def __init__(self, state_dim, latent_dim=32):
        super(DAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x):
        """노이즈가 섞인 state를 넣으면 재구성오차를 반환 (reward shaping용)"""
        with torch.no_grad():
            x_recon = self.forward(x)
            error = F.mse_loss(x_recon, x, reduction='none').mean(dim=-1, keepdim=True)
        return error  # shape: (batch_size, 1)


def pretrain_dae(dae, replay_buffer, device, noise_std=0.1, epochs=50, batch_size=256, lr=1e-3):
    """
    정상 데이터로 DAE 사전학습.
    clean state에 노이즈를 추가한 뒤 원본 clean state를 복원하도록 학습.
    """
    optimizer = torch.optim.Adam(dae.parameters(), lr=lr)
    dae.train()

    print(f"\n[DAE Pretrain] epochs={epochs}, noise_std={noise_std}, batch_size={batch_size}")
    for epoch in range(epochs):
        batch = replay_buffer.sample(batch_size)
        states = batch[0].to(device)  # clean state

        x_noisy = states + torch.randn_like(states) * noise_std
        x_recon = dae(x_noisy)
        loss = F.mse_loss(x_recon, states)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}]  recon_loss: {loss.item():.6f}")

    dae.eval()
    print("[DAE Pretrain] Done.\n")
    return dae