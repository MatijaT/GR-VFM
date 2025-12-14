import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 1. Data: 2D mixture of Gaussians
# ----------------------------

def sample_mog_2d(n_samples):
    """
    2D mixture of two Gaussians centered at (-2,0) and (2,0).
    """
    n = n_samples
    # mixture component indicator (0 or 1)
    mix = torch.bernoulli(0.5 * torch.ones(n, 1, device=device))  # 1 = left, 0 = right

    # left and right component samples
    left  = torch.stack([
        -2.0 + 0.2 * torch.randn(n, device=device),
         0.0 + 0.2 * torch.randn(n, device=device)
    ], dim=-1)  # (n, 2)

    right = torch.stack([
         2.0 + 0.2 * torch.randn(n, device=device),
         0.0 + 0.2 * torch.randn(n, device=device)
    ], dim=-1)  # (n, 2)

    x = mix * left + (1.0 - mix) * right  # broadcast over 2 dims
    return x

# Pre-generate a large data buffer for convenience
N_DATA = 20_000
data_tensor = sample_mog_2d(N_DATA)  # (N_DATA, 2)

# ----------------------------
# 2. Network for mu_phi(x,t)
# ----------------------------

class EndpointPosteriorNet(nn.Module):
    """
    Network to parameterize the mean mu_phi(x, t) of q_phi(x1 | x,t).
    Input: x in R^2, t in [0,1]
    Output: mu in R^2
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),  # (x1, x2, t)
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, t):
        # x: (batch, 2), t: (batch, 1)
        inp = torch.cat([x, t], dim=-1)  # (batch, 3)
        mu = self.net(inp)               # (batch, 2)
        return mu

net = EndpointPosteriorNet(hidden_dim=64).to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-3)

sigma = 0.1  # fixed variance for q_phi (not used explicitly except in scaling)

# ----------------------------
# 3. VFM training step
# ----------------------------

def vfm_train_step(batch_size=512):
    """
    One VFM step:
      - sample t, x0, x1, form x_t
      - predict mu_phi(x_t, t)
      - loss ~ MSE between x1 and mu_phi
    """
    net.train()

    # 1. t ~ Uniform[0,1]
    t = torch.rand(batch_size, 1, device=device)

    # 2. base x0 ~ N(0, I)
    x0 = torch.randn(batch_size, 2, device=device)

    # 3. data x1 from mixture of Gaussians
    idx = torch.randint(0, N_DATA, (batch_size,), device=device)
    x1 = data_tensor[idx]  # (batch, 2)

    # 4. interpolation path: x_t = (1-t)x0 + t x1
    x_t = (1.0 - t) * x0 + t * x1

    # 5. predict mean of q_phi(x1 | x_t, t)
    mu = net(x_t, t)  # (batch, 2)

    # 6. VFM loss: negative log-likelihood under Gaussian
    #    For fixed sigma^2, NLL is proportional to squared error.
    mse = torch.mean((x1 - mu) ** 2)
    loss = mse / (2.0 * sigma**2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# ----------------------------
# 4. Training loop
# ----------------------------

num_steps = 4000
losses = []

for step in range(1, num_steps + 1):
    loss = vfm_train_step(batch_size=512)
    losses.append(loss)
    if step % 200 == 0:
        print(f"Step {step}/{num_steps}, loss = {loss:.4f}")

plt.figure()
plt.plot(losses)
plt.xlabel("Training step")
plt.ylabel("VFM loss (approx NLL)")
plt.title("VFM training on 2D mixture of Gaussians")
plt.show()

# ----------------------------
# 5. Vector field and ODE sampling
# ----------------------------

@torch.no_grad()
def v_phi(x, t):
    """
    Compute v_phi(x,t) = (mu_phi(x,t) - x)/(1-t)
    x: (batch, 2)
    t: (batch, 1)
    """
    mu = net(x, t)
    return (mu - x) / (1.0 - t + 1e-4)  # small epsilon to avoid division at t=1

def sample_from_vfm(n_samples=5000, n_steps=100):
    """
    Sample from the learned flow by integrating dx/dt = v_phi(x,t)
    with Euler steps from t=0 to 1.
    """
    net.eval()
    with torch.no_grad():
        # Start from base distribution x0 ~ N(0,I)
        x = torch.randn(n_samples, 2, device=device)
        T = 1.0
        dt = T / n_steps
        for k in range(n_steps):
            t_val = k * dt
            t = torch.full((n_samples, 1), t_val, device=device)
            v = v_phi(x, t)
            x = x + dt * v
    return x.cpu().numpy()

samples = sample_from_vfm(n_samples=5000, n_steps=150)

# ----------------------------
# 6. Visualizations
# ----------------------------

# Target data
data_np = data_tensor[:5000].cpu().numpy()
plt.figure()
plt.scatter(data_np[:, 0], data_np[:, 1], s=5, alpha=0.5)
plt.axis("equal")
plt.title("Target data: 2D mixture of Gaussians")
plt.show()

# Base Gaussian
base_np = torch.randn(5000, 2).cpu().numpy()
plt.figure()
plt.scatter(base_np[:, 0], base_np[:, 1], s=5, alpha=0.5)
plt.axis("equal")
plt.title("Base distribution: N(0, I)")
plt.show()

# VFM samples
plt.figure()
plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5)
plt.axis("equal")
plt.title("Samples from VFM flow")
plt.show()
