import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Target data: 2D ring distribution
def sample_ring(n_samples, radius=2.0, noise=0.1):
    angles = np.random.uniform(0, 2 * np.pi, size=n_samples)
    r = radius + np.random.normal(0.0, noise, size=n_samples)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    data = np.stack([x, y], axis=1).astype(np.float32)
    return data

# Generate a fixed data buffer
N_DATA = 20_000
data_array = sample_ring(N_DATA)
data_tensor = torch.from_numpy(data_array).to(device)

class VectorField(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, t):
        """
        x: (batch, 2)
        t: (batch, 1)
        returns: v_theta(x, t) of shape (batch, 2)
        """
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)

model = VectorField().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def flow_matching_train_step(batch_size=512):
    model.train()

    # 1. Sample time t ~ Uniform[0, 1]
    t = torch.rand(batch_size, 1, device=device)

    # 2. Sample base points x0 ~ N(0, I)
    x0 = torch.randn(batch_size, 2, device=device)

    # 3. Sample data points x1 from the target ring
    idx = torch.randint(0, N_DATA, (batch_size,), device=device)
    x1 = data_tensor[idx]

    # 4. Interpolation path: x_t = (1 - t) * x0 + t * x1
    x_t = (1.0 - t) * x0 + t * x1

    # 5. Conditional velocity along the interpolation path: u = x1 - x0
    target_velocity = x1 - x0

    # 6. Neural prediction v_theta(x_t, t)
    pred_velocity = model(x_t, t)

    # 7. MSE loss between predicted and target velocities
    loss = torch.mean((pred_velocity - target_velocity) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

num_steps = 4000  # you can start with 1000 just to see it working
losses = []

for step in range(1, num_steps + 1):
    loss = flow_matching_train_step(batch_size=512)
    losses.append(loss)
    if step % 200 == 0:
        print(f"Step {step}/{num_steps}, loss = {loss:.4f}")

# Plot training loss
plt.figure()
plt.plot(losses)
plt.xlabel("Training step")
plt.ylabel("Flow Matching loss (MSE)")
plt.title("Training loss for Flow Matching on 2D ring")
plt.show()

def sample_from_flow(n_samples=5000, n_steps=100):
    model.eval()
    with torch.no_grad():
        # Start from base distribution x0 ~ N(0, I)
        x = torch.randn(n_samples, 2, device=device)
        T = 1.0
        dt = T / n_steps

        # Forward integration from t=0 to t=1
        for k in range(n_steps):
            t_val = k * dt  # scalar time
            t = torch.full((n_samples, 1), t_val, device=device)
            v = model(x, t)
            x = x + dt * v   # Euler step: x_{t+dt} = x_t + dt * v(x_t, t)

    return x.cpu().numpy()

samples = sample_from_flow(n_samples=5000, n_steps=100)

# Target data (ring)
plt.figure()
plt.scatter(data_array[:5000, 0], data_array[:5000, 1], s=5, alpha=0.5)
plt.axis("equal")
plt.title("Target data: 2D ring distribution")
plt.show()

# Base distribution (Gaussian)
base_samples = np.random.randn(5000, 2).astype(np.float32)
plt.figure()
plt.scatter(base_samples[:, 0], base_samples[:, 1], s=5, alpha=0.5)
plt.axis("equal")
plt.title("Base distribution: 2D standard Gaussian")
plt.show()

# Samples from the Flow Matching model
plt.figure()
plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5)
plt.axis("equal")
plt.title("Samples from Flow Matching model")
plt.show()
