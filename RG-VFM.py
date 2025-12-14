import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------
# 1. S^2 helpers
# --------------------------------------------------------

def normalize_s2(x, eps=1e-8):
    """Project vectors to S^2 by normalizing along last dim."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def sample_uniform_s2(n):
    """Uniform-ish samples on S^2 via Gaussian normalization."""
    x = torch.randn(n, 3, device=device)
    return normalize_s2(x)

def sample_spherical_cluster(n, center, kappa=20.0):
    """
    Approx spherical Gaussian around 'center' on S^2:
    sample ~ N(kappa * center, I), then normalize.
    """
    center = center.to(device)
    mean = kappa * center
    x = mean + torch.randn(n, 3, device=device)
    return normalize_s2(x)

def geodesic_distance_s2(x, y, eps=1e-6):
    """
    Geodesic distance on S^2: d(x,y) = arccos(<x,y>).
    x, y: (batch, 3)
    """
    dot = (x * y).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(dot)
    return theta  # shape (batch,)

def log_map_s2(x, y, eps=1e-6):
    """
    Log map on S^2: log_x(y) in T_x S^2.
    x, y: (batch, 3) on S^2.
    Returns tangent vector v at x such that exp_x(v) ~ y.
    """
    dot = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(dot)               # (batch, 1)
    sin_theta = torch.sin(theta)

    # Handle small angle: v ~ 0
    small = theta < (1e-4)
    # Direction orthogonal to x:
    # y - <x,y> x has norm sin(theta)
    v = y - dot * x
    v = v / (sin_theta + eps) * theta     # length ~ theta

    # For very small theta, just use zero vector
    v = torch.where(small, torch.zeros_like(v), v)
    return v  # (batch, 3), tangent at x

def exp_map_s2(x, v, eps=1e-6):
    """
    Exponential map on S^2: exp_x(v).
    x: (batch, 3), v: (batch, 3) in T_x S^2.
    """
    norm_v = v.norm(dim=-1, keepdim=True)  # (batch, 1)
    small = norm_v < 1e-6

    # Normalized direction
    dir_v = v / (norm_v + eps)
    # Geodesic step
    cos_term = torch.cos(norm_v)
    sin_term = torch.sin(norm_v)

    y = cos_term * x + sin_term * dir_v
    y = torch.where(small, x, y)  # if v ~ 0, stay at x
    return normalize_s2(y)

def slerp_s2(x0, x1, t, eps=1e-6):
    """
    Spherical linear interpolation on S^2.
    x0, x1: (batch, 3) on S^2
    t: (batch, 1) in [0, 1]
    returns: x_t (batch, 3)
    """
    dot = (x0 * x1).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(dot)          # (batch, 1)
    sin_theta = torch.sin(theta)     # (batch, 1)

    w0 = torch.sin((1.0 - t) * theta) / (sin_theta + eps)
    w1 = torch.sin(t * theta) / (sin_theta + eps)

    x_t = w0 * x0 + w1 * x1
    return normalize_s2(x_t)

# --------------------------------------------------------
# 2. Target data on S^2 (two clusters)
# --------------------------------------------------------

N_DATA = 10_000

center1 = torch.tensor([0.0, 0.0, 1.0])   # north
center2 = torch.tensor([0.0, 0.0, -1.0])  # south

data_cluster1 = sample_spherical_cluster(N_DATA // 2, center1, kappa=30.0)
data_cluster2 = sample_spherical_cluster(N_DATA // 2, center2, kappa=30.0)
data_tensor = torch.cat([data_cluster1, data_cluster2], dim=0)  # (N_DATA, 3)

# --------------------------------------------------------
# 3. RG-VFM-style network: predict endpoint mu_phi(x_t, t) in S^2
# --------------------------------------------------------

class EndpointNetS2(nn.Module):
    """
    RG-VFM-style: given (x_t, t), predict endpoint mu_phi(x_t, t) on S^2.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),  # (x (3), t (1)) -> 4
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=-1)   # (batch, 4)
        raw = self.net(inp)              # unconstrained in R^3
        mu = normalize_s2(raw)           # project to S^2
        return mu                        # (batch, 3)

net = EndpointNetS2(hidden_dim=64).to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-3)

# --------------------------------------------------------
# 4. RG-VFM training step on S^2 (endpoint geodesic loss)
# --------------------------------------------------------

def rg_vfm_train_step_s2(batch_size=512):
    """
    One RG-VFM-style step:
      - sample t, x0 (base), x1 (data)
      - compute x_t via geodesic interpolation
      - predict endpoint mu_phi(x_t, t)
      - loss = squared geodesic distance d_S2(x1, mu_phi(x_t,t))^2
    """
    net.train()

    # 1. t ~ Uniform[0,1]
    t = torch.rand(batch_size, 1, device=device)

    # 2. base x0 ~ uniform on S^2
    x0 = sample_uniform_s2(batch_size)

    # 3. data x1 from target on S^2
    idx = torch.randint(0, data_tensor.shape[0], (batch_size,), device=device)
    x1 = data_tensor[idx]  # (batch, 3)

    # 4. geodesic interpolation x_t on S^2
    x_t = slerp_s2(x0, x1, t)  # (batch, 3)

    # 5. predict endpoint mu_phi(x_t, t)
    mu = net(x_t, t)           # (batch, 3) on S^2

    # 6. RG-VFM loss: squared geodesic distance between true and predicted endpoints
    dist = geodesic_distance_s2(x1, mu)   # (batch,)
    loss = (dist ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# --------------------------------------------------------
# 5. Train
# --------------------------------------------------------

num_steps = 2000
losses = []

for step in range(1, num_steps + 1):
    loss = rg_vfm_train_step_s2(batch_size=512)
    losses.append(loss)
    if step % 200 == 0:
        print(f"Step {step}/{num_steps}, RG-VFM loss = {loss:.4f}")

plt.figure()
plt.plot(losses)
plt.xlabel("Training step")
plt.ylabel("RG-VFM endpoint loss (geodesic MSE)")
plt.title("RG-VFM-style training on S^2 (bimodal data)")
plt.show()

# --------------------------------------------------------
# 6. Define vector field v_phi from predicted endpoint
# --------------------------------------------------------

@torch.no_grad()
def v_phi_s2(x, t, eps=1e-4):
    """
    Compute RG-VFM-style vector field:
      v_phi(x,t) = (1 / (1 - t)) * log_x(mu_phi(x,t)).
    x: (batch, 3) on S^2
    t: (batch, 1)
    returns: tangent vector in T_x S^2
    """
    mu = net(x, t)             # predicted endpoint on S^2
    v_dir = log_map_s2(x, mu)  # tangent vector towards mu
    v = v_dir / (1.0 - t + eps)
    return v

# --------------------------------------------------------
# 7. Sampling from the learned flow on S^2
# --------------------------------------------------------

def sample_from_rg_vfm_s2(n_samples=5000, n_steps=80):
    """
    Sample from the learned RG-VFM flow:
      - start from base ~ uniform on S^2
      - integrate dx/dt = v_phi_s2(x,t)
      - use exponential map for each Euler step to stay on S^2
    """
    net.eval()
    with torch.no_grad():
        x = sample_uniform_s2(n_samples)   # base on S^2
        T = 1.0
        dt = T / n_steps

        for k in range(n_steps):
            t_val = (k + 0.5) * dt  # midpoints in (0,1)
            t = torch.full((n_samples, 1), t_val, device=device)
            v = v_phi_s2(x, t)      # tangent vector
            step_v = dt * v
            x = exp_map_s2(x, step_v)

    return x.cpu().numpy()

samples = sample_from_rg_vfm_s2(n_samples=5000, n_steps=80)

# --------------------------------------------------------
# 8. Simple 3D visualization on the sphere
# --------------------------------------------------------

def plot_unit_sphere(ax, alpha=0.15, resolution=40):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, rstride=2, cstride=2,
                    alpha=alpha, edgecolor="none")

def plot_points_on_s2(points, title="Points on S^2", color="C0"):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    plot_unit_sphere(ax, alpha=0.15)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=5, alpha=0.7, c=color)
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    plt.show()

# Plot target data
idx_plot = torch.randperm(data_tensor.shape[0])[:2000]
plot_points_on_s2(data_tensor[idx_plot], title="Target data on S^2", color="C0")

# Plot RG-VFM samples
plot_points_on_s2(samples, title="RG-VFM samples on S^2", color="C1")
