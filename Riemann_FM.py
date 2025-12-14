import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Basic S^2 helpers -----

def normalize_s2(x):
    """Project vectors to S^2 by normalizing along the last dimension."""
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

def sample_uniform_s2(n):
    """Uniform-ish samples on S^2 via Gaussian normalize."""
    x = torch.randn(n, 3, device=device)
    return normalize_s2(x)

def sample_spherical_cluster(n, center, kappa=20.0):
    """
    Approx spherical Gaussian centered at 'center' on S^2:
    sample normal(mean = kappa * center), then normalize.
    center: (3,) tensor on S^2
    """
    center = center.to(device)
    mean = kappa * center
    x = mean + torch.randn(n, 3, device=device)
    return normalize_s2(x)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots in some setups

def plot_unit_sphere(ax, alpha=0.15, resolution=40):
    """
    Draw a translucent unit sphere on the given 3D axis.
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, rstride=2, cstride=2,
                    alpha=alpha, edgecolor="none")

def plot_points_on_s2(points, title="Points on S^2", color="C0"):
    """
    points: (N, 3) array-like of Cartesian points on S^2.
            Can be a NumPy array or a torch tensor.
    """
    if hasattr(points, "detach"):  # torch tensor
        points = points.detach().cpu().numpy()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Draw sphere
    plot_unit_sphere(ax, alpha=0.15, resolution=40)

    # Scatter points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=5, alpha=0.7, c=color)

    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    plt.show()

def plot_s2_data_and_samples(data_points, sample_points,
                             data_color="C0", sample_color="C1"):
    """
    Quick comparison plot: data vs samples on the same sphere.
    """
    # Convert to numpy if needed
    if hasattr(data_points, "detach"):
        data_points = data_points.detach().cpu().numpy()
    if hasattr(sample_points, "detach"):
        sample_points = sample_points.detach().cpu().numpy()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    plot_unit_sphere(ax, alpha=0.15, resolution=40)

    ax.scatter(data_points[:, 0], data_points[:, 1], data_points[:, 2],
               s=5, alpha=0.5, c=data_color, label="data")
    ax.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2],
               s=5, alpha=0.5, c=sample_color, label="samples")

    ax.set_title("Data vs Flow Matching samples on S^2")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.tight_layout()
    plt.show()


# Target "data" on S^2: two clusters (like a spherical mixture)
N_DATA = 10_000

center1 = torch.tensor([0.0, 0.0, 1.0])   # "north pole"
center2 = torch.tensor([0.0, 0.0, -1.0])  # "south pole"

data_cluster1 = sample_spherical_cluster(N_DATA // 2, center1, kappa=30.0)
data_cluster2 = sample_spherical_cluster(N_DATA // 2, center2, kappa=30.0)
data_tensor = torch.cat([data_cluster1, data_cluster2], dim=0)


data_np = data_tensor.cpu().numpy()
'''
plt.figure()
plt.scatter(data_np[:, 0], data_np[:, 1], s=3, alpha=0.4)
plt.axis("equal")
plt.title("Target data on S^2 (projected to x-y)")
plt.show()
'''
def slerp_s2(x0, x1, t, eps=1e-6):
    """
    Spherical linear interpolation on S^2.
    x0, x1: (batch, 3) on S^2
    t: (batch, 1) in [0, 1]
    returns: x_t (batch, 3), theta, sin_theta
    """
    # Dot product (clamped to avoid numerical issues)
    dot = (x0 * x1).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(dot)          # (batch, 1)
    sin_theta = torch.sin(theta)     # (batch, 1)

    # weights
    w0 = torch.sin((1.0 - t) * theta) / (sin_theta + eps)
    w1 = torch.sin(t * theta) / (sin_theta + eps)

    x_t = w0 * x0 + w1 * x1
    x_t = normalize_s2(x_t)  # stay on S^2
    return x_t, theta, sin_theta

def geodesic_velocity_s2(x0, x1, t, theta, sin_theta, eps=1e-6):
    """
    Time derivative of the S^2 geodesic (slerp) at time t.
    Input shapes:
        x0, x1: (batch, 3)
        t, theta, sin_theta: (batch, 1)
    returns: v_geo (batch, 3)
    """
    # d/dt [ sin((1-t)θ)/sinθ ] = -θ cos((1-t)θ)/sinθ
    dA_dt = -theta * torch.cos((1.0 - t) * theta) / (sin_theta + eps)
    # d/dt [ sin(tθ)/sinθ ] = θ cos(tθ)/sinθ
    dB_dt =  theta * torch.cos(t * theta) / (sin_theta + eps)

    v = dA_dt * x0 + dB_dt * x1   # (batch, 3)
    return v

def project_to_tangent(x, v):
    """
    Project v onto tangent space at x on S^2.
    x, v: (batch, 3)
    """
    dot = (v * x).sum(dim=-1, keepdim=True)
    return v - dot * x


class S2VectorField(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # input: (x (3), t (1)) -> 4D -> hidden -> 3D
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x, t):
        """
        x: (batch, 3) on S^2
        t: (batch, 1)
        returns tangent vector v_theta(x,t) in R^3, orthogonal to x
        """
        inp = torch.cat([x, t], dim=-1)  # (batch, 4)
        w = self.net(inp)               # unconstrained (batch, 3)
        v = project_to_tangent(x, w)    # ensure tangent
        return v

model = S2VectorField(hidden_dim=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def flow_matching_train_step_s2(batch_size=512):
    model.train()

    # 1. Sample time t ~ Uniform[0,1]
    t = torch.rand(batch_size, 1, device=device)

    # 2. Sample base points x0 ~ base distribution on S^2
    x0 = sample_uniform_s2(batch_size)

    # 3. Sample data points x1 from the target "data" on S^2
    idx = torch.randint(0, data_tensor.shape[0], (batch_size,), device=device)
    x1 = data_tensor[idx]

    # 4. Geodesic interpolation x_t on S^2
    x_t, theta, sin_theta = slerp_s2(x0, x1, t)

    # 5. Conditional geodesic velocity along this path
    v_geo = geodesic_velocity_s2(x0, x1, t, theta, sin_theta)
    v_geo = project_to_tangent(x_t, v_geo)  # ensure tangent

    # 6. Neural prediction of tangent velocity
    v_pred = model(x_t, t)

    # 7. MSE loss between predicted and target velocities
    loss = torch.mean((v_pred - v_geo) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


num_steps = 1500
losses = []

for step in range(1, num_steps + 1):
    loss = flow_matching_train_step_s2(batch_size=512)
    losses.append(loss)
    if step % 100 == 0:
        print(f"Step {step}/{num_steps}, loss = {loss:.4f}")

plt.figure()
plt.plot(losses)
plt.xlabel("Training step")
plt.ylabel("Flow Matching loss")
plt.title("Riemannian Flow Matching on S^2 (bimodal target)")
plt.show()


def sample_from_flow_s2(n_samples=5000, n_steps=80):
    model.eval()
    with torch.no_grad():
        x = sample_uniform_s2(n_samples)  # base distribution on S^2
        T = 1.0
        dt = T / n_steps
        for k in range(n_steps):
            t_val = k * dt
            t = torch.full((n_samples, 1), t_val, device=device)
            v = model(x, t)
            x = x + dt * v
            x = normalize_s2(x)  # re-project to S^2
    return x.cpu().numpy()

samples = sample_from_flow_s2(n_samples=5000, n_steps=80)

# Project data and samples on (x,y) plane for 2D scatter
'''
plt.figure()
plt.scatter(data_np[:, 0], data_np[:, 1], s=3, alpha=0.4, label="data")
plt.axis("equal")
plt.title("Target data on S^2 (x-y projection)")
plt.legend()
plt.show()

samples_np = samples
plt.figure()
plt.scatter(samples_np[:, 0], samples_np[:, 1], s=3, alpha=0.4, label="FM samples")
plt.axis("equal")
plt.title("Flow Matching samples on S^2 (x-y projection)")
plt.legend()
plt.show()
'''

plot_points_on_s2(data_np[:10000], title="Target data on S^2")
plot_points_on_s2(samples[:1000], title="Flow Matching samples on S^2", color="C1")
base_samples = sample_uniform_s2(1000)
plot_points_on_s2(base_samples, title="Base uniform samples on S^2", color="C2")