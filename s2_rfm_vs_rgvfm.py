
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# S^2 helpers
# -----------------------------

def normalize_s2(x, eps=1e-8):
    """Project vectors onto S^2."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def sample_uniform_s2(n):
    """Uniform-ish samples on S^2 via Gaussian normalization."""
    x = torch.randn(n, 3, device=device)
    return normalize_s2(x)


def sample_spherical_cluster(n, center, kappa=20.0):
    """
    Approximate 'spherical Gaussian' around center on S^2:
    sample ~ N(kappa * center, I) and normalize.
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
    return theta  # (batch,)


def log_map_s2(x, y, eps=1e-6):
    """
    Log map on S^2: log_x(y) in T_x S^2.
    x, y: (batch, 3) on S^2.
    Returns v in T_x S^2 s.t. exp_x(v) ~ y.
    """
    dot = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(dot)                # (batch, 1)
    sin_theta = torch.sin(theta)

    # direction orthogonal to x
    v = y - dot * x
    v = v / (sin_theta + eps) * theta      # length ~ theta

    small = theta < 1e-4
    v = torch.where(small, torch.zeros_like(v), v)
    return v  # (batch, 3)


def exp_map_s2(x, v, eps=1e-6):
    """
    Exponential map on S^2: exp_x(v).
    x: (batch, 3), v: (batch, 3) in T_x S^2.
    """
    norm_v = v.norm(dim=-1, keepdim=True)  # (batch, 1)
    small = norm_v < 1e-6

    dir_v = v / (norm_v + eps)
    cos_term = torch.cos(norm_v)
    sin_term = torch.sin(norm_v)

    y = cos_term * x + sin_term * dir_v
    y = torch.where(small, x, y)
    return normalize_s2(y)


def slerp_s2(x0, x1, t, eps=1e-6):
    """
    Spherical linear interpolation on S^2.
    x0, x1: (batch, 3) on S^2
    t: (batch, 1) in [0,1]
    """
    dot = (x0 * x1).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    w0 = torch.sin((1.0 - t) * theta) / (sin_theta + eps)
    w1 = torch.sin(t * theta) / (sin_theta + eps)

    x_t = w0 * x0 + w1 * x1
    return normalize_s2(x_t)


def project_to_tangent(x, v):
    """
    Project v in R^3 to the tangent plane T_x S^2.
    """
    return v - (v * x).sum(dim=-1, keepdim=True) * x


# -----------------------------
# Data on S^2
# -----------------------------

N_DATA = 8000

center1 = torch.tensor([0.0, 0.0, 1.0])
center2 = torch.tensor([0.0, 0.0, -1.0])

data_cluster1 = sample_spherical_cluster(N_DATA // 2, center1, kappa=30.0)
data_cluster2 = sample_spherical_cluster(N_DATA // 2, center2, kappa=30.0)
data_tensor = torch.cat([data_cluster1, data_cluster2], dim=0)  # (N_DATA, 3)


# -----------------------------
# Networks
# -----------------------------

class RFMNetS2(nn.Module):
    """
    RFM-style network: (x_t, t) -> velocity in T_{x_t} S^2.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),   # x (3) + t (1)
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=-1)
        v = self.net(inp)
        v_tan = project_to_tangent(x, v)
        return v_tan


class EndpointNetS2(nn.Module):
    """
    RG-VFM-style network: (x_t, t) -> endpoint mu(x_t,t) on S^2.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=-1)
        raw = self.net(inp)
        mu = normalize_s2(raw)  # constrain to S^2
        return mu


rfm_net = RFMNetS2(hidden_dim=64).to(device)
rg_net = EndpointNetS2(hidden_dim=64).to(device)

opt_rfm = optim.Adam(rfm_net.parameters(), lr=1e-3)
opt_rg = optim.Adam(rg_net.parameters(), lr=1e-3)


# -----------------------------
# Training steps
# -----------------------------

def rfm_train_step(batch_size=512):
    """
    One RFM update step on S^2.
    Loss: velocity MSE in T_{x_t} S^2.
    """
    rfm_net.train()

    # 1. t ~ Unif[0,1]
    t = torch.rand(batch_size, 1, device=device)

    # 2. base x0 ~ uniform S^2
    x0 = sample_uniform_s2(batch_size)

    # 3. data x1
    idx = torch.randint(0, data_tensor.shape[0], (batch_size,), device=device)
    x1 = data_tensor[idx]

    # 4. geodesic interpolation
    x_t = slerp_s2(x0, x1, t)

    # 5. target velocity via log map
    eps = 1e-4
    target_v = log_map_s2(x_t, x1) / (1.0 - t + eps)

    # 6. predicted velocity
    pred_v = rfm_net(x_t, t)

    loss = ((pred_v - target_v) ** 2).sum(dim=-1).mean()

    opt_rfm.zero_grad()
    loss.backward()
    opt_rfm.step()

    return loss.item()


def rg_vfm_train_step(batch_size=512):
    """
    One RG-VFM-style update step on S^2.
    Loss: squared geodesic distance between true and predicted endpoints.
    """
    rg_net.train()

    t = torch.rand(batch_size, 1, device=device)
    x0 = sample_uniform_s2(batch_size)

    idx = torch.randint(0, data_tensor.shape[0], (batch_size,), device=device)
    x1 = data_tensor[idx]

    x_t = slerp_s2(x0, x1, t)

    mu = rg_net(x_t, t)               # predicted endpoint on S^2
    dist = geodesic_distance_s2(x1, mu)
    loss = (dist ** 2).mean()

    opt_rg.zero_grad()
    loss.backward()
    opt_rg.step()

    return loss.item()


# -----------------------------
# Vector fields and sampling
# -----------------------------

@torch.no_grad()
def v_theta_rfm(x, t):
    """RFM vector field v_theta(x,t) in T_x S^2."""
    rfm_net.eval()
    return rfm_net(x, t)


@torch.no_grad()
def v_phi_rg(x, t, eps=1e-4):
    """RG-VFM-style vector field v_phi(x,t) from endpoint prediction."""
    rg_net.eval()
    mu = rg_net(x, t)
    v_dir = log_map_s2(x, mu)
    v = v_dir / (1.0 - t + eps)
    return v


def sample_from_flow(net_type="rfm", n_samples=4000, n_steps=80):
    """
    Sample from a learned flow on S^2 by integrating dx/dt = v(x,t).
    net_type: "rfm" or "rg".
    """
    with torch.no_grad():
        x = sample_uniform_s2(n_samples)
        T = 1.0
        dt = T / n_steps

        for k in range(n_steps):
            t_val = (k + 0.5) * dt
            t = torch.full((n_samples, 1), t_val, device=device)
            if net_type == "rfm":
                v = v_theta_rfm(x, t)
            else:
                v = v_phi_rg(x, t)
            step_v = dt * v
            x = exp_map_s2(x, step_v)

    return x.cpu().numpy()


# -----------------------------
# Plotting helpers
# -----------------------------

def plot_unit_sphere(ax, alpha=0.15, resolution=40):
    """Draw a translucent unit sphere for context."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, rstride=2, cstride=2,
                    alpha=alpha, edgecolor="none")


# -----------------------------
# Main training + comparison
# -----------------------------

def main(num_steps=1500):
    losses_rfm = []
    losses_rg = []

    for step in range(1, num_steps + 1):
        loss_rfm = rfm_train_step(batch_size=512)
        loss_rg = rg_vfm_train_step(batch_size=512)

        losses_rfm.append(loss_rfm)
        losses_rg.append(loss_rg)

        if step % 100 == 0:
            print(
                f"Step {step}/{num_steps}, "
                f"RFM loss = {loss_rfm:.4f}, RG-VFM loss = {loss_rg:.4f}"
            )

    # -------------------------
    # Loss curves
    # -------------------------
    plt.figure()
    plt.plot(losses_rfm, label="RFM: velocity MSE")
    plt.plot(losses_rg, label="RG-VFM: endpoint geodesic MSE")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("RFM vs RG-VFM-style training on S^2")
    plt.tight_layout()
    plt.savefig("loss_curves_s2.png", dpi=200)

    # -------------------------
    # Samples
    # -------------------------
    print("Sampling from RFM...")
    samples_rfm = sample_from_flow("rfm", n_samples=4000, n_steps=80)
    print("Sampling from RG-VFM-style model...")
    samples_rg = sample_from_flow("rg", n_samples=4000, n_steps=80)
    data_np = data_tensor[:8000].cpu().numpy()

    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    plot_unit_sphere(ax1, alpha=0.15)
    ax1.scatter(data_np[:, 0], data_np[:, 1], data_np[:, 2],
                s=5, alpha=0.7)
    ax1.set_title("Target data on S^2")
    ax1.set_box_aspect([1, 1, 1])

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    plot_unit_sphere(ax2, alpha=0.15)
    ax2.scatter(samples_rfm[:, 0], samples_rfm[:, 1], samples_rfm[:, 2],
                s=5, alpha=0.7)
    ax2.set_title("RFM samples on S^2")
    ax2.set_box_aspect([1, 1, 1])

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    plot_unit_sphere(ax3, alpha=0.15)
    ax3.scatter(samples_rg[:, 0], samples_rg[:, 1], samples_rg[:, 2],
                s=5, alpha=0.7)
    ax3.set_title("RG-VFM-style samples on S^2")
    ax3.set_box_aspect([1, 1, 1])

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    plt.tight_layout()
    plt.savefig("samples_s2.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    # You can increase num_steps if you have more time/compute.
    main(num_steps=2000)
