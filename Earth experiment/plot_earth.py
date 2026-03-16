import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import normalize_torch, extrinsic_to_latlon_rad_torch, extrinsic_to_mollweide_rad_torch

def _prepare_plot_inputs(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    y: torch.Tensor | None = None,
    plot_all_samples: bool = False,
):
    """
    Normalize plot inputs.

    Args
    ----
    x      : (N, 3)
    x_hat  : (N, 3) or (N, K, 3)
    y      : optional (N, 3)
    plot_all_samples :
        If x_hat is (N,K,3):
          - False -> use mean over K, returning (N,3)
          - True  -> flatten to (N*K,3) and repeat x,y accordingly

    Returns
    -------
    x_plot     : (M, 3)
    xhat_plot  : (M, 3)
    y_plot     : None or (M, 3)
    """
    x = normalize_torch(x.detach().cpu())
    x_hat = normalize_torch(x_hat.detach().cpu())
    y_plot = None if y is None else normalize_torch(y.detach().cpu())

    if x_hat.ndim == 2:
        return x, x_hat, y_plot

    if x_hat.ndim != 3:
        raise ValueError(f"x_hat must have shape (N,3) or (N,K,3), got {tuple(x_hat.shape)}")

    N, K, d = x_hat.shape
    if d != 3:
        raise ValueError(f"Expected x_hat last dim = 3, got {d}")

    if not plot_all_samples:
        xhat_plot = normalize_torch(x_hat.mean(dim=1))
        return x, xhat_plot, y_plot

    # plot all K reconstructions
    xhat_plot = x_hat.reshape(N * K, 3)
    x_plot = x[:, None, :].expand(N, K, 3).reshape(N * K, 3)

    if y_plot is not None:
        y_plot = y_plot[:, None, :].expand(N, K, 3).reshape(N * K, 3)

    return x_plot, xhat_plot, y_plot

def _wrap_longitude_jumps_deg(lon_deg: np.ndarray) -> np.ndarray:
    lon_deg = np.asarray(lon_deg, dtype=float)
    out = [lon_deg[0]]
    for j in range(1, len(lon_deg)):
        prev = lon_deg[j - 1]
        curr = lon_deg[j]
        if abs(curr - prev) > 180.0:
            out.append(np.nan)
        out.append(curr)
    return np.asarray(out)


def _latlon_arc_arrays(x: torch.Tensor, y: torch.Tensor, arc_points: int = 100):
    arc = great_circle_interpolation_torch(x, y, n_points=arc_points)  # (B,M,3)
    ll = extrinsic_to_latlon_deg_torch(arc.reshape(-1, 3)).reshape(arc.shape[0], arc.shape[1], 2)
    ll_np = ll.detach().cpu().numpy()

    lat_list, lon_list = [], []
    for i in range(ll_np.shape[0]):
        lat_i = ll_np[i, :, 0]
        lon_i = ll_np[i, :, 1]

        lon_wrapped = _wrap_longitude_jumps_deg(lon_i)

        lat_plot = []
        src_idx = 0
        for val in lon_wrapped:
            if np.isnan(val):
                lat_plot.append(np.nan)
            else:
                lat_plot.append(lat_i[src_idx])
                src_idx += 1

        lat_list.append(np.asarray(lat_plot))
        lon_list.append(lon_wrapped)

    return lat_list, lon_list

def extrinsic_to_latlon_deg_torch(x: torch.Tensor) -> torch.Tensor:
    x = normalize_torch(x)

    xx = x[:, 0]
    yy = x[:, 1]
    zz = x[:, 2]

    lat = -torch.arcsin(zz)
    lon = torch.atan2(yy, xx)

    return torch.stack([lat, lon], dim=-1) * (180.0 / torch.pi)


def plot_earth_latlon(
    x: torch.Tensor,
    title: str = "Earth data (lat/lon)",
    s: float = 6.0,
    alpha: float = 0.5,
    color: str = "tab:blue",
    figsize=(8, 4),
):
    latlon = extrinsic_to_latlon_deg_torch(x).detach().cpu().numpy()
    lat = latlon[:, 0]
    lon = latlon[:, 1]

    plt.figure(figsize=figsize)
    plt.scatter(lon, lat, s=s, alpha=alpha, c=color)
    plt.xlabel("longitude (deg)")
    plt.ylabel("latitude (deg)")
    plt.title(title)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_earth_train_val_test(
    train_loader,
    val_loader=None,
    test_loader=None,
    title: str = "Earth train/val/test split",
    train_color: str = "tab:blue",
    val_color: str = "tab:orange",
    test_color: str = "tab:green",
    s: float = 5.0,
    alpha: float = 0.5,
    figsize=(10, 5.6),
):
    """
    Plot train/val/test points on a Mollweide Earth projection.

    Assumes points from the loaders are already extrinsic coordinates on S^2.
    Uses extrinsic_to_mollweide_rad_torch for plotting coordinates.
    """
    def collect_points(loader):
        if loader is None:
            return None
        xs = []
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            xs.append(x.detach().cpu())
        return torch.cat(xs, dim=0) if xs else None

    x_train = collect_points(train_loader)
    x_val = collect_points(val_loader)
    x_test = collect_points(test_loader)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="mollweide")

    if x_train is not None:
        ll = extrinsic_to_mollweide_rad_torch(normalize_torch(x_train)).numpy()
        lat = ll[:, 0]
        lon = ll[:, 1]
        ax.scatter(lon, lat, s=s, alpha=alpha, c=train_color, label="train")

    if x_val is not None:
        ll = extrinsic_to_mollweide_rad_torch(normalize_torch(x_val)).numpy()
        lat = ll[:, 0]
        lon = ll[:, 1]
        ax.scatter(lon, lat, s=s, alpha=alpha, c=val_color, label="val")

    if x_test is not None:
        ll = extrinsic_to_mollweide_rad_torch(normalize_torch(x_test)).numpy()
        lat = ll[:, 0]
        lon = ll[:, 1]
        ax.scatter(lon, lat, s=s, alpha=alpha, c=test_color, label="test")

    ax.grid(True, alpha=0.25)
    ax.set_xticklabels(
        ["150°W", "120°W", "90°W", "60°W", "30°W", "0°", "30°E", "60°E", "90°E", "120°E", "150°E"]
    )
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_sphere_3d(
    x: torch.Tensor,
    title: str = "Sphere data",
    s: float = 8.0,
    alpha: float = 0.5,
    color: str = "tab:blue",
    figsize=(7, 7),
):
    x_np = x.detach().cpu().numpy()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, 80)
    v = np.linspace(0, np.pi, 40)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, linewidth=0.3, alpha=0.15)

    ax.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2], s=s, alpha=alpha, c=color)

    ax.set_title(title)
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-1.05, 1.05])
    ax.set_zlim([-1.05, 1.05])
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()

 # ============================================================
# great-circle interpolation for visualization
# ============================================================

def great_circle_interpolation_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    n_points: int = 50,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Returns points along the shortest great-circle arc from x to y.

    Args:
        x, y: (B, 3)
        n_points: number of interpolation points along each arc

    Returns:
        arc: (B, n_points, 3)
    """
    x = normalize_torch(x)
    y = normalize_torch(y)

    dot = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.arccos(dot)  # (B,1)

    s = torch.linspace(0.0, 1.0, n_points, device=x.device, dtype=x.dtype)
    s = s.view(1, n_points, 1)  # (1,M,1)

    x_exp = x.unsqueeze(1)      # (B,1,3)
    y_exp = y.unsqueeze(1)      # (B,1,3)
    theta_exp = theta.unsqueeze(1)  # (B,1,1)

    sin_theta = torch.sin(theta_exp).clamp_min(eps)

    # if x and y are extremely close, fall back to linear interpolation + renorm
    close = theta_exp.abs() < 1e-6

    arc_slerp = (
        torch.sin((1.0 - s) * theta_exp) / sin_theta * x_exp
        + torch.sin(s * theta_exp) / sin_theta * y_exp
    )

    arc_lerp = normalize_torch((1.0 - s) * x_exp + s * y_exp)

    arc = torch.where(close, arc_lerp, arc_slerp)
    return normalize_torch(arc)


# ============================================================
# great-circle plotting
# ============================================================

def plot_sphere_3d_with_arcs(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    y: torch.Tensor | None = None,
    plot_all_samples: bool = False,
    n_show: int = 20,
    arc_points: int = 50,
    s_points: float = 20.0,
    alpha_points: float = 0.8,
    alpha_arc: float = 0.6,
    figsize=(8, 8),
    title: str = "Great-circle reconstruction error on S^2",
):
    """
    Plot x, optional y, and x_hat on the sphere.

    x_hat can be:
      - (N,3): single reconstruction per point
      - (N,K,3): K sampled reconstructions per point

    If plot_all_samples=False and x_hat is (N,K,3), plot mean over K.
    If plot_all_samples=True and x_hat is (N,K,3), plot all K samples.
    """
    x_plot, xhat_plot, y_plot = _prepare_plot_inputs(
        x=x, x_hat=x_hat, y=y, plot_all_samples=plot_all_samples
    )

    n = min(n_show, x_plot.shape[0])
    x_plot = x_plot[:n]
    xhat_plot = xhat_plot[:n]
    if y_plot is not None:
        y_plot = y_plot[:n]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, 80)
    v = np.linspace(0, np.pi, 40)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, linewidth=0.3, alpha=0.15)

    x_np = x_plot.numpy()
    xhat_np = xhat_plot.numpy()

    ax.scatter(
        x_np[:, 0], x_np[:, 1], x_np[:, 2],
        s=s_points, alpha=alpha_points, label="x", color="tab:blue"
    )

    if y_plot is not None:
        y_np = y_plot.numpy()
        ax.scatter(
            y_np[:, 0], y_np[:, 1], y_np[:, 2],
            s=s_points, alpha=alpha_points, label="y", color="tab:green"
        )

    ax.scatter(
        xhat_np[:, 0], xhat_np[:, 1], xhat_np[:, 2],
        s=s_points, alpha=alpha_points, label="x_hat", color="tab:orange"
    )

    if y_plot is None:
        arcs = great_circle_interpolation_torch(x_plot, xhat_plot, n_points=arc_points).numpy()
        for i in range(n):
            ax.plot(
                arcs[i, :, 0], arcs[i, :, 1], arcs[i, :, 2],
                alpha=alpha_arc, linewidth=1.2, color="black",
            )
    else:
        arcs_xy = great_circle_interpolation_torch(x_plot, y_plot, n_points=arc_points).numpy()
        arcs_yh = great_circle_interpolation_torch(y_plot, xhat_plot, n_points=arc_points).numpy()

        for i in range(n):
            ax.plot(
                arcs_xy[i, :, 0], arcs_xy[i, :, 1], arcs_xy[i, :, 2],
                alpha=alpha_arc, linewidth=1.2, color="tab:green",
            )
            ax.plot(
                arcs_yh[i, :, 0], arcs_yh[i, :, 1], arcs_yh[i, :, 2],
                alpha=alpha_arc, linewidth=1.2, color="black",
            )

    ax.set_title(title)
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-1.05, 1.05])
    ax.set_zlim([-1.05, 1.05])
    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_latlon_pairs(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    y: torch.Tensor | None = None,
    plot_all_samples: bool = False,
    n_show: int = 50,
    arc_points: int = 100,
    figsize=(9, 4),
    title: str = "Lat/lon comparison",
):
    """
    Plot x, optional y, and x_hat in latitude-longitude coordinates.

    Great-circle arcs are sampled on S^2 and wrapped at the dateline.

    x_hat can be:
      - (N,3)
      - (N,K,3), with plot_all_samples controlling mean-vs-all display.
    """
    x_plot, xhat_plot, y_plot = _prepare_plot_inputs(
        x=x, x_hat=x_hat, y=y, plot_all_samples=plot_all_samples
    )

    n = min(n_show, x_plot.shape[0])
    x_plot = x_plot[:n]
    xhat_plot = xhat_plot[:n]
    if y_plot is not None:
        y_plot = y_plot[:n]

    ll_x = extrinsic_to_latlon_deg_torch(x_plot).numpy()
    ll_hat = extrinsic_to_latlon_deg_torch(xhat_plot).numpy()

    plt.figure(figsize=figsize)

    plt.scatter(ll_x[:, 1], ll_x[:, 0], s=25, alpha=0.8, label="x", color="tab:blue")

    if y_plot is not None:
        ll_y = extrinsic_to_latlon_deg_torch(y_plot).numpy()
        plt.scatter(ll_y[:, 1], ll_y[:, 0], s=25, alpha=0.8, label="y", color="tab:green")

    plt.scatter(ll_hat[:, 1], ll_hat[:, 0], s=25, alpha=0.8, label="x_hat", color="tab:orange")

    if y_plot is None:
        lat_arcs, lon_arcs = _latlon_arc_arrays(x_plot, xhat_plot, arc_points=arc_points)
        for lat_i, lon_i in zip(lat_arcs, lon_arcs):
            plt.plot(lon_i, lat_i, color="black", alpha=0.35, linewidth=0.8)
    else:
        lat_xy, lon_xy = _latlon_arc_arrays(x_plot, y_plot, arc_points=arc_points)
        lat_yh, lon_yh = _latlon_arc_arrays(y_plot, xhat_plot, arc_points=arc_points)

        for lat_i, lon_i in zip(lat_xy, lon_xy):
            plt.plot(lon_i, lat_i, color="tab:green", alpha=0.45, linewidth=0.9)
        for lat_i, lon_i in zip(lat_yh, lon_yh):
            plt.plot(lon_i, lat_i, color="black", alpha=0.35, linewidth=0.8)

    plt.xlabel("longitude (deg)")
    plt.ylabel("latitude (deg)")
    plt.title(title)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_points_on_earth(
    *point_sets,
    labels=None,
    colors=None,
    markers=None,
    sizes=None,
    title="Points on Earth (Mollweide projection)",
    alpha=0.9,):
    """
    Plot one or more sets of points on S^2 using a Mollweide Earth projection.

    Parameters
    ----------
    point_sets : tensors
        Each tensor should have shape (N,3) or (3,)
        representing unit vectors on S^2.

    labels : list[str]
        Legend labels.

    colors : list[str]
        Colors for each point set.

    markers : list[str]
        Matplotlib markers.

    sizes : list[float]
        Marker sizes.

    Example
    -------
    plot_points_on_earth(
        x_true,
        recon_samples,
        labels=["true x", "reconstructions"],
        colors=["red", "blue"],
        markers=["*", "."]
    )
    """

    fig = plt.figure(figsize=(10,5.5))
    ax = fig.add_subplot(111, projection="mollweide")

    for i, pts in enumerate(point_sets):

        pts = torch.as_tensor(pts)

        if pts.ndim == 1:
            pts = pts[None]

        pts = normalize_torch(pts)

        latlon = extrinsic_to_latlon_rad_torch(pts).cpu().numpy()
        lat = -latlon[:,0]
        lon = latlon[:,1]

        label = labels[i] if labels is not None else None
        color = colors[i] if colors is not None else None
        marker = markers[i] if markers is not None else "o"
        size = sizes[i] if sizes is not None else 40

        ax.scatter(
            lon,
            lat,
            s=size,
            color=color,
            marker=marker,
            alpha=alpha,
            label=label,
            edgecolors="black" if marker=="*" else None,
            linewidths=0.5 if marker=="*" else 0,
        )

    ax.grid(True, alpha=0.3)

    ax.set_xticklabels([
        "150°W","120°W","90°W","60°W","30°W",
        "0°",
        "30°E","60°E","90°E","120°E","150°E"
    ])

    ax.set_title(title)

    if labels is not None:
        ax.legend()

    plt.tight_layout()
    plt.show()