import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

try:
    from scipy.stats import gaussian_kde
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


from DPnP import dPnP_sampler_torch_batched
from bel import get_bel
from utils import (
    normalize_torch,
    sample_geodesic_gaussian_on_sphere,
    sphere_distance_torch,
    sample_xtrue_y_batches_from_dataloader,
    spherical_kde_vmf,
    sphere_mean_torch,
    extrinsic_to_latlon_rad_torch,
    extrinsic_to_mollweide_rad_torch,
    mollweide_plot_grid_xyz_torch,
    sample_vmf_s2,
    seismic_signal,
    get_seismic_f_fn,
    get_two_point_seismic_f_fn,
)




# =========================================================
# Aggregate reconstruction gap vs steps
# =========================================================

@torch.no_grad()
def cosine_similarity_vs_steps_for_particle_counts(
    dataloader,
    q_score,
    p_score,
    eta,
    sigma_y,
    particle_counts=(1, 5, 10, 15, 20),
    out_samples=64,
    grw_steps=5,
    num_pairs=64,
    batch_eval_size=16,
    device="mps",
    dtype=torch.float32,
    seed=0,
    show_stderr=True,
):
    """
    Sample x_true and y as before. For each DPnP step and each requested particle
    count N, compute the cosine similarity between x_true and the spherical mean
    of the first N sampled particles. Then overlay the cosine-vs-step curves for
    all requested N values.

    Parameters
    ----------
    particle_counts : iterable of int
        Values of N for which to compute:
            cos( sphere_mean( X_steps[:, :, :N, :] ), x_true )

        Each N must satisfy 1 <= N <= out_samples.

    Returns
    -------
    results : dict
        {
          "steps": np.ndarray shape (S+1,),
          "obs_cos_baseline": float,
          "by_particle_count": {
              N: {
                  "mean_cos": np.ndarray shape (S+1,),
                  "stderr_cos": np.ndarray shape (S+1,),
              },
              ...
          }
        }
    """
    device = torch.device(device)

    particle_counts = sorted(set(int(n) for n in particle_counts))
    if len(particle_counts) == 0:
        raise ValueError("particle_counts must contain at least one positive integer")
    if min(particle_counts) < 1:
        raise ValueError("all particle counts must be >= 1")
    if max(particle_counts) > out_samples:
        raise ValueError(
            f"max requested particle count = {max(particle_counts)} exceeds "
            f"out_samples = {out_samples}"
        )

    x_true_all, y_all = sample_xtrue_y_batches_from_dataloader(
        dataloader=dataloader,
        sigma_y=sigma_y,
        num_pairs=num_pairs,
        device=device,
        dtype=dtype,
        seed=seed,
    )

    # store chunked cosine values separately for each N
    cos_chunks_by_N = {N: [] for N in particle_counts}

    running_seed = int(seed)

    for start in range(0, num_pairs, batch_eval_size):
        end = min(start + batch_eval_size, num_pairs)

        x_batch = x_true_all[start:end]   # (B, d)
        y_batch = y_all[start:end]        # (B, d)

        X_steps = dPnP_sampler_torch_batched(
            q_score=q_score,
            p_score=p_score,
            y=y_batch,
            out_samples=out_samples,
            eta=eta,
            grw_steps=grw_steps,
            seed=running_seed,
            end_only=False,
            device=device,
            dtype=dtype,
        )
        running_seed += 1

        # X_steps: (S+1, B, P, d)
        S_plus_1, B, P, d = X_steps.shape

        for N in particle_counts:
            X_subset = X_steps[:, :, :N, :]                     # (S+1, B, N, d)
            X_subset_mean = sphere_mean_torch(X_subset, dim=2)  # (S+1, B, d)

            cos_subset_mean = (X_subset_mean * x_batch[None, :, :]).sum(dim=-1)  # (S+1, B)
            cos_chunks_by_N[N].append(cos_subset_mean.cpu())

    # concatenate over chunks
    all_cos_by_N = {
        N: torch.cat(cos_chunks_by_N[N], dim=1)   # (S+1, num_pairs)
        for N in particle_counts
    }

    mean_cos_by_N = {
        N: all_cos_by_N[N].mean(dim=1).numpy()
        for N in particle_counts
    }

    stderr_cos_by_N = {
        N: (
            all_cos_by_N[N].std(dim=1, unbiased=False).numpy()
            / np.sqrt(all_cos_by_N[N].shape[1])
        )
        for N in particle_counts
    }

    steps = np.arange(S_plus_1)

    # observation baseline
    obs_cos = (x_true_all * y_all).sum(dim=-1)
    mean_obs_cos = obs_cos.mean().item()

    # shared y-limits
    ymin_cos = min(
        [mean_obs_cos] + [float(mean_cos_by_N[N].min()) for N in particle_counts]
    )
    ymax_cos = max(
        [mean_obs_cos] + [float(mean_cos_by_N[N].max()) for N in particle_counts]
    )
    pad_cos = 0.05 * max(1e-8, ymax_cos - ymin_cos)
    ymin_cos -= pad_cos
    ymax_cos += pad_cos

    # print summary
    print(f"mean_obs_cos = {mean_obs_cos:.6f}")
    for N in particle_counts:
        print(f"N = {N:>3d}, step 0 mean cos = {mean_cos_by_N[N][0]:.6f}")

    # overlay plot
    plt.figure(figsize=(8, 5))

    plt.axhline(
        mean_obs_cos,
        linestyle="--",
        linewidth=2.5,
        color="black",
        alpha=0.9,
        zorder=1,
        label="mean cosine from observation y",
    )

    for N in particle_counts:
        plt.plot(
            steps,
            mean_cos_by_N[N],
            marker="o",
            linewidth=2,
            zorder=3,
            label=f"N = {N}",
        )

        if show_stderr:
            plt.fill_between(
                steps,
                mean_cos_by_N[N] - stderr_cos_by_N[N],
                mean_cos_by_N[N] + stderr_cos_by_N[N],
                alpha=0.12,
                zorder=2,
            )

    plt.xlabel("DPnP step")
    plt.ylabel("mean cosine similarity")
    plt.title(
        "Cosine similarity vs DPnP step\n"
        "spherical mean reconstruction using N particles"
    )
    plt.ylim(ymin_cos, ymax_cos)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "steps": steps,
        "obs_cos_baseline": mean_obs_cos,
        "by_particle_count": {
            N: {
                "mean_cos": mean_cos_by_N[N],
                "stderr_cos": stderr_cos_by_N[N],
            }
            for N in particle_counts
        },
    }
# =========================================================
# KDE plot
# =========================================================

@torch.no_grad()
def plot_spherical_kde_mollweide(
    X_final: torch.Tensor,
    x_true: torch.Tensor | None = None,
    y_obs: torch.Tensor | None = None,
    x_recon_mean: torch.Tensor | None = None,
    b_index: int = 0,
    kappa: float = 25.0,
    n_lon: int = 240,
    n_lat: int = 120,
    xy_marker_size: float = 40,
    recon_marker_size: float = 40,
    sample_marker_size: float = 10,
    y_marker_size: float = 20,
    overlay_samples: bool = True,
    overlay_sample_count: int = 400,
    overlay_y: bool = True,
    overlay_y_count: int | None = None,
    annotate_xy: bool = False,
    annotate_recon_mean: bool = False,
    title: str = "Spherical KDE of final samples (Mollweide projection)",
    levels: int = 20
):
    X_final = X_final.detach().cpu()
    if X_final.ndim == 2:
        X_final = X_final[None, :, :]

    samples = normalize_torch(X_final[b_index])
    device = samples.device
    dtype = samples.dtype

    Lon, Lat, grid_xyz = mollweide_plot_grid_xyz_torch(
        n_lon=n_lon,
        n_lat=n_lat,
        device=device,
        dtype=dtype,
    )

    dens = spherical_kde_vmf(samples, grid_xyz, kappa=kappa, normalize=True)
    dens_np = dens.reshape(Lon.shape).cpu().numpy()

    lon_np = Lon.cpu().numpy()
    lat_np = Lat.cpu().numpy()

    fig = plt.figure(figsize=(10, 5.6))
    ax = fig.add_subplot(111, projection="mollweide")

    cf = ax.contourf(lon_np, lat_np, dens_np, levels=levels, cmap="viridis")
    fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.08, label="spherical KDE")

    if overlay_samples:
        ll_samp = extrinsic_to_mollweide_rad_torch(samples).cpu().numpy()
        lat_s = ll_samp[:, 0]
        lon_s = ll_samp[:, 1]

        if ll_samp.shape[0] > overlay_sample_count:
            idx = np.random.choice(ll_samp.shape[0], overlay_sample_count, replace=False)
            lat_s = lat_s[idx]
            lon_s = lon_s[idx]

        ax.scatter(
            lon_s,
            lat_s,
            s=sample_marker_size,
            color="orange",
            alpha=0.5,
            edgecolors="none",
            zorder=2,
            label="final samples",
        )

    if x_true is not None:
        x_true = x_true.detach().cpu()
        if x_true.ndim == 1:
            x0 = x_true
        elif x_true.ndim == 2:
            x0 = x_true[b_index]
        else:
            raise ValueError(f"x_true must have shape (3,) or (B,3), got {tuple(x_true.shape)}")

        ll_x = extrinsic_to_mollweide_rad_torch(normalize_torch(x0[None, :])).numpy()[0]
        lat_x, lon_x = ll_x[0], ll_x[1]

        ax.scatter(
            [lon_x],
            [lat_x],
            s=xy_marker_size,
            color = 'tab:red',
            edgecolors="black",
            linewidths=1.4,
            zorder=5,
            label="true x",
        )
        if annotate_xy:
            ax.text(
                lon_x + 0.08,
                lat_x + 0.05,
                "x",
                fontsize=12,
                weight="bold",
                color="black",
                zorder=6,
            )

    if x_recon_mean is not None:
        x_recon_mean = x_recon_mean.detach().cpu()
        if x_recon_mean.ndim == 1:
            xm = x_recon_mean
        elif x_recon_mean.ndim == 2:
            xm = x_recon_mean[b_index]
        else:
            raise ValueError(
                f"x_recon_mean must have shape (3,) or (B,3), got {tuple(x_recon_mean.shape)}"
            )

        ll_m = extrinsic_to_mollweide_rad_torch(normalize_torch(xm[None, :])).numpy()[0]
        lat_m, lon_m = ll_m[0], ll_m[1]

        ax.scatter(
            [lon_m],
            [lat_m],
            s=recon_marker_size,
            color="tab:pink",
            marker="o",
            edgecolors="black",
            linewidths=1.4,
            zorder=5,
            label="mean reconstruction",
        )
        if annotate_recon_mean:
            ax.text(
                lon_m + 0.08,
                lat_m + 0.05,
                r"$\hat{x}$",
                fontsize=12,
                weight="bold",
                color="black",
                zorder=6,
            )

    if y_obs is not None and overlay_y:
        y_obs = y_obs.detach().cpu()

        if y_obs.ndim == 1:
            y_plot = y_obs[None, :]
        elif y_obs.ndim == 2:
            if y_obs.shape[-1] != 3:
                raise ValueError(f"y_obs last dimension must be 3, got {tuple(y_obs.shape)}")
            if y_obs.shape[0] == X_final.shape[0]:
                y_plot = y_obs[b_index][None, :]
            else:
                y_plot = y_obs
        elif y_obs.ndim == 3:
            y_plot = y_obs[b_index]
        else:
            raise ValueError(
                "y_obs must have shape (3,), (B,3), (M,3), or (B,M,3), "
                f"got {tuple(y_obs.shape)}"
            )

        y_plot = normalize_torch(y_plot)

        if overlay_y_count is not None and y_plot.shape[0] > overlay_y_count:
            idx = np.random.choice(y_plot.shape[0], overlay_y_count, replace=False)
            y_plot = y_plot[idx]

        ll_y = extrinsic_to_mollweide_rad_torch(y_plot).numpy()
        lat_y = ll_y[:, 0]
        lon_y = ll_y[:, 1]

        ax.scatter(
            lon_y,
            lat_y,
            s=y_marker_size,
            color="tab:green",
            marker="D",
            edgecolors="none",
            alpha=0.4,
            zorder=5,
            label="observed y",
        )

        if annotate_xy and y_plot.shape[0] == 1:
            ax.text(
                lon_y[0] + 0.08,
                lat_y[0] + 0.05,
                "y",
                fontsize=12,
                weight="bold",
                color="black",
                zorder=6,
            )

    ax.grid(True, alpha=0.28)
    ax.set_xticklabels(
        ["150°W", "120°W", "90°W", "60°W", "30°W", "0°", "30°E", "60°E", "90°E", "120°E", "150°E"]
    )
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_multi_xtrue_with_reconstruction_kde(
    recon_means: torch.Tensor,
    x_true_all: torch.Tensor,
    kappa: float = 25.0,
    n_lon: int = 240,
    n_lat: int = 120,
    x_marker_size: float = 40,
    recon_marker_size: float = 70,
    recon_alpha = 0.8,
    title: str = "KDE of averaged DPnP reconstructions over multiple $x_{true}$",
    overlay_recon_points: bool = True,
    overlay_recon_count: int = 400,
    levels :int = 20,
    color:str = 'white'
):
    recon_means = normalize_torch(recon_means.detach().cpu())
    x_true_all = normalize_torch(x_true_all.detach().cpu())

    device = recon_means.device
    dtype = recon_means.dtype

    Lon, Lat, grid_xyz = mollweide_plot_grid_xyz_torch(
        n_lon=n_lon,
        n_lat=n_lat,
        device=device,
        dtype=dtype,
    )

    dens = spherical_kde_vmf(recon_means, grid_xyz, kappa=kappa, normalize=True)
    dens_np = dens.reshape(Lon.shape).cpu().numpy()

    lon_np = Lon.cpu().numpy()
    lat_np = Lat.cpu().numpy()

    fig = plt.figure(figsize=(10.5, 5.8))
    ax = fig.add_subplot(111, projection="mollweide")

    cf = ax.contourf(lon_np, lat_np, dens_np, levels=levels, cmap="viridis")
    fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.08, label="spherical KDE")

    if overlay_recon_points:
        ll_r = extrinsic_to_mollweide_rad_torch(recon_means).numpy()
        lat_r = ll_r[:, 0]
        lon_r = ll_r[:, 1]

        if ll_r.shape[0] > overlay_recon_count:
            idx = np.random.choice(ll_r.shape[0], overlay_recon_count, replace=False)
            lat_r = lat_r[idx]
            lon_r = lon_r[idx]

        ax.scatter(
            lon_r,
            lat_r,
            s=recon_marker_size,
            color=color,
            alpha=recon_alpha,
            edgecolors="none",
            zorder=5,
            label="averaged reconstructions",
        )

    ll_x = extrinsic_to_mollweide_rad_torch(x_true_all).numpy()
    lat_x = ll_x[:, 0]
    lon_x = ll_x[:, 1]

    ax.scatter(
        lon_x,
        lat_x,
        s=x_marker_size,
        color="tab:red",
        #marker="*",
        edgecolors="black",
        linewidths=1.2,
        zorder=2,
        label=r"multiple $x_{\mathrm{true}}$",
    )

    ax.grid(True, alpha=0.28)
    ax.set_xticklabels(
        ["150°W", "120°W", "90°W", "60°W", "30°W", "0°", "30°E", "60°E", "90°E", "120°E", "150°E"]
    )
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_dpnp_reconstruction_kde(
    recon_samples: torch.Tensor,
    x_true: torch.Tensor,
    recon_mean: torch.Tensor,
    kappa: float = 25.0,
    n_lon: int = 240,
    n_lat: int = 120,
    x_marker_size: float = 30,
    recon_mean_marker_size: float = 60,
    recond_marke_a = 0.7,
    recon_marker_size: float = 10,
    y_marker_size: float = 45,
    y_marker_a = 0.7, 
    title: str = "DPnP reconstruction KDE",
    overlay_samples: bool = True,
    overlay_sample_count: int = 400,
    y_obs: torch.Tensor = None,
    overlay_y_obs: bool = False,
    overlay_y_obs_count: int = 200,
    y_mean: torch.Tensor = None,
    levels: int = 20,
):
    recon_samples = normalize_torch(recon_samples.detach().cpu())
    x_true = normalize_torch(x_true.detach().cpu())
    recon_mean = normalize_torch(recon_mean.detach().cpu())

    if y_obs is not None:
        y_obs = normalize_torch(y_obs.detach().cpu())
    if y_mean is not None:
        y_mean = normalize_torch(y_mean.detach().cpu())

    device = recon_samples.device
    dtype = recon_samples.dtype

    Lon, Lat, grid_xyz = mollweide_plot_grid_xyz_torch(
        n_lon=n_lon,
        n_lat=n_lat,
        device=device,
        dtype=dtype,
    )

    dens = spherical_kde_vmf(recon_samples, grid_xyz, kappa=kappa, normalize=True)
    dens_np = dens.reshape(Lon.shape).cpu().numpy()

    lon_np = Lon.cpu().numpy()
    lat_np = Lat.cpu().numpy()

    fig = plt.figure(figsize=(10.5, 5.8))
    ax = fig.add_subplot(111, projection="mollweide")

    cf = ax.contourf(lon_np, lat_np, dens_np, levels=levels, cmap="viridis")
    fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.08, label="spherical KDE")

    if overlay_samples:
        ll_r = extrinsic_to_mollweide_rad_torch(recon_samples).numpy()
        lat_r = ll_r[:, 0]
        lon_r = ll_r[:, 1]

        if ll_r.shape[0] > overlay_sample_count:
            idx = np.random.choice(ll_r.shape[0], overlay_sample_count, replace=False)
            lat_r = lat_r[idx]
            lon_r = lon_r[idx]

        ax.scatter(
            lon_r,
            lat_r,
            s=recon_marker_size,
            color="white",
            alpha=recond_marke_a,
            edgecolors="none",
            zorder=5,
            label="recon samples",
        )

    if overlay_y_obs and y_obs is not None:
        ll_y = extrinsic_to_mollweide_rad_torch(y_obs).numpy()
        lat_y = ll_y[:, 0]
        lon_y = ll_y[:, 1]

        if ll_y.shape[0] > overlay_y_obs_count:
            idx = np.random.choice(ll_y.shape[0], overlay_y_obs_count, replace=False)
            lat_y = lat_y[idx]
            lon_y = lon_y[idx]

        ax.scatter(
            lon_y,
            lat_y,
            s=y_marker_size,
            color="pink",
            alpha=y_marker_a,
            edgecolors="none",
            zorder=6,
            label=r"$y_{\mathrm{obs}}$",
        )


    ll_rm = extrinsic_to_mollweide_rad_torch(recon_mean[None, :]).numpy()
    ax.scatter(
        ll_rm[:, 1],
        ll_rm[:, 0],
        s=recon_mean_marker_size,
        color="gold",
        edgecolors="black",
        linewidths=1.2,
        zorder=9,
        label="recon mean",
    )

    ll_x = extrinsic_to_mollweide_rad_torch(x_true[None, :]).numpy()
    ax.scatter(
        ll_x[:, 1],
        ll_x[:, 0],
        s=x_marker_size,
        color="tab:red",
        edgecolors="black",
        linewidths=1.2,
        zorder=10,
        label=r"$x_{\mathrm{true}}$",
    )

    ax.grid(True, alpha=0.28)
    ax.set_xticklabels(
        ["150°W", "120°W", "90°W", "60°W", "30°W", "0°", "30°E", "60°E", "90°E", "120°E", "150°E"]
    )
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def plot_observation_distribution(
    y_obs: torch.Tensor,
    x_true: torch.Tensor,
    y_mean: torch.Tensor | None = None,
    kappa: float = 25.0,
    show_kde: bool = False,
    show_y_mean: bool = False,
    overlay_sample_count: int = 200,
    title: str = "Observed y samples",
    levels:int = 20
):
    y_obs = normalize_torch(y_obs.detach().cpu())
    x_true = normalize_torch(x_true.detach().cpu())

    if y_mean is None:
        y_mean = sphere_mean_torch(y_obs, dim=0)

    y_mean = normalize_torch(y_mean.detach().cpu())

    if show_kde:
        plot_spherical_kde_mollweide(
            X_final=y_obs,
            x_true=x_true,
            x_recon_mean=y_mean,
            overlay_samples=True,
            overlay_sample_count=overlay_sample_count,
            kappa=kappa,
            title=title,
            levels=levels
        )
    else:
        ll_y = extrinsic_to_mollweide_rad_torch(y_obs).numpy()
        lat = ll_y[:, 0]
        lon = ll_y[:, 1]

        if len(lat) > overlay_sample_count:
            idx = np.random.choice(len(lat), overlay_sample_count, replace=False)
            lat = lat[idx]
            lon = lon[idx]

        ll_x = extrinsic_to_mollweide_rad_torch(x_true[None]).numpy()[0]
        ll_m = extrinsic_to_mollweide_rad_torch(y_mean[None]).numpy()[0]

        fig = plt.figure(figsize=(10, 5.6))
        ax = fig.add_subplot(111, projection="mollweide")

        ax.scatter(
            lon,
            lat,
            s=40,
            color="tab:green",
            edgecolors="black",
            linewidths=0.8,
            alpha=0.6,
            label="observed y",
        )

        ax.scatter(
            [ll_x[1]],
            [ll_x[0]],
            s=40,
            color="tab:red",
            edgecolors="black",
            linewidths=1.3,
            label="true x",
        )

        if show_y_mean:
            ax.scatter(
                [ll_m[1]],
                [ll_m[0]],
                s=150,
                color="tab:blue",
                marker="o",
                edgecolors="black",
                linewidths=1.3,
                label="mean y",
            )

        ax.set_title(title)
        ax.grid(True, alpha=0.28)
        ax.set_xticklabels(
            ["150°W", "120°W", "90°W", "60°W", "30°W", "0°", "30°E", "60°E", "90°E", "120°E", "150°E"]
        )
        ax.legend()
        plt.tight_layout()
        plt.show()

@torch.no_grad()
def test_dpnp_sampler_multiple_xtrue(
    dataloader,
    q_score,
    p_score,
    eta,
    sigma_y,
    num_xtrue=4,
    num_y_per_xtrue=32,
    out_samples=64,
    grw_steps=5,
    device="mps",
    dtype=torch.float32,
    seed=0,
    kappa=25.0,
    show_y_kde=False,
    overlay_recon_sample_count=400,
    overlay_y_sample_count=200,
    plot_results=True,
    make_aggregate_plot=True,
    levels: int = 20,
    color: str = "white",
):
    """
    For each chosen x_true:
      1. sample x_true from dataloader
      2. sample many y's centered at x_true from the vMF / spherical observation model
      3. run DPnP from those y's
      4. make:
           - DPnP KDE using particle-mean reconstructions
           - DPnP KDE using all pooled final particles
           - observed y distribution

    Finally, make an aggregate plot showing:
      - all selected x_true points
      - all per-y averaged reconstructions pooled together
      - KDE of those pooled per-y averaged reconstructions

    This version uses cosine similarity instead of reconstruction gap.
    """
    device = torch.device(device)

    x_true_all, _ = sample_xtrue_y_batches_from_dataloader(
        dataloader=dataloader,
        sigma_y=sigma_y,
        num_pairs=num_xtrue,
        device=device,
        dtype=dtype,
        seed=seed,
    )

    x_true_all = normalize_torch(x_true_all)

    results = []
    aggregate_mean_recons = []

    for i in range(num_xtrue):
        x_true = normalize_torch(x_true_all[i])

        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed + 1000 + i))

        x_rep = x_true[None, :].expand(num_y_per_xtrue, 3)
        y_batch = sample_vmf_s2(
            mu=x_rep,
            kappa=sigma_y,
            generator=gen,
        )
        y_batch = normalize_torch(y_batch)

        X_out = dPnP_sampler_torch_batched(
            q_score=q_score,
            p_score=p_score,
            y=y_batch,
            out_samples=out_samples,
            eta=eta,
            grw_steps=grw_steps,
            seed=int(seed + 2000 + i),
            end_only=True,
            device=device,
            dtype=dtype,
        )

        if X_out.ndim == 4:
            X_final = X_out[-1]
        elif X_out.ndim == 3:
            X_final = X_out
        else:
            raise ValueError(f"Unexpected DPnP output shape: {tuple(X_out.shape)}")

        X_final = normalize_torch(X_final)

        recon_mean_per_y = sphere_mean_torch(X_final, dim=1)
        recon_mean_per_y = normalize_torch(recon_mean_per_y)
        aggregate_mean_recons.append(recon_mean_per_y.detach().cpu())

        y_global_mean = sphere_mean_torch(y_batch, dim=0)
        y_global_mean = normalize_torch(y_global_mean)

        # Observation cosine similarities
        obs_cos = (y_batch * x_true[None, :]).sum(dim=-1)
        mean_obs_cos = obs_cos.mean().item()
        std_obs_cos = obs_cos.std(unbiased=False).item()

        per_xtrue_result = {
            "x_true": x_true.detach().cpu(),
            "y_batch": y_batch.detach().cpu(),
            "X_final": X_final.detach().cpu(),
            "recon_mean_per_y": recon_mean_per_y.detach().cpu(),
            "y_global_mean": y_global_mean.detach().cpu(),
            "mean_obs_cos": mean_obs_cos,
            "std_obs_cos": std_obs_cos,
            "particle_mean": {},
            "all_particles": {},
        }

        for use_particle_mean in [True, False]:
            if use_particle_mean:
                recon_samples = recon_mean_per_y
                recon_global_mean = sphere_mean_torch(recon_mean_per_y, dim=0)
                recon_global_mean = normalize_torch(recon_global_mean)

                recon_cos = (recon_mean_per_y * x_true[None, :]).sum(dim=-1)

                recon_mode_label = "particle-mean reconstructions"
                result_key = "particle_mean"
            else:
                recon_samples = X_final.reshape(-1, 3)
                recon_samples = normalize_torch(recon_samples)

                recon_global_mean = sphere_mean_torch(recon_samples, dim=0)
                recon_global_mean = normalize_torch(recon_global_mean)

                recon_cos = (recon_samples * x_true[None, :]).sum(dim=-1)

                recon_mode_label = "all pooled final particles"
                result_key = "all_particles"

            mean_recon_cos = recon_cos.mean().item()
            std_recon_cos = recon_cos.std(unbiased=False).item()

            if plot_results:
                plot_dpnp_reconstruction_kde(
                    recon_samples=recon_samples,
                    x_true=x_true,
                    recon_mean=recon_global_mean,
                    kappa=kappa,
                    overlay_samples=True,
                    overlay_sample_count=overlay_recon_sample_count,
                    title=(
                        f"x_true #{i+1}: DPnP {recon_mode_label}\n"
                        f"mean cosine similarity = {mean_recon_cos:.4f}, "
                        f"std = {std_recon_cos:.4f}"
                    ),
                    levels=levels,
                )

            per_xtrue_result[result_key] = {
                "recon_samples": recon_samples.detach().cpu(),
                "recon_global_mean": recon_global_mean.detach().cpu(),
                "mean_recon_cos": mean_recon_cos,
                "std_recon_cos": std_recon_cos,
            }

        if plot_results:
            plot_observation_distribution(
                y_obs=y_batch,
                x_true=x_true,
                y_mean=y_global_mean,
                kappa=kappa,
                show_kde=show_y_kde,
                overlay_sample_count=overlay_y_sample_count,
                title=(
                    f"x_true #{i+1}: observed y\n"
                    f"mean cosine similarity = {mean_obs_cos:.4f}, "
                    f"std = {std_obs_cos:.4f}"
                ),
            )

        results.append(per_xtrue_result)

    if make_aggregate_plot:
        aggregate_mean_recons = torch.cat(aggregate_mean_recons, dim=0)

        plot_multi_xtrue_with_reconstruction_kde(
            recon_means=aggregate_mean_recons,
            x_true_all=x_true_all.detach().cpu(),
            kappa=kappa,
            title=(
                "All selected $x_{true}$ overlaid with KDE of pooled "
                "per-$y$ averaged reconstructions"
            ),
            overlay_recon_points=True,
            overlay_recon_count=min(400, aggregate_mean_recons.shape[0]),
            color=color,
        )

    return results


@torch.no_grad()
def make_aggregate_plot_from_results(
    results,
    mode="x_mean",           # "x_mean", "x_all", "x_global_mean", or "y"
    kappa=25.0,
    title=None,
    overlay_points=True,
    overlay_count=400,
    levels=20,
    color="white",
    max_kde_samples=5000,    # automatic subsampling cap for x_all
    seed=0,
    x_size = 20,
    recon_size = 20,
    recon_alpha = 0.6
):
    """
    Aggregate KDE plot from test_dpnp_sampler_multiple_xtrue results.

    mode="x_mean"        -> pooled per-y averaged reconstructions
                           (recon_mean_per_y across all x_true and y)
    mode="x_all"         -> pooled non-averaged final reconstruction particles
                           (all particles across all x_true and y; automatically
                           subsampled for KDE if too large)
    mode="x_global_mean" -> one aggregate mean hat{x} for each x_true, where
                           hat{x}_i = sphere_mean(recon_mean_per_y for that x_true)
    mode="y"             -> pooled sampled observations
    """

    if len(results) == 0:
        raise ValueError("results is empty")

    x_true_all = torch.stack([r["x_true"] for r in results], dim=0)

    if mode == "x_mean":
        samples = torch.cat(
            [r["recon_mean_per_y"] for r in results],
            dim=0,
        )
        default_title = (
            "All selected $x_{true}$ overlaid with KDE of pooled "
            "per-$y$ averaged reconstructions"
        )

    elif mode == "x_all":
        samples = torch.cat(
            [r["all_particles"]["recon_samples"] for r in results],
            dim=0,
        )
        default_title = (
            "All selected $x_{true}$ overlaid with KDE of pooled "
            "non-averaged final reconstruction particles"
        )

    elif mode == "x_global_mean":
        samples = torch.stack(
            [
                normalize_torch(sphere_mean_torch(r["recon_mean_per_y"], dim=0))
                for r in results
            ],
            dim=0,
        )
        default_title = (
            "All selected $x_{true}$ overlaid with KDE of aggregate mean "
            "reconstruction for each $x_{true}$"
        )

    elif mode == "y":
        samples = torch.cat(
            [r["y_batch"] for r in results],
            dim=0,
        )
        default_title = (
            "All selected $x_{true}$ overlaid with KDE of sampled $y$ values"
        )

    else:
        raise ValueError('mode must be "x_mean", "x_all", "x_global_mean", or "y"')

    # -------------------------------------------------------
    # automatic KDE subsampling ONLY for x_all
    # -------------------------------------------------------
    samples_for_kde = samples
    subsampled = False

    if mode == "x_all" and samples.shape[0] > max_kde_samples:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        idx = torch.randperm(samples.shape[0], generator=gen)[:max_kde_samples]
        samples_for_kde = samples[idx]
        subsampled = True

    if title is None:
        title = default_title
        if subsampled:
            title += (
                f"\n(KDE built from random subset of "
                f"{samples_for_kde.shape[0]} particles)"
            )

    plot_multi_xtrue_with_reconstruction_kde(
        recon_means=samples_for_kde,
        x_true_all=x_true_all,
        kappa=kappa,
        title=title,
        overlay_recon_points=overlay_points,
        overlay_recon_count=min(overlay_count, samples.shape[0]),
        levels=levels,
        color=color,
        x_marker_size= x_size,
        recon_marker_size= recon_size,
        recon_alpha=recon_alpha
    )
@torch.no_grad()
def replot_dpnp_results_for_xtrue(
    results,
    xtrue_idx,
    kappa=25.0,
    overlay_recon_sample_count=400,
    overlay_y_sample_count=200,
    levels=20,
):
    """
    Re-make two per-x_true KDE plots from the stored output of
    test_dpnp_sampler_multiple_xtrue(...), with y_obs overlaid on top
    of each reconstruction KDE plot.

    Plot 1:
        KDE of particle-mean reconstructions + overlaid y_obs
    Plot 2:
        KDE of all pooled final particles + overlaid y_obs
    """
    if not (0 <= xtrue_idx < len(results)):
        raise IndexError(
            f"xtrue_idx={xtrue_idx} out of range for results of length {len(results)}"
        )

    r = results[xtrue_idx]

    x_true = r["x_true"]
    y_batch = r["y_batch"]
    y_global_mean = r["y_global_mean"]

    mean_obs_cos = r["mean_obs_cos"]
    std_obs_cos = r["std_obs_cos"]

    # ---------------------------
    # 1) particle-mean KDE plot
    # ---------------------------
    recon_samples_pm = r["particle_mean"]["recon_samples"]
    recon_global_mean_pm = r["particle_mean"]["recon_global_mean"]
    mean_recon_cos_pm = r["particle_mean"]["mean_recon_cos"]
    std_recon_cos_pm = r["particle_mean"]["std_recon_cos"]

    plot_dpnp_reconstruction_kde(
        recon_samples=recon_samples_pm,
        x_true=x_true,
        recon_mean=recon_global_mean_pm,
        kappa=kappa,
        overlay_samples=True,
        overlay_sample_count=overlay_recon_sample_count,
        y_obs=y_batch,
        overlay_y_obs=True,
        overlay_y_obs_count=overlay_y_sample_count,
        y_mean=y_global_mean,
        recon_marker_size= 20,
        y_marker_size=20,
        title=(
            f"x_true #{xtrue_idx+1}: DPnP particle-mean reconstructions + observed y\n"
            f"recon mean cos = {mean_recon_cos_pm:.4f}, recon std = {std_recon_cos_pm:.4f}\n"
            f"obs mean cos = {mean_obs_cos:.4f}, obs std = {std_obs_cos:.4f}"
        ),
        levels=levels,
    )

    # ---------------------------
    # 2) all-particles KDE plot
    # ---------------------------
    recon_samples_all = r["all_particles"]["recon_samples"]
    recon_global_mean_all = r["all_particles"]["recon_global_mean"]
    mean_recon_cos_all = r["all_particles"]["mean_recon_cos"]
    std_recon_cos_all = r["all_particles"]["std_recon_cos"]

    plot_dpnp_reconstruction_kde(
        recon_samples=recon_samples_all,
        x_true=x_true,
        recon_mean=recon_global_mean_all,
        kappa=kappa,
        overlay_samples=True,
        overlay_sample_count=overlay_recon_sample_count,
        y_obs=y_batch,
        overlay_y_obs=True,
        overlay_y_obs_count=overlay_y_sample_count,
        y_mean=y_global_mean,
        recon_marker_size= 10,
        y_marker_size=20,
        title=(
            f"x_true #{xtrue_idx+1}: DPnP all pooled final particles + observed y\n"
            f"recon mean cos = {mean_recon_cos_all:.4f}, recon std = {std_recon_cos_all:.4f}\n"
            f"obs mean cos = {mean_obs_cos:.4f}, obs std = {std_obs_cos:.4f}"
        ),
        levels=levels,
    )


# =========================================================
# Two-point seismic inverse problem – cosine vs steps
# =========================================================

def _reconstruction_score(x_recon: torch.Tensor, x_true: torch.Tensor, metric: str) -> torch.Tensor:
    """
    Compute per-sample reconstruction quality.

    metric="cosine"  : dot product <x_recon, x_true> in [-1, 1], higher is better.
    metric="geodesic": great-circle distance in degrees in [0, 180], lower is better.
    """
    dot = (x_recon * x_true).sum(dim=-1).clamp(-1.0, 1.0)
    if metric == "cosine":
        return dot
    elif metric == "geodesic":
        return torch.rad2deg(torch.arccos(dot))
    else:
        raise ValueError(f"metric must be 'cosine' or 'geodesic', got '{metric}'")


@torch.no_grad()
def cosine_similarity_vs_steps_two_point(
    dataloader,
    p_score,
    eta,
    y1=None,
    y2=None,
    beta: float = 10.0,
    sigma2: float = 0.05,
    kappa_sensors: float | None = None,
    metric: str = "cosine",
    n_bel_paths: int = 5000,
    n_bel_steps: int = 5,
    particle_counts=(1, 5, 10, 20),
    out_samples: int = 64,
    grw_steps: int = 5,
    num_pairs: int = 64,
    batch_eval_size: int = 8,
    device="mps",
    dtype=torch.float32,
    seed: int = 0,
    show_stderr: bool = True,
):
    """
    Two-point seismic analog of cosine_similarity_vs_steps_for_particle_counts.

    For each x_true sampled from the dataloader, draws scalar observations
    (u1, u2) from the seismic model and runs DPnP with the joint two-sensor
    likelihood as q_score, recording reconstruction quality at every DPnP step.

        I(x, y) = exp(beta * (<x, y> - 1))
        u1 ~ N(I(x_true, y1), sigma2),  u2 ~ N(I(x_true, y2), sigma2)

    Parameters
    ----------
    dataloader    : DataLoader  -- source of x_true samples
    p_score       : callable    -- trained prior score p_score(x, t)
    eta           : 1-D array   -- DPnP time schedule
    y1, y2        : (3,) or None -- fixed sensor locations on S^2.
                    Required when kappa_sensors is None.
    beta          : float       -- seismic signal decay rate
    sigma2        : float       -- observation noise variance
    kappa_sensors : float or None
                    If given, sample y1_i, y2_i ~ vMF(x_true_i, kappa_sensors)
                    independently for each x_true instead of using fixed y1, y2.
                    Larger kappa = sensors closer to x_true.
    metric        : "cosine" (default) or "geodesic"
                    "cosine"   -- dot product in [-1,1], higher is better.
                    "geodesic" -- great-circle distance in degrees, lower is better.
    n_bel_paths, n_bel_steps : BEL hyper-parameters
    particle_counts : N values to evaluate
    out_samples   : total particles per DPnP call (>= max(particle_counts))
    grw_steps     : SDE integration steps per DPnP step
    num_pairs     : number of (x_true, u1, u2) triples to average over
    batch_eval_size : how many triples to batch into one DPnP call
    seed          : random seed

    Returns
    -------
    dict with keys:
        "steps", "metric",
        "by_particle_count": {N: {"mean_score", "stderr_score"}}
    """
    if kappa_sensors is None and (y1 is None or y2 is None):
        raise ValueError("Provide y1 and y2, or set kappa_sensors to sample them per x_true.")
    if metric not in {"cosine", "geodesic"}:
        raise ValueError("metric must be 'cosine' or 'geodesic'")

    device = torch.device(device)
    particle_counts = sorted(set(int(n) for n in particle_counts))
    if min(particle_counts) < 1:
        raise ValueError("all particle_counts must be >= 1")
    if max(particle_counts) > out_samples:
        raise ValueError(f"max particle count {max(particle_counts)} > out_samples {out_samples}")

    # Fixed sensors (only used when kappa_sensors is None)
    if y1 is not None:
        y1 = normalize_torch(torch.as_tensor(y1, device=device, dtype=dtype))
    if y2 is not None:
        y2 = normalize_torch(torch.as_tensor(y2, device=device, dtype=dtype))

    sigma = math.sqrt(sigma2)

    # Sample x_true; ignore the vMF y returned by helper
    x_true_all, _ = sample_xtrue_y_batches_from_dataloader(
        dataloader=dataloader,
        sigma_y=1.0,
        num_pairs=num_pairs,
        device=device,
        dtype=dtype,
        seed=seed,
    )

    score_chunks_by_N = {N: [] for N in particle_counts}
    running_seed = seed

    for start in range(0, num_pairs, batch_eval_size):
        end = min(start + batch_eval_size, num_pairs)
        x_batch = x_true_all[start:end]          # (B, 3)
        B = x_batch.shape[0]

        gen = torch.Generator(device=device)
        gen.manual_seed(running_seed)

        # ---- sensor locations for this mini-batch ----
        if kappa_sensors is not None:
            # Sample y1_b, y2_b per x_true ~ vMF(x_true, kappa_sensors)
            y1_b = sample_vmf_s2(mu=x_batch, kappa=kappa_sensors, generator=gen)  # (B, 3)
            y2_b = sample_vmf_s2(mu=x_batch, kappa=kappa_sensors, generator=gen)  # (B, 3)
        else:
            y1_b = y1   # (3,) broadcast for all items in batch
            y2_b = y2

        # ---- seismic signals & observations ----
        I1_true = seismic_signal(x_batch, y1_b, beta)   # (B,)
        I2_true = seismic_signal(x_batch, y2_b, beta)

        u1_batch = I1_true + sigma * torch.randn(B, device=device, dtype=dtype, generator=gen)
        u2_batch = I2_true + sigma * torch.randn(B, device=device, dtype=dtype, generator=gen)

        # ---- DPnP with batched f_fn ----
        f_fn = get_two_point_seismic_f_fn(y1_b, y2_b, u1_batch, u2_batch, beta, sigma2)
        q_score = get_bel(f_fn=f_fn, n_paths=n_bel_paths, n_steps=n_bel_steps,
                          device=device, dtype=dtype)

        # Dummy y sets B; use y1_b (per-item) or broadcast y1
        dummy_y = y1_b if kappa_sensors is not None else y1.unsqueeze(0).expand(B, 3)

        X_steps = dPnP_sampler_torch_batched(
            q_score=q_score,
            p_score=p_score,
            y=dummy_y,
            out_samples=out_samples,
            eta=eta,
            grw_steps=grw_steps,
            seed=running_seed,
            end_only=False,
            device=device,
            dtype=dtype,
        )   # (S+1, B, P, 3)
        running_seed += 1

        S_plus_1 = X_steps.shape[0]

        for N in particle_counts:
            X_sub = X_steps[:, :, :N, :]                       # (S+1, B, N, 3)
            X_sub_mean = sphere_mean_torch(X_sub, dim=2)        # (S+1, B, 3)
            sc = _reconstruction_score(X_sub_mean, x_batch[None], metric)  # (S+1, B)
            score_chunks_by_N[N].append(sc.cpu())

    all_scores_by_N = {
        N: torch.cat(score_chunks_by_N[N], dim=1)   # (S+1, num_pairs)
        for N in particle_counts
    }
    mean_score_by_N  = {N: all_scores_by_N[N].mean(dim=1).numpy()  for N in particle_counts}
    stderr_score_by_N = {
        N: (all_scores_by_N[N].std(dim=1, unbiased=False).numpy()
            / np.sqrt(all_scores_by_N[N].shape[1]))
        for N in particle_counts
    }

    steps = np.arange(S_plus_1)
    ylabel = "mean cosine similarity" if metric == "cosine" else "mean geodesic distance (°)"
    better = "↑ better" if metric == "cosine" else "↓ better"

    # ---- summary print ----
    for N in particle_counts:
        final = mean_score_by_N[N][-1]
        print(f"N = {N:>3d} | final step mean {metric} = {final:.4f}")

    # ---- plot ----
    ymin = min(float(v.min()) for v in mean_score_by_N.values())
    ymax = max(float(v.max()) for v in mean_score_by_N.values())
    pad = 0.05 * max(1e-8, ymax - ymin)

    sensor_desc = (f"vMF sensors kappa={kappa_sensors}" if kappa_sensors is not None
                   else "fixed sensors")
    plt.figure(figsize=(8, 5))
    for N in particle_counts:
        plt.plot(steps, mean_score_by_N[N], marker="o", linewidth=2, label=f"N = {N}")
        if show_stderr:
            plt.fill_between(
                steps,
                mean_score_by_N[N] - stderr_score_by_N[N],
                mean_score_by_N[N] + stderr_score_by_N[N],
                alpha=0.12,
            )

    plt.xlabel("DPnP step")
    plt.ylabel(f"{ylabel}  ({better})")
    plt.title(
        f"Two-point seismic: {ylabel} vs DPnP step\n"
        f"{sensor_desc}, spherical mean over N particles"
    )
    plt.ylim(ymin - pad, ymax + pad)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "steps": steps,
        "metric": metric,
        "by_particle_count": {
            N: {"mean_score": mean_score_by_N[N], "stderr_score": stderr_score_by_N[N]}
            for N in particle_counts
        },
    }


# =========================================================
# Two-point seismic demo – single x_true, multiple u samples
# =========================================================

@torch.no_grad()
def run_two_point_earthquake_demo(
    x_true,
    y1=None,
    y2=None,
    beta: float = 10.0,
    sigma2: float = 0.05,
    p_score=None,
    eta=None,
    n_trials: int = 20,
    kappa_sensors: float | None = None,
    metric: str = "cosine",
    n_bel_paths: int = 5000,
    n_bel_steps: int = 5,
    out_samples: int = 32,
    grw_steps: int = 5,
    particle_counts=(1, 5, 10, 20),
    kde_kappa: float = 25.0,
    device="mps",
    dtype=torch.float32,
    seed: int = 0,
    show_stderr: bool = True,
):
    """
    Two-point seismic earthquake reconstruction demo.

    For a known earthquake location x_true and two sensor positions y1, y2,
    repeatedly samples scalar measurements

        u1_k ~ N(I(x_true, y1), sigma2)
        u2_k ~ N(I(x_true, y2), sigma2)
        I(x, y) = exp(beta * (<x, y> - 1))

    for k = 1 … n_trials and runs DPnP with the joint two-sensor likelihood as
    q_score.

    Displays:
      1. Mollweide KDE of per-trial averaged reconstructions (with x_true and sensors).
      2. Reconstruction quality vs DPnP step for different particle counts N.

    Parameters
    ----------
    x_true        : (3,) tensor -- true earthquake location on S^2
    y1, y2        : (3,) tensors or None -- sensor locations on S^2.
                    Required when kappa_sensors is None.
    beta          : float -- seismic signal decay rate (beta > 0)
    sigma2        : float -- observation noise variance
    p_score       : callable -- trained prior score p_score(x, t)
    eta           : 1-D array -- DPnP time schedule
    n_trials      : int -- K, number of (u1, u2) realisations to average
    kappa_sensors : float or None
                    If given, sample y1, y2 ~ vMF(x_true, kappa_sensors) once
                    instead of using provided y1, y2.
                    Larger kappa = sensors closer to x_true.
    metric        : "cosine" (default) or "geodesic"
                    "cosine"   -- dot product, higher is better.
                    "geodesic" -- great-circle distance in degrees, lower is better.
    n_bel_paths, n_bel_steps : BEL hyper-parameters
    out_samples   : DPnP particles P per trial
    grw_steps     : SDE integration steps per DPnP step
    particle_counts : N values for the quality-vs-step plot
    kde_kappa     : vMF bandwidth for the Mollweide KDE
    seed          : base random seed

    Returns
    -------
    dict with keys:
        x_true, y1, y2,
        u1_all (K,), u2_all (K,),
        X_finals (K, P, 3)      -- final DPnP particles per trial
        trial_means (K, 3)      -- per-trial spherical mean reconstruction
        overall_mean (3,)       -- spherical mean over all trial means
        score_trials (K,)       -- per-trial metric value for the mean reconstruction
        score_steps_by_N        -- {N: (S+1, K) ndarray}
        metric                  -- the metric string used
    """
    if p_score is None or eta is None:
        raise ValueError("p_score and eta are required.")
    if kappa_sensors is None and (y1 is None or y2 is None):
        raise ValueError("Provide y1 and y2, or set kappa_sensors to sample them from vMF.")
    if metric not in {"cosine", "geodesic"}:
        raise ValueError("metric must be 'cosine' or 'geodesic'")

    device = torch.device(device)
    eta_t = torch.as_tensor(eta, device=device, dtype=dtype)
    S = int(eta_t.numel())
    particle_counts = sorted(set(int(n) for n in particle_counts))
    if max(particle_counts) > out_samples:
        raise ValueError(f"max particle count {max(particle_counts)} > out_samples {out_samples}")

    x_true = normalize_torch(torch.as_tensor(x_true, device=device, dtype=dtype))
    sigma = math.sqrt(sigma2)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # ---- 1. Sensor locations ----
    if kappa_sensors is not None:
        y1 = sample_vmf_s2(mu=x_true, kappa=kappa_sensors, generator=gen)  # (3,)
        y2 = sample_vmf_s2(mu=x_true, kappa=kappa_sensors, generator=gen)  # (3,)
        print(f"Sampled sensors from vMF(x_true, kappa={kappa_sensors})")
    else:
        y1 = normalize_torch(torch.as_tensor(y1, device=device, dtype=dtype))
        y2 = normalize_torch(torch.as_tensor(y2, device=device, dtype=dtype))

    # ---- 2. Compute true signals ----
    I1_true = seismic_signal(x_true, y1, beta).item()
    I2_true = seismic_signal(x_true, y2, beta).item()
    print(f"True signals:  I(x, y1) = {I1_true:.4f},  I(x, y2) = {I2_true:.4f}")

    # ---- 3. Sample K observations ----
    u1_all = I1_true + sigma * torch.randn(n_trials, device=device, dtype=dtype, generator=gen)
    u2_all = I2_true + sigma * torch.randn(n_trials, device=device, dtype=dtype, generator=gen)
    print(f"Sampled {n_trials} (u1, u2) pairs.  "
          f"u1 mean={u1_all.mean():.4f},  u2 mean={u2_all.mean():.4f}")

    # ---- 4. Run DPnP for all K trials in one batched call ----
    f_fn = get_two_point_seismic_f_fn(y1, y2, u1_all, u2_all, beta, sigma2)
    q_score = get_bel(f_fn=f_fn, n_paths=n_bel_paths, n_steps=n_bel_steps,
                      device=device, dtype=dtype)

    dummy_y = y1.unsqueeze(0).expand(n_trials, 3)   # (K, 3) – sets B = K

    X_steps = dPnP_sampler_torch_batched(
        q_score=q_score,
        p_score=p_score,
        y=dummy_y,
        out_samples=out_samples,
        eta=eta_t,
        grw_steps=grw_steps,
        seed=seed,
        end_only=False,
        device=device,
        dtype=dtype,
    )   # (S+1, K, P, 3)

    X_finals    = X_steps[-1]                                   # (K, P, 3)
    trial_means = sphere_mean_torch(X_finals, dim=1)             # (K, 3)
    overall_mean = sphere_mean_torch(trial_means, dim=0)         # (3,)

    # ---- 5. Metric for final reconstructions ----
    score_trials = _reconstruction_score(trial_means, x_true.unsqueeze(0).expand_as(trial_means), metric)  # (K,)
    score_overall = _reconstruction_score(overall_mean.unsqueeze(0), x_true.unsqueeze(0), metric).item()

    ylabel = "cosine similarity" if metric == "cosine" else "geodesic distance (°)"
    better = "(↑ better)" if metric == "cosine" else "(↓ better)"
    print(f"Trial {ylabel}: mean={score_trials.mean():.4f}  std={score_trials.std():.4f}  {better}")
    print(f"Overall mean {ylabel}: {score_overall:.4f}")

    # ---- 6. Per-step metric for each N and trial ----
    score_steps_by_N = {}
    for N in particle_counts:
        X_sub      = X_steps[:, :, :N, :]                             # (S+1, K, N, 3)
        X_sub_mean = sphere_mean_torch(X_sub, dim=2)                   # (S+1, K, 3)
        sc = _reconstruction_score(
            X_sub_mean,
            x_true.unsqueeze(0).unsqueeze(0).expand_as(X_sub_mean),
            metric,
        )   # (S+1, K)
        score_steps_by_N[N] = sc.cpu()

    # ---- 7. Plot: Mollweide KDE ----
    Lon, Lat, grid_xyz = mollweide_plot_grid_xyz_torch(
        n_lon=240, n_lat=120, device=trial_means.device, dtype=dtype
    )
    dens = spherical_kde_vmf(trial_means, grid_xyz, kappa=kde_kappa, normalize=True)
    dens_np = dens.reshape(Lon.shape).cpu().numpy()

    sensor_desc = (f"vMF sensors (kappa={kappa_sensors})" if kappa_sensors is not None
                   else "fixed sensors")
    fig, ax = plt.subplots(1, 1, figsize=(11, 5.5), subplot_kw={"projection": "mollweide"})
    cf = ax.contourf(Lon.cpu().numpy(), Lat.cpu().numpy(), dens_np, levels=20, cmap="viridis")
    fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.08, label="spherical KDE")

    ll_tm = extrinsic_to_mollweide_rad_torch(trial_means.cpu()).numpy()
    ax.scatter(ll_tm[:, 1], ll_tm[:, 0], s=18, color="white", alpha=0.65,
               edgecolors="none", zorder=5, label="trial means")

    ll_om = extrinsic_to_mollweide_rad_torch(overall_mean.cpu()[None]).numpy()[0]
    ax.scatter([ll_om[1]], [ll_om[0]], s=120, color="gold", marker="*",
               edgecolors="black", linewidths=1.2, zorder=8, label="overall mean")

    ll_x = extrinsic_to_mollweide_rad_torch(x_true.cpu()[None]).numpy()[0]
    ax.scatter([ll_x[1]], [ll_x[0]], s=90, color="tab:red",
               edgecolors="black", linewidths=1.2, zorder=9, label=r"$x_{\rm true}$")

    for yi, lbl, col in [(y1, r"$y_1$", "tab:cyan"), (y2, r"$y_2$", "tab:orange")]:
        ll_yi = extrinsic_to_mollweide_rad_torch(yi.cpu()[None]).numpy()[0]
        ax.scatter([ll_yi[1]], [ll_yi[0]], s=80, color=col, marker="D",
                   edgecolors="black", linewidths=1.0, zorder=7, label=lbl)

    ax.grid(True, alpha=0.28)
    ax.set_xticklabels(
        ["150°W","120°W","90°W","60°W","30°W","0°","30°E","60°E","90°E","120°E","150°E"]
    )
    ax.set_title(
        f"Two-point seismic reconstruction — {sensor_desc}\n"
        f"n_trials={n_trials}, beta={beta}, sigma2={sigma2}"
    )
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()

    # ---- 8. Plot: quality vs DPnP steps ----
    steps_arr = np.arange(S + 1)
    mean_score_by_N = {N: score_steps_by_N[N].mean(dim=1).numpy() for N in particle_counts}
    stderr_score_by_N = {
        N: score_steps_by_N[N].std(dim=1, unbiased=False).numpy() / np.sqrt(n_trials)
        for N in particle_counts
    }

    ymin = min(float(v.min()) for v in mean_score_by_N.values())
    ymax = max(float(v.max()) for v in mean_score_by_N.values())
    pad = 0.05 * max(1e-8, ymax - ymin)

    plt.figure(figsize=(8, 5))
    for N in particle_counts:
        plt.plot(steps_arr, mean_score_by_N[N], marker="o", linewidth=2, label=f"N = {N}")
        if show_stderr:
            plt.fill_between(
                steps_arr,
                mean_score_by_N[N] - stderr_score_by_N[N],
                mean_score_by_N[N] + stderr_score_by_N[N],
                alpha=0.12,
            )

    plt.xlabel("DPnP step")
    plt.ylabel(f"mean {ylabel}  {better}")
    plt.title(
        f"Two-point seismic: {ylabel} vs DPnP step\n"
        f"{sensor_desc}, averaged over {n_trials} (u1, u2) realisations"
    )
    plt.ylim(ymin - pad, ymax + pad)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "x_true": x_true.cpu(),
        "y1": y1.cpu(),
        "y2": y2.cpu(),
        "u1_all": u1_all.cpu(),
        "u2_all": u2_all.cpu(),
        "X_finals": X_finals.cpu(),
        "trial_means": trial_means.cpu(),
        "overall_mean": overall_mean.cpu(),
        "score_trials": score_trials.cpu(),
        "score_steps_by_N": {N: score_steps_by_N[N].numpy() for N in particle_counts},
        "metric": metric,
    }