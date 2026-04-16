import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

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
    sample_kent_s2,
    seismic_signal,
    get_seismic_f_fn,
    get_two_point_seismic_f_fn,
    mcmc_mh_s2_batched,
    get_two_point_seismic_log_target,
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
    aggregate_mode="mean",   # "mean" or "min_overall"
):
    """
    Sample x_true and y as before. For each DPnP step and each requested particle
    count N, compute the cosine similarity between x_true and the spherical mean
    of the first N sampled particles.

    aggregate_mode:
      - "mean": current behavior
      - "min_overall": use the maximum cosine over all sampled pairs overall,
                       both for the dashed observation line and for each DPnP curve
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
    if aggregate_mode not in {"mean", "min_overall"}:
        raise ValueError('aggregate_mode must be "mean" or "min_overall"')

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

    if aggregate_mode == "mean":
        curve_by_N = {
            N: all_cos_by_N[N].mean(dim=1).numpy()
            for N in particle_counts
        }
        stderr_by_N = {
            N: (
                all_cos_by_N[N].std(dim=1, unbiased=False).numpy()
                / np.sqrt(all_cos_by_N[N].shape[1])
            )
            for N in particle_counts
        }

        obs_cos = (x_true_all * y_all).sum(dim=-1)
        obs_cos_baseline = obs_cos.mean().item()

        ylabel = "mean cosine similarity"
        baseline_label = "mean cosine from observation y"
        curve_label_fmt = "N = {N}"
        plot_title = (
            "Cosine similarity vs DPnP step\n"
            "spherical mean reconstruction using N particles"
        )

    else:  # aggregate_mode == "min_overall"
        curve_by_N = {
            N: all_cos_by_N[N].min(dim=1).values.numpy()
            for N in particle_counts
        }

        # no stderr for a max-overall summary
        stderr_by_N = {N: None for N in particle_counts}

        obs_cos = (x_true_all * y_all).sum(dim=-1)
        obs_cos_baseline = obs_cos.min().item()

        ylabel = "min cosine similarity overall"
        baseline_label = "min cosine from observation y overall"
        curve_label_fmt = "N = {N}"
        plot_title = (
            "Min cosine similarity vs DPnP step\n"
            "spherical mean reconstruction using N particles"
        )

    steps = np.arange(S_plus_1)

    ymin_cos = min(
        [obs_cos_baseline] + [float(curve_by_N[N].min()) for N in particle_counts]
    )
    ymax_cos = max(
        [obs_cos_baseline] + [float(curve_by_N[N].max()) for N in particle_counts]
    )
    pad_cos = 0.05 * max(1e-8, ymax_cos - ymin_cos)
    ymin_cos -= pad_cos
    ymax_cos += pad_cos

    print(f"{baseline_label} = {obs_cos_baseline:.6f}")
    for N in particle_counts:
        print(f"N = {N:>3d}, step 0 value = {curve_by_N[N][0]:.6f}")

    plt.figure(figsize=(8, 5))

    plt.axhline(
        obs_cos_baseline,
        linestyle="--",
        linewidth=2.5,
        color="black",
        alpha=0.9,
        zorder=1,
        label=baseline_label,
    )

    for N in particle_counts:
        plt.plot(
            steps,
            curve_by_N[N],
            marker="o",
            linewidth=2,
            zorder=3,
            label=curve_label_fmt.format(N=N),
        )

        if show_stderr and aggregate_mode == "mean":
            plt.fill_between(
                steps,
                curve_by_N[N] - stderr_by_N[N],
                curve_by_N[N] + stderr_by_N[N],
                alpha=0.12,
                zorder=2,
            )

    plt.xlabel("DPnP step")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel(ylabel)
    plt.title(plot_title)
    plt.ylim(ymin_cos, ymax_cos)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "steps": steps,
        "aggregate_mode": aggregate_mode,
        "obs_cos_baseline": obs_cos_baseline,
        "by_particle_count": {
            N: {
                "curve": curve_by_N[N],
                "stderr": stderr_by_N[N],
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
    levels: int = 20,
    color_bar = False
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
    if color_bar:
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
    color:str = 'white',
    color_bar = False
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
    if color_bar:
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
    color_bar = False
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
    if color_bar:
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
    x_size=20,
    recon_size=20,
    recon_alpha=0.6,
):
    """
    Aggregate KDE plot from test_dpnp_sampler_multiple_xtrue results.

    mode="x_mean"        -> pooled per-y averaged reconstructions
                           (recon_mean_per_y across all x_true and y)
    mode="x_all"         -> pooled non-averaged final reconstruction particles
                           (all particles across all x_true and y; automatically
                           subsampled for KDE if too large)
    mode="x_global_mean" -> refactored to also use pooled per-y averaged
                           reconstructions for BOTH the KDE and overlayed points
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
        # Refactor requested:
        # use all per-y reconstructions for BOTH KDE and overlayed points
        samples = torch.cat(
            [r["recon_mean_per_y"] for r in results],
            dim=0,
        )
        default_title = (
            "All selected $x_{true}$ overlaid with KDE of pooled "
            "per-$y$ averaged reconstructions"
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
        x_marker_size=x_size,
        recon_marker_size=recon_size,
        recon_alpha=recon_alpha,
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
    metric="geodesic": great-circle distance in radians in [0, π], lower is better.
    """
    dot = (x_recon * x_true).sum(dim=-1).clamp(-1.0, 1.0)
    if metric == "cosine":
        return dot
    elif metric == "geodesic":
        return torch.arccos(dot)
    else:
        raise ValueError(f"metric must be 'cosine' or 'geodesic', got '{metric}'")


def _reflect_through_great_circle(
    x: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
) -> torch.Tensor:
    """
    Reflect x through the great circle spanned by y1 and y2.

    The two-point seismic likelihood I(x,y_i) = exp(β(<x,y_i>−1)) depends on
    x only through its inner products with y1 and y2.  Hence any x that
    satisfies <x,y1>=<x_true,y1> AND <x,y2>=<x_true,y2> is observationally
    *identical* to x_true from both sensors.  The unique other such point on
    S² is the reflection of x_true through the plane span(y1,y2):

        x' = x − 2 (x · n) n,  n = normalize(y1 × y2).

    Parameters
    ----------
    x  : (..., 3)
    y1 : (3,) or (S, 3) -- broadcast-compatible with x
    y2 : (3,) or (S, 3)

    Returns
    -------
    x' : (..., 3), normalised onto S^2.  Equals x when x lies in span(y1,y2).
    """
    cross = torch.linalg.cross(y1, y2, dim=-1)                 # (..., 3)
    n = cross / cross.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    proj = (x * n).sum(dim=-1, keepdim=True)                   # (..., 1)
    return normalize_torch(x - 2.0 * proj * n)


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
    kent_beta: float | None = None,
    metric: str = "cosine",
    n_bel_paths: int = 5000,
    n_bel_steps: int = 5,
    particle_counts=(1, 5, 10, 20),
    out_samples: int = 64,
    grw_steps: int = 5,
    num_pairs: int = 64,
    batch_eval_size: int = 8,
    include_baseline: bool = True,
    mcmc_tau: float = 0.1,
    mcmc_use_mala: bool = False,
    mcmc_n_burnin: int = 500,
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
                    If given, sample y1_i, y2_i from a distribution centred on
                    x_true_i independently for each x_true instead of fixed y1/y2.
                    Larger kappa → sensors closer to x_true.
    kent_beta     : float or None (default None)
                    If set together with kappa_sensors, sensors are drawn from
                    the Kent distribution Kent(x_true, kappa_sensors, kent_beta).
                    Requires 0 ≤ 2*kent_beta < kappa_sensors.
                    When None, plain vMF is used.
    metric        : "cosine" (default) or "geodesic"
                    "cosine"   -- dot product in [-1,1], higher is better.
                    "geodesic" -- great-circle distance in radians, lower is better.
    n_bel_paths, n_bel_steps : BEL hyper-parameters
    particle_counts : N values to evaluate
    out_samples   : total particles per DPnP call (>= max(particle_counts))
    grw_steps     : SDE integration steps per DPnP step
    num_pairs     : number of (x_true, u1, u2) triples to average over
    batch_eval_size : how many triples to batch into one DPnP call
    include_baseline : bool (default True)
                    If True, also sample from the likelihood-only posterior
                    q(x) ∝ p(u1|x,y1) p(u2|x,y2) using MCMC on S² and
                    overlay results as horizontal dashed lines on the
                    quality-vs-step plot.
                    Note: DPnP with p_score≡0 does NOT achieve this — the
                    BEL bridge p(x0|xt) ∝ p(xt|x0) p0(x0) implicitly uses
                    the earthquake prior, so the baseline must be independent
                    of the learned score model.
    mcmc_tau        : step size for MCMC baseline (default 0.1).
    mcmc_use_mala   : bool (default False). If True, use manifold MALA.
    mcmc_n_burnin   : burn-in steps for MCMC (default 500).
    seed          : random seed

    Returns
    -------
    dict with keys:
        "steps", "metric",
        "by_particle_count": {N: {"mean_score", "stderr_score"}},
        "baseline_by_particle_count": {N: {"mean_score", "stderr_score"}}   # if include_baseline
    """
    _uniform_sensors = (kappa_sensors is None and kent_beta is None
                        and y1 is None and y2 is None)
    if not _uniform_sensors and kappa_sensors is None and (y1 is None or y2 is None):
        raise ValueError(
            "Provide y1/y2, set kappa_sensors, or leave all sensor args None "
            "for uniform random sensors."
        )
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

    score_chunks_by_N    = {N: [] for N in particle_counts}
    score_chunks_by_N_bl = {N: [] for N in particle_counts} if include_baseline else {}
    running_seed = seed

    for start in range(0, num_pairs, batch_eval_size):
        end = min(start + batch_eval_size, num_pairs)
        x_batch = x_true_all[start:end]          # (B, 3)
        B = x_batch.shape[0]

        gen = torch.Generator(device=device)
        gen.manual_seed(running_seed)

        # ---- sensor locations for this mini-batch ----
        if kappa_sensors is not None:
            if kent_beta is not None:
                # Sample y1_b, y2_b per x_true ~ Kent(x_true, kappa_sensors, kent_beta)
                y1_b = sample_kent_s2(
                    mu=x_batch, kappa=kappa_sensors, beta=kent_beta, generator=gen,
                )   # (B, 3)
                y2_b = sample_kent_s2(
                    mu=x_batch, kappa=kappa_sensors, beta=kent_beta, generator=gen,
                )   # (B, 3)
            else:
                # Sample y1_b, y2_b per x_true ~ vMF(x_true, kappa_sensors)
                y1_b = sample_vmf_s2(mu=x_batch, kappa=kappa_sensors, generator=gen)  # (B, 3)
                y2_b = sample_vmf_s2(mu=x_batch, kappa=kappa_sensors, generator=gen)  # (B, 3)
        elif _uniform_sensors:
            # Uniform random on S²
            y1_b = normalize_torch(torch.randn(B, 3, device=device, dtype=dtype, generator=gen))
            y2_b = normalize_torch(torch.randn(B, 3, device=device, dtype=dtype, generator=gen))
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
        q_score_fn = get_bel(f_fn=f_fn, n_paths=n_bel_paths, n_steps=n_bel_steps,
                             device=device, dtype=dtype)

        # Dummy y sets B; use y1_b (per-item) or broadcast y1
        dummy_y = (y1_b if (kappa_sensors is not None or _uniform_sensors)
                   else y1.unsqueeze(0).expand(B, 3))

        X_steps = dPnP_sampler_torch_batched(
            q_score=q_score_fn,
            p_score=p_score,
            y=dummy_y,
            out_samples=out_samples,
            eta=eta,
            grw_steps=grw_steps,
            seed=running_seed,
            end_only=False,
            device=device,
            dtype=dtype,
        )   # (STEPS+1, B, P, 3)

        if include_baseline:
            # MCMC on S² — the only correct way to sample from the likelihood-only
            # baseline q(x) ∝ p(u1|x,y1)p(u2|x,y2).  DPnP with p_score≡0 is NOT
            # equivalent: BEL bridge paths use p(x0|xt)∝p(xt|x0)p0(x0), which
            # encodes the earthquake prior p0 regardless of what p_score returns.
            log_tgt = get_two_point_seismic_log_target(
                y1_b if (kappa_sensors is not None or _uniform_sensors) else y1.unsqueeze(0).expand(B, 3),
                y2_b if (kappa_sensors is not None or _uniform_sensors) else y2.unsqueeze(0).expand(B, 3),
                u1_batch, u2_batch, beta, sigma2,
            )
            x_init_bl = (None if _uniform_sensors
                         else y1_b if kappa_sensors is not None
                         else y1.unsqueeze(0).expand(B, 3))
            X_mcmc_bl, _ = mcmc_mh_s2_batched(
                log_target_fn=log_tgt,
                B=B,
                n_samples=out_samples,
                tau=mcmc_tau,
                n_burnin=mcmc_n_burnin,
                x_init=x_init_bl,
                use_mala=mcmc_use_mala,
                device=device,
                dtype=dtype,
                seed=running_seed + 1,
            )   # (out_samples, B, 3)
            X_final_bl = X_mcmc_bl.permute(1, 0, 2)             # (B, out_samples, 3)

        running_seed += 1

        S_plus_1 = X_steps.shape[0]

        for N in particle_counts:
            X_sub = X_steps[:, :, :N, :]                        # (STEPS+1, B, N, 3)
            X_sub_mean = sphere_mean_torch(X_sub, dim=2)         # (STEPS+1, B, 3)
            sc = _reconstruction_score(X_sub_mean, x_batch[None], metric)  # (STEPS+1, B)
            score_chunks_by_N[N].append(sc.cpu())
            if include_baseline:
                X_sub_bl = X_final_bl[:, :N, :]                  # (B, N, 3)
                X_sub_mean_bl = sphere_mean_torch(X_sub_bl, dim=1)# (B, 3)
                sc_bl = _reconstruction_score(X_sub_mean_bl, x_batch, metric)  # (B,)
                score_chunks_by_N_bl[N].append(sc_bl.cpu())

    all_scores_by_N = {
        N: torch.cat(score_chunks_by_N[N], dim=1)   # (STEPS+1, num_pairs)
        for N in particle_counts
    }
    mean_score_by_N  = {N: all_scores_by_N[N].mean(dim=1).numpy()  for N in particle_counts}
    stderr_score_by_N = {
        N: (all_scores_by_N[N].std(dim=1, unbiased=False).numpy()
            / np.sqrt(all_scores_by_N[N].shape[1]))
        for N in particle_counts
    }

    # Aggregate MCMC baseline chunks: each is (B,) → cat → (num_pairs,) scalar
    mean_score_by_N_bl   = {}
    stderr_score_by_N_bl = {}
    if include_baseline:
        for N in particle_counts:
            cat_bl = torch.cat(score_chunks_by_N_bl[N], dim=0)  # (num_pairs,)
            mean_score_by_N_bl[N]   = float(cat_bl.mean())
            stderr_score_by_N_bl[N] = float(cat_bl.std(unbiased=False)
                                             / np.sqrt(cat_bl.shape[0]))

    steps = np.arange(S_plus_1)
    ylabel = "mean cosine similarity" if metric == "cosine" else "mean geodesic distance (rad)"
    better = "↑ better" if metric == "cosine" else "↓ better"

    # ---- summary print ----
    print("DPnP:")
    for N in particle_counts:
        print(f"  N = {N:>3d} | final step mean {metric} = {mean_score_by_N[N][-1]:.4f}")
    if include_baseline:
        print("MCMC baseline (likelihood-only):")
        for N in particle_counts:
            print(f"  N = {N:>3d} | mean {metric} = {mean_score_by_N_bl[N]:.4f}")

    # ---- plot ----

    if _uniform_sensors:
        sensor_desc = "uniform random sensors"
    elif kappa_sensors is None:
        sensor_desc = "fixed sensors"
    elif kent_beta is not None:
        sensor_desc = f"Kent sensors kappa={kappa_sensors}, beta={kent_beta}"
    else:
        sensor_desc = f"vMF sensors kappa={kappa_sensors}"

    bl_flat = [mean_score_by_N_bl[N] for N in particle_counts] if include_baseline else []
    all_flat = [float(x) for arr in mean_score_by_N.values() for x in arr] + bl_flat
    ymin = min(all_flat)
    ymax = max(all_flat)
    pad = 0.05 * max(1e-8, ymax - ymin)

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.figure(figsize=(8, 5))
    for i, N in enumerate(particle_counts):
        col = prop_cycle[i % len(prop_cycle)]
        plt.plot(steps, mean_score_by_N[N], marker="o", linewidth=2,
                 color=col, label=f"DPnP N={N}")
        if show_stderr:
            plt.fill_between(steps,
                             mean_score_by_N[N] - stderr_score_by_N[N],
                             mean_score_by_N[N] + stderr_score_by_N[N],
                             alpha=0.12, color=col)
        if N in mean_score_by_N_bl:
            # MCMC has no step axis → horizontal dashed reference line
            plt.axhline(mean_score_by_N_bl[N], color=col, linestyle="--",
                        linewidth=1.5, alpha=0.65,
                        label=f"MCMC baseline N={N}")
            if show_stderr:
                plt.axhspan(mean_score_by_N_bl[N] - stderr_score_by_N_bl[N],
                            mean_score_by_N_bl[N] + stderr_score_by_N_bl[N],
                            alpha=0.06, color=col)

    plt.xlabel("DPnP step")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel(f"{ylabel}  ({better})")
    plt.title(
        f"Two-point seismic: {ylabel} vs DPnP step\n"
        f"{sensor_desc}, spherical mean over N particles"
        + (" | dashed = MCMC likelihood-only baseline" if include_baseline else "")
    )
    plt.ylim(ymin - pad, ymax + pad)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()

    out = {
        "steps": steps,
        "metric": metric,
        "by_particle_count": {
            N: {"mean_score": mean_score_by_N[N], "stderr_score": stderr_score_by_N[N]}
            for N in particle_counts
        },
    }
    if include_baseline:
        out["baseline_by_particle_count"] = {
            N: {"mean_score": mean_score_by_N_bl[N], "stderr_score": stderr_score_by_N_bl[N]}
            for N in particle_counts
        }
    return out


# =========================================================
# Two-point seismic demo – one x_true, multiple (y1,y2) pairs,
#                          multiple (u1,u2) trials per sensor pair
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
    n_sensor_pairs: int = 5,
    n_trials_per_sensor: int = 10,
    kappa_sensors: float | None = None,
    metric: str = "cosine",
    n_bel_paths: int = 5000,
    n_bel_steps: int = 5,
    out_samples: int = 32,
    grw_steps: int = 5,
    particle_counts=(1, 5, 10, 20),
    kent_beta: float | None = None,
    kde_kappa: float = 25.0,
    plot_per_sensor: bool = True,
    plot_global: bool = True,
    plot_combined: bool = False,
    plot_quality_vs_steps: bool = True,
    plot_3d: bool = False,
    show_kde: bool = True,
    include_baseline: bool = True,
    mcmc_tau: float = 0.1,
    mcmc_use_mala: bool = False,
    mcmc_n_burnin: int = 500,
    device="mps",
    dtype=torch.float32,
    seed: int = 0,
    show_stderr: bool = True,
):
    """
    Two-point seismic earthquake reconstruction demo.

    Hierarchy:
        x_true
        └── sensor pair s = 1 … n_sensor_pairs  (y1_s, y2_s)
            └── u-trial  t = 1 … n_trials_per_sensor
                └── DPnP → out_samples particles → trial_mean[s, t]

    All S × T trials are batched into **one** DPnP call for efficiency.

    Parameters
    ----------
    x_true              : (3,) tensor -- true earthquake location on S^2
    y1, y2              : (3,) or (S, 3) tensors, or None.
                          Required when kappa_sensors is None.
                          (3,) tensors are broadcast to all S sensor pairs.
    beta                : float -- seismic signal decay rate (beta > 0)
    sigma2              : float -- observation noise variance
    p_score             : callable -- trained prior score p_score(x, t)
    eta                 : 1-D array -- DPnP time schedule
    n_sensor_pairs      : int (S) -- number of (y1, y2) configurations
    n_trials_per_sensor : int (T) -- number of (u1, u2) draws per sensor pair
    kappa_sensors       : float or None
                          Sample y1_s, y2_s from a distribution centred on
                          x_true, independently for each s.  When set, y1/y2
                          are ignored.  Larger kappa → sensors closer to x_true.
    kent_beta           : float or None (default None)
                          If set together with kappa_sensors, sensors are sampled
                          from the Kent (FB5) distribution
                              p(x|μ,κ,β) ∝ exp(κ⟨μ,x⟩ + β(⟨γ₁,x⟩²−⟨γ₂,x⟩²))
                          with μ=x_true, κ=kappa_sensors, β=kent_beta.
                          Requires 0 ≤ 2β < κ.  When None, plain vMF is used.
    metric              : "cosine" (default) or "geodesic"
    n_bel_paths, n_bel_steps : BEL hyper-parameters
    out_samples         : DPnP particles P per trial
    grw_steps           : SDE integration steps per DPnP outer step
    particle_counts     : N values for the quality-vs-step plot
    kde_kappa           : vMF bandwidth for KDE plots (ignored when show_kde=False)
    plot_per_sensor     : bool (default True)
                          One plot per sensor pair.
    plot_global         : bool (default True)
                          One plot across all sensor pairs.
    plot_quality_vs_steps : bool (default True)
                          Quality metric vs DPnP step, averaged over all S*T.
    plot_3d             : bool (default False)
                          If True, render sphere plots in 3D (matplotlib 3D axes)
                          so x_true and the sample cloud are visible on S².
                          If False, use Mollweide projections (default).
    show_kde            : bool (default True)
                          If True, overlay a spherical KDE on the sphere surface.
                          If False, scatter raw sample points only.
    include_baseline    : bool (default True)
                          If True, also draw samples from the likelihood-only
                          posterior q(x) ∝ p(u1|x,y1) p(u2|x,y2) using
                          MCMC on S² (no score network), and overlay results
                          on all plots.
                          Note: DPnP with p_score≡0 does NOT achieve this —
                          BEL bridge paths use p(x0|xt)∝p(xt|x0)p0(x0),
                          embedding the earthquake prior regardless of p_score.
    mcmc_tau            : float (default 0.1)
                          Step size for the MCMC baseline (τ for random walk,
                          η for MALA).
    mcmc_use_mala       : bool (default False)
                          If True, use manifold MALA (gradient-guided proposals)
                          instead of geodesic random walk.
    mcmc_n_burnin       : int (default 500)
                          Burn-in steps for the MCMC baseline.
    seed                : base random seed

    Returns
    -------
    dict with keys
        x_true               : (3,)
        y1_sensors           : (S, 3)
        y2_sensors           : (S, 3)
        x_reflected          : (S, 3)  -- reflection of x_true through span(y1_s,y2_s)
                                          has identical seismic likelihood as x_true
        u1_all               : (S, T)
        u2_all               : (S, T)
        X_finals             : (S, T, P, 3) -- final DPnP particles
        trial_means          : (S, T, 3)    -- spherical mean over P per trial
        sensor_means         : (S, 3)       -- spherical mean over T per sensor pair
        global_mean          : (3,)         -- spherical mean over all S*T trials
        score_trials         : (S, T)       -- metric at trial_means
        score_sensors        : (S,)         -- metric at sensor_means
        score_global         : float        -- metric at global_mean
        score_steps_by_N     : {N: (STEPS+1, S, T) ndarray}
        -- if include_baseline --
        X_finals_bl          : (S, T, P, 3) -- baseline (no prior) final particles
        trial_means_bl       : (S, T, 3)
        score_trials_bl      : (S, T)
        score_global_bl      : float
        score_steps_by_N_bl  : {N: (STEPS+1, S, T) ndarray}
        metric               : str
    """
    if p_score is None or eta is None:
        raise ValueError("p_score and eta are required.")
    _uniform_sensors = (kappa_sensors is None and kent_beta is None
                        and y1 is None and y2 is None)
    if not _uniform_sensors and kappa_sensors is None and (y1 is None or y2 is None):
        raise ValueError(
            "Provide y1/y2, set kappa_sensors, or leave all sensor args None "
            "for uniform random sensors."
        )
    if metric not in {"cosine", "geodesic"}:
        raise ValueError("metric must be 'cosine' or 'geodesic'")

    device = torch.device(device)
    eta_t = torch.as_tensor(eta, device=device, dtype=dtype)
    STEPS = int(eta_t.numel())

    x_true = normalize_torch(torch.as_tensor(x_true, device=device, dtype=dtype))  # (3,)
    S = int(n_sensor_pairs)
    T = int(n_trials_per_sensor)
    P = int(out_samples)
    sigma = math.sqrt(sigma2)
    particle_counts = sorted(set(int(n) for n in particle_counts))
    if max(particle_counts) > P:
        raise ValueError(f"max particle count {max(particle_counts)} > out_samples {P}")

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # ── 1. Sensor locations (S, 3) ────────────────────────────────────────────
    if kappa_sensors is not None:
        if kent_beta is not None:
            y1_sensors = sample_kent_s2(
                mu=x_true, kappa=kappa_sensors, beta=kent_beta,
                n_samples=S, generator=gen,
            )   # (S, 3)
            y2_sensors = sample_kent_s2(
                mu=x_true, kappa=kappa_sensors, beta=kent_beta,
                n_samples=S, generator=gen,
            )   # (S, 3)
            sensor_desc = f"Kent sensors (kappa={kappa_sensors}, beta={kent_beta})"
            print(f"Sampled {S} sensor pairs from Kent(x_true, kappa={kappa_sensors}, beta={kent_beta})")
        else:
            y1_sensors = sample_vmf_s2(mu=x_true, kappa=kappa_sensors,
                                       n_samples=S, generator=gen)   # (S, 3)
            y2_sensors = sample_vmf_s2(mu=x_true, kappa=kappa_sensors,
                                       n_samples=S, generator=gen)   # (S, 3)
            sensor_desc = f"vMF sensors (kappa={kappa_sensors})"
            print(f"Sampled {S} sensor pairs from vMF(x_true, kappa={kappa_sensors})")
    elif _uniform_sensors:
        # Sample S sensor pairs uniformly on S²
        y1_sensors = normalize_torch(
            torch.randn(S, 3, device=device, dtype=dtype, generator=gen))
        y2_sensors = normalize_torch(
            torch.randn(S, 3, device=device, dtype=dtype, generator=gen))
        sensor_desc = "uniform random sensors"
        print(f"Sampled {S} sensor pairs uniformly on S²")
    else:
        y1_t = normalize_torch(torch.as_tensor(y1, device=device, dtype=dtype))
        y2_t = normalize_torch(torch.as_tensor(y2, device=device, dtype=dtype))
        if y1_t.dim() == 1:          # (3,) → broadcast
            y1_sensors = y1_t.unsqueeze(0).expand(S, 3).clone()
            y2_sensors = y2_t.unsqueeze(0).expand(S, 3).clone()
        else:                         # already (S, 3)
            y1_sensors = y1_t
            y2_sensors = y2_t
        sensor_desc = "fixed sensors"

    ylabel  = "cosine similarity" if metric == "cosine" else "geodesic distance (rad)"
    better  = "(↑ better)"        if metric == "cosine" else "(↓ better)"

    # ── 2. True seismic signals and noisy observations ────────────────────────
    # I_true: (S,)  — signal at each sensor pair
    I1_true = seismic_signal(x_true, y1_sensors, beta)   # (S,)
    I2_true = seismic_signal(x_true, y2_sensors, beta)   # (S,)

    # u_all: (S, T)  — T u-draws per sensor pair
    noise1 = torch.randn(S, T, device=device, dtype=dtype, generator=gen)
    noise2 = torch.randn(S, T, device=device, dtype=dtype, generator=gen)
    u1_all = I1_true[:, None] + sigma * noise1   # (S, T)
    u2_all = I2_true[:, None] + sigma * noise2   # (S, T)
    print(f"Sampled {S}×{T} = {S*T} (u1, u2) pairs.")

    # ── 3. Batch all S×T trials into one DPnP call ───────────────────────────
    # Flatten: item k = s*T + t  →  sensor s, trial t
    y1_flat = y1_sensors.unsqueeze(1).expand(S, T, 3).reshape(S * T, 3)   # (S*T, 3)
    y2_flat = y2_sensors.unsqueeze(1).expand(S, T, 3).reshape(S * T, 3)
    u1_flat = u1_all.reshape(S * T)                                        # (S*T,)
    u2_flat = u2_all.reshape(S * T)

    f_fn = get_two_point_seismic_f_fn(y1_flat, y2_flat, u1_flat, u2_flat, beta, sigma2)
    q_score = get_bel(f_fn=f_fn, n_paths=n_bel_paths, n_steps=n_bel_steps,
                      device=device, dtype=dtype)

    X_steps = dPnP_sampler_torch_batched(
        q_score=q_score,
        p_score=p_score,
        y=y1_flat,            # (S*T, 3) — sets DPnP batch B = S*T
        out_samples=P,
        eta=eta_t,
        grw_steps=grw_steps,
        seed=seed,
        end_only=False,
        device=device,
        dtype=dtype,
    )   # (STEPS+1, S*T, P, 3)

    # ── 4. Reshape and compute hierarchical means ─────────────────────────────
    X_finals     = X_steps[-1].reshape(S, T, P, 3)                  # (S, T, P, 3)
    trial_means  = sphere_mean_torch(X_finals, dim=2)                # (S, T, 3)
    sensor_means = sphere_mean_torch(trial_means, dim=1)             # (S, 3)
    global_mean  = sphere_mean_torch(sensor_means, dim=0)            # (3,)

    # ── 5. Reconstruction quality metrics ─────────────────────────────────────
    xt_ST = x_true.expand(S, T, 3)
    xt_S  = x_true.expand(S, 3)
    score_trials  = _reconstruction_score(trial_means,  xt_ST, metric)          # (S, T)
    score_sensors = _reconstruction_score(sensor_means, xt_S,  metric)          # (S,)
    score_global  = _reconstruction_score(global_mean[None], x_true[None], metric).item()

    print(f"\nTrial {ylabel}:  mean={score_trials.mean():.4f}  "
          f"std={score_trials.std():.4f}  {better}")
    print(f"Sensor-mean {ylabel}: mean={score_sensors.mean():.4f}  "
          f"std={score_sensors.std():.4f}")
    print(f"Global mean {ylabel}: {score_global:.4f}")

    # ── 5b. Reflection points (likelihood-ambiguous partners) ─────────────────
    # For sensor pair s, x'_s = reflect(x_true, span(y1_s, y2_s)).
    # At x'_s: <x'_s, y_i> = <x_true, y_i> for both i, so the seismic
    # likelihood is exactly the same — this is the point the sensors cannot
    # distinguish from x_true without prior information.
    x_reflected = _reflect_through_great_circle(
        x_true.expand(S, 3), y1_sensors, y2_sensors
    )   # (S, 3)
    score_reflected = _reconstruction_score(x_reflected, x_true.expand(S, 3), metric)
    print(f"Reflection point {ylabel}: mean={score_reflected.mean():.4f}  "
          f"min={score_reflected.min():.4f}  max={score_reflected.max():.4f}")

    # ── 6. Per-step metric for quality-vs-steps plot ──────────────────────────
    score_steps_by_N = {}
    for N in particle_counts:
        X_sub      = X_steps[:, :, :N, :]                           # (STEPS+1, S*T, N, 3)
        X_sub_mean = sphere_mean_torch(X_sub, dim=2)                 # (STEPS+1, S*T, 3)
        sc = _reconstruction_score(
            X_sub_mean,
            x_true.expand(S * T, 3)[None].expand(STEPS + 1, S * T, 3),
            metric,
        )   # (STEPS+1, S*T)
        score_steps_by_N[N] = sc.reshape(STEPS + 1, S, T).cpu()     # (STEPS+1, S, T)

    # ── 6b. Baseline: MCMC on S² (likelihood-only) ───────────────────────────
    # Samples from q(x) ∝ p(u1|x,y1) p(u2|x,y2) via geodesic random-walk Metropolis.
    # DPnP with p_score≡0 is NOT a valid baseline: BEL bridge paths use
    # p(x0|xt) ∝ p(xt|x0) p0(x0), which embeds the earthquake prior p0 in
    # the score estimate regardless of what p_score returns.
    X_finals_bl      = None
    trial_means_bl   = None
    score_trials_bl  = None
    score_global_bl  = None
    score_steps_by_N_bl = {}

    if include_baseline:
        mala_str = "MALA" if mcmc_use_mala else "geodesic RW"
        print(f"\nRunning likelihood-only baseline ({mala_str} on S²)…")
        log_target = get_two_point_seismic_log_target(
            y1_flat, y2_flat, u1_flat, u2_flat, beta, sigma2
        )
        X_mcmc, accept_rate = mcmc_mh_s2_batched(
            log_target_fn=log_target,
            B=S * T,
            n_samples=P,
            tau=mcmc_tau,
            n_burnin=mcmc_n_burnin,
            x_init=None if _uniform_sensors else y1_flat,
            use_mala=mcmc_use_mala,
            device=device,
            dtype=dtype,
            seed=seed + 1,
        )   # (P, S*T, 3)
        print(f"  MCMC acceptance rate: {accept_rate:.3f}")

        X_finals_bl    = X_mcmc.permute(1, 0, 2).reshape(S, T, P, 3)   # (S, T, P, 3)
        trial_means_bl = sphere_mean_torch(X_finals_bl, dim=2)           # (S, T, 3)
        score_trials_bl = _reconstruction_score(trial_means_bl, xt_ST, metric)  # (S, T)
        score_global_bl = _reconstruction_score(
            sphere_mean_torch(trial_means_bl.reshape(S * T, 3), dim=0)[None],
            x_true[None], metric,
        ).item()
        print(f"Baseline trial {ylabel}: mean={score_trials_bl.mean():.4f}  "
              f"std={score_trials_bl.std():.4f}  {better}")
        print(f"Baseline global {ylabel}: {score_global_bl:.4f}")

        for N in particle_counts:
            X_sub_bl      = X_finals_bl[:, :, :N, :]
            X_sub_mean_bl = sphere_mean_torch(X_sub_bl, dim=2)    # (S, T, 3)
            sc_bl = _reconstruction_score(X_sub_mean_bl, xt_ST, metric)
            score_steps_by_N_bl[N] = sc_bl.cpu()                  # (S, T)

    # ── 7. Plot helpers ───────────────────────────────────────────────────────
    _lon_ax_labels = [
        "150°W","120°W","90°W","60°W","30°W","0°",
        "30°E","60°E","90°E","120°E","150°E"
    ]

    def _mollweide_ax(samples_3d, title_str, *, extra_markers=None):
        """Draw a Mollweide projection with optional KDE contour."""
        samp = normalize_torch(samples_3d.detach().cpu())

        fig, ax = plt.subplots(1, 1, figsize=(11, 5.5),
                               subplot_kw={"projection": "mollweide"})

        if show_kde:
            Lon, Lat, grid_xyz = mollweide_plot_grid_xyz_torch(
                n_lon=240, n_lat=120, device=samp.device, dtype=samp.dtype)
            dens = spherical_kde_vmf(samp, grid_xyz, kappa=kde_kappa, normalize=True)
            dens_np = dens.reshape(Lon.shape).cpu().numpy()
            cf = ax.contourf(Lon.cpu().numpy(), Lat.cpu().numpy(), dens_np,
                             levels=20, cmap="viridis")
            fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.08, label="spherical KDE")

        # x_true
        ll_x = extrinsic_to_mollweide_rad_torch(x_true.cpu()[None]).numpy()[0]
        ax.scatter([ll_x[1]], [ll_x[0]], s=100, color="tab:red",
                   edgecolors="black", linewidths=1.2, zorder=10,
                   label=r"$x_{\rm true}$")

        if extra_markers:
            for pts, kwargs in extra_markers:
                ll = extrinsic_to_mollweide_rad_torch(
                    normalize_torch(pts.detach().cpu())).numpy()
                if ll.ndim == 1:
                    ll = ll[None]
                ax.scatter(ll[:, 1], ll[:, 0], **kwargs)

        ax.grid(True, alpha=0.28)
        ax.set_xticklabels(_lon_ax_labels)
        ax.set_title(title_str)
        ax.legend(loc="upper right", fontsize=7)
        plt.tight_layout()
        plt.show()
        return fig, ax

    def _sphere_plotly_ax(samples_3d, title_str, *, extra_markers=None):
        """Rotatable Plotly 3D sphere — KDE surface when show_kde=True, wireframe otherwise."""
        import plotly.graph_objects as go
        import matplotlib.colors as mcolors

        samp = normalize_torch(samples_3d.detach().cpu())

        u_ang = np.linspace(0, 2 * np.pi, 80)
        v_ang = np.linspace(0,     np.pi, 40)
        Ug, Vg = np.meshgrid(u_ang, v_ang)
        Xs = np.sin(Vg) * np.cos(Ug)
        Ys = np.sin(Vg) * np.sin(Ug)
        Zs = np.cos(Vg)

        fig = go.Figure()

        # Shading: colour the sphere surface by a simple diffuse lighting model
        # so the camera-facing hemisphere appears lighter than the back side.
        # This gives a strong depth cue regardless of KDE mode.
        _light_dir = np.array([1.0, 1.0, 1.0])
        _light_dir = _light_dir / np.linalg.norm(_light_dir)
        _normals   = np.stack([Xs, Ys, Zs], axis=-1)        # unit normals = position on S²
        _diffuse   = np.clip((_normals * _light_dir).sum(-1), 0, 1)

        if show_kde:
            grid_np = np.stack([Xs, Ys, Zs], axis=-1).reshape(-1, 3)
            grid_t  = torch.tensor(grid_np, dtype=dtype).cpu()
            dens    = spherical_kde_vmf(samp, grid_t, kappa=kde_kappa, normalize=True)
            dens_np = dens.numpy().reshape(Vg.shape)
            fig.add_trace(go.Surface(
                x=Xs, y=Ys, z=Zs,
                surfacecolor=dens_np,
                colorscale="Viridis",
                opacity=0.92,
                showscale=True,
                colorbar=dict(title="KDE", thickness=15, len=0.65),
                lighting=dict(ambient=0.35, diffuse=0.65, specular=0.15,
                              roughness=0.5, fresnel=0.1),
                lightposition=dict(x=1000, y=1000, z=1000),
                name="KDE",
                showlegend=False,
            ))
        else:
            # Shaded opaque sphere so back-side markers are hidden
            fig.add_trace(go.Surface(
                x=Xs, y=Ys, z=Zs,
                surfacecolor=_diffuse,
                colorscale=[[0, "rgba(40,60,90,0.85)"],
                            [0.5, "rgba(100,149,237,0.70)"],
                            [1, "rgba(200,220,255,0.55)"]],
                opacity=0.55,
                showscale=False,
                lighting=dict(ambient=0.45, diffuse=0.55, specular=0.1,
                              roughness=0.6, fresnel=0.05),
                lightposition=dict(x=1000, y=1000, z=1000),
                showlegend=False,
            ))
            for v in np.linspace(0, np.pi, 10)[1:-1]:      # latitude circles
                lx = np.sin(v) * np.cos(u_ang)
                ly = np.sin(v) * np.sin(u_ang)
                lz = np.full_like(u_ang, np.cos(v))
                fig.add_trace(go.Scatter3d(
                    x=lx, y=ly, z=lz, mode="lines",
                    line=dict(color="steelblue", width=1.5),
                    showlegend=False, hoverinfo="skip",
                ))
            for u in np.linspace(0, 2 * np.pi, 13)[:-1]:  # longitude lines
                lx = np.sin(v_ang) * np.cos(u)
                ly = np.sin(v_ang) * np.sin(u)
                lz = np.cos(v_ang)
                fig.add_trace(go.Scatter3d(
                    x=lx, y=ly, z=lz, mode="lines",
                    line=dict(color="steelblue", width=1.5),
                    showlegend=False, hoverinfo="skip",
                ))

        def _c(color, alpha=1.0):
            try:
                r, g, b, a = mcolors.to_rgba(color, alpha=alpha)
                return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.3f})"
            except Exception:
                return "blue"

        # Plotly 3D only supports: circle, circle-open, cross, diamond,
        # diamond-open, square, square-open, x
        _sym = {None: "circle", "o": "circle", ".": "circle",
                "*": "diamond", "D": "diamond", "d": "diamond-open",
                "X": "x", "x": "x", "s": "square",
                "^": "circle", "v": "circle"}

        # x_true
        xt_np = x_true.cpu().numpy()
        fig.add_trace(go.Scatter3d(
            x=[xt_np[0]], y=[xt_np[1]], z=[xt_np[2]],
            mode="markers",
            marker=dict(size=12, color="red",
                        line=dict(color="black", width=2)),
            name="x_true",
        ))

        if extra_markers:
            for pts, kwargs in extra_markers:
                pts_np = normalize_torch(pts.detach().cpu()).numpy()
                if pts_np.ndim == 1:
                    pts_np = pts_np[None]
                s_mpl   = kwargs.get("s", 20)
                color   = kwargs.get("color", "blue")
                alpha   = kwargs.get("alpha", 1.0)
                ec      = kwargs.get("edgecolors", "none")
                lw      = kwargs.get("linewidths", 0)
                sym_mpl = kwargs.get("marker", "o")
                label   = kwargs.get("label", None)

                c_str  = _c(color, alpha)
                sym    = _sym.get(sym_mpl, "circle")
                psize  = max(4, int(np.sqrt(s_mpl) * 1.2))
                if ec not in (None, "none", "face"):
                    line_kw = dict(color=_c(ec), width=max(1, int(lw * 1.5)))
                else:
                    line_kw = dict(width=0)

                fig.add_trace(go.Scatter3d(
                    x=pts_np[:, 0], y=pts_np[:, 1], z=pts_np[:, 2],
                    mode="markers",
                    marker=dict(size=psize, color=c_str, symbol=sym, line=line_kw),
                    name=label or "",
                    showlegend=label is not None,
                ))

        fig.update_layout(
            title=dict(text=title_str, font=dict(size=13)),
            scene=dict(
                xaxis=dict(showticklabels=False, title="",
                           showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, title="",
                           showgrid=False, zeroline=False),
                zaxis=dict(showticklabels=False, title="",
                           showgrid=False, zeroline=False),
                aspectmode="cube",
            ),
            legend=dict(x=0.01, y=0.99, font=dict(size=9)),
            width=720, height=680,
            margin=dict(l=0, r=0, t=50, b=0),
        )
        fig.show()
        return fig

    def _plot_ax(samples_3d, title_str, *, extra_markers=None):
        """Dispatch to 3D (Plotly) or Mollweide plot depending on plot_3d flag."""
        if plot_3d:
            return _sphere_plotly_ax(samples_3d, title_str, extra_markers=extra_markers)
        else:
            return _mollweide_ax(samples_3d, title_str, extra_markers=extra_markers)

    sensor_colors = plt.cm.tab10(np.linspace(0, 1, max(S, 2)))

    # ── 8. Per-sensor plots ───────────────────────────────────────────────────
    if plot_per_sensor:
        for s in range(S):
            # KDE from all T*P particles for this sensor pair
            all_particles_s = X_finals[s].reshape(T * P, 3).cpu()
            trial_means_s   = trial_means[s].cpu()     # (T, 3)
            sensor_mean_s   = sensor_means[s].cpu()    # (3,)
            y1s = y1_sensors[s].cpu()
            y2s = y2_sensors[s].cpu()
            xr_s = x_reflected[s].cpu()                # reflection (3,)

            dpnp_trial_color = "white" if show_kde else "gold"
            extra = [
                # per-trial mean markers (DPnP)
                (trial_means_s,
                 dict(s=20, color=dpnp_trial_color, alpha=0.7, edgecolors="none",
                      zorder=6, label=f"DPnP trial means (T={T})")),
                # sensor-pair mean (DPnP)
                (sensor_mean_s[None],
                 dict(s=130, color="gold", marker="*", edgecolors="black",
                      linewidths=1.2, zorder=9, label="DPnP reconstruction mean")),
                # sensors
                (y1s[None],
                 dict(s=80, color="tab:cyan", marker="D", edgecolors="black",
                      linewidths=1.0, zorder=8, label=r"$y_1$")),
                (y2s[None],
                 dict(s=80, color="tab:orange", marker="D", edgecolors="black",
                      linewidths=1.0, zorder=8, label=r"$y_2$")),
                # reflection point: identical likelihood as x_true
                (xr_s[None],
                 dict(s=110, color="tab:green", marker="X", edgecolors="black",
                      linewidths=1.0, zorder=10,
                      label=r"$x'$ (reflection, same likelihood)")),
            ]

            # optionally overlay baseline trial means and reconstruction mean
            if include_baseline and trial_means_bl is not None:
                extra.append((
                    trial_means_bl[s].cpu(),
                    dict(s=20, color="tab:pink", alpha=0.7, edgecolors="none",
                         zorder=6, label=f"baseline trial means (T={T})"),
                ))
                bl_sensor_mean_s = sphere_mean_torch(trial_means_bl[s].cpu(), dim=0)
                extra.append((
                    bl_sensor_mean_s[None],
                    dict(s=130, color="tab:pink", marker="*", edgecolors="black",
                         linewidths=1.2, zorder=9, label="baseline reconstruction mean"),
                ))

            score_s = score_sensors[s].item()
            _plot_ax(
                all_particles_s,
                title_str=(
                    f"Sensor pair {s + 1}/{S} — {sensor_desc}\n"
                    f"{'KDE' if show_kde else 'Scatter'} of {T}×{P} DPnP particles | "
                    f"sensor-mean {ylabel}={score_s:.4f} {better}"
                ),
                extra_markers=extra,
            )

    # ── 9. Global plot (across all sensor pairs) ──────────────────────────────
    if plot_global:
        # KDE from all S*T trial means
        all_trial_means = trial_means.reshape(S * T, 3).cpu()   # (S*T, 3)

        extra_global = []
        for s in range(S):
            sm = sensor_means[s].cpu()
            col = sensor_colors[s % len(sensor_colors)]
            extra_global.append((
                sm[None],
                dict(s=70, color=col, marker="o", edgecolors="black",
                     linewidths=1.0, zorder=7,
                     label=f"DPnP reconstruction mean {s+1}" if s < 6 else None,
                     alpha=0.85),
            ))
            # reflection point for each sensor pair
            extra_global.append((
                x_reflected[s].cpu()[None],
                dict(s=55, color="tab:purple", marker="X", edgecolors="black",
                     linewidths=0.8, zorder=8, alpha=0.65,
                     label=r"$x'$ reflections" if s == 0 else None),
            ))
        extra_global.append((
            global_mean.cpu()[None],
            dict(s=160, color="gold", marker="*", edgecolors="black",
                 linewidths=1.4, zorder=9, label="DPnP global mean"),
        ))
        if include_baseline and trial_means_bl is not None:
            bl_sensor_means = sphere_mean_torch(trial_means_bl.cpu(), dim=1)  # (S, 3)
            for s in range(S):
                extra_global.append((
                    bl_sensor_means[s][None],
                    dict(s=70, color="tab:pink", marker="o", edgecolors="black",
                         linewidths=1.0, zorder=7, alpha=0.85,
                         label=f"baseline reconstruction mean {s+1}" if s < 6 else None),
                ))
            bl_global_mean = sphere_mean_torch(bl_sensor_means, dim=0)
            extra_global.append((
                bl_global_mean[None],
                dict(s=160, color="tab:pink", marker="*", edgecolors="black",
                     linewidths=1.4, zorder=9, label="baseline global mean"),
            ))

        _plot_ax(
            all_trial_means,
            title_str=(
                f"Global view — {sensor_desc}\n"
                f"{'KDE' if show_kde else 'Scatter'} of all {S}×{T} DPnP trial means | "
                f"global {ylabel}={score_global:.4f} {better}"
            ),
            extra_markers=extra_global,
        )

    # ── 9b. Combined plot ─────────────────────────────────────────────────────
    # Single plot: all sensor pairs + global aggregate + baseline + sensors + reflections
    if plot_combined:
        all_particles_c = X_finals.reshape(S * T * P, 3).cpu()
        extra_c = []

        for s in range(S):
            col = sensor_colors[s % len(sensor_colors)]
            sl  = f"s={s + 1}"
            # Per-sensor trial means (small, sensor-coloured)
            extra_c.append((
                trial_means[s].cpu(),
                dict(s=18, color=col, alpha=0.6, edgecolors="none",
                     zorder=6, label=f"DPnP trials {sl}" if s < 5 else None),
            ))
            # Per-sensor reconstruction mean (medium circle)
            extra_c.append((
                sensor_means[s].cpu()[None],
                dict(s=110, color=col, marker="o", edgecolors="black",
                     linewidths=1.2, zorder=8, alpha=0.9,
                     label=f"DPnP mean {sl}" if s < 5 else None),
            ))
            # Sensor locations y1, y2 (diamonds / triangles)
            extra_c.append((
                y1_sensors[s].cpu()[None],
                dict(s=65, color=col, marker="D", edgecolors="black",
                     linewidths=0.8, zorder=7, alpha=0.85,
                     label=r"$y_1$" if s == 0 else None),
            ))
            extra_c.append((
                y2_sensors[s].cpu()[None],
                dict(s=65, color=col, marker="^", edgecolors="black",
                     linewidths=0.8, zorder=7, alpha=0.85,
                     label=r"$y_2$" if s == 0 else None),
            ))
            # Reflection point
            extra_c.append((
                x_reflected[s].cpu()[None],
                dict(s=85, color="tab:purple", marker="X", edgecolors="black",
                     linewidths=0.9, zorder=9, alpha=0.75,
                     label=r"$x'$ reflections" if s == 0 else None),
            ))
        # Global DPnP mean
        extra_c.append((
            global_mean.cpu()[None],
            dict(s=210, color="gold", marker="*", edgecolors="black",
                 linewidths=1.5, zorder=11, label="DPnP global mean"),
        ))

        if include_baseline and trial_means_bl is not None:
            bl_smeans_c = sphere_mean_torch(trial_means_bl.cpu(), dim=1)  # (S, 3)
            for s in range(S):
                extra_c.append((
                    trial_means_bl[s].cpu(),
                    dict(s=18, color="tab:pink", alpha=0.35, edgecolors="none",
                         zorder=5, label="baseline trials" if s == 0 else None),
                ))
                extra_c.append((
                    bl_smeans_c[s][None],
                    dict(s=110, color="tab:pink", marker="o", edgecolors="black",
                         linewidths=1.0, zorder=8, alpha=0.9,
                         label=f"baseline mean s={s+1}" if s < 5 else None),
                ))
            bl_gmean_c = sphere_mean_torch(bl_smeans_c, dim=0)
            extra_c.append((
                bl_gmean_c[None],
                dict(s=210, color="tab:pink", marker="*", edgecolors="black",
                     linewidths=1.5, zorder=11, label="baseline global mean"),
            ))

        bl_info = (f"  baseline={score_global_bl:.4f}" if include_baseline and score_global_bl is not None else "")
        _plot_ax(
            all_particles_c,
            title_str=(
                f"Combined — {sensor_desc}\n"
                f"DPnP global {ylabel}={score_global:.4f}{bl_info}  {better}"
            ),
            extra_markers=extra_c,
        )

    # ── 10. Quality vs DPnP steps ─────────────────────────────────────────────
    if plot_quality_vs_steps:
        steps_arr = np.arange(STEPS + 1)
        n_total   = S * T
        mean_sc_N   = {N: score_steps_by_N[N].mean(dim=(1, 2)).numpy() for N in particle_counts}
        stderr_sc_N = {
            N: score_steps_by_N[N].reshape(STEPS + 1, n_total).std(dim=1, unbiased=False).numpy()
               / np.sqrt(n_total)
            for N in particle_counts
        }

        # MCMC baseline: scalar per N → horizontal dashed reference line
        mean_bl_N   = {}
        stderr_bl_N = {}
        if include_baseline and score_steps_by_N_bl:
            mean_bl_N   = {N: float(score_steps_by_N_bl[N].mean())
                           for N in particle_counts}
            stderr_bl_N = {N: float(score_steps_by_N_bl[N].std() / np.sqrt(n_total))
                           for N in particle_counts}

        bl_flat  = list(mean_bl_N.values())
        all_flat = [float(v) for arr in mean_sc_N.values() for v in arr] + bl_flat
        ymin = min(all_flat)
        ymax = max(all_flat)
        pad  = 0.05 * max(1e-8, ymax - ymin)

        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        plt.figure(figsize=(8, 5))
        for i, N in enumerate(particle_counts):
            col = prop_cycle[i % len(prop_cycle)]
            plt.plot(steps_arr, mean_sc_N[N], marker="o", linewidth=2,
                     color=col, label=f"DPnP N={N}")
            if show_stderr:
                plt.fill_between(steps_arr,
                                 mean_sc_N[N] - stderr_sc_N[N],
                                 mean_sc_N[N] + stderr_sc_N[N],
                                 alpha=0.12, color=col)
            if N in mean_bl_N:
                # MCMC has no step axis → horizontal dashed reference line
                plt.axhline(mean_bl_N[N], color=col, linestyle="--",
                            linewidth=1.5, alpha=0.65,
                            label=f"MCMC baseline N={N}")
                if show_stderr:
                    plt.axhspan(mean_bl_N[N] - stderr_bl_N[N],
                                mean_bl_N[N] + stderr_bl_N[N],
                                alpha=0.06, color=col)

        plt.xlabel("DPnP step")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel(f"mean {ylabel}  {better}")
        plt.title(
            f"Two-point seismic: {ylabel} vs DPnP step\n"
            f"{sensor_desc}, averaged over {S}×{T} trials"
            + (" | dashed = MCMC likelihood-only baseline" if include_baseline else "")
        )
        plt.ylim(ymin - pad, ymax + pad)
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.show()

    out = {
        "x_true":           x_true.cpu(),
        "y1_sensors":       y1_sensors.cpu(),      # (S, 3)
        "y2_sensors":       y2_sensors.cpu(),      # (S, 3)
        "x_reflected":      x_reflected.cpu(),     # (S, 3)
        "u1_all":           u1_all.cpu(),           # (S, T)
        "u2_all":           u2_all.cpu(),           # (S, T)
        "X_finals":         X_finals.cpu(),         # (S, T, P, 3)
        "trial_means":      trial_means.cpu(),      # (S, T, 3)
        "sensor_means":     sensor_means.cpu(),     # (S, 3)
        "global_mean":      global_mean.cpu(),      # (3,)
        "score_trials":     score_trials.cpu(),     # (S, T)
        "score_sensors":    score_sensors.cpu(),    # (S,)
        "score_global":     score_global,           # float
        "score_reflected":  score_reflected.cpu(),  # (S,)
        "score_steps_by_N": {N: score_steps_by_N[N].numpy()
                             for N in particle_counts},
        "metric":           metric,
    }
    if include_baseline:
        out.update({
            "X_finals_bl":         X_finals_bl.cpu() if X_finals_bl is not None else None,
            "trial_means_bl":      trial_means_bl.cpu() if trial_means_bl is not None else None,
            "score_trials_bl":     score_trials_bl.cpu() if score_trials_bl is not None else None,
            "score_global_bl":     score_global_bl,
            "score_steps_by_N_bl": {N: score_steps_by_N_bl[N].numpy()
                                    for N in particle_counts},
        })
    return out