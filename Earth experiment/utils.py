import torch
import numpy as np
import math
DEFAULT_DTYPE = torch.float32
EPS64 = 1e-12
EPS32 = 1e-7
sig2 = 0.1

def get_eps(dtype):
    return EPS32 if dtype == torch.float32 else EPS64

def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ============================================================
# S^2 geometry utilities
# ============================================================

def normalize_torch(x: torch.Tensor, eps: float = 1e-12):
    """
    Normalize vectors along the last dimension.

    Supports shapes:
        (d,)
        (N,d)
        (B,P,d)
        (...,d)
    """
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

def tangent_project_torch(x: torch.Tensor, v: torch.Tensor):
    """
    Project v onto the tangent space at x on S^{d-1}.

    Both tensors must have shape (..., d).

    Returns tensor of shape (..., d).
    """
    dot = (x * v).sum(dim=-1, keepdim=True)
    return v - dot * x

def sphere_exp_map_torch(x: torch.Tensor, v: torch.Tensor, eps: float = 1e-12):
    """
    Exponential map on the unit sphere.

    x : (..., d) point on S^{d-1}
    v : (..., d) tangent vector

    Returns:
        (..., d)
    """
    v_norm = v.norm(dim=-1, keepdim=True)

    # safe normalized direction
    v_dir = v / v_norm.clamp_min(eps)

    cos = torch.cos(v_norm)
    sin = torch.sin(v_norm)

    y = cos * x + sin * v_dir

    return normalize_torch(y, eps)

def sphere_log_map_torch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12):
    """
    Logarithmic map on the unit sphere (inverse of the exponential map).

    Returns the tangent vector v ∈ T_x S^{d-1} such that Exp_x(v) = y.

    x : (..., d) base point on S^{d-1}
    y : (..., d) target point on S^{d-1}

    Returns:
        (..., d) tangent vector at x.  Zero when x ≈ y.
    """
    dot = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    y_perp = y - dot * x                                         # component ⊥ x
    y_perp_norm = y_perp.norm(dim=-1, keepdim=True).clamp_min(eps)
    theta = torch.arccos(dot)                                     # geodesic distance
    return theta * y_perp / y_perp_norm


def sphere_distance_torch(x, y, euclidean=True, eps=None):
    if eps is None:
        eps = get_eps(x.dtype)

    x = normalize_torch(x, eps=eps)
    y = normalize_torch(y, eps=eps)

    dot = torch.sum(x * y, dim=-1).clamp(-1.0, 1.0)

    if euclidean:
        return torch.sqrt(torch.clamp(2.0 * (1.0 - dot), min=0.0))
    else:
        sin_theta = torch.linalg.norm(x - dot.unsqueeze(-1) * y, dim=-1)
        return torch.atan2(sin_theta, dot)


def parallel_transport_S2_torch(x, y, V, eps=None):
    if eps is None:
        eps = get_eps(x.dtype)

    x = normalize_torch(x, eps=eps)
    y = normalize_torch(y, eps=eps)

    dot = torch.sum(x * y, dim=-1, keepdim=True)
    denom = torch.clamp(1.0 + dot, min=eps)

    yTV = torch.sum(y.unsqueeze(-1) * V, dim=-2, keepdim=True)
    corr = (yTV / denom.unsqueeze(-1)) * (x + y).unsqueeze(-1)
    W = V - corr

    W = W - torch.sum(y.unsqueeze(-1) * W, dim=-2, keepdim=True) * y.unsqueeze(-1)
    return W


def orthonormalize_frames_torch(U, eps=None):
    if eps is None:
        eps = get_eps(U.dtype)

    v0 = U[..., :, 0]
    n0 = torch.linalg.norm(v0, dim=-1, keepdim=True)
    q0 = v0 / torch.clamp(n0, min=eps)

    v1 = U[..., :, 1]
    proj = torch.sum(q0 * v1, dim=-1, keepdim=True)
    v1 = v1 - proj * q0
    n1 = torch.linalg.norm(v1, dim=-1, keepdim=True)
    q1 = v1 / torch.clamp(n1, min=eps)

    return torch.stack([q0, q1], dim=-1)


def make_tangent_frame_S2_batch_torch(X, eps=None):
    if eps is None:
        eps = get_eps(X.dtype)

    X = normalize_torch(X, eps=eps)

    use_x = torch.abs(X[..., 0]) < 0.9
    A_x = torch.tensor([1.0, 0.0, 0.0], dtype=X.dtype, device=X.device)
    A_y = torch.tensor([0.0, 1.0, 0.0], dtype=X.dtype, device=X.device)
    A = torch.where(use_x.unsqueeze(-1), A_x, A_y)

    e1 = tangent_project_torch(X, A)
    e1 = normalize_torch(e1, eps=eps)

    e2 = torch.cross(X, e1, dim=-1)
    e2 = normalize_torch(e2, eps=eps)

    return torch.stack([e1, e2], dim=-1)
####################### VMF sampler #######################
def tangent_basis_s2(mu: torch.Tensor, eps: float = 1e-12):
    """
    Build an orthonormal tangent basis at each mu on S^2.

    Parameters
    ----------
    mu : (..., 3) tensor

    Returns
    -------
    e1, e2 : (..., 3) tensors
        Orthonormal basis vectors spanning T_mu S^2.
    """
    if mu.shape[-1] != 3:
        raise ValueError("mu must have shape (..., 3)")

    mu = normalize_torch(mu, eps=eps)

    ref_x = torch.tensor([1.0, 0.0, 0.0], device=mu.device, dtype=mu.dtype)
    ref_y = torch.tensor([0.0, 1.0, 0.0], device=mu.device, dtype=mu.dtype)

    # If mu is too aligned with x-axis, use y-axis instead
    use_y = (mu[..., 0].abs() > 0.9)[..., None]
    ref = torch.where(use_y, ref_y.expand_as(mu), ref_x.expand_as(mu))

    e1 = ref - (ref * mu).sum(dim=-1, keepdim=True) * mu
    e1 = normalize_torch(e1, eps=eps)

    e2 = torch.cross(mu, e1, dim=-1)
    e2 = normalize_torch(e2, eps=eps)

    return e1, e2


def _prepare_kappa(kappa, target_shape, device, dtype):
    """
    Convert kappa to tensor broadcastable to target_shape.
    target_shape is the leading shape, i.e. mu.shape[:-1].
    """
    if not torch.is_tensor(kappa):
        kappa = torch.tensor(kappa, device=device, dtype=dtype)
    else:
        kappa = kappa.to(device=device, dtype=dtype)

    try:
        kappa = torch.broadcast_to(kappa, target_shape)
    except RuntimeError as e:
        raise ValueError(
            f"kappa with shape {tuple(kappa.shape)} is not broadcastable "
            f"to target leading shape {tuple(target_shape)}"
        ) from e
    return kappa


@torch.no_grad()
def sample_vmf_s2(
    mu: torch.Tensor,
    kappa,
    n_samples: int | None = None,
    generator=None,
    eps: float = 1e-12,
    keep_sample_dim: bool = False,
) -> torch.Tensor:
    """
    Sample from the von Mises-Fisher distribution on S^2:
        p(x | mu, kappa) ∝ exp(kappa * <mu, x>)

    Supports mu with shape:
        (3,)
        (N, 3)
        (B, P, 3)
        (..., 3)

    Parameters
    ----------
    mu : (..., 3) tensor
        Mean direction(s) on S^2.
    kappa : scalar or tensor broadcastable to mu.shape[:-1]
        Concentration parameter(s).
    n_samples : int or None
        Number of samples per mu.
        If None, returns one sample per mu with output shape (..., 3).
        If integer m, returns shape (..., m, 3), unless
        m == 1 and keep_sample_dim=False, in which case returns (..., 3).
    generator : torch.Generator, optional
    eps : float
    keep_sample_dim : bool
        Whether to keep the sample dimension when n_samples == 1.

    Returns
    -------
    x : tensor
        Shape (..., 3) or (..., n_samples, 3).
    """
    if mu.shape[-1] != 3:
        raise ValueError("mu must have shape (..., 3)")

    mu = normalize_torch(mu, eps=eps)
    device, dtype = mu.device, mu.dtype
    lead_shape = mu.shape[:-1]

    if n_samples is None:
        n_samples = 1
        squeeze_output = True and (not keep_sample_dim)
    else:
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        squeeze_output = (n_samples == 1) and (not keep_sample_dim)

    kappa = _prepare_kappa(kappa, lead_shape, device, dtype)

    # Shapes:
    #   mu         : (..., 3)
    #   e1, e2     : (..., 3)
    #   phi, u, w  : (..., n_samples)
    #   x          : (..., n_samples, 3)

    phi = 2.0 * math.pi * torch.rand(
        *lead_shape, n_samples, device=device, dtype=dtype, generator=generator
    )
    u = torch.rand(
        *lead_shape, n_samples, device=device, dtype=dtype, generator=generator
    )

    kappa_exp = kappa[..., None]  # (..., 1)
    small = torch.abs(kappa_exp) < 1e-8

    # For kappa = 0, uniform on sphere
    w_uniform = 2.0 * u - 1.0

    # Inverse CDF for vMF on S^2:
    #   F(w) = (exp(kappa w) - exp(-kappa)) / (exp(kappa) - exp(-kappa))
    # so
    #   w = -1 + log(1 + u * (exp(2kappa)-1)) / kappa
    w_vmf = -1.0 + torch.log1p(u * torch.expm1(2.0 * kappa_exp)) / kappa_exp

    w = torch.where(small, w_uniform, w_vmf).clamp(-1.0, 1.0)
    sin_theta = torch.sqrt((1.0 - w**2).clamp_min(0.0))

    e1, e2 = tangent_basis_s2(mu, eps=eps)  # (..., 3)

    tangent_dir = (
        torch.cos(phi)[..., None] * e1[..., None, :]
        + torch.sin(phi)[..., None] * e2[..., None, :]
    )  # (..., n_samples, 3)

    x = w[..., None] * mu[..., None, :] + sin_theta[..., None] * tangent_dir
    x = normalize_torch(x, eps=eps)

    if squeeze_output:
        x = x.squeeze(-2)

    return x


def _kent_frame(mu: torch.Tensor, eps: float = 1e-12):
    """
    Compute a right-handed orthonormal frame (γ₁, γ₂) in the tangent plane of μ.
    γ₁ is chosen perpendicular to μ by Gram-Schmidt against the coordinate axis
    with the smallest absolute projection on μ.  γ₂ = μ × γ₁.

    Parameters
    ----------
    mu : (..., 3) tensor, unit vectors on S²

    Returns
    -------
    gamma1, gamma2 : (..., 3) tensors — orthonormal, tangent to S² at μ
    """
    mu = normalize_torch(mu, eps=eps)
    # Axis least aligned with mu
    idx = mu.abs().argmin(dim=-1, keepdim=True)        # (..., 1)
    e = torch.zeros_like(mu)
    e.scatter_(-1, idx, 1.0)                           # canonical axis
    # Gram-Schmidt
    gamma1 = normalize_torch(e - (e * mu).sum(-1, keepdim=True) * mu, eps=eps)
    gamma2 = torch.linalg.cross(mu, gamma1, dim=-1)
    return gamma1, gamma2


@torch.no_grad()
def sample_kent_s2(
    mu: torch.Tensor,
    kappa: float,
    beta: float,
    n_samples: int | None = None,
    generator=None,
    eps: float = 1e-12,
    keep_sample_dim: bool = False,
    max_rejection_iters: int = 500,
) -> torch.Tensor:
    """
    Sample from the Kent (FB5) distribution on S²:
        p(x | μ, κ, β) ∝ exp(κ⟨μ,x⟩ + β(⟨γ₁,x⟩² − ⟨γ₂,x⟩²))

    Uses rejection sampling with vMF(μ, κ) as the envelope.
    Since the Kent/vMF ratio is exp(β(⟨γ₁,x⟩²−⟨γ₂,x⟩²)) ≤ exp(β),
    the normalised acceptance probability is:
        p(accept | x) = exp(β(⟨γ₁,x⟩² − ⟨γ₂,x⟩²) − β)   ∈ (0, 1]

    Requires 0 ≤ 2β < κ for the distribution to be proper.

    Parameters
    ----------
    mu        : (..., 3) tensor — mean direction(s) on S²
    kappa     : float           — concentration κ  (κ > 0)
    beta      : float           — ovalness β       (0 ≤ 2β < κ)
    n_samples : int or None     — samples per mu  (None → one per mu)
    generator : torch.Generator, optional
    eps       : float
    keep_sample_dim : bool
    max_rejection_iters : int   — safety cap on rejection loop

    Returns
    -------
    x : (..., 3) or (..., n_samples, 3)
    """
    if beta < 0:
        raise ValueError("beta must be >= 0")
    if 2.0 * beta >= kappa:
        raise ValueError(
            f"Kent distribution requires 2*beta < kappa, got 2*beta={2*beta}, kappa={kappa}"
        )

    mu = normalize_torch(mu, eps=eps)
    gamma1, gamma2 = _kent_frame(mu, eps=eps)
    device, dtype = mu.device, mu.dtype
    lead_shape = mu.shape[:-1]

    if n_samples is None:
        n_samples = 1
        squeeze_output = not keep_sample_dim
    else:
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        squeeze_output = (n_samples == 1) and (not keep_sample_dim)

    # Flatten leading dims to (B, 3) for rejection loop
    B = int(math.prod(lead_shape)) if lead_shape else 1
    mu_flat     = mu.reshape(B, 3)
    gamma1_flat = gamma1.reshape(B, 3)
    gamma2_flat = gamma2.reshape(B, 3)

    # collected[b] = list of accepted (k, 3) tensors; counts[b] = accepted so far
    collected = [[] for _ in range(B)]
    counts    = [0]   * B

    for _ in range(max_rejection_iters):
        if all(c >= n_samples for c in counts):
            break
        # How many more we need in the worst case
        needed_max = max(n_samples - c for c in counts)
        # Oversample 4× to keep the loop short
        n_draw = needed_max * 4

        # Draw from vMF(μ_flat, κ) for all B simultaneously
        cands = sample_vmf_s2(
            mu_flat, kappa,
            n_samples=n_draw,
            generator=generator,
            keep_sample_dim=True,
        )   # (B, n_draw, 3)

        # Acceptance log-probability
        c1 = (cands * gamma1_flat[:, None, :]).sum(-1)   # (B, n_draw)
        c2 = (cands * gamma2_flat[:, None, :]).sum(-1)
        log_accept = beta * (c1 ** 2 - c2 ** 2) - beta   # ≤ 0

        u = torch.rand(
            log_accept.shape, device=device, dtype=dtype, generator=generator
        )
        accept_mask = u < log_accept.exp()                # (B, n_draw) bool

        for b in range(B):
            if counts[b] >= n_samples:
                continue
            accepted_b = cands[b][accept_mask[b]]         # (k, 3)
            need_more  = n_samples - counts[b]
            take       = min(accepted_b.shape[0], need_more)
            if take > 0:
                collected[b].append(accepted_b[:take])
                counts[b] += take

    # Assemble result
    result_parts = []
    for b in range(B):
        if collected[b]:
            samp_b = torch.cat(collected[b], dim=0)[:n_samples]   # (n_samples, 3)
        else:
            # Fallback: use plain vMF (should not happen in practice)
            # mu_flat[b] is (3,), so output is (n_samples, 3)
            samp_b = sample_vmf_s2(
                mu_flat[b], kappa, n_samples=n_samples,
                generator=generator, keep_sample_dim=True,
            )   # (n_samples, 3)
        result_parts.append(samp_b)

    result = torch.stack(result_parts, dim=0)            # (B, n_samples, 3)
    result = result.reshape(*lead_shape, n_samples, 3)
    result = normalize_torch(result, eps=eps)

    if squeeze_output:
        result = result.squeeze(-2)
    return result


####################################################################
@torch.no_grad()
def sample_geodesic_gaussian_on_sphere(
    x: torch.Tensor,
    sigma=None,
    generator=None,
) -> torch.Tensor:
    """
    Sample y ~ exp_x(N(0, sigma)) on S^2.

    Supports x with shape:
        (3,)
        (N,3)
        (B,P,3)
        (...,3)

    sigma options:
        None          -> 0.05 * I_3
        scalar        -> isotropic variance
        (3,3)         -> covariance matrix
        (...,3,3)     -> batched covariance
    """

    x = normalize_torch(x)

    d = x.shape[-1]
    batch_shape = x.shape[:-1]

    # default covariance
    if sigma is None:
        sigma = sig2 * torch.eye(d, device=x.device, dtype=x.dtype)
    else:
        sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)

    # ambient Gaussian noise
    z = torch.randn(
        *batch_shape, d,
        device=x.device,
        dtype=x.dtype,
        generator=generator,
    )

    v = tangent_project_torch(x, z)

    # apply covariance
    if sigma.ndim == 0:
        v = sigma * v
    elif sigma.ndim == 2:
        v = torch.matmul(v, sigma.T)
    else:
        v = torch.matmul(v.unsqueeze(-2), sigma).squeeze(-2)

    y = sphere_exp_map_torch(x, v)

    return normalize_torch(y)

# ============================================================
# Default f_fn examples
# ============================================================

def gaussian_kernel_f(X, y, sig2=sig2, euclidean=False):
    """
    X: (B,P,3)
    y: (3,) or broadcastable to (B,P,3)
    returns: (B,P)
    """
    dist = sphere_distance_torch(X, y, euclidean=euclidean)
    return torch.exp((-1 / (2 * sig2)) * dist * dist)

def get_gaussian_kernel_f(sig2):
    def f(X,y):
        return gaussian_kernel_f(X,y, sig2=sig2, euclidean = False)
    return f

def vmf_s2_unnormalized_pdf(
    x: torch.Tensor,
    mu: torch.Tensor,
    kappa,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Unnormalized vMF density on S^2:
        f(x) = exp(kappa * <mu, x>)
    """
    if x.shape[-1] != 3 or mu.shape[-1] != 3:
        raise ValueError("x and mu must both have shape (..., 3)")

    x = normalize_torch(x, eps=eps)
    mu = normalize_torch(mu, eps=eps)

    dot = (x * mu).sum(dim=-1)

    if not torch.is_tensor(kappa):
        kappa = torch.tensor(kappa, device=dot.device, dtype=dot.dtype)
    else:
        kappa = kappa.to(device=dot.device, dtype=dot.dtype)

    return torch.exp(kappa * dot)


def get_vmf_f(kappa):
    def f(x,y):
        return vmf_s2_unnormalized_pdf(x=y,mu=x,kappa=kappa)
    return f


# ============================================================
# Seismic / "two-point" inverse problem likelihood kernels
# ============================================================

def seismic_signal(x: torch.Tensor, y: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Seismic signal model: I(x, y) = exp(beta * (<x, y> - 1))

    Maximum I = 1 when x = y; decays smoothly as geodesic distance grows.

    x, y : (..., 3) tensors on S^2 (need not be unit-normalised on input).
    Returns (...) tensor of values in (0, 1].
    """
    ip = (normalize_torch(x) * normalize_torch(y)).sum(dim=-1)
    return torch.exp(beta * (ip - 1.0))


def get_seismic_f_fn(y_sensor, u_obs, beta: float, sigma2: float):
    """
    Single-sensor seismic likelihood kernel for use with get_bel.

    Implements:
        f(x) = exp( -(u - I(x, y))^2 / (2 sigma^2) )
        I(x, y) = exp(beta * (<x, y> - 1))

    Parameters
    ----------
    y_sensor : array-like, shape (3,)
        Sensor location on S^2.
    u_obs : float
        Scalar observation.
    beta : float
        Signal decay rate (beta > 0).
    sigma2 : float
        Observation noise variance.

    Returns
    -------
    f_fn : callable
        f_fn(X, dummy_y) -> tensor matching the leading dimensions of X
        with the last spatial dimension reduced.
        Compatible with bel_gradlog_u_S2_batch_chunked_torch (X shape: (B, P, 3)).
    """
    def f_fn(X: torch.Tensor, dummy_y: torch.Tensor) -> torch.Tensor:
        y_t = torch.as_tensor(y_sensor, device=X.device, dtype=X.dtype)
        u_t = torch.as_tensor(u_obs, device=X.device, dtype=X.dtype)
        ip = (X * y_t).sum(dim=-1)          # (...) via broadcasting (3,) onto last dim
        I = torch.exp(beta * (ip - 1.0))
        return torch.exp(-(u_t - I) ** 2 / (2.0 * sigma2))
    return f_fn


def get_two_point_seismic_f_fn(
    y1, y2, u1_obs, u2_obs, beta: float, sigma2: float
):
    """
    Two-sensor seismic likelihood kernel for use with get_bel.

    Joint likelihood for observations (u1, y1) and (u2, y2):
        f(x) = exp( -(u1 - I(x,y1))^2 / (2 sigma^2) )
             * exp( -(u2 - I(x,y2))^2 / (2 sigma^2) )
        I(x, y) = exp(beta * (<x, y> - 1))

    Parameters
    ----------
    y1, y2 : array-like
        Sensor locations on S^2.
        - Shape (3,)   : one shared sensor for all batch items.
        - Shape (K, 3) : per-item sensors (K must match len(u1_obs)).
    u1_obs, u2_obs : float or 1-D tensor of shape (K,)
        Observations at each sensor.
        - Scalar : single trial; f_fn works for any DPnP batch size B.
        - Shape (K,) : K batched trials; DPnP must be called with batch size B = K.
          BEL receives X of shape (K*P, n_bel, 3) and recovers P = (K*P) // K.
    beta : float
        Signal decay rate.
    sigma2 : float
        Observation noise variance.

    Returns
    -------
    f_fn : callable
        f_fn(X, dummy_y) -> (B_total, n_bel) tensor.
        Compatible with bel_gradlog_u_S2_batch_chunked_torch.
    """
    u1_t = torch.as_tensor(u1_obs) if not torch.is_tensor(u1_obs) else u1_obs.detach().cpu()
    u2_t = torch.as_tensor(u2_obs) if not torch.is_tensor(u2_obs) else u2_obs.detach().cpu()
    batched = u1_t.dim() > 0 and u1_t.numel() > 1
    K = int(u1_t.numel()) if batched else None

    # Pre-convert sensors; keep on CPU and move inside f_fn
    y1_t = torch.as_tensor(y1).float().detach().cpu()
    y2_t = torch.as_tensor(y2).float().detach().cpu()
    y1_per_item = y1_t.dim() == 2  # True when shape is (K, 3)
    y2_per_item = y2_t.dim() == 2

    def f_fn(X: torch.Tensor, dummy_y: torch.Tensor) -> torch.Tensor:
        dev, dt = X.device, X.dtype
        u1 = u1_t.to(device=dev, dtype=dt)
        u2 = u2_t.to(device=dev, dtype=dt)
        _y1 = y1_t.to(device=dev, dtype=dt)
        _y2 = y2_t.to(device=dev, dtype=dt)

        B_total = X.shape[0]

        if batched:
            # X: (K * P_dpnp, n_bel, 3)
            P_dpnp = B_total // K

            # ---- sensor inner products ----
            if y1_per_item:
                # _y1: (K, 3) → (K, P_dpnp, 1, 3) → (B_total, 1, 3)
                y1_exp = _y1.unsqueeze(1).expand(K, P_dpnp, 3).reshape(B_total, 1, 3)
                ip1 = (X * y1_exp).sum(dim=-1)          # (B_total, n_bel)
            else:
                ip1 = (X * _y1).sum(dim=-1)             # broadcast (3,)

            if y2_per_item:
                y2_exp = _y2.unsqueeze(1).expand(K, P_dpnp, 3).reshape(B_total, 1, 3)
                ip2 = (X * y2_exp).sum(dim=-1)
            else:
                ip2 = (X * _y2).sum(dim=-1)

            # ---- observation broadcast ----
            u1_exp = u1.unsqueeze(1).expand(K, P_dpnp).reshape(B_total, 1)
            u2_exp = u2.unsqueeze(1).expand(K, P_dpnp).reshape(B_total, 1)
        else:
            ip1 = (X * _y1).sum(dim=-1)
            ip2 = (X * _y2).sum(dim=-1)
            u1_exp = u1
            u2_exp = u2

        I1 = torch.exp(beta * (ip1 - 1.0))
        I2 = torch.exp(beta * (ip2 - 1.0))
        like1 = torch.exp(-(u1_exp - I1) ** 2 / (2.0 * sigma2))
        like2 = torch.exp(-(u2_exp - I2) ** 2 / (2.0 * sigma2))
        return like1 * like2

    return f_fn


# =========================================================
# Geometry helpers on S^2
# =========================================================
def extrinsic_to_latlon_deg_torch(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = normalize_torch(x, eps)
    xx, yy, zz = x[..., 0], x[..., 1], x[..., 2]
    lat = torch.rad2deg(torch.arcsin(zz.clamp(-1.0, 1.0)))
    lon = torch.rad2deg(torch.atan2(yy, xx))
    return torch.stack([lat, lon], dim=-1)


def wrap_longitude_relative(lon_deg: np.ndarray, center_deg: float) -> np.ndarray:
    return ((lon_deg - center_deg + 180.0) % 360.0) - 180.0 + center_deg


def great_circle_interp_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    n_points: int = 80,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Spherical linear interpolation between two points on S^2.
    a, b: (3,)
    returns: (n_points, 3)
    """
    a = normalize_torch(a, eps)
    b = normalize_torch(b, eps)

    dot = (a * b).sum().clamp(-1.0, 1.0)
    omega = torch.arccos(dot)

    if omega.abs() < 1e-8:
        return a[None, :].repeat(n_points, 1)

    t = torch.linspace(0.0, 1.0, n_points, device=a.device, dtype=a.dtype)
    so = torch.sin(omega)
    pts = (
        torch.sin((1 - t) * omega)[:, None] / so * a[None, :]
        + torch.sin(t * omega)[:, None] / so * b[None, :]
    )
    return normalize_torch(pts, eps)


def sphere_mean_torch(x: torch.Tensor, dim: int = -2, eps: float = 1e-12) -> torch.Tensor:
    """
    Extrinsic mean projected back to S^2.
    x: (..., P, 3)
    """
    return normalize_torch(x.mean(dim=dim), eps)


def extrinsic_to_latlon_rad_torch(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    x: (..., 3) on S^2
    returns (..., 2) = [lat_rad, lon_rad]
    """
    x = normalize_torch(x, eps)
    xx, yy, zz = x[..., 0], x[..., 1], x[..., 2]
    lat = torch.arcsin(zz.clamp(-1.0, 1.0))
    lon = torch.atan2(yy, xx)
    return torch.stack([lat, lon], dim=-1)


def spherical_kde_vmf(
    samples: torch.Tensor,
    eval_points: torch.Tensor,
    kappa: float = 25.0,
    normalize: bool = True,
) -> torch.Tensor:
    """
    von Mises-Fisher kernel density estimate on S^2.

    samples     : (N,3)
    eval_points : (M,3)
    returns     : (M,)

    Kernel:
        K_kappa(z, x) ~ exp(kappa * <z, x>)

    If normalize=True, includes the S^2 vMF normalizing constant:
        c(kappa) = kappa / (4*pi*sinh(kappa))
    """
    samples = normalize_torch(samples)
    eval_points = normalize_torch(eval_points)

    dots = eval_points @ samples.T   # (M,N)
    vals = torch.exp(kappa * dots)

    if normalize:
        c_kappa = kappa / (4.0 * np.pi * np.sinh(kappa))
        vals = c_kappa * vals

    return vals.mean(dim=1)

# =========================================================
# Data sampling helper
# =========================================================

@torch.no_grad()
def sample_xtrue_y_batches_from_dataloader(
    dataloader,
    sigma_y: float,
    num_pairs: int,
    device="mps",
    dtype=torch.float32,
    seed: int = 0,
):
    """
    Uniformly sample num_pairs x_true from the dataset underlying the dataloader.
    Correctly handles torch.utils.data.Subset, so if dataloader is test_loader,
    sampling is restricted to the test split.

    Then sample y | x_true from vmf
    """
    device = torch.device(device)

    # CPU generator for index sampling
    gen_cpu = torch.Generator(device="cpu")
    gen_cpu.manual_seed(int(seed))

    ds = dataloader.dataset

    if hasattr(ds, "dataset") and hasattr(ds, "indices"):
        base_dataset = ds.dataset
        subset_indices = torch.as_tensor(ds.indices, dtype=torch.long)  # stays on CPU
        N = subset_indices.shape[0]

        if num_pairs > N:
            raise ValueError(f"num_pairs={num_pairs} exceeds subset size {N}")

        chosen_subset_pos = torch.randperm(N, generator=gen_cpu)[:num_pairs]
        chosen_base_idx = subset_indices[chosen_subset_pos].tolist()

        xs = []
        for idx in chosen_base_idx:
            item = base_dataset[idx]
            x = item[0] if isinstance(item, (tuple, list)) else item
            x = torch.as_tensor(x, device=device, dtype=dtype)
            xs.append(x)

        x_true_all = normalize_torch(torch.stack(xs, dim=0))

    else:
        N = len(ds)

        if num_pairs > N:
            raise ValueError(f"num_pairs={num_pairs} exceeds dataset size {N}")

        chosen_idx = torch.randperm(N, generator=gen_cpu)[:num_pairs].tolist()

        xs = []
        for idx in chosen_idx:
            item = ds[idx]
            x = item[0] if isinstance(item, (tuple, list)) else item
            x = torch.as_tensor(x, device=device, dtype=dtype)
            xs.append(x)

        x_true_all = normalize_torch(torch.stack(xs, dim=0))

    # Separate generator for sphere sampling on target device
    gen_device = torch.Generator(device=device)
    gen_device.manual_seed(int(seed) + 1)

    y_all = sample_vmf_s2(
        mu=x_true_all,
        kappa=sigma_y,
        generator=gen_device,
    )
    y_all = normalize_torch(y_all)

    return x_true_all, y_all

# =========================================================
# Plotting-coordinate helpers
# =========================================================

def extrinsic_to_mollweide_rad_torch(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Convert extrinsic S^2 points to plotting coordinates for matplotlib Mollweide.

    Returns (..., 2) = [plot_lat_rad, plot_lon_rad]
    where plot_lat_rad includes the sign flip needed for this dataset.
    """
    ll = extrinsic_to_latlon_rad_torch(x, eps=eps)
    out = ll.clone()
    out[..., 0] = -out[..., 0]   # flip latitude for plotting only
    return out


def mollweide_plot_grid_xyz_torch(
    n_lon: int = 240,
    n_lat: int = 120,
    device=None,
    dtype=torch.float32,
):
    """
    Build a Mollweide plotting grid and the corresponding physical S^2 points.

    Returns
    -------
    Lon_plot : (n_lon, n_lat)
    Lat_plot : (n_lon, n_lat)
    grid_xyz : (n_lon*n_lat, 3)

    Important:
    plot_lat = - physical_lat
    so the physical z-coordinate is z = sin(physical_lat) = -sin(plot_lat).
    """
    lon = torch.linspace(-np.pi, np.pi, n_lon, device=device, dtype=dtype)
    lat_plot = torch.linspace(-0.5 * np.pi, 0.5 * np.pi, n_lat, device=device, dtype=dtype)

    Lon_plot, Lat_plot = torch.meshgrid(lon, lat_plot, indexing="xy")

    xg = torch.cos(Lat_plot) * torch.cos(Lon_plot)
    yg = torch.cos(Lat_plot) * torch.sin(Lon_plot)
    zg = -torch.sin(Lat_plot)

    grid_xyz = torch.stack([xg, yg, zg], dim=-1).reshape(-1, 3)
    return Lon_plot, Lat_plot, grid_xyz


# ============================================================
# Batched MCMC sampler on S^2 (geodesic random-walk or MALA)
# ============================================================

def _riemannian_grad_log_target(x, log_target_fn):
    """
    Compute the Riemannian gradient grad_{S^2} log q(x) via autograd.

    Returns (grad, log_q) where grad ∈ T_x S^2 and log_q = log q(x).
    """
    with torch.enable_grad():
        x_req = x.detach().requires_grad_(True)
        log_q = log_target_fn(x_req)
        ambient_grad = torch.autograd.grad(log_q.sum(), x_req)[0]
    return tangent_project_torch(x, ambient_grad.detach()), log_q.detach()


@torch.no_grad()
def mcmc_mh_s2_batched(
    log_target_fn,
    B: int,
    n_samples: int,
    tau: float = 0.1,
    n_burnin: int = 500,
    x_init: torch.Tensor | None = None,
    use_mala: bool = False,
    device="cpu",
    dtype=torch.float32,
    seed: int = 0,
) -> torch.Tensor:
    """
    Batched MCMC sampler on S^2 with geodesic proposals.

    Supports two modes:

    **Random walk** (use_mala=False, default):
        v = τ (I − x xᵀ) z,   z ~ N(0, I₃)
        x* = Exp_x(v)
        α  = min{1, q(x*)/q(x)}        (symmetric proposal)

    **MALA** (use_mala=True):
        g  = grad_{S²} log q(x)         (via autograd)
        ξ  = (I − x xᵀ) z,  z ~ N(0, I₃)
        v  = τ g + √(2τ) ξ
        x* = Exp_x(v)
        α  = min{1, q(x*) q_rev(x|x*) / [q(x) q_fwd(x*|x)]}

    Runs B independent chains in parallel.

    Parameters
    ----------
    log_target_fn : callable
        log_target_fn(x) where x has shape (B, 3) → (B,).
        Must be differentiable w.r.t. x when use_mala=True.
    B             : int   -- number of independent chains
    n_samples     : int   -- samples to keep per chain (after burn-in)
    tau           : float -- step size (geodesic scale for RW; η for MALA)
    n_burnin      : int   -- burn-in steps discarded before recording
    x_init        : (B, 3) or (3,) tensor, or None
                    Initial chain positions.  If None, initialise uniformly
                    on S².  Providing a point near the high-density region
                    (e.g. a sensor location) dramatically improves mixing.
    use_mala      : bool  -- if True, use manifold MALA instead of random walk
    device, dtype, seed

    Returns
    -------
    samples : (n_samples, B, 3) tensor on S^2
    accept_rate : float   -- mean acceptance rate over all steps and chains
    """
    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # ── Initialise chains ────────────────────────────────────────────────
    if x_init is not None:
        x_init_t = normalize_torch(
            torch.as_tensor(x_init, device=device, dtype=dtype)
        )
        if x_init_t.dim() == 1:
            x_init_t = x_init_t.unsqueeze(0).expand(B, 3)
        x = x_init_t.clone()
    else:
        x = normalize_torch(
            torch.randn(B, 3, device=device, dtype=dtype, generator=gen)
        )

    # ── Pre-compute initial state ────────────────────────────────────────
    if use_mala:
        grad_x, log_q = _riemannian_grad_log_target(x, log_target_fn)
    else:
        log_q = log_target_fn(x)                               # (B,)

    samples = torch.empty(n_samples, B, 3, device=device, dtype=dtype)
    n_total  = n_burnin + n_samples
    accepted_total = 0.0

    for step in range(n_total):
        # ── Tangent noise ξ ~ N(0, P_x) ─────────────────────────────────
        z = torch.randn(B, 3, device=device, dtype=dtype, generator=gen)
        xi = tangent_project_torch(x, z)                       # (B, 3)

        if use_mala:
            # ── MALA proposal ────────────────────────────────────────────
            #   v = η grad_{S²} log q(x)  +  √(2η) ξ
            v = tau * grad_x + math.sqrt(2.0 * tau) * xi
            x_prop = sphere_exp_map_torch(x, v)                # (B, 3)

            grad_prop, log_q_prop = _riemannian_grad_log_target(
                x_prop, log_target_fn
            )

            # MH correction for the asymmetric proposal
            v_fwd = sphere_log_map_torch(x, x_prop)            # Log_x(x*)
            v_rev = sphere_log_map_torch(x_prop, x)            # Log_{x*}(x)

            fwd_sq = (v_fwd - tau * grad_x).pow(2).sum(dim=-1)
            rev_sq = (v_rev - tau * grad_prop).pow(2).sum(dim=-1)

            log_alpha = (
                (log_q_prop - log_q)
                + (fwd_sq - rev_sq) / (4.0 * tau)
            ).clamp(max=0.0)
        else:
            # ── Geodesic random-walk proposal ────────────────────────────
            x_prop = sphere_exp_map_torch(x, tau * xi)         # (B, 3)
            log_q_prop = log_target_fn(x_prop)                 # (B,)
            log_alpha = (log_q_prop - log_q).clamp(max=0.0)

        # ── Accept / reject ──────────────────────────────────────────────
        u = torch.rand(B, device=device, dtype=dtype, generator=gen)
        accept = u < log_alpha.exp()                            # (B,)

        x     = torch.where(accept.unsqueeze(-1), x_prop, x)
        log_q = torch.where(accept, log_q_prop, log_q)
        if use_mala:
            grad_x = torch.where(accept.unsqueeze(-1), grad_prop, grad_x)

        accepted_total += accept.float().mean().item()

        if step >= n_burnin:
            samples[step - n_burnin] = x

    accept_rate = accepted_total / n_total
    return samples, accept_rate


def get_two_point_seismic_log_target(
    y1: torch.Tensor,
    y2: torch.Tensor,
    u1_obs: torch.Tensor,
    u2_obs: torch.Tensor,
    beta: float,
    sigma2: float,
):
    """
    Batched log-target for the two-point seismic likelihood baseline.

        log q(x_b) = log p(u1_b | x_b, y1_b) + log p(u2_b | x_b, y2_b)
                   = -(u1_b - I(x_b,y1_b))^2/(2σ²)
                     -(u2_b - I(x_b,y2_b))^2/(2σ²)

    Parameters
    ----------
    y1, y2    : (B, 3) sensor locations (one per chain)
    u1_obs, u2_obs : (B,) scalar observations (one per chain)
    beta, sigma2   : seismic model parameters

    Returns
    -------
    log_target_fn : callable  (B, 3) → (B,)
    """
    _y1  = y1.detach().clone()
    _y2  = y2.detach().clone()
    _u1  = u1_obs.detach().clone()
    _u2  = u2_obs.detach().clone()

    def log_target_fn(x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3)
        y1_d = _y1.to(device=x.device, dtype=x.dtype)
        y2_d = _y2.to(device=x.device, dtype=x.dtype)
        u1_d = _u1.to(device=x.device, dtype=x.dtype)
        u2_d = _u2.to(device=x.device, dtype=x.dtype)

        x_n = normalize_torch(x)
        ip1 = (x_n * normalize_torch(y1_d)).sum(dim=-1)   # (B,)
        ip2 = (x_n * normalize_torch(y2_d)).sum(dim=-1)

        I1 = torch.exp(beta * (ip1 - 1.0))
        I2 = torch.exp(beta * (ip2 - 1.0))

        ll1 = -(u1_d - I1) ** 2 / (2.0 * sigma2)
        ll2 = -(u2_d - I2) ** 2 / (2.0 * sigma2)
        return ll1 + ll2

    return log_target_fn

