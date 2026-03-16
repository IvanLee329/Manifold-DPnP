import torch
from utils import (
    gaussian_kernel_f,
    normalize_torch,
    make_tangent_frame_S2_batch_torch,
    parallel_transport_S2_torch,
    orthonormalize_frames_torch,
    sphere_exp_map_torch,
    get_eps,
    DEFAULT_DTYPE,
    get_best_device,
)

# ============================================================
# utilities
# ============================================================

def _flatten_points_lastdim(x: torch.Tensor):
    """
    x: (..., d)
    returns:
        x_flat: (N, d)
        lead_shape: original leading shape
    """
    if x.ndim < 1:
        raise ValueError("x must have at least 1 dimension")
    d = x.shape[-1]
    lead_shape = x.shape[:-1]
    x_flat = x.reshape(-1, d)
    return x_flat, lead_shape


def _expand_y_for_x(y: torch.Tensor, x_shape, dtype, device):
    """
    Expand y to match x_shape = (..., 3).

    Supported y shapes:
        (3,)
        (B,3)
        (B,P,3)
        (...,3) exactly matching x_shape
    """
    y = torch.as_tensor(y, dtype=dtype, device=device)

    if len(x_shape) < 2:
        raise ValueError("x_shape must correspond to (..., 3)")

    lead_shape = x_shape[:-1]
    d = x_shape[-1]
    if d != 3:
        raise ValueError(f"Expected ambient dimension 3, got {d}")

    if y.ndim == 1:
        if y.shape[0] != d:
            raise ValueError(f"y must have shape (3,), got {tuple(y.shape)}")
        return y.view(*([1] * len(lead_shape)), d).expand(*lead_shape, d)

    if y.shape[-1] != d:
        raise ValueError(f"Last dimension of y must be 3, got {tuple(y.shape)}")

    if tuple(y.shape) == tuple(x_shape):
        return y

    # Special case: x is (B,P,3), y is (B,3)
    if len(lead_shape) == 2 and y.ndim == 2 and y.shape[0] == lead_shape[0]:
        return y[:, None, :].expand(lead_shape[0], lead_shape[1], d)

    # Special case: x is (B,3), y is (B,3)
    if len(lead_shape) == 1 and y.ndim == 2 and y.shape[0] == lead_shape[0]:
        return y

    raise ValueError(
        f"Could not broadcast y with shape {tuple(y.shape)} to x shape {tuple(x_shape)}"
    )



# ============================================================
#  batched BEL estimator for X0 shape (B,3)
# ============================================================
@torch.no_grad()
def bel_gradlog_u_S2_batch_chunked_torch(
    X0, y, t,
    f_fn=gaussian_kernel_f,
    n_paths=8000,
    n_steps=5,
    generator=None,
    device=None,
    dtype=DEFAULT_DTYPE,
    reorthonormalize_every=1,
    f_kwargs=None,
    grad_only=True,
):
    """
    BEL estimator (all paths simulated at once).

    This keeps the same function signature so existing code
    (like the DPnP sampler) continues to work.
    """

    if f_kwargs is None:
        f_kwargs = {}

    if device is None:
        device = get_best_device()

    X0 = torch.as_tensor(X0, dtype=dtype, device=device)
    y = torch.as_tensor(y, dtype=dtype, device=device)

    if X0.ndim != 2 or X0.shape[-1] != 3:
        raise ValueError(f"X0 must have shape (B,3), got {tuple(X0.shape)}")

    eps = get_eps(dtype)

    X0 = normalize_torch(X0, eps=eps)
    U0 = make_tangent_frame_S2_batch_torch(X0, eps=eps)

    B = X0.shape[0]
    P = int(n_paths)

    # broadcast y
    if y.ndim == 1:
        y = y[None, None, :].expand(B, P, 3)
    elif y.ndim == 2:
        y = y[:, None, :].expand(B, P, 3)
    else:
        raise ValueError("y must have shape (3,) or (B,3)")

    t = torch.as_tensor(t, dtype=dtype, device=device)
    dt = t / n_steps
    sqrt_dt = torch.sqrt(dt)

    # initial states
    X = X0[:, None, :].expand(B, P, 3).clone()
    U = U0[:, None, :, :].expand(B, P, 3, 2).clone()

    I = torch.zeros(B, P, 2, dtype=dtype, device=device)

    for k in range(n_steps):

        tk = (k + 0.5) * dt
        M_scalar = torch.exp(-0.5 * tk)

        dW = torch.randn(
            B, P, 2,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        dW.mul_(sqrt_dt)

        # tangent increment
        V = U[..., 0] * dW[..., 0:1]
        V.add_(U[..., 1] * dW[..., 1:2])

        X_new = sphere_exp_map_torch(X, V, eps=eps)
        U_new = parallel_transport_S2_torch(X, X_new, U, eps=eps)

        if reorthonormalize_every > 0 and ((k + 1) % reorthonormalize_every == 0):
            U_new = orthonormalize_frames_torch(U_new, eps=eps)

        I.add_(M_scalar * dW)

        X = X_new
        U = U_new

    # evaluate likelihood kernel
    fvals = f_fn(X, y, **f_kwargs)   # (B,P)

    if fvals.shape != (B, P):
        raise ValueError(
            f"f_fn must return shape {(B, P)}, but got {tuple(fvals.shape)}"
        )

    sum_f = fvals.sum(dim=1)                        # (B,)
    sum_fI = (fvals.unsqueeze(-1) * I).sum(dim=1)   # (B,2)

    n_paths_t = torch.as_tensor(float(n_paths), dtype=dtype, device=device)

    u_hat = sum_f / n_paths_t

    denom = t * sum_f.unsqueeze(-1) + 1e-30
    gradlog_intr = sum_fI / denom

    gradlog = U0[..., 0] * gradlog_intr[..., 0:1]
    gradlog.add_(U0[..., 1] * gradlog_intr[..., 1:2])

    gradlog = gradlog - (gradlog * X0).sum(dim=-1, keepdim=True) * X0

    return (u_hat, gradlog) if not grad_only else gradlog



# ============================================================
# General BEL wrapper compatible with DPnP q_score
# ============================================================
@torch.no_grad()
def bel_general(
    X0, t,y,
    f_fn=gaussian_kernel_f,
    n_paths=5000,
    n_steps=5,
    generator=None,
    device=None,
    dtype=DEFAULT_DTYPE,
    reorthonormalize_every=1,
    f_kwargs=None,
    grad_only=True,
):
    if f_kwargs is None:
        f_kwargs = {}

    if device is None:
        device = get_best_device()

    X0 = torch.as_tensor(X0, dtype=dtype, device=device)
    if X0.shape[-1] != 3:
        raise ValueError(f"X0 must have last dimension 3, got {tuple(X0.shape)}")

    X0 = normalize_torch(X0, eps=get_eps(dtype))
    x_shape = X0.shape

    y_full = _expand_y_for_x(y, x_shape, dtype=dtype, device=device)

    X0_flat, lead_shape = _flatten_points_lastdim(X0)
    y_flat, _ = _flatten_points_lastdim(y_full)

    out = bel_gradlog_u_S2_batch_chunked_torch(
        X0=X0_flat,
        y=y_flat,
        t=t,
        f_fn=f_fn,
        n_paths=n_paths,
        n_steps=n_steps,
        generator=generator,
        device=device,
        dtype=dtype,
        reorthonormalize_every=reorthonormalize_every,
        f_kwargs=f_kwargs,
        grad_only=grad_only,
    )

    if grad_only:
        return out.reshape(*lead_shape, 3)

    u_hat_flat, grad_flat = out
    return u_hat_flat.reshape(*lead_shape), grad_flat.reshape(*lead_shape, 3)

def get_bel(f_fn=gaussian_kernel_f,
    n_paths=5000,
    n_steps=5,
    generator=None,
    device=None,
    dtype=DEFAULT_DTYPE):
    def bel(X0,t,y):
        return bel_general(X0, t,y,
    f_fn=f_fn,
    n_paths=n_paths,
    n_steps=n_steps,
    generator=generator,
    device=device,
    dtype=dtype,)
    return bel