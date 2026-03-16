import torch
from sde import GRW_SDE_path_integrator
from utils import normalize_torch

import torch


@torch.no_grad()
def dPnP_sampler_torch_batched(
    q_score,
    p_score,
    y,
    out_samples,
    eta,
    grw_steps=5,
    seed=0,
    end_only=False,
    device="mps",
    dtype=torch.float32,
):
    """
    Batched DPnP sampler on S^2 / manifold embedded in R^d.

    Parameters
    ----------
    q_score : callable
        Should accept:
            q_score(x, t, y=...)
        with
            x : (B, P, d)
            y : (B, P, d)
        and return a drift/score of shape (B, P, d)

    p_score : callable
        Should accept:
            p_score(x, t)
        with
            x : (B, P, d)
        and return shape (B, P, d)

    y : tensor-like
        Shape (d,) or (B, d)

    out_samples : int
        Number of DPnP samples per observation y_i

    eta : 1D tensor-like
        Time schedule, length = number of outer steps

    Returns
    -------
    If end_only=True:
        X_final : (B, out_samples, d)
    else:
        X : (steps+1, B, out_samples, d)
    """
    device = torch.device(device)
    eta = torch.as_tensor(eta, device=device, dtype=dtype)
    steps = eta.numel()

    y = torch.as_tensor(y, device=device, dtype=dtype)
    if y.ndim == 1:
        y = y[None, :]   # (1,d)
    elif y.ndim != 2:
        raise ValueError("y must have shape (d,) or (B,d)")

    B, d = y.shape
    P = int(out_samples)
    eps = 1e-12

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    # X has shape (steps+1, B, P, d)
    X = torch.empty((steps + 1, B, P, d), device=device, dtype=dtype)

    # initialize on the sphere
    X0 = torch.randn((B, P, d), device=device, dtype=dtype, generator=gen)
    X[0] = normalize_torch(X0, eps=eps)

    # expand y once to (B,P,d)
    y_expanded = y[:, None, :].expand(B, P, d)
    
    for i in range(int(steps)):
        t_i = eta[i]

        # Plug-and-play step with data-dependent score
        X_pl = GRW_SDE_path_integrator(
            b=lambda x, t: q_score(x, t, y=y_expanded),
            sig = lambda x, t: 1.0,
            x=X[i],
            T=t_i,
            n_steps=grw_steps,
            generator=gen,
            return_path=False,
        )   # expected shape (B,P,d)

        # Prior step
        X_prior = GRW_SDE_path_integrator(
            b=p_score,
            sig = lambda x, t: 1.0,
            x=X_pl,
            T=t_i,
            n_steps=grw_steps,
            generator=gen,
            return_path=False,
        )   # expected shape (B,P,d)

        X[i + 1] = normalize_torch(X_prior, eps=eps)

    return X[-1] if end_only else X