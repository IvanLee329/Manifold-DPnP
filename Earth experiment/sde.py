import torch
from utils import normalize_torch,tangent_project_torch,sphere_exp_map_torch


@torch.no_grad()
def GRW_SDE_path_integrator(
    b,
    sig,
    x: torch.Tensor,
    T: float,
    n_steps: int = 5,
    generator=None,
    return_path: bool = True,
):
    x = normalize_torch(x)
    added_batch_dim = False

    if x.ndim == 1:
        x = x[None, :]
        added_batch_dim = True

    d = x.shape[-1]
    batch_shape = x.shape[:-1]

    T = torch.as_tensor(T, device=x.device, dtype=x.dtype)
    dt = T / n_steps
    sqrt_dt = torch.sqrt(dt)

    if return_path:
        path = []

    for k in range(n_steps):
        tk_mid = dt * (k + 0.5)

        Z = torch.randn(
            *batch_shape, d,
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )

        V = (sqrt_dt * sig(x, tk_mid) * Z) + (dt * b(x, tk_mid))
        V = tangent_project_torch(x, V)

        x = sphere_exp_map_torch(x, V)
        x = normalize_torch(x)

        if return_path:
            path.append(x.clone())

    out = torch.stack(path, dim=0) if return_path else x

    if added_batch_dim:
        out = out[:, 0, :] if return_path else out[0]

    return out