import torch


################################ exact divergence ##################################
def riemannian_divergence_s2(score_fn, x, t):
    x = x.requires_grad_(True)
    s = score_fn(x, t)

    B, d = x.shape
    J_cols = []

    for i in range(d):
        grad_si = torch.autograd.grad(
            s[:, i].sum(),
            x,
            create_graph=True,
            retain_graph=True,
        )[0]  # (B, d)
        J_cols.append(grad_si.unsqueeze(1))

    J = torch.cat(J_cols, dim=1)   # (B, d, d), J[:, i, j] = d s_i / d x_j

    I = torch.eye(d, device=x.device, dtype=x.dtype).unsqueeze(0)   # (1,d,d)
    P = I - x.unsqueeze(-1) * x.unsqueeze(-2)                       # (B,d,d)

    PJ = torch.matmul(P, J)
    return torch.diagonal(PJ, dim1=-2, dim2=-1).sum(dim=-1)


def ism_exact(score_model, x_t, t, weight=None):
    """
    Intrinsic ISM loss on S^2.

    score_model(x_t, t) must return tangent vectors in ambient R^3 coords.
    """
    x_t = x_t.requires_grad_(True)
    score = score_model(x_t, t)                      # (B, 3)
    sq_norm = (score ** 2).sum(dim=-1)              # (B,)
    div_score = riemannian_divergence_s2(score_model, x_t, t)

    losses = 0.5 * sq_norm + div_score

    if weight is not None:
        if weight.ndim == 2 and weight.shape[-1] == 1:
            weight = weight.squeeze(-1)
        losses = weight * losses

    loss = losses.mean()
    return loss

