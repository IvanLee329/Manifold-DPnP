import torch
import torch.nn as nn
from utils import normalize_torch, tangent_project_torch


def ambient_generators_s2(x: torch.Tensor) -> torch.Tensor:
    x = normalize_torch(x)
    eye = torch.eye(3, device=x.device, dtype=x.dtype)
    batch_shape = x.shape[:-1]
    eye = eye.expand(*batch_shape, 3, 3)
    xxT = x.unsqueeze(-1) * x.unsqueeze(-2)
    return eye - xxT


class ConcatEmbedding(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == x.ndim - 1:
            t = t.unsqueeze(-1)
        return torch.cat([x, t], dim=-1)


class SineLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.linear(x))


class SineMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_hidden_layers: int = 3):
        super().__init__()
        layers = [SineLayer(in_dim, hidden_dim)]
        for _ in range(n_hidden_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim))
        self.hidden = nn.Sequential(*layers)
        self.final = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.hidden(x))


class AmbientGeneratorScoreNet(nn.Module):
    """
    s_theta(x,t) = sum_i f_i(x,t) X_i(x),
    with X_i(x) = (I - x x^T)e_i.
    """
    def __init__(self, hidden_dim: int = 128, n_hidden_layers: int = 3):
        super().__init__()
        self.embedding = ConcatEmbedding()
        self.net = SineMLP(
            in_dim=4,
            hidden_dim=hidden_dim,
            out_dim=3,
            n_hidden_layers=n_hidden_layers,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = normalize_torch(x)
        feat = self.embedding(x, t)
        fi = self.net(feat)
        Xi = ambient_generators_s2(x)
        score = torch.einsum("ni,ndi->nd", fi, Xi)
        return tangent_project_torch(x, score)
    

class DPnPScoreWrapper(nn.Module):
    """
    Wraps a score network with forward(x_flat, t_flat) so it can accept x of shape (...,3).
    """
    def __init__(self, score_model: nn.Module):
        super().__init__()
        self.score_model = score_model

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        if x_shape[-1] != 3:
            raise ValueError(f"x must have last dimension 3, got {tuple(x_shape)}")

        x_flat = x.reshape(-1, 3)

        t = torch.as_tensor(t, device=x.device, dtype=x.dtype)
        if t.ndim == 0:
            t_flat = t.expand(x_flat.shape[0])
        else:
            t_flat = t.reshape(-1)
            if t_flat.numel() == 1:
                t_flat = t_flat.expand(x_flat.shape[0])
            elif t_flat.numel() != x_flat.shape[0]:
                raise ValueError(
                    f"t has {t_flat.numel()} entries but x has {x_flat.shape[0]} points"
                )

        out = self.score_model(x_flat, t_flat)

        if out.shape != x_flat.shape:
            raise ValueError(
                f"score_model must return shape {tuple(x_flat.shape)}, got {tuple(out.shape)}"
            )

        return out.reshape(x_shape)





