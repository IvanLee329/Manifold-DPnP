import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from utils import normalize_torch


def latlon_deg_to_extrinsic_torch(latlon_deg: torch.Tensor) -> torch.Tensor:
    intrinsic = torch.pi * (latlon_deg / 180.0) + torch.tensor(
        [torch.pi / 2, torch.pi],
        dtype=latlon_deg.dtype,
        device=latlon_deg.device,
    )

    theta = intrinsic[:, 0]
    phi = intrinsic[:, 1]

    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    out = torch.stack([x, y, z], dim=-1)
    return normalize_torch(out)


def extrinsic_to_latlon_deg_torch(x: torch.Tensor) -> torch.Tensor:
    x = normalize_torch(x)

    xx = x[:, 0]
    yy = x[:, 1]
    zz = x[:, 2]

    lat = -torch.arcsin(zz)
    lon = torch.atan2(yy, xx)

    return torch.stack([lat, lon], dim=-1) * (180.0 / torch.pi)


class EarthSphericalDataset(Dataset):
    FILE_MAP = {
        "earthquake": ("quakes_all.csv", 4),
        "fire": ("fire.csv", 1),
        "flood": ("flood.csv", 2),
        "volcano": ("volerup.csv", 2),
    }

    def __init__(
        self,
        data_dir: str = "data",
        name: str = "earthquake",
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        super().__init__()

        if name not in self.FILE_MAP:
            raise ValueError(
                f"Unknown earth dataset '{name}'. "
                f"Choose from {list(self.FILE_MAP.keys())}."
            )

        filename, skip_header = self.FILE_MAP[name]
        path = os.path.join(data_dir, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find dataset file: {path}")

        raw = np.genfromtxt(path, delimiter=",", skip_header=skip_header)
        raw = np.asarray(raw, dtype=np.float32)

        if raw.ndim != 2 or raw.shape[1] < 2:
            raise ValueError(
                f"Expected at least 2 columns for lat/lon, got shape {raw.shape}"
            )

        latlon_deg = torch.tensor(raw[:, :2], dtype=dtype, device=device)
        extrinsic = latlon_deg_to_extrinsic_torch(latlon_deg)

        self.name = name
        self.latlon_deg = latlon_deg
        self.data = extrinsic

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        x = self.data[idx]
        context = None
        return x, context


def earth_collate_fn(batch):
    xs = torch.stack([item[0] for item in batch], dim=0)
    contexts = [item[1] for item in batch]
    context = None if all(c is None for c in contexts) else contexts
    return xs, context


def make_earth_dataloaders(
    data_dir: str = "data",
    name: str = "earthquake",
    batch_size: int = 512,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 0,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
):


    dataset = EarthSphericalDataset(
        data_dir=data_dir,
        name=name,
        dtype=dtype,
        device=device,
    )

    n = len(dataset)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    n_test = n - n_train - n_val

    g = torch.Generator()
    g.manual_seed(seed)

    train_ds, val_ds, test_ds = random_split(
        dataset,
        lengths=[n_train, n_val, n_test],
        generator=g,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=earth_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=len(val_ds),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=earth_collate_fn,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=len(test_ds),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=earth_collate_fn,
    )

    return train_loader, val_loader, test_loader, dataset