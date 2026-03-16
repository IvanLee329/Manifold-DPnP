import torch
import torch.nn as nn
import math
from sde import GRW_SDE_path_integrator
from loss import ism_exact
from utils import normalize_torch
from torch.optim.lr_scheduler import LambdaLR
import os
############################ schedules #############################
class BetaSchedule(nn.Module):
    def beta_t(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def reverse(self):
        raise NotImplementedError


class LinearBetaSchedule(BetaSchedule):
    def __init__(
        self,
        tf: float = 1.0,
        t0: float = 0.0,
        beta_0: float = 1e-3,
        beta_f: float = 5.0,
    ):
        super().__init__()
        self.tf = tf
        self.t0 = t0
        self.beta_0 = beta_0
        self.beta_f = beta_f

    def beta_t(self, t: torch.Tensor) -> torch.Tensor:
        normed_t = (t - self.t0) / (self.tf - self.t0)
        return self.beta_0 + normed_t * (self.beta_f - self.beta_0)

    def log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        normed_t = (t - self.t0) / (self.tf - self.t0)
        return -0.5 * (
            0.5 * normed_t**2 * (self.beta_f - self.beta_0)
            + normed_t * self.beta_0
        )

    def rescale_t(self, t: torch.Tensor) -> torch.Tensor:
        return -2.0 * self.log_mean_coeff(t)

    def reverse(self):
        return LinearBetaSchedule(
            tf=self.t0,
            t0=self.tf,
            beta_f=self.beta_0,
            beta_0=self.beta_f,
        )


class ConstantBetaSchedule(LinearBetaSchedule):
    def __init__(self, tf: float = 1.0, value: float = 1.0):
        super().__init__(tf=tf, t0=0.0, beta_0=value, beta_f=value)

def make_linear_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 1000,
    total_steps: int = 100000,
    min_lr_ratio: float = 0.0,
):
    """
    Learning-rate multiplier schedule:
      - linearly ramps from 0 to 1 over `warmup_steps`
      - then follows cosine decay from 1 to `min_lr_ratio`

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Your optimizer.
    warmup_steps : int
        Number of warmup steps. Default: 1000
    total_steps : int
        Total number of training steps.
    min_lr_ratio : float
        Final LR is `base_lr * min_lr_ratio`.
        Use 0.0 for cosine decay to zero.

    Returns
    -------
    scheduler : torch.optim.lr_scheduler.LambdaLR
    """

    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be nonnegative.")
    if warmup_steps > total_steps:
        raise ValueError("warmup_steps cannot exceed total_steps.")
    if not (0.0 <= min_lr_ratio <= 1.0):
        raise ValueError("min_lr_ratio must lie in [0, 1].")

    def lr_lambda(current_step: int):
        # Linear warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # Cosine decay
        if total_steps == warmup_steps:
            return min_lr_ratio

        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


##########################################Check point ##########################################

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    step,
    epoch,
    best_metric,
):
    """
    Save training checkpoint.

    Parameters
    ----------
    path : str
        File path to save checkpoint.
    model : torch.nn.Module
    optimizer : torch.optim.Optimizer
    scheduler : torch.optim.lr_scheduler
    step : int
        Global training step.
    epoch : int
        Current epoch.
    best_metric : float
        Best validation metric so far.
    """

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "epoch": epoch,
        "best_metric": best_metric,
    }

    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, device="mps"):
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model"])

    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and ckpt["scheduler"] is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    global_step = ckpt["step"]
    epoch = ckpt["epoch"]
    best_val_loss = ckpt["best_metric"]

    return model, optimizer, scheduler, global_step, epoch, best_val_loss

########################################## training loop ##########################################

def ism_training_step_pathwise(
    model,
    optimizer,
    x0: torch.Tensor,
    T: float,
    n_steps: int,
    generator=None,
):
    model.train()

    if x0.ndim == 1:
        x0 = x0.unsqueeze(0)

    x0 = normalize_torch(x0)
    device = x0.device
    N = x0.shape[0]

    k_idx = torch.randint(
        low=1,
        high=n_steps + 1,
        size=(N,),
        device=device,
        generator=generator,
    )

    # Match the actual discrete path times
    t = T * (k_idx.float() / n_steps)
    t = t.unsqueeze(-1)

    beta_schedule = LinearBetaSchedule()

    with torch.no_grad():
        zero_drift = lambda x, tau: torch.zeros_like(x)

        path = GRW_SDE_path_integrator(
            b=zero_drift,
            sig=lambda x, tau: torch.sqrt(beta_schedule.beta_t(tau)),
            x=x0,
            T=T,
            n_steps=n_steps,
            generator=generator,
            return_path=True,
        )
        x_t = path[k_idx - 1, torch.arange(N, device=device)]

    loss = ism_exact(model, x_t, t)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
    optimizer.step()
    return loss.item(), x_t.detach(), t.detach()


def evaluate_ism(
    model,
    data_loader,
    T: float,
    n_steps: int,
    n_eval_repeats: int = 8,
    eval_seed: int = 12345,
    device=None,
):
    model.eval()
    losses = []

    if device is None:
        device = next(model.parameters()).device

    beta_schedule = LinearBetaSchedule()

    # Fixed generator so eval is reproducible across calls
    g = torch.Generator(device=device)
    g.manual_seed(eval_seed)

    for batch in data_loader:
        if isinstance(batch, (list, tuple)):
            x0 = batch[0]
        else:
            x0 = batch

        x0 = x0.to(device)

        if x0.ndim == 1:
            x0 = x0.unsqueeze(0)

        x0 = normalize_torch(x0)
        N = x0.shape[0]

        batch_loss = 0.0

        for _ in range(n_eval_repeats):
            k_idx = torch.randint(
                low=1,
                high=n_steps + 1,
                size=(N,),
                device=device,
                generator=g,
            )

            # Match path time grid
            t = T * (k_idx.float() / n_steps)
            t = t.unsqueeze(-1)

            zero_drift = lambda x, tau: torch.zeros_like(x)

            path = GRW_SDE_path_integrator(
                b=zero_drift,
                sig=lambda x, tau: torch.sqrt(beta_schedule.beta_t(tau)),
                x=x0,
                T=T,
                n_steps=n_steps,
                generator=g,
                return_path=True,
            )

            x_t = path[k_idx - 1, torch.arange(N, device=device)]
            loss = ism_exact(model, x_t, t)
            batch_loss += loss.item()

        losses.append(batch_loss / n_eval_repeats)

    return float(sum(losses) / len(losses)) if losses else float("nan")



def train_ism_pathwise(
    model,
    train_loader,
    val_loader,
    test_loader,
    T: float = 1.0,
    lr: float = 2e-4,
    n_epochs: int = 20,
    n_steps: int = 100,
    device: str = "mps",
    generator=None,
    log_every: int = 100,
    eval_every: int = 1000,
    checkpoint_dir="checkpoint",
    resume_path=None,
):
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device(device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
    )

    total_steps = n_epochs * len(train_loader)
    scheduler = make_linear_warmup_cosine_scheduler(
        optimizer=optimizer,
        warmup_steps=1000,
        total_steps=total_steps,
    )

    global_step = 0
    start_epoch = 0
    best_val_loss = float("inf")
    loss_history = []

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

        global_step = ckpt["step"]
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_metric"]

        print(
            f"Resumed from epoch={start_epoch} "
            f"step={global_step} "
            f"best_val_loss={best_val_loss:.6f}"
        )

    for epoch in range(start_epoch, n_epochs):
        model.train()

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x0 = batch[0]
            else:
                x0 = batch

            x0 = x0.to(device)

            loss_value, _, t = ism_training_step_pathwise(
                model=model,
                optimizer=optimizer,
                x0=x0,
                T=T,
                n_steps=n_steps,
                generator=generator,
            )

            scheduler.step()
            global_step += 1

            if global_step % log_every == 0:
                print(
                    f"epoch={epoch:03d} "
                    f"step={global_step:06d} "
                    f"loss={loss_value:.6f} "
                    f"t_mean={t.mean().item():.4f}"
                )

            if global_step % eval_every == 0:
                val_loss = evaluate_ism(
                    model=model,
                    data_loader=val_loader,
                    T=T,
                    n_steps=n_steps,
                )
                print(f"[eval] step={global_step:06d} val_loss={val_loss:.6f}")
                loss_history.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        f"{checkpoint_dir}/best.pt",
                        model,
                        optimizer,
                        scheduler,
                        global_step,
                        epoch,
                        best_val_loss,
                    )
                    print("New best checkpoint saved")

                save_checkpoint(
                    f"{checkpoint_dir}/latest.pt",
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                    epoch,
                    best_val_loss,
                )

    test_loss = evaluate_ism(
        model=model,
        data_loader=test_loader,
        T=T,
        n_steps=n_steps,
    )
    print(f"[final] test_loss={test_loss:.6f}")

    model.loss_history = loss_history
    return model