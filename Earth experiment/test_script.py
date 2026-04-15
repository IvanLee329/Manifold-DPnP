import torch
import numpy as np
from data import make_earth_dataloaders
from model import AmbientGeneratorScoreNet, DPnPScoreWrapper
from train import train_ism_pathwise
from bel import get_bel
from test_earth import  test_dpnp_sampler_multiple_xtrue
import matplotlib.pyplot as plt
from utils import get_vmf_f
device = "cuda" if torch.cuda.is_available() else "mps"

hidden_dim = 512
n_hidden_layers = 5
MODEL_PATH = "p_score_model_weights.pth" 
KAPPA = 30

def p_score_model(train_loader, val_loader, test_loader, hidden_dim = hidden_dim, n_hidden_layers = n_hidden_layers, T = 1, lr = 2e-4, n_epochs = 20000,n_steps = 100):
    device = "cuda" if torch.cuda.is_available() else "mps"
    model = AmbientGeneratorScoreNet(hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers)
    trained_model = train_ism_pathwise(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        T=T,
        lr=lr,
        n_epochs=n_epochs,
        n_steps=n_steps,
        device=device,
        log_every=100,
        eval_every=1000,
    )
    return trained_model

batch_size = 512
data_name = 'earthquake'
seed = 0
train_loader, val_loader, test_loader, dataset = make_earth_dataloaders(
        data_dir="data",
        name=data_name,   # change to: "fire", "flood", "volcano"
        batch_size=batch_size,
        seed=seed,
        device="cpu",        # keep raw data on CPU; train loop moves batches
    )

p_model_loaded2= AmbientGeneratorScoreNet(hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers)
p_model_loaded2.load_state_dict(torch.load('checkpoint/best.pt',weights_only=False)['model'])
p_model_loaded2.eval().to(device)
p_score2 = DPnPScoreWrapper(p_model_loaded2)

def aneal_schedule(K:int = 20, K0:int = 5, eta0:float = 0.45, etaK:float = 0.15):
    etas = np.zeros(K)
    for i in range(K):
        if i < K0:
            etas[i] = eta0
        else:
            frac = (i-K0)/(K-K0)
            etas[i] = eta0 * ((etaK/eta0) ** frac)
    return etas

q_score = get_bel(f_fn= get_vmf_f(kappa=KAPPA))
eta = aneal_schedule(K=20, K0=5, eta0  = 0.1, etaK  = 0.05)


results2 = test_dpnp_sampler_multiple_xtrue(
    dataloader=test_loader,
    q_score=q_score,
    p_score=p_score2,
    eta=eta,
    sigma_y=KAPPA,
    num_xtrue=10,
    num_y_per_xtrue=32,
    out_samples=64,
    grw_steps=5,
    device=device,
    dtype=torch.float32,
    seed=2,
    kappa=25.0,
    plot_results=True
)
save_dict = {
    "results": results2,
    "sigma_y": KAPPA,
    "eta": eta,
    "num_y_per_xtrue": 32,
    "out_samples": 64,
    "grw_steps": 5,
    'num_xtrue': 80,
    'seed': 2,
    'kappa':25
}

torch.save(save_dict, "dpnp_test_result_2.pt")