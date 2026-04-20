"""
PINN for the 1D quantum BEC droplet (Petrov-Astrakharchik 2016).

Problem
-------
Stationary 1D extended Gross-Pitaevskii equation with repulsive cubic
(mean-field) and attractive quadratic (Lee-Huang-Yang) nonlinearities:

    -0.5 * psi_xx  -  |psi| * psi  +  |psi|^2 * psi  -  mu * psi  =  0

Analytical ground state (with mu < 0, |mu| < 2/9):
    B   = sqrt(1 - 4.5 * |mu|)
    k   = sqrt(2 * |mu|)
    psi = 3 * |mu| / (1 + B * cosh(k * x))          (positive-psi gauge)

Analytical norm (Petrov-Astrakharchik eq. 10):
    a     = sqrt(4.5 * |mu|)                        (= sqrt(-9 mu / 2))
    N(mu) = (4/3) * [ ln((a + 1) / B) - a ]
(verified to machine precision against numerical integration.)

Network
-------
MLP: (x, mu) -> psi     -- one network amortizes across the whole droplet family.

Total training loss
-------------------
    L_total  =  w_pde * L_pde  +  w_bc * L_bc  +  w_uniq * L_uniq

with:

  L_pde     =  mean over N_col collocation points (x_i, mu_i) of the
               squared PDE residual:
                   r(x, mu) := -0.5 psi_xx - |psi| psi + |psi|^2 psi - mu psi
                   L_pde    :=  (1/N_col) * sum_i  r(x_i, mu_i)^2

  L_bc      =  mean squared wavefunction at x = +/- L (soft Dirichlet):
                   L_bc  :=  (1/N_bc) * sum_j  psi(x_j, mu_j)^2,   x_j in {-L, +L}

  L_uniq    =  one of four choices, selected by uniqueness_method:

      (a) "norm"       :  match integrated |psi|^2 to the analytical N(mu).
                          For M mu-samples:
                              L_uniq := (1/M) * sum_m ( N_pred(mu_m) - N(mu_m) )^2
                          where N_pred is a trapezoidal integral of |psi|^2
                          on a fixed x-grid.

      (b) "anchor"     :  pin psi at a single x (= anchor_x, default 0) for
                          each mu to its analytical value:
                              L_uniq := (1/M) * sum_m ( psi(anchor_x, mu_m)
                                                        - psi_exact(anchor_x, mu_m) )^2

      (c) "semi_supervised" :
                          mean squared error against N_sup = 50 frozen
                          (x, mu, psi_exact) triples sampled uniformly:
                              L_uniq := (1/N_sup) * sum_s ( psi(x_s, mu_s)
                                                            - psi_exact(x_s, mu_s) )^2

      (d) "none"       :  L_uniq := 0   (control; should collapse to psi=0)

The loss weights w_pde, w_bc, w_uniq are hyperparameters (see config.yaml).
Default values: w_pde=1, w_bc=10, w_uniq=10 — standard PINN weighting that
compensates for the small number of BC points and emphasises the uniqueness
constraint to break the trivial-solution degeneracy.

Everything in this file is deliberately small and hackable. The
`run_experiment.py` entry point + `config.yaml` drive it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# NumPy 2.0 renamed np.trapz -> np.trapezoid. Support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# =============================================================================
#  Analytical solution & norm
# =============================================================================

def psi_exact_np(x: np.ndarray, mu: float) -> np.ndarray:
    """Petrov-Astrakharchik analytical ground state (NumPy)."""
    abs_mu = abs(mu)
    assert 0.0 < abs_mu < 2.0 / 9.0, f"|mu|={abs_mu} outside (0, 2/9)"
    B = math.sqrt(1.0 - 4.5 * abs_mu)
    k = math.sqrt(2.0 * abs_mu)
    return (3.0 * abs_mu) / (1.0 + B * np.cosh(k * x))


def psi_exact_torch(x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Analytical solution (Torch, broadcastable). mu is negative."""
    abs_mu = mu.abs()
    B = torch.sqrt(torch.clamp(1.0 - 4.5 * abs_mu, min=1e-12))
    k = torch.sqrt(2.0 * abs_mu)
    return (3.0 * abs_mu) / (1.0 + B * torch.cosh(k * x))


def norm_exact(mu: float) -> float:
    """Exact norm N = integral |psi|^2 dx, from Petrov-Astrakharchik eq. (10):

        N = (4/3) * [ ln( (a + 1) / B ) - a ]

    with a = sqrt(-9 mu / 2) = sqrt(9 |mu| / 2),
         B = sqrt(1 + 9 mu / 2) = sqrt(1 - 9 |mu| / 2).

    Verified to machine precision against numerical trapezoid integration
    (see _sanity_check_norm at the bottom of this file).
    """
    abs_mu = abs(mu)
    a = math.sqrt(4.5 * abs_mu)          # sqrt(-9 mu / 2)
    B = math.sqrt(1.0 - 4.5 * abs_mu)    # sqrt(1 + 9 mu / 2)
    return (4.0 / 3.0) * (math.log((a + 1.0) / B) - a)


# =============================================================================
#  Network
# =============================================================================

class MLP(nn.Module):
    """Plain fully-connected MLP with tanh activations, (x, mu) -> psi."""

    def __init__(self, depth: int = 5, width: int = 64,
                 activation: str = "tanh"):
        super().__init__()
        act = {"tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}[activation]
        layers: list[nn.Module] = [nn.Linear(2, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        # Sensible init: smaller output layer to keep initial psi ~ O(0.1)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    def forward(self, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        # x, mu expected shape [N,1]; we rescale mu into roughly [-1,1] for
        # better conditioning (mu lives in [-0.22, 0]).
        mu_scaled = mu / 0.2
        h = torch.cat([x, mu_scaled], dim=-1)
        return self.net(h).squeeze(-1)


# =============================================================================
#  PDE residual
# =============================================================================

def pde_residual(model: MLP, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Residual of  -0.5 psi_xx - |psi| psi + |psi|^2 psi - mu psi  = 0.

    x, mu : tensors of shape [N,1] with requires_grad=True on x.
    """
    psi = model(x, mu)                                             # [N]
    grad_x = torch.autograd.grad(
        psi, x, grad_outputs=torch.ones_like(psi),
        create_graph=True, retain_graph=True,
    )[0]                                                           # [N,1]
    psi_x = grad_x.squeeze(-1)
    grad_xx = torch.autograd.grad(
        psi_x, x, grad_outputs=torch.ones_like(psi_x),
        create_graph=True, retain_graph=True,
    )[0]
    psi_xx = grad_xx.squeeze(-1)

    abs_psi = psi.abs()
    mu_flat = mu.squeeze(-1)
    res = -0.5 * psi_xx - abs_psi * psi + abs_psi**2 * psi - mu_flat * psi
    return res


# =============================================================================
#  Training
# =============================================================================

@dataclass
class TrainConfig:
    # Domain
    L: float = 30.0
    mu_min: float = -0.20        # nearer flat-top limit (|mu| = 0.2222...)
    mu_max: float = -0.02        # nearer NLSE soliton limit

    # Network
    depth: int = 5
    width: int = 64
    activation: str = "tanh"

    # Collocation
    n_collocation: int = 4000
    n_mu_train: int = 32         # number of mu samples per resample
    n_boundary: int = 64

    # Uniqueness ablation: "norm", "anchor", "semi_supervised",
    #                      "norm_plus_anchor", "full_profile", or "none"
    uniqueness_method: str = "norm"
    # Only for uniqueness_method == "norm" (or "norm_plus_anchor"):
    # how to compare N_pred vs N(mu).
    #   "absolute" : (N_pred - N_true)^2            -- biased toward large |mu|
    #   "relative" : ((N_pred - N_true) / N_true)^2 -- scale-invariant (recommended)
    #   "log"      : (log N_pred - log N_true)^2    -- also scale-invariant
    norm_loss_type: str = "relative"
    # Only for "norm_plus_anchor": relative weight of the peak-anchor term
    # inside the hybrid uniqueness loss.
    anchor_weight_in_hybrid: float = 1.0
    # Only for "norm_plus_anchor": list of anchor x-positions, in units of
    # 1/k(mu) = 1/sqrt(2|mu|). So [0.0, 1.0, -1.0] places anchors at the
    # droplet peak and at +/- one characteristic half-width -- automatically
    # scaled with mu. Using more anchors adds more analytical shape
    # information per mu.
    anchor_x_list: tuple = (0.0,)
    n_supervised_points: int = 50       # only for semi_supervised
    anchor_x: float = 0.0               # only for anchor / norm_plus_anchor
    # Only for "full_profile": mu values at which the complete analytical
    # psi(x) is given on a dense x-grid. All must be in (-2/9, 0).
    mu_anchor_full: tuple = ()
    # x-grid density for the full-profile supervision.
    n_x_anchor_full: int = 201

    # Loss weights
    w_pde: float = 1.0
    w_bc: float = 10.0           # psi(+-L) ~ 0
    w_uniq: float = 10.0         # norm / anchor / semi-supervised

    # Optimization
    adam_iters: int = 20000
    adam_lr: float = 1e-3
    adam_lr_final: float = 1e-5  # cosine decay target
    lbfgs_iters: int = 50000
    lbfgs_history: int = 50
    lbfgs_tol_grad: float = 1e-9
    lbfgs_tol_change: float = 1e-12
    resample_every: int = 500

    # Misc
    seed: int = 0
    log_every: int = 500
    device: str = "cuda"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_collocation(cfg: TrainConfig, device: torch.device
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (x, mu) interior points. Returns requires_grad=True on x."""
    # x uniformly in [-L, L]; mu uniformly in [mu_min, mu_max]
    x = (torch.rand(cfg.n_collocation, 1, device=device) * 2 - 1) * cfg.L
    # For each collocation point, pair with a mu drawn from a small pool
    mu_pool = (torch.rand(cfg.n_mu_train, 1, device=device)
               * (cfg.mu_max - cfg.mu_min) + cfg.mu_min)
    idx = torch.randint(0, cfg.n_mu_train, (cfg.n_collocation,), device=device)
    mu = mu_pool[idx]
    x.requires_grad_(True)
    return x, mu


def sample_boundary(cfg: TrainConfig, device: torch.device
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Boundary points at +/- L, paired with random mu."""
    half = cfg.n_boundary // 2
    x_left = -cfg.L * torch.ones(half, 1, device=device)
    x_right = cfg.L * torch.ones(cfg.n_boundary - half, 1, device=device)
    x = torch.cat([x_left, x_right], dim=0)
    mu = (torch.rand(cfg.n_boundary, 1, device=device)
          * (cfg.mu_max - cfg.mu_min) + cfg.mu_min)
    return x, mu


def compute_norm_on_grid(model: MLP, mu_values: torch.Tensor,
                         x_grid: torch.Tensor) -> torch.Tensor:
    """Trapezoidal integral of |psi|^2 for each mu in mu_values.

    mu_values : [M,1]
    x_grid    : [Nx,1]
    returns   : [M]
    """
    M = mu_values.shape[0]
    Nx = x_grid.shape[0]
    x_rep = x_grid.repeat(M, 1)                    # [M*Nx, 1]
    mu_rep = mu_values.repeat_interleave(Nx, dim=0)  # [M*Nx, 1]
    psi = model(x_rep, mu_rep).reshape(M, Nx)
    dx = (x_grid[1, 0] - x_grid[0, 0]).item()
    # trapezoidal on a uniform grid
    integrand = psi.abs() ** 2
    integral = (integrand[:, :-1] + integrand[:, 1:]).sum(dim=1) * dx * 0.5
    return integral


def build_uniqueness_term(cfg: TrainConfig, model: MLP,
                          device: torch.device
                          ) -> Callable[[], torch.Tensor]:
    """Factory returning a CLOSURE that computes the chosen uniqueness loss.

    The returned closure samples random mu values internally at each call.
    This is the stochastic version used during the Adam phase. For the L-BFGS
    phase (which needs deterministic objectives), use
    build_uniqueness_term_deterministic.
    """
    if cfg.uniqueness_method == "none":
        return lambda: torch.tensor(0.0, device=device)

    # x-grid used for trapezoidal norm integration (when needed).
    Nx_grid = 1001
    x_grid = torch.linspace(-cfg.L, cfg.L, Nx_grid,
                            device=device).unsqueeze(-1)
    norm_loss_type = getattr(cfg, "norm_loss_type", "relative")
    anchor_weight = getattr(cfg, "anchor_weight_in_hybrid", 1.0)
    anchor_x_list = tuple(getattr(cfg, "anchor_x_list", (0.0,)))

    if cfg.uniqueness_method in ("norm", "norm_plus_anchor"):
        use_anchor = (cfg.uniqueness_method == "norm_plus_anchor")

        def loss_fn() -> torch.Tensor:
            M = 16
            mu_vals = (torch.rand(M, 1, device=device)
                       * (cfg.mu_max - cfg.mu_min) + cfg.mu_min)
            return _norm_or_hybrid_loss(
                model, mu_vals, x_grid, norm_loss_type,
                use_anchor=use_anchor,
                anchor_x_list=anchor_x_list,
                anchor_weight=anchor_weight,
            )

        return loss_fn

    if cfg.uniqueness_method == "anchor":
        def loss_fn() -> torch.Tensor:
            M = 32
            mu_vals = (torch.rand(M, 1, device=device)
                       * (cfg.mu_max - cfg.mu_min) + cfg.mu_min)
            x_anchor = cfg.anchor_x * torch.ones(M, 1, device=device)
            psi_pred = model(x_anchor, mu_vals)
            psi_true = psi_exact_torch(x_anchor, mu_vals).squeeze(-1)
            return ((psi_pred - psi_true) ** 2).mean()

        return loss_fn

    if cfg.uniqueness_method == "semi_supervised":
        # freeze a fixed supervised dataset from the analytical solution
        rng = np.random.default_rng(cfg.seed)
        xs = rng.uniform(-cfg.L, cfg.L, cfg.n_supervised_points)
        mus = rng.uniform(cfg.mu_min, cfg.mu_max, cfg.n_supervised_points)
        ys = np.array([psi_exact_np(np.array([xi]), mi)[0]
                       for xi, mi in zip(xs, mus)])
        x_sup = torch.tensor(xs, dtype=torch.float32,
                             device=device).unsqueeze(-1)
        mu_sup = torch.tensor(mus, dtype=torch.float32,
                              device=device).unsqueeze(-1)
        y_sup = torch.tensor(ys, dtype=torch.float32, device=device)

        def loss_fn() -> torch.Tensor:
            psi_pred = model(x_sup, mu_sup)
            return ((psi_pred - y_sup) ** 2).mean()

        return loss_fn

    if cfg.uniqueness_method == "full_profile":
        # For each mu in mu_anchor_full, pre-compute the complete analytical
        # psi(x) on a dense x-grid and freeze it as a constant tensor.
        # MSE over all (x, mu) pairs in those frozen slices.
        # Already deterministic, so works unchanged in the L-BFGS phase.
        if len(cfg.mu_anchor_full) == 0:
            raise ValueError(
                "uniqueness_method='full_profile' requires at least one "
                "mu value in mu_anchor_full."
            )
        x_grid_np = np.linspace(-cfg.L, cfg.L, cfg.n_x_anchor_full)
        x_grid_t = torch.tensor(
            x_grid_np, dtype=torch.float32, device=device,
        ).unsqueeze(-1)                                         # [Nx, 1]
        profiles: list[tuple[torch.Tensor, torch.Tensor]] = []
        for mu_val in cfg.mu_anchor_full:
            psi_np = psi_exact_np(x_grid_np, float(mu_val))
            mu_t = torch.full_like(x_grid_t, float(mu_val))    # [Nx, 1]
            psi_t = torch.tensor(psi_np, dtype=torch.float32, device=device)
            profiles.append((mu_t, psi_t))

        def loss_fn() -> torch.Tensor:
            per_mu = []
            for mu_t, psi_true in profiles:
                psi_pred = model(x_grid_t, mu_t)               # [Nx]
                per_mu.append(((psi_pred - psi_true) ** 2).mean())
            return torch.stack(per_mu).mean()

        return loss_fn

    raise ValueError(f"unknown uniqueness_method: {cfg.uniqueness_method}")


def build_uniqueness_term_deterministic(
    cfg: TrainConfig, model: MLP, device: torch.device,
    n_mu_freeze: int = 64,
) -> Callable[[], torch.Tensor]:
    """Like build_uniqueness_term but with ALL random samples frozen at
    construction time.

    L-BFGS calls the closure multiple times per step for its line search and
    cannot make progress if the objective is stochastic. This wrapper freezes
    the mu samples into constant tensors so the closure is a deterministic
    function of the model weights.
    """
    if cfg.uniqueness_method == "none":
        return lambda: torch.tensor(0.0, device=device)

    Nx_grid = 1001
    x_grid = torch.linspace(-cfg.L, cfg.L, Nx_grid,
                            device=device).unsqueeze(-1)
    norm_loss_type = getattr(cfg, "norm_loss_type", "relative")
    anchor_weight = getattr(cfg, "anchor_weight_in_hybrid", 1.0)
    anchor_x_list = tuple(getattr(cfg, "anchor_x_list", (0.0,)))

    if cfg.uniqueness_method in ("norm", "norm_plus_anchor"):
        use_anchor = (cfg.uniqueness_method == "norm_plus_anchor")
        mu_vals_frozen = (
            torch.rand(n_mu_freeze, 1, device=device)
            * (cfg.mu_max - cfg.mu_min) + cfg.mu_min
        )

        def loss_fn() -> torch.Tensor:
            return _norm_or_hybrid_loss(
                model, mu_vals_frozen, x_grid, norm_loss_type,
                use_anchor=use_anchor,
                anchor_x_list=anchor_x_list,
                anchor_weight=anchor_weight,
            )

        return loss_fn

    if cfg.uniqueness_method == "anchor":
        mu_vals_frozen = (
            torch.rand(n_mu_freeze, 1, device=device)
            * (cfg.mu_max - cfg.mu_min) + cfg.mu_min
        )
        x_anchor_frozen = cfg.anchor_x * torch.ones_like(mu_vals_frozen)

        def loss_fn() -> torch.Tensor:
            psi_pred = model(x_anchor_frozen, mu_vals_frozen)
            psi_true = psi_exact_torch(x_anchor_frozen,
                                       mu_vals_frozen).squeeze(-1)
            return ((psi_pred - psi_true) ** 2).mean()

        return loss_fn

    # semi_supervised and full_profile already use frozen datasets -- reuse
    # the stochastic builder (it produces a deterministic closure for these).
    if cfg.uniqueness_method in ("semi_supervised", "full_profile"):
        return build_uniqueness_term(cfg, model, device)

    raise ValueError(f"unknown uniqueness_method: {cfg.uniqueness_method}")


def _norm_or_hybrid_loss(
    model: MLP, mu_vals: torch.Tensor, x_grid: torch.Tensor,
    norm_loss_type: str,
    use_anchor: bool,
    anchor_x_list: tuple,
    anchor_weight: float,
) -> torch.Tensor:
    """Compute the (possibly hybrid) norm-based uniqueness loss.

    Components:
      - Norm matching:  compares integral |psi|^2 to analytical N(mu).
      - Peak / shape anchors (optional, 'norm_plus_anchor'):
            compares psi(x_i, mu) to analytical psi_exact(x_i, mu) at a list
            of anchor positions. Anchors are specified in units of 1/k(mu),
            so an anchor at "1.0" sits at a distance of one characteristic
            droplet half-width from the center -- automatically scaling with
            mu to cover the same relative portion of the droplet profile.
            All anchors use relative MSE so each carries equal weight
            regardless of the local psi scale.

    The anchor term uses only analytical scalars (closed-form ground state
    values at a small set of points), not labelled ground-truth shapes -- so
    it remains in the spirit of a physics-informed constraint rather than
    supervised learning.
    """
    device = mu_vals.device

    # --- Norm-matching term ------------------------------------------------
    N_pred = compute_norm_on_grid(model, mu_vals, x_grid)            # [M]
    N_true = torch.stack([
        torch.tensor(norm_exact(m.item()), device=device)
        for m in mu_vals.squeeze(-1)
    ])
    if norm_loss_type == "absolute":
        loss_norm = ((N_pred - N_true) ** 2).mean()
    elif norm_loss_type == "relative":
        loss_norm = (((N_pred - N_true) / N_true) ** 2).mean()
    elif norm_loss_type == "log":
        N_pred_safe = torch.clamp(N_pred, min=1e-12)
        loss_norm = ((torch.log(N_pred_safe) - torch.log(N_true)) ** 2).mean()
    else:
        raise ValueError(f"unknown norm_loss_type: {norm_loss_type}")

    if not use_anchor or len(anchor_x_list) == 0:
        return loss_norm

    # --- Shape-anchor term (multi-point) -----------------------------------
    # For each anchor position a_i (in units of 1/k(mu)), compare
    #   psi(a_i / k(mu), mu)  vs  psi_exact(a_i / k(mu), mu)
    # using relative MSE with a clamped denominator to avoid blow-up at
    # small peak amplitudes.
    k = torch.sqrt(2.0 * mu_vals.abs())                               # [M,1]
    anchor_losses = []
    for a_scaled in anchor_x_list:
        # x-position of this anchor per mu (broadcast over M)
        x_anc = (float(a_scaled) / k)                                 # [M,1]
        psi_pred_a = model(x_anc, mu_vals)                            # [M]
        psi_true_a = psi_exact_torch(x_anc, mu_vals).squeeze(-1)      # [M]
        scale = psi_true_a.abs().clamp(min=1e-2)
        anchor_losses.append(
            (((psi_pred_a - psi_true_a) / scale) ** 2).mean()
        )
    loss_anchor = torch.stack(anchor_losses).mean()

    return loss_norm + anchor_weight * loss_anchor


def train(cfg: TrainConfig, out_dir: Path) -> tuple[MLP, dict]:
    device = torch.device(cfg.device if torch.cuda.is_available()
                          else "cpu")
    print(f"[train] device={device}")
    set_seed(cfg.seed)

    model = MLP(cfg.depth, cfg.width, cfg.activation).to(device)
    uniqueness_loss_fn = build_uniqueness_term(cfg, model, device)

    history: dict[str, list] = {"step": [], "loss": [], "loss_pde": [],
                                "loss_bc": [], "loss_uniq": []}

    # ---------------------------- Adam phase ----------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=cfg.adam_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.adam_iters, eta_min=cfg.adam_lr_final,
    )

    x_int, mu_int = sample_collocation(cfg, device)
    x_bc, mu_bc = sample_boundary(cfg, device)

    t0 = time.time()
    for it in range(cfg.adam_iters):
        if it > 0 and it % cfg.resample_every == 0:
            x_int, mu_int = sample_collocation(cfg, device)
            x_bc, mu_bc = sample_boundary(cfg, device)

        opt.zero_grad(set_to_none=True)
        res = pde_residual(model, x_int, mu_int)
        loss_pde = (res ** 2).mean()

        psi_bc = model(x_bc, mu_bc)
        loss_bc = (psi_bc ** 2).mean()

        loss_uniq = uniqueness_loss_fn()

        loss = (cfg.w_pde * loss_pde + cfg.w_bc * loss_bc
                + cfg.w_uniq * loss_uniq)
        loss.backward()
        opt.step()
        sched.step()

        if it % cfg.log_every == 0 or it == cfg.adam_iters - 1:
            history["step"].append(it)
            history["loss"].append(loss.item())
            history["loss_pde"].append(loss_pde.item())
            history["loss_bc"].append(loss_bc.item())
            history["loss_uniq"].append(float(loss_uniq.detach())
                                        if torch.is_tensor(loss_uniq)
                                        else float(loss_uniq))
            print(f"[adam {it:6d}] loss={loss.item():.3e}  "
                  f"pde={loss_pde.item():.3e}  bc={loss_bc.item():.3e}  "
                  f"uniq={float(history['loss_uniq'][-1]):.3e}  "
                  f"lr={sched.get_last_lr()[0]:.2e}")

    t_adam = time.time() - t0
    print(f"[train] Adam phase done in {t_adam:.1f}s")

    # --------------------------- L-BFGS phase ---------------------------------

    cfg_lbfgs_points = max(cfg.n_collocation, 8000)
    cfg_n_mu_fixed = max(cfg.n_mu_train, 64)
    x_int = (torch.rand(cfg_lbfgs_points, 1, device=device) * 2 - 1) * cfg.L
    mu_pool = (torch.rand(cfg_n_mu_fixed, 1, device=device)
               * (cfg.mu_max - cfg.mu_min) + cfg.mu_min)
    idx = torch.randint(0, cfg_n_mu_fixed, (cfg_lbfgs_points,), device=device)
    mu_int = mu_pool[idx]
    x_int.requires_grad_(True)
    x_bc_fixed, mu_bc_fixed = sample_boundary(cfg, device)

    # Build a DETERMINISTIC uniqueness loss for L-BFGS: fix the mu samples so
    # closure() returns the same value for the same weights.
    uniqueness_loss_fn_lbfgs = build_uniqueness_term_deterministic(
        cfg, model, device,
    )

    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter=cfg.lbfgs_iters,
        history_size=cfg.lbfgs_history,
        tolerance_grad=cfg.lbfgs_tol_grad,
        tolerance_change=cfg.lbfgs_tol_change,
        line_search_fn="strong_wolfe",
    )

    step_counter = {"n": 0}
    t1 = time.time()

    def closure():
        lbfgs.zero_grad(set_to_none=True)
        # zero_grad only clears model-parameter grads. x_int has
        # requires_grad=True for psi_xx, so backward() accumulates into
        # x_int.grad on every closure call. Clear it explicitly.
        if x_int.grad is not None:
            x_int.grad = None
        res = pde_residual(model, x_int, mu_int)
        loss_pde = (res ** 2).mean()
        psi_bc = model(x_bc_fixed, mu_bc_fixed)
        loss_bc = (psi_bc ** 2).mean()
        loss_uniq = uniqueness_loss_fn_lbfgs()
        loss = (cfg.w_pde * loss_pde + cfg.w_bc * loss_bc
                + cfg.w_uniq * loss_uniq)
        loss.backward()
        step_counter["n"] += 1
        # Log on first step AND at regular intervals
        if step_counter["n"] == 1 or step_counter["n"] % cfg.log_every == 0:
            uniq_val = (float(loss_uniq.detach())
                        if torch.is_tensor(loss_uniq) else float(loss_uniq))
            print(f"[lbfgs {step_counter['n']:6d}] loss={loss.item():.3e}  "
                  f"pde={loss_pde.item():.3e}  bc={loss_bc.item():.3e}  "
                  f"uniq={uniq_val:.3e}")
            history["step"].append(cfg.adam_iters + step_counter["n"])
            history["loss"].append(loss.item())
            history["loss_pde"].append(loss_pde.item())
            history["loss_bc"].append(loss_bc.item())
            history["loss_uniq"].append(uniq_val)
        return loss

    # NOTE: no try/except here. If L-BFGS fails we want the traceback visible.
    lbfgs.step(closure)

    t_lbfgs = time.time() - t1
    print(f"[train] L-BFGS phase done in {t_lbfgs:.1f}s  "
          f"({step_counter['n']} closure calls)")

    history["train_time_adam_s"] = t_adam
    history["train_time_lbfgs_s"] = t_lbfgs
    return model, history


# =============================================================================
#  Evaluation & plots
# =============================================================================

@torch.no_grad()
def evaluate(model: MLP, cfg: TrainConfig, mu_eval: np.ndarray,
             n_x: int = 2001) -> dict:
    """Evaluate the model on a fine grid for each mu in mu_eval.

    Returns a dict with arrays shaped [len(mu_eval), n_x].
    """
    device = next(model.parameters()).device
    x_np = np.linspace(-cfg.L, cfg.L, n_x)
    x_t = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(-1)

    preds = np.zeros((len(mu_eval), n_x))
    exacts = np.zeros_like(preds)
    for i, mu in enumerate(mu_eval):
        mu_t = torch.full_like(x_t, float(mu))
        psi_pred = model(x_t, mu_t).cpu().numpy()
        psi_true = psi_exact_np(x_np, float(mu))
        preds[i] = psi_pred
        exacts[i] = psi_true

    # Sign-correct: the network may learn the +/- gauge freely. Align to exact
    # by matching the sign at x=0.
    center_idx = n_x // 2
    for i in range(len(mu_eval)):
        if preds[i, center_idx] * exacts[i, center_idx] < 0:
            preds[i] = -preds[i]

    abs_err = np.abs(preds - exacts)                          # pointwise
    max_exact = np.max(np.abs(exacts), axis=1, keepdims=True) # per-mu peak
    pointwise_rel = abs_err / max_exact                       # [M, Nx]

    l2_err = np.sqrt(_trapezoid(abs_err**2, x_np, axis=1))
    l2_true = np.sqrt(_trapezoid(exacts**2, x_np, axis=1))
    l2_rel = l2_err / l2_true                                 # [M]

    return {
        "x": x_np,
        "mu": np.asarray(mu_eval),
        "psi_pred": preds,
        "psi_exact": exacts,
        "pointwise_rel_err": pointwise_rel,
        "l2_rel_err": l2_rel,
        "l2_rel_err_mean": float(l2_rel.mean()),
        "l2_rel_err_max": float(l2_rel.max()),
    }


def _style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def plot_profiles(eval_out: dict, out_path: Path,
                  title_suffix: str = "") -> None:
    _style()
    mus = eval_out["mu"]
    x = eval_out["x"]
    n = len(mus)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows),
                             squeeze=False)
    for i, mu in enumerate(mus):
        ax = axes[i // cols][i % cols]
        ax.plot(x, eval_out["psi_exact"][i], "k-", lw=2, label="exact")
        ax.plot(x, eval_out["psi_pred"][i], "r--", lw=1.5, label="PINN")
        ax.set_title(f"mu = {mu:.4f}  |  L2 rel = "
                     f"{eval_out['l2_rel_err'][i]:.2e}")
        ax.set_xlabel("x")
        ax.set_ylabel("psi(x)")
        ax.legend(loc="upper right", fontsize=8)
        # Zoom to droplet support
        k = math.sqrt(2.0 * abs(mu))
        half = max(8.0 / k, 5.0)
        ax.set_xlim(-half, half)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"PINN vs analytical droplet  {title_suffix}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pointwise_error(eval_out: dict, out_path: Path,
                         title_suffix: str = "") -> None:
    _style()
    mus = eval_out["mu"]
    x = eval_out["x"]
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.get_cmap("viridis")
    for i, mu in enumerate(mus):
        ax.semilogy(x, eval_out["pointwise_rel_err"][i],
                    color=cmap(i / max(1, len(mus) - 1)),
                    label=f"mu={mu:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("|psi_pred - psi_exact| / max|psi_exact|")
    ax.set_title(f"Pointwise relative error  {title_suffix}")
    # Zoom to droplet cores
    k_max = math.sqrt(2.0 * max(abs(m) for m in mus))
    ax.set_xlim(-12.0 / k_max, 12.0 / k_max)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_l2_vs_mu(eval_out: dict, out_path: Path,
                  title_suffix: str = "") -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(eval_out["mu"], eval_out["l2_rel_err"], "o-")
    ax.set_xlabel("mu")
    ax.set_ylabel("integrated L2 relative error")
    ax.set_title(f"L2 relative error vs mu  {title_suffix}")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_loss_history(history: dict, out_path: Path,
                      title_suffix: str = "") -> None:
    _style()
    fig, ax = plt.subplots(figsize=(7, 4))
    steps = history["step"]
    ax.semilogy(steps, history["loss"], label="total")
    ax.semilogy(steps, history["loss_pde"], label="pde")
    ax.semilogy(steps, history["loss_bc"], label="bc")
    ax.semilogy(steps, history["loss_uniq"], label="uniq")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"Training loss  {title_suffix}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_train_vs_test_profiles(
    model: MLP, cfg: TrainConfig,
    mu_train: np.ndarray, mu_test: np.ndarray,
    out_path: Path, title_suffix: str = "",
    n_x: int = 2001,
) -> None:
    """Two-row comparison plot:
        row 1: mu values INSIDE the training range [mu_min, mu_max]   ("train")
        row 2: mu values OUTSIDE that range                           ("test")
    Each panel overlays PINN prediction against the analytical solution.

    Note: with the (x, mu) -> psi formulation there is no discrete set of
    "trained mu" values -- collocation is dense in the 2D rectangle. So
    "train" means in-distribution mu and "test" means out-of-distribution mu
    (i.e. generalization / extrapolation).
    """
    _style()
    device = next(model.parameters()).device
    x_np = np.linspace(-cfg.L, cfg.L, n_x)
    x_t = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(-1)

    def _eval_one(mu: float) -> tuple[np.ndarray, np.ndarray, float]:
        with torch.no_grad():
            mu_t = torch.full_like(x_t, float(mu))
            pred = model(x_t, mu_t).cpu().numpy()
        exact = psi_exact_np(x_np, float(mu))
        # sign-align on the central value
        if pred[n_x // 2] * exact[n_x // 2] < 0:
            pred = -pred
        num = np.sqrt(_trapezoid((pred - exact) ** 2, x_np))
        den = np.sqrt(_trapezoid(exact ** 2, x_np))
        l2 = num / den
        return pred, exact, float(l2)

    n_train = len(mu_train)
    n_test = len(mu_test)
    n_cols = max(n_train, n_test)
    fig, axes = plt.subplots(
        2, n_cols, figsize=(3.6 * n_cols, 5.6), squeeze=False,
    )

    for i, mu in enumerate(mu_train):
        pred, exact, l2 = _eval_one(mu)
        ax = axes[0][i]
        ax.plot(x_np, exact, "k-", lw=2, label="exact")
        ax.plot(x_np, pred, "r--", lw=1.5, label="PINN")
        ax.set_title(f"train: mu={mu:.4f}  |  L2rel={l2:.2e}", fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("psi")
        ax.legend(fontsize=7, loc="upper right")
        k = math.sqrt(2.0 * abs(mu)); ax.set_xlim(-max(8.0 / k, 5.0),
                                                   max(8.0 / k, 5.0))
    for j in range(n_train, n_cols):
        axes[0][j].axis("off")

    for i, mu in enumerate(mu_test):
        pred, exact, l2 = _eval_one(mu)
        ax = axes[1][i]
        ax.plot(x_np, exact, "k-", lw=2, label="exact")
        ax.plot(x_np, pred, "b--", lw=1.5, label="PINN")
        ax.set_title(f"test: mu={mu:.4f}  |  L2rel={l2:.2e}", fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("psi")
        ax.legend(fontsize=7, loc="upper right")
        k = math.sqrt(2.0 * abs(mu)); ax.set_xlim(-max(8.0 / k, 5.0),
                                                   max(8.0 / k, 5.0))
    for j in range(n_test, n_cols):
        axes[1][j].axis("off")

    # Row labels
    axes[0][0].annotate(
        "in-distribution\n(mu in [mu_min, mu_max])",
        xy=(-0.32, 0.5), xycoords="axes fraction",
        ha="center", va="center", rotation=90, fontsize=9, color="darkred",
    )
    axes[1][0].annotate(
        "out-of-distribution\n(mu outside training range)",
        xy=(-0.32, 0.5), xycoords="axes fraction",
        ha="center", va="center", rotation=90, fontsize=9, color="darkblue",
    )

    fig.suptitle(f"PINN vs analytical — train vs test mu   {title_suffix}",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_error_heatmap(
    model: MLP, cfg: TrainConfig, out_path: Path,
    title_suffix: str = "",
    n_mu: int = 80, n_x: int = 401,
    mu_min_eval: float | None = None,
    mu_max_eval: float | None = None,
) -> None:
    """Heatmap of pointwise relative error |psi_pred - psi_exact| / max|psi_exact|
    over a dense grid of (x, mu).

    The vertical extent of the heatmap spans [mu_min_eval, mu_max_eval], which
    defaults to the training range but can be extended to visualize OOD behavior.
    The training range is outlined on the plot for reference.
    """
    _style()
    device = next(model.parameters()).device
    if mu_min_eval is None:
        mu_min_eval = cfg.mu_min
    if mu_max_eval is None:
        mu_max_eval = cfg.mu_max

    x_np = np.linspace(-cfg.L, cfg.L, n_x)
    mu_np = np.linspace(mu_min_eval, mu_max_eval, n_mu)

    err = np.zeros((n_mu, n_x))
    for i, mu in enumerate(mu_np):
        x_t = torch.tensor(x_np, dtype=torch.float32,
                           device=device).unsqueeze(-1)
        mu_t = torch.full_like(x_t, float(mu))
        pred = model(x_t, mu_t).cpu().numpy()
        exact = psi_exact_np(x_np, float(mu))
        if pred[n_x // 2] * exact[n_x // 2] < 0:
            pred = -pred
        denom = max(np.max(np.abs(exact)), 1e-30)
        err[i] = np.abs(pred - exact) / denom

    # Zoom in x to the droplet support for readability
    k_max = math.sqrt(2.0 * max(abs(mu_min_eval), abs(mu_max_eval)))
    k_min = math.sqrt(2.0 * min(abs(mu_min_eval), abs(mu_max_eval)))
    x_half = 12.0 / k_min

    fig, ax = plt.subplots(figsize=(8, 5))
    from matplotlib.colors import LogNorm
    im = ax.imshow(
        err,
        extent=[x_np[0], x_np[-1], mu_np[0], mu_np[-1]],
        aspect="auto", origin="lower",
        cmap="viridis",
        norm=LogNorm(vmin=max(err.min(), 1e-8), vmax=max(err.max(), 1e-7)),
    )
    cbar = fig.colorbar(im, ax=ax, label="|psi_pred - psi_exact| / max|psi_exact|")
    ax.set_xlabel("x")
    ax.set_ylabel("mu")
    ax.set_xlim(-x_half, x_half)
    ax.set_title(f"Pointwise relative error  {title_suffix}")
    # Outline training range
    ax.axhline(cfg.mu_min, color="red", lw=0.8, ls=":", alpha=0.7)
    ax.axhline(cfg.mu_max, color="red", lw=0.8, ls=":", alpha=0.7,
               label="training mu range")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
#  Self-check (analytical norm formula)
# =============================================================================

def _sanity_check_norm(verbose: bool = True) -> None:
    """Numerically verify norm_exact against trapezoidal integral."""
    for mu in [-0.02, -0.05, -0.1, -0.15, -0.2]:
        x = np.linspace(-200.0, 200.0, 200001)
        psi = psi_exact_np(x, mu)
        N_num = _trapezoid(psi**2, x)
        N_ana = norm_exact(mu)
        if verbose:
            print(f"mu={mu:+.3f}  N_numerical={N_num:.6f}  "
                  f"N_analytical={N_ana:.6f}  rel_err="
                  f"{abs(N_num - N_ana) / N_ana:.2e}")


if __name__ == "__main__":
    _sanity_check_norm()