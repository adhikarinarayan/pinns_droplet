"""
Single entry point for running a PINN droplet experiment.

Usage
-----
    python run_experiment.py --config config.yaml

This will:
  1. Load the config (YAML), build a TrainConfig.
  2. Train Adam -> L-BFGS.
  3. Evaluate on the eval mu-grid.
  4. Save, under results/<experiment.name>/:
       - config.yaml         (a copy of the config used)
       - metrics.json        (scalar + per-mu metrics)
       - history.npz         (loss curves)
       - eval.npz            (predictions, exact, errors on the grid)
       - profiles.png        (psi_pred vs psi_exact per mu)
       - pointwise_error.png (|psi_pred - psi_exact|/max|psi_exact| per mu)
       - l2_vs_mu.png        (integrated L2 rel error vs mu)
       - loss_history.png    (all loss components on log scale)
       - notes.md            (a copy of the `experiment.notes` field)
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from pinn_droplet import (
    TrainConfig,
    evaluate,
    plot_error_heatmap,
    plot_l2_vs_mu,
    plot_loss_history,
    plot_pointwise_error,
    plot_profiles,
    plot_train_vs_test_profiles,
    train,
)


def _build_train_config(cfg_yaml: dict) -> TrainConfig:
    """Flatten the nested YAML into TrainConfig kwargs."""
    p = cfg_yaml["problem"]
    n = cfg_yaml["network"]
    c = cfg_yaml["collocation"]
    u = cfg_yaml["uniqueness"]
    w = cfg_yaml["loss_weights"]
    o = cfg_yaml["optimization"]
    m = cfg_yaml["misc"]
    return TrainConfig(
        L=float(p["L"]),
        mu_min=float(p["mu_min"]),
        mu_max=float(p["mu_max"]),
        depth=int(n["depth"]),
        width=int(n["width"]),
        activation=str(n["activation"]),
        n_collocation=int(c["n_collocation"]),
        n_mu_train=int(c["n_mu_train"]),
        n_boundary=int(c["n_boundary"]),
        resample_every=int(c["resample_every"]),
        uniqueness_method=str(u["method"]),
        norm_loss_type=str(u.get("norm_loss_type", "relative")),
        anchor_weight_in_hybrid=float(u.get("anchor_weight_in_hybrid", 1.0)),
        anchor_x_list=tuple(u.get("anchor_x_list", [0.0])),
        n_supervised_points=int(u["n_supervised_points"]),
        anchor_x=float(u["anchor_x"]),
        mu_anchor_full=tuple(float(v) for v in u.get("mu_anchor_full", [])),
        n_x_anchor_full=int(u.get("n_x_anchor_full", 201)),
        w_pde=float(w["w_pde"]),
        w_bc=float(w["w_bc"]),
        w_uniq=float(w["w_uniq"]),
        adam_iters=int(o["adam_iters"]),
        adam_lr=float(o["adam_lr"]),
        adam_lr_final=float(o["adam_lr_final"]),
        lbfgs_iters=int(o["lbfgs_iters"]),
        lbfgs_history=int(o["lbfgs_history"]),
        lbfgs_tol_grad=float(o["lbfgs_tol_grad"]),
        lbfgs_tol_change=float(o["lbfgs_tol_change"]),
        seed=int(m["seed"]),
        log_every=int(m["log_every"]),
        device=str(m["device"]),
    )


def main(config_path: str, results_root: str = "results") -> None:
    cfg_path = Path(config_path)
    with cfg_path.open() as f:
        cfg_yaml = yaml.safe_load(f)

    exp_name = cfg_yaml["experiment"]["name"]
    out_dir = Path(results_root) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] experiment: {exp_name}")
    print(f"[run] output dir: {out_dir.resolve()}")

    # Save a copy of the config and notes so results are self-documenting
    shutil.copy(cfg_path, out_dir / "config.yaml")
    (out_dir / "notes.md").write_text(
        cfg_yaml["experiment"].get("notes", "").strip() + "\n",
    )

    # Build TrainConfig and train
    train_cfg = _build_train_config(cfg_yaml)
    model, history = train(train_cfg, out_dir)

    # Save loss history
    np.savez(out_dir / "history.npz", **{
        k: np.array(v) if isinstance(v, list) else np.array([v])
        for k, v in history.items()
    })

    # Evaluate
    mu_eval = np.array(cfg_yaml["evaluation"]["mu_eval"], dtype=float)
    n_x_eval = int(cfg_yaml["evaluation"]["n_x_eval"])
    ev = evaluate(model, train_cfg, mu_eval, n_x=n_x_eval)

    # Save raw arrays
    np.savez(
        out_dir / "eval.npz",
        x=ev["x"],
        mu=ev["mu"],
        psi_pred=ev["psi_pred"],
        psi_exact=ev["psi_exact"],
        pointwise_rel_err=ev["pointwise_rel_err"],
        l2_rel_err=ev["l2_rel_err"],
    )

    # Save scalar metrics
    metrics = {
        "experiment_name": exp_name,
        "l2_rel_err_mean": ev["l2_rel_err_mean"],
        "l2_rel_err_max": ev["l2_rel_err_max"],
        "l2_rel_err_per_mu": {
            f"{float(mu):.4f}": float(err)
            for mu, err in zip(ev["mu"], ev["l2_rel_err"])
        },
        "train_time_adam_s": history.get("train_time_adam_s"),
        "train_time_lbfgs_s": history.get("train_time_lbfgs_s"),
        "final_loss": history["loss"][-1] if history["loss"] else None,
        "final_loss_pde": history["loss_pde"][-1] if history["loss_pde"] else None,
        "train_config": asdict(train_cfg),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Save all plots
    plot_profiles(ev, out_dir / "profiles.png", f"({exp_name})")
    plot_pointwise_error(ev, out_dir / "pointwise_error.png", f"({exp_name})")
    plot_l2_vs_mu(ev, out_dir / "l2_vs_mu.png", f"({exp_name})")
    plot_loss_history(history, out_dir / "loss_history.png", f"({exp_name})")

    # Train vs test (in-distribution vs out-of-distribution mu) comparison
    mu_train_plot = np.array(
        cfg_yaml["evaluation"].get("mu_train_plot", []), dtype=float,
    )
    mu_test_plot = np.array(
        cfg_yaml["evaluation"].get("mu_test_plot", []), dtype=float,
    )
    if len(mu_train_plot) > 0 or len(mu_test_plot) > 0:
        plot_train_vs_test_profiles(
            model, train_cfg,
            mu_train=mu_train_plot, mu_test=mu_test_plot,
            out_path=out_dir / "train_vs_test_profiles.png",
            title_suffix=f"({exp_name})",
            n_x=n_x_eval,
        )

    # Pointwise error heatmap over (x, mu)
    plot_error_heatmap(
        model, train_cfg,
        out_path=out_dir / "error_heatmap.png",
        title_suffix=f"({exp_name})",
        n_mu=int(cfg_yaml["evaluation"].get("n_mu_heatmap", 80)),
        n_x=int(cfg_yaml["evaluation"].get("n_x_heatmap", 401)),
        mu_min_eval=cfg_yaml["evaluation"].get("mu_min_heatmap"),
        mu_max_eval=cfg_yaml["evaluation"].get("mu_max_heatmap"),
    )

    # Console summary
    print(f"\n[run] ===== SUMMARY: {exp_name} =====")
    print(f"[run] mean L2 rel err (across mu grid): {ev['l2_rel_err_mean']:.3e}")
    print(f"[run]  max L2 rel err (across mu grid): {ev['l2_rel_err_max']:.3e}")
    print(f"[run] per-mu L2 rel errors:")
    for mu, err in zip(ev["mu"], ev["l2_rel_err"]):
        print(f"[run]   mu={mu:+.4f}  ->  {err:.3e}")
    print(f"[run] results written to: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the YAML config file.")
    parser.add_argument("--results_root", type=str, default="results",
                        help="Directory under which experiment folders live.")
    args = parser.parse_args()
    main(args.config, args.results_root)