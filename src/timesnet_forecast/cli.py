from __future__ import annotations

import argparse
import json
import copy
import time
from typing import Dict, Any, List
import optuna

from .config import PipelineConfig, load_yaml
from .train import train_once
from .predict import predict_once
from .utils.logging import console
from .config import save_yaml
from .utils import io as io_utils


def _apply_trial_to_cfg(cfg: Dict[str, Any], space: Dict[str, Any], trial: optuna.Trial) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    for key, spec in space.items():
        path = key.split(".")
        t = spec.get("type")
        if t == "int":
            low, high = int(spec["low"]), int(spec["high"])
            step = int(spec.get("step", 1))
            val = trial.suggest_int(key, low=low, high=high, step=step)
        elif t == "float":
            low, high = float(spec["low"]), float(spec["high"])
            log = bool(spec.get("log", False))
            val = trial.suggest_float(key, low=low, high=high, log=log)
        elif t == "categorical":
            choices = spec["choices"]
            val = trial.suggest_categorical(key, choices)
        else:
            raise ValueError(f"Unknown type: {t}")
        # set
        cur = out
        for p in path[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[path[-1]] = val
    return out


def cmd_train(args: argparse.Namespace) -> None:
    cfg = PipelineConfig.from_files(args.config, overrides=args.override)
    train_once(cfg)


def cmd_predict(args: argparse.Namespace) -> None:
    cfg = PipelineConfig.from_files(args.config, overrides=args.override)
    predict_once(cfg)


def cmd_tune(args: argparse.Namespace) -> None:
    base_cfg = PipelineConfig.from_files(args.config, overrides=args.override)
    base = base_cfg.to_dict()
    space = load_yaml(args.space)

    if base["tuning"]["sampler"] == "tpe_multivariate":
        sampler = optuna.samplers.TPESampler(multivariate=True, seed=base["tuning"]["seed"])
    else:
        sampler = optuna.samplers.TPESampler(seed=base["tuning"]["seed"])
    if base["tuning"]["pruner"] == "median":
        pruner = optuna.pruners.MedianPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    timeout = None
    if base["tuning"]["timeout_min"] is not None:
        timeout = int(base["tuning"]["timeout_min"]) * 60

    def objective(trial: optuna.Trial) -> float:
        cfg_dict = _apply_trial_to_cfg(base, space, trial)
        trial_cfg = PipelineConfig.from_mapping(cfg_dict)
        # train_once returns best validation NLL
        val_nll, _ = train_once(trial_cfg)
        trial.report(val_nll, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return val_nll

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=int(args.n_trials), timeout=timeout)

    console().print(f"[bold magenta]Best value: {study.best_value:.6f}[/bold magenta]")
    console().print(f"[bold]Best params:[/bold] {study.best_trial.params}")

    # Save best params
    art_dir = base["artifacts"]["dir"]
    io_utils.save_json(study.best_trial.params, f"{art_dir}/best_params.json")
    # Optionally save merged config
    best_cfg = _apply_trial_to_cfg(base, space, study.best_trial)
    best_cfg_normalized = PipelineConfig.from_mapping(best_cfg).to_dict()
    save_yaml(best_cfg_normalized, f"{art_dir}/{base['artifacts']['config_file']}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="timesnet-forecast")
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train")
    p_train.add_argument("--config", type=str, default="configs/default.yaml")
    p_train.add_argument("--override", nargs="*", default=[])
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--config", type=str, default="configs/default.yaml")
    p_pred.add_argument("--override", nargs="*", default=[])
    p_pred.set_defaults(func=cmd_predict)

    p_tune = sub.add_parser("tune")
    p_tune.add_argument("--config", type=str, default="configs/default.yaml")
    p_tune.add_argument("--space", type=str, default="configs/search_space.yaml")
    p_tune.add_argument("--n-trials", type=int, default=30)
    p_tune.add_argument("--override", nargs="*", default=[])
    p_tune.set_defaults(func=cmd_tune)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
