from __future__ import annotations

import torch
from .config import PipelineConfig
from .utils.logging import console
from .utils.seed import seed_everything


def bootstrap(cfg: dict) -> torch.device:
    want = cfg["train"]["device"]
    if want == "cuda" and not torch.cuda.is_available():
        console().print("[yellow]CUDA not available; falling back to CPU.[/yellow]")
    device = torch.device("cuda:0" if (want == "cuda" and torch.cuda.is_available()) else "cpu")
    deterministic = bool(cfg["train"].get("deterministic", False))
    torch.set_float32_matmul_precision(cfg["train"]["matmul_precision"])
    seed_everything(int(cfg.get("tuning", {}).get("seed", 2025)), deterministic=deterministic)
    return device


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    cfg = PipelineConfig.from_files(args.config).to_dict()
    device = bootstrap(cfg)
    console().print(f"[bold green]Bootstrap complete. Device: {device}[/bold green]")


if __name__ == "__main__":
    main()
