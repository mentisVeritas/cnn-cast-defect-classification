"""Utility helpers: config loading, device selection, seeding, logging."""

from __future__ import annotations

import logging
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve config paths relative to project root (works from notebook/any cwd)."""
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (root or project_root()) / p


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        path = resolve_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(config: dict[str, Any]) -> torch.device:
    device_cfg = str(config["training"].get("device", "auto")).lower()

    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_cfg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_num_workers(config: dict[str, Any]) -> int:
    nw = config["training"].get("num_workers", "auto")
    if str(nw).lower() == "auto":
        # Windows/macOS: 0 avoids DataLoader multiprocessing issues
        if platform.system() in ("Darwin", "Windows"):
            return 0
        return 4
    return int(nw)


def get_dataloader_kwargs(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    num_workers = get_num_workers(config)
    kwargs: dict[str, Any] = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def describe_runtime(config: dict[str, Any], device: torch.device) -> str:
    parts = [
        f"OS={platform.system()}",
        f"Device={device}",
        f"num_workers={get_num_workers(config)}",
    ]
    if device.type == "cuda":
        parts.append(f"GPU={torch.cuda.get_device_name(0)}")
    return " | ".join(parts)


def setup_logging(config: dict[str, Any]) -> logging.Logger:
    logger = logging.getLogger("cnn_cast_defect")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.addHandler(logging.StreamHandler())
    logger.handlers[-1].setFormatter(formatter)

    if config["logging"].get("log_to_file", True):
        log_path = resolve_path(config["logging"]["log_file"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def ensure_output_dirs(config: dict[str, Any]) -> None:
    paths = [
        config["logging"]["log_file"],
        config["checkpoint"]["latest_checkpoint"],
        config["checkpoint"]["best_model"],
        config["evaluation"]["metrics_file"],
        config["evaluation"]["classification_report_file"],
        config["evaluation"]["confusion_matrix_plot"],
        config["visualization"]["loss_curve"],
        config["visualization"]["accuracy_curve"],
    ]
    for p in paths:
        resolve_path(p).parent.mkdir(parents=True, exist_ok=True)
