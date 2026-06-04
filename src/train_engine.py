"""Training loop: loss, accuracy, checkpoints, early stopping."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import build_model
from src.utils import ensure_output_dirs, resolve_path

logger = logging.getLogger("cnn_cast_defect")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
    desc: str = "Train",
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc=desc, leave=False)
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            bs = images.size(0)
            running_loss += loss.item() * bs
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += bs

            loop.set_postfix(loss=f"{running_loss / total:.3f}", acc=f"{correct / total:.2f}")

    return running_loss / max(total, 1), correct / max(total, 1)


def save_checkpoint(path: Path, epoch: int, model, optimizer, best_val_loss, patience, history, class_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "patience_counter": patience,
            "history": history,
            "class_names": class_names,
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer, device):
    data = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(data["model_state_dict"])
    optimizer.load_state_dict(data["optimizer_state_dict"])
    return data


class Trainer:
    def __init__(
        self,
        config,
        train_loader,
        val_loader,
        class_names,
        device,
        model: nn.Module | None = None,
        latest_checkpoint: str | Path | None = None,
        best_checkpoint: str | Path | None = None,
        resume_training: bool | None = None,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names
        self.device = device

        self.model = (model if model is not None else build_model(config, len(class_names))).to(
            device
        )
        lr = config["training"]["learning_rate"]
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }
        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_time_seconds = 0.0

        self._latest_checkpoint = resolve_path(
            latest_checkpoint or config["checkpoint"]["latest_checkpoint"]
        )
        self._best_checkpoint = resolve_path(
            best_checkpoint or config["checkpoint"]["best_model"]
        )
        if resume_training is None:
            self._resume_training = config["checkpoint"].get("resume_training", True)
        else:
            self._resume_training = resume_training

        ensure_output_dirs(config)
        self._maybe_resume()

    def _maybe_resume(self):
        path = self._latest_checkpoint
        if self.config["checkpoint"].get("start_from_scratch", False):
            logger.info("Training from scratch.")
            return
        if not self._resume_training or not path.exists():
            logger.info("Training from scratch.")
            return

        data = load_checkpoint(path, self.model, self.optimizer, self.device)
        self.start_epoch = data["epoch"] + 1
        self.history = data.get("history", self.history)
        self.best_val_loss = data.get("best_val_loss", data.get("best_metric", self.best_val_loss))
        self.patience_counter = data.get("patience_counter", 0)
        logger.info(f"Resumed from epoch {data['epoch']}, next epoch: {self.start_epoch}")

    def train(self) -> dict:
        epochs = self.config["training"]["epochs"]
        latest = self._latest_checkpoint
        best = self._best_checkpoint
        es = self.config["training"].get("early_stopping", {})
        es_on = es.get("enabled", False)
        es_patience = es.get("patience", 5)

        t0 = time.perf_counter()
        for epoch in range(self.start_epoch, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            train_loss, train_acc = _run_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.device,
                self.optimizer,
                desc="Training",
            )
            val_loss, val_acc = _run_epoch(
                self.model,
                self.val_loader,
                self.criterion,
                self.device,
                desc="Validation",
            )

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_accuracy"].append(val_acc)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                if self.config["checkpoint"].get("save_best_model", True):
                    save_checkpoint(
                        best,
                        epoch,
                        self.model,
                        self.optimizer,
                        self.best_val_loss,
                        self.patience_counter,
                        self.history,
                        self.class_names,
                    )
            elif es_on:
                self.patience_counter += 1

            save_checkpoint(
                latest,
                epoch,
                self.model,
                self.optimizer,
                self.best_val_loss,
                self.patience_counter,
                self.history,
                self.class_names,
            )

            msg = (
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"train_acc={train_acc:.2%} val_acc={val_acc:.2%}"
            )
            print(msg)
            logger.info(f"Epoch {epoch}/{epochs} {msg}")

            if es_on and self.patience_counter >= es_patience:
                print(f"Early stopping (patience={es_patience})")
                break

        self.training_time_seconds = time.perf_counter() - t0

        out = resolve_path(self.config["evaluation"]["metrics_file"]).parent / "training_history.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        print(f"\nDone. Best model: {best}")
        return self.history
