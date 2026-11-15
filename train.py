"""Train the three FashionShift models on Fashion-MNIST.

Models:
  1. lr        - multinomial logistic regression on standardized raw pixels (scikit-learn)
  2. cnn       - SmallCNN, plain training
  3. cnn_aug   - SmallCNN, trained with rotation/translation/noise augmentation

Resumable: --epochs adds epochs on top of any existing checkpoint, so long
training fits inside short shell calls. Deterministic seeds throughout.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from src.data import (load_raw, split_train_val, normalization_stats,
                      write_split_manifest, DATA_DIR)
from src.models import SmallCNN, param_count

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(ROOT, "artifacts", "checkpoints")
SEED = 20261116


def set_seeds(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(False)


def tensors(x, y=None):
    xt = torch.from_numpy(x.reshape(len(x), -1))
    if y is None:
        return xt
    return xt, torch.from_numpy(y)


def standardize(x, mean, std):
    return (x - mean) / std


def augment_batch(xb, rng):
    """xb: (N,784) torch tensor in [0,1]. Rotation +-15 deg, translation +-2 px, gaussian noise."""
    n = xb.size(0)
    x = xb.view(-1, 1, 28, 28)
    # random rotation + translation via affine grid
    angles = torch.from_numpy(rng.uniform(-15, 15, n).astype(np.float32)) * (np.pi / 180.0)
    shifts = torch.from_numpy(rng.randint(-2, 3, (n, 2)).astype(np.float32)) / 14.0  # fraction of half-width
    cos, sin = torch.cos(angles), torch.sin(angles)
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 0, 2] = shifts[:, 0]
    theta[:, 1, 2] = shifts[:, 1]
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
    x = torch.nn.functional.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    noise = torch.from_numpy(rng.normal(0, 0.05, x.shape).astype(np.float32))
    return (x + noise).clamp(0, 1).view(n, 784)


def train_cnn(name, epochs, x_tr, y_tr, x_va, y_va, mean, std, augment=False):
    set_seeds(SEED + (1 if augment else 0))
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"{name}.pt")
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_epoch = 0
    history = []
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_epoch = state["epoch"]
        history = state["history"]
        print(f"[{name}] resuming at epoch {start_epoch}")
    x_tr_s, y_tr_t = tensors(standardize(x_tr, mean, std), y_tr)
    x_va_s, y_va_t = tensors(standardize(x_va, mean, std), y_va)
    x_tr_raw = torch.from_numpy(x_tr)  # [0,1] for augmentation
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.RandomState(SEED + (100 if augment else 0) + start_epoch)
    bs = 128
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(x_tr_s.size(0))
        total_loss, seen = 0.0, 0
        for i in range(0, x_tr_s.size(0), bs):
            idx = perm[i:i + bs]
            if augment:
                xb = standardize(augment_batch(x_tr_raw[idx], rng), mean, std)
            else:
                xb = x_tr_s[idx]
            yb = y_tr_t[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
            seen += len(idx)
        model.eval()
        with torch.no_grad():
            va_logits = model(x_va_s)
            va_loss = loss_fn(va_logits, y_va_t).item()
            va_acc = (va_logits.argmax(1) == y_va_t).float().mean().item()
        history.append({"epoch": epoch + 1, "train_loss": total_loss / seen,
                        "val_loss": va_loss, "val_acc": va_acc})
        print(f"[{name}] epoch {epoch+1} train_loss={total_loss/seen:.4f} "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} ({time.time()-t0:.0f}s)")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch + 1, "history": history,
                    "mean": mean, "std": std, "params": param_count(model)}, ckpt_path)
    return history


def train_lr(epochs, x_tr, y_tr, x_va, y_va, mean, std):
    """Multinomial logistic regression: a single Linear(784->10) trained with
    cross-entropy (torch), resumable like the CNNs."""
    set_seeds(SEED + 2)
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, "lr.pt")
    model = nn.Linear(784, 10)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_epoch, history = 0, []
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_epoch = state["epoch"]
        history = state["history"]
        print(f"[lr] resuming at epoch {start_epoch}")
    x_tr_s, y_tr_t = tensors(standardize(x_tr, mean, std), y_tr)
    x_va_s, y_va_t = tensors(standardize(x_va, mean, std), y_va)
    loss_fn = nn.CrossEntropyLoss()
    bs = 512
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(x_tr_s.size(0))
        total_loss, seen = 0.0, 0
        for i in range(0, x_tr_s.size(0), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(x_tr_s[idx]), y_tr_t[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
            seen += len(idx)
        model.eval()
        with torch.no_grad():
            va_logits = model(x_va_s)
            va_loss = loss_fn(va_logits, y_va_t).item()
            va_acc = (va_logits.argmax(1) == y_va_t).float().mean().item()
        history.append({"epoch": epoch + 1, "train_loss": total_loss / seen,
                        "val_loss": va_loss, "val_acc": va_acc})
        print(f"[lr] epoch {epoch+1} train_loss={total_loss/seen:.4f} "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} ({time.time()-t0:.0f}s)")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch + 1, "history": history,
                    "mean": mean, "std": std,
                    "params": sum(p.numel() for p in model.parameters())}, ckpt_path)
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lr", "cnn", "cnn_aug", "all"], default="all")
    ap.add_argument("--epochs", type=int, default=3, help="epochs to ADD per CNN this run")
    args = ap.parse_args()

    x_full, y_full, _, _, hashes = load_raw(include_test=False)
    x_tr, y_tr, x_va, y_va, tr_idx, va_idx = split_train_val(x_full, y_full)
    del x_full, y_full
    mean, std = normalization_stats(x_tr)
    print(f"train={len(x_tr)} val={len(x_va)} mean={mean:.4f} std={std:.4f}")
    write_split_manifest(tr_idx, va_idx, hashes,
                         os.path.join(ROOT, "artifacts", "split_manifest.json"))
    with open(os.path.join(ROOT, "artifacts", "norm.json"), "w") as f:
        json.dump({"mean": mean, "std": std}, f)

    if args.model in ("lr", "all"):
        train_lr(args.epochs, x_tr, y_tr, x_va, y_va, mean, std)
    if args.model in ("cnn", "all"):
        train_cnn("cnn", args.epochs, x_tr, y_tr, x_va, y_va, mean, std, augment=False)
    if args.model in ("cnn_aug", "all"):
        train_cnn("cnn_aug", args.epochs, x_tr, y_tr, x_va, y_va, mean, std, augment=True)


if __name__ == "__main__":
    main()
