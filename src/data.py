"""Fashion-MNIST loading, deterministic split and normalization.

Data source: https://github.com/zalandoresearch/fashion-mnist (MIT License).
The official test set (t10k) is NEVER used for training or model selection.
"""
import gzip
import hashlib
import json
import os
import struct

import numpy as np

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic {magic} in {path}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows, cols).astype(np.float32) / 255.0


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic {magic} in {path}"
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(data_dir=DATA_DIR, include_test=True):
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    paths = {k: os.path.join(data_dir, v) for k, v in files.items()}
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} missing. Download the four official .gz files from "
                "https://github.com/zalandoresearch/fashion-mnist into data/ (see data/README.md)."
            )
    x_train_full = _read_idx_images(paths["train_images"])
    y_train_full = _read_idx_labels(paths["train_labels"])
    if include_test:
        x_test = _read_idx_images(paths["test_images"])
        y_test = _read_idx_labels(paths["test_labels"])
    else:
        x_test = y_test = None
    hashes = {k: sha256_file(p) for k, p in paths.items()}
    return x_train_full, y_train_full, x_test, y_test, hashes


def split_train_val(x, y, val_size=10000, seed=20261116):
    """Deterministic stratified-ish split: shuffle with fixed seed, take last val_size."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(x))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    return (x[train_idx], y[train_idx], x[val_idx], y[val_idx],
            train_idx.tolist(), val_idx.tolist())


def normalization_stats(x_train):
    """Mean/std over the training split only (flattened pixels)."""
    return float(x_train.mean()), float(x_train.std() + 1e-8)


def write_split_manifest(train_idx, val_idx, hashes, out_path, seed=20261116):
    manifest = {
        "seed": seed,
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "test_count": 10000,
        "train_indices_sha256": hashlib.sha256(json.dumps(train_idx).encode()).hexdigest(),
        "val_indices_sha256": hashlib.sha256(json.dumps(val_idx).encode()).hexdigest(),
        "test_set": "official t10k, untouched by training/model selection",
        "source_files_sha256": hashes,
    }
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
