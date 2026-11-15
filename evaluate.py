"""Evaluate the three trained models on the UNTOUCHED official test set.

Produces artifacts/metrics.json and artifacts/predictions.csv.
All perturbations are seeded and deterministic; the same transforms are
re-implemented in the browser for the live stress lab.
"""
import csv
import json
import os

import numpy as np
import torch

from src.data import load_raw, CLASSES
from src.models import SmallCNN

ROOT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(ROOT, "artifacts")
MODELS = ["lr", "cnn", "cnn_aug"]
ECE_BINS = 15


def softmax(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def ece(probs, labels, n_bins=ECE_BINS):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    per_bin = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        n = int(m.sum())
        if n:
            acc_b = float(correct[m].mean())
            conf_b = float(conf[m].mean())
            ece_val += n / len(labels) * abs(acc_b - conf_b)
            per_bin.append({"lo": float(lo), "hi": float(hi), "n": n,
                            "acc": acc_b, "conf": conf_b})
        else:
            per_bin.append({"lo": float(lo), "hi": float(hi), "n": 0,
                            "acc": None, "conf": None})
    return float(ece_val), per_bin


def brier(probs, labels):
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def coverage_risk(probs, labels, steps=20):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    curve = []
    for t in np.linspace(0.0, 0.99, steps):
        keep = conf >= t
        cov = float(keep.mean())
        risk = float(1.0 - correct[keep].mean()) if keep.any() else None
        curve.append({"threshold": float(t), "coverage": cov, "risk": risk})
    return curve


def per_class(probs, labels):
    pred = probs.argmax(axis=1)
    out = []
    for c, name in enumerate(CLASSES):
        tp = int(((pred == c) & (labels == c)).sum())
        fp = int(((pred == c) & (labels != c)).sum())
        fn = int(((pred != c) & (labels == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out.append({"class": name, "precision": prec, "recall": rec, "f1": f1,
                    "support": int((labels == c).sum())})
    return out


def confusion(probs, labels):
    pred = probs.argmax(axis=1)
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(labels, pred):
        cm[t][p] += 1
    return cm.tolist()


# ---------------- deterministic perturbations (mirror js/inference.js) --------

def blur3x3(imgs):
    a = imgs.reshape(-1, 28, 28)
    p = np.pad(a, ((0, 0), (1, 1), (1, 1)), mode="edge")
    k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    o = np.zeros_like(a)
    for kr in range(3):
        for kc in range(3):
            o += p[:, kr:kr + 28, kc:kc + 28] * k[kr, kc]
    return o.reshape(-1, 784).astype(np.float32)


def occlude(imgs, size=8):
    out = imgs.copy()
    for i in range(len(out)):
        rng = np.random.RandomState(7777 + i)
        r = rng.randint(0, 28 - size + 1)
        c = rng.randint(0, 28 - size + 1)
        a = out[i].reshape(28, 28)
        a[r:r + size, c:c + size] = 0.0
    return out


def translate(imgs, max_shift=4):
    out = np.zeros_like(imgs)
    for i in range(len(out)):
        rng = np.random.RandomState(5555 + i)
        dy = rng.randint(-max_shift, max_shift + 1)
        dx = rng.randint(-max_shift, max_shift + 1)
        a = imgs[i].reshape(28, 28)
        o = np.zeros_like(a)
        rs = slice(max(0, dy), min(28, 28 + dy))
        cs = slice(max(0, dx), min(28, 28 + dx))
        rr = slice(max(0, -dy), min(28, 28 - dy))
        cr = slice(max(0, -dx), min(28, 28 - dx))
        o[rs, cs] = a[rr, cr]
        out[i] = o.reshape(-1)
    return out


STRESSES = {"blur": blur3x3, "occlusion": occlude, "translation": translate}


def load_models():
    models = {}
    for name in MODELS:
        state = torch.load(os.path.join(ART, "checkpoints", f"{name}.pt"),
                           map_location="cpu", weights_only=False)
        if name == "lr":
            m = torch.nn.Linear(784, 10)
        else:
            m = SmallCNN()
        m.load_state_dict(state["model"])
        m.eval()
        models[name] = (m, state["mean"], state["std"], state.get("history", []),
                        state.get("params"))
    return models


def predict_probs(model, x, mean, std, bs=512):
    xs = (x - mean) / std
    outs = []
    with torch.no_grad():
        for i in range(0, len(xs), bs):
            outs.append(model(torch.from_numpy(xs[i:i + bs])).numpy())
    return softmax(np.concatenate(outs))


def evaluate_set(name, probs, labels):
    pred = probs.argmax(axis=1)
    acc = float((pred == labels).mean())
    pc = per_class(probs, labels)
    macro_f1 = float(np.mean([c["f1"] for c in pc]))
    nll = float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)) .mean())
    e, bins = ece(probs, labels)
    return {
        "accuracy": acc, "macro_f1": macro_f1, "nll": nll, "brier": brier(probs, labels),
        "ece": e, "ece_bins": bins, "per_class": pc,
        "confusion": confusion(probs, labels),
        "coverage_risk": coverage_risk(probs, labels),
    }


def main():
    os.makedirs(ART, exist_ok=True)
    _, _, x_test, y_test, _ = load_raw(include_test=True)
    x_test = x_test.reshape(len(x_test), -1)
    models = load_models()
    metrics = {"test_count": len(x_test), "classes": CLASSES, "models": {}}
    probs_cache = {}
    for name, (m, mean, std, hist, params) in models.items():
        probs = predict_probs(m, x_test, mean, std)
        probs_cache[name] = probs
        res = evaluate_set(name, probs, y_test)
        res["params"] = params
        res["val_history"] = hist
        metrics["models"][name] = res
        print(f"[{name}] acc={res['accuracy']:.4f} macroF1={res['macro_f1']:.4f} "
              f"nll={res['nll']:.4f} brier={res['brier']:.4f} ece={res['ece']:.4f}")

    stress_results = {}
    for sname, fn in STRESSES.items():
        xs = fn(x_test)
        stress_results[sname] = {}
        for name, (m, mean, std, _, _) in models.items():
            probs = predict_probs(m, xs, mean, std)
            pred = probs.argmax(axis=1)
            pc = per_class(probs, y_test)
            stress_results[sname][name] = {
                "accuracy": float((pred == y_test).mean()),
                "macro_f1": float(np.mean([c["f1"] for c in pc])),
                "mean_confidence": float(probs.max(axis=1).mean()),
                "mean_confidence_correct": float(probs.max(axis=1)[pred == y_test].mean()),
                "mean_confidence_wrong": float(probs.max(axis=1)[pred != y_test].mean()),
            }
        print(f"[stress {sname}] " + " ".join(
            f"{n}: acc={stress_results[sname][n]['accuracy']:.4f}" for n in MODELS))
    metrics["stress"] = stress_results

    # predictions CSV
    with open(os.path.join(ART, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        header = ["idx", "true"]
        for n in MODELS:
            header += [f"{n}_pred", f"{n}_conf"]
        w.writerow(header)
        preds = {n: probs_cache[n].argmax(axis=1) for n in MODELS}
        confs = {n: probs_cache[n].max(axis=1) for n in MODELS}
        for i in range(len(y_test)):
            row = [i, int(y_test[i])]
            for n in MODELS:
                row += [int(preds[n][i]), round(float(confs[n][i]), 5)]
            w.writerow(row)

    with open(os.path.join(ART, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("saved artifacts/metrics.json and artifacts/predictions.csv")


if __name__ == "__main__":
    main()
