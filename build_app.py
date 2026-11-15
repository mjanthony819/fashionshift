"""Assemble the single-file offline app index.html from app/template.html.

Injects: weights.json, metrics.json, a gallery of test images, and a failure
gallery. Also writes artifacts/build_manifest.json with SHA256 hashes.
"""
import base64
import hashlib
import json
import os

import numpy as np

from src.data import load_raw

ROOT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(ROOT, "artifacts")
GALLERY_N = 96
FAIL_N = 36


def b64_bytes(arr):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.uint8).tobytes()).decode("ascii")


def main():
    _, _, x_test, y_test, _ = load_raw(include_test=True)
    x_u8 = (x_test.reshape(len(x_test), -1) * 255).round().astype(np.uint8)

    with open(os.path.join(ART, "weights.json")) as f:
        weights = json.load(f)
    with open(os.path.join(ART, "metrics.json")) as f:
        metrics = json.load(f)

    rng = np.random.RandomState(4242)
    gal_idx = rng.choice(len(x_test), GALLERY_N, replace=False)
    gallery = [{"i": int(i), "y": int(y_test[i]), "p": b64_bytes(x_u8[i])} for i in gal_idx]

    # failure gallery: confident mistakes of cnn_aug
    import torch
    from src.models import SmallCNN
    state = torch.load(os.path.join(ART, "checkpoints", "cnn_aug.pt"),
                       map_location="cpu", weights_only=False)
    model = SmallCNN()
    model.load_state_dict(state["model"])
    model.eval()
    with torch.no_grad():
        xs = torch.from_numpy(((x_test.reshape(len(x_test), -1) - state["mean"]) / state["std"]).astype(np.float32))
        probs = torch.softmax(model(xs), dim=1).numpy()
    pred = probs.argmax(1)
    conf = probs.max(1)
    wrong = np.where(pred != y_test)[0]
    # prefer confident mistakes, spread across classes
    order = wrong[np.argsort(-conf[wrong])]
    seen, fails = set(), []
    for i in order:
        c = int(y_test[i])
        if seen.count(c) if isinstance(seen, list) else 0:
            pass
        fails.append({"i": int(i), "y": c, "pred": int(pred[i]), "conf": float(conf[i]),
                      "p": b64_bytes(x_u8[i])})
        if len(fails) >= FAIL_N:
            break

    with open(os.path.join(ROOT, "app", "template.html")) as f:
        html = f.read()
    with open(os.path.join(ROOT, "js", "inference.js")) as f:
        inference_js = f.read()

    html = (html
            .replace("__WEIGHTS__", json.dumps(weights))
            .replace("__METRICS__", json.dumps(metrics))
            .replace("__GALLERY__", json.dumps(gallery))
            .replace("__FAILURES__", json.dumps(fails))
            .replace("__INFERENCE_JS__", inference_js))

    out = os.path.join(ROOT, "index.html")
    with open(out, "w") as f:
        f.write(html)

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    manifest = {
        "index_html": {"bytes": os.path.getsize(out), "sha256": sha(out)},
        "weights_json": {"bytes": os.path.getsize(os.path.join(ART, "weights.json")),
                         "sha256": sha(os.path.join(ART, "weights.json"))},
        "metrics_json_sha256": sha(os.path.join(ART, "metrics.json")),
        "gallery_count": len(gallery), "failure_count": len(fails),
    }
    with open(os.path.join(ART, "build_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"index.html {manifest['index_html']['bytes']/1048576:.2f} MB, "
          f"{len(gallery)} gallery, {len(fails)} failures")


if __name__ == "__main__":
    main()
