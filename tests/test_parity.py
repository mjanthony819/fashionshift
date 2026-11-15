"""Generate reference probabilities from the PyTorch models for parity tests."""
import json
import os

import numpy as np
import torch

from src.data import load_raw
from src.models import SmallCNN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")
N = 48


def main():
    _, _, x_test, y_test, _ = load_raw(include_test=True)
    x_test = x_test.reshape(len(x_test), -1)
    rng = np.random.RandomState(99)
    idx = rng.choice(len(x_test), N, replace=False)
    xs = x_test[idx]
    ref = {"idx": idx.tolist(), "images_b64": None, "models": {}}
    import base64
    ref["images_b64"] = base64.b64encode(
        (xs * 255).round().astype(np.uint8).tobytes()).decode("ascii")
    for name in ["lr", "cnn", "cnn_aug"]:
        state = torch.load(os.path.join(ART, "checkpoints", f"{name}.pt"),
                           map_location="cpu", weights_only=False)
        model = torch.nn.Linear(784, 10) if name == "lr" else SmallCNN()
        model.load_state_dict(state["model"])
        model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(((xs - state["mean"]) / state["std"]).astype(np.float32))
            probs = torch.softmax(model(xt), dim=1).numpy()
        ref["models"][name] = probs.round(6).tolist()
    with open(os.path.join(ART, "parity_reference.json"), "w") as f:
        json.dump(ref, f)
    print(f"reference for {N} images x 3 models written")


if __name__ == "__main__":
    main()
