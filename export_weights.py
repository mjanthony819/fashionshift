"""Export trained weights to JSON (base64 Float32) for the browser app and parity tests."""
import base64
import json
import os

import numpy as np
import torch

from src.models import SmallCNN

ROOT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(ROOT, "artifacts")


def b64(arr):
    a = np.ascontiguousarray(arr, dtype=np.float32)
    return base64.b64encode(a.tobytes()).decode("ascii")


def export():
    out = {"format": "f32le-base64", "models": {}}
    for name in ["lr", "cnn", "cnn_aug"]:
        state = torch.load(os.path.join(ART, "checkpoints", f"{name}.pt"),
                           map_location="cpu", weights_only=False)
        sd = state["model"]
        entry = {"mean": state["mean"], "std": state["std"], "params": state.get("params")}
        if name == "lr":
            entry["arch"] = "linear"
            entry["layers"] = {
                "W": {"shape": list(sd["weight"].shape), "data": b64(sd["weight"].numpy())},
                "b": {"shape": list(sd["bias"].shape), "data": b64(sd["bias"].numpy())},
            }
        else:
            entry["arch"] = "smallcnn"
            entry["layers"] = {
                "conv1_w": {"shape": list(sd["conv1.weight"].shape), "data": b64(sd["conv1.weight"].numpy())},
                "conv1_b": {"shape": list(sd["conv1.bias"].shape), "data": b64(sd["conv1.bias"].numpy())},
                "conv2_w": {"shape": list(sd["conv2.weight"].shape), "data": b64(sd["conv2.weight"].numpy())},
                "conv2_b": {"shape": list(sd["conv2.bias"].shape), "data": b64(sd["conv2.bias"].numpy())},
                "fc1_w": {"shape": list(sd["fc1.weight"].shape), "data": b64(sd["fc1.weight"].numpy())},
                "fc1_b": {"shape": list(sd["fc1.bias"].shape), "data": b64(sd["fc1.bias"].numpy())},
                "fc2_w": {"shape": list(sd["fc2.weight"].shape), "data": b64(sd["fc2.weight"].numpy())},
                "fc2_b": {"shape": list(sd["fc2.bias"].shape), "data": b64(sd["fc2.bias"].numpy())},
            }
        out["models"][name] = entry
    path = os.path.join(ART, "weights.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"wrote {path} ({os.path.getsize(path)//1024} KB)")


if __name__ == "__main__":
    export()
