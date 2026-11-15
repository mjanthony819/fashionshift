"""Sanity checks for the metric implementations against sklearn references."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import ece, brier, softmax


def main():
    rng = np.random.RandomState(0)
    probs = softmax(rng.normal(size=(2000, 10)))
    labels = rng.randint(0, 10, 2000)

    from sklearn.metrics import brier_score_loss, log_loss
    # multi-class Brier (sum over classes) == sum of per-class sklearn brier scores
    ref = sum(brier_score_loss((labels == c).astype(int), probs[:, c]) for c in range(10))
    mine = brier(probs, labels)
    assert abs(ref - mine) < 1e-9, (ref, mine)
    print(f"brier matches sklearn: {mine:.6f}")

    nll = float(-np.log(np.clip(probs[np.arange(2000), labels], 1e-12, 1)).mean())
    assert abs(nll - log_loss(labels, probs, labels=list(range(10)))) < 1e-9
    print(f"nll matches sklearn: {nll:.6f}")

    # ECE of genuinely calibrated probs (labels sampled from them) is small;
    # of confident-and-usually-wrong is large
    cal_labels = np.array([rng.choice(10, p=row) for row in probs])
    e_cal, _ = ece(probs, cal_labels)
    bad = np.zeros_like(probs); bad[:, 0] = 1.0
    e_bad, _ = ece(bad, labels)
    assert 0 <= e_cal < 0.08 and e_bad > 0.5, (e_cal, e_bad)
    print(f"ece sane: calibrated {e_cal:.4f} vs overconfident {e_bad:.4f}")

    # perturbation determinism
    from evaluate import blur3x3, occlude, translate
    x = rng.rand(16, 784).astype(np.float32)
    assert np.array_equal(occlude(x), occlude(x))
    assert np.array_equal(translate(x), translate(x))
    assert np.allclose(blur3x3(np.full((2, 784), 0.5, dtype=np.float32)), 0.5)
    print("perturbations deterministic, blur preserves constants")
    print("METRIC TESTS PASSED")


if __name__ == "__main__":
    main()
