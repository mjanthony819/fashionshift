# FashionShift

**A vision-model audit workbench.** Three models are trained on Fashion-MNIST and then
compared honestly on the untouched official test set: clean accuracy, calibration,
abstention behaviour, and what happens when the images shift (blur, occlusion,
translation). The whole workbench runs offline in a single HTML file — inference is a
hand-written JavaScript re-implementation that is parity-tested against PyTorch.

Built for the *Build Your First AI App* hackathon (Devpost). Built with AI assistance
(disclosed below).

## Live app

Open `index.html` in any modern browser — no server, no install, no network needed.
(Or visit the GitHub Pages deployment linked from the repository page.)

The first tab is **Classify a garment**: drop in any PNG/JPG (or a CSV batch of
784-value rows) and all three trained models classify it live in your browser,
showing exactly what the models see. The bench, evidence and stress tabs audit
those same models afterwards.

**Update, November 16, 2026:** the app was reworked so the trained models drive
the main workflow end to end (classify-first) and the interface moved from a
stock dark theme to the project's charcoal/lime design. Models, data, metrics
and evaluation are unchanged. The narrated demo under `demo/` was recorded for
the original submission and shows its earlier interface.

## What it shows

| Model | Accuracy | Macro F1 | NLL | Brier | ECE |
|---|---|---|---|---|---|
| Logistic regression | 0.8386 | 0.8374 | 0.4529 | 0.2290 | 0.0066 |
| CNN | 0.9114 | 0.9115 | 0.2490 | 0.1275 | 0.0196 |
| CNN + augmentation | 0.8849 | 0.8839 | 0.3219 | 0.1663 | 0.0259 |

Stress audit (full 10,000-image test set, seeded perturbations):

| Perturbation | LR | CNN | CNN + aug |
|---|---|---|---|
| Blur 3x3 | 0.8222 | 0.8849 | 0.8398 |
| Occlusion 8x8 | 0.7789 | 0.8236 | 0.7858 |
| Translation ±4px | 0.4101 | 0.5775 | **0.8014** |

Honest findings:

- Augmentation buys large translation robustness (+22 points over the plain CNN) but
  *costs* 2.7 points of clean accuracy and does **not** help under blur or occlusion here.
- Confidence stays high when models are wrong under stress (wrong-call confidence averages
  0.60–0.75), so abstaining on confidence alone is not a shift detector.
- Shirt / T-shirt-top / Pullover are the weak classes in every model.

## Scope and limits

Fashion-MNIST is 28x28 grayscale catalog clothing photos from Zalando Research (MIT
license). This is a model-audit bench, **not** a real-world photo classifier and not a
shopping or recommendation tool. Phone photos, drawings and other garment styles are out
of distribution — expect confident nonsense, and the workbench exists to show exactly that.

## Repository layout

```
index.html            the complete offline workbench (built artifact)
app/template.html     app source template
js/inference.js       shared JS inference (browser + Node)
train.py              trains lr / cnn / cnn_aug (resumable, seeded)
evaluate.py           metrics + stress audit on the untouched test set
export_weights.py     checkpoints -> artifacts/weights.json (f32 base64)
build_app.py          template + weights + metrics + gallery -> index.html
src/                  data loading, model definition
tests/                parity (JS vs PyTorch), unit, and metric tests
artifacts/            checkpoints, metrics.json, predictions.csv, manifests
data/                 raw dataset files (download separately, see data/README.md)
docs/                 interview manual (DOCX/PDF)
demo/                 narrated captioned demo video + captions
```

## Reproduce

```bash
pip install -r requirements.txt          # torch (CPU), numpy, scikit-learn, matplotlib, pillow
# download the 4 official .gz files into data/  (see data/README.md)
python3 train.py --model all --epochs 10 # LR finishes in ~8 epochs
python3 evaluate.py                      # writes artifacts/metrics.json + predictions.csv
python3 export_weights.py
python3 build_app.py                     # writes index.html
python3 tests/test_metrics.py
PYTHONPATH=. python3 tests/test_parity.py
node tests/parity_check.js               # JS vs PyTorch, 1e-4 tolerance
node tests/test_unit.js
```

## AI assistance disclosure

This project was designed, coded, trained, evaluated and documented with substantial
assistance from an AI coding agent (Instinct) acting for the author. The author directed
the concept, scope and honesty requirements; the agent wrote the code, ran the training
and built the app. All results are measured by the scripts above, not asserted.

## Data license

Fashion-MNIST: MIT License, Copyright (c) 2017 Zalando SE.
See https://github.com/zalandoresearch/fashion-mnist and `data/README.md`.
