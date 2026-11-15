"""Generate docs/FashionShift_Interview_Manual.docx from the measured artifacts."""
import json
import os

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(ROOT, "artifacts")
DOCS = os.path.join(ROOT, "docs")

with open(os.path.join(ART, "metrics.json")) as f:
    M = json.load(f)
with open(os.path.join(ART, "split_manifest.json")) as f:
    S = json.load(f)

ML = {"lr": "Logistic regression", "cnn": "CNN", "cnn_aug": "CNN + augmentation"}


def f4(x):
    return f"{x:.4f}"


doc = Document()
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(11)


def h(text, level=1):
    par = doc.add_heading(text, level=level)
    par.paragraph_format.keep_with_next = True


def p(text, bold=False, italic=False):
    par = doc.add_paragraph()
    r = par.add_run(text)
    r.bold = bold
    r.italic = italic
    return par


def bullets(items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def qa(i, q, a):
    par = doc.add_paragraph()
    r = par.add_run(f"Q{i}. {q}")
    r.bold = True
    par.paragraph_format.keep_with_next = True
    doc.add_paragraph(a)


# ---------------- cover ----------------
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("\n\nFashionShift\n")
r.font.size = Pt(40)
r.bold = True
r.font.color.rgb = RGBColor(0x25, 0x31, 0x19)
s = doc.add_paragraph()
s.alignment = WD_ALIGN_PARAGRAPH.CENTER
s.add_run("Classify and route garments through a simulated sorting gate, then audit the models - a Fashion-MNIST lab\n\n"
          "Project manual and interview preparation\n"
          "Build Your First AI App hackathon (Devpost)\n\n"
          "M J Anthony\nNovember 2026\n\n"
          "Built with disclosed AI assistance.").font.size = Pt(14)
doc.add_page_break()

# ---------------- 1 overview ----------------
h("1. What FashionShift is")
p("FashionShift is a workbench that trains three image classifiers on Fashion-MNIST and "
  "then audits them where models actually fail: distribution shift. Instead of stopping at "
  "a single accuracy number, it measures calibration, abstention behaviour, per-class "
  "weaknesses, and robustness to blur, occlusion and translation, on a test set that was "
  "never touched during training or model selection.")
p("The product is a single offline HTML file. All inference in the browser is a "
  "hand-written JavaScript re-implementation of the PyTorch models, proven to match the "
  "originals to within 1e-6 on 1,440 reference probabilities. Nothing is uploaded anywhere.")
h("2. Why it exists (motivation)")
p("Benchmark scores flatter models. A classifier that tops a clean leaderboard can collapse "
  "when real inputs arrive rotated, shifted, blurred or partly hidden - and it can stay "
  "confident while being wrong. FashionShift makes that behaviour visible and measurable in "
  "an interactive bench, an evidence dashboard, a live stress lab and an import tool.")

# ---------------- 3 data ----------------
h("3. Data and split discipline")
bullets([
    "Dataset: Fashion-MNIST, 70,000 grayscale 28x28 clothing images in 10 classes, "
    "(c) 2017 Zalando SE, MIT License.",
    f"Split: 50,000 train / 10,000 validation from the official training set "
    f"(seed {S['seed']}, indices hashed in artifacts/split_manifest.json).",
    "Test: the official 10,000-image t10k set, never used for training or model selection.",
    "Normalization: mean/std computed on the training split only "
    f"(mean 0.2860, std 0.3531).",
    "Source file SHA256 hashes are recorded in the split manifest for reproducibility.",
])
p("Classes: " + ", ".join(M["classes"]) + ".")

# ---------------- 4 models ----------------
h("4. The three models")
h("4.1 Logistic regression (baseline)", 2)
p(f"A single linear layer (784 -> 10) trained with cross-entropy. "
  f"{M['models']['lr']['params']:,} parameters. Fast, transparent, and the honesty anchor: "
  "if a neural net cannot beat this, something is wrong.")
h("4.2 CNN", 2)
p(f"conv(1->16, 3x3) + ReLU + maxpool2, conv(16->32, 3x3) + ReLU + maxpool2, "
  f"fc(1568->128) + ReLU, fc(128->10). {M['models']['cnn']['params']:,} parameters. "
  "Adam, lr 1e-3, batch 128, 10 epochs, fixed seeds.")
h("4.3 CNN + augmentation", 2)
p("Identical architecture, trained with random rotation (+/-15 deg), translation (+/-2 px) "
  "and Gaussian noise (sigma 0.05) applied per batch. Same epochs, same optimizer.")

# ---------------- 5 results ----------------
h("5. Measured results (untouched official test set, 10,000 images)")
tbl = doc.add_table(rows=1, cols=6)
tbl.style = "Light Grid Accent 1"
for i, c in enumerate(["Model", "Accuracy", "Macro F1", "NLL", "Brier", "ECE"]):
    tbl.rows[0].cells[i].text = c
for m in ["lr", "cnn", "cnn_aug"]:
    r = M["models"][m]
    cells = tbl.add_row().cells
    cells[0].text = ML[m]
    for i, k in enumerate(["accuracy", "macro_f1", "nll", "brier", "ece"]):
        cells[i + 1].text = f4(r[k])
p("")
bullets([
    f"Plain CNN is the clean-data winner: {f4(M['models']['cnn']['accuracy'])} accuracy.",
    f"Augmentation costs {100*(M['models']['cnn']['accuracy']-M['models']['cnn_aug']['accuracy']):.1f} points of clean accuracy - a deliberate trade.",
    "Shirt (F1 0.679), Pullover (0.791) and Coat (0.773) are the weakest classes for the "
    "augmented CNN; Shirt vs T-shirt/top vs Pullover confusion dominates the failure gallery.",
    "Logistic regression is the best-calibrated model (ECE 0.0066) - a known property of "
    "linear models - while being the least accurate.",
])

h("5.1 Per-class detail (augmented CNN)", 2)
tbl = doc.add_table(rows=1, cols=5)
tbl.style = "Light Grid Accent 1"
for i, c in enumerate(["Class", "Precision", "Recall", "F1", "Support"]):
    tbl.rows[0].cells[i].text = c
for c in M["models"]["cnn_aug"]["per_class"]:
    cells = tbl.add_row().cells
    cells[0].text = c["class"]
    cells[1].text = f"{c['precision']:.3f}"
    cells[2].text = f"{c['recall']:.3f}"
    cells[3].text = f"{c['f1']:.3f}"
    cells[4].text = str(c["support"])
p("")

# ---------------- 6 calibration ----------------
h("6. Calibration and abstention")
p("For every model the workbench reports negative log-likelihood, Brier score and "
  "expected calibration error (15 bins), plus a coverage-versus-risk curve: the error rate "
  "on kept predictions as the confidence threshold rises from 0 to 0.99.")
bullets([
    "Raising the threshold monotonically reduces risk on clean data for all three models.",
    "That curve is measured on clean data only. Under stress (Section 7), confidence no "
    "longer separates right from wrong, which is why the bench shows both.",
])

# ---------------- 7 stress ----------------
h("7. Stress audit (full test set, seeded perturbations)")
tbl = doc.add_table(rows=1, cols=5)
tbl.style = "Light Grid Accent 1"
for i, c in enumerate(["Perturbation", "Model", "Accuracy", "Conf (correct)", "Conf (wrong)"]):
    tbl.rows[0].cells[i].text = c
PN = {"blur": "Blur 3x3", "occlusion": "Occlusion 8x8", "translation": "Translation +/-4px"}
for sn in ["blur", "occlusion", "translation"]:
    for m in ["lr", "cnn", "cnn_aug"]:
        r = M["stress"][sn][m]
        cells = tbl.add_row().cells
        cells[0].text = PN[sn]
        cells[1].text = ML[m]
        cells[2].text = f4(r["accuracy"])
        cells[3].text = f"{r['mean_confidence_correct']:.3f}"
        cells[4].text = f"{r['mean_confidence_wrong']:.3f}"
p("")
bullets([
    "Translation is the killer: logistic regression collapses to 0.4101, the plain CNN to "
    "0.5775, while the augmented CNN holds 0.8014 - the clearest measured benefit of "
    "augmentation.",
    "Under blur and occlusion the augmented CNN is no better (slightly worse) than the plain "
    "CNN. Augmentation is not a generic robustness guarantee; it helps the shifts it trains "
    "on.",
    "Wrong-call confidence stays 0.60-0.75 under every stress. Confidence-based abstention "
    "alone does not detect shifted inputs.",
])

# ---------------- 8 app ----------------
h("8. The app (engineering)")
bullets([
    "One 2.3 MB HTML file: weights (f32, base64), 96-image test gallery, 36-image failure "
    "gallery, all metrics, all code. No server, no CDN, no analytics.",
    "Classify a garment (first tab): drop any PNG/JPG, see exactly what the models see "
    "(grayscale, center-crop, 28x28, optional invert), and get all three model calls with "
    "confidence, live in the browser. CSV batch import (784 columns + optional label) runs "
    "the same models over a whole file with strict validation before anything runs. A sorting "
    "gate (chosen model + confidence threshold) then routes every result: AUTO-SORT into the "
    "predicted class bin when confidence clears the gate, otherwise MANUAL REVIEW. Single "
    "images get a routing banner with per-model routing chips; CSV batches get a routing "
    "column, a per-bin routing summary with the manual-review share, and a downloadable "
    "routing CSV. The gate is a benchmark-data simulator: it demonstrates confidence-gated "
    "routing logic, not factory-ready QC - under distribution shift the models stay "
    "confidently wrong, so a real gate would misroute exactly when it looks sure.",
    "Live bench: random test images, three model chips each, abstain-threshold slider, "
    "batch accuracy readout.",
    "Evidence tab: metrics table, confusion matrix, per-class F1, reliability diagram, "
    "coverage-risk chart, real failure gallery.",
    "Stress lab: blur / occlusion / translation applied live in JS with immediate re-inference.",
    "Session history in localStorage only; one-click model card and metrics.json export.",
    "Parity: tests/parity_check.js compares JS inference against PyTorch on 48 test images x "
    "3 models - 1,440 probabilities, max abs error 1e-6.",
    "18 Node unit tests (inference primitives, perturbations, CSV validation, bundle "
    "self-containment) and metric tests cross-checked with scikit-learn.",
])

# ---------------- 9 repro ----------------
h("9. Reproducibility")
bullets([
    "All seeds fixed (20261116 for split/training, separate seeds for reference inputs).",
    "Split indices hashed in artifacts/split_manifest.json; source-file SHA256 recorded.",
    "Checkpoints under artifacts/checkpoints/; every number in this manual comes from "
    "artifacts/metrics.json, regenerated by evaluate.py.",
    "Full commands in README.md: train -> evaluate -> export -> build -> test.",
    "Training runs CPU-only in about two minutes per model.",
])

# ---------------- 10 limitations ----------------
h("10. Honest limitations")
bullets([
    "Ten classes of 28x28 grayscale catalog photos. Not validated on phone photos, drawings "
    "or other garments; not a shopping or recommendation tool.",
    "Augmentation result is benchmark-specific: it helps translation here, not blur or "
    "occlusion. No universal robustness claim is made.",
    "The stress set is synthetic and seeded; real-world shift is broader.",
    "The augmented model's clean-accuracy cost is reported, not hidden.",
    "Calibration numbers (ECE) are computed on in-distribution test data only.",
])

# ---------------- 11 AI disclosure ----------------
h("11. AI assistance disclosure")
p("FashionShift was designed, coded, trained, evaluated and documented with substantial "
  "assistance from an AI coding agent (Instinct), acting under the author's direction. "
  "The author chose the concept (a model-audit workbench rather than another accuracy demo), "
  "set the honesty requirements, and reviewed the results. All quantitative claims come from "
  "the training/evaluation scripts, not from the assistant.")

h("12. How it maps to the judging criteria")
bullets([
    "Creativity and originality (25%): an audit workbench for model shift, not another "
    "chatbot or accuracy demo; the live stress lab is the distinctive piece.",
    "Implementation and functionality (30%): real trained models, real browser inference "
    "with 1e-6 parity, import pipelines, session persistence, model-card export, 20+ "
    "automated checks.",
    "Usefulness and real-world impact (25%): the workflow - clean metrics, calibration, "
    "stress audit, model card - is exactly what a responsible small-model release needs.",
    "Presentation and documentation (20%): this manual, the README with reproduce "
    "commands, the narrated captioned demo, and the in-app model card.",
])

doc.add_page_break()

# ---------------- 13 Q&A ----------------
h("13. Interview questions and answers")
QAS = [
    ("What problem does FashionShift solve?",
     "It answers a question single-number benchmarks hide: does a model that scores well on "
     "clean data survive when inputs shift? It trains three Fashion-MNIST classifiers and "
     "audits accuracy, calibration, abstention and robustness to blur, occlusion and "
     "translation on a test set the models never saw."),
    ("Why Fashion-MNIST and not MNIST?",
     "MNIST is essentially solved - near-100% accuracy leaves nothing to audit. Fashion-MNIST "
     "has genuinely hard classes (shirt vs pullover vs tee) where models still make real "
     "mistakes worth studying."),
    ("Why three models?",
     "A linear baseline anchors honesty, a plain CNN shows what a standard network buys, and "
     "the same CNN trained with augmentation isolates exactly what augmentation changes. Same "
     "architecture for the two CNNs means the comparison is controlled."),
    ("How did you avoid test-set leakage?",
     "The official t10k test set was never loaded by the training script. A seeded split "
     "carved 10k validation images out of the training set for model selection; split indices "
     "and dataset file hashes are recorded in a manifest."),
    ("Walk me through the CNN.",
     "Two 3x3 conv layers (1->16->32 channels), each followed by ReLU and 2x2 max-pooling, "
     "then a 1568->128 fully connected layer and a 128->10 head. About 207k parameters, "
     "trained with Adam at 1e-3 for 10 epochs on CPU."),
    ("What augmentation did you use and why?",
     "Random rotation up to 15 degrees, translation up to 2 pixels and Gaussian noise, applied "
     "per batch with a seeded generator. These mimic the shifts real cameras introduce."),
    ("What did augmentation actually buy?",
     "Under a seeded +/-4px translation stress it raised accuracy from 0.578 to 0.801. Under "
     "blur and occlusion it did not help - it was slightly worse. And it cost 2.7 points of "
     "clean accuracy. All three facts are reported side by side."),
    ("Which model is 'best'?",
     "It depends on the deployment shift, which is the whole point of the workbench. On clean "
     "data the plain CNN wins; under translation the augmented CNN wins by 22 points; for "
     "calibrated probabilities the linear model has the lowest ECE."),
    ("What is ECE and why report it?",
     "Expected calibration error: the gap between a model's confidence and its actual accuracy "
     "across confidence bins. A model can be accurate and still badly calibrated - dangerous "
     "if downstream logic trusts its probabilities."),
    ("What is the coverage-risk curve?",
     "Sort predictions by confidence, keep only those above a threshold, and plot the error "
     "rate of what remains against how much was kept. It quantifies how useful abstention is "
     "for each model - on clean data."),
    ("Your key finding on confidence under stress?",
     "Wrong predictions under stress still average 0.60-0.75 confidence for all three models. "
     "So 'just abstain when unsure' fails precisely when inputs are shifted - you need "
     "shift detection, not just a confidence threshold."),
    ("Why does logistic regression collapse under translation?",
     "A linear model keys on absolute pixel positions. Move every pixel four steps and the "
     "learned weights point at empty space. Convolutions have local translation tolerance, "
     "and augmentation teaches it explicitly - hence 0.41 vs 0.58 vs 0.80."),
    ("Why is augmentation slightly worse under blur and occlusion?",
     "Its training distribution spent capacity on rotations and shifts, not blur or occluders, "
     "and noise regularization slightly blurs decision boundaries. You get robustness to the "
     "shifts you train for, not for free."),
    ("What are the weakest classes and why?",
     "Shirt, pullover and T-shirt/top. At 28x28 grayscale they share silhouette and texture, "
     "so even the best model only reaches F1 0.679 on Shirt. The failure gallery in the app "
     "shows real examples."),
    ("How does the browser run inference without a server?",
     "Weights are exported as base64 Float32 and embedded in the HTML. A hand-written JS "
     "implementation reproduces conv2d, max-pool, ReLU, the linear layers and softmax."),
    ("How do you know the JS matches PyTorch?",
     "A parity test: 48 seeded test images are run through both implementations, all three "
     "models, and 1,440 output probabilities are compared. Max absolute error is about 1e-6, "
     "inside the 1e-4 tolerance."),
    ("What other tests exist?",
     "18 Node unit tests cover the inference primitives, the perturbations, CSV validation "
     "rules and bundle self-containment. Python metric tests cross-check Brier and NLL "
     "against scikit-learn and verify perturbation determinism."),
    ("How is the app kept fully offline?",
     "A unit test asserts the built index.html contains no external script or stylesheet "
     "references. Fonts are system fonts; charts are hand-drawn canvas."),
    ("How does image import work?",
     "The image is center-cropped square, downscaled to 28x28, converted to luma "
     "(0.299/0.587/0.114 weights), optionally inverted, previewed at exactly what the model "
     "sees, then scored by all three models in-browser."),
    ("How does CSV import validate data?",
     "Every row must have exactly 784 pixel values (plus an optional 0-9 label), all numeric "
     "and within 0-255. Bad rows are rejected with specific messages before any inference "
     "runs; files over 20 MB are refused."),
    ("What would you do with one more week?",
     "Add a real shift-detector (e.g. an autoencoder reconstruction-error score or Mahalanobis "
     "distance on penultimate features) and measure whether it beats confidence under stress; "
     "add rotation stress; quantify augmentation cost across dataset sizes."),
    ("How would you scale this to production?",
     "Batch scoring service with the same evaluation harness wired into CI: every model "
     "candidate must publish its clean metrics, stress metrics and calibration before "
     "deploy. The workbench UI already reads exactly that artifact format."),
    ("What is the model card?",
     "A one-click export from the app: markdown listing data, split discipline, all measured "
     "metrics, stress results and limitations - the same numbers as this manual, generated "
     "from the same metrics.json."),
    ("Where does session data go?",
     "Nowhere. History lives in the browser's localStorage; images and CSVs never leave the "
     "page. There is no backend at all."),
    ("Why a single HTML file instead of a framework app?",
     "Portability and verifiability: one file anyone can save, hash, diff and run offline. It "
     "also proves the inference really is self-contained rather than calling a hidden API."),
    ("How long did training take and on what hardware?",
     "About two minutes per model on a 2-core CPU sandbox - 10 epochs over 50k images at "
     "batch 128. No GPU was needed, which is deliberate: the whole project reproduces on "
     "any laptop."),
    ("What is the biggest honest weakness of the project?",
     "The stress set is synthetic and small in variety. Real deployment shift includes "
     "lighting, backgrounds, pose and fabric variation that 28x28 grayscale cannot express. "
     "The bench demonstrates method, not production readiness."),
    ("How did AI assistance work on this project?",
     "An AI coding agent wrote the code, trained the models and assembled the app under the "
     "author's direction and honesty constraints. Every number it reports is regenerated by "
     "the committed scripts; the disclosure is in the README and on the submission."),
    ("Why should a judge pick this over a chatbot demo?",
     "Because it does something harder than call an API: it trains real models, measures them "
     "honestly, ships working in-browser inference with proven numerical parity, and surfaces "
     "a result most demos hide - where and how confidently models fail."),
    ("Explain the confusion matrix findings.",
     "The augmented CNN confuses Shirt with T-shirt/top (134) and Pullover (110) most. "
     "Trouser, Bag, Sneaker and Ankle boot exceed F1 0.95 - distinct silhouettes are easy, "
     "upper-body garments are not."),
    ("What metrics matter beyond accuracy and why?",
     "Macro F1 guards against per-class collapse, NLL and Brier score the probabilities not "
     "just the argmax, and ECE summarizes calibration. A model can win accuracy and lose "
     "every probability metric - ours is reported on all of them."),
    ("Is FashionShift useful beyond a hackathon?",
     "Yes, as a template: the same train-evaluate-stress-audit workflow applies to any small "
     "vision model before release, and the import path lets a curator probe the exact "
     "failure modes on their own images."),
]
for i, (q, a) in enumerate(QAS, 1):
    qa(i, q, a)

p("")
p("End of manual. All figures regenerate from artifacts/metrics.json via make_manual.py.",
  italic=True)

os.makedirs(DOCS, exist_ok=True)
out = os.path.join(DOCS, "FashionShift_Interview_Manual.docx")
doc.save(out)
print("saved", out, f"({len(QAS)} Q&As)")
