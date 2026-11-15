/* Unit tests: inference primitives, perturbations, and the CSV validation rules
 * (duplicated here against the same spec the app enforces). */
"use strict";
const fs = require("fs");
const path = require("path");
const assert = require("assert");
const FS = require(path.join(__dirname, "..", "js", "inference.js"));

let n = 0;
function ok(name, cond) { n++; if (!cond) { console.error("FAIL:", name); process.exitCode = 1; } else console.log("ok", n, "-", name); }

// softmax sums to 1
const p = FS.softmax(new Float32Array([1, -2, 3, 0, 0.5, -1, 2, 0.1, -0.3, 0.7]));
ok("softmax sums to 1", Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-5);

// linear layer against hand computation
const W = new Float32Array([1, 0, 0, 1]); const b = new Float32Array([0.5, -0.5]);
const out = FS.linear(new Float32Array([2, 3]), W, b, 2, 2);
ok("linear hand-check", Math.abs(out[0] - 2.5) < 1e-6 && Math.abs(out[1] - 2.5) < 1e-6);

// maxpool halves dimensions and takes max
const mp = FS.maxpool2(new Float32Array([1, 2, 3, 4]), 1, 2);
ok("maxpool 2x2", mp.length === 1 && mp[0] === 4);

// blur preserves constant images
const ones = new Float32Array(784).fill(0.5);
const bl = FS.blur3x3(ones);
ok("blur keeps constant image", Math.abs(bl[400] - 0.5) < 1e-6 && Math.abs(bl[0] - 0.5) < 1e-6);

// occlusion zeroes the block and nothing else
const occ = FS.occlude(ones, 10, 12, 8);
let zeroInside = true, okOutside = true;
for (let r = 0; r < 28; r++) for (let c = 0; c < 28; c++) {
  const inside = r >= 10 && r < 18 && c >= 12 && c < 20;
  if (inside && occ[r * 28 + c] !== 0) zeroInside = false;
  if (!inside && Math.abs(occ[r * 28 + c] - 0.5) > 1e-6) okOutside = false;
}
ok("occlusion zeroes block only", zeroInside && okOutside);

// translation shifts content, zero-fills
const timg = new Float32Array(784); timg[5 * 28 + 7] = 1;
const tr = FS.translate(timg, 2, 3);
ok("translation moves pixel by (dy,dx)", tr[7 * 28 + 10] === 1 && tr[5 * 28 + 7] === 0);

// base64 decoder round-trip
const f32 = new Float32Array([1.5, -2.25, 0, 1e-6]);
const b64 = Buffer.from(f32.buffer).toString("base64");
const back = FS.b64ToF32(b64);
ok("b64 decode round-trip", f32.every((v, i) => Math.abs(back[i] - v) < 1e-12));

// weights decode shapes
const weights = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "artifacts", "weights.json"), "utf8"));
const cnn = FS.decodeModel(weights.models.cnn);
ok("cnn layer shapes", cnn.layers.conv1_w.shape.join("x") === "16x1x3x3" &&
  cnn.layers.fc2_w.shape.join("x") === "10x128");
const lr = FS.decodeModel(weights.models.lr);
ok("lr shape", lr.layers.W.shape.join("x") === "10x784");

// prediction contract on a real image
const ref = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "artifacts", "parity_reference.json"), "utf8"));
const imgs = Buffer.from(ref.images_b64, "base64");
const px = new Float32Array(784);
for (let j = 0; j < 784; j++) px[j] = imgs[j] / 255;
const r0 = FS.predict(cnn, px);
ok("predict returns 10 probs and argmax", r0.probs.length === 10 && r0.pred >= 0 && r0.pred < 10 && r0.conf === r0.probs[r0.pred]);

// CSV validation spec (mirrors app rules)
function validateCsvLine(line, rowNum) {
  const parts = line.split(",").map(s => s.trim());
  if (parts.length !== 784 && parts.length !== 785) return `row ${rowNum}: ${parts.length} columns, expected 784 pixels (+ optional label)`;
  const pxv = parts.slice(0, 784).map(Number);
  if (pxv.some(v => !isFinite(v) || v < 0 || v > 255)) return `row ${rowNum}: pixel values must be numbers in 0–255`;
  if (parts.length === 785) { const l = Number(parts[784]); if (!Number.isInteger(l) || l < 0 || l > 9) return `row ${rowNum}: label must be an integer 0–9`; }
  return null;
}
const goodRow = new Array(784).fill("128").join(",");
ok("csv good row passes", validateCsvLine(goodRow, 1) === null);
ok("csv good row+label passes", validateCsvLine(goodRow + ",4", 1) === null);
ok("csv short row rejected", validateCsvLine("1,2,3", 1) !== null);
ok("csv out-of-range pixel rejected", validateCsvLine(new Array(783).fill("0").concat(["999"]).join(","), 1) !== null);
ok("csv bad label rejected", validateCsvLine(goodRow + ",99", 1) !== null);
ok("csv text pixel rejected", validateCsvLine(new Array(783).fill("0").concat(["abc"]).join(","), 1) !== null);

// app bundle self-containment: no external URLs in index.html
const html = fs.readFileSync(path.join(__dirname, "..", "index.html"), "utf8");
ok("app has no external scripts/styles", !/<script[^>]+src=|<link[^>]+stylesheet/i.test(html));
ok("app embeds data", html.includes('"weights"') && html.includes('"gallery"'));

console.log(process.exitCode ? "UNIT TESTS FAILED" : `ALL ${n} UNIT TESTS PASSED`);
