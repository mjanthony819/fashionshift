/* Parity: JS inference must match PyTorch reference probabilities within 1e-4. */
"use strict";
const fs = require("fs");
const path = require("path");
const FS = require(path.join(__dirname, "..", "js", "inference.js"));

const ART = path.join(__dirname, "..", "artifacts");
const ref = JSON.parse(fs.readFileSync(path.join(ART, "parity_reference.json"), "utf8"));
const weights = JSON.parse(fs.readFileSync(path.join(ART, "weights.json"), "utf8"));

const imgs = Buffer.from(ref.images_b64, "base64");
const N = ref.idx.length;
let failures = 0, checked = 0, maxErr = 0;

for (const name of ["lr", "cnn", "cnn_aug"]) {
  const model = FS.decodeModel(weights.models[name]);
  for (let i = 0; i < N; i++) {
    const px = new Float32Array(784);
    for (let j = 0; j < 784; j++) px[j] = imgs[i * 784 + j] / 255;
    const r = FS.predict(model, px);
    const refP = ref.models[name][i];
    for (let c = 0; c < 10; c++) {
      const err = Math.abs(r.probs[c] - refP[c]);
      if (err > maxErr) maxErr = err;
      checked++;
      if (err > 1e-4) {
        failures++;
        if (failures < 5) console.log(`MISMATCH ${name} img ${i} class ${c}: js=${r.probs[c]} torch=${refP[c]}`);
      }
    }
  }
  console.log(`${name}: ${N} images compared`);
}
console.log(`checked ${checked} probabilities, max abs error ${maxErr.toExponential(2)}, failures ${failures}`);
if (failures > 0) { console.error("PARITY FAILED"); process.exit(1); }
console.log("PARITY OK");
