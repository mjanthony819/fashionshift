/* FashionShift pure-JS inference.
 * Mirrors src/models.py (SmallCNN) and the torch Linear baseline exactly.
 * Runs in the browser (window.FSInference) and in Node (module.exports).
 * Parity with PyTorch is enforced by tests/parity_check.js.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.FSInference = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  function b64ToF32(b64) {
    let bin;
    if (typeof atob !== "undefined") {
      bin = atob(b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      return new Float32Array(bytes.buffer);
    }
    const buf = Buffer.from(b64, "base64");
    return new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
  }

  function decodeModel(entry) {
    const layers = {};
    for (const [k, v] of Object.entries(entry.layers)) {
      layers[k] = { shape: v.shape, data: b64ToF32(v.data) };
    }
    return { arch: entry.arch, mean: entry.mean, std: entry.std, layers };
  }

  function softmax(logits) {
    let m = -Infinity;
    for (const v of logits) if (v > m) m = v;
    let s = 0;
    const e = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) { e[i] = Math.exp(logits[i] - m); s += e[i]; }
    for (let i = 0; i < e.length; i++) e[i] /= s;
    return e;
  }

  function linear(x, W, b, inF, outF) {
    const out = new Float32Array(outF);
    for (let o = 0; o < outF; o++) {
      let s = b[o];
      const wrow = o * inF;
      for (let i = 0; i < inF; i++) s += W[wrow + i] * x[i];
      out[o] = s;
    }
    return out;
  }

  // 3x3 conv, padding=1 (zeros), single input image set: in (C,H,W) -> out (F,H,W)
  function conv3x3(input, inC, size, W, B, outF) {
    const out = new Float32Array(outF * size * size);
    for (let f = 0; f < outF; f++) {
      for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
          let s = B[f];
          for (let ic = 0; ic < inC; ic++) {
            const wBase = ((f * inC + ic) * 9);
            const iBase = ic * size * size;
            for (let kr = -1; kr <= 1; kr++) {
              const rr = r + kr;
              if (rr < 0 || rr >= size) continue;
              for (let kc = -1; kc <= 1; kc++) {
                const cc = c + kc;
                if (cc < 0 || cc >= size) continue;
                s += W[wBase + (kr + 1) * 3 + (kc + 1)] * input[iBase + rr * size + cc];
              }
            }
          }
          out[f * size * size + r * size + c] = s;
        }
      }
    }
    return out;
  }

  function relu(x) { for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0; return x; }

  function maxpool2(input, C, size) {
    const half = size / 2;
    const out = new Float32Array(C * half * half);
    for (let c = 0; c < C; c++) {
      const base = c * size * size;
      const oBase = c * half * half;
      for (let r = 0; r < half; r++) {
        for (let col = 0; col < half; col++) {
          const r0 = r * 2, c0 = col * 2;
          let m = input[base + r0 * size + c0];
          m = Math.max(m, input[base + r0 * size + c0 + 1]);
          m = Math.max(m, input[base + (r0 + 1) * size + c0]);
          m = Math.max(m, input[base + (r0 + 1) * size + c0 + 1]);
          out[oBase + r * half + col] = m;
        }
      }
    }
    return out;
  }

  function standardize(pixels, mean, std) {
    const out = new Float32Array(pixels.length);
    for (let i = 0; i < pixels.length; i++) out[i] = (pixels[i] - mean) / std;
    return out;
  }

  // pixels: 784 values in [0,1]. Returns {probs, pred, conf}
  function predict(model, pixels) {
    const x = standardize(pixels, model.mean, model.std);
    let logits;
    if (model.arch === "linear") {
      logits = linear(x, model.layers.W.data, model.layers.b.data, 784, 10);
    } else {
      let a = conv3x3(x, 1, 28, model.layers.conv1_w.data, model.layers.conv1_b.data, 16);
      a = maxpool2(relu(a), 16, 28);
      a = conv3x3(a, 16, 14, model.layers.conv2_w.data, model.layers.conv2_b.data, 32);
      a = maxpool2(relu(a), 32, 14);
      const h = relu(linear(a, model.layers.fc1_w.data, model.layers.fc1_b.data, 32 * 7 * 7, 128));
      logits = linear(h, model.layers.fc2_w.data, model.layers.fc2_b.data, 128, 10);
    }
    const probs = softmax(logits);
    let pred = 0;
    for (let i = 1; i < 10; i++) if (probs[i] > probs[pred]) pred = i;
    return { probs: Array.from(probs), pred, conf: probs[pred], logits: Array.from(logits) };
  }

  // ---------- deterministic perturbations (mirror evaluate.py) ----------

  function blur3x3(pixels) {
    const k = [1, 2, 1, 2, 4, 2, 1, 2, 1];
    const out = new Float32Array(784);
    for (let r = 0; r < 28; r++) {
      for (let c = 0; c < 28; c++) {
        let s = 0;
        for (let kr = -1; kr <= 1; kr++) {
          const rr = Math.min(27, Math.max(0, r + kr)); // edge padding
          for (let kc = -1; kc <= 1; kc++) {
            const cc = Math.min(27, Math.max(0, c + kc));
            s += k[(kr + 1) * 3 + (kc + 1)] * pixels[rr * 28 + cc];
          }
        }
        out[r * 28 + c] = s / 16;
      }
    }
    return out;
  }

  // tiny LCG-free deterministic PRNG matching nothing external; Python side uses
  // RandomState(base+i) with the same derived (r,c)/(dy,dx) choices precomputed
  // and shipped in the app bundle so live lab matches the audit exactly.
  function occlude(pixels, r, c, size) {
    const out = Float32Array.from(pixels);
    for (let rr = r; rr < r + size; rr++)
      for (let cc = c; cc < c + size; cc++) out[rr * 28 + cc] = 0;
    return out;
  }

  function translate(pixels, dy, dx) {
    const out = new Float32Array(784);
    for (let r = 0; r < 28; r++) {
      for (let c = 0; c < 28; c++) {
        const sr = r - dy, sc = c - dx;
        if (sr >= 0 && sr < 28 && sc >= 0 && sc < 28) out[r * 28 + c] = pixels[sr * 28 + sc];
      }
    }
    return out;
  }

  return { b64ToF32, decodeModel, predict, softmax, blur3x3, occlude, translate, linear, conv3x3, maxpool2 };
});
