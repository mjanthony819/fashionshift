# Data

Download the four official Fashion-MNIST files into this directory:

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

From any official mirror listed at https://github.com/zalandoresearch/fashion-mnist, e.g.:

```bash
cd data
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
  curl -sSLO "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/$f"
done
```

Fashion-MNIST is (c) 2017 Zalando SE, released under the MIT License
(https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE).

The training split, validation split and file hashes used for the reported results are
recorded in `../artifacts/split_manifest.json`. The official test set (`t10k`) was never
used for training or model selection.
