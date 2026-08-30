# k-means-compression
 
A from-scratch implementation of K-means clustering, used to compress images by reducing them to a small palette of colors. The core algorithm, every part of it (initialization, cluster assignment, centroid updates, convergence) is implemented and tested by hand.
 
## Example
 
| Original | Compressed (k=4) |
|---|---|
| ![original-cat](examples/cat.jpg) | ![compressed4-cat](examples/cat-4.jpg) |  

| Compressed (k=8) | Compressed (k=128) | 
|---|---|
| ![compressed8-cat](examples/cat-8.jpg) | ![compressed128-cat](examples/cat-128.jpg) |

| Original | Compressed (k=4) |
|---|---|
| ![original-landscape](examples/small-landscape.jpg) | ![compressed4-landscape](examples/small-landscape-4.jpg) |

| Compressed (k=8) | Compressed (k=16) |
|---|---|
| ![compressed8-landscape](examples/small-landscape-8.jpg) | ![compressed16-landscape](examples/small-landscape-16.jpg) |

| Compressed (k=32) | Compressed (k=64) |
|---|---|
| ![compressed32-landscape](examples/small-landscape-32.jpg) | ![compressed64-landscape](examples/small-landscape-64.jpg) |


| Compressed (k=128) | Compressed (k=256) |
|---|---|
| ![compressed128-landscape](examples/small-landscape-128.jpg) | ![compressed256-landscape](examples/small-landscape-256.jpg) |

### Example output for `k=16`:
```
python3 cli.py -k 16 --load examples/cat.jpg --save examples/c
at-16.jpg
100%|████████████████████████| 100/100 [07:19<00:00,  4.39s/it]
Saved the compressed image to: examples/cat-16.jpg
Colors: 88,681 -> 16
Size: 65,744,640 bytes -> 10,957,488 bytes (83.3% smaller)
```
 
That 83.3% is theoretical: storing 8 RGB colors as a palette, plus a 3-bit index per pixel, instead of full 24-bit RGB per pixel. The real PNG files on disk compress even further, since PNG applies its own lossless compression on top.
 
## How it works
 
Every pixel is a point in 3D space (R, G, B). K-means clusters the image's pixels into K groups by color similarity, then replaces every pixel with its cluster's average color. Fewer distinct colors means the image can be stored far more compactly: a small color palette plus a per pixel index, instead of a full RGB triple for every pixel.
 
## Installation
 
```bash
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```
 
## Usage
 
```bash
python3 cli.py -k 16 --load photo.jpg --save compressed.png
```
 
| Flag | Description | Default |
|---|---|---|
| `-k` | Number of colors (1-256) | required |
| `--load` | Input image path | required |
| `--save` | Output image path | required |
| `--seed` | Random seed, for reproducible results | random |
| `--max_iter` | Max iterations before giving up | 100 |
| `--eps` | Convergence tolerance | 1e-4 |
| `--backend` | `scratch` (from-scratch, this project's implementation) or `sklearn` (fast path for large images / high K) | `scratch` |
 
> The `scratch` backend is the point of this project, but it's a pure Python/numpy implementation and can take a while on large images at high K! 
 
## Architecture
 
- `kmeans.py` — the algorithm: initialization, cluster assignment (vectorized,
  batched to bound memory usage on large images), centroid updates, and the main fit loop
- `kmeans_sklearn.py` — an alternative backend 
- `image_io.py` — loading images
- `compressor.py` — the pipeline that ties loading, clustering, and saving together
- `stats.py` — computes the compression statistics (unique colors, theoretical size,
  savings) shown after every run
- `cli.py` — the command-line interface and input validation

