# torchPersLay

`torchPersLay` is a native [PyTorch](https://pytorch.org/) implementation of **PersLay**, a neural-network layer for persistence diagrams in topological data analysis. The package is published on PyPI as `torchPersLay`; the Python import package is `perslay`.

This repository focuses on PersLay representations with learnable point weights, Gaussian bandwidths, and exact persistence-image integration. Version 0.4 adds a globally learnable nonuniform persistence-image grid on top of the fused operators introduced in version 0.3. The library separates four Gaussian feature maps:

- `GridSampledGaussianPerslayPhi`: Gaussian surface values at pixel centres;
- `IntegratedGaussianPerslayPhi`: exact Gaussian mass integrated over a fixed pixel grid;
- `LearnableGridIntegratedGaussianPerslayPhi`: exact Gaussian mass integrated over a globally learned nonuniform pixel grid; and
- `GaussianPerslayPhi`: the pre-0.2 boundary-grid implementation retained for checkpoint compatibility.

The distinction matters: centre samples represent local surface heights, whereas integrated pixels represent regional Gaussian mass.

When either explicit Gaussian map is used with `perm_op=torch.sum`, `Perslay`
automatically exploits Gaussian separability. It evaluates only one-dimensional
birth and persistence factors and contracts them with a batched matrix
multiplication instead of materializing a `[batch, points, ny, nx]` tensor.
This is mathematically equivalent to the pointwise construction and supports
exact point chunking through `chunk_size` for very large diagrams.

## Installation

Install the released package from PyPI:

```bash
pip install torchperslay
```

Install the local repository in editable mode:

```bash
git clone https://github.com/jhnrckmnznrs/torchPersLay.git
cd torchPersLay
pip install -e .
```

To run the experiment scripts in this repository, also install the experiment extra or the repository requirements:

```bash
pip install -e ".[experiments]"
# or
pip install -r requirements.txt
```

## Quick start

The example below builds a PersLay regressor for a small batch of padded persistence diagrams. Diagrams are expected as `(birth, death)` pairs with shape `[batch, num_points, 2]`.

```python
import torch

from perslay import (
    ConstantPerslayWeight,
    GridSampledGaussianPerslayPhi,
    Perslay,
)
from perslay.models import FlattenRho

image_size = [10, 10]
image_bnds = [[0.0, 1.0], [0.0, 1.0]]  # birth bounds, persistence bounds

weight = ConstantPerslayWeight()
phi = GridSampledGaussianPerslayPhi(
    image_size=image_size,
    image_bnds=image_bnds,
    sigma_x=0.1,
    sigma_y=0.1,
    normalize=False,
    learnable_sigma=True,
    flatten_order="legacy-x-major",
    chunk_size=4096,  # optional; bounds peak memory for large diagrams
)

perslay = Perslay(
    weight=weight,
    phi=phi,
    perm_op=torch.sum,
    rho=FlattenRho(),
)

# Two padded diagrams with three points each.
diagrams = torch.tensor(
    [
        [[0.0, 0.4], [0.2, 0.8], [0.0, 0.0]],
        [[0.1, 0.5], [0.3, 0.9], [0.4, 0.95]],
    ],
    dtype=torch.float32,
)
mask = torch.tensor(
    [
        [True, True, False],
        [True, True, True],
    ]
)

features = perslay(diagrams, mask=mask)
print(features.shape)  # torch.Size([2, 100])
```

With constant weights, sum aggregation, centre coordinates, omitted Gaussian
normalization, and `legacy-x-major` flattening, this matches the mathematical
construction of the Rust grid-sampled persistence surface. Setting
`normalize=True` adds `1 / (2*pi*sigma_x*sigma_y)` but still returns point
samples rather than pixel integrals.

For canonical integrated persistence-image pixels, replace the feature map by:

```python
from perslay import IntegratedGaussianPerslayPhi

phi = IntegratedGaussianPerslayPhi(
    image_size=[10, 10],
    image_bnds=[[0.0, 1.0], [0.0, 1.0]],
    sigma_x=0.1,
    sigma_y=0.1,
    normalize=True,
    learnable_sigma=True,
    flatten_order="row-major",
    chunk_size=4096,
)
```

`chunk_size=None` processes all points in one fused contraction. A positive
integer processes points in additive chunks. Chunking changes only the order
of floating-point additions, not the mathematical representation. The generic
materialized PersLay path remains available for non-sum permutation operators.


## Learnable nonuniform persistence-image grids

`LearnableGridIntegratedGaussianPerslayPhi` keeps the outer birth/persistence
bounds and the number of pixels fixed but learns the internal pixel boundaries.
The grid is **global**: one set of learned boundaries is shared by every diagram
processed by the layer.

```python
from perslay import LearnableGridIntegratedGaussianPerslayPhi

phi = LearnableGridIntegratedGaussianPerslayPhi(
    image_size=[20, 20],
    image_bnds=[[0.0, 1.0], [0.0, 1.0]],
    sigma_x=0.05,
    sigma_y=0.05,
    normalize=True,
    learnable_sigma=True,
    min_cell_fraction=0.05,
    grid_regularization_strength=1e-3,
)

x_edges, y_edges = phi.grid_edges()       # differentiable tensors
x_widths, y_widths = phi.grid_width_fractions()
penalty = phi.regularization_loss()       # add to the task loss if desired
```

For an axis with `n` cells and trainable logits `theta`, normalized widths are

```text
q = r / n + (1 - r) * softmax(theta)
```

where `r=min_cell_fraction`. Zero logits give exactly the ordinary uniform
grid. The minimum-width floor prevents cells from collapsing, while cumulative
sums of `q` guarantee that boundaries remain ordered and the outer bounds never
move.

The construction is differentiable because a pixel mass has the form

```text
Phi((x_right - birth) / sigma) - Phi((x_left - birth) / sigma).
```

Differentiating with respect to a shared boundary evaluates the Gaussian
density at that boundary: the same infinitesimal mass is removed from one cell
and added to its neighbour. Thus standard backpropagation can move grid
boundaries without hard bin assignment. `grid_regularization()` returns a
dimensionless squared deviation from uniform cell widths; the configured
`regularization_loss()` is zero at uniform initialization.

## Regression models

For most regression experiments, use the provided model wrappers:

```python
from perslay.models import PerslayRegressor, MultiPerslayRegressor

model = PerslayRegressor(
    image_size=[20, 20],
    image_bnds=[[0.0, 1.0], [0.0, 1.0]],
    sigma_x=0.05,
    sigma_y=0.05,
    hidden_dim=128,
    weight_type="constant",
    feature_map_type="grid_sampled",
    normalize=False,
    learnable_sigma=True,
    flatten_order="legacy-x-major",
    chunk_size=4096,
)
```

Available `weight_type` values are:

- `constant`
- `persistence`
- `linear_ramp` (also supply `weight_threshold`)
- `learnable_power`
- `normalized_learnable_power`
- `mlp`

Available `feature_map_type` values are:

- `grid_sampled`: pixel-centre surface values;
- `integrated`: exact pixel integrals on a fixed grid;
- `integrated_learnable_grid`: exact pixel integrals on a global learned nonuniform grid; and
- `legacy_gaussian`: the pre-0.2 boundary-grid behavior.

For `grid_sampled` or `integrated`, set `learnable_sigma=False` to hold the
specified bandwidths fixed. Trainable bandwidths remain positive through a
logarithmic parameterization.

## Performance

The fused Gaussian sum reduces the dominant intermediate from approximately
`B * N * nx * ny` values to `B * C * (nx + ny)` values, where `C` is the
configured chunk size, or `N` when unchunked. Version 0.3.1 also removes an
unnecessary unchunked accumulator and reduces elementwise work in the sampled
Gaussian factors. The integrated representation uses tail-stable Gaussian
interval formulas that retain both values and gradients at numerically
difficult pixel boundaries.

Run the retained materialized implementation against the fused eager path with:

```bash
python benchmarks/benchmark_gaussian_aggregation.py --device cpu --threads 5
python benchmarks/benchmark_gaussian_aggregation.py --device cuda --points 10000
```

For a fixed-shape workload, benchmark TorchInductor as well:

```bash
python benchmarks/benchmark_gaussian_aggregation.py \
  --device cuda --points 10000 --compile --compile-mode reduce-overhead
```

The benchmark verifies numerical agreement before timing. Compilation time is
reported separately and excluded from steady-state timing. `torch.compile` is
intentionally opt-in: compilation has a first-call cost, and the speedup depends
on device, representation, diagram length, and shape stability. For training,
you can compile the complete model directly:

```python
model = torch.compile(model, fullgraph=True, mode="reduce-overhead")
```

On the development CPU (PyTorch 2.10, float32, 5 threads), with batch size 8,
5,000 points per diagram, and a `30 x 20` grid, representative eager runs kept
the fused sampled operator around an order of magnitude faster than explicit
materialization and the fused integrated operator around 3x faster. The
dominant intermediate estimate for this case falls from about 91.6 MiB to
7.6 MiB when unchunked. These numbers are environment-specific; use the
included benchmark for any performance claim.

`chunk_size` is also a tuning parameter rather than only a memory limit. A
moderate chunk can improve CPU cache locality on large diagrams, while very
small chunks add Python/kernel-launch overhead. Keep `chunk_size=None` as the
general default and benchmark representative values such as 512, 2048, and
8192 on the target CPU/GPU before freezing an experiment configuration.

CUDA peak temporary allocation is reported automatically when the benchmark is
run with `--device cuda`. This release candidate was validated in a CPU-only
environment, so GPU speed and memory numbers should be measured on the target
CUDA system before publication.

## Repository layout

```text
src/perslay/
  data.py      Dataset and collate utilities for persistence-diagram CSV files
  layers.py    PersLay layers, feature maps, and point-weight functions
  models.py    Single-branch and multi-branch PersLay regressors

experiments/
  train_regression.py       Config-driven regression training script
  configs/degree2_regression.yaml
  configs/grid_sampled_regression.yaml
  configs/integrated_regression.yaml
  configs/h0_h2_regression.yaml

benchmarks/
  benchmark_gaussian_aggregation.py
```

## Running experiments

Prepare persistence diagram CSV files with columns similar to:

```csv
birth,death
0.1,0.4
0.2,0.8
```

Prepare a target CSV with:

```csv
filename,target
sample_001.csv,3.14
```

Then run:

```bash
python experiments/train_regression.py --config experiments/configs/degree2_regression.yaml
```

To run the explicit centre-sampled model:

```bash
python experiments/train_regression.py \
  --config experiments/configs/grid_sampled_regression.yaml
```

The sampled and integrated example configurations both use a `15 x 15` grid,
zero padding, normalized Gaussian kernels, row-major flattening, and normalized
learnable-power point weights. They differ only in centre sampling versus exact
pixel integration. Bandwidths are initialized from pixel sizes and then
learned. Use `learnable_sigma: false` for a fixed-bandwidth ablation. The Rust-
compatible unnormalized, constant-weight setup is shown separately in the
quick-start example above.

Run the matched integrated example with:

```bash
python experiments/train_regression.py \
  --config experiments/configs/integrated_regression.yaml
```

For multi-diagram input, use:

```bash
python experiments/train_regression.py --config experiments/configs/h0_h2_regression.yaml
```

Training writes checkpoints and resolved configs under the configured `output.run_dir`.

### Synthetic apparent-modulus study

The manuscript experiment uses repeated grouped cross-validation rather than
the simple random split in `train_regression.py`. Baseline, uniformly thinned,
and locally weakened siblings sharing a `structure_XXXX` identifier always
remain in the same fold. See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the expected
data layout, complete commands, resumption behavior, and analysis steps.

Run a short validation experiment before starting the full study:

```bash
python experiments/cross_validate_bone_modulus_perslay.py \
  --data-root data \
  --output-dir outputs/feature_map_smoke \
  --variants baseline uniform weakened \
  --folds 2 --repeats 1 --epochs 3 --patience 2 \
  --models variant_residual \
  --dimension-sets h0 \
  --feature-map-types grid_sampled integrated \
  --gaussian-normalization normalized \
  --flatten-order row-major \
  --device auto
```

For the controlled sampled-versus-integrated comparison, use identical
Gaussian normalization and vector ordering so that the discretization is the
intended difference:

```bash
python experiments/cross_validate_bone_modulus_perslay.py \
  --data-root data \
  --output-dir outputs/bone_modulus_discretization_comparison \
  --variants baseline uniform weakened \
  --folds 5 --repeats 10 \
  --models variant_residual \
  --dimension-sets h0 h2 h0_h2 \
  --feature-map-types grid_sampled integrated \
  --gaussian-normalization normalized \
  --flatten-order row-major \
  --learnable-sigma \
  --image-size 10 10 \
  --bounds-padding-fraction 0.05 \
  --sigma-multiplier 1.0 \
  --weight-type normalized_learnable_power \
  --hidden-dimension 128 --dropout 0.10 \
  --batch-size 4 --epochs 300 --patience 40 \
  --learning-rate 0.0005 --weight-decay 0.0001 \
  --random-seed 2026 \
  --device auto
```

To compare each representation in its conventional form instead, replace
`--gaussian-normalization normalized` and `--flatten-order row-major` with:

```bash
--gaussian-normalization representation-default \
--flatten-order representation-default
```

The representation defaults are an unnormalized, legacy-x-major
centre-sampled surface and a normalized, row-major pixel-integrated image.
Both feature maps use the same grouped folds and the same initialization seed
within each fold. The resolved choices are written to `config.json`, every
per-fold `result.json`, and the output CSV files.

Compare their repeated out-of-fold predictions with a structure-level paired
bootstrap:

```bash
python experiments/compare_bone_modulus_feature_maps.py \
  --cv-output outputs/bone_modulus_discretization_comparison \
  --model perslay_variant_residual \
  --dimension-set h0_h2 \
  --feature-map-types grid_sampled integrated \
  --bootstrap 5000 \
  --seed 2026
```

Analyze one representation against the variant-aware BV/TV baseline with:

```bash
python experiments/analyze_bone_modulus_cv.py \
  --cv-output outputs/bone_modulus_discretization_comparison \
  --output-dir outputs/bone_modulus_discretization_comparison/post_analysis_integrated \
  --model perslay_variant_residual \
  --dimension-set h0_h2 \
  --feature-map-type integrated \
  --baseline variant_bvtv \
  --bootstrap 5000 \
  --seed 2026
```

Reusing the same output directory resumes incomplete runs. Use a new output
directory for an independent rerun. `--overwrite-output` deliberately removes
the selected output directory, while `--retrain-existing` retrains completed
per-fold runs without deleting other files.

## Citation

If you use this package in research, please cite the original PersLay paper:

> Mathieu Carrière, Frédéric Chazal, Yuichi Ike, Théo Lacombe, Martin Royer, and Yuhei Umeda. **PersLay: A Neural Network Layer for Persistence Diagrams and New Graph Topological Signatures.** AISTATS 2020, PMLR 108:2786–2796.

## License

This project is distributed under the MIT License. See `LICENSE` for details.


## v0.4.1: memory-aware training on CPU

The explicit Gaussian feature maps support optional non-reentrant activation
checkpointing of the fused point-chunk matrix products. Supply both an explicit
chunk size and `checkpoint_chunks=True`:

```python
from perslay import IntegratedGaussianPerslayPhi

phi = IntegratedGaussianPerslayPhi(
    image_size=[20, 20],
    image_bnds=[[0.0, 1.0], [0.0, 1.0]],
    sigma_x=0.05,
    sigma_y=0.05,
    chunk_size=128,
    checkpoint_chunks=True,
)
```

This affects training/backward only; inference does not need checkpointing.
Checkpointing recomputes intermediate factors during backpropagation, usually
lowering saved activation memory at the expense of additional runtime. The
optimal chunk size depends on the device, batch, number of points, and grid.
The same settings can be passed to `PerslayRegressor`, `MultiPerslayRegressor`,
or `make_perslay_phi` for `grid_sampled`, `integrated`, and
`integrated_learnable_grid` feature maps. Legacy Gaussian maps are unchanged.

Input CSV datasets drop non-finite points (including essential infinite-death
classes) and nonpositive-persistence pairs. Padding masks are boolean `True`
for valid points. All-masked batches produce zeros, and masked mean divides by
the number of real points. Do not use non-finite **unmasked** points; model them
separately or preprocess them before calling the layer.

A small CPU-only benchmark is available:

```bash
PYTHONPATH=src python benchmarks/benchmark_checkpoint_memory.py
```

CPU benchmark outcomes and validation limitations are recorded in
[`VALIDATION_0.4.1.md`](VALIDATION_0.4.1.md).
