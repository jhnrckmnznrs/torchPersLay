# torchPersLay

**Differentiable persistence-diagram layers in PyTorch.**

`torchPersLay` transforms batches of persistence diagrams into fixed-size, trainable features for neural networks. The import name is `perslay`. It supports point-weight functions, Gaussian persistence surfaces, exact pixel-integrated persistence images, globally learnable nonuniform pixel boundaries, masking, and simple single-/multi-branch regression models.

## Install

To install the current source directly from GitHub:

```bash
pip install 'git+https://github.com/jhnrckmnznrs/torchPersLay.git'
```

The PyPI distribution uses the same project name (`torchPersLay`), but the GitHub source may contain improvements not yet uploaded to PyPI.

Requires Python 3.9 or newer and PyTorch. The package includes no datasets, trained checkpoints, or experiment outputs.

## Basic usage

```python
import torch
from perslay import PerslayRegressor

model = PerslayRegressor(
    image_size=[24, 24],
    image_bnds=[[0.0, 1.0], [0.0, 1.0]],  # birth and persistence ranges
    sigma_x=0.08,
    sigma_y=0.08,
    weight_type="learnable_power",
    feature_map_type="integrated_learnable_grid",
    normalize=True,
    backward_backend="analytic",       # optional memory-saving first-order VJP
    aggregation_backend="auto",         # exact histogram if beneficial
)

# Each point is [birth, death]; masked padded points are ignored.
diagrams = torch.tensor([
    [[0.1, 0.4], [0.2, 0.6]],
    [[0.0, 0.3], [0.3, 0.7]],
])
result = model(diagrams)
loss = result.square().mean() + model.regularization_loss()
loss.backward()
```

Use `feature_map_type="integrated"` for fixed pixel boundaries, `"grid_sampled"` for Gaussian centre samples, and `"legacy_gaussian"` for the original PersLay-style grid map. The canonical image uses **birth/persistence** coordinates even though the input diagrams use **birth/death** coordinates.

## Exact computation backends

Pixel-integrated Gaussian images have separable responses, so their weighted sum can be evaluated as `Y.T @ diag(weights) @ X` without constructing a `[batch, points, height, width]` tensor. All exact backends compute the same mathematical features; floating-point accumulation orders can differ.

- **`aggregation_backend="fused"`** (default): separable pointwise reduction; robust for arbitrary continuous diagrams.
- **`aggregation_backend="auto"`**: selects a dense coordinate histogram or sparse grouping of exact birth–persistence pairs if its cost and memory limits indicate an advantage; otherwise uses the fused path. The two coordinate axes are sorted only once.
- **`aggregation_backend="histogram"`**: attempts dense coordinate grouping, subject to the memory limit.
- **`aggregation_backend="pairs"`**: attempts grouping by occupied coordinate pairs without constructing the Cartesian-product histogram. Both explicit grouping modes fall back to fused computation if their safety limits are exceeded.

Coordinate grouping uses exact floating-point equality, **never rounding**. Identical birth–persistence pairs are evaluated only once while retaining differentiability of weights, bandwidths and learned grid boundaries. It does not group diagrams whose coordinates require gradients; those use the fused backend. Dynamic grouping is bypassed under `torch.compile` in favor of the fused backend.

### Reuse fixed-diagram coordinate grouping

When the same diagram tensor is reused across training steps, calculate the grouping once. Learned weights, Gaussian bandwidths, and grid boundaries are still updated on every step:

```python
from perslay import prepare_coordinate_pairs, prepare_histogram

# Use prepare_histogram() for a compact Cartesian-product grid.
# Use prepare_coordinate_pairs() when relatively few distinct pairs occur.
plan = prepare_coordinate_pairs(diagrams)  # optional: mask=boolean_mask
for step in range(10):
    model.zero_grad(set_to_none=True)
    prediction = model(diagrams, histogram_plan=plan)
    prediction.square().mean().backward()
    # optimizer.step() with your optimizer
```

A plan checks the exact diagram coordinates, shape, device, and dtype before use. After changing diagrams, padding masks, preprocessing, or device/dtype, build a new plan. The cache stores **no model parameters** and is not saved in `state_dict`. Dense preparation defaults to at most 131,072 histogram cells and 32 MiB per batch. Sparse pair preparation defaults to at most 16,384 distinct pairs and a 32 MiB plan budget, with an additional runtime-memory guard in automatic dispatch. Over-limit diagrams should use the regular fused backend. For masks, pass `mask` both when preparing the plan and when calling the model.

### Backward options

For integrated Gaussian maps, choose `backward_backend="autograd"` (default) or `"analytic"`. Analytical backward recomputes compact Gaussian factors, reducing saved activations and supporting first-order reverse-mode derivatives. It does not currently support double backward or `torch.func`; use autograd for those. The unnormalized analytical path reuses its existing weight gradient to evaluate the Gaussian scale-factor derivative, avoiding a redundant pass. `checkpoint_chunks=True` with an explicit `chunk_size` is another memory-saving choice when using autograd.

## CPU benchmarks

A synthetic 4-diagram, 8,192-point-per-diagram workload with an 80 × 80 learnable grid and four CPU threads gave these illustrative median forward+backward times (PyTorch 2.10 CPU, 4 timed steps after warmup):

| Input / backend | Median step time |
|---|---:|
| Discrete coordinates, analytical fused | 104 ms |
| Discrete coordinates, automatic histogram | 5.5 ms |
| Discrete coordinates, cached histogram | 2.6 ms |
| Continuous coordinates, analytical fused | 109 ms |

A separate correlated-coordinate benchmark (4 diagrams × 8,192 points, 512 distinct pairs, 80 × 80 grid, four CPU threads) measured about 84–90 ms for fused analytical backward, 8–10 ms for automatic sparse-pair dispatch, and 3–5 ms per step for cached pairs (approximately 5 ms one-time preparation). On continuous inputs, automatic dispatch fell back to fused computation with comparable runtime.

The cached times exclude one-time grouping preparation. Results depend strongly on actual distinct coordinate counts, bandwidths, CPU and thread settings; these are not general speed guarantees. On small cached workloads, **one CPU thread can be faster than several**. CUDA-specific performance has not been validated.

## Test and develop

```bash
pip install -e .
pip install pytest
pytest -q
```

The tests cover Gaussian output and gradient agreement, differentiability of learned grid boundaries and bandwidths, numerical edge cases, masking, cached-plan invalidation, full models, checkpoint loading, and PyTorch compilation smoke tests. The compact source tree intentionally excludes large datasets, saved model weights, and generated outputs.

## License and citation

MIT license; see `LICENSE`. For the underlying PersLay construction, cite Carrière, Chazal, Ike, Lacombe, Royer and Umeda (2020), *PersLay: A Neural Network Layer for Persistence Diagrams*, AISTATS / PMLR 108.
