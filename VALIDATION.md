# Validation record for version 0.4.0

Version 0.4.0 adds a globally learned nonuniform grid to the exact
pixel-integrated persistence-image implementation while retaining the fused
aggregation, chunking, numerical-stability, and compile behavior from 0.3.1.

## Learnable-grid mathematics and invariants

For an axis containing `n` cells, unconstrained trainable logits `theta` are
converted to normalized widths by

```text
q = r / n + (1 - r) * softmax(theta),
```

where `0 <= r < 1` is `min_cell_fraction`. Consequently the widths sum to one,
remain positive (with an explicit floor when `r > 0`), and generate strictly
ordered internal boundaries through cumulative sums. The outer image bounds
are constants. Zero logits generate exactly uniform widths, so version 0.4.0
starts from the canonical fixed-grid representation rather than a random grid.

For a Gaussian with mean `mu` and scale `sigma`, one one-dimensional cell mass
is

```text
A_j = Phi((x_{j+1} - mu) / sigma) - Phi((x_j - mu) / sigma).
```

An internal edge `x_j` is the upper edge of one cell and the lower edge of the
next. By the fundamental theorem of calculus its derivative contributes the
Gaussian density with opposite signs to those adjacent cells. Thus the exact
integrated representation is differentiable with respect to grid geometry and
does not require hard or straight-through bin assignments.

## Correctness checks

The full local suite validates all 0.3.1 behaviors plus:

- uniform learned-grid initialization versus the fixed integrated persistence
  image in float64;
- exact fixed outer bounds, ordered internal edges, normalized cell widths,
  and the configured minimum-width floor under extreme logits;
- learned-grid autograd derivatives versus centered finite differences;
- a two-class synthetic task in which gradient optimization moves the only
  birth-axis internal boundary from 0.500 to approximately 0.300, the
  discriminative midpoint between Gaussian centers at 0.2 and 0.4;
- increased representation separation on that synthetic task;
- zero uniform-grid regularization at initialization and positive penalty after
  grid deformation;
- model-factory and `PerslayRegressor` integration;
- learned-grid checkpoint save/load round trips;
- exact chunked versus unchunked adaptive aggregation; and
- full-graph `torch.compile` capture of the adaptive fused operator.

The release-candidate result is:

```text
45 passed, 7 subtests passed
```

## Existing fused/numerical checks retained

The suite continues to cover sampled/integrated materialized-versus-fused
outputs, gradients with respect to diagrams/weights/bandwidths, masking, empty
diagrams, row-major and legacy layouts, tail-stable Gaussian interval masses,
pixel-edge gradients, and compiled full-model backward propagation.

The fused learned-grid result remains mathematically
`Y^T diag(w) X`; only the one-dimensional cell edges used to form `X` and `Y`
are learned. Point chunking remains an exact additive partition of the same
sum up to floating-point addition order.

## Commands used

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src tests benchmarks experiments examples
```

This environment is CPU-only. CUDA benchmarking remains deferred until a CUDA
machine is available; no GPU-specific performance claim is made for 0.4.0.
