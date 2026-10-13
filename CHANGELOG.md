## 0.5.0 (candidate; CPU validated)

- Add opt-in `backward_backend="analytic"` to exact integrated and learnable-grid persistence-image layers and model factories, retaining the existing autograd default.
- Compute first-order gradients of point coordinates, learned weights, Gaussian bandwidths, and learned pixel edges with a chunked analytical vector-Jacobian product.
- Preserve all v0.4.1 checkpoint keys and model defaults; support `torch.load` / `state_dict` across the backend switch.
- Reject `checkpoint_chunks=True` together with `backward_backend="analytic"` to avoid ambiguous memory settings.
- Document first-order-only scope and CPU benchmark caveats; experimental Chebyshev interpolation remains outside the stable package.

# Changelog

## 0.4.1 (release candidate)

- Make single- and multi-diagram CSV datasets consistently discard non-finite points and nonpositive-persistence pairs, and support empty input files and empty batches.
- Sanitize masked coordinates before nonlinear point weighting to prevent NaN/Inf padding from contaminating embeddings; reject mismatched masks.
- Correct max/min masked aggregation, implement true masked means (including the fused mean path), and fix `topkN` parsing, insufficient points, and all-masked cases.
- Add opt-in `checkpoint_chunks=True` for the sampled, integrated, and learnable-grid fused sum paths. Use non-reentrant activation checkpointing and recomputation during backward to lower saved-tensor memory at some compute cost.
- Propagate checkpoint settings through the PersLay model factory and single-/multi-branch regressors.
- Add correctness, gradient-parity, and CPU saved-tensor-memory tests and a CPU benchmark script.

# Changelog

## 0.4.0

- Add `LearnableGridIntegratedGaussianPerslayPhi`, a canonical exact persistence-image layer with globally learned nonuniform birth and persistence pixel boundaries.
- Parameterize cell widths by softmax logits with a configurable minimum-width floor, guaranteeing ordered boundaries, fixed outer bounds, and positive widths throughout optimization.
- Initialize width logits at zero so the learnable-grid representation exactly reproduces the ordinary uniform integrated persistence image before training.
- Preserve the fused `Y^T diag(w) X` aggregation and exact point chunking for learned grids.
- Add `grid_edges()`, `grid_width_fractions()`, `grid_regularization()`, and `regularization_loss()` inspection/training helpers.
- Add `feature_map_type="integrated_learnable_grid"` to the model factory and single-/multi-branch regressors.
- Validate learned-boundary gradients against centered finite differences, checkpoint round trips, chunked equivalence, full-graph `torch.compile`, and a synthetic optimization in which a boundary moves from 0.5 to the analytically informative location 0.3.

## 0.3.1

- Reduce eager-mode arithmetic in the centre-sampled Gaussian factors by using
  precomputed inverse variance multipliers instead of per-element division.
- Avoid a zero-initialized accumulator on the common unchunked fused path;
  chunked execution retains exact additive accumulation.
- Rework canonical integrated-pixel interval masses to preserve three
  properties simultaneously: float32 tail accuracy, tiny central-interval
  accuracy, and correct gradients when a Gaussian mean lies exactly on a pixel
  edge.
- Add float64 reference, tiny-central-mass, and pixel-edge gradient tests for
  the integrated Gaussian interval calculation.
- Verify that both fused Gaussian operators and a complete integrated
  `PerslayRegressor` capture as a single `torch.compile` graph and that the
  compiled regressor backpropagates through all trainable parameters.
- Extend the Gaussian benchmark with dtype, CPU-thread, `torch.compile` mode,
  first-call compilation cost, and CUDA temporary-memory reporting.
- Document `torch.compile` and chunk-size tuning as workload-dependent options
  rather than enabling either globally by default.

## 0.3.0

- Add a fused sum-aggregation path for `GridSampledGaussianPerslayPhi` and
  `IntegratedGaussianPerslayPhi` that avoids materializing the full
  `[batch, points, ny, nx]` tensor.
- Exploit separability of both sampled Gaussian surfaces and exact integrated
  Gaussian pixels using batched matrix multiplication.
- Add optional exact point chunking through `chunk_size` for memory-bounded
  processing of large persistence diagrams.
- Preserve the existing pointwise `forward()` API and automatically use the
  fused path only when `Perslay` receives `perm_op=torch.sum`.
- Replace cancellation-prone Gaussian CDF subtraction in the integrated map
  with a tail-stable `erf`/`erfc` interval-mass calculation.
- Make sampled-grid buffers contiguous so `state_dict` checkpoint loading is
  safe on PyTorch versions where `meshgrid` returns overlapping expanded views.
- Add output, gradient, masking, chunking, tail-accuracy, empty-diagram, and
  checkpoint round-trip regression tests.
- Add `benchmarks/benchmark_gaussian_aggregation.py` for reproducible
  materialized-versus-fused CPU/GPU timing.

## 0.2.1

- Wire the legacy, centre-sampled, and pixel-integrated Gaussian maps into the
  repeated grouped synthetic-modulus cross-validation pipeline.
- Permit multiple feature maps in one run while reusing identical grouped
  folds and initialization seeds.
- Record feature-map type, kernel normalization, bandwidth learning, and
  flattening order in run signatures, JSON metadata, and result tables.
- Add a matched paired analysis with structure-level bootstrap intervals for
  sampled-versus-integrated predictions.
- Make the post-analysis representation-aware and prevent accidental pooling
  of predictions from different feature maps.
- Consolidate the grouped experiment under one canonical entry point while
  retaining the former v2 script as a compatibility wrapper.
- Add experiment-configuration and representation-aware aggregation tests.

## 0.2.0

- Add `GridSampledGaussianPerslayPhi`, which evaluates Gaussian persistence surfaces at pixel centres and can reproduce the Rust sampled representation.
- Add `IntegratedGaussianPerslayPhi`, which computes exact Gaussian probability mass over each pixel using differentiable error-function differences.
- Preserve the pre-0.2 `GaussianPerslayPhi` as `feature_map_type: legacy_gaussian` for checkpoint compatibility.
- Add normalized or unnormalized kernels, fixed or trainable bandwidths, and row-major or legacy x-major flattening to the new feature maps.
- Add fixed persistence and linear-ramp point weights.
- Extend `PerslayRegressor` and `MultiPerslayRegressor` with configurable feature-map types and weighting parameters.
- Add explicit grid-sampled and integrated regression configuration examples.
- Add numerical tests for grid centres, normalization, masking, pixel integration, weighting, flattening, and model construction.

## 0.1.5

- Native PyTorch PersLay implementation with Gaussian feature maps, learnable weights, single-branch regression, and multi-branch regression.
