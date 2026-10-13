# torchPersLay v0.5.0 candidate — CPU validation

Date: 2026-10-13. Local release candidate; not a claim of a PyPI release.

## Scope

Introduced an optional `backward_backend="analytic"` for exact integrated
Gaussian and learned-grid integrated Gaussian representations. The default
remains `"autograd"`, and old `state_dict` entries load unchanged. The fused
forward computes exactly the same mathematical weighted image representation,
subject to floating-point accumulation-order differences. The backwards
implementation is an analytical first-order vector-Jacobian product with
bounded, recomputed point chunks.

## Test summary

Command: `PYTHONPATH=src pytest -q`

**93 passed, 21 subtests passed** on Python 3.13 / PyTorch 2.10.0+cpu.

Validation includes:
- Float32/float64 first-order forward/backward parity, normalized and unnormalized
- Fixed and globally learned grids, chunked and unchunked paths
- Float64 `gradcheck` for points, weights, sigmas and grid boundaries
- Masked nonfinite padded inputs, empty diagrams, positive/negative weights
- Full single-branch and multi-branch regressor loss/backward
- Strict loading of v0.4.1-compatible state_dict, default behavior preserved
- Narrow and wide sigma sweeps (0.015 to 1.2)
- Explicit rejection of incompatible `checkpoint_chunks=True` and analytical mode
- `torch.compile` smoke tests: fullgraph eager plus a small fullgraph CPU-Inductor

Limitations: CPU-only environment, no AMP, GPU, distributed training or multiple
PyTorch versions. Higher-order derivatives and `torch.func` transforms are not
supported by the analytical backend; select `"autograd"` for those workflows.

## Independent-process, end-to-end CPU benchmark

4 threads. Full `PerslayRegressor`, learned grid and learned point weights;
forward + loss + backward, 3 repetitions per process. Peak RSS reflects the
whole process (Python + Torch runtime), not only tensor payload. Each backend
ran in a separate Python process. All use identical random seeds and model
initializations. Benchmarks are hardware- and workload-dependent.

| Batch × points | Grid | Backend | Median time | Peak RSS |
|---|---|---|---:|---:|
| 4 × 8192 | 60 × 60 | Standard autograd | 87.63 ms | 459.9 MiB |
| 4 × 8192 | 60 × 60 | Analytical backward | 70.97 ms | 311.1 MiB |
| 4 × 8192 | 60 × 60 | Checkpointed autograd | 105.86 ms | 371.6 MiB |
| 4 × 16384 | 80 × 80 | Standard autograd | 221.94 ms | 788.4 MiB |
| 4 × 16384 | 80 × 80 | Analytical backward | 171.44 ms | 337.3 MiB |
| 4 × 16384 | 80 × 80 | Checkpointed autograd | 306.40 ms | 421.3 MiB |

On the larger case, analytical backward was ~1.29× faster and used ~57.2%
less peak RSS than standard autograd. Smaller inputs can have different
performance characteristics. No claims about CUDA performance are warranted.

Repeat using e.g.

```bash
PYTHONPATH=src python benchmarks/benchmark_analytic_model_cpu.py --backend autograd
PYTHONPATH=src python benchmarks/benchmark_analytic_model_cpu.py --backend analytic
PYTHONPATH=src python benchmarks/benchmark_analytic_model_cpu.py --backend checkpoint
```

## Experimental approximate branch

The Chebyshev study is **not** part of the v0.5.0 stable API. The separate
research archive has an opt-in runtime probe-based tolerance check, per-axis
rank adaptation, and exact fallback for out-of-bounds or insufficient rank.
The numerical probe cannot guarantee global or gradient error bounds. It
must be validated for every target scientific workflow before consideration
for the stable library.
