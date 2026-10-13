# Research branch — Chebyshev-compressed persistence images

**Experimental CPU research; not included in the stable torchPersLay v0.5.0
API.** Requires the `perslay` package (the 0.5.0 candidate is sufficient),
PyTorch and NumPy. No GPU or scientific dataset is needed to reproduce these
tests.

The prototype approximates each one-dimensional exact Gaussian pixel response
using barycentric Chebyshev interpolation, then accumulates weighted low-rank
moments before reconstructing the image:

\[
X_{ik}\approx \sum_r \ell_r(b_i) A_{rk},\qquad
Y_{ij}\approx\sum_s m_s(p_i) B_{sj},\qquad
P\approx B^T (M) A.
\]

The accumulation cost changes from roughly \(O(BNn_xn_y)\) to
\(O(BNR_xR_y)\), plus factorization/reconstruction. The method is not exact.

## Optional rank selection with exact fallback

```python
from adaptive_chebyshev import controlled_chebyshev_sum

image, decision = controlled_chebyshev_sum(
    phi, diagrams, weights,
    initial_rank=16,
    max_rank=96,
    tolerance=1e-4,
    probe_points=65,
    on_failure='exact',
    return_decision=True,
)
print(decision)  # approximate or exact-fallback, ranks and probe errors
```

For each axis, the implementation tests ranks 16, 32, 64, 96 (as permitted),
compares interpolated Gaussian integrated masses with the exact expression at
65 deterministic coordinate probes, and chooses the first passing rank. It
re-tests whenever sigma/grid parameters change, including during learning.
Insufficient rank, out-of-bounds inputs, or nonfinite inputs trigger exact
fallback unless `on_failure='raise'` is requested. Inputs inside bounds use
float64 interpolation basis by default to improve mean gradients.

**Important:** The error check is an *empirical finite probe*, not a rigorous
uniform bound, not an end-to-end training guarantee, and does not control
input, sigma, or grid-boundary gradient error. In particular, narrow Gaussian
bandwidths can cause much larger errors between probe locations. The stable
exact backend must remain the default; the approximation needs domain-specific
validation or mathematical error bounds before use in scientific conclusions.

## Tests

```bash
PYTHONPATH=/path/to/torchPersLay-0.5.0/src:$PWD pytest -q tests/
```

The tests cover broad-bandwidth success, narrow-bandwidth fallback, dynamically
changing learned sigma, differentiability, invalid rank and out-of-bounds
fallback/exception. Synthetic CPU-only examples; no external datasets.

The included `cheb_fused.py` remains an explicit approximate kernel; the
`adaptive_chebyshev.py` wrapper adds the safety checks and exact fallback.
See the earlier deep optimization report for exploratory speed/error curves.
