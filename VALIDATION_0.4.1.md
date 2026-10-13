# torchPersLay 0.4.1 — CPU validation report

Status: local release candidate; **not published to GitHub or PyPI**.

## Scope

This release fixes non-finite points and masked reductions, and adds an opt-in
non-reentrant PyTorch activation-checkpointing path for fused Gaussian sums.
The mathematical feature maps, trainable parameters, and default operation
(`checkpoint_chunks=False`) have been preserved.

## Correctness

On a Linux CPU environment using Python 3.13, PyTorch 2.10.0+cpu:

```text
PYTHONPATH=src pytest -q
57 passed, 21 subtests passed
python -m compileall -q src tests benchmarks experiments examples
```

New tests include: non-finite CSV input and entirely empty diagrams, padding with
`NaN`/`inf`, masked sum/mean/max/min/top-k, batch-empty reductions, parameter
and input gradient parity of checkpointed versus ordinary fused algorithms,
end-to-end regressor gradient parity, and fullgraph PyTorch compile using the
`eager` backend. The inherited v0.4.0 tests include the 180-step CPU synthetic
experiment that moves a learned boundary from 0.5 to approximately 0.3.

**Not performed:** real-dataset training/generalization evaluation, GPU profiling,
TorchInductor checkpointed full-graph evaluation, mixed precision benchmarking,
or end-to-end memory profiling for an entire production model.

## CPU benchmark: activation payload

```text
PYTHONPATH=src python benchmarks/benchmark_checkpoint_memory.py --repeats 5

PyTorch 2.10.0+cpu, threads=4, batch=2, points=1024, grid=20x20, chunk=128
Representation    Checkpoint   Median fwd+bwd      Saved autograd tensor payload
sampled           False             2.42 ms                        0.952 MiB
sampled           True              7.56 ms                        0.030 MiB
integrated        False             8.79 ms                        2.105 MiB
integrated        True             14.16 ms                        0.027 MiB
learnable_grid    False             9.59 ms                        2.106 MiB
learnable_grid    True             17.53 ms                        0.027 MiB
```

The payload counts tensors saved through `torch.autograd.graph.saved_tensors_hooks`.
It is not peak process RSS, peak allocated memory, or a proof of equivalent
speed/memory tradeoffs on other workloads. The smaller sampled-kernel timing
varied considerably between runs; rebenchmark on the target machine.

## CPU benchmark: peak process resident memory

```text
PYTHONPATH=src python benchmarks/benchmark_checkpoint_peak_rss.py

batch=4, points=16384, grid=80x80, chunk=256, threads=4
3 independent processes per mode, median:

                Compute time       Peak process RSS
standard             0.326 s           637.4 MiB
checkpoint           1.006 s           347.4 MiB
```

This workload showed approximately 45.5% less total peak RSS but 3.1 times
longer forward+backward compute when checkpointing was enabled. Peak RSS includes
Python/PyTorch runtime memory and allocator/cache overhead, so differences on
small workloads can be muted or even reversed. On a smaller CPU run of four
4096-point diagrams with a 40x40 grid, checkpointing **increased** observed peak
RSS from approximately 305 MiB to 318 MiB. For small datasets, the ordinary fused
path is the recommended default.

## Release policy

Keep checkpointing disabled by default. Enable it only after measuring peak RSS
and wall time on the actual workload. Consider `checkpoint_chunks=True` with
an explicit `chunk_size` for sufficiently large training jobs constrained by RAM.
The finite persistence-image feature maps intentionally exclude essential
classes; modeling essential classes requires a separate finite encoding.
