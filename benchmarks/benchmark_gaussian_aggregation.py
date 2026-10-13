"""Benchmark fused Gaussian PersLay aggregation.

Examples:

    python benchmarks/benchmark_gaussian_aggregation.py --device cpu
    python benchmarks/benchmark_gaussian_aggregation.py --device cuda --points 10000
    python benchmarks/benchmark_gaussian_aggregation.py --device cuda --compile

The materialized path is retained only as a correctness/performance reference.
Compilation time is reported separately and excluded from steady-state timing.
"""

import argparse
import statistics
import time

import torch

from perslay import GridSampledGaussianPerslayPhi, IntegratedGaussianPerslayPhi


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(function, device, warmup=5, repeats=20):
    for _ in range(warmup):
        function()
    synchronize(device)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        synchronize(device)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def materialized(phi, diagrams, weights):
    values, _ = phi(diagrams)
    weighted = values * weights[..., None, None, None]
    return weighted.sum(dim=1)


def cuda_peak_delta(function, device):
    if device.type != "cuda":
        return None
    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    function()
    synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return max(0, peak - baseline)


def compile_function(function, mode):
    kwargs = {"fullgraph": True}
    if mode != "default":
        kwargs["mode"] = mode
    return torch.compile(function, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--points", type=int, default=5000)
    parser.add_argument("--nx", type=int, default=30)
    parser.add_argument("--ny", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    if args.compile and not hasattr(torch, "compile"):
        raise SystemExit("--compile requires a PyTorch version with torch.compile")
    if args.threads is not None:
        if args.threads <= 0:
            raise SystemExit("--threads must be positive")
        torch.set_num_threads(args.threads)

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    birth = torch.rand(args.batch_size, args.points, device=device, dtype=dtype)
    persistence = torch.rand(
        args.batch_size, args.points, device=device, dtype=dtype
    )
    diagrams = torch.stack((birth, birth + persistence), dim=-1)
    weights = torch.rand(args.batch_size, args.points, device=device, dtype=dtype)

    common = dict(
        image_size=[args.nx, args.ny],
        image_bnds=[[0.0, 1.0], [0.0, 1.0]],
        sigma_x=0.08,
        sigma_y=0.1,
        normalize=True,
        learnable_sigma=False,
        flatten_order="row-major",
        chunk_size=args.chunk_size,
    )

    print(
        f"device={device} dtype={args.dtype} batch={args.batch_size} "
        f"points={args.points} grid={args.nx}x{args.ny} "
        f"chunk_size={args.chunk_size} threads={torch.get_num_threads()}"
    )
    if args.compile:
        print(f"torch.compile mode={args.compile_mode}; first-call time excluded")

    header = "representation           materialized_ms   fused_eager_ms   speedup"
    if args.compile:
        header += "   compiled_ms   compile_gain   first_call_s"
    print(header)

    for name, constructor in (
        ("grid_sampled", GridSampledGaussianPerslayPhi),
        ("integrated", IntegratedGaussianPerslayPhi),
    ):
        phi = constructor(**common).to(device=device, dtype=dtype)
        expected = materialized(phi, diagrams, weights)
        actual = phi.weighted_sum(diagrams, weights)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

        materialized_time = time_call(
            lambda: materialized(phi, diagrams, weights),
            device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        eager_time = time_call(
            lambda: phi.weighted_sum(diagrams, weights),
            device,
            warmup=args.warmup,
            repeats=args.repeats,
        )

        line = (
            f"{name:22s} {materialized_time * 1e3:15.3f} "
            f"{eager_time * 1e3:16.3f} "
            f"{materialized_time / eager_time:9.2f}x"
        )

        if args.compile:
            compiled = compile_function(phi.weighted_sum, args.compile_mode)
            start = time.perf_counter()
            compiled_actual = compiled(diagrams, weights)
            synchronize(device)
            first_call = time.perf_counter() - start
            torch.testing.assert_close(
                compiled_actual, actual, rtol=2e-5, atol=2e-6
            )
            compiled_time = time_call(
                lambda: compiled(diagrams, weights),
                device,
                warmup=max(1, min(args.warmup, 3)),
                repeats=args.repeats,
            )
            line += (
                f" {compiled_time * 1e3:13.3f} "
                f"{eager_time / compiled_time:12.2f}x "
                f"{first_call:12.3f}"
            )

        print(line)

        if device.type == "cuda":
            materialized_peak = cuda_peak_delta(
                lambda: materialized(phi, diagrams, weights), device
            )
            fused_peak = cuda_peak_delta(
                lambda: phi.weighted_sum(diagrams, weights), device
            )
            print(
                f"  CUDA temporary peak estimate: materialized="
                f"{materialized_peak / 2**20:.1f} MiB, "
                f"fused={fused_peak / 2**20:.1f} MiB"
            )

    element_size = diagrams.element_size()
    materialized_bytes = (
        args.batch_size * args.points * args.nx * args.ny * element_size
    )
    separable_bytes = (
        args.batch_size
        * min(args.points, args.chunk_size or args.points)
        * (args.nx + args.ny)
        * element_size
    )
    print()
    print(
        "dominant intermediate estimate: "
        f"materialized={materialized_bytes / 2**20:.1f} MiB, "
        f"fused-axis={separable_bytes / 2**20:.1f} MiB"
    )


if __name__ == "__main__":
    main()
