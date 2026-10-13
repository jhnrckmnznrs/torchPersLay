"""Compare actual peak process RSS in independent CPU processes.

Defaults intentionally use a larger feature map than the microbenchmark,
because checkpointing has nontrivial bookkeeping overhead that can dominate
small workloads. Peak RSS includes the Python/PyTorch runtime and native caches.
Run with: PYTHONPATH=src python benchmarks/benchmark_checkpoint_peak_rss.py
"""
import argparse
import resource
import statistics
import subprocess
import sys
import time

import torch

from perslay import LearnableGridIntegratedGaussianPerslayPhi


def run_child(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(26)
    phi = LearnableGridIntegratedGaussianPerslayPhi(
        image_size=[args.grid, args.grid],
        image_bnds=[[0.0, 1.0], [0.0, 1.5]],
        sigma_x=0.09,
        sigma_y=0.13,
        chunk_size=args.chunk,
        checkpoint_chunks=args.mode == "checkpoint",
    )
    diagrams = torch.rand(args.batch, args.points, 2)
    diagrams[..., 1] = diagrams[..., 0] + torch.rand(
        args.batch, args.points
    ) * 0.5
    weights = torch.rand(args.batch, args.points)
    t0 = time.perf_counter()
    phi.weighted_sum(diagrams, weights).square().mean().backward()
    elapsed = time.perf_counter() - t0
    # Linux ru_maxrss is KiB, unlike macOS which reports bytes.
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"{elapsed:.6f},{peak_mib:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--points", type=int, default=16384)
    parser.add_argument("--grid", type=int, default=80)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", choices=("standard", "checkpoint"))
    args = parser.parse_args()
    if min(args.batch, args.points, args.grid, args.chunk, args.threads, args.repeats) < 1:
        parser.error("all numeric arguments must be positive")
    if args.mode:
        run_child(args)
        return
    print(f"CPU, batch={args.batch}, points={args.points}, "
          f"grid={args.grid}x{args.grid}, chunk={args.chunk}, "
          f"threads={args.threads}; separate processes, {args.repeats} runs/mode")
    for mode in ("standard", "checkpoint"):
        data = []
        for _ in range(args.repeats):
            command = [sys.executable, __file__,
                "--mode", mode, "--batch", str(args.batch),
                "--points", str(args.points), "--grid", str(args.grid),
                "--chunk", str(args.chunk), "--threads", str(args.threads),
                "--repeats", "1"]
            output = subprocess.check_output(command, text=True).strip()
            data.append([float(value) for value in output.split(",")])
        median_time = statistics.median(value[0] for value in data)
        median_rss = statistics.median(value[1] for value in data)
        print(f"{mode:>10}: median compute {median_time:.3f}s, "
              f"median peak process RSS {median_rss:.1f} MiB")


if __name__ == "__main__":
    main()
