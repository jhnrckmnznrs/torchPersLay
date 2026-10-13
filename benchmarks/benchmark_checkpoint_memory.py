"""CPU-only saved-tensor footprint and backward-time comparison.

The saved-tensor payload measures unique tensors retained by autograd hooks
(without allocating copies). It is NOT process peak RSS, which also includes
model weights, PyTorch runtime caches, temporary allocations, and interpreter
memory. Values are intended for relative comparison on the same workload.
"""
import argparse
import statistics
import time

import torch

from perslay import (
    GridSampledGaussianPerslayPhi,
    IntegratedGaussianPerslayPhi,
    LearnableGridIntegratedGaussianPerslayPhi,
)


CLASSES = {
    "sampled": GridSampledGaussianPerslayPhi,
    "integrated": IntegratedGaussianPerslayPhi,
    "learnable_grid": LearnableGridIntegratedGaussianPerslayPhi,
}


def measured_step(phi, diagrams, weights):
    # Saved tensors are counted at the time of saving. This represents a
    # cumulative activation payload, not unique tensor storage or peak RSS.
    saved_bytes = 0

    def save_hook(tensor):
        nonlocal saved_bytes
        saved_bytes += tensor.numel() * tensor.element_size()
        return tensor

    phi.zero_grad(set_to_none=True)
    with torch.autograd.graph.saved_tensors_hooks(save_hook, lambda tensor: tensor):
        beginning = time.perf_counter()
        output = phi.weighted_sum(diagrams, weights)
        loss = output.square().mean()
        loss.backward()
        elapsed = time.perf_counter() - beginning
    return elapsed, saved_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=20)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(vars(args).values()) <= 0:
        parser.error("all numeric arguments must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(73)
    d = torch.rand(args.batch, args.points, 2)
    d[..., 1] = d[..., 0] + torch.rand(args.batch, args.points) * 0.5
    w = torch.rand(args.batch, args.points)
    print(f"PyTorch {torch.__version__}, CPU, threads={args.threads}, batch={args.batch}, "
          f"points={args.points}, grid=({args.nx},{args.ny}), chunk={args.chunk}")
    print("Method | checkpoint | median forward+backward (ms) | saved tensor payload (MiB)")
    for name, cls in CLASSES.items():
        for mode in [False, True]:
            phi = cls(
                image_size=[args.nx, args.ny],
                image_bnds=[[0.0, 1.0], [0.0, 1.6]],
                sigma_x=0.1, sigma_y=0.14,
                normalize=True, chunk_size=args.chunk,
                checkpoint_chunks=mode,
            )
            # Warm up once and measure with identical settings.
            measured_step(phi, d, w)
            measurements = [measured_step(phi, d, w) for _ in range(args.repeats)]
            median_time = statistics.median(v[0] for v in measurements) * 1000
            median_memory = statistics.median(v[1] for v in measurements) / (1024**2)
            print(f"{name:<14} | {str(mode):<10} | {median_time:>12.2f} | {median_memory:>8.3f}")


if __name__ == "__main__":
    main()
