"""Run in fresh processes to compare full-model CPU forward + backward RSS.

Example: PYTHONPATH=src python benchmarks/benchmark_analytic_model_cpu.py --backend analytic
"""
import argparse
import resource
import statistics
import time

import torch

from perslay.models import PerslayRegressor


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['autograd','analytic','checkpoint'],required=True)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--points',type=int,default=16384)
    parser.add_argument('--grid',type=int,default=80)
    parser.add_argument('--chunk',type=int,default=2048)
    parser.add_argument('--repeats',type=int,default=4)
    parser.add_argument('--threads',type=int,default=4)
    a=parser.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(28)
    layer=PerslayRegressor(
        image_size=[a.grid,a.grid],image_bnds=[[0,1],[0,1]],
        sigma_x=.1,sigma_y=.07,feature_map_type='integrated_learnable_grid',
        weight_type='learnable_power',normalize=True,learnable_sigma=True,
        backward_backend='analytic' if a.backend=='analytic' else 'autograd',
        checkpoint_chunks=a.backend=='checkpoint',
        chunk_size=a.chunk if a.backend!='autograd' else None,
        hidden_dim=16,
    )
    diagram=torch.rand(a.batch_size,a.points,2)
    diagram[...,1]+=diagram[...,0]
    target=torch.linspace(0,1,a.batch_size)
    def run():
        layer.zero_grad(set_to_none=True)
        predictions=layer(diagram)
        loss=(predictions-target).square().mean()+.1*layer.regularization_loss()
        loss.backward()
        return float(loss.detach())
    times=[]
    for _ in range(a.repeats):
        start=time.perf_counter()
        loss=run()
        times.append((time.perf_counter()-start)*1000)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f'backend={a.backend} grid={a.grid} points={a.points} '
          f'B={a.batch_size} time_ms_median={statistics.median(times):.2f} '
          f'peak_rss_mib={peak:.1f} loss={loss:.6f} torch={torch.__version__}',flush=True)


if __name__=='__main__':
    main()
