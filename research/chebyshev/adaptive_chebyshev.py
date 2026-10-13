"""RESEARCH ONLY: accuracy-tested Chebyshev persistence-image approximation.

Error control is measured on a deterministic, finite coordinate probe grid; it
is not a formal supremum-norm bound and does not guarantee input gradients.
Inappropriate for silent replacement of the exact backend.
"""

from dataclasses import dataclass
import torch

from cheb_fused import (_basis, _nodes_weights, cheb_weighted_sum)
from perslay.layers import IntegratedGaussianPerslayPhi


@dataclass(frozen=True)
class ApproximationDecision:
    backend: str
    rank_x: int | None
    rank_y: int | None
    probe_relative_error_x: float | None
    probe_relative_error_y: float | None


def _axis_error(phi, axis, rank, samples, device, dtype):
    bounds=phi.image_bnds[axis]
    sigma=phi._sigmas()[axis].detach()
    edges=(phi.grid_edges() if hasattr(phi,'grid_edges')
           else (phi.x_edges,phi.y_edges))[axis].detach()
    # Chebyshev nodes and interpolation basis calculated in high precision to
    # avoid cancellation close to interpolation nodes.
    n,w=_nodes_weights(bounds,rank,torch.float64,device)
    t=(torch.arange(samples,device=device,dtype=torch.float64)+0.5)/samples
    p=bounds[0]+(bounds[1]-bounds[0])*t
    interp=_basis(p,n,w)
    # Exact interval masses in 64-bit are a stricter probe than float32 math.
    target=IntegratedGaussianPerslayPhi._normal_interval_masses(
        edges.double(),p,sigma.double())
    from_nodes=IntegratedGaussianPerslayPhi._normal_interval_masses(
        edges.double(),n,sigma.double())
    predicted=interp@from_nodes
    relative=(predicted-target).norm()/target.norm().clamp_min(1e-12)
    return float(relative)


def controlled_chebyshev_sum(phi, diagrams, weights, *,
                              initial_rank=16, max_rank=96,
                              tolerance=1e-4, probe_points=65,
                              on_failure='exact', return_decision=False):
    """Try a verified interpolation rank per axis; optionally fall back exact.

    Probe errors test 1D Gaussian **values**, not training gradients. Thus,
    even when successful, approximation error is not a global guarantee.
    The exact fallback uses phi.weighted_sum and respects its configured
    backward backend. For an approximate mode, gradients flow through weights,
    means, learned sigma, and globally learned pixel edges.
    """
    if on_failure not in ('exact','raise'):
        raise ValueError('on_failure must be exact or raise')
    if not isinstance(initial_rank,int) or not 2<=initial_rank:
        raise ValueError('initial_rank must be an integer >=2')
    if not isinstance(max_rank,int) or max_rank<initial_rank:
        raise ValueError('max_rank must be >= initial_rank')
    if not isinstance(probe_points,int) or probe_points<8:
        raise ValueError('probe_points must be an integer >=8')
    if not (0<tolerance<1):
        raise ValueError('tolerance must be in (0,1)')
    if diagrams.ndim!=3 or diagrams.shape[-1]!=2 or weights.shape!=diagrams.shape[:2]:
        raise ValueError('expected diagrams [B,N,2] and weights [B,N]')
    if diagrams.shape[1]==0:
        output=phi.weighted_sum(diagrams,weights)
        decision=ApproximationDecision('exact-empty',None,None,None,None)
        return (output,decision) if return_decision else output
    b=diagrams[...,0]
    p=diagrams[...,1]-b
    bounds=phi.image_bnds
    in_bounds=(torch.isfinite(diagrams).all() and torch.isfinite(weights).all()
               and torch.all((b>=bounds[0][0]) & (b<=bounds[0][1]))
               and torch.all((p>=bounds[1][0]) & (p<=bounds[1][1])))
    if not bool(in_bounds):
        if on_failure=='raise':
            raise ValueError('nonfinite or out-of-bounds inputs cannot be approximated')
        output=phi.weighted_sum(diagrams,weights)
        decision=ApproximationDecision('exact-fallback',None,None,None,None)
        return (output,decision) if return_decision else output
    selected=[]
    errors=[]
    # Probes intentionally detached, while the actual approximate computation
    # remains fully differentiable.
    with torch.no_grad():
        for axis in (0,1):
            rank=initial_rank
            best_error=float('inf')
            best_rank=None
            while True:
                error=_axis_error(phi,axis,rank,probe_points,diagrams.device,diagrams.dtype)
                best_error=min(best_error,error)
                if error<=tolerance:
                    best_rank=rank
                    break
                if rank==max_rank:
                    break
                rank=min(2*rank,max_rank)
            selected.append(best_rank)
            errors.append(best_error)
    if any(rank is None for rank in selected):
        if on_failure=='raise':
            raise ValueError(f'Chebyshev probe tolerance not met: errors={errors}')
        output=phi.weighted_sum(diagrams,weights)
        decision=ApproximationDecision('exact-fallback',None,None,*errors)
    else:
        output=cheb_weighted_sum(phi,diagrams,weights,*selected,
                                 high_precision_basis=True)
        decision=ApproximationDecision('approximate',*selected,*errors)
    return (output,decision) if return_decision else output
