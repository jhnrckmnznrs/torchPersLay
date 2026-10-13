"""Exact coordinate-histogram aggregation for fixed persistence diagrams.

The dense histogram is never constructed unless its size fits a configurable
memory budget. Plans are immutable snapshots of the input (no model parameters
are cached), and reuse validates the exact coordinates and mask.
"""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class HistogramPlan:
    coordinates: torch.Tensor
    birth_values: torch.Tensor
    persistence_values: torch.Tensor
    flat_indices: torch.Tensor
    batch_size: int
    points_per_diagram: int

    @property
    def cell_count(self):
        return self.birth_values.numel() * self.persistence_values.numel()

    def validate(self, diagrams):
        if diagrams.requires_grad:
            raise ValueError("Histogram grouping does not differentiate coordinates")
        if (diagrams.shape != self.coordinates.shape
                or diagrams.device != self.coordinates.device
                or diagrams.dtype != self.coordinates.dtype
                or not torch.equal(diagrams, self.coordinates)):
            raise ValueError("Histogram plan is stale: rebuild after changing diagrams or masks")


def prepare_histogram(diagrams, mask=None, *, max_cells=131072,
                      max_bytes=32*1024*1024):
    """Prepare a reusable plan for fixed diagrams (birth, death).

    If padding is present, provide its boolean mask. Masked coordinates are
    standardized to zero; no invalid values are allowed in active coordinates.
    """
    if diagrams.ndim != 3 or diagrams.shape[-1] != 2:
        raise ValueError("diagrams must be [B,N,2]")
    if diagrams.requires_grad:
        raise ValueError("Histogram grouping does not differentiate coordinates")
    if not isinstance(max_cells, int) or max_cells < 1:
        raise ValueError("max_cells must be positive")
    if not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if mask is not None:
        if mask.dtype != torch.bool or mask.shape != diagrams.shape[:2]:
            raise ValueError("mask must be boolean [B,N]")
        diagrams = torch.where(mask.unsqueeze(-1), diagrams, 0.0)
    if not bool(torch.isfinite(diagrams).all()):
        raise ValueError("Active persistence points must be finite")
    batch_size, n, _ = diagrams.shape
    if batch_size == 0 or n == 0:
        raise ValueError("Empty diagrams should use the fused backend")
    with torch.no_grad():
        birth = diagrams[..., 0].flatten()
        persistence = (diagrams[..., 1]-diagrams[..., 0]).flatten()
        ux,ix = torch.unique(birth, sorted=True, return_inverse=True)
        uy,iy = torch.unique(persistence, sorted=True, return_inverse=True)
        cells = ux.numel()*uy.numel()
        if cells > max_cells or batch_size*cells*diagrams.element_size()>max_bytes:
            raise ValueError("Dense histogram exceeds size limit")
        bidx = torch.arange(batch_size,device=diagrams.device).repeat_interleave(n)
        indices = bidx*cells + iy*ux.numel()+ix
        return HistogramPlan(diagrams.detach().clone(),ux,uy,indices,batch_size,n)


def evaluate_histogram(phi, diagrams, weights, plan):
    """Evaluate differentiable weights, sigma, and grid on cached coordinates."""
    plan.validate(diagrams)
    if weights.shape != diagrams.shape[:2]:
        raise ValueError("weights must have shape [B,N]")
    if weights.device != diagrams.device or weights.dtype != diagrams.dtype:
        raise ValueError("weights must share diagram dtype and device")
    ux,uy = plan.birth_values,plan.persistence_values
    B=plan.batch_size
    Xn,Yn=ux.numel(),uy.numel()
    H=weights.new_zeros(B*Xn*Yn).index_add(0,plan.flat_indices,weights.reshape(-1)).view(B,Yn,Xn)
    xe,ye = phi.grid_edges() if hasattr(phi,'grid_edges') else (phi.x_edges,phi.y_edges)
    sx,sy=phi._sigmas()
    X=phi._normal_interval_masses(xe,ux.unsqueeze(0),sx).squeeze(0)
    Y=phi._normal_interval_masses(ye,uy.unsqueeze(0),sy).squeeze(0)
    nx,ny=phi.image_size
    left_cost=B*(ny*Yn*Xn+ny*Xn*nx)
    right_cost=B*(Yn*Xn*nx+ny*Yn*nx)
    if left_cost<right_cost:
        values=torch.matmul(torch.matmul(Y.T,H),X)
    else:
        values=torch.matmul(Y.T,torch.matmul(H,X))
    if not phi.normalize:
        values=values*(2*math.pi*sx*sy)
    return phi._format_aggregated_output(values)


def _looks_continuous(diagrams):
    """Cheap, deterministic rejection before globally sorting all coordinates."""
    all_birth=diagrams[...,0].reshape(-1)
    all_p=(diagrams[...,1]-diagrams[...,0]).reshape(-1)
    # sample 1,024 positions spread across the entire batch
    stride=max(1,all_birth.numel()//1024)
    bs=all_birth[::stride][:1024]
    ps=all_p[::stride][:1024]
    return (torch.unique(bs).numel()>0.96*bs.numel()
            and torch.unique(ps).numel()>0.96*ps.numel())


def try_histogram(phi,diagrams,weights,*,mode="auto",plan=None,max_cells=131072,
                  max_bytes=32*1024*1024):
    """Return a histogram result or None to request the exact fused fallback."""
    B,N=diagrams.shape[:2]
    if B==0 or N==0 or diagrams.requires_grad:
        return None
    if plan is not None:
        return evaluate_histogram(phi,diagrams,weights,plan)
    if mode=="auto" and (B*N<512 or _looks_continuous(diagrams)):
        return None
    try:
        plan=prepare_histogram(diagrams,max_cells=max_cells,max_bytes=max_bytes)
    except ValueError as exc:
        if str(exc) in ("Dense histogram exceeds size limit", "Empty diagrams should use the fused backend"):
            return None
        raise
    if mode=="auto":
        ux,uy=plan.birth_values.numel(),plan.persistence_values.numel()
        nx,ny=phi.image_size
        cost=min(B*(ny*uy*ux+ny*ux*nx),B*(uy*ux*nx+ny*uy*nx))
        # Cheap enough to group only when it is clearly less expensive than
        # the full pointwise separable Gaussian reduction.
        if cost*2>B*N*nx*ny:
            return None
    return evaluate_histogram(phi,diagrams,weights,plan)
