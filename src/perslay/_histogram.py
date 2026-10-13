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


def try_histogram(phi, diagrams, weights, *, mode="auto", plan=None,
                  max_cells=131072, max_bytes=32*1024*1024):
    """Select exact coordinate grouping or request the fused fallback.

    Automatic dispatch sorts each coordinate axis only once, then chooses a
    dense histogram or sparse occupied-pair representation using cost bounds.
    """
    B, N = diagrams.shape[:2]
    if plan is not None:
        if isinstance(plan, CoordinatePairPlan):
            return evaluate_coordinate_pairs(phi, diagrams, weights, plan)
        if isinstance(plan, HistogramPlan):
            return evaluate_histogram(phi, diagrams, weights, plan)
        raise TypeError("Expected HistogramPlan or CoordinatePairPlan")
    if B == 0 or N == 0 or diagrams.requires_grad:
        return None
    if mode == "histogram":
        try:
            return evaluate_histogram(phi, diagrams, weights,
                prepare_histogram(diagrams, max_cells=max_cells, max_bytes=max_bytes))
        except ValueError as exc:
            if str(exc) == "Dense histogram exceeds size limit":
                return None
            raise
    if mode == "pairs":
        try:
            sparse = prepare_coordinate_pairs(diagrams, max_bytes=max_bytes)
        except ValueError as exc:
            if str(exc) == "Sparse coordinate pairs exceed size limit":
                return None
            raise
        if not _pair_runtime_safe(phi, B, N, sparse.pair_count,
                                  diagrams.element_size(), max_bytes, mode=mode):
            return None
        return evaluate_coordinate_pairs(phi, diagrams, weights, sparse)
    if mode != "auto" or B * N < 512 or _looks_continuous(diagrams):
        return None

    # One pair of axis sorts supports either exact aggregation layout.
    dense = None
    sparse = None
    with torch.no_grad():
        birth = diagrams[..., 0].reshape(-1)
        persistence = (diagrams[..., 1] - diagrams[..., 0]).reshape(-1)
        if not bool(torch.isfinite(diagrams).all()):
            raise ValueError("Active persistence points must be finite")
        ux, ix = torch.unique(birth, sorted=True, return_inverse=True)
        uy, iy = torch.unique(persistence, sorted=True, return_inverse=True)
        cells = ux.numel() * uy.numel()
        nx, ny = phi.image_size
        dense_ok = (cells <= max_cells and
                    B * cells * diagrams.element_size() <= max_bytes)
        if dense_ok:
            work = min(B * (ny * uy.numel() * ux.numel() + ny * ux.numel() * nx),
                       B * (uy.numel() * ux.numel() * nx + ny * uy.numel() * nx))
            if work * 2 <= B * N * nx * ny:
                bidx = torch.arange(B, device=diagrams.device).repeat_interleave(N)
                flat = bidx * cells + iy * ux.numel() + ix
                dense = HistogramPlan(diagrams.detach().clone(), ux, uy, flat, B, N)

        # The Cartesian product is expensive or sparsely occupied. Encode
        # exactly occupied coordinate pairs without another axis sort.
        if dense is None:
            packed = ix + ux.numel() * iy
            pairs, inverse = torch.unique(packed, sorted=True, return_inverse=True)
            K = pairs.numel()
            required_bytes = ((diagrams.numel() + 2 * K) * diagrams.element_size()
                              + (B * N + B * K) * 8)
            if K > 16384 or required_bytes > max_bytes:
                return None
            if not _pair_runtime_safe(phi, B, N, K, diagrams.element_size(),
                                      max_bytes, mode=mode):
                return None
            bidx = torch.arange(B, device=diagrams.device).repeat_interleave(N)
            flat = bidx * K + inverse
            sparse = CoordinatePairPlan(diagrams.detach().clone(), ux[pairs % ux.numel()],
                            uy[torch.div(pairs, ux.numel(), rounding_mode='floor')],
                            flat, B, N)
    if dense is not None:
        return evaluate_histogram(phi, diagrams, weights, dense)
    return evaluate_coordinate_pairs(phi, diagrams, weights, sparse)


@dataclass(frozen=True)
class CoordinatePairPlan:
    """Reusable exact grouping of identical (birth, persistence) coordinate pairs.

    No learned parameters are cached. Diagrams must remain fixed, although point
    weights, bandwidths and learned pixel edges remain fully differentiable.
    """
    coordinates: torch.Tensor
    births: torch.Tensor
    persistences: torch.Tensor
    flat_indices: torch.Tensor
    batch_size: int
    points_per_diagram: int

    @property
    def pair_count(self):
        return self.births.numel()

    def validate(self, diagrams):
        if diagrams.requires_grad:
            raise ValueError("Coordinate grouping does not differentiate input coordinates")
        if (diagrams.shape != self.coordinates.shape
                or diagrams.device != self.coordinates.device
                or diagrams.dtype != self.coordinates.dtype
                or not torch.equal(diagrams, self.coordinates)):
            raise ValueError("Coordinate-pair plan is stale: rebuild when diagrams or masks change")


def prepare_coordinate_pairs(diagrams, mask=None, *, max_pairs=16384,
                             max_bytes=32 * 1024 * 1024):
    """Group identical birth/persistence pairs without allocating a dense grid.

    The integer encoding is injective: (birth_id, persistence_id) is mapped to
    birth_id + unique_birth_count * persistence_id. Floating-point values are
    never rounded or quantized. A boolean mask can replace invalid padding.
    """
    if diagrams.ndim != 3 or diagrams.shape[-1] != 2:
        raise ValueError("diagrams must be [B,N,2]")
    if diagrams.requires_grad:
        raise ValueError("Coordinate grouping does not differentiate input coordinates")
    if not isinstance(max_pairs, int) or max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    if not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if mask is not None:
        if mask.dtype != torch.bool or mask.shape != diagrams.shape[:2]:
            raise ValueError("mask must be boolean [B,N]")
        diagrams = torch.where(mask.unsqueeze(-1), diagrams, 0.0)
    if not bool(torch.isfinite(diagrams).all()):
        raise ValueError("Active persistence points must be finite")
    B, N, _ = diagrams.shape
    if not B or not N:
        raise ValueError("Empty diagrams should use the fused backend")
    with torch.no_grad():
        birth = diagrams[..., 0].reshape(-1)
        persistence = (diagrams[..., 1] - diagrams[..., 0]).reshape(-1)
        ux, ix = torch.unique(birth, sorted=True, return_inverse=True)
        uy, iy = torch.unique(persistence, sorted=True, return_inverse=True)
        n_birth = ux.numel()
        # Integer pack avoids a slower two-dimensional floating-point sort.
        packed = ix + n_birth * iy
        pairs, inverse = torch.unique(packed, sorted=True, return_inverse=True)
        K = pairs.numel()
        # Include the snapshot, per-point indices, and grouped scalar weights.
        required_bytes = (diagrams.numel() + 2 * K) * diagrams.element_size() \
                         + (B * N + B * K) * 8
        if K > max_pairs or required_bytes > max_bytes:
            raise ValueError("Sparse coordinate pairs exceed size limit")
        bidx = torch.arange(B, device=diagrams.device).repeat_interleave(N)
        flat = bidx * K + inverse
        return CoordinatePairPlan(diagrams.detach().clone(), ux[pairs % n_birth],
                                  uy[torch.div(pairs, n_birth, rounding_mode='floor')],
                                  flat, B, N)


def evaluate_coordinate_pairs(phi, diagrams, weights, plan):
    """Differentiable evaluation of an exact coordinate-pair plan."""
    plan.validate(diagrams)
    if weights.shape != diagrams.shape[:2]:
        raise ValueError("weights must have shape [B,N]")
    if weights.device != diagrams.device or weights.dtype != diagrams.dtype:
        raise ValueError("weights must share diagram dtype and device")
    B, K = plan.batch_size, plan.pair_count
    xe, ye = phi.grid_edges() if hasattr(phi, 'grid_edges') else (phi.x_edges, phi.y_edges)
    sx, sy = phi._sigmas()
    X = phi._normal_interval_masses(xe, plan.births.unsqueeze(0), sx).squeeze(0)
    Y = phi._normal_interval_masses(ye, plan.persistences.unsqueeze(0), sy).squeeze(0)
    H = weights.new_zeros(B * K).index_add(0, plan.flat_indices, weights.reshape(-1)).view(B, K)
    # Multiply point weights into the smaller axis factor, then use batched GEMM.
    nx, ny = phi.image_size
    if nx <= ny:
        values = torch.bmm(Y.T.expand(B, -1, -1), X.unsqueeze(0) * H.unsqueeze(-1))
    else:
        values = torch.bmm((Y.unsqueeze(0) * H.unsqueeze(-1)).transpose(1, 2),
                           X.unsqueeze(0).expand(B, -1, -1))
    if not phi.normalize:
        values = values * (2 * math.pi * sx * sy)
    return phi._format_aggregated_output(values)


def _pair_runtime_safe(phi, B, N, K, element_size, max_bytes, *, mode):
    nx, ny = phi.image_size
    # Upper bound for X, Y, weighted factors and H, not including output.
    factor_bytes = (K * (nx + ny) + B * K * (min(nx, ny) + 1)) * element_size
    if factor_bytes > max_bytes:
        return False
    # Avoid an expensive grouping path on nearly all-unique diagrams. The
    # grouping overhead is substantial, hence the conservative 4x threshold.
    if mode == 'auto':
        pair_work = B * K * nx * ny
        fused_work = B * N * nx * ny
        if pair_work * 4 >= fused_work or B * N < 512:
            return False
    return True
