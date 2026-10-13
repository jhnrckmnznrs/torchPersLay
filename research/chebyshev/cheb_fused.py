"""EXPERIMENTAL Chebyshev basis contraction for PI.
Approximate; only guarantees interpolation inside the specified image bounds.
"""
import math
import torch
from perslay.layers import IntegratedGaussianPerslayPhi


def _nodes_weights(bounds,num,dtype,device):
    i=torch.arange(num,dtype=dtype,device=device)
    theta=math.pi*(i+.5)/num
    x=(bounds[1]+bounds[0])*.5+(bounds[1]-bounds[0])*.5*torch.cos(theta)
    # barycentric interpolation weights, first-kind Chebyshev roots
    weights=torch.where((i.long()%2)==0,torch.ones_like(i),-torch.ones_like(i))*torch.sin(theta)
    return x,weights


def _basis(values,nodes,weights):
    diffs=values.unsqueeze(-1)-nodes
    # replace exact zeros for stable barycentric interpolation
    close=(diffs.abs()<=1e-14) # float32 equality optional
    safe=torch.where(close,torch.ones_like(diffs),diffs)
    terms=weights/safe
    result=terms/terms.sum(dim=-1,keepdim=True)
    correct=close.any(dim=-1,keepdim=True)
    return torch.where(correct,close.to(result.dtype),result)


def cheb_weighted_sum(phi,diagrams,weights,rx=24,ry=24,high_precision_basis=False):
    b=diagrams[...,0]
    p=diagrams[...,1]-b
    if not torch.all((b>=phi.image_bnds[0][0]) & (b<=phi.image_bnds[0][1])):
        raise ValueError('all birth coordinates must lie inside the image bounds')
    if not torch.all((p>=phi.image_bnds[1][0]) & (p<=phi.image_bnds[1][1])):
        raise ValueError('all persistence coordinates must lie inside the image bounds')
    sx,sy=phi._sigmas()
    if hasattr(phi,'grid_edges'):xe,ye=phi.grid_edges()
    else:xe,ye=phi.x_edges,phi.y_edges
    xn,wx=_nodes_weights(phi.image_bnds[0],rx,b.dtype,b.device)
    yn,wy=_nodes_weights(phi.image_bnds[1],ry,b.dtype,b.device)
    AX=IntegratedGaussianPerslayPhi._normal_interval_masses(xe,xn,sx) # [rx,nx]
    AY=IntegratedGaussianPerslayPhi._normal_interval_masses(ye,yn,sy) # [ry,ny]
    if high_precision_basis:
        # Barycentric rational weights are sensitive near interpolation nodes.
        xn64,wx64=_nodes_weights(phi.image_bnds[0],rx,torch.float64,b.device)
        yn64,wy64=_nodes_weights(phi.image_bnds[1],ry,torch.float64,b.device)
        BX=_basis(b.double(),xn64,wx64).to(b.dtype)
        BY=_basis(p.double(),yn64,wy64).to(p.dtype)
    else:
        BX=_basis(b,xn,wx)
        BY=_basis(p,yn,wy)
    moments=torch.bmm(BY.transpose(1,2),BX*weights.unsqueeze(-1))
    out=AY.T @ moments @ AX
    if not phi.normalize:out=out*(2*math.pi*sx*sy)
    return phi._format_aggregated_output(out)
