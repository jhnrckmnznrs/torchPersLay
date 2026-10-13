"""First-order analytical VJP for exact integrated persistence-image aggregation.

The kernel returns the same mathematical features as the eager PyTorch backend,
with floating-point addition-order differences. It does not support higher-order
autograd or torch.func transforms.
"""
import math
import torch

from torch.autograd.function import once_differentiable


def axis_masses(edges,means,sigma):
    from .layers import IntegratedGaussianPerslayPhi
    return IntegratedGaussianPerslayPhi._normal_interval_masses(edges, means, sigma)

def axis_pdfs(edges,means,sigma):
    z=(edges-means.unsqueeze(-1))/sigma
    return torch.exp(-0.5*z.square())/(math.sqrt(2.0*math.pi)*sigma)

class AnalyticFusedIntegrated(torch.autograd.Function):
    @staticmethod
    def forward(ctx,b,p,w,sx,sy,xe,ye,chunk,normalize):
        B,N=b.shape
        nx,ny=xe.numel()-1,ye.numel()-1
        c=chunk or max(1, min(N, 2048))
        out=b.new_zeros((B,ny,nx))
        with torch.no_grad():
            for a in range(0,N,c):
                z=slice(a,min(a+c,N))
                xx=axis_masses(xe,b[:,z],sx)
                yy=axis_masses(ye,p[:,z],sy)
                out.add_(torch.bmm(yy.transpose(1,2),xx*w[:,z,None]))
            if not normalize:
                out.mul_(2*math.pi*sx*sy)
        ctx.save_for_backward(b,p,w,sx,sy,xe,ye)
        ctx.chunk=c
        ctx.normalize=normalize
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        b,p,w,sx,sy,xe,ye=ctx.saved_tensors
        B,N=b.shape
        c=ctx.chunk
        with torch.no_grad():
            dsx=sx.new_zeros(())
            dsy=sy.new_zeros(())
            db=torch.zeros_like(b)
            dp=torch.zeros_like(p)
            dw=torch.zeros_like(w)
            dxe=torch.zeros_like(xe)
            dye=torch.zeros_like(ye)
            if not ctx.normalize:
                gout=g*(2*math.pi*sx*sy)
            else:
                gout=g
            for a in range(0,N,c):
                sl=slice(a,min(a+c,N))
                bc=b[:,sl]
                pc=p[:,sl]
                wc=w[:,sl]
                X=axis_masses(xe,bc,sx)
                Y=axis_masses(ye,pc,sy)
                YG=torch.bmm(Y,gout)
                XGt=torch.bmm(X,gout.transpose(1,2))
                dw[:,sl]=(X*YG).sum(-1)
                dX=YG*wc.unsqueeze(-1)
                dY=XGt*wc.unsqueeze(-1)
                px=axis_pdfs(xe,bc,sx)
                py=axis_pdfs(ye,pc,sy)
                db[:,sl]=(dX*(px[...,:-1]-px[...,1:])).sum(-1)
                dp[:,sl]=(dY*(py[...,:-1]-py[...,1:])).sum(-1)
                dxlower=xe[:-1]-bc.unsqueeze(-1)
                dxupper=xe[1:]-bc.unsqueeze(-1)
                dylower=ye[:-1]-pc.unsqueeze(-1)
                dyupper=ye[1:]-pc.unsqueeze(-1)
                dsx+=((dX*(dxlower*px[...,:-1]-dxupper*px[...,1:])).sum()/sx)
                dsy+=((dY*(dylower*py[...,:-1]-dyupper*py[...,1:])).sum()/sy)
                dxe[0] -= (dX[...,0]*px[...,0]).sum()
                dxe[-1] += (dX[...,-1]*px[...,-1]).sum()
                dye[0] -= (dY[...,0]*py[...,0]).sum()
                dye[-1] += (dY[...,-1]*py[...,-1]).sum()
                if xe.numel()>2:
                    dxe[1:-1] += ((dX[...,:-1]-dX[...,1:])*px[...,1:-1]).sum(dim=(0,1))
                if ye.numel()>2:
                    dye[1:-1] += ((dY[...,:-1]-dY[...,1:])*py[...,1:-1]).sum(dim=(0,1))
            if not ctx.normalize:
                # The prefactor's contribution is recovered without another
                # complete Gaussian reduction: dw is already the gradient of
                # the weighted, unnormalized image w.r.t. the point weights.
                # Therefore (w * dw).sum() == <g, output>, up to summation
                # order, by linearity of the weighted image in w.
                prefactor_vjp=(w*dw).sum()
                dsx+=prefactor_vjp/sx
                dsy+=prefactor_vjp/sy
        return db,dp,dw,dsx,dsy,dxe,dye,None,None


def analytic_weighted_sum(phi,diagrams,weights,chunk_size=None):
    b=diagrams[...,0]
    p=diagrams[...,1]-b
    sx,sy=phi._sigmas()
    if hasattr(phi,'grid_edges'):
        xe,ye=phi.grid_edges()
    else:
        xe,ye=phi.x_edges,phi.y_edges
    out=AnalyticFusedIntegrated.apply(b,p,weights,sx,sy,xe,ye,chunk_size,phi.normalize)
    return phi._format_aggregated_output(out)
