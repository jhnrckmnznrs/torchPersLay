"""Release-candidate regression tests for the exact analytical backend."""
import copy
import io

import pytest
import torch

from perslay.layers import (
    ConstantPerslayWeight,
    IntegratedGaussianPerslayPhi,
    LearnableGridIntegratedGaussianPerslayPhi,
    Perslay,
)
from perslay.models import FlattenRho, MultiPerslayRegressor, PerslayRegressor
from perslay._analytic import AnalyticFusedIntegrated


@pytest.mark.parametrize('cls', [IntegratedGaussianPerslayPhi, LearnableGridIntegratedGaussianPerslayPhi])
@pytest.mark.parametrize('normalize', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('chunk', [None, 9])
def test_kernel_gradient_parity(cls, normalize, dtype, chunk):
    torch.manual_seed(81)
    params=dict(image_size=[7, 5], image_bnds=[[0, 1], [0, 1]],
                sigma_x=0.08, sigma_y=0.12, normalize=normalize,
                chunk_size=chunk)
    std=cls(**params, backward_backend='autograd').to(dtype)
    fast=cls(**params, backward_backend='analytic').to(dtype)
    fast.load_state_dict(std.state_dict(), strict=True)
    if isinstance(std, LearnableGridIntegratedGaussianPerslayPhi):
        with torch.no_grad():
            std.x_width_logits.copy_(torch.randn_like(std.x_width_logits)*0.1)
            fast.load_state_dict(std.state_dict())
    diag=torch.rand(2, 31, 2, dtype=dtype)
    diag[..., 1]+=diag[..., 0]
    weights=torch.randn(2,31,dtype=dtype)
    def run(phi):
        d=diag.detach().clone().requires_grad_()
        w=weights.detach().clone().requires_grad_()
        out=phi.weighted_sum(d,w)
        grads=torch.autograd.grad(out.square().sum(), (d,w,*phi.parameters()))
        return out, grads
    o1,g1=run(std);o2,g2=run(fast)
    torch.testing.assert_close(o2,o1,atol=1e-5,rtol=4e-5)
    for a,b in zip(g1,g2):
        torch.testing.assert_close(a,b,atol=2e-4 if dtype==torch.float32 else 1e-8,
                                   rtol=3e-3 if dtype==torch.float32 else 1e-6)


def test_analytic_gradcheck():
    phi=LearnableGridIntegratedGaussianPerslayPhi([3,4],[[0,1],[0,1]],0.2,0.3).double()
    b=torch.tensor([[0.2,0.51]],dtype=torch.double,requires_grad=True)
    p=torch.tensor([[0.33,0.66]],dtype=torch.double,requires_grad=True)
    w=torch.tensor([[1.,-0.4]],dtype=torch.double,requires_grad=True)
    sx,sy=phi._sigmas();xe,ye=phi.grid_edges()
    fn=lambda *xs:AnalyticFusedIntegrated.apply(*xs,1,True)
    assert torch.autograd.gradcheck(fn,(b,p,w,sx,sy,xe,ye), eps=1e-6,atol=1e-5,rtol=1e-3)


@pytest.mark.parametrize('cls', [IntegratedGaussianPerslayPhi, LearnableGridIntegratedGaussianPerslayPhi])
def test_masked_empty_and_nonfinite_slots(cls):
    params=dict(image_size=[6, 5], image_bnds=[[0,1],[0,1]],
                sigma_x=0.07,sigma_y=0.15,normalize=True)
    fast=Perslay(ConstantPerslayWeight(),cls(**params,backward_backend='analytic'),torch.sum,FlattenRho())
    std=Perslay(ConstantPerslayWeight(),cls(**params,backward_backend='autograd'),torch.sum,FlattenRho())
    std.load_state_dict(fast.state_dict())
    diagrams=torch.tensor([[[0.12,0.5],[float('inf'),float('nan')]],
                           [[float('nan'),float('inf')],[float('inf'),float('nan')]]])
    mask=torch.tensor([[True, False],[False, False]])
    a=fast(diagrams,mask);b=std(diagrams,mask)
    assert torch.isfinite(a).all()
    torch.testing.assert_close(a,b)
    assert torch.count_nonzero(a[1])==0
    empty=fast(torch.empty(2,0,2),torch.empty(2,0,dtype=torch.bool))
    assert empty.shape==(2,30)
    assert torch.count_nonzero(empty)==0


@pytest.mark.parametrize('map_type',['integrated','integrated_learnable_grid'])
def test_complete_model_training_and_old_checkpoint(map_type):
    options=dict(image_size=[8,6],image_bnds=[[0,1],[0,1]],sigma_x=.13,
                 sigma_y=.09,feature_map_type=map_type,weight_type='learnable_power',
                 normalize=True,chunk_size=7,hidden_dim=9)
    std=PerslayRegressor(**options,backward_backend='autograd')
    analytic=PerslayRegressor(**options,backward_backend='analytic')
    # The existing state_dict keys must be interchangeable.
    saved=io.BytesIO();torch.save(std.state_dict(),saved);saved.seek(0)
    analytic.load_state_dict(torch.load(saved,weights_only=True),strict=True)
    torch.manual_seed(9)
    d=torch.rand(3,17,2);d[...,1]+=d[...,0]
    mask=torch.tensor([[True]*17,[True]*12+[False]*5,[True]*8+[False]*9])
    d=torch.where(mask[...,None],d,float('nan'))
    d[~mask,0]=float('inf')
    for model in (std,analytic):
        model.zero_grad(set_to_none=True)
    outstd=std(d,mask);outfast=analytic(d,mask)
    torch.testing.assert_close(outfast,outstd,atol=1e-5,rtol=2e-4)
    outstd.square().sum().backward()
    outfast.square().sum().backward()
    for (name,p1),(_,p2) in zip(std.named_parameters(),analytic.named_parameters()):
        assert p1.grad is not None and p2.grad is not None, name
        torch.testing.assert_close(p2.grad,p1.grad,atol=1e-4,rtol=3e-3,msg=name)


def test_multi_branch_model():
    config={'h0':dict(image_size=[5,4],image_bnds=[[0,1],[0,1]],sigma_x=.1,sigma_y=.1),
            'h2':dict(image_size=[4,4],image_bnds=[[0,1],[0,1]],sigma_x=.2,sigma_y=.1,
                      feature_map_type='integrated_learnable_grid')}
    settings=dict(branch_configs=config,feature_map_type='integrated',hidden_dim=6,
                  weight_type='persistence',chunk_size=6,normalize=True)
    fast=MultiPerslayRegressor(**settings,backward_backend='analytic')
    std=MultiPerslayRegressor(**settings,backward_backend='autograd')
    std.load_state_dict(fast.state_dict())
    d={}
    for key,n in [('h0',13),('h2',19)]:
        x=torch.rand(2,n,2);x[...,1]+=x[...,0]
        d[key]=x
    p1=fast(d);p2=std(d)
    torch.testing.assert_close(p1,p2,atol=1e-5,rtol=1e-4)
    p1.square().sum().backward()
    assert all(p.grad is not None for p in fast.parameters())


def test_higher_order_rejected_and_autograd_fallback():
    p=IntegratedGaussianPerslayPhi([3,3],[[0,1],[0,1]],.2,backward_backend='analytic')
    d=torch.tensor([[[.1,.5]]],requires_grad=True)
    w=torch.ones(1,1)
    v=p.weighted_sum(d,w).sum()
    g=torch.autograd.grad(v,d,create_graph=True)[0]
    with pytest.raises(RuntimeError):
        torch.autograd.grad(g.sum(),d)
    default=IntegratedGaussianPerslayPhi([3,3],[[0,1],[0,1]],.2,backward_backend='autograd')
    v=default.weighted_sum(d,w).sum()
    g=torch.autograd.grad(v,d,create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(g.sum(),d)[0]).all()


def test_parameter_validation():
    kw=dict(image_size=[4,4],image_bnds=[[0,1],[0,1]],sigma_x=.2)
    with pytest.raises(ValueError,match='mutually exclusive'):
        IntegratedGaussianPerslayPhi(**kw,chunk_size=2,checkpoint_chunks=True,backward_backend='analytic')
    with pytest.raises(ValueError,match='backward_backend'):
        IntegratedGaussianPerslayPhi(**kw,backward_backend='other')


@pytest.mark.parametrize('sigma',[0.015, 0.025, 0.05, 0.2, 1.2])
@pytest.mark.parametrize('normalize',[False,True])
def test_sigma_extremes_finite_parity(sigma,normalize):
    """Check difficult, but representable Gaussian bandwidths in float64."""
    torch.manual_seed(32)
    base=dict(image_size=[12,10],image_bnds=[[0,1],[0,1]],
              sigma_x=sigma,sigma_y=sigma*1.1,normalize=normalize,
              chunk_size=13)
    p=IntegratedGaussianPerslayPhi(**base,backward_backend='autograd').double()
    a=IntegratedGaussianPerslayPhi(**base,backward_backend='analytic').double()
    a.load_state_dict(p.state_dict())
    d=torch.rand(2,61,2,dtype=torch.double);d[...,1]+=d[...,0]
    w=torch.randn(2,61,dtype=torch.double)
    def evaluate(layer):
        x=d.detach().requires_grad_(True)
        z=w.detach().requires_grad_(True)
        y=layer.weighted_sum(x,z)
        grad=torch.autograd.grad(y.square().sum(),[x,z,*layer.parameters()])
        return y,grad
    y,dy=evaluate(p);v,dv=evaluate(a)
    torch.testing.assert_close(y,v,rtol=1e-10,atol=1e-10)
    for q,r in zip(dy,dv):
        torch.testing.assert_close(q,r,rtol=1e-4,atol=1e-7)


def test_reject_analytical_backend_for_unsupported_feature_maps():
    from perslay.models import make_perslay_phi
    p=dict(image_size=[4,4], image_bnds=[[0,1],[0,1]],sigma_x=.2,
           sigma_y=.2,normalize=True,learnable_sigma=True,
           flatten_order='row-major',backward_backend='analytic')
    for typ in ['legacy_gaussian','grid_sampled']:
        with pytest.raises(ValueError,match='integrated'):
            make_perslay_phi(feature_map_type=typ,**p)
