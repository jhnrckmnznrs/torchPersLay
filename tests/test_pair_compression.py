"""Exact sparse coordinate pairs, adaptive dispatch and VJP regression tests."""
import copy

import pytest
import torch

from perslay import (
    CoordinatePairPlan, IntegratedGaussianPerslayPhi,
    LearnableGridIntegratedGaussianPerslayPhi,
    PerslayRegressor, prepare_coordinate_pairs,
)


def repeated_pairs(B=3, N=300, levels=48, dtype=torch.float64):
    ids = torch.randint(levels, (B, N))
    birth = ids.to(dtype) / 2048
    persistence = (ids.to(dtype) + 2) / 2048
    return torch.stack((birth, birth + persistence), dim=-1)


@pytest.mark.parametrize('grid',[False,True])
@pytest.mark.parametrize('normalize',[False,True])
@pytest.mark.parametrize('dtype',[torch.float32, torch.float64])
def test_exact_output_and_gradients(grid, normalize, dtype):
    torch.manual_seed(31)
    cls = LearnableGridIntegratedGaussianPerslayPhi if grid else IntegratedGaussianPerslayPhi
    ref = cls(image_size=[11,9],image_bnds=[[0,1],[0,1]],sigma_x=.04,
              sigma_y=.06,normalize=normalize,aggregation_backend='fused',
              backward_backend='autograd').to(dtype)
    sparse = copy.deepcopy(ref);sparse.aggregation_backend='pairs'
    d = repeated_pairs(dtype=dtype)
    plan = prepare_coordinate_pairs(d)
    assert isinstance(plan, CoordinatePairPlan)
    assert plan.pair_count == 48
    wa = torch.randn(*d.shape[:2],dtype=dtype,requires_grad=True)
    wb = wa.detach().clone().requires_grad_()
    a = ref.weighted_sum(d,wa)
    b = sparse.weighted_sum(d,wb,histogram_plan=plan)
    tol = 7e-5 if dtype == torch.float32 else 3e-10
    torch.testing.assert_close(a,b,atol=tol,rtol=tol)
    grads_a = torch.autograd.grad(a.square().sum(),[wa,*ref.parameters()])
    grads_b = torch.autograd.grad(b.square().sum(),[wb,*sparse.parameters()])
    for ga,gb in zip(grads_a,grads_b):
        torch.testing.assert_close(ga,gb,atol=tol,rtol=tol)


def test_sparse_mask_and_full_regressor():
    torch.manual_seed(32)
    a = PerslayRegressor(image_size=[9,10],image_bnds=[[0,1],[0,1]],
            sigma_x=.05,sigma_y=.08,feature_map_type='integrated_learnable_grid',
            aggregation_backend='fused',backward_backend='analytic').double()
    b = copy.deepcopy(a)
    d = repeated_pairs(B=2,N=200)
    m = torch.ones(d.shape[:2],dtype=torch.bool);m[:,::7] = False
    d[:,::7] = float('nan')
    plan = prepare_coordinate_pairs(d,mask=m)
    r1 = a(d,mask=m)
    r2 = b(d,mask=m,histogram_plan=plan)
    torch.testing.assert_close(r1,r2,atol=2e-10,rtol=2e-10)
    r1.square().sum().backward();r2.square().sum().backward()
    for pa,pb in zip(a.parameters(),b.parameters()):
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad,pb.grad,atol=2e-9,rtol=2e-9)


def test_pair_plan_stale_and_bounds():
    d = repeated_pairs(B=2,N=200)
    p = prepare_coordinate_pairs(d)
    with pytest.raises(ValueError,match='stale'):
        p.validate(d + .0001)
    with pytest.raises(ValueError,match='size limit'):
        prepare_coordinate_pairs(d,max_pairs=2)
    with pytest.raises(ValueError,match='size limit'):
        prepare_coordinate_pairs(d,max_bytes=10)
    with pytest.raises(ValueError,match='coordinates'):
        prepare_coordinate_pairs(d.requires_grad_())
    with pytest.raises(ValueError,match='boolean'):
        prepare_coordinate_pairs(d.detach(),mask=torch.ones(d.shape[:2]))
    with pytest.raises(ValueError,match='finite'):
        prepare_coordinate_pairs(torch.full_like(d.detach(),float('inf')))


def test_automated_pair_dispatch_and_continuous_fallback(monkeypatch):
    import perslay._histogram as H
    torch.manual_seed(33)
    d = repeated_pairs(B=3,N=2500,levels=512,dtype=torch.float32)
    phi = IntegratedGaussianPerslayPhi(image_size=[18,18],image_bnds=[[0,1],[0,1]],
                   sigma_x=.03,sigma_y=.04,aggregation_backend='auto').float()
    weights = torch.randn(d.shape[:2])
    events=[]
    orig=H.evaluate_coordinate_pairs
    def track(*args,**kwargs):
        events.append(True)
        return orig(*args,**kwargs)
    monkeypatch.setattr(H,'evaluate_coordinate_pairs',track)
    torch.testing.assert_close(phi.weighted_sum(d,weights),phi._weighted_sum_fused(d,weights),atol=1e-4,rtol=1e-4)
    assert events, 'Expected sparse plan in automatic mode for correlated coordinates'
    events.clear()
    cont=torch.rand(3,2500,2);cont[...,1]+=cont[...,0]
    torch.testing.assert_close(phi.weighted_sum(cont,weights),phi._weighted_sum_fused(cont,weights))
    assert not events
    cont.requires_grad_(True)
    r = phi.weighted_sum(cont,weights)
    assert torch.isfinite(torch.autograd.grad(r.sum(),cont)[0]).all()


def test_second_derivatives_and_changeable_weights():
    torch.manual_seed(34)
    phi = LearnableGridIntegratedGaussianPerslayPhi(image_size=[7,8],
                image_bnds=[[0,1],[0,1]],sigma_x=.04,sigma_y=.07,
                aggregation_backend='pairs').double()
    d = repeated_pairs(B=2,N=60)
    plan = prepare_coordinate_pairs(d)
    for factor in [1.,-2.]:
        w=(torch.randn(d.shape[:2],dtype=d.dtype)*factor).requires_grad_()
        out=phi.weighted_sum(d,w,histogram_plan=plan)
        g=torch.autograd.grad(out.square().sum(),phi.log_sigma,create_graph=True)[0]
        assert torch.isfinite(torch.autograd.grad(g.sum(),phi.log_sigma)[0]).all()


@pytest.mark.parametrize('grid',[False,True])
def test_unnormalized_analytic_gradients_after_recompute_elimination(grid):
    torch.manual_seed(35)
    cls = LearnableGridIntegratedGaussianPerslayPhi if grid else IntegratedGaussianPerslayPhi
    ref=cls(image_size=[13,11],image_bnds=[[0,1],[0,1]],sigma_x=.03,
           sigma_y=.08,normalize=False,backward_backend='autograd').double()
    ana=copy.deepcopy(ref);ana.backward_backend='analytic'
    d=torch.rand(2,97,2,dtype=torch.float64)
    d[...,1]+=d[...,0]
    w=torch.randn(2,97,dtype=torch.float64)
    wa=w.clone().requires_grad_();wb=w.clone().requires_grad_()
    ra=ref.weighted_sum(d,wa);rb=ana.weighted_sum(d,wb)
    torch.testing.assert_close(ra,rb,atol=1e-11,rtol=1e-11)
    grads_ref=torch.autograd.grad(ra.square().sum(),[wa,*ref.parameters()])
    grads_ana=torch.autograd.grad(rb.square().sum(),[wb,*ana.parameters()])
    for a,b in zip(grads_ref,grads_ana):torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)


def test_auto_dense_path_preserves_autograd():
    # Avoid accidentally placing a differentiable evaluation in a no_grad
    # grouping block; this is a regression specific to automatic dispatch.
    torch.manual_seed(36)
    d = repeated_pairs(B=2,N=1000,levels=12,dtype=torch.float64)
    phi=LearnableGridIntegratedGaussianPerslayPhi(image_size=[9,10],
        image_bnds=[[0,1],[0,1]],sigma_x=.08,sigma_y=.07,
        aggregation_backend='auto',backward_backend='autograd').double()
    w=torch.randn(2,1000,dtype=torch.float64,requires_grad=True)
    result=phi.weighted_sum(d,w)
    result.square().sum().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()
    assert phi.log_sigma.grad is not None and torch.isfinite(phi.log_sigma.grad).all()
    assert phi.x_width_logits.grad is not None and torch.isfinite(phi.x_width_logits.grad).all()


def test_sparse_pair_plan_not_required_for_correct_automatic_outputs():
    # Sparse grouping uses floating-point equality rather than rounding; it
    # must be bitwise value-preserving even with nearby distinct coordinates.
    d = repeated_pairs(B=2,N=1100,levels=100,dtype=torch.float64)
    d[0,1,0] = torch.nextafter(d[0,1,0],torch.tensor(float('inf')))
    d[0,1,1] = d[0,1,0] + .0009765625
    phi=IntegratedGaussianPerslayPhi(image_size=[20,21],image_bnds=[[0,1],[0,1]],
        sigma_x=.02,sigma_y=.025,aggregation_backend='auto').double()
    w=torch.randn(2,1100,dtype=torch.float64)
    torch.testing.assert_close(phi.weighted_sum(d,w),phi._weighted_sum_fused(d,w),
                               rtol=2e-10,atol=2e-10)
