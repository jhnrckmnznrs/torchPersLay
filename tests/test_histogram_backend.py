"""Correctness and safe dispatch for exact histogram factorization."""
import copy
import pytest
import torch
from perslay import (prepare_histogram, PerslayRegressor, MultiPerslayRegressor,
                     IntegratedGaussianPerslayPhi, LearnableGridIntegratedGaussianPerslayPhi)


def quantized(B=3,N=300,levels=16,dtype=torch.float64):
    b=torch.randint(levels,(B,N)).to(dtype)/128
    p=(torch.randint(levels,(B,N)).to(dtype)+1)/128
    return torch.stack([b,b+p],-1)


@pytest.mark.parametrize("learnable",[False,True])
@pytest.mark.parametrize("normalized",[False,True])
@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_exact_value_and_all_gradients(learnable,normalized,dtype):
    torch.manual_seed(123)
    cls=LearnableGridIntegratedGaussianPerslayPhi if learnable else IntegratedGaussianPerslayPhi
    base=cls(image_size=[12,15],image_bnds=[[0,1],[0,1]],sigma_x=.07,sigma_y=.11,
             normalize=normalized,aggregation_backend="fused",backward_backend="autograd").to(dtype)
    hist=copy.deepcopy(base);hist.aggregation_backend="histogram"
    d=quantized(dtype=dtype)
    w1=torch.randn(d.shape[:2],dtype=dtype,requires_grad=True)
    w2=w1.detach().clone().requires_grad_()
    r=base.weighted_sum(d,w1)
    h=hist.weighted_sum(d,w2)
    tol=4e-5 if dtype==torch.float32 else 2e-10
    torch.testing.assert_close(r,h,atol=tol,rtol=tol)
    loss1=r.square().sum();loss2=h.square().sum()
    g1=torch.autograd.grad(loss1,[w1,*base.parameters()])
    g2=torch.autograd.grad(loss2,[w2,*hist.parameters()])
    for a,b in zip(g1,g2): torch.testing.assert_close(a,b,atol=tol,rtol=tol)


def test_mask_and_invalid_padding_full_model():
    torch.manual_seed(4)
    b=PerslayRegressor(image_size=[10,13],image_bnds=[[0,1],[0,1]],sigma_x=.06,sigma_y=.09,
         feature_map_type="integrated_learnable_grid",backward_backend="analytic",
         aggregation_backend="auto",normalize=True).double()
    h=copy.deepcopy(b);h.perslay.phi.aggregation_backend="histogram"
    d=quantized(B=3,N=100,dtype=torch.float64)
    mask=torch.ones(d.shape[:2],dtype=torch.bool);mask[:,::5]=False
    d[:,::5]=float('nan')
    plan=prepare_histogram(d,mask)
    ref=b(d,mask=mask)
    result=h(d,mask=mask,histogram_plan=plan)
    torch.testing.assert_close(ref,result,rtol=1e-10,atol=1e-10)
    ref.sum().backward();result.sum().backward()
    for pa,pb in zip(b.parameters(),h.parameters()):
        if pa.grad is not None:torch.testing.assert_close(pa.grad,pb.grad,rtol=1e-9,atol=1e-9)
    assert torch.isfinite(result).all()


def test_plan_invalidation_and_cached_weight_updates():
    d=quantized(B=2,N=99)
    plan=prepare_histogram(d)
    phi=IntegratedGaussianPerslayPhi(image_size=[8,9],image_bnds=[[0,1],[0,1]],sigma_x=.08,
          aggregation_backend="fused").double()
    for factor in [1.,5.,-2.]:
        w=torch.randn(d.shape[:2],dtype=d.dtype,requires_grad=True)*factor
        result=phi.weighted_sum(d,w,histogram_plan=plan)
        expect=phi._weighted_sum_fused(d,w)
        torch.testing.assert_close(result,expect,atol=1e-10,rtol=1e-10)
    changed=d.clone();changed[0,0,0]+=.01
    with pytest.raises(ValueError,match="stale"):
        phi.weighted_sum(changed,torch.ones(changed.shape[:2],dtype=d.dtype),histogram_plan=plan)


def test_fallback_continuous_or_coordinate_grad():
    torch.manual_seed(6)
    d=torch.rand(2,2048,2);d[...,1]+=d[...,0]
    phi=IntegratedGaussianPerslayPhi(image_size=[12,13],image_bnds=[[0,2],[0,1]],sigma_x=.2,
       aggregation_backend="auto",backward_backend="analytic")
    w=torch.randn(d.shape[:2],requires_grad=True)
    torch.testing.assert_close(phi.weighted_sum(d,w),phi._weighted_sum_fused(d,w))
    d.requires_grad_()
    torch.testing.assert_close(phi.weighted_sum(d,w),phi._weighted_sum_fused(d,w))
    assert torch.isfinite(torch.autograd.grad(phi.weighted_sum(d,w).sum(),d)[0]).all()


def test_limit_empty_and_mask_errors():
    d=quantized(B=2,N=40)
    with pytest.raises(ValueError,match="limit"):
        prepare_histogram(d,max_bytes=1)
    with pytest.raises(ValueError,match="limit"):
        prepare_histogram(d,max_cells=2)
    with pytest.raises(ValueError,match="boolean"):
        prepare_histogram(d,mask=torch.ones(d.shape[:2]))
    phi=IntegratedGaussianPerslayPhi(image_size=[9,8],image_bnds=[[0,1],[0,1]],sigma_x=.05,
        aggregation_backend="auto")
    assert phi.weighted_sum(d[:,:0],torch.empty(2,0,dtype=d.dtype)).shape==(2,8,9,1)


def test_double_backward_parameters():
    torch.manual_seed(10)
    phi=LearnableGridIntegratedGaussianPerslayPhi(image_size=[7,9],image_bnds=[[0,1],[0,1]],
             sigma_x=.07,sigma_y=.06,backward_backend="autograd",aggregation_backend="histogram").double()
    d=quantized(B=2,N=45)
    w=torch.randn(d.shape[:2],dtype=d.dtype,requires_grad=True)
    out=phi.weighted_sum(d,w)
    g=torch.autograd.grad(out.square().sum(),phi.log_sigma,create_graph=True)[0]
    gg=torch.autograd.grad(g.sum(),phi.log_sigma)[0]
    assert gg.isfinite().all()


def test_multibranch_caching_and_checkpoint_compatibility():
    config={k:dict(image_size=[6,6],image_bnds=[[0,1],[0,1]],sigma_x=.06,sigma_y=.07,
                      feature_map_type="integrated",aggregation_backend="histogram") for k in ['h0','h2']}
    model=MultiPerslayRegressor(branch_configs=config,feature_map_type="integrated")
    diagrams={k:quantized(2,50,dtype=torch.float32) for k in config}
    plans={k:prepare_histogram(diagrams[k]) for k in diagrams}
    output=model(diagrams,histogram_plans=plans)
    torch.testing.assert_close(output,model(diagrams),atol=1e-5,rtol=1e-5)
    state=model.state_dict()
    new=MultiPerslayRegressor(branch_configs=config,feature_map_type="integrated")
    new.load_state_dict(state,strict=True)
    torch.testing.assert_close(model(diagrams),new(diagrams))


def test_torch_compile_fallback():
    if not hasattr(torch,"compile"):pytest.skip("torch.compile unavailable")
    d=quantized(2,70,dtype=torch.float32)
    phi=IntegratedGaussianPerslayPhi(image_size=[6,7],image_bnds=[[0,1],[0,1]],sigma_x=.1,
       aggregation_backend="auto").float()
    # Eager exact grouping can have dynamic shapes; compiled model uses fused path.
    fun=torch.compile(phi.weighted_sum, backend="eager",fullgraph=True)
    w=torch.ones(d.shape[:2]);result=fun(d,w)
    torch.testing.assert_close(result,phi._weighted_sum_fused(d,w))
