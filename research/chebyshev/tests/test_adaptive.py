import torch
import pytest
from adaptive_chebyshev import controlled_chebyshev_sum
from perslay.layers import LearnableGridIntegratedGaussianPerslayPhi


def make(sigma):
    return LearnableGridIntegratedGaussianPerslayPhi([32,32], [[0,1],[0,1]],
                sigma_x=sigma,sigma_y=sigma,normalize=True)


def data():
    torch.manual_seed(3)
    x=torch.rand(2,200,2)
    x[...,1]+=x[...,0]
    return x,torch.randn(2,200)


def test_adaptive_broad_sigma_and_gradients():
    phi=make(.13)
    x,w=data();x.requires_grad_();w.requires_grad_()
    z,choice=controlled_chebyshev_sum(phi,x,w,tolerance=1e-5,return_decision=True)
    assert choice.backend=='approximate'
    ref=phi.weighted_sum(x,w)
    torch.testing.assert_close(z,ref,atol=2e-4,rtol=1e-4)
    z.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert phi.x_width_logits.grad is not None


def test_narrow_sigma_fallback():
    phi=make(.012)
    x,w=data()
    z,choice=controlled_chebyshev_sum(phi,x,w,max_rank=32,tolerance=1e-4,
                                       return_decision=True)
    assert choice.backend=='exact-fallback'
    torch.testing.assert_close(z,phi.weighted_sum(x,w))


def test_out_of_bounds_fallback_or_error():
    phi=make(.12)
    x,w=data()
    x[0,0,0]=5.
    z,choice=controlled_chebyshev_sum(phi,x,w,return_decision=True)
    assert choice.backend=='exact-fallback'
    with pytest.raises(ValueError,match='out-of-bounds'):
        controlled_chebyshev_sum(phi,x,w,on_failure='raise')


def test_dynamic_sigma_revalidates():
    phi=make(.12)
    x,w=data()
    _,good=controlled_chebyshev_sum(phi,x,w,initial_rank=16,max_rank=32,
                     tolerance=1e-4,return_decision=True)
    with torch.no_grad():phi.log_sigma[:]=torch.log(torch.tensor(.012))
    _,bad=controlled_chebyshev_sum(phi,x,w,initial_rank=16,max_rank=32,
                     tolerance=1e-4,return_decision=True)
    assert good.backend=='approximate'
    assert bad.backend=='exact-fallback'


def test_rank_validation():
    phi=make(.1)
    x,w=data()
    with pytest.raises(ValueError,match='initial_rank'):
        controlled_chebyshev_sum(phi,x,w,initial_rank=1)
