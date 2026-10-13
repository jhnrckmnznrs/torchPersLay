"""Small optional CPU graph-compile smoke test for the analytical backend."""
import pytest
import torch
from perslay.models import PerslayRegressor


@pytest.mark.parametrize('backend',['eager'])
def test_torch_compile_fullgraph(backend):
    if not hasattr(torch,'compile'):
        pytest.skip('torch.compile not available')
    model=PerslayRegressor(image_size=[4,3],image_bnds=[[0,1],[0,1]],
           sigma_x=.15,sigma_y=.13,feature_map_type='integrated_learnable_grid',
           weight_type='learnable_power',normalize=True,backward_backend='analytic',
           chunk_size=5,hidden_dim=4)
    x=torch.rand(2,12,2);x[...,1]+=x[...,0]
    try:
        compiled=torch.compile(model,backend=backend,fullgraph=True)
        loss=compiled(x).square().sum()
        loss.backward()
    except Exception as exc:
        pytest.fail(f'fullgraph={True} backend={backend} failed: {exc}')
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
