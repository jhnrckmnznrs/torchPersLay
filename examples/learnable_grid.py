"""Minimal demonstration that an integrated persistence-image edge can learn."""

import torch

from perslay import LearnableGridIntegratedGaussianPerslayPhi


torch.manual_seed(0)
phi = LearnableGridIntegratedGaussianPerslayPhi(
    image_size=[2, 1],
    image_bnds=[[0.0, 1.0], [0.0, 1.0]],
    sigma_x=0.08,
    sigma_y=0.15,
    normalize=True,
    learnable_sigma=False,
    min_cell_fraction=0.0,
)

left = torch.tensor([[[0.2, 0.7]]])
right = torch.tensor([[[0.4, 0.9]]])
weights = torch.ones((1, 1))
optimizer = torch.optim.Adam([phi.x_width_logits], lr=0.08)

print("initial x edges:", phi.grid_edges()[0].detach().tolist())
for _ in range(180):
    optimizer.zero_grad()
    left_features = phi.weighted_sum(left, weights)
    right_features = phi.weighted_sum(right, weights)
    loss = -torch.square(left_features - right_features).sum()
    loss.backward()
    optimizer.step()

print("learned x edges:", phi.grid_edges()[0].detach().tolist())
