import torch
import torch.nn as nn

from .layers import (
    ConstantPerslayWeight,
    GaussianPerslayPhi,
    LearnablePowerPerslayWeight,
    MLPPerslayWeight,
    NormalizedLearnablePowerPerslayWeight,
    Perslay,
)


class FlattenRho(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class PerslayRegressor(nn.Module):
    def __init__(
        self,
        image_size,
        image_bnds,
        sigma_x,
        sigma_y,
        hidden_dim=64,
        weight_type="learnable_power",
        weight_hidden_dim=16,
    ):
        super().__init__()

        if weight_type == "constant":
            weight = ConstantPerslayWeight()

        elif weight_type == "learnable_power":
            weight = LearnablePowerPerslayWeight(
                init_scale=1.0,
                init_power=1.0,
            )

        elif weight_type == "mlp":
            weight = MLPPerslayWeight(
                image_bnds=image_bnds,
                hidden_dim=weight_hidden_dim,
            )

        elif weight_type == "normalized_learnable_power":
            weight = NormalizedLearnablePowerPerslayWeight(
                image_bnds=image_bnds,
                init_scale=1.0,
                init_power=1.0,
            )

        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")

        phi = GaussianPerslayPhi(
            image_size=image_size,
            image_bnds=image_bnds,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            normalize=False,
        )

        rho = FlattenRho()

        self.perslay = Perslay(
            weight=weight,
            phi=phi,
            perm_op=torch.sum,
            rho=rho,
        )

        feature_dim = image_size[0] * image_size[1]

        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, diagrams, mask=None):
        x = self.perslay(diagrams, mask=mask)
        y = self.regressor(x).squeeze(-1)
        return y


class MultiPerslayRegressor(nn.Module):
    """
    Multi-branch PersLay regressor.

    Example input:
        diagrams = {
            "h0": Tensor[B, N0, 2],
            "h2": Tensor[B, N2, 2],
        }

        mask = {
            "h0": Tensor[B, N0],
            "h2": Tensor[B, N2],
        }
    """

    def __init__(
        self,
        branch_configs,
        hidden_dim=64,
        weight_type="learnable_power",
        weight_hidden_dim=16,
        dropout=0.0,
    ):
        super().__init__()

        self.branch_names = list(branch_configs.keys())
        self.branches = nn.ModuleDict()

        total_feature_dim = 0

        for name, cfg in branch_configs.items():
            branch = PerslayRegressor(
                image_size=cfg["image_size"],
                image_bnds=cfg["image_bnds"],
                sigma_x=cfg["sigma_x"],
                sigma_y=cfg["sigma_y"],
                hidden_dim=hidden_dim,
                weight_type=weight_type,
                weight_hidden_dim=weight_hidden_dim,
            )

            # Use only the PersLay feature extractor from the branch.
            self.branches[name] = branch.perslay

            total_feature_dim += cfg["image_size"][0] * cfg["image_size"][1]

        self.regressor = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, diagrams, mask=None):
        features = []

        for name in self.branch_names:
            branch_mask = None
            if mask is not None:
                branch_mask = mask[name]

            x = self.branches[name](
                diagrams[name],
                mask=branch_mask,
            )

            features.append(x)

        x = torch.cat(features, dim=-1)
        y = self.regressor(x).squeeze(-1)

        return y
