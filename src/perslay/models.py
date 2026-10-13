import torch
import torch.nn as nn

from .layers import (
    ConstantPerslayWeight,
    GaussianPerslayPhi,
    GridSampledGaussianPerslayPhi,
    IntegratedGaussianPerslayPhi,
    LearnableGridIntegratedGaussianPerslayPhi,
    LearnablePowerPerslayWeight,
    LinearRampPerslayWeight,
    MLPPerslayWeight,
    NormalizedLearnablePowerPerslayWeight,
    PersistencePerslayWeight,
    Perslay,
)


class FlattenRho(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


def make_perslay_weight(
    weight_type,
    image_bnds,
    weight_hidden_dim=16,
    weight_threshold=1.0,
):
    if weight_type == "constant":
        return ConstantPerslayWeight()
    if weight_type == "persistence":
        return PersistencePerslayWeight()
    if weight_type == "linear_ramp":
        return LinearRampPerslayWeight(threshold=weight_threshold)
    if weight_type == "learnable_power":
        return LearnablePowerPerslayWeight(init_scale=1.0, init_power=1.0)
    if weight_type == "mlp":
        return MLPPerslayWeight(
            image_bnds=image_bnds,
            hidden_dim=weight_hidden_dim,
        )
    if weight_type == "normalized_learnable_power":
        return NormalizedLearnablePowerPerslayWeight(
            image_bnds=image_bnds,
            init_scale=1.0,
            init_power=1.0,
        )
    raise ValueError(f"Unknown weight_type: {weight_type}")


def make_perslay_phi(
    feature_map_type,
    image_size,
    image_bnds,
    sigma_x,
    sigma_y,
    normalize,
    learnable_sigma,
    flatten_order,
    chunk_size=None,
    min_cell_fraction=0.05,
    grid_regularization_strength=0.0,
    checkpoint_chunks=False,
    backward_backend="autograd",
):
    common = {
        "image_size": image_size,
        "image_bnds": image_bnds,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "normalize": normalize,
    }

    if feature_map_type == "legacy_gaussian":
        if backward_backend != "autograd":
            raise ValueError("analytical backward supports integrated Gaussian maps only")
        if not learnable_sigma:
            raise ValueError(
                "legacy_gaussian always has learnable bandwidths; use "
                "grid_sampled or integrated for fixed bandwidths"
            )
        return GaussianPerslayPhi(**common)
    if feature_map_type == "grid_sampled":
        if backward_backend != "autograd":
            raise ValueError("analytical backward supports integrated Gaussian maps only")
        return GridSampledGaussianPerslayPhi(
            **common,
            learnable_sigma=learnable_sigma,
            flatten_order=flatten_order,
            chunk_size=chunk_size,
            checkpoint_chunks=checkpoint_chunks,
        )
    if feature_map_type == "integrated":
        return IntegratedGaussianPerslayPhi(
            **common,
            learnable_sigma=learnable_sigma,
            flatten_order=flatten_order,
            chunk_size=chunk_size,
            checkpoint_chunks=checkpoint_chunks,
            backward_backend=backward_backend,
        )
    if feature_map_type == "integrated_learnable_grid":
        return LearnableGridIntegratedGaussianPerslayPhi(
            **common,
            learnable_sigma=learnable_sigma,
            flatten_order=flatten_order,
            chunk_size=chunk_size,
            checkpoint_chunks=checkpoint_chunks,
            backward_backend=backward_backend,
            min_cell_fraction=min_cell_fraction,
            grid_regularization_strength=grid_regularization_strength,
        )
    raise ValueError(
        "Unknown feature_map_type: "
        f"{feature_map_type}. Expected legacy_gaussian, grid_sampled, integrated, "
        "or integrated_learnable_grid."
    )


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
        weight_threshold=1.0,
        feature_map_type="legacy_gaussian",
        normalize=False,
        learnable_sigma=True,
        flatten_order="legacy-x-major",
        chunk_size=None,
        min_cell_fraction=0.05,
        grid_regularization_strength=0.0,
        checkpoint_chunks=False,
        backward_backend="autograd",
    ):
        super().__init__()
        weight = make_perslay_weight(
            weight_type=weight_type,
            image_bnds=image_bnds,
            weight_hidden_dim=weight_hidden_dim,
            weight_threshold=weight_threshold,
        )
        phi = make_perslay_phi(
            feature_map_type=feature_map_type,
            image_size=image_size,
            image_bnds=image_bnds,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            normalize=normalize,
            learnable_sigma=learnable_sigma,
            flatten_order=flatten_order,
            chunk_size=chunk_size,
            checkpoint_chunks=checkpoint_chunks,
            backward_backend=backward_backend,
            min_cell_fraction=min_cell_fraction,
            grid_regularization_strength=grid_regularization_strength,
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

    def regularization_loss(self):
        regularizer = getattr(self.perslay.phi, "regularization_loss", None)
        if regularizer is None:
            return next(self.parameters()).new_zeros(())
        return regularizer()


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
        weight_threshold=1.0,
        feature_map_type="legacy_gaussian",
        normalize=False,
        learnable_sigma=True,
        flatten_order="legacy-x-major",
        chunk_size=None,
        min_cell_fraction=0.05,
        grid_regularization_strength=0.0,
        dropout=0.0,
        checkpoint_chunks=False,
        backward_backend="autograd",
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
                weight_threshold=cfg.get("weight_threshold", weight_threshold),
                feature_map_type=cfg.get("feature_map_type", feature_map_type),
                normalize=cfg.get("normalize", normalize),
                learnable_sigma=cfg.get("learnable_sigma", learnable_sigma),
                flatten_order=cfg.get("flatten_order", flatten_order),
                chunk_size=cfg.get("chunk_size", chunk_size),
                checkpoint_chunks=cfg.get("checkpoint_chunks", checkpoint_chunks),
                backward_backend=cfg.get("backward_backend", backward_backend),
                min_cell_fraction=cfg.get(
                    "min_cell_fraction", min_cell_fraction
                ),
                grid_regularization_strength=cfg.get(
                    "grid_regularization_strength", grid_regularization_strength
                ),
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

    def regularization_loss(self):
        losses = []
        for branch in self.branches.values():
            regularizer = getattr(branch.phi, "regularization_loss", None)
            if regularizer is not None:
                losses.append(regularizer())
        if losses:
            return torch.stack(losses).sum()
        return next(self.parameters()).new_zeros(())
