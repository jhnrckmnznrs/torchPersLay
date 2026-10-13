"""PyTorch PersLay layers and model helpers."""

try:
    from importlib.metadata import version
except ImportError:  # pragma: no cover
    from importlib_metadata import version

from .data import (
    MultiPersistenceDiagramDataset,
    PersistenceDiagramDataset,
    collate_multi_persistence_diagrams,
    collate_persistence_diagrams,
    compute_diagram_bounds,
    compute_sigma_from_bounds,
)
from .layers import (
    ConstantPerslayWeight,
    FlatPerslayPhi,
    GaussianMixturePerslayWeight,
    GaussianPerslayPhi,
    GridSampledGaussianPerslayPhi,
    GridPerslayWeight,
    IntegratedGaussianPerslayPhi,
    LearnableGridIntegratedGaussianPerslayPhi,
    LearnablePowerPerslayWeight,
    LinearRampPerslayWeight,
    MLPPerslayWeight,
    NormalizedLearnablePowerPerslayWeight,
    Perslay,
    PersistencePerslayWeight,
    PowerPerslayWeight,
)
from .models import (
    FlattenRho,
    MultiPerslayRegressor,
    PerslayRegressor,
    make_perslay_phi,
    make_perslay_weight,
)

try:
    __version__ = version("torchPersLay")
except Exception:  # pragma: no cover
    __version__ = "0.4.1"

__all__ = [
    "__version__",
    "PersistenceDiagramDataset",
    "MultiPersistenceDiagramDataset",
    "collate_persistence_diagrams",
    "collate_multi_persistence_diagrams",
    "compute_diagram_bounds",
    "compute_sigma_from_bounds",
    "Perslay",
    "PowerPerslayWeight",
    "ConstantPerslayWeight",
    "PersistencePerslayWeight",
    "LinearRampPerslayWeight",
    "LearnablePowerPerslayWeight",
    "NormalizedLearnablePowerPerslayWeight",
    "MLPPerslayWeight",
    "GridPerslayWeight",
    "GaussianMixturePerslayWeight",
    "GaussianPerslayPhi",
    "GridSampledGaussianPerslayPhi",
    "IntegratedGaussianPerslayPhi",
    "LearnableGridIntegratedGaussianPerslayPhi",
    "FlatPerslayPhi",
    "FlattenRho",
    "PerslayRegressor",
    "MultiPerslayRegressor",
    "make_perslay_weight",
    "make_perslay_phi",
]
