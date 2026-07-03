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
    GridPerslayWeight,
    LearnablePowerPerslayWeight,
    MLPPerslayWeight,
    NormalizedLearnablePowerPerslayWeight,
    Perslay,
    PowerPerslayWeight,
)
from .models import FlattenRho, MultiPerslayRegressor, PerslayRegressor

try:
    __version__ = version("torchPersLay")
except Exception:  # pragma: no cover
    __version__ = "0.1.5"

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
    "LearnablePowerPerslayWeight",
    "NormalizedLearnablePowerPerslayWeight",
    "MLPPerslayWeight",
    "GridPerslayWeight",
    "GaussianMixturePerslayWeight",
    "GaussianPerslayPhi",
    "FlatPerslayPhi",
    "FlattenRho",
    "PerslayRegressor",
    "MultiPerslayRegressor",
]
