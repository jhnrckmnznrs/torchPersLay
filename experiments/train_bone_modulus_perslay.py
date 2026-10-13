#!/usr/bin/env python3
"""Leakage-safe PersLay regression for apparent Young's modulus.

The experiment combines the baseline and uniformly thinned dual-lattice
structures.  Each structure contributes two related samples, but both variants
are assigned to the same train/validation/test split to prevent paired-structure
leakage.

Inputs expected under --data-root (default: data):

    fea_results_baseline_ds3.csv
    fea_results_uniform_ds3.csv
    diagrams/h0_baseline/structure_0000.csv ...
    diagrams/h2_baseline/structure_0000.csv ...
    diagrams/h0_uniform/structure_0000.csv ...
    diagrams/h2_uniform/structure_0000.csv ...

The persistence diagrams are compressed exactly by replacing duplicate
birth/death points with one point plus its multiplicity.  Because PersLay uses a
sum aggregation, this produces exactly the same embedding as expanding all
repeated points, while reducing these data from tens of thousands of rows to a
few hundred or thousand unique points per diagram.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCRIPT_VERSION = 1
CACHE_VERSION = 1
VALID_SPLITS = ("train", "validation", "test")
DEFAULT_VARIANTS = ("baseline", "uniform")
DEFAULT_DIMENSIONS = (0, 2)


class ExperimentError(RuntimeError):
    """Invalid input, dependency, cache, or training state."""


@dataclass(frozen=True)
class SampleRecord:
    index: int
    sample_id: str
    structure_id: str
    variant: str
    source_filename: str
    target: float
    bvtv: float
    diagram_paths: Mapping[str, Path]


@dataclass(frozen=True)
class TargetNormalization:
    mean: float
    std: float


@dataclass(frozen=True)
class Metrics:
    number_of_samples: int
    mae: float
    rmse: float
    r2: float


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def finite_float(value: Any, *, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ExperimentError(f"{context} must be numeric; found {value!r}.") from error
    if not math.isfinite(result):
        raise ExperimentError(f"{context} must be finite; found {value!r}.")
    return result


def structure_id_from_filename(filename: str) -> str:
    match = re.search(r"(structure_\d+)", filename, flags=re.IGNORECASE)
    if match is None:
        raise ExperimentError(
            f"Could not extract a structure_XXXX identifier from {filename!r}."
        )
    return match.group(1).lower()


def dimension_name(dimension: int) -> str:
    return f"h{dimension}"


def require_columns(
    fieldnames: Sequence[str] | None,
    required: Iterable[str],
    path: Path,
) -> None:
    if fieldnames is None:
        raise ExperimentError(f"CSV has no header: {path}")
    missing = [name for name in required if name not in fieldnames]
    if missing:
        raise ExperimentError(
            f"{path} is missing required column(s): {', '.join(missing)}."
        )


def read_records(
    data_root: Path,
    *,
    variants: Sequence[str],
    dimensions: Sequence[int],
    target_column: str,
    bvtv_column: str,
) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    seen_sample_ids: set[str] = set()

    for variant in variants:
        fea_path = data_root / f"fea_results_{variant}_ds3.csv"
        if not fea_path.is_file():
            raise ExperimentError(f"FEA result CSV does not exist: {fea_path}")

        with fea_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            require_columns(
                reader.fieldnames,
                ("relative_image", "status", "converged", target_column, bvtv_column),
                fea_path,
            )

            for row_number, row in enumerate(reader, start=2):
                status = str(row["status"]).strip().lower()
                converged = parse_bool(row["converged"])
                if status != "success" or not converged:
                    continue

                source_filename = str(row["relative_image"]).strip()
                structure_id = structure_id_from_filename(source_filename)
                sample_id = f"{variant}__{structure_id}"
                if sample_id in seen_sample_ids:
                    raise ExperimentError(
                        f"Duplicate sample {sample_id!r} in {fea_path}."
                    )
                seen_sample_ids.add(sample_id)

                diagram_paths: dict[str, Path] = {}
                for dimension in dimensions:
                    name = dimension_name(dimension)
                    diagram_path = (
                        data_root
                        / "diagrams"
                        / f"{name}_{variant}"
                        / f"{structure_id}.csv"
                    )
                    if not diagram_path.is_file():
                        raise ExperimentError(
                            f"Missing {name.upper()} diagram for {sample_id}: "
                            f"{diagram_path}"
                        )
                    diagram_paths[name] = diagram_path.resolve()

                target = finite_float(
                    row[target_column],
                    context=f"{fea_path}:{row_number} {target_column}",
                )
                bvtv = finite_float(
                    row[bvtv_column],
                    context=f"{fea_path}:{row_number} {bvtv_column}",
                )
                if target <= 0.0:
                    raise ExperimentError(
                        f"{fea_path}:{row_number} target must be positive."
                    )
                if bvtv <= 0.0:
                    raise ExperimentError(
                        f"{fea_path}:{row_number} BV/TV must be positive."
                    )

                records.append(
                    SampleRecord(
                        index=len(records),
                        sample_id=sample_id,
                        structure_id=structure_id,
                        variant=variant,
                        source_filename=source_filename,
                        target=target,
                        bvtv=bvtv,
                        diagram_paths=diagram_paths,
                    )
                )

    if not records:
        raise ExperimentError("No successful and converged FEA records were found.")

    expected_variants = set(variants)
    variants_by_structure: dict[str, set[str]] = {}
    for record in records:
        variants_by_structure.setdefault(record.structure_id, set()).add(record.variant)
    incomplete = {
        structure_id: sorted(expected_variants - present)
        for structure_id, present in variants_by_structure.items()
        if present != expected_variants
    }
    if incomplete:
        examples = list(incomplete.items())[:10]
        raise ExperimentError(
            "Each structure must have every selected variant. Missing examples: "
            f"{examples}"
        )

    return records


def split_structure_groups(
    records: Sequence[SampleRecord],
    *,
    train_fraction: float,
    validation_fraction: float,
    random_seed: int,
) -> dict[int, str]:
    if not (0.0 < train_fraction < 1.0):
        raise ExperimentError("--train-fraction must be between 0 and 1.")
    if not (0.0 < validation_fraction < 1.0):
        raise ExperimentError("--validation-fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1.0:
        raise ExperimentError(
            "--train-fraction + --validation-fraction must be less than 1."
        )

    structure_ids = sorted({record.structure_id for record in records})
    if len(structure_ids) < 3:
        raise ExperimentError("At least three structure groups are required.")

    rng = np.random.default_rng(random_seed)
    shuffled = list(structure_ids)
    rng.shuffle(shuffled)

    number_of_groups = len(shuffled)
    number_train = max(1, int(round(train_fraction * number_of_groups)))
    number_validation = max(
        1, int(round(validation_fraction * number_of_groups))
    )
    if number_train + number_validation >= number_of_groups:
        number_validation = max(1, number_of_groups - number_train - 1)
    number_test = number_of_groups - number_train - number_validation
    if number_test < 1:
        raise ExperimentError("Split fractions leave no test structure groups.")

    train_groups = set(shuffled[:number_train])
    validation_groups = set(
        shuffled[number_train : number_train + number_validation]
    )
    test_groups = set(shuffled[number_train + number_validation :])

    assignments: dict[int, str] = {}
    for record in records:
        if record.structure_id in train_groups:
            assignments[record.index] = "train"
        elif record.structure_id in validation_groups:
            assignments[record.index] = "validation"
        elif record.structure_id in test_groups:
            assignments[record.index] = "test"
        else:
            raise AssertionError("Unassigned structure group.")

    # Explicit leakage guard.
    splits_by_structure: dict[str, set[str]] = {}
    for record in records:
        splits_by_structure.setdefault(record.structure_id, set()).add(
            assignments[record.index]
        )
    leaked = {
        structure_id: sorted(splits)
        for structure_id, splits in splits_by_structure.items()
        if len(splits) != 1
    }
    if leaked:
        raise ExperimentError(f"Paired-structure split leakage detected: {leaked}")

    return assignments


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def cache_key(
    record: SampleRecord,
    *,
    minimum_persistence: float,
) -> str:
    payload = {
        "cache_version": CACHE_VERSION,
        "sample_id": record.sample_id,
        "minimum_persistence": float(minimum_persistence),
        "sources": {
            name: file_signature(path)
            for name, path in sorted(record.diagram_paths.items())
        },
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def read_and_compress_diagram(
    path: Path,
    *,
    minimum_persistence: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    try:
        values = np.genfromtxt(
            path,
            delimiter=",",
            names=True,
            dtype=np.float64,
            encoding="utf-8-sig",
        )
    except (OSError, ValueError) as error:
        raise ExperimentError(f"Could not read persistence diagram {path}: {error}") from error

    if values.dtype.names is None or not {"birth", "death"}.issubset(values.dtype.names):
        raise ExperimentError(f"Diagram must contain birth and death columns: {path}")

    birth = np.atleast_1d(values["birth"]).astype(np.float64, copy=False)
    death = np.atleast_1d(values["death"]).astype(np.float64, copy=False)
    points = np.column_stack((birth, death))

    finite = np.isfinite(points).all(axis=1)
    persistence = points[:, 1] - points[:, 0]
    keep = finite & (persistence > minimum_persistence)
    points = points[keep]
    original_count = int(points.shape[0])

    if original_count == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            0,
        )

    unique_points, counts = np.unique(points, axis=0, return_counts=True)
    return (
        np.ascontiguousarray(unique_points, dtype=np.float32),
        np.ascontiguousarray(counts, dtype=np.float32),
        original_count,
    )


def build_or_load_cache(
    records: Sequence[SampleRecord],
    *,
    cache_directory: Path,
    minimum_persistence: float,
    rebuild: bool,
) -> tuple[
    dict[int, dict[str, tuple[np.ndarray, np.ndarray]]],
    dict[str, dict[str, float]],
]:
    cache_directory.mkdir(parents=True, exist_ok=True)
    cached: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    totals: dict[str, list[tuple[int, int]]] = {}

    for position, record in enumerate(records, start=1):
        key = cache_key(record, minimum_persistence=minimum_persistence)
        cache_path = cache_directory / f"{record.sample_id}__{key}.npz"

        arrays: dict[str, np.ndarray] = {}
        original_counts: dict[str, int] = {}
        valid_cache = cache_path.is_file() and not rebuild

        if valid_cache:
            try:
                with np.load(cache_path, allow_pickle=False) as archive:
                    if int(archive["cache_version"]) != CACHE_VERSION:
                        valid_cache = False
                    else:
                        for name in record.diagram_paths:
                            arrays[f"{name}_points"] = np.ascontiguousarray(
                                archive[f"{name}_points"], dtype=np.float32
                            )
                            arrays[f"{name}_counts"] = np.ascontiguousarray(
                                archive[f"{name}_counts"], dtype=np.float32
                            )
                            original_counts[name] = int(archive[f"{name}_original_count"])
            except (OSError, KeyError, ValueError):
                valid_cache = False

        if not valid_cache:
            for name, path in record.diagram_paths.items():
                points, counts, original_count = read_and_compress_diagram(
                    path,
                    minimum_persistence=minimum_persistence,
                )
                arrays[f"{name}_points"] = points
                arrays[f"{name}_counts"] = counts
                original_counts[name] = original_count

            payload: dict[str, Any] = {
                "cache_version": np.asarray(CACHE_VERSION, dtype=np.int64)
            }
            for name in record.diagram_paths:
                payload[f"{name}_points"] = arrays[f"{name}_points"]
                payload[f"{name}_counts"] = arrays[f"{name}_counts"]
                payload[f"{name}_original_count"] = np.asarray(
                    original_counts[name], dtype=np.int64
                )
            np.savez_compressed(cache_path, **payload)

        sample_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in record.diagram_paths:
            points = arrays[f"{name}_points"]
            counts = arrays[f"{name}_counts"]
            if points.shape[0] != counts.shape[0] or points.shape[1:] != (2,):
                raise ExperimentError(f"Invalid compressed cache shape in {cache_path}.")
            sample_arrays[name] = (points, counts)
            totals.setdefault(name, []).append(
                (original_counts[name], int(points.shape[0]))
            )
        cached[record.index] = sample_arrays

        if position == 1 or position % 25 == 0 or position == len(records):
            print(f"Prepared diagram cache: {position}/{len(records)} samples")

    summary: dict[str, dict[str, float]] = {}
    for name, rows in totals.items():
        original = np.asarray([row[0] for row in rows], dtype=np.float64)
        unique = np.asarray([row[1] for row in rows], dtype=np.float64)
        summary[name] = {
            "mean_original_points": float(original.mean()),
            "mean_unique_points": float(unique.mean()),
            "maximum_unique_points": float(unique.max()),
            "compression_ratio": float(original.sum() / max(unique.sum(), 1.0)),
        }
    return cached, summary


def compute_training_bounds(
    cached: Mapping[int, Mapping[str, tuple[np.ndarray, np.ndarray]]],
    assignments: Mapping[int, str],
    branch_names: Sequence[str],
    *,
    padding_fraction: float,
) -> dict[str, list[list[float]]]:
    bounds: dict[str, list[list[float]]] = {}
    for name in branch_names:
        birth_min = math.inf
        birth_max = -math.inf
        persistence_min = math.inf
        persistence_max = -math.inf
        found = False

        for record_index, split in assignments.items():
            if split != "train":
                continue
            points = cached[record_index][name][0]
            if points.shape[0] == 0:
                continue
            persistence = points[:, 1] - points[:, 0]
            birth_min = min(birth_min, float(points[:, 0].min()))
            birth_max = max(birth_max, float(points[:, 0].max()))
            persistence_min = min(persistence_min, float(persistence.min()))
            persistence_max = max(persistence_max, float(persistence.max()))
            found = True

        if not found:
            raise ExperimentError(f"All training diagrams are empty for {name}.")

        birth_range = max(birth_max - birth_min, 1.0e-6)
        persistence_range = max(persistence_max - persistence_min, 1.0e-6)
        bounds[name] = [
            [
                birth_min - padding_fraction * birth_range,
                birth_max + padding_fraction * birth_range,
            ],
            [
                max(0.0, persistence_min - padding_fraction * persistence_range),
                persistence_max + padding_fraction * persistence_range,
            ],
        ]
    return bounds


def compute_training_mass_scales(
    cached: Mapping[int, Mapping[str, tuple[np.ndarray, np.ndarray]]],
    assignments: Mapping[int, str],
    branch_names: Sequence[str],
) -> dict[str, float]:
    """Return a fixed training-only point-mass scale for each branch.

    Dividing every branch embedding by one positive constant does not discard
    any information and can be absorbed by the first linear layer.  It prevents
    the very large diagram cardinalities in these voxel data from producing
    poorly scaled neural-network inputs.
    """
    scales: dict[str, float] = {}
    for name in branch_names:
        masses = [
            float(cached[index][name][1].sum())
            for index, split in assignments.items()
            if split == "train"
        ]
        if not masses or max(masses) <= 0.0:
            raise ExperimentError(f"Training diagrams have zero mass for {name}.")
        scales[name] = max(float(np.mean(masses)), 1.0)
    return scales


def import_dependencies() -> Any:
    try:
        import torch
        from perslay.models import PerslayRegressor
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as error:
        raise ExperimentError(
            "PyTorch and the local perslay package are required. From the "
            "repository root run: python -m pip install -e ."
        ) from error

    class Namespace:
        pass

    result = Namespace()
    result.torch = torch
    result.nn = nn
    result.DataLoader = DataLoader
    result.Dataset = Dataset
    result.PerslayRegressor = PerslayRegressor
    return result


def seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(requested: str, torch: Any) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ExperimentError("CUDA was requested but is not available.")
    return device


def make_dataset_class(dependencies: Any) -> type:
    torch = dependencies.torch
    Dataset = dependencies.Dataset

    class CompressedDiagramDataset(Dataset):
        def __init__(
            self,
            records: Sequence[SampleRecord],
            indices: Sequence[int],
            cached: Mapping[int, Mapping[str, tuple[np.ndarray, np.ndarray]]],
        ) -> None:
            self.records = records
            self.indices = list(indices)
            self.cached = cached

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, item: int) -> tuple[Any, Any, Any, Any, Any]:
            record_index = self.indices[item]
            record = self.records[record_index]
            diagrams = {
                name: torch.from_numpy(values[0])
                for name, values in self.cached[record_index].items()
            }
            multiplicities = {
                name: torch.from_numpy(values[1])
                for name, values in self.cached[record_index].items()
            }
            return (
                diagrams,
                multiplicities,
                torch.tensor(record.target, dtype=torch.float32),
                torch.tensor(record.bvtv, dtype=torch.float32),
                torch.tensor(record_index, dtype=torch.long),
            )

    return CompressedDiagramDataset


def collate_compressed(batch: Sequence[Any]) -> tuple[Any, Any, Any, Any, Any]:
    import torch
    from torch.nn.utils.rnn import pad_sequence

    diagram_rows, multiplicity_rows, targets, bvtvs, record_ids = zip(*batch)
    branch_names = tuple(diagram_rows[0].keys())
    padded_diagrams: dict[str, Any] = {}
    padded_multiplicities: dict[str, Any] = {}
    masks: dict[str, Any] = {}

    for name in branch_names:
        diagrams = [row[name] for row in diagram_rows]
        multiplicities = [row[name] for row in multiplicity_rows]
        lengths = torch.tensor([diagram.shape[0] for diagram in diagrams])

        if int(lengths.max().item()) == 0:
            padded_diagrams[name] = torch.zeros(
                len(diagrams), 1, 2, dtype=torch.float32
            )
            padded_multiplicities[name] = torch.zeros(
                len(diagrams), 1, dtype=torch.float32
            )
            masks[name] = torch.zeros(len(diagrams), 1, dtype=torch.bool)
        else:
            padded_diagrams[name] = pad_sequence(
                diagrams, batch_first=True, padding_value=0.0
            )
            padded_multiplicities[name] = pad_sequence(
                multiplicities, batch_first=True, padding_value=0.0
            )
            maximum = padded_diagrams[name].shape[1]
            masks[name] = (
                torch.arange(maximum).unsqueeze(0) < lengths.unsqueeze(1)
            )

    return (
        padded_diagrams,
        padded_multiplicities,
        masks,
        torch.stack(targets),
        torch.stack(bvtvs),
        torch.stack(record_ids),
    )


def sigma_from_bounds(
    bounds: Sequence[Sequence[float]],
    image_size: Sequence[int],
    *,
    multiplier: float,
) -> tuple[float, float]:
    return (
        multiplier * (float(bounds[0][1]) - float(bounds[0][0])) / int(image_size[0]),
        multiplier * (float(bounds[1][1]) - float(bounds[1][0])) / int(image_size[1]),
    )


def make_model(
    *,
    branch_names: Sequence[str],
    image_bounds: Mapping[str, Sequence[Sequence[float]]],
    image_size: Sequence[int],
    sigma_multiplier: float,
    branch_mass_scales: Mapping[str, float],
    hidden_dimension: int,
    dropout: float,
    weight_type: str,
    weight_hidden_dimension: int,
    dependencies: Any,
    feature_map_type: str = "legacy_gaussian",
    normalize_gaussian: bool = False,
    learnable_sigma: bool = True,
    flatten_order: str = "legacy-x-major",
) -> tuple[Any, dict[str, dict[str, Any]]]:
    torch = dependencies.torch
    nn = dependencies.nn
    branch_settings: dict[str, dict[str, Any]] = {}

    class ExactCompressedMultiPerslay(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            branches: dict[str, Any] = {}
            feature_dimension = 0
            for name in branch_names:
                sigma_x, sigma_y = sigma_from_bounds(
                    image_bounds[name],
                    image_size,
                    multiplier=sigma_multiplier,
                )
                wrapper = dependencies.PerslayRegressor(
                    image_size=list(image_size),
                    image_bnds=image_bounds[name],
                    sigma_x=sigma_x,
                    sigma_y=sigma_y,
                    hidden_dim=hidden_dimension,
                    weight_type=weight_type,
                    weight_hidden_dim=weight_hidden_dimension,
                    feature_map_type=feature_map_type,
                    normalize=normalize_gaussian,
                    learnable_sigma=learnable_sigma,
                    flatten_order=flatten_order,
                )
                branches[name] = wrapper.perslay
                branch_feature_dimension = int(image_size[0]) * int(image_size[1])
                feature_dimension += branch_feature_dimension
                branch_settings[name] = {
                    "image_size": [int(value) for value in image_size],
                    "image_bounds": [
                        [float(value) for value in image_bounds[name][0]],
                        [float(value) for value in image_bounds[name][1]],
                    ],
                    "initial_sigma_x": float(sigma_x),
                    "initial_sigma_y": float(sigma_y),
                    "feature_map_type": feature_map_type,
                    "normalize_gaussian": bool(normalize_gaussian),
                    "learnable_sigma": bool(learnable_sigma),
                    "flatten_order": flatten_order,
                    "feature_dimension": branch_feature_dimension,
                }
            self.branch_names = tuple(branch_names)
            self.branches = nn.ModuleDict(branches)
            for branch_name in self.branch_names:
                self.register_buffer(
                    f"mass_scale__{branch_name}",
                    torch.tensor(
                        float(branch_mass_scales[branch_name]),
                        dtype=torch.float32,
                    ),
                )
            self.regressor = nn.Sequential(
                nn.Linear(feature_dimension, hidden_dimension),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dimension, 1),
            )

        @staticmethod
        def compressed_branch_forward(
            branch: Any,
            diagrams: Any,
            multiplicities: Any,
            mask: Any,
            mass_scale: Any,
        ) -> Any:
            # This is algebraically identical to branch(diagrams, mask) on the
            # fully expanded diagram, because duplicate points have identical
            # point weights and identical phi values under sum aggregation.
            vector, _ = branch.phi(diagrams)
            point_weight = branch.weight(diagrams)
            point_weight = point_weight * (
                multiplicities.to(point_weight.dtype) / mass_scale
            )
            point_weight = point_weight * mask.to(point_weight.dtype)
            for _ in range(vector.ndim - point_weight.ndim):
                point_weight = point_weight.unsqueeze(-1)
            vector = torch.sum(vector * point_weight, dim=1)
            return branch.rho(vector)

        def forward(
            self,
            diagrams: Mapping[str, Any],
            multiplicities: Mapping[str, Any],
            masks: Mapping[str, Any],
        ) -> Any:
            features = [
                self.compressed_branch_forward(
                    self.branches[name],
                    diagrams[name],
                    multiplicities[name],
                    masks[name],
                    getattr(self, f"mass_scale__{name}"),
                )
                for name in self.branch_names
            ]
            return self.regressor(torch.cat(features, dim=1)).view(-1)

    return ExactCompressedMultiPerslay(), branch_settings


def move_mapping(values: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {
        name: tensor.to(device, non_blocking=True)
        for name, tensor in values.items()
    }


def target_signal(
    target: Any,
    bvtv: Any,
    *,
    target_mode: str,
    bvtv_intercept: float,
    bvtv_exponent: float,
    torch_module: Any | None = None,
) -> Any:
    if target_mode == "raw":
        return target
    if target_mode != "bvtv_residual":
        raise ExperimentError(f"Unknown target mode: {target_mode}")
    if torch_module is None:
        return np.log(target) - (bvtv_intercept + bvtv_exponent * np.log(bvtv))
    return torch_module.log(target) - (
        bvtv_intercept + bvtv_exponent * torch_module.log(bvtv)
    )


def reconstruct_target(
    signal: Any,
    bvtv: Any,
    *,
    target_mode: str,
    bvtv_intercept: float,
    bvtv_exponent: float,
    torch_module: Any,
) -> Any:
    if target_mode == "raw":
        return signal
    if target_mode != "bvtv_residual":
        raise ExperimentError(f"Unknown target mode: {target_mode}")
    return torch_module.exp(
        bvtv_intercept + bvtv_exponent * torch_module.log(bvtv) + signal
    )


def regression_metrics(rows: Sequence[tuple[int, float, float]]) -> Metrics:
    if not rows:
        raise ExperimentError("Cannot calculate metrics for an empty set.")
    truth = np.asarray([row[1] for row in rows], dtype=np.float64)
    prediction = np.asarray([row[2] for row in rows], dtype=np.float64)
    residual = prediction - truth
    denominator = float(np.sum(np.square(truth - truth.mean())))
    return Metrics(
        number_of_samples=len(rows),
        mae=float(np.mean(np.abs(residual))),
        rmse=float(np.sqrt(np.mean(np.square(residual)))),
        r2=(
            float(1.0 - np.sum(np.square(residual)) / denominator)
            if denominator > 0.0
            else float("nan")
        ),
    )


def evaluate(
    model: Any,
    loader: Any,
    *,
    records: Sequence[SampleRecord],
    normalization: TargetNormalization,
    target_mode: str,
    bvtv_intercept: float,
    bvtv_exponent: float,
    device: Any,
    torch: Any,
) -> list[tuple[int, float, float]]:
    model.eval()
    rows: list[tuple[int, float, float]] = []
    with torch.no_grad():
        for diagrams, multiplicities, masks, _, bvtvs, record_ids in loader:
            diagrams = move_mapping(diagrams, device)
            multiplicities = move_mapping(multiplicities, device)
            masks = move_mapping(masks, device)
            bvtvs = bvtvs.to(device, non_blocking=True)
            prediction_normalized = model(diagrams, multiplicities, masks)
            predicted_signal = (
                prediction_normalized * normalization.std + normalization.mean
            )
            predictions = reconstruct_target(
                predicted_signal,
                bvtvs,
                target_mode=target_mode,
                bvtv_intercept=bvtv_intercept,
                bvtv_exponent=bvtv_exponent,
                torch_module=torch,
            )
            for record_id, prediction in zip(
                record_ids.tolist(), predictions.cpu().tolist()
            ):
                index = int(record_id)
                rows.append((index, records[index].target, float(prediction)))
    return rows


def mean_baseline_metrics(
    records: Sequence[SampleRecord],
    indices: Sequence[int],
    *,
    training_mean: float,
) -> Metrics:
    return regression_metrics(
        [(index, records[index].target, training_mean) for index in indices]
    )


def fit_bvtv_power_law(
    records: Sequence[SampleRecord],
    indices: Sequence[int],
) -> tuple[float, float]:
    x = np.log(np.asarray([records[index].bvtv for index in indices]))
    y = np.log(np.asarray([records[index].target for index in indices]))
    design = np.column_stack((np.ones_like(x), x))
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def bvtv_power_law_metrics(
    records: Sequence[SampleRecord],
    indices: Sequence[int],
    *,
    intercept: float,
    exponent: float,
) -> Metrics:
    return regression_metrics(
        [
            (
                index,
                records[index].target,
                float(math.exp(intercept) * records[index].bvtv**exponent),
            )
            for index in indices
        ]
    )


def metrics_by_variant(
    rows: Sequence[tuple[int, float, float]],
    records: Sequence[SampleRecord],
) -> dict[str, Metrics]:
    variants = sorted({records[row[0]].variant for row in rows})
    return {
        variant: regression_metrics(
            [row for row in rows if records[row[0]].variant == variant]
        )
        for variant in variants
    }


def learned_parameters(model: Any, torch: Any) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    functional = torch.nn.functional
    for name, branch in model.branches.items():
        values: dict[str, float] = {}
        sigma = torch.exp(branch.phi.log_sigma).detach().cpu()
        values["sigma_x"] = float(sigma[0].item())
        values["sigma_y"] = float(sigma[1].item())
        weight = branch.weight
        if hasattr(weight, "raw_scale"):
            values["weight_scale"] = float(
                functional.softplus(weight.raw_scale).detach().cpu().item()
            )
        if hasattr(weight, "raw_power"):
            values["persistence_power"] = float(
                functional.softplus(weight.raw_power).detach().cpu().item()
            )
        result[name] = values
    return result


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metrics_to_json(metrics: Metrics) -> dict[str, Any]:
    result = asdict(metrics)
    if not math.isfinite(result["r2"]):
        result["r2"] = None
    return result


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train an exact duplicate-compressed H0/H2 PersLay regressor for "
            "apparent Young's modulus using baseline and uniform structures."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bone_modulus_h0_h2"),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=DEFAULT_VARIANTS,
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        choices=(0, 1, 2),
        default=list(DEFAULT_DIMENSIONS),
    )
    parser.add_argument(
        "--target-column", default="apparent_youngs_modulus_MPa"
    )
    parser.add_argument(
        "--target-mode",
        choices=("raw", "bvtv_residual"),
        default="raw",
        help=(
            "raw predicts modulus directly from diagrams; bvtv_residual fits "
            "the training-only BV/TV power law first and asks PersLay to "
            "predict the remaining log-modulus correction."
        ),
    )
    parser.add_argument("--bvtv-column", default="retained_density_bvtv")
    parser.add_argument("--minimum-persistence", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--image-size", nargs=2, type=int, default=(10, 10))
    parser.add_argument("--bounds-padding-fraction", type=float, default=0.05)
    parser.add_argument("--sigma-multiplier", type=float, default=1.0)
    parser.add_argument("--hidden-dimension", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument(
        "--weight-type",
        choices=(
            "constant",
            "learnable_power",
            "normalized_learnable_power",
            "mlp",
        ),
        default="normalized_learnable_power",
    )
    parser.add_argument("--weight-hidden-dimension", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--minimum-improvement", type=float, default=1.0e-4)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser


def run(arguments: argparse.Namespace) -> None:
    if arguments.minimum_persistence < 0.0:
        raise ExperimentError("--minimum-persistence cannot be negative.")
    if min(arguments.image_size) < 2:
        raise ExperimentError("Each image-size dimension must be at least 2.")
    if arguments.batch_size < 1 or arguments.epochs < 1:
        raise ExperimentError("Batch size and epochs must be positive.")

    data_root = arguments.data_root.resolve()
    output_directory = arguments.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    cache_directory = (
        arguments.cache_dir.resolve()
        if arguments.cache_dir is not None
        else output_directory / "diagram_cache"
    )

    dimensions = tuple(dict.fromkeys(arguments.dimensions))
    branch_names = tuple(dimension_name(value) for value in dimensions)
    variants = tuple(dict.fromkeys(arguments.variants))

    records = read_records(
        data_root,
        variants=variants,
        dimensions=dimensions,
        target_column=arguments.target_column,
        bvtv_column=arguments.bvtv_column,
    )
    assignments = split_structure_groups(
        records,
        train_fraction=arguments.train_fraction,
        validation_fraction=arguments.validation_fraction,
        random_seed=arguments.random_seed,
    )
    split_indices = {
        split: [
            record.index
            for record in records
            if assignments[record.index] == split
        ]
        for split in VALID_SPLITS
    }
    split_groups = {
        split: sorted(
            {
                records[index].structure_id
                for index in split_indices[split]
            }
        )
        for split in VALID_SPLITS
    }

    cached, compression_summary = build_or_load_cache(
        records,
        cache_directory=cache_directory,
        minimum_persistence=arguments.minimum_persistence,
        rebuild=arguments.rebuild_cache,
    )
    image_bounds = compute_training_bounds(
        cached,
        assignments,
        branch_names,
        padding_fraction=arguments.bounds_padding_fraction,
    )
    branch_mass_scales = compute_training_mass_scales(
        cached, assignments, branch_names
    )

    bvtv_intercept, bvtv_exponent = fit_bvtv_power_law(
        records, split_indices["train"]
    )
    training_targets = np.asarray(
        [records[index].target for index in split_indices["train"]],
        dtype=np.float64,
    )
    training_bvtv = np.asarray(
        [records[index].bvtv for index in split_indices["train"]],
        dtype=np.float64,
    )
    training_signals = np.asarray(
        target_signal(
            training_targets,
            training_bvtv,
            target_mode=arguments.target_mode,
            bvtv_intercept=bvtv_intercept,
            bvtv_exponent=bvtv_exponent,
        ),
        dtype=np.float64,
    )
    signal_std = float(training_signals.std(ddof=0))
    if signal_std <= 0.0:
        raise ExperimentError("Training target signal has zero variance.")
    normalization = TargetNormalization(
        mean=float(training_signals.mean()),
        std=signal_std,
    )

    dependencies = import_dependencies()
    torch = dependencies.torch
    seed_everything(arguments.random_seed, torch)
    device = choose_device(arguments.device, torch)
    print(f"Using device: {device}")
    print(
        "Grouped split: "
        + ", ".join(
            f"{split}={len(split_groups[split])} structures/"
            f"{len(split_indices[split])} samples"
            for split in VALID_SPLITS
        )
    )
    signal_units = "MPa" if arguments.target_mode == "raw" else "log residual"
    print(
        f"Training target mode={arguments.target_mode}; "
        f"signal mean={normalization.mean:.6f}, "
        f"std={normalization.std:.6f} ({signal_units})"
    )
    print(
        f"Training-only BV/TV power law: log(E)={bvtv_intercept:.6f} "
        f"+ {bvtv_exponent:.6f} log(BV/TV)"
    )
    for name in branch_names:
        print(
            f"{name.upper()} exact duplicate compression: "
            f"mean {compression_summary[name]['mean_original_points']:.1f} -> "
            f"{compression_summary[name]['mean_unique_points']:.1f} points "
            f"({compression_summary[name]['compression_ratio']:.1f}x); "
            f"training mass scale={branch_mass_scales[name]:.1f}"
        )

    DatasetClass = make_dataset_class(dependencies)
    datasets = {
        split: DatasetClass(records, split_indices[split], cached)
        for split in VALID_SPLITS
    }
    loaders = {
        split: dependencies.DataLoader(
            datasets[split],
            batch_size=arguments.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate_compressed,
            num_workers=arguments.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(arguments.num_workers > 0),
        )
        for split in VALID_SPLITS
    }

    model, branch_settings = make_model(
        branch_names=branch_names,
        image_bounds=image_bounds,
        image_size=arguments.image_size,
        sigma_multiplier=arguments.sigma_multiplier,
        branch_mass_scales=branch_mass_scales,
        hidden_dimension=arguments.hidden_dimension,
        dropout=arguments.dropout,
        weight_type=arguments.weight_type,
        weight_hidden_dimension=arguments.weight_hidden_dimension,
        dependencies=dependencies,
    )
    model = model.to(device)
    number_of_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Trainable parameters: {number_of_parameters:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=arguments.scheduler_factor,
        patience=arguments.scheduler_patience,
    )
    loss_function = torch.nn.MSELoss()

    history: list[dict[str, Any]] = []
    best_validation_rmse = math.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, arguments.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for diagrams, multiplicities, masks, targets, bvtvs, _ in loaders["train"]:
            diagrams = move_mapping(diagrams, device)
            multiplicities = move_mapping(multiplicities, device)
            masks = move_mapping(masks, device)
            targets = targets.to(device, non_blocking=True)
            bvtvs = bvtvs.to(device, non_blocking=True)
            signals = target_signal(
                targets,
                bvtvs,
                target_mode=arguments.target_mode,
                bvtv_intercept=bvtv_intercept,
                bvtv_exponent=bvtv_exponent,
                torch_module=torch,
            )
            targets_normalized = (
                signals - normalization.mean
            ) / normalization.std

            optimizer.zero_grad(set_to_none=True)
            predictions = model(diagrams, multiplicities, masks)
            loss = loss_function(predictions, targets_normalized)
            if not torch.isfinite(loss):
                raise ExperimentError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            if arguments.gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), arguments.gradient_clip
                )
            optimizer.step()

            batch_size = int(targets.shape[0])
            loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size

        training_mse_normalized = loss_sum / max(sample_count, 1)
        validation_rows = evaluate(
            model,
            loaders["validation"],
            records=records,
            normalization=normalization,
            target_mode=arguments.target_mode,
            bvtv_intercept=bvtv_intercept,
            bvtv_exponent=bvtv_exponent,
            device=device,
            torch=torch,
        )
        validation_metrics = regression_metrics(validation_rows)
        scheduler.step(validation_metrics.rmse)

        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "training_mse_normalized": training_mse_normalized,
                "validation_mae_MPa": validation_metrics.mae,
                "validation_rmse_MPa": validation_metrics.rmse,
                "validation_r2": validation_metrics.r2,
            }
        )

        improved = (
            validation_metrics.rmse
            < best_validation_rmse - arguments.minimum_improvement
        )
        if improved:
            best_validation_rmse = validation_metrics.rmse
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            print(
                f"Epoch {epoch:4d} | train normalized MSE "
                f"{training_mse_normalized:.6f} | val RMSE "
                f"{validation_metrics.rmse:.6f} MPa | val R2 "
                f"{validation_metrics.r2:.6f}"
            )

        if epochs_without_improvement >= arguments.patience:
            print(
                f"Early stopping at epoch {epoch}; best epoch was {best_epoch}."
            )
            break

    if best_state is None:
        raise ExperimentError("Training did not produce a valid model state.")
    model.load_state_dict(best_state)

    final_rows: dict[str, list[tuple[int, float, float]]] = {}
    model_metrics: dict[str, Metrics] = {}
    model_metrics_by_variant: dict[str, dict[str, Metrics]] = {}
    for split in VALID_SPLITS:
        rows = evaluate(
            model,
            loaders[split],
            records=records,
            normalization=normalization,
            target_mode=arguments.target_mode,
            bvtv_intercept=bvtv_intercept,
            bvtv_exponent=bvtv_exponent,
            device=device,
            torch=torch,
        )
        final_rows[split] = rows
        model_metrics[split] = regression_metrics(rows)
        model_metrics_by_variant[split] = metrics_by_variant(rows, records)

    mean_baselines = {
        split: mean_baseline_metrics(
            records,
            split_indices[split],
            training_mean=normalization.mean,
        )
        for split in VALID_SPLITS
    }
    bvtv_baselines = {
        split: bvtv_power_law_metrics(
            records,
            split_indices[split],
            intercept=bvtv_intercept,
            exponent=bvtv_exponent,
        )
        for split in VALID_SPLITS
    }

    learned = learned_parameters(model, torch)
    checkpoint = {
        "script_version": SCRIPT_VERSION,
        "model_type": "exact_duplicate_compressed_multi_perslay",
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "branch_settings": branch_settings,
        "branch_mass_scales_training_only": branch_mass_scales,
        "learned_parameters": learned,
        "normalization": asdict(normalization),
        "dimensions": list(dimensions),
        "variants": list(variants),
        "minimum_persistence": arguments.minimum_persistence,
        "target_mode": arguments.target_mode,
        "bvtv_power_law_training_only": {
            "intercept": bvtv_intercept,
            "exponent": bvtv_exponent,
        },
        "hidden_dimension": arguments.hidden_dimension,
        "dropout": arguments.dropout,
        "weight_type": arguments.weight_type,
        "weight_hidden_dimension": arguments.weight_hidden_dimension,
    }
    torch.save(checkpoint, output_directory / "best_model.pt")

    write_csv(
        output_directory / "training_history.csv",
        (
            "epoch",
            "learning_rate",
            "training_mse_normalized",
            "validation_mae_MPa",
            "validation_rmse_MPa",
            "validation_r2",
        ),
        history,
    )
    write_csv(
        output_directory / "split_assignments.csv",
        (
            "sample_id",
            "structure_id",
            "variant",
            "source_filename",
            "split",
            "apparent_youngs_modulus_MPa",
            "retained_density_bvtv",
        ),
        (
            {
                "sample_id": record.sample_id,
                "structure_id": record.structure_id,
                "variant": record.variant,
                "source_filename": record.source_filename,
                "split": assignments[record.index],
                "apparent_youngs_modulus_MPa": record.target,
                "retained_density_bvtv": record.bvtv,
            }
            for record in records
        ),
    )
    prediction_rows: list[dict[str, Any]] = []
    for split in VALID_SPLITS:
        for record_index, truth, prediction in final_rows[split]:
            record = records[record_index]
            prediction_rows.append(
                {
                    "sample_id": record.sample_id,
                    "structure_id": record.structure_id,
                    "variant": record.variant,
                    "split": split,
                    "target_MPa": truth,
                    "prediction_MPa": prediction,
                    "error_MPa": prediction - truth,
                    "absolute_error_MPa": abs(prediction - truth),
                    "retained_density_bvtv": record.bvtv,
                }
            )
    write_csv(
        output_directory / "predictions.csv",
        (
            "sample_id",
            "structure_id",
            "variant",
            "split",
            "target_MPa",
            "prediction_MPa",
            "error_MPa",
            "absolute_error_MPa",
            "retained_density_bvtv",
        ),
        sorted(
            prediction_rows,
            key=lambda row: (
                VALID_SPLITS.index(str(row["split"])),
                str(row["structure_id"]),
                str(row["variant"]),
            ),
        ),
    )

    result = {
        "script_version": SCRIPT_VERSION,
        "model_type": "exact_duplicate_compressed_multi_perslay",
        "scientific_design": {
            "selected_variants": list(variants),
            "excluded_variant": "weakened",
            "grouping_variable": "structure_id",
            "paired_variants_kept_in_same_split": True,
            "test_used_only_after_validation_model_selection": True,
            "diagram_only_model": True,
            "bvtv_not_given_to_perslay": arguments.target_mode == "raw",
            "bvtv_used_only_for_training_fitted_residualization": (
                arguments.target_mode == "bvtv_residual"
            ),
            "duplicate_compression_is_exact_under_sum_aggregation": True,
        },
        "target_column": arguments.target_column,
        "target_mode": arguments.target_mode,
        "bvtv_column": arguments.bvtv_column,
        "dimensions": list(dimensions),
        "minimum_persistence": arguments.minimum_persistence,
        "random_seed": arguments.random_seed,
        "device": str(device),
        "number_of_parameters": number_of_parameters,
        "best_epoch": best_epoch,
        "split_group_counts": {
            split: len(split_groups[split]) for split in VALID_SPLITS
        },
        "split_sample_counts": {
            split: len(split_indices[split]) for split in VALID_SPLITS
        },
        "target_normalization_training_only": asdict(normalization),
        "image_bounds_training_only": image_bounds,
        "branch_settings": branch_settings,
        "branch_mass_scales_training_only": branch_mass_scales,
        "learned_parameters": learned,
        "compression_summary": compression_summary,
        "model_metrics": {
            split: metrics_to_json(metrics)
            for split, metrics in model_metrics.items()
        },
        "model_metrics_by_variant": {
            split: {
                variant: metrics_to_json(metrics)
                for variant, metrics in variant_metrics.items()
            }
            for split, variant_metrics in model_metrics_by_variant.items()
        },
        "training_mean_baseline": {
            split: metrics_to_json(metrics)
            for split, metrics in mean_baselines.items()
        },
        "bvtv_power_law_baseline": {
            "formula": "E = exp(intercept) * BVTV**exponent",
            "intercept": bvtv_intercept,
            "exponent": bvtv_exponent,
            "metrics": {
                split: metrics_to_json(metrics)
                for split, metrics in bvtv_baselines.items()
            },
        },
    }
    with (output_directory / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    print("Final metrics:")
    for split in VALID_SPLITS:
        metrics = model_metrics[split]
        bvtv_metrics = bvtv_baselines[split]
        print(
            f"  {split:10s} PersLay MAE={metrics.mae:.4f} MPa "
            f"RMSE={metrics.rmse:.4f} MPa R2={metrics.r2:.4f} | "
            f"BV/TV-only RMSE={bvtv_metrics.rmse:.4f} MPa "
            f"R2={bvtv_metrics.r2:.4f}"
        )
    print(f"Results written to: {output_directory}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argument_parser()
    arguments = parser.parse_args(argv)
    try:
        run(arguments)
    except ExperimentError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
