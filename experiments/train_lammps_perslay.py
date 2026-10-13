#!/usr/bin/env python3
"""Train a leakage-safe PersLay baseline on LAMMPS persistence diagrams.

This experiment reads ``graph_targets.csv`` and the processed ``*_nodes.csv``
files produced by ``prepare_persistence_graphs_for_pyg.py``.  Each selected
homology dimension is treated as an independent persistence diagram containing
only ``(birth_value, death_value)`` points.  No persistence-pairing edges or
representative-size features are used.

The default model has separate H1 and H2 Gaussian PersLay branches.  Their
embeddings are concatenated with the normalized filtration endpoint and passed
to a small regression head.  This provides a diagram-only baseline against the
two-branch persistence-pairing GNN while keeping the graph-level covariate used
by that GNN.  Pass ``--exclude-filtration-endpoint`` for a purely topological
ablation.

Scientific safeguards
---------------------
* an external seed-grouped ``split_assignments.csv`` is required;
* diagram bounds, graph-feature normalization, and target normalization are
  fitted on the training seeds only;
* the test set is evaluated only after validation-based model selection;
* timestep is excluded unless ``--include-timestep`` is requested;
* essential intervals use the finite endpoint already written by the graph
  preprocessor, or can be removed with ``--essential-policy drop``;
* node identifiers, representative sizes, graph edges, and PotEng never enter
  the PersLay diagram features.

Example
-------
    python3 experiments/train_lammps_perslay.py \
        --data-root ~/Desktop/gel_lammps_test/pyg_data \
        --split-assignments \
          ~/Desktop/gel_lammps_test/gnn_results/dimension_ablation/H1_H2/split_assignments.csv \
        --output-dir \
          ~/Desktop/gel_lammps_test/perslay_results/H1_H2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCRIPT_VERSION = 1
CACHE_VERSION = 1
DEFAULT_TARGET = "PotEng"
VALID_SPLITS = ("train", "validation", "test")
ORDER_TOLERANCE = 1.0e-6


class PerslayExperimentError(RuntimeError):
    """An invalid input, option, dependency, cache, or training-state error."""


@dataclass(frozen=True)
class DiagramRecord:
    index: int
    graph_id: str
    seed: int
    timestep: int
    filtration_endpoint: float
    target: float
    nodes_path: Path


@dataclass(frozen=True)
class Normalization:
    graph_mean: tuple[float, ...]
    graph_std: tuple[float, ...]
    target_mean: float
    target_std: float


@dataclass(frozen=True)
class Metrics:
    number_of_samples: int
    mae: float
    rmse: float
    r2: float


def finite_float(raw: str, *, context: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise PerslayExperimentError(
            f"{context} must be numeric; found {raw!r}."
        ) from error
    if not math.isfinite(value):
        raise PerslayExperimentError(
            f"{context} must be finite; found {raw!r}."
        )
    return value


def integer(raw: str, *, context: str) -> int:
    text = str(raw).strip()
    try:
        value = int(text)
    except (TypeError, ValueError) as error:
        raise PerslayExperimentError(
            f"{context} must be an integer; found {raw!r}."
        ) from error
    if text not in {str(value), f"+{value}"}:
        raise PerslayExperimentError(
            f"{context} must be an integer; found {raw!r}."
        )
    return value


def require_columns(
    fieldnames: Sequence[str] | None,
    required: Iterable[str],
    path: Path,
) -> list[str]:
    if fieldnames is None:
        raise PerslayExperimentError(f"CSV has no header: {path}")
    names = list(fieldnames)
    if len(names) != len(set(names)):
        raise PerslayExperimentError(f"CSV has duplicate columns: {path}")
    missing = [name for name in required if name not in names]
    if missing:
        raise PerslayExperimentError(
            f"{path} is missing required column(s): {', '.join(missing)}."
        )
    return names


def check_row(
    row: Mapping[str | None, Any],
    path: Path,
    row_number: int,
) -> None:
    if None in row:
        raise PerslayExperimentError(
            f"{path}:{row_number} has more values than header columns."
        )
    missing = [name for name, value in row.items() if value is None]
    if missing:
        raise PerslayExperimentError(
            f"{path}:{row_number} is missing: "
            + ", ".join(str(name) for name in missing)
            + "."
        )


def resolve_beneath(root: Path, relative: str, *, context: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise PerslayExperimentError(
            f"{context} escapes the data root: {relative!r}."
        )
    if not candidate.is_file():
        raise PerslayExperimentError(f"{context} does not exist: {candidate}")
    return candidate


def read_graph_index(
    data_root: Path,
    *,
    index_name: str,
    target_column: str,
) -> tuple[list[DiagramRecord], Path]:
    index_path = (data_root / index_name).resolve()
    if not index_path.is_file():
        raise PerslayExperimentError(
            f"Graph index does not exist: {index_path}"
        )

    required = {
        "graph_id",
        "seed",
        "timestep",
        "filtration_endpoint",
        target_column,
        "nodes_csv",
    }
    records: list[DiagramRecord] = []
    seen_graph_ids: set[str] = set()

    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, required, index_path)
        for row_number, row in enumerate(reader, start=2):
            check_row(row, index_path, row_number)
            graph_id = str(row["graph_id"]).strip()
            if not graph_id:
                raise PerslayExperimentError(
                    f"{index_path}:{row_number} graph_id is empty."
                )
            if graph_id in seen_graph_ids:
                raise PerslayExperimentError(
                    f"Duplicate graph_id {graph_id!r} in {index_path}."
                )
            seen_graph_ids.add(graph_id)
            records.append(
                DiagramRecord(
                    index=len(records),
                    graph_id=graph_id,
                    seed=integer(
                        str(row["seed"]),
                        context=f"{index_path}:{row_number} seed",
                    ),
                    timestep=integer(
                        str(row["timestep"]),
                        context=f"{index_path}:{row_number} timestep",
                    ),
                    filtration_endpoint=finite_float(
                        str(row["filtration_endpoint"]),
                        context=(
                            f"{index_path}:{row_number} "
                            "filtration_endpoint"
                        ),
                    ),
                    target=finite_float(
                        str(row[target_column]),
                        context=(
                            f"{index_path}:{row_number} {target_column}"
                        ),
                    ),
                    nodes_path=resolve_beneath(
                        data_root,
                        str(row["nodes_csv"]),
                        context=f"{index_path}:{row_number} nodes_csv",
                    ),
                )
            )

    if not records:
        raise PerslayExperimentError(
            f"No graph records were read from {index_path}."
        )
    return records, index_path


def read_split_assignments(
    path: Path,
    records: Sequence[DiagramRecord],
) -> dict[int, str]:
    if not path.is_file():
        raise PerslayExperimentError(
            f"Split-assignment CSV does not exist: {path}"
        )
    record_by_graph_id = {record.graph_id: record for record in records}
    assignments: dict[int, str] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(
            reader.fieldnames,
            {"graph_id", "seed", "timestep", "split"},
            path,
        )
        for row_number, row in enumerate(reader, start=2):
            check_row(row, path, row_number)
            graph_id = str(row["graph_id"]).strip()
            record = record_by_graph_id.get(graph_id)
            if record is None:
                continue
            if record.index in assignments:
                raise PerslayExperimentError(
                    f"Duplicate assignment for {graph_id!r} in {path}."
                )
            seed = integer(
                str(row["seed"]),
                context=f"{path}:{row_number} seed",
            )
            timestep = integer(
                str(row["timestep"]),
                context=f"{path}:{row_number} timestep",
            )
            if seed != record.seed or timestep != record.timestep:
                raise PerslayExperimentError(
                    f"{path}:{row_number} metadata disagree with "
                    f"graph_targets.csv for {graph_id!r}."
                )
            split = str(row["split"]).strip()
            if split not in VALID_SPLITS:
                raise PerslayExperimentError(
                    f"{path}:{row_number} split must be one of "
                    f"{', '.join(VALID_SPLITS)}; found {split!r}."
                )
            assignments[record.index] = split

    missing = [
        record.graph_id
        for record in records
        if record.index not in assignments
    ]
    if missing:
        preview = ", ".join(repr(value) for value in missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise PerslayExperimentError(
            f"{path} has no assignment for {len(missing)} graph(s): "
            f"{preview}{suffix}"
        )

    seed_splits: dict[int, str] = {}
    for record in records:
        split = assignments[record.index]
        previous = seed_splits.setdefault(record.seed, split)
        if previous != split:
            raise PerslayExperimentError(
                f"Seed {record.seed} occurs in both {previous} and {split}."
            )
    for split in VALID_SPLITS:
        if split not in assignments.values():
            raise PerslayExperimentError(
                f"External assignments leave {split} empty."
            )
    return assignments


def branch_name(dimension: int) -> str:
    return f"H{dimension}"


def read_selected_diagrams(
    path: Path,
    *,
    dimensions: Sequence[int],
    essential_policy: str,
    minimum_persistence: float,
) -> dict[str, np.ndarray]:
    selected = set(dimensions)
    points: dict[int, list[tuple[float, float]]] = {
        dimension: [] for dimension in dimensions
    }
    required = {
        "dimension",
        "birth_value",
        "death_value",
        "is_essential",
    }

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, required, path)
        for row_number, row in enumerate(reader, start=2):
            check_row(row, path, row_number)
            dimension = integer(
                str(row["dimension"]),
                context=f"{path}:{row_number} dimension",
            )
            if dimension not in selected:
                continue
            essential = integer(
                str(row["is_essential"]),
                context=f"{path}:{row_number} is_essential",
            )
            if essential not in (0, 1):
                raise PerslayExperimentError(
                    f"{path}:{row_number} is_essential must be 0 or 1."
                )
            if essential_policy == "drop" and essential == 1:
                continue

            birth = finite_float(
                str(row["birth_value"]),
                context=f"{path}:{row_number} birth_value",
            )
            death = finite_float(
                str(row["death_value"]),
                context=f"{path}:{row_number} death_value",
            )
            if death < birth - ORDER_TOLERANCE:
                raise PerslayExperimentError(
                    f"{path}:{row_number} death_value {death} is below "
                    f"birth_value {birth}."
                )
            if death < birth:
                death = birth
            if death - birth <= minimum_persistence:
                continue
            points[dimension].append((birth, death))

    result: dict[str, np.ndarray] = {}
    for dimension in dimensions:
        rows = points[dimension]
        result[branch_name(dimension)] = (
            np.asarray(rows, dtype=np.float32).reshape(-1, 2)
            if rows
            else np.empty((0, 2), dtype=np.float32)
        )
    return result


def cache_configuration(
    records: Sequence[DiagramRecord],
    assignments: Mapping[int, str],
    *,
    dimensions: Sequence[int],
    essential_policy: str,
    minimum_persistence: float,
    bounds_padding_fraction: float,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    for record in records:
        try:
            status = record.nodes_path.stat()
        except OSError as error:
            raise PerslayExperimentError(
                f"Could not inspect {record.nodes_path}: {error}"
            ) from error
        row = [
            record.graph_id,
            record.seed,
            record.timestep,
            record.filtration_endpoint,
            record.target,
            assignments[record.index],
            status.st_size,
            status.st_mtime_ns,
        ]
        digest.update(
            json.dumps(
                row,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return {
        "cache_version": CACHE_VERSION,
        "dimensions": list(dimensions),
        "essential_policy": essential_policy,
        "minimum_persistence": minimum_persistence,
        "bounds_padding_fraction": bounds_padding_fraction,
        "number_of_graphs": len(records),
        "inputs_sha256": digest.hexdigest(),
    }


def calculate_image_bounds(
    extrema: Mapping[int, Mapping[str, float]],
    *,
    dimensions: Sequence[int],
    padding_fraction: float,
) -> dict[str, list[list[float]]]:
    bounds: dict[str, list[list[float]]] = {}
    for dimension in dimensions:
        values = extrema[dimension]
        birth_min = values["birth_min"]
        birth_max = values["birth_max"]
        persistence_max = values["persistence_max"]
        if not all(
            math.isfinite(value)
            for value in (birth_min, birth_max, persistence_max)
        ):
            raise PerslayExperimentError(
                f"No H{dimension} points occur in the training split after "
                "filtering."
            )
        birth_range = birth_max - birth_min
        birth_scale = (
            birth_range
            if birth_range > 1.0e-12
            else max(abs(birth_min), 1.0)
        )
        persistence_scale = max(persistence_max, 1.0e-12)
        birth_padding = padding_fraction * birth_scale
        persistence_padding = padding_fraction * persistence_scale
        bounds[branch_name(dimension)] = [
            [birth_min - birth_padding, birth_max + birth_padding],
            [0.0, persistence_max + persistence_padding],
        ]
    return bounds


def validate_cached_bounds(
    raw: Any,
    *,
    dimensions: Sequence[int],
) -> dict[str, list[list[float]]]:
    if not isinstance(raw, dict):
        raise ValueError("image_bounds is not an object")
    result: dict[str, list[list[float]]] = {}
    for dimension in dimensions:
        name = branch_name(dimension)
        value = raw.get(name)
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(not isinstance(axis, list) or len(axis) != 2 for axis in value)
        ):
            raise ValueError(f"{name} has malformed image bounds")
        converted = [
            [float(value[0][0]), float(value[0][1])],
            [float(value[1][0]), float(value[1][1])],
        ]
        flattened = [number for axis in converted for number in axis]
        if not all(math.isfinite(number) for number in flattened):
            raise ValueError(f"{name} image bounds are non-finite")
        if converted[0][1] <= converted[0][0]:
            raise ValueError(f"{name} birth bounds have nonpositive width")
        if converted[1][1] <= converted[1][0]:
            raise ValueError(
                f"{name} persistence bounds have nonpositive width"
            )
        result[name] = converted
    return result


def prepare_diagram_cache(
    records: Sequence[DiagramRecord],
    assignments: Mapping[int, str],
    *,
    output_directory: Path,
    dimensions: Sequence[int],
    essential_policy: str,
    minimum_persistence: float,
    bounds_padding_fraction: float,
    rebuild: bool,
) -> tuple[Path, dict[str, list[list[float]]], dict[str, dict[str, int]]]:
    cache_directory = output_directory / "diagram_cache"
    cache_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_directory / "cache_manifest.json"
    configuration = cache_configuration(
        records,
        assignments,
        dimensions=dimensions,
        essential_policy=essential_policy,
        minimum_persistence=minimum_persistence,
        bounds_padding_fraction=bounds_padding_fraction,
    )

    if manifest_path.is_file() and not rebuild:
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            if (
                isinstance(manifest, dict)
                and manifest.get("configuration") == configuration
            ):
                missing = [
                    index
                    for index in range(len(records))
                    if not (cache_directory / f"{index:08d}.npz").is_file()
                ]
                if not missing:
                    image_bounds = validate_cached_bounds(
                        manifest.get("image_bounds"),
                        dimensions=dimensions,
                    )
                    point_counts = manifest.get("point_counts")
                    if not isinstance(point_counts, dict):
                        raise ValueError("point_counts is not an object")
                    print(f"Diagram cache: {cache_directory}")
                    return cache_directory, image_bounds, point_counts
                print(
                    f"Diagram cache is missing {len(missing)} file(s); "
                    "rebuilding."
                )
            else:
                print(
                    "Diagram cache does not match the current inputs; "
                    "rebuilding."
                )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(
                f"Diagram cache manifest is unusable ({error}); rebuilding.",
                file=sys.stderr,
            )
    elif rebuild and manifest_path.exists():
        print("Ignoring diagram cache because --rebuild-cache was set.")

    extrema = {
        dimension: {
            "birth_min": math.inf,
            "birth_max": -math.inf,
            "persistence_max": -math.inf,
        }
        for dimension in dimensions
    }
    point_counts = {
        split: {branch_name(dimension): 0 for dimension in dimensions}
        for split in VALID_SPLITS
    }

    for position, record in enumerate(records, start=1):
        diagrams = read_selected_diagrams(
            record.nodes_path,
            dimensions=dimensions,
            essential_policy=essential_policy,
            minimum_persistence=minimum_persistence,
        )
        cache_path = cache_directory / f"{record.index:08d}.npz"
        temporary_path = cache_directory / f".{record.index:08d}.tmp.npz"
        try:
            with temporary_path.open("wb") as handle:
                np.savez(handle, **diagrams)
            temporary_path.replace(cache_path)
        except OSError as error:
            raise PerslayExperimentError(
                f"Could not write diagram cache {cache_path}: {error}"
            ) from error

        split = assignments[record.index]
        for dimension in dimensions:
            name = branch_name(dimension)
            diagram = diagrams[name]
            point_counts[split][name] += int(diagram.shape[0])
            if split != "train" or diagram.shape[0] == 0:
                continue
            birth = diagram[:, 0].astype(np.float64, copy=False)
            persistence = (
                diagram[:, 1] - diagram[:, 0]
            ).astype(np.float64, copy=False)
            extrema[dimension]["birth_min"] = min(
                extrema[dimension]["birth_min"],
                float(birth.min()),
            )
            extrema[dimension]["birth_max"] = max(
                extrema[dimension]["birth_max"],
                float(birth.max()),
            )
            extrema[dimension]["persistence_max"] = max(
                extrema[dimension]["persistence_max"],
                float(persistence.max()),
            )

        if (
            position == 1
            or position % 250 == 0
            or position == len(records)
        ):
            print(f"Diagram cache: {position}/{len(records)} graphs")

    image_bounds = calculate_image_bounds(
        extrema,
        dimensions=dimensions,
        padding_fraction=bounds_padding_fraction,
    )
    manifest = {
        "configuration": configuration,
        "image_bounds": image_bounds,
        "point_counts": point_counts,
    }
    temporary_manifest = cache_directory / ".cache_manifest.json.tmp"
    try:
        with temporary_manifest.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        temporary_manifest.replace(manifest_path)
    except OSError as error:
        raise PerslayExperimentError(
            f"Could not write diagram-cache manifest {manifest_path}: {error}"
        ) from error
    print(f"Diagram cache written: {cache_directory}")
    return cache_directory, image_bounds, point_counts


def graph_feature_names(
    *,
    exclude_filtration_endpoint: bool,
    include_timestep: bool,
) -> tuple[str, ...]:
    names: list[str] = []
    if not exclude_filtration_endpoint:
        names.append("filtration_endpoint")
    if include_timestep:
        names.append("timestep")
    return tuple(names)


def graph_values(
    record: DiagramRecord,
    *,
    exclude_filtration_endpoint: bool,
    include_timestep: bool,
) -> list[float]:
    values: list[float] = []
    if not exclude_filtration_endpoint:
        values.append(record.filtration_endpoint)
    if include_timestep:
        values.append(float(record.timestep))
    return values


def compute_normalization(
    records: Sequence[DiagramRecord],
    assignments: Mapping[int, str],
    *,
    exclude_filtration_endpoint: bool,
    include_timestep: bool,
) -> Normalization:
    training_records = [
        record
        for record in records
        if assignments[record.index] == "train"
    ]
    graph_rows = [
        graph_values(
            record,
            exclude_filtration_endpoint=exclude_filtration_endpoint,
            include_timestep=include_timestep,
        )
        for record in training_records
    ]
    targets = np.asarray(
        [record.target for record in training_records],
        dtype=np.float64,
    )

    if graph_rows and graph_rows[0]:
        graph_array = np.asarray(graph_rows, dtype=np.float64)
        graph_mean_array = graph_array.mean(axis=0)
        graph_std_array = graph_array.std(axis=0)
        graph_std_array = np.where(
            graph_std_array > 1.0e-12,
            graph_std_array,
            1.0,
        )
        graph_mean = tuple(float(value) for value in graph_mean_array)
        graph_std = tuple(float(value) for value in graph_std_array)
    else:
        graph_mean = ()
        graph_std = ()

    target_mean = float(targets.mean())
    target_std = float(targets.std())
    if target_std <= 1.0e-12:
        target_std = 1.0
    return Normalization(
        graph_mean=graph_mean,
        graph_std=graph_std,
        target_mean=target_mean,
        target_std=target_std,
    )


class CachedDiagramDataset:
    """Module-level dataset compatible with Python 3.14 forkserver workers."""

    def __init__(
        self,
        records: Sequence[DiagramRecord],
        indices: Sequence[int],
        *,
        cache_directory: Path,
        dimensions: Sequence[int],
        normalization: Normalization,
        exclude_filtration_endpoint: bool,
        include_timestep: bool,
    ) -> None:
        self.records = tuple(records[index] for index in indices)
        self.cache_directory = cache_directory
        self.branch_names = tuple(
            branch_name(dimension) for dimension in dimensions
        )
        self.normalization = normalization
        self.exclude_filtration_endpoint = exclude_filtration_endpoint
        self.include_timestep = include_timestep
        self.graph_mean = np.asarray(
            normalization.graph_mean,
            dtype=np.float32,
        )
        self.graph_std = np.asarray(
            normalization.graph_std,
            dtype=np.float32,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, position: int) -> tuple[Any, Any, Any, Any]:
        import torch

        record = self.records[position]
        cache_path = self.cache_directory / f"{record.index:08d}.npz"
        try:
            with np.load(cache_path, allow_pickle=False) as archive:
                diagrams = {
                    name: torch.from_numpy(
                        np.ascontiguousarray(
                            np.asarray(archive[name], dtype=np.float32)
                        )
                    )
                    for name in self.branch_names
                }
        except (OSError, KeyError, ValueError) as error:
            raise PerslayExperimentError(
                f"Could not read diagram cache {cache_path}: {error}"
            ) from error

        raw_graph = np.asarray(
            graph_values(
                record,
                exclude_filtration_endpoint=(
                    self.exclude_filtration_endpoint
                ),
                include_timestep=self.include_timestep,
            ),
            dtype=np.float32,
        )
        normalized_graph = (
            (raw_graph - self.graph_mean) / self.graph_std
            if raw_graph.size
            else raw_graph
        )
        normalized_target = (
            record.target - self.normalization.target_mean
        ) / self.normalization.target_std
        return (
            diagrams,
            torch.from_numpy(np.ascontiguousarray(normalized_graph)),
            torch.tensor(normalized_target, dtype=torch.float32),
            torch.tensor(record.index, dtype=torch.long),
        )


def collate_cached_diagrams(
    batch: Sequence[tuple[Any, Any, Any, Any]],
) -> tuple[Any, Any, Any, Any, Any]:
    import torch
    from torch.nn.utils.rnn import pad_sequence

    diagram_rows, graph_rows, targets, record_ids = zip(*batch)
    names = tuple(diagram_rows[0].keys())
    padded: dict[str, Any] = {}
    masks: dict[str, Any] = {}
    for name in names:
        diagrams = [row[name] for row in diagram_rows]
        lengths = torch.tensor(
            [diagram.shape[0] for diagram in diagrams],
            dtype=torch.long,
        )
        if int(lengths.max().item()) == 0:
            padded[name] = torch.zeros(
                len(diagrams),
                1,
                2,
                dtype=torch.float32,
            )
            masks[name] = torch.zeros(
                len(diagrams),
                1,
                dtype=torch.bool,
            )
        else:
            padded[name] = pad_sequence(
                diagrams,
                batch_first=True,
                padding_value=0.0,
            )
            maximum_length = padded[name].shape[1]
            masks[name] = (
                torch.arange(maximum_length).unsqueeze(0)
                < lengths.unsqueeze(1)
            )
    return (
        padded,
        masks,
        torch.stack(graph_rows),
        torch.stack(targets),
        torch.stack(record_ids),
    )


def import_ml_dependencies() -> SimpleNamespace:
    try:
        import torch
        from perslay.models import PerslayRegressor
        from torch import nn
        from torch.nn import functional
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise PerslayExperimentError(
            "PyTorch and the local perslay package are required. From the "
            "repository root, run: python3 -m pip install -e ."
        ) from error
    return SimpleNamespace(
        torch=torch,
        nn=nn,
        functional=functional,
        DataLoader=DataLoader,
        PerslayRegressor=PerslayRegressor,
    )


def sigma_from_bounds(
    bounds: Sequence[Sequence[float]],
    image_size: Sequence[int],
    *,
    multiplier: float,
) -> tuple[float, float]:
    sigma_x = (
        multiplier
        * (float(bounds[0][1]) - float(bounds[0][0]))
        / int(image_size[0])
    )
    sigma_y = (
        multiplier
        * (float(bounds[1][1]) - float(bounds[1][0]))
        / int(image_size[1])
    )
    return sigma_x, sigma_y


def make_model(
    *,
    image_bounds: Mapping[str, Sequence[Sequence[float]]],
    image_size: Sequence[int],
    sigma_multiplier: float,
    number_of_graph_features: int,
    hidden_dimension: int,
    dropout: float,
    weight_type: str,
    weight_hidden_dimension: int,
    dependencies: SimpleNamespace,
) -> tuple[Any, dict[str, dict[str, Any]]]:
    nn = dependencies.nn
    branch_settings: dict[str, dict[str, Any]] = {}
    for name, bounds in image_bounds.items():
        sigma_x, sigma_y = sigma_from_bounds(
            bounds,
            image_size,
            multiplier=sigma_multiplier,
        )
        branch_settings[name] = {
            "image_size": [int(value) for value in image_size],
            "image_bounds": [
                [float(bounds[0][0]), float(bounds[0][1])],
                [float(bounds[1][0]), float(bounds[1][1])],
            ],
            "initial_sigma_x": sigma_x,
            "initial_sigma_y": sigma_y,
        }

    class LammpsMultiPerslayRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.branch_names = tuple(branch_settings)
            branches = {}
            diagram_width = 0
            for name, settings in branch_settings.items():
                wrapper = dependencies.PerslayRegressor(
                    image_size=settings["image_size"],
                    image_bnds=settings["image_bounds"],
                    sigma_x=settings["initial_sigma_x"],
                    sigma_y=settings["initial_sigma_y"],
                    hidden_dim=hidden_dimension,
                    weight_type=weight_type,
                    weight_hidden_dim=weight_hidden_dimension,
                )
                branches[name] = wrapper.perslay
                feature_dimension = int(
                    np.prod(wrapper.perslay.phi.mu.shape[1:])
                )
                settings["feature_dimension"] = feature_dimension
                diagram_width += feature_dimension
            self.branches = nn.ModuleDict(branches)
            self.regressor = nn.Sequential(
                nn.Linear(
                    diagram_width + number_of_graph_features,
                    hidden_dimension,
                ),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dimension, 1),
            )

        def forward(
            self,
            diagrams: Mapping[str, Any],
            graph_attributes: Any,
            *,
            mask: Mapping[str, Any],
        ) -> Any:
            features = [
                self.branches[name](diagrams[name], mask=mask[name])
                for name in self.branch_names
            ]
            if graph_attributes.shape[1] > 0:
                features.append(graph_attributes)
            return self.regressor(
                dependencies.torch.cat(features, dim=1)
            ).view(-1)

    return LammpsMultiPerslayRegressor(), branch_settings


def choose_device(requested: str, torch: Any) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PerslayExperimentError(
            "CUDA was requested but is not available."
        )
    return device


def move_mapping_to_device(values: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {
        name: tensor.to(device, non_blocking=True)
        for name, tensor in values.items()
    }


def evaluate_model(
    model: Any,
    loader: Any,
    *,
    records: Sequence[DiagramRecord],
    device: Any,
    normalization: Normalization,
    dependencies: SimpleNamespace,
) -> list[tuple[int, float, float]]:
    torch = dependencies.torch
    model.eval()
    rows: list[tuple[int, float, float]] = []
    with torch.no_grad():
        for (
            diagrams,
            masks,
            graph_attributes,
            _,
            record_ids,
        ) in loader:
            diagrams = move_mapping_to_device(diagrams, device)
            masks = move_mapping_to_device(masks, device)
            graph_attributes = graph_attributes.to(
                device,
                non_blocking=True,
            )
            prediction_normalized = model(
                diagrams,
                graph_attributes,
                mask=masks,
            )
            predictions = (
                prediction_normalized * normalization.target_std
                + normalization.target_mean
            )
            for record_id, prediction in zip(
                record_ids.tolist(),
                predictions.cpu().tolist(),
            ):
                sample = int(record_id)
                rows.append(
                    (
                        sample,
                        records[sample].target,
                        float(prediction),
                    )
                )
    return rows


def regression_metrics(
    rows: Sequence[tuple[int, float, float]],
) -> Metrics:
    if not rows:
        raise PerslayExperimentError(
            "Cannot calculate metrics for an empty split."
        )
    truth = np.asarray([row[1] for row in rows], dtype=np.float64)
    prediction = np.asarray([row[2] for row in rows], dtype=np.float64)
    residual = prediction - truth
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    denominator = float(np.sum(np.square(truth - truth.mean())))
    r2 = (
        float(1.0 - np.sum(np.square(residual)) / denominator)
        if denominator > 0.0
        else float("nan")
    )
    return Metrics(len(rows), mae, rmse, r2)


def mean_baseline_metrics(
    records: Sequence[DiagramRecord],
    indices: Sequence[int],
    *,
    training_mean: float,
) -> Metrics:
    return regression_metrics(
        [
            (index, records[index].target, training_mean)
            for index in indices
        ]
    )


def fit_filtration_endpoint_baseline(
    records: Sequence[DiagramRecord],
    training_indices: Sequence[int],
) -> tuple[float, float]:
    endpoints = np.asarray(
        [
            records[index].filtration_endpoint
            for index in training_indices
        ],
        dtype=np.float64,
    )
    targets = np.asarray(
        [records[index].target for index in training_indices],
        dtype=np.float64,
    )
    design = np.column_stack((np.ones(endpoints.shape[0]), endpoints))
    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        targets,
        rcond=None,
    )
    return float(coefficients[0]), float(coefficients[1])


def filtration_endpoint_baseline_metrics(
    records: Sequence[DiagramRecord],
    indices: Sequence[int],
    *,
    intercept: float,
    slope: float,
) -> Metrics:
    return regression_metrics(
        [
            (
                index,
                records[index].target,
                intercept + slope * records[index].filtration_endpoint,
            )
            for index in indices
        ]
    )


def learned_branch_parameters(
    model: Any,
    dependencies: SimpleNamespace,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    functional = dependencies.functional
    torch = dependencies.torch
    for name, branch in model.branches.items():
        values: dict[str, float] = {}
        if hasattr(branch.phi, "log_sigma"):
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


def json_safe_metrics(metrics: Metrics) -> dict[str, Any]:
    result = asdict(metrics)
    if not math.isfinite(result["r2"]):
        result["r2"] = None
    return result


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train an H1/H2 Gaussian PersLay regression baseline directly "
            "from processed persistence-graph node CSVs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--split-assignments",
        required=True,
        type=Path,
        help=(
            "The GNN split_assignments.csv to reuse exactly. A seed-grouped "
            "external split is required for a valid comparison."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--index-name", default="graph_targets.csv")
    parser.add_argument("--target-column", default=DEFAULT_TARGET)
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Homology dimensions used as independent PersLay branches.",
    )
    parser.add_argument(
        "--essential-policy",
        choices=("capped", "drop"),
        default="capped",
        help=(
            "Keep essential intervals with their preprocessed finite endpoint "
            "or remove them."
        ),
    )
    parser.add_argument(
        "--minimum-persistence",
        type=float,
        default=0.0,
        help="Discard intervals with persistence at most this value.",
    )
    parser.add_argument(
        "--bounds-padding-fraction",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("BIRTH_PIXELS", "PERSISTENCE_PIXELS"),
        default=(10, 10),
    )
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
    parser.add_argument(
        "--weight-hidden-dimension",
        type=int,
        default=16,
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--minimum-improvement", type=float, default=1.0e-5)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--exclude-filtration-endpoint",
        action="store_true",
        help=(
            "Remove the graph-level filtration endpoint, producing a purely "
            "diagram-based model. By default it is included to match the GNN."
        ),
    )
    parser.add_argument(
        "--include-timestep",
        action="store_true",
        help="Include normalized timestep as an explicit covariate.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore and rebuild the binary persistence-diagram cache.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing result files in output-dir.",
    )
    return parser


def validate_arguments(arguments: argparse.Namespace) -> None:
    if not arguments.dimensions:
        raise PerslayExperimentError(
            "--dimensions must contain at least one value."
        )
    if len(arguments.dimensions) != len(set(arguments.dimensions)):
        raise PerslayExperimentError("--dimensions contains duplicates.")
    if any(dimension < 0 for dimension in arguments.dimensions):
        raise PerslayExperimentError(
            "--dimensions values must be nonnegative."
        )
    if (
        not math.isfinite(arguments.minimum_persistence)
        or arguments.minimum_persistence < 0.0
    ):
        raise PerslayExperimentError(
            "--minimum-persistence must be nonnegative."
        )
    if (
        not math.isfinite(arguments.bounds_padding_fraction)
        or arguments.bounds_padding_fraction < 0.0
    ):
        raise PerslayExperimentError(
            "--bounds-padding-fraction must be nonnegative."
        )
    if any(value < 1 for value in arguments.image_size):
        raise PerslayExperimentError(
            "--image-size entries must be positive."
        )
    if (
        not math.isfinite(arguments.sigma_multiplier)
        or arguments.sigma_multiplier <= 0.0
    ):
        raise PerslayExperimentError(
            "--sigma-multiplier must be positive."
        )
    if arguments.hidden_dimension < 1:
        raise PerslayExperimentError(
            "--hidden-dimension must be positive."
        )
    if arguments.weight_hidden_dimension < 1:
        raise PerslayExperimentError(
            "--weight-hidden-dimension must be positive."
        )
    if not (0.0 <= arguments.dropout < 1.0):
        raise PerslayExperimentError("--dropout must be in [0, 1).")
    for name in (
        "batch_size",
        "epochs",
        "patience",
        "scheduler_patience",
    ):
        if getattr(arguments, name) < 1:
            raise PerslayExperimentError(
                f"--{name.replace('_', '-')} must be positive."
            )
    if (
        not math.isfinite(arguments.minimum_improvement)
        or arguments.minimum_improvement < 0.0
    ):
        raise PerslayExperimentError(
            "--minimum-improvement must be nonnegative."
        )
    if (
        not math.isfinite(arguments.learning_rate)
        or arguments.learning_rate <= 0.0
    ):
        raise PerslayExperimentError(
            "--learning-rate must be positive."
        )
    if (
        not math.isfinite(arguments.weight_decay)
        or arguments.weight_decay < 0.0
    ):
        raise PerslayExperimentError(
            "--weight-decay must be nonnegative."
        )
    if (
        not math.isfinite(arguments.gradient_clip)
        or arguments.gradient_clip < 0.0
    ):
        raise PerslayExperimentError(
            "--gradient-clip must be nonnegative."
        )
    if not (
        math.isfinite(arguments.scheduler_factor)
        and 0.0 < arguments.scheduler_factor < 1.0
    ):
        raise PerslayExperimentError(
            "--scheduler-factor must lie strictly between 0 and 1."
        )
    if arguments.num_workers < 0:
        raise PerslayExperimentError(
            "--num-workers must be nonnegative."
        )


def run(arguments: argparse.Namespace) -> None:
    validate_arguments(arguments)
    dimensions = tuple(sorted(arguments.dimensions))
    data_root = arguments.data_root.expanduser().resolve()
    split_path = arguments.split_assignments.expanduser().resolve()
    output_directory = arguments.output_dir.expanduser().resolve()
    if not data_root.is_dir():
        raise PerslayExperimentError(
            f"Data root is not a directory: {data_root}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    result_paths = [
        output_directory / "metrics.json",
        output_directory / "predictions.csv",
        output_directory / "split_assignments.csv",
        output_directory / "training_history.csv",
        output_directory / "best_model.pt",
    ]
    existing = [path for path in result_paths if path.exists()]
    if existing and not arguments.overwrite:
        preview = "\n  ".join(str(path) for path in existing)
        raise PerslayExperimentError(
            "Result files already exist; pass --overwrite to replace them:\n  "
            + preview
        )

    records, index_path = read_graph_index(
        data_root,
        index_name=arguments.index_name,
        target_column=arguments.target_column,
    )
    assignments = read_split_assignments(split_path, records)
    split_indices = {
        split: [
            record.index
            for record in records
            if assignments[record.index] == split
        ]
        for split in VALID_SPLITS
    }
    split_seed_counts = {
        split: len({records[index].seed for index in indices})
        for split, indices in split_indices.items()
    }
    print(
        "Samples: "
        + ", ".join(
            f"{split}={len(split_indices[split])} "
            f"({split_seed_counts[split]} seeds)"
            for split in VALID_SPLITS
        )
    )
    print(
        "PersLay branches: "
        + ", ".join(branch_name(dimension) for dimension in dimensions)
    )
    print(f"Split source: {split_path}")

    dependencies = import_ml_dependencies()
    torch = dependencies.torch
    cache_directory, image_bounds, point_counts = prepare_diagram_cache(
        records,
        assignments,
        output_directory=output_directory,
        dimensions=dimensions,
        essential_policy=arguments.essential_policy,
        minimum_persistence=arguments.minimum_persistence,
        bounds_padding_fraction=arguments.bounds_padding_fraction,
        rebuild=arguments.rebuild_cache,
    )
    normalization = compute_normalization(
        records,
        assignments,
        exclude_filtration_endpoint=(
            arguments.exclude_filtration_endpoint
        ),
        include_timestep=arguments.include_timestep,
    )

    random.seed(arguments.random_seed)
    np.random.seed(arguments.random_seed)
    torch.manual_seed(arguments.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.random_seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    datasets = {
        split: CachedDiagramDataset(
            records,
            indices,
            cache_directory=cache_directory,
            dimensions=dimensions,
            normalization=normalization,
            exclude_filtration_endpoint=(
                arguments.exclude_filtration_endpoint
            ),
            include_timestep=arguments.include_timestep,
        )
        for split, indices in split_indices.items()
    }
    device = choose_device(arguments.device, torch)
    generator = torch.Generator()
    generator.manual_seed(arguments.random_seed)
    loader_arguments = {
        "batch_size": arguments.batch_size,
        "num_workers": arguments.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": arguments.num_workers > 0,
        "collate_fn": collate_cached_diagrams,
    }
    loaders = {
        "train": dependencies.DataLoader(
            datasets["train"],
            shuffle=True,
            generator=generator,
            **loader_arguments,
        ),
        "validation": dependencies.DataLoader(
            datasets["validation"],
            shuffle=False,
            **loader_arguments,
        ),
        "test": dependencies.DataLoader(
            datasets["test"],
            shuffle=False,
            **loader_arguments,
        ),
    }

    feature_names = graph_feature_names(
        exclude_filtration_endpoint=(
            arguments.exclude_filtration_endpoint
        ),
        include_timestep=arguments.include_timestep,
    )
    model, branch_settings = make_model(
        image_bounds=image_bounds,
        image_size=arguments.image_size,
        sigma_multiplier=arguments.sigma_multiplier,
        number_of_graph_features=len(feature_names),
        hidden_dimension=arguments.hidden_dimension,
        dropout=arguments.dropout,
        weight_type=arguments.weight_type,
        weight_hidden_dimension=arguments.weight_hidden_dimension,
        dependencies=dependencies,
    )
    model = model.to(device)
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
    number_of_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    print(f"Device: {device}")
    print(f"Parameters: {number_of_parameters:,}")
    print(
        "Diagram bounds and all normalization statistics use the training "
        "split only."
    )

    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_validation_rmse = math.inf
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, arguments.epochs + 1):
        model.train()
        total_squared_error = 0.0
        number_examples = 0
        for (
            diagrams,
            masks,
            graph_attributes,
            target,
            _,
        ) in loaders["train"]:
            diagrams = move_mapping_to_device(diagrams, device)
            masks = move_mapping_to_device(masks, device)
            graph_attributes = graph_attributes.to(
                device,
                non_blocking=True,
            )
            target = target.to(device, non_blocking=True).view(-1)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                diagrams,
                graph_attributes,
                mask=masks,
            )
            loss = dependencies.functional.mse_loss(
                prediction,
                target,
            )
            loss.backward()
            if arguments.gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    arguments.gradient_clip,
                )
            optimizer.step()
            total_squared_error += float(loss.item()) * target.numel()
            number_examples += target.numel()

        training_mse_normalized = total_squared_error / number_examples
        validation_rows = evaluate_model(
            model,
            loaders["validation"],
            records=records,
            device=device,
            normalization=normalization,
            dependencies=dependencies,
        )
        validation_metrics = regression_metrics(validation_rows)
        scheduler.step(validation_metrics.rmse)
        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "training_mse_normalized": training_mse_normalized,
                "validation_mae": validation_metrics.mae,
                "validation_rmse": validation_metrics.rmse,
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
                f"Epoch {epoch:4d} | "
                f"train normalized MSE {training_mse_normalized:.6f} | "
                f"val RMSE {validation_metrics.rmse:.6f} | "
                f"val R2 {validation_metrics.r2:.6f}"
            )
        if epochs_without_improvement >= arguments.patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch was {best_epoch}."
            )
            break

    if best_state is None:
        raise PerslayExperimentError(
            "Training did not produce a valid model state."
        )
    model.load_state_dict(best_state)

    final_metrics: dict[str, Metrics] = {}
    mean_baselines: dict[str, Metrics] = {}
    endpoint_baselines: dict[str, Metrics] = {}
    prediction_rows: list[dict[str, Any]] = []
    endpoint_intercept, endpoint_slope = fit_filtration_endpoint_baseline(
        records,
        split_indices["train"],
    )
    for split in VALID_SPLITS:
        rows = evaluate_model(
            model,
            loaders[split],
            records=records,
            device=device,
            normalization=normalization,
            dependencies=dependencies,
        )
        final_metrics[split] = regression_metrics(rows)
        mean_baselines[split] = mean_baseline_metrics(
            records,
            split_indices[split],
            training_mean=normalization.target_mean,
        )
        endpoint_baselines[split] = (
            filtration_endpoint_baseline_metrics(
                records,
                split_indices[split],
                intercept=endpoint_intercept,
                slope=endpoint_slope,
            )
        )
        for record_id, truth, prediction in rows:
            record = records[record_id]
            prediction_rows.append(
                {
                    "graph_id": record.graph_id,
                    "seed": record.seed,
                    "timestep": record.timestep,
                    "split": split,
                    "target": truth,
                    "prediction": prediction,
                    "error": prediction - truth,
                }
            )

    learned_parameters = learned_branch_parameters(model, dependencies)
    checkpoint = {
        "script_version": SCRIPT_VERSION,
        "model_type": "multi_branch_gaussian_perslay",
        "model_state_dict": best_state,
        "target_column": arguments.target_column,
        "dimensions": dimensions,
        "branch_settings": branch_settings,
        "learned_branch_parameters": learned_parameters,
        "graph_feature_names": feature_names,
        "normalization": asdict(normalization),
        "hidden_dimension": arguments.hidden_dimension,
        "dropout": arguments.dropout,
        "weight_type": arguments.weight_type,
        "weight_hidden_dimension": arguments.weight_hidden_dimension,
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, output_directory / "best_model.pt")

    write_csv(
        output_directory / "training_history.csv",
        (
            "epoch",
            "learning_rate",
            "training_mse_normalized",
            "validation_mae",
            "validation_rmse",
            "validation_r2",
        ),
        history,
    )
    write_csv(
        output_directory / "split_assignments.csv",
        ("graph_id", "seed", "timestep", "split"),
        (
            {
                "graph_id": record.graph_id,
                "seed": record.seed,
                "timestep": record.timestep,
                "split": assignments[record.index],
            }
            for record in records
        ),
    )
    write_csv(
        output_directory / "predictions.csv",
        (
            "graph_id",
            "seed",
            "timestep",
            "split",
            "target",
            "prediction",
            "error",
        ),
        sorted(
            prediction_rows,
            key=lambda row: (
                str(row["split"]),
                int(row["seed"]),
                int(row["timestep"]),
            ),
        ),
    )

    result = {
        "script_version": SCRIPT_VERSION,
        "model_type": "multi_branch_gaussian_perslay",
        "target_column": arguments.target_column,
        "index_path": str(index_path),
        "split_source": str(split_path),
        "best_epoch": best_epoch,
        "device": str(device),
        "number_of_parameters": number_of_parameters,
        "dimensions": list(dimensions),
        "diagram_coordinates": ["birth_value", "death_value"],
        "uses_pairing_graph_edges": False,
        "uses_representative_sizes": False,
        "essential_policy": arguments.essential_policy,
        "minimum_persistence": arguments.minimum_persistence,
        "image_size": list(arguments.image_size),
        "image_bounds_training_only": image_bounds,
        "branch_settings": branch_settings,
        "learned_branch_parameters": learned_parameters,
        "weight_type": arguments.weight_type,
        "graph_feature_names": list(feature_names),
        "include_timestep": arguments.include_timestep,
        "normalization": asdict(normalization),
        "seed_group_counts": split_seed_counts,
        "sample_counts": {
            split: len(indices)
            for split, indices in split_indices.items()
        },
        "diagram_point_counts": point_counts,
        "model_metrics": {
            split: json_safe_metrics(metrics)
            for split, metrics in final_metrics.items()
        },
        "training_mean_baseline": {
            split: json_safe_metrics(metrics)
            for split, metrics in mean_baselines.items()
        },
        "filtration_endpoint_linear_baseline": {
            "intercept": endpoint_intercept,
            "slope": endpoint_slope,
            "metrics": {
                split: json_safe_metrics(metrics)
                for split, metrics in endpoint_baselines.items()
            },
        },
        "scientific_note": (
            "This is a persistence-diagram-only topological baseline: each "
            "branch receives birth/death coordinates but no graph edges, "
            "representative sizes, or essential indicator. Filtration endpoint "
            "is included by default solely to match the graph-level covariate "
            "used by the GNN and CNN; timestep remains excluded by default."
        ),
    }
    with (output_directory / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    print("Final metrics:")
    for split in VALID_SPLITS:
        metrics = final_metrics[split]
        mean_baseline = mean_baselines[split]
        endpoint_baseline = endpoint_baselines[split]
        print(
            f"  {split:10s} MAE={metrics.mae:.6f} "
            f"RMSE={metrics.rmse:.6f} R2={metrics.r2:.6f} "
            f"(mean baseline={mean_baseline.rmse:.6f}, "
            f"endpoint-linear baseline={endpoint_baseline.rmse:.6f})"
        )
    print(f"Results written to {output_directory}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argument_parser()
    arguments = parser.parse_args(argv)
    try:
        run(arguments)
    except PerslayExperimentError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
