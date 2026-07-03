import csv
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Subset


class PersistenceDiagramDataset(Dataset):
    """
    Dataset for persistence diagrams stored as CSV files.

    Each diagram CSV should have headers:

        birth,death
        0.1,0.4
        0.2,0.8

    The target CSV should have:

        filename,target
        sample_001.csv,3.14
    """

    def __init__(self, diagram_dir, targets_csv):
        self.diagram_dir = Path(diagram_dir)

        self.targets = {}
        with open(targets_csv, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.targets[row["filename"]] = float(row["target"])

        self.files = sorted(self.diagram_dir.glob("*.csv"))
        self.files = [path for path in self.files if path.name in self.targets]

        if len(self.files) == 0:
            raise ValueError("No diagram CSV files with matching targets were found.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        diagram = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1,
            usecols=(0, 1),
            dtype=np.float32,
        )

        if diagram.ndim == 1:
            diagram = diagram[None, :]

        diagram = torch.from_numpy(diagram)
        target = torch.tensor(self.targets[path.name], dtype=torch.float32)

        return diagram, target, path.name


class MultiPersistenceDiagramDataset(Dataset):
    """
    Dataset for multiple persistence diagrams per sample.

    Example:
        diagram_dirs = {
            "h0": "data/h0_structures90",
            "h2": "data/h2_structures90",
        }

    Each directory must contain CSV files with the same filenames.
    """

    def __init__(self, diagram_dirs, targets_csv):
        self.diagram_dirs = {name: Path(path) for name, path in diagram_dirs.items()}

        self.targets = {}
        with open(targets_csv, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.targets[row["filename"]] = float(row["target"])

        file_sets = []
        for diagram_dir in self.diagram_dirs.values():
            file_sets.append({path.name for path in diagram_dir.glob("*.csv")})

        common_files = set(self.targets.keys())
        for file_set in file_sets:
            common_files = common_files.intersection(file_set)

        self.files = sorted(common_files)

        if len(self.files) == 0:
            raise ValueError(
                "No matching diagram CSV files were found across all diagram directories."
            )

    def __len__(self):
        return len(self.files)

    def _load_diagram(self, path):
        diagram = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1,
            usecols=(0, 1),
            dtype=np.float32,
        )

        if diagram.ndim == 1:
            diagram = diagram[None, :]

        # Important for H0: remove essential classes with death = inf.
        diagram = diagram[np.isfinite(diagram).all(axis=1)]

        # Optional but usually sensible: remove non-positive persistence.
        persistence = diagram[:, 1] - diagram[:, 0]
        diagram = diagram[persistence > 0.0]

        if diagram.shape[0] == 0:
            diagram = np.zeros((0, 2), dtype=np.float32)

        return torch.from_numpy(diagram.astype(np.float32))

    def __getitem__(self, idx):
        filename = self.files[idx]

        diagrams = {
            name: self._load_diagram(diagram_dir / filename)
            for name, diagram_dir in self.diagram_dirs.items()
        }

        target = torch.tensor(self.targets[filename], dtype=torch.float32)

        return diagrams, target, filename


def collate_multi_persistence_diagrams(batch):
    """
    Collate function for MultiPersistenceDiagramDataset.

    Returns:
        diagrams_padded: dict[str, Tensor[B, N_max, 2]]
        masks:           dict[str, Tensor[B, N_max]]
        targets:         Tensor[B]
        filenames:       tuple[str]
    """
    diagrams_dicts, targets, filenames = zip(*batch)

    diagram_names = diagrams_dicts[0].keys()

    padded_by_name = {}
    mask_by_name = {}

    for name in diagram_names:
        diagrams = [d[name] for d in diagrams_dicts]
        lengths = torch.tensor([d.shape[0] for d in diagrams], dtype=torch.long)

        max_len = int(lengths.max().item())

        if max_len == 0:
            diagrams_padded = torch.zeros(
                len(diagrams),
                1,
                2,
                dtype=torch.float32,
            )
            mask = torch.zeros(
                len(diagrams),
                1,
                dtype=torch.bool,
            )
        else:
            diagrams_padded = pad_sequence(
                diagrams,
                batch_first=True,
                padding_value=0.0,
            )

            max_len = diagrams_padded.shape[1]
            mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

        padded_by_name[name] = diagrams_padded
        mask_by_name[name] = mask

    targets = torch.stack(targets)

    return padded_by_name, mask_by_name, targets, filenames


def collate_persistence_diagrams(batch):
    """
    Pads variable-size persistence diagrams inside one batch.

    Returns:
        diagrams_padded: [B, N_max, 2]
        mask:            [B, N_max]
        targets:         [B]
        filenames:       tuple[str]
    """
    diagrams, targets, filenames = zip(*batch)

    lengths = torch.tensor([d.shape[0] for d in diagrams], dtype=torch.long)

    diagrams_padded = pad_sequence(
        diagrams,
        batch_first=True,
        padding_value=0.0,
    )

    max_len = diagrams_padded.shape[1]
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    targets = torch.stack(targets)

    return diagrams_padded, mask, targets, filenames


def compute_diagram_bounds(
    dataset,
    padding_fraction=0.05,
    persistence_min=0.0,
    diagram_key=None,
):
    """
    Compute global image bounds from a dataset or torch.utils.data.Subset.

    Bounds are computed after converting diagrams from (birth, death)
    to (birth, persistence).

    Returns:
        image_bnds = [[birth_min, birth_max], [persistence_min, persistence_max]]
    """

    birth_values = []
    persistence_values = []

    # Handle both a full Dataset and a Subset from random_split.
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = dataset.indices
    else:
        base_dataset = dataset
        indices = range(len(dataset))

    for idx in indices:
        diagram, _, _ = base_dataset[idx]

        if isinstance(diagram, dict):
            if diagram_key is None:
                raise ValueError(
                    "diagram_key must be provided for multi-diagram datasets."
                )
            diagram = diagram[diagram_key]

        if diagram.shape[0] == 0:
            continue

        birth = diagram[:, 0]
        death = diagram[:, 1]
        persistence = death - birth

        birth_values.append(birth)
        persistence_values.append(persistence)

    if len(birth_values) == 0:
        raise ValueError(
            "Cannot compute bounds because all selected diagrams are empty."
        )

    birth_values = torch.cat(birth_values)
    persistence_values = torch.cat(persistence_values)

    birth_min = birth_values.min().item()
    birth_max = birth_values.max().item()

    pers_min = max(persistence_min, persistence_values.min().item())
    pers_max = persistence_values.max().item()

    birth_range = birth_max - birth_min
    pers_range = pers_max - pers_min

    # Avoid zero-width bounds if all values are identical.
    if birth_range == 0.0:
        birth_range = 1.0

    if pers_range == 0.0:
        pers_range = 1.0

    birth_pad = padding_fraction * birth_range
    pers_pad = padding_fraction * pers_range

    image_bnds = [
        [birth_min - birth_pad, birth_max + birth_pad],
        [pers_min, pers_max + pers_pad],
    ]

    return image_bnds


def compute_sigma_from_bounds(image_bnds, image_size, multiplier=1.0):
    """
    Choose initial anisotropic Gaussian bandwidths from image bounds.

    The default choice is:

        sigma_x = multiplier * pixel_width
        sigma_y = multiplier * pixel_height

    where pixel_width and pixel_height are determined by the image bounds
    and image resolution.

    Args:
        image_bnds: [[birth_min, birth_max], [pers_min, pers_max]]
        image_size: [nx, ny]
        multiplier: how many pixel widths/heights to use as the initial sigma

    Returns:
        sigma_x, sigma_y
    """
    birth_width = image_bnds[0][1] - image_bnds[0][0]
    pers_width = image_bnds[1][1] - image_bnds[1][0]

    sigma_x = multiplier * birth_width / image_size[0]
    sigma_y = multiplier * pers_width / image_size[1]

    return float(sigma_x), float(sigma_y)
