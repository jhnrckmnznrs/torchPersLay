# torchPersLay

`torchPersLay` is a native [PyTorch](https://pytorch.org/) implementation of **PersLay**, a neural-network layer for persistence diagrams in topological data analysis. The package is published on PyPI as `torchPersLay`; the Python import package is `perslay`.

This repository focuses on PersLay-style persistence-image features with learnable point weights and Gaussian bandwidths. Earlier DeepSet experiment code has been removed to keep the package focused on the successful PersLay implementation.

## Installation

Install the released package from PyPI:

```bash
pip install torchperslay
```

Install the local repository in editable mode:

```bash
git clone https://github.com/jhnrckmnznrs/torchPersLay.git
cd torchPersLay
pip install -e .
```

To run the experiment scripts in this repository, also install the experiment extra or the repository requirements:

```bash
pip install -e ".[experiments]"
# or
pip install -r requirements.txt
```

## Quick start

The example below builds a PersLay regressor for a small batch of padded persistence diagrams. Diagrams are expected as `(birth, death)` pairs with shape `[batch, num_points, 2]`.

```python
import torch

from perslay import (
    GaussianPerslayPhi,
    NormalizedLearnablePowerPerslayWeight,
    Perslay,
)
from perslay.models import FlattenRho

image_size = [10, 10]
image_bnds = [[0.0, 1.0], [0.0, 1.0]]  # birth bounds, persistence bounds

weight = NormalizedLearnablePowerPerslayWeight(image_bnds=image_bnds)
phi = GaussianPerslayPhi(
    image_size=image_size,
    image_bnds=image_bnds,
    sigma_x=0.1,
    sigma_y=0.1,
    normalize=False,
)

perslay = Perslay(
    weight=weight,
    phi=phi,
    perm_op=torch.sum,
    rho=FlattenRho(),
)

# Two padded diagrams with three points each.
diagrams = torch.tensor(
    [
        [[0.0, 0.4], [0.2, 0.8], [0.0, 0.0]],
        [[0.1, 0.5], [0.3, 0.9], [0.4, 0.95]],
    ],
    dtype=torch.float32,
)
mask = torch.tensor(
    [
        [True, True, False],
        [True, True, True],
    ]
)

features = perslay(diagrams, mask=mask)
print(features.shape)  # torch.Size([2, 100])
```

## Regression models

For most regression experiments, use the provided model wrappers:

```python
from perslay.models import PerslayRegressor, MultiPerslayRegressor

model = PerslayRegressor(
    image_size=[20, 20],
    image_bnds=[[0.0, 1.0], [0.0, 1.0]],
    sigma_x=0.05,
    sigma_y=0.05,
    hidden_dim=128,
    weight_type="normalized_learnable_power",
)
```

Available `weight_type` values are:

- `constant`
- `learnable_power`
- `normalized_learnable_power`
- `mlp`

`MultiPerslayRegressor` supports multiple homology dimensions or multiple diagram sources per sample. See `experiments/configs/h0_h2_regression.yaml` for an example configuration.

## Repository layout

```text
src/perslay/
  data.py      Dataset and collate utilities for persistence-diagram CSV files
  layers.py    PersLay layers, feature maps, and point-weight functions
  models.py    Single-branch and multi-branch PersLay regressors

experiments/
  train_regression.py       Config-driven regression training script
  configs/degree2_regression.yaml
  configs/h0_h2_regression.yaml
```

## Running experiments

Prepare persistence diagram CSV files with columns similar to:

```csv
birth,death
0.1,0.4
0.2,0.8
```

Prepare a target CSV with:

```csv
filename,target
sample_001.csv,3.14
```

Then run:

```bash
python experiments/train_regression.py --config experiments/configs/degree2_regression.yaml
```

For multi-diagram input, use:

```bash
python experiments/train_regression.py --config experiments/configs/h0_h2_regression.yaml
```

Training writes checkpoints and resolved configs under the configured `output.run_dir`.

## Citation

If you use this package in research, please cite the original PersLay paper:

> Mathieu Carrière, Frédéric Chazal, Yuichi Ike, Théo Lacombe, Martin Royer, and Yuhei Umeda. **PersLay: A Neural Network Layer for Persistence Diagrams and New Graph Topological Signatures.** AISTATS 2020, PMLR 108:2786–2796.

## License

This project is distributed under the MIT License. See `LICENSE` for details.
