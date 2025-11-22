<div align="justify">

# torchPersLay

torchPersLay is a <a href="https://pytorch.org/">PyTorch</a> implementation of <a href = "http://proceedings.mlr.press/v108/carriere20a/carriere20a.pdf">PersLay</a>, a neural network layer for processing persistence diagrams in topological data analysis (TDA). The original PersLay architecture is available in <a href = "https://gudhi.inria.fr/python/latest/representations_tflow_itf_ref.html">GUDHI</a>, but only in <a href="https://www.tensorflow.org/">TensorFlow</a>. This project provides a native, modular, and extensible PyTorch version suitable for modern deep-learning pipelines.

## Citation

If you use this neural network layer in your research, please cite the original paper:

**PersLay: A Neural Network Layer for Persistence Diagrams and New Graph Topological Signatures**  
Mathieu Carrière, Frédéric Chazal, Yuichi Ike, Théo Lacombe, Martin Royer, Yuhei Umeda  
Proceedings of the Twenty-Third International Conference on Artificial Intelligence and Statistics (AISTATS),  
PMLR 108:2786–2796, 2020.

## Installation

Package is not yet available in the Python Package Index. You may copy torchPersLay.py in your present working directory by downloading the file itself or cloning this repository:

```
git clone https://github.com/jhnrckmnznrs/torchPersLay.git
```

## Example Usage

This example usage is concerned with a simple regression model that uses PersLay as a single hidden layer.

### Import Packages

Import necessary packages. Ensure that all packages are installed in your system.

```
from torchPersLay import *

import gudhi.representations as gdr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
```

### Define PersLay Layers

Choose the PersLay layer that you want. Consult the official PersLay documentation for the possible options.

```
constant = 1.0
power = 0.0

weight = PowerPerslayWeight(constant=constant, power=power)
rho = nn.Identity()

image_size = (5, 5)
image_bnds = ((-0.5, 1.5), (-0.5, 1.5))
variance = 0.1

phi = GaussianPerslayPhi(
    image_size=image_size,
    image_bnds=image_bnds,
    variance=variance,
)

perm_op = torch.sum

perslay = Perslay(weight=weight, phi=phi, perm_op=perm_op, rho=rho)
```

### Import Data

This is an example data. Import your own data.

```
diagrams = [
    np.array([[0.0, 4.0], [1.0, 2.0], [3.0, 8.0], [6.0, 8.0]]),
    np.array([[1.0, 3.0], [2.0, 2.5], [4.0, 7.0], [7.0, 7.5]]),
]

scaler = gdr.DiagramScaler(use=True, scalers=[([0, 1], MinMaxScaler())])
diagrams = scaler.fit_transform(diagrams)
diagrams = torch.from_numpy(np.array(diagrams, dtype=np.float32))

y = torch.tensor([[1.0], [3.0]])
```

### Define Model

This is the creation of the model. This is a simple example. You may add other layers as usual and/or use known architectures to concatenate feature vectors.

```
class PersLayRegressor(nn.Module):
    def __init__(self, perslay, image_size=(5, 5)):
        super().__init__()
        self.perslay = perslay
        feature_dim = image_size[0] * image_size[1]

        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, diagrams):
        x = self.perslay(diagrams)       # [B, 5, 5, 1]
        x = x.view(x.shape[0], -1)       # [B, 25]
        return self.regressor(x)

model = PersLayRegressor(perslay)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

### Train Model

This is the model training for PyTorch models.

```
for epoch in range(100):
    optimizer.zero_grad()
    preds = model(diagrams)
    loss = criterion(preds, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}, Loss = {loss.item():.6f}")
```

### (Optional) Inspect Learned Parameters

You may optionally see the learned parameters.

```
for name, param in perslay.named_parameters():
    print(name, param.data)
```
