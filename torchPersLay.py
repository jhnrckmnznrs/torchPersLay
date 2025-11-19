import math
import gudhi.representations as gdr
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class PowerPerslayWeight(nn.Module):
    """
    Differentiable weight function for persistence diagram points.
    Weight = constant * (distance to diagonal) ** power
    """

    def __init__(self, constant, power):
        """
        Parameters:
            constant (float): trainable multiplier
            power (float): exponent applied to distance to diagonal
        """
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(float(constant)))
        self.power = power

    def forward(self, diagrams):
        """
        Parameters:
            diagrams: Tensor of shape [B, N, 2], padded if necessary

        Returns:
            weight: Tensor of shape [B, N] with pointwise weights
        """
        # distance to diagonal = |death - birth|
        distance = torch.abs(diagrams[..., 1] - diagrams[..., 0])
        weight = self.constant * torch.pow(distance, self.power)
        return weight

class GridPerslayWeight(nn.Module):
    """
    This is a class for computing a differentiable weight function for persistence diagram points.
    This function is defined from an array that contains its values on a 2D grid.
    """

    def __init__(self, grid, grid_bnds, **kwargs):
        """
        Constructor for the GridPerslayWeight class.

        Parameters:
            grid (n x n numpy array): grid of values.
            grid_bnds (2 x 2 numpy array): boundaries of the grid, of the form [[min_x, max_x], [min_y, max_y]].
        """
        super().__init__()
        self.grid = nn.Parameter(torch.tensor(grid, dtype=torch.float32))
        self.grid_bnds = grid_bnds

    def forward(self, diagrams):
        """
        Apply GridPerslayWeight on a list of persistence diagrams.

        Parameters:
            diagrams (list of n tensors of shape (num_points x 2)): list containing n persistence diagrams.

        Returns:
            weight (list of n tensors): list containing the weights of the points in the n persistence diagrams.
        """class GridPerslayWeight(nn.Module):
            """
            This is a class for computing a differentiable weight function for persistence diagram points.
            This function is defined from an array that contains its values on a 2D grid.
            """

            def __init__(self, grid, grid_bnds, **kwargs):
                """
                Constructor for the GridPerslayWeight class.

                Parameters:
                    grid (n x n numpy array): grid of values.
                    grid_bnds (2 x 2 numpy array): boundaries of the grid, of the form [[min_x, max_x], [min_y, max_y]].
                """
                super().__init__()
                self.grid = nn.Parameter(torch.tensor(grid, dtype=torch.float32))
                self.grid_bnds = grid_bnds

            def forward(self, diagrams):
                """
                Apply GridPerslayWeight on a list of persistence diagrams.

                Parameters:
                    diagrams (list of n tensors of shape (num_points x 2)): list containing n persistence diagrams.

                Returns:
                    weight (list of n tensors): list containing the weights of the points in the n persistence diagrams.
                """
                # grid = torch.from_numpy(self.grid)
                grid_shape = self.grid.shape
                weights = []

                for diagram in diagrams:
                    indices = []
                    for dim in range(2):
                        m, M = self.grid_bnds[dim]
                        coords = diagram[:, dim]

                        # Match TensorFlow scaling (uses grid_shape, NOT grid_shape - 1)
                        ids = grid_shape[dim] * (coords - m) / (M - m)

                        # Match TensorFlow truncation (TensorFlow cast → truncates toward zero)
                        ids = ids.to(torch.int32)

                        indices.append(ids)

                    # No clamping → out-of-range indices behave like in TF
                    weight = self.grid[indices[0], indices[1]]
                    weights.append(weight)

                return torch.stack(weights, dim=0)
        # grid = torch.from_numpy(self.grid)
        grid_shape = self.grid.shape
        weights = []

        for diagram in diagrams:
            indices = []
            for dim in range(2):
                m, M = self.grid_bnds[dim]
                coords = diagram[:, dim]

                # Match TensorFlow scaling (uses grid_shape, NOT grid_shape - 1)
                ids = grid_shape[dim] * (coords - m) / (M - m)

                # Match TensorFlow truncation (TensorFlow cast → truncates toward zero)
                ids = ids.to(torch.int32)

                indices.append(ids)

            # No clamping → out-of-range indices behave like in TF
            weight = self.grid[indices[0], indices[1]]
            weights.append(weight)

        return torch.stack(weights, dim=0)

class GaussianMixturePerslayWeight(nn.Module):
    """
    This is a class for computing a differentiable weight function for persistence diagram points.
    This function is defined from a mixture of Gaussian functions.
    """

    def __init__(self, gaussians, **kwargs):
        """
        Constructor for the GaussianMixturePerslayWeight class.

        Parameters:
            gaussians (4 x n numpy array): parameters of the n Gaussian functions, of the form
                transpose([[mu_x^1, mu_y^1, sigma_x^1, sigma_y^1], ..., [mu_x^n, mu_y^n, sigma_x^n, sigma_y^n]]).
        """
        super().__init__()
        self.W = nn.Parameter(torch.tensor(gaussians, dtype=torch.float32))

    def forward(self, diagrams):
        """
        Apply GaussianMixturePerslayWeight on a list of persistence diagrams.

        Parameters:
            diagrams (list of n tensors of shape (num_points x 2)): list containing n persistence diagrams.

        Returns:
            weight (list of n tensors): list containing the weights of the points in the n persistence diagrams.
        """
        means = self.W[:2, :].unsqueeze(0).unsqueeze(0)  # (1, 1, 2, n_gaussians)
        variances = self.W[2:, :].unsqueeze(0).unsqueeze(0)  # (1, 1, 2, n_gaussians)

        weights = []
        for diagram in diagrams:
            # diagram: (num_points, 2)
            diagram_expanded = diagram.unsqueeze(-1)  # (num_points, 2, 1)
            dists = torch.square(diagram_expanded - means[0, 0]) / torch.square(
                variances[0, 0]
            )
            weight = torch.sum(torch.exp(-torch.sum(dists, dim=1)), dim=1)
            weights.append(weight)

        return torch.stack(weights, dim=0)

class GaussianPerslayPhi(nn.Module):
    """
    This is a class for computing a transformation function for persistence diagram points.
    This function turns persistence diagram points into 2D Gaussian functions centered on the points,
    that are then evaluated on a regular 2D grid.
    """

    def __init__(self, image_size, image_bnds, variance, **kwargs):
        """
        Constructor for the GaussianPerslayPhi class.

        Parameters:
            image_size (int numpy array): number of grid elements on each grid axis, of the form [n_x, n_y].
            image_bnds (2 x 2 numpy array): boundaries of the grid, of the form [[min_x, max_x], [min_y, max_y]].
            variance (float): variance of the Gaussian functions.
        """
        super().__init__()
        self.image_size = image_size
        self.image_bnds = image_bnds
        self.log_variance = nn.Parameter(torch.log(torch.tensor(variance)))
        # self.variance = nn.Parameter(torch.tensor(variance, dtype=torch.float32))

        step = [
            (self.image_bnds[i][1] - self.image_bnds[i][0]) / self.image_size[i]
            for i in range(2)
        ]
        coords = [
            torch.arange(self.image_bnds[i][0], self.image_bnds[i][1], step[i])
            for i in range(2)
        ]
        self.M = torch.meshgrid(*coords, indexing="xy")
        self.mu = torch.stack(self.M, dim=0)  # (2, image_size[0], image_size[1])

    def forward(self, diagrams):
        """
        Apply GaussianPerslayPhi on a list of persistence diagrams.

        Parameters:
            diagrams (list of n tensors of shape (num_points x 2)): list containing n persistence diagrams.

        Returns:
            output (list of n tensors of shape (num_points x image_size[0] x image_size[1] x 1)):
                list containing the evaluations on the 2D grid of the 2D Gaussian functions.
            output_shape (tuple): shape of the output tensor.
        """

        variance = torch.exp(self.log_variance)

        # Transform diagram: (birth, death) -> (birth, persistence)
        birth = diagrams[..., 0:1]  # [B, N, 1]
        lifetime = diagrams[..., 1:2] - birth  # [B, N, 1]
        diagrams_d = torch.cat([birth, lifetime], dim=-1)  # [B, N, 2]

        # Expand dimensions for broadcasting
        diagrams_d = diagrams_d.unsqueeze(-1).unsqueeze(-1)  # (num_points, 2, 1, 1)

        mu = self.mu.unsqueeze(0).unsqueeze(0).squeeze(3)

        dists = torch.square(diagrams_d - mu) / (2 * torch.square(variance))
        gauss = torch.exp(-dists.sum(dim=2)) / (2 * math.pi * torch.square(variance))
        output = gauss.unsqueeze(-1)  # (num_points, image_size[0], image_size[1], 1)

        output_shape = self.M[0].shape + tuple([1])
        return output, output_shape

class TentPerslayPhi(nn.Module):
    """
    This is a class for computing a transformation function for persistence diagram points.
    This function turns persistence diagram points into 1D tent functions centered on the points,
    that are then evaluated on a regular 1D grid.
    """

    def __init__(self, samples, **kwargs):
        """
        Constructor for the TentPerslayPhi class.

        Parameters:
            samples (float numpy array): grid elements on which to evaluate the tent functions, of the form [x_1, ..., x_n].
        """
        super().__init__()
        self.samples = nn.Parameter(torch.tensor(samples, dtype=torch.float32))

    def forward(self, diagrams):
        """
        Apply TentPerslayPhi on a list of persistence diagrams.

        Parameters:
            diagrams (list of n tensors of shape (num_points x 2)): list containing n persistence diagrams.

        Returns:
            output (list of n tensors of shape (num_points x num_samples)):
                list containing the evaluations on the 1D grid of the 1D tent functions.
            output_shape (tuple): shape of the output tensor.
        """
        samples_d = self.samples.unsqueeze(0).unsqueeze(0)  # (1, 1, num_samples)

        outputs = []
        for diagram in diagrams:
            xs = diagram[:, 0:1]  # (num_points, 1)
            ys = diagram[:, 1:2]  # (num_points, 1)
            output = torch.maximum(
                0.5 * (ys - xs) - torch.abs(samples_d - 0.5 * (ys + xs)),
                torch.tensor(0.0),
            )
            outputs.append(output.squeeze(1))  # (num_points, num_samples)

        output_shape = self.samples.shape
        return torch.stack(outputs, dim=0), output_shape

class FlatPerslayPhi(nn.Module):
    """
    This is a class for computing a transformation function for persistence diagram points.
    This function turns persistence diagram points into 1D constant functions that are evaluated on a regular 1D grid.
    """

    def __init__(self, samples, theta, **kwargs):
        """
        Constructor for the FlatPerslayPhi class.

        Parameters:
            samples (float numpy array): grid elements on which to evaluate the constant functions, of the form [x_1, ..., x_n].
            theta (float): sigmoid parameter used to approximate the constant function with a differentiable sigmoid function.
        """
        super().__init__()
        self.samples = nn.Parameter(torch.tensor(samples, dtype=torch.float32))
        self.theta = nn.Parameter(torch.tensor(theta, dtype=torch.float32))

    def forward(self, diagrams):
        """
        Apply FlatPerslayPhi on a list of persistence diagrams.

        Parameters:
            diagrams (list of n tensors of shape (num_points x 2)): list containing n persistence diagrams.

        Returns:
            output (list of n tensors of shape (num_points x num_samples)):
                list containing the evaluations on the 1D grid of the 1D constant functions.
            output_shape (tuple): shape of the output tensor.
        """
        samples_d = self.samples.unsqueeze(0).unsqueeze(0)  # (1, 1, num_samples)

        outputs = []
        for diagram in diagrams:
            xs = diagram[:, 0:1]  # (num_points, 1)
            ys = diagram[:, 1:2]  # (num_points, 1)
            output = 1.0 / (
                1.0
                + torch.exp(
                    -self.theta
                    * (0.5 * (ys - xs) - torch.abs(samples_d - 0.5 * (ys + xs)))
                )
            )
            outputs.append(output.squeeze(1))  # (num_points, num_samples)

        output_shape = self.samples.shape
        return torch.stack(outputs, dim=0), output_shape

class Perslay(nn.Module):
    """
    Vectorizes persistence diagrams in a differentiable way, implementing PersLay.
    Reference: http://proceedings.mlr.press/v108/carriere20a.html
    """

    def __init__(self, weight, phi, perm_op, rho):
        """
        Parameters:
            weight: callable that computes weights for persistence diagram points
            phi: callable that transforms persistence diagram points
            perm_op: permutation-invariant function (sum, mean, max, min) or "topk{number}"
            rho: postprocessing function (nn.Module)
        """
        super().__init__()
        self.weight = weight
        self.phi = phi
        self.perm_op = perm_op
        self.rho = rho

    def forward(self, diagrams):
        """
        Parameters:
            diagrams: tensor of shape [B, N, 2], padded if necessary

        Returns:
            vector: tensor of shape [B, output_dim] representing PersLay embeddings
        """
        # phi should return (vector, dim) like in TF version
        # vector: [B, N, ...] or [B, N, grid_x, grid_y, 1]
        vector, dim = self.phi(diagrams)
        weightTensor = self.weight(diagrams)
        # weight broadcasting: match vector dims from 2 onwards
        for _ in range(vector.ndim - weightTensor.ndim):
            weightTensor = weightTensor.unsqueeze(-1)
        vector = vector * weightTensor

        # Apply permutation invariant operation
        permop = self.perm_op
        if isinstance(permop, str) and permop[:3] == "top":
            k = int(permop[3:])
            # Flatten spatial dimensions if necessary
            vector = vector.view(
                vector.shape[0], vector.shape[1], -1
            )  # [B, N, features]
            # Take top-k along the point axis
            topk_vals, _ = torch.topk(vector.transpose(1, 2), k=k, dim=2)
            vector = topk_vals.reshape(vector.shape[0], -1)
        else:
            # Apply perm_op along points axis (axis=1)
            vector = permop(vector, dim=1)

        # Apply postprocessing function rho
        vector = self.rho(vector)

        return vector
