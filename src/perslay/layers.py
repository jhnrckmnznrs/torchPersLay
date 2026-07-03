import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ConstantPerslayWeight(nn.Module):
    """
    Constant point weight.

    Every persistence point receives weight 1.
    This is useful as a baseline to test whether persistence-based weighting helps.
    """

    def forward(self, diagrams):
        return torch.ones_like(diagrams[..., 0])


class LearnablePowerPerslayWeight(nn.Module):
    """
    Learnable persistence-based weight function.

    Weight = scale * (persistence + eps) ** power

    Both scale and power are learned, while constrained to be positive.
    """

    def __init__(self, init_scale=1.0, init_power=1.0, eps=1e-6):
        super().__init__()
        self.raw_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.raw_power = nn.Parameter(torch.tensor(float(init_power)))
        self.eps = eps

    def forward(self, diagrams):
        birth = diagrams[..., 0]
        death = diagrams[..., 1]
        persistence = (death - birth).clamp_min(self.eps)

        scale = F.softplus(self.raw_scale)
        power = F.softplus(self.raw_power)

        return scale * torch.pow(persistence, power)


class MLPPerslayWeight(nn.Module):
    """
    Learnable pointwise weight function.

    Takes each persistence point as (birth, persistence) and returns a
    positive scalar weight.
    """

    def __init__(self, image_bnds, hidden_dim=16, eps=1e-6):
        super().__init__()

        self.eps = eps

        mins = torch.tensor(
            [image_bnds[0][0], image_bnds[1][0]],
            dtype=torch.float32,
        )
        maxs = torch.tensor(
            [image_bnds[0][1], image_bnds[1][1]],
            dtype=torch.float32,
        )

        self.register_buffer("mins", mins)
        self.register_buffer("ranges", (maxs - mins).clamp_min(eps))

        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, diagrams):
        birth = diagrams[..., 0:1]
        death = diagrams[..., 1:2]
        persistence = death - birth

        x = torch.cat([birth, persistence], dim=-1)

        # Normalize coordinates to roughly [0, 1] using image bounds.
        x = (x - self.mins) / self.ranges

        weight = self.net(x).squeeze(-1)

        # Positive weights.
        return F.softplus(weight)


class NormalizedLearnablePowerPerslayWeight(nn.Module):
    """
    Learnable persistence-based weight function using normalized persistence.

    Weight = scale * (normalized_persistence + eps) ** power

    Normalizing persistence makes training more stable when raw persistence
    values have a large range.
    """

    def __init__(self, image_bnds, init_scale=1.0, init_power=1.0, eps=1e-6):
        super().__init__()

        self.eps = eps

        pers_min = float(image_bnds[1][0])
        pers_max = float(image_bnds[1][1])

        self.register_buffer("pers_min", torch.tensor(pers_min, dtype=torch.float32))
        self.register_buffer(
            "pers_range",
            torch.tensor(max(pers_max - pers_min, eps), dtype=torch.float32),
        )

        # These are raw parameters. softplus keeps scale and power positive.
        self.raw_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.raw_power = nn.Parameter(torch.tensor(float(init_power)))

    def forward(self, diagrams):
        birth = diagrams[..., 0]
        death = diagrams[..., 1]
        persistence = death - birth

        persistence = (persistence - self.pers_min) / self.pers_range
        persistence = persistence.clamp_min(self.eps)

        scale = torch.nn.functional.softplus(self.raw_scale)
        power = torch.nn.functional.softplus(self.raw_power)

        return scale * torch.pow(persistence, power)


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
    Differentiable persistence-image-style PersLay feature map using
    learnable anisotropic Gaussian bandwidths.

    Each persistence point is transformed from (birth, death) to
    (birth, persistence), then evaluated on a regular 2D grid using an
    anisotropic Gaussian kernel.

    If normalize=False, the Gaussian normalization constant is omitted.
    In that case, each point contributes peak value 1, and the output is
    not a probability density.
    """

    def __init__(
        self,
        image_size,
        image_bnds,
        sigma_x,
        sigma_y=None,
        normalize=False,
        eps=1e-6,
        **kwargs,
    ):
        """
        Parameters:
            image_size: number of grid elements on each axis, [n_x, n_y]
            image_bnds: grid bounds [[min_x, max_x], [min_y, max_y]]
            sigma_x: initial Gaussian bandwidth in the birth direction
            sigma_y: initial Gaussian bandwidth in the persistence direction.
                     If None, sigma_y is initialized equal to sigma_x.
            normalize: whether to include the Gaussian normalization constant
            eps: small positive constant for numerical stability
        """
        super().__init__()

        if sigma_y is None:
            sigma_y = sigma_x

        self.image_size = image_size
        self.image_bnds = image_bnds
        self.normalize = normalize
        self.eps = eps

        # Learn log-sigmas so that sigma_x and sigma_y remain positive.
        self.log_sigma = nn.Parameter(
            torch.log(torch.tensor([sigma_x, sigma_y], dtype=torch.float32))
        )

        step = [
            (self.image_bnds[i][1] - self.image_bnds[i][0]) / self.image_size[i]
            for i in range(2)
        ]

        coords = [
            torch.arange(
                self.image_bnds[i][0],
                self.image_bnds[i][1],
                step[i],
                dtype=torch.float32,
            )
            for i in range(2)
        ]

        M = torch.meshgrid(*coords, indexing="xy")
        mu = torch.stack(M, dim=0)  # [2, grid_y, grid_x]

        # Register as buffer so it moves with .to(device), .cuda(), etc.
        self.register_buffer("mu", mu)

    def forward(self, diagrams):
        """
        Parameters:
            diagrams: tensor of shape [B, N, 2], containing (birth, death)

        Returns:
            output: tensor of shape [B, N, grid_y, grid_x, 1]
            output_shape: shape of one transformed point, [grid_y, grid_x, 1]
        """

        # Positive learnable bandwidths.
        sigma = torch.exp(self.log_sigma).clamp_min(self.eps)
        sigma_x = sigma[0]
        sigma_y = sigma[1]

        # Transform diagram: (birth, death) -> (birth, persistence)
        birth = diagrams[..., 0:1]
        persistence = diagrams[..., 1:2] - birth
        diagrams_d = torch.cat([birth, persistence], dim=-1)  # [B, N, 2]

        # [B, N, 2, 1, 1]
        diagrams_d = diagrams_d.unsqueeze(-1).unsqueeze(-1)

        # [1, 1, 2, grid_y, grid_x]
        mu = self.mu.unsqueeze(0).unsqueeze(0)

        dx2 = torch.square(diagrams_d[:, :, 0:1] - mu[:, :, 0:1])
        dy2 = torch.square(diagrams_d[:, :, 1:2] - mu[:, :, 1:2])

        exponent = -(dx2 / (2.0 * sigma_x.square()) + dy2 / (2.0 * sigma_y.square()))

        gauss = torch.exp(exponent).squeeze(2)  # [B, N, grid_y, grid_x]

        if self.normalize:
            gauss = gauss / (2.0 * math.pi * sigma_x * sigma_y)

        output = gauss.unsqueeze(-1)  # [B, N, grid_y, grid_x, 1]
        output_shape = self.mu[0].shape + (1,)

        return output, output_shape


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

    def forward(self, diagrams, mask=None):
        """
        Parameters:
            diagrams: tensor of shape [B, N, 2], padded if necessary
            mask: optional boolean tensor of shape [B, N].
                  True means real persistence point; False means padding.

        Returns:
            vector: tensor of shape [B, output_dim] representing PersLay embeddings
        """
        vector, dim = self.phi(diagrams)
        weight_tensor = self.weight(diagrams)

        if mask is not None:
            weight_tensor = weight_tensor * mask.to(weight_tensor.dtype)

        for _ in range(vector.ndim - weight_tensor.ndim):
            weight_tensor = weight_tensor.unsqueeze(-1)

        vector = vector * weight_tensor

        perm_op = self.perm_op

        if isinstance(perm_op, str) and perm_op[:3] == "top":
            k = int(perm_op[3:])

            vector = vector.view(vector.shape[0], vector.shape[1], -1)

            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).expand_as(vector)
                vector = vector.masked_fill(~mask_expanded, float("-inf"))

            topk_vals, _ = torch.topk(vector.transpose(1, 2), k=k, dim=2)
            vector = topk_vals.reshape(vector.shape[0], -1)
        else:
            vector = perm_op(vector, dim=1)

        vector = self.rho(vector)

        return vector
