import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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


class PersistencePerslayWeight(nn.Module):
    """Weight each point by its nonnegative persistence ``death - birth``."""

    def forward(self, diagrams):
        return (diagrams[..., 1] - diagrams[..., 0]).clamp_min(0.0)


class LinearRampPerslayWeight(nn.Module):
    """Bounded diagonal-vanishing weight ``min(persistence / threshold, 1)``."""

    def __init__(self, threshold):
        super().__init__()
        threshold = float(threshold)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("threshold must be finite and positive")
        self.register_buffer(
            "threshold", torch.tensor(threshold, dtype=torch.float32)
        )

    def forward(self, diagrams):
        persistence = (diagrams[..., 1] - diagrams[..., 0]).clamp_min(0.0)
        return (persistence / self.threshold).clamp(max=1.0)


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


class _ExplicitGridGaussianPhi(nn.Module):
    """Shared validation and tensor layout for explicit Gaussian grid maps."""

    _VALID_FLATTEN_ORDERS = {"row-major", "legacy-x-major"}

    def __init__(
        self,
        image_size,
        image_bnds,
        sigma_x,
        sigma_y=None,
        normalize=False,
        learnable_sigma=True,
        flatten_order="legacy-x-major",
        chunk_size=None,
        eps=1e-6,
        checkpoint_chunks=False,
    ):
        super().__init__()

        if sigma_y is None:
            sigma_y = sigma_x
        if len(image_size) != 2 or any(
            int(value) <= 0 or float(value) != int(value) for value in image_size
        ):
            raise ValueError("image_size must contain two positive integers")
        if len(image_bnds) != 2 or any(len(bounds) != 2 for bounds in image_bnds):
            raise ValueError(
                "image_bnds must be [[birth_min, birth_max], "
                "[persistence_min, persistence_max]]"
            )

        bounds = [[float(value) for value in axis] for axis in image_bnds]
        for lower, upper in bounds:
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError("each pair of image bounds must be finite and increasing")
        for name, value in (("sigma_x", sigma_x), ("sigma_y", sigma_y)):
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if flatten_order not in self._VALID_FLATTEN_ORDERS:
            raise ValueError(
                "flatten_order must be 'row-major' or 'legacy-x-major'"
            )
        if chunk_size is not None:
            if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
                raise TypeError("chunk_size must be a positive integer or None")
            if chunk_size <= 0:
                raise ValueError("chunk_size must be a positive integer or None")
        if not isinstance(checkpoint_chunks, bool):
            raise TypeError("checkpoint_chunks must be a boolean")
        if checkpoint_chunks and chunk_size is None:
            raise ValueError("checkpoint_chunks requires an explicit chunk_size")
        eps = float(eps)
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be finite and positive")

        self.image_size = [int(image_size[0]), int(image_size[1])]
        self.image_bnds = bounds
        self.normalize = bool(normalize)
        self.learnable_sigma = bool(learnable_sigma)
        self.sigma_is_learnable = self.learnable_sigma
        self.flatten_order = flatten_order
        self.chunk_size = chunk_size
        self.checkpoint_chunks = checkpoint_chunks
        self.eps = eps

        initial_log_sigma = torch.log(
            torch.tensor([sigma_x, sigma_y], dtype=torch.float32)
        )
        if self.learnable_sigma:
            self.log_sigma = nn.Parameter(initial_log_sigma)
        else:
            self.register_buffer("log_sigma", initial_log_sigma)

    def _sigmas(self):
        sigma = torch.exp(self.log_sigma).clamp_min(self.eps)
        return sigma[0], sigma[1]

    @staticmethod
    def _birth_persistence(diagrams):
        birth = diagrams[..., 0]
        persistence = diagrams[..., 1] - birth
        return birth, persistence

    def _format_output(self, values):
        """Format ``[B, N, ny, nx]`` values for PersLay aggregation."""
        if self.flatten_order == "legacy-x-major":
            values = values.transpose(-2, -1)  # [B, N, nx, ny]
        output = values.unsqueeze(-1)
        output_shape = tuple(values.shape[-2:]) + (1,)
        return output, output_shape

    def _format_aggregated_output(self, values):
        """Format fused ``[B, ny, nx]`` values after point aggregation."""
        if self.flatten_order == "legacy-x-major":
            values = values.transpose(-2, -1)  # [B, nx, ny]
        return values.unsqueeze(-1)

    def _chunk_slices(self, number_of_points):
        """Yield point slices for exact memory-bounded aggregation."""
        if number_of_points == 0:
            return
        chunk_size = self.chunk_size or number_of_points
        for start in range(0, number_of_points, chunk_size):
            yield slice(start, min(start + chunk_size, number_of_points))


class GridSampledGaussianPerslayPhi(_ExplicitGridGaussianPhi):
    """Gaussian persistence surface evaluated at pixel centres.

    With ``ConstantPerslayWeight`` and sum aggregation, this layer reproduces
    the mathematical construction used by the Rust ``sampled`` mode:

    ``S_j(D) = sum_i w_i exp(-0.5 * distance_sigma(z_j, u_i)^2)``.

    ``normalize=True`` additionally multiplies every Gaussian by
    ``1 / (2*pi*sigma_x*sigma_y)``.  Values remain point samples, not pixel
    integrals.  ``flatten_order='legacy-x-major'`` matches the Rust feature
    ordering; ``'row-major'`` is convenient for conventional image display.
    """

    def __init__(
        self,
        image_size,
        image_bnds,
        sigma_x,
        sigma_y=None,
        normalize=False,
        learnable_sigma=True,
        flatten_order="legacy-x-major",
        chunk_size=None,
        eps=1e-6,
        checkpoint_chunks=False,
        **kwargs,
    ):
        super().__init__(
            image_size=image_size,
            image_bnds=image_bnds,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            normalize=normalize,
            learnable_sigma=learnable_sigma,
            flatten_order=flatten_order,
            chunk_size=chunk_size,
            eps=eps,
            checkpoint_chunks=checkpoint_chunks,
        )

        nx, ny = self.image_size
        x_min, x_max = self.image_bnds[0]
        y_min, y_max = self.image_bnds[1]
        x_width = (x_max - x_min) / nx
        y_height = (y_max - y_min) / ny

        x_centres = x_min + (torch.arange(nx, dtype=torch.float32) + 0.5) * x_width
        y_centres = y_min + (torch.arange(ny, dtype=torch.float32) + 0.5) * y_height
        grid_y, grid_x = torch.meshgrid(y_centres, x_centres, indexing="ij")
        # meshgrid returns expanded views with overlapping storage on current
        # PyTorch versions.  Register contiguous copies so state_dict loading
        # can safely copy into these buffers while preserving checkpoint keys.
        self.register_buffer("grid_x", grid_x.contiguous())
        self.register_buffer("grid_y", grid_y.contiguous())

    def _axis_values(self, birth, persistence, sigma_x, sigma_y):
        x_centres = self.grid_x[0, :]
        y_centres = self.grid_y[:, 0]
        dx2 = torch.square(birth.unsqueeze(-1) - x_centres)
        dy2 = torch.square(persistence.unsqueeze(-1) - y_centres)
        inv_two_sigma_x2 = -0.5 / sigma_x.square()
        inv_two_sigma_y2 = -0.5 / sigma_y.square()
        x_values = torch.exp(dx2 * inv_two_sigma_x2)
        y_values = torch.exp(dy2 * inv_two_sigma_y2)
        return x_values, y_values

    def forward(self, diagrams):
        sigma_x, sigma_y = self._sigmas()
        birth, persistence = self._birth_persistence(diagrams)
        x_values, y_values = self._axis_values(
            birth, persistence, sigma_x, sigma_y
        )
        values = y_values.unsqueeze(-1) * x_values.unsqueeze(-2)
        if self.normalize:
            values = values / (2.0 * math.pi * sigma_x * sigma_y)
        return self._format_output(values)

    def _chunk_product(self, birth, persistence, weights, sigma_x, sigma_y):
        """Compute the separable, weighted matrix product for a point chunk."""
        x_values, y_values = self._axis_values(
            birth, persistence, sigma_x, sigma_y
        )
        return torch.bmm(
            y_values.transpose(1, 2), x_values * weights.unsqueeze(-1)
        )

    def weighted_sum(self, diagrams, weights):
        """Exactly aggregate weighted sampled surfaces without ``B*N*H*W``.

        The Gaussian kernel is separable, so the point sum can be evaluated as
        a batched matrix product.  ``chunk_size`` only changes the order of
        floating-point additions; mathematically the result is identical to
        materializing all pointwise surfaces and summing them.
        """
        if diagrams.ndim != 3 or diagrams.shape[-1] != 2:
            raise ValueError("diagrams must have shape [B, N, 2]")
        if weights.shape != diagrams.shape[:2]:
            raise ValueError("weights must have shape [B, N]")

        batch_size, number_of_points = diagrams.shape[:2]
        nx, ny = self.image_size
        if number_of_points == 0:
            values = diagrams.new_zeros((batch_size, ny, nx))
            return self._format_aggregated_output(values)

        sigma_x, sigma_y = self._sigmas()
        birth, persistence = self._birth_persistence(diagrams)

        if not self.checkpoint_chunks and (
            self.chunk_size is None or self.chunk_size >= number_of_points
        ):
            values = self._chunk_product(
                birth, persistence, weights, sigma_x, sigma_y
            )
        else:
            values = diagrams.new_zeros((batch_size, ny, nx))
            for point_slice in self._chunk_slices(number_of_points):
                inputs = (
                    birth[:, point_slice],
                    persistence[:, point_slice],
                    weights[:, point_slice],
                    sigma_x,
                    sigma_y,
                )
                if self.checkpoint_chunks and torch.is_grad_enabled():
                    # Non-reentrant checkpointing recomputes intermediate
                    # Gaussian factors on backward instead of retaining them.
                    contribution = checkpoint(
                        self._chunk_product, *inputs, use_reentrant=False
                    )
                else:
                    contribution = self._chunk_product(*inputs)
                values = values + contribution

        if self.normalize:
            values = values / (2.0 * math.pi * sigma_x * sigma_y)
        return self._format_aggregated_output(values)


class IntegratedGaussianPerslayPhi(_ExplicitGridGaussianPhi):
    """Exact Gaussian mass integrated over each persistence-image pixel.

    This is the canonical persistence-image pixelization.  With
    ``normalize=True`` (the default), each point's Gaussian integrates to one
    over the full plane before weighting.  Setting ``normalize=False`` returns
    the integral of a peak-one, unnormalized Gaussian instead.
    """

    def __init__(
        self,
        image_size,
        image_bnds,
        sigma_x,
        sigma_y=None,
        normalize=True,
        learnable_sigma=True,
        flatten_order="row-major",
        chunk_size=None,
        eps=1e-6,
        checkpoint_chunks=False,
        **kwargs,
    ):
        super().__init__(
            image_size=image_size,
            image_bnds=image_bnds,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            normalize=normalize,
            learnable_sigma=learnable_sigma,
            flatten_order=flatten_order,
            chunk_size=chunk_size,
            eps=eps,
            checkpoint_chunks=checkpoint_chunks,
        )

        nx, ny = self.image_size
        self.register_buffer(
            "x_edges",
            torch.linspace(
                self.image_bnds[0][0],
                self.image_bnds[0][1],
                nx + 1,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "y_edges",
            torch.linspace(
                self.image_bnds[1][0],
                self.image_bnds[1][1],
                ny + 1,
                dtype=torch.float32,
            ),
        )

    @staticmethod
    def _normal_interval_masses(edges, means, sigma):
        """Stable normal probability mass for adjacent edge intervals.

        Direct CDF subtraction is inaccurate in far positive/negative tails,
        while an absolute-tail shortcut has an ambiguous derivative at zero.
        This implementation uses two stable primitives per standardized edge:

        * ``0.5 * erfc(abs(z))`` supplies same-sign tail probabilities;
        * ``erf(z)`` supplies intervals that touch or cross zero.

        Strict sign tests route intervals with an edge exactly at the Gaussian
        mean through the smooth ``erf`` expression, preserving the correct
        derivative there.  The central expression also retains tiny float32
        masses that would disappear when subtracting two CDF values near 0.5.
        """
        scale = sigma * math.sqrt(2.0)
        z = (edges - means.unsqueeze(-1)) / scale
        lower = z[..., :-1]
        upper = z[..., 1:]

        tail = 0.5 * torch.erfc(torch.abs(z))
        lower_tail = tail[..., :-1]
        upper_tail = tail[..., 1:]
        positive_mass = lower_tail - upper_tail
        negative_mass = upper_tail - lower_tail

        erf_values = torch.erf(z)
        central_mass = 0.5 * (erf_values[..., 1:] - erf_values[..., :-1])

        masses = torch.where(
            lower > 0.0,
            positive_mass,
            torch.where(upper < 0.0, negative_mass, central_mass),
        )
        return masses.clamp_min(0.0)

    def _axis_masses(self, birth, persistence, sigma_x, sigma_y):
        x_mass = self._normal_interval_masses(self.x_edges, birth, sigma_x)
        y_mass = self._normal_interval_masses(
            self.y_edges, persistence, sigma_y
        )
        return x_mass, y_mass

    def forward(self, diagrams):
        sigma_x, sigma_y = self._sigmas()
        birth, persistence = self._birth_persistence(diagrams)
        x_mass, y_mass = self._axis_masses(
            birth, persistence, sigma_x, sigma_y
        )
        values = y_mass.unsqueeze(-1) * x_mass.unsqueeze(-2)  # [B, N, ny, nx]

        if not self.normalize:
            values = values * (2.0 * math.pi * sigma_x * sigma_y)
        return self._format_output(values)

    def _chunk_product(self, birth, persistence, weights, sigma_x, sigma_y):
        """Compute the separable, weighted matrix product for a point chunk."""
        x_values, y_values = self._axis_masses(
            birth, persistence, sigma_x, sigma_y
        )
        return torch.bmm(
            y_values.transpose(1, 2), x_values * weights.unsqueeze(-1)
        )

    def weighted_sum(self, diagrams, weights):
        """Exactly aggregate canonical persistence images in bounded memory."""
        if diagrams.ndim != 3 or diagrams.shape[-1] != 2:
            raise ValueError("diagrams must have shape [B, N, 2]")
        if weights.shape != diagrams.shape[:2]:
            raise ValueError("weights must have shape [B, N]")

        batch_size, number_of_points = diagrams.shape[:2]
        nx, ny = self.image_size
        if number_of_points == 0:
            values = diagrams.new_zeros((batch_size, ny, nx))
            return self._format_aggregated_output(values)

        sigma_x, sigma_y = self._sigmas()
        birth, persistence = self._birth_persistence(diagrams)

        if not self.checkpoint_chunks and (
            self.chunk_size is None or self.chunk_size >= number_of_points
        ):
            values = self._chunk_product(
                birth, persistence, weights, sigma_x, sigma_y
            )
        else:
            values = diagrams.new_zeros((batch_size, ny, nx))
            for point_slice in self._chunk_slices(number_of_points):
                inputs = (
                    birth[:, point_slice],
                    persistence[:, point_slice],
                    weights[:, point_slice],
                    sigma_x,
                    sigma_y,
                )
                if self.checkpoint_chunks and torch.is_grad_enabled():
                    # Non-reentrant checkpointing recomputes intermediate
                    # Gaussian factors on backward instead of retaining them.
                    contribution = checkpoint(
                        self._chunk_product, *inputs, use_reentrant=False
                    )
                else:
                    contribution = self._chunk_product(*inputs)
                values = values + contribution

        if not self.normalize:
            values = values * (2.0 * math.pi * sigma_x * sigma_y)
        return self._format_aggregated_output(values)


class LearnableGridIntegratedGaussianPerslayPhi(IntegratedGaussianPerslayPhi):
    """Canonical persistence image on a globally learned nonuniform grid.

    The outer image bounds and number of pixels are fixed.  Internal pixel
    boundaries are learned through positive width fractions shared by every
    persistence diagram processed by this module.  For an axis with ``n``
    cells and unconstrained logits ``theta``, the normalized widths are

    ``q = r / n + (1 - r) * softmax(theta)``,

    where ``r=min_cell_fraction``.  Thus every cell remains strictly positive,
    the widths sum to one, and a zero-logit initialization reproduces the
    ordinary uniform persistence-image grid exactly.

    Gaussian mass is still integrated exactly over each (now learned) pixel,
    so gradients with respect to internal boundaries follow directly from the
    fundamental theorem of calculus.  The grid is global/model-level rather
    than input-dependent, which keeps one common vectorization for all samples.
    """

    def __init__(
        self,
        image_size,
        image_bnds,
        sigma_x,
        sigma_y=None,
        normalize=True,
        learnable_sigma=True,
        flatten_order="row-major",
        chunk_size=None,
        min_cell_fraction=0.05,
        grid_regularization_strength=0.0,
        eps=1e-6,
        checkpoint_chunks=False,
        **kwargs,
    ):
        min_cell_fraction = float(min_cell_fraction)
        if not math.isfinite(min_cell_fraction) or not 0.0 <= min_cell_fraction < 1.0:
            raise ValueError("min_cell_fraction must be finite and in [0, 1)")
        grid_regularization_strength = float(grid_regularization_strength)
        if (
            not math.isfinite(grid_regularization_strength)
            or grid_regularization_strength < 0.0
        ):
            raise ValueError(
                "grid_regularization_strength must be finite and nonnegative"
            )

        super().__init__(
            image_size=image_size,
            image_bnds=image_bnds,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            normalize=normalize,
            learnable_sigma=learnable_sigma,
            flatten_order=flatten_order,
            chunk_size=chunk_size,
            checkpoint_chunks=checkpoint_chunks,
            eps=eps,
        )

        # The fixed-grid parent creates edge buffers.  This subclass computes
        # the edges from learned widths instead, so remove the unused buffers
        # rather than carrying misleading state_dict entries.
        del self.x_edges
        del self.y_edges

        nx, ny = self.image_size
        self.x_width_logits = nn.Parameter(torch.zeros(nx, dtype=torch.float32))
        self.y_width_logits = nn.Parameter(torch.zeros(ny, dtype=torch.float32))
        self.min_cell_fraction = min_cell_fraction
        self.grid_regularization_strength = grid_regularization_strength

    def _width_fractions(self, logits):
        number_of_cells = logits.numel()
        probabilities = torch.softmax(logits, dim=0)
        # ``min_cell_fraction`` is expressed relative to a uniform cell width:
        # r=0.05 means every cell is at least 5% of 1/n of the full axis span.
        floor = self.min_cell_fraction / number_of_cells
        return floor + (1.0 - self.min_cell_fraction) * probabilities

    @staticmethod
    def _edges_from_widths(widths, lower, upper):
        lower_tensor = widths.new_tensor(lower)
        upper_tensor = widths.new_tensor(upper)
        if widths.numel() == 1:
            return torch.stack((lower_tensor, upper_tensor))
        span = upper_tensor - lower_tensor
        internal = lower_tensor + span * torch.cumsum(widths, dim=0)[:-1]
        return torch.cat((lower_tensor.unsqueeze(0), internal, upper_tensor.unsqueeze(0)))

    def grid_width_fractions(self):
        """Return normalized learned cell widths for birth and persistence axes."""
        return (
            self._width_fractions(self.x_width_logits),
            self._width_fractions(self.y_width_logits),
        )

    def grid_edges(self):
        """Return current differentiable birth and persistence pixel edges."""
        x_widths, y_widths = self.grid_width_fractions()
        x_edges = self._edges_from_widths(
            x_widths, self.image_bnds[0][0], self.image_bnds[0][1]
        )
        y_edges = self._edges_from_widths(
            y_widths, self.image_bnds[1][0], self.image_bnds[1][1]
        )
        return x_edges, y_edges

    def _axis_masses(self, birth, persistence, sigma_x, sigma_y):
        x_edges, y_edges = self.grid_edges()
        x_mass = self._normal_interval_masses(x_edges, birth, sigma_x)
        y_mass = self._normal_interval_masses(y_edges, persistence, sigma_y)
        return x_mass, y_mass

    def grid_regularization(self):
        """Dimensionless squared deviation of cell widths from a uniform grid.

        The value is zero at initialization.  Multiplication by the number of
        cells makes the normalized target width equal to one on both axes,
        keeping the scale interpretable across different grid resolutions.
        """
        x_widths, y_widths = self.grid_width_fractions()
        x_penalty = torch.mean(torch.square(x_widths * x_widths.numel() - 1.0))
        y_penalty = torch.mean(torch.square(y_widths * y_widths.numel() - 1.0))
        return 0.5 * (x_penalty + y_penalty)

    def regularization_loss(self):
        """Return the configured uniform-grid regularization contribution."""
        return self.grid_regularization_strength * self.grid_regularization()


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
        """Embed batched birth/death diagrams with an optional padding mask.

        Only unmasked finite points are meaningful. Padded slots are replaced
        with finite coordinates *before* evaluating nonlinear weights and maps,
        and excluded from every reduction; this also avoids NaN propagation
        when padding contains infinite or NaN sentinels.
        """
        if diagrams.ndim != 3 or diagrams.shape[-1] != 2:
            raise ValueError("diagrams must have shape [B, N, 2]")
        if mask is not None:
            if mask.shape != diagrams.shape[:2] or mask.dtype != torch.bool:
                raise ValueError("mask must be boolean with shape [B, N]")
            diagrams = torch.where(mask.unsqueeze(-1), diagrams, 0.0)

        weight_tensor = self.weight(diagrams)
        if mask is not None:
            weight_tensor = torch.where(mask, weight_tensor, 0.0)

        # Gaussian separability permits much smaller intermediate tensors for
        # both sum and masked mean; other reductions retain the generic path.
        if self.perm_op in (torch.sum, torch.mean) and hasattr(self.phi, "weighted_sum"):
            vector = self.phi.weighted_sum(diagrams, weight_tensor)
            if self.perm_op is torch.mean:
                counts = (
                    mask.sum(dim=1) if mask is not None
                    else diagrams.new_full((diagrams.shape[0],), diagrams.shape[1])
                )
                vector = vector / counts.clamp_min(1).to(vector.dtype).view(-1, 1, 1, 1)
            return self.rho(vector)

        vector, _ = self.phi(diagrams)
        for _ in range(vector.ndim - weight_tensor.ndim):
            weight_tensor = weight_tensor.unsqueeze(-1)
        vector = vector * weight_tensor

        perm_op = self.perm_op
        if isinstance(perm_op, str) and perm_op.startswith("top"):
            prefix = "topk" if perm_op.startswith("topk") else "top"
            number = perm_op[len(prefix):]
            if not number.isdecimal() or int(number) <= 0:
                raise ValueError("topk permutation operation requires a positive integer")
            k = int(number)
            values = vector.flatten(start_dim=2)
            if mask is not None:
                values = values.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            # Pad missing entries with -inf, then explicitly zero missing ranks.
            if k > values.shape[1]:
                values = F.pad(values, (0, 0, 0, k - values.shape[1]), value=float("-inf"))
            topk_vals = torch.topk(values, k=k, dim=1).values
            vector = torch.where(torch.isfinite(topk_vals), topk_vals, 0.0)
            vector = vector.transpose(1, 2).reshape(vector.shape[0], -1)
        elif perm_op in (torch.sum, torch.mean, torch.max, torch.min):
            if perm_op is torch.sum:
                vector = vector.sum(dim=1)
            elif perm_op is torch.mean:
                vector = vector.sum(dim=1)
                counts = (
                    mask.sum(dim=1) if mask is not None
                    else diagrams.new_full((diagrams.shape[0],), diagrams.shape[1])
                )
                vector = vector / counts.clamp_min(1).to(vector.dtype).view(
                    -1, *([1] * (vector.ndim - 1))
                )
            else:
                if mask is not None:
                    fill_value = float("-inf") if perm_op is torch.max else float("inf")
                    vector = vector.masked_fill(
                        ~mask.view(mask.shape + (1,) * (vector.ndim - 2)),
                        fill_value,
                    )
                if vector.shape[1] == 0:
                    vector = vector.new_zeros((vector.shape[0], *vector.shape[2:]))
                else:
                    vector = perm_op(vector, dim=1).values
                    if mask is not None:
                        has_points = mask.any(dim=1).view(
                            -1, *([1] * (vector.ndim - 1))
                        )
                        vector = torch.where(has_points, vector, 0.0)
        else:
            # Preserve support for user-supplied permutation operators.
            vector = perm_op(vector, dim=1)
        return self.rho(vector)
