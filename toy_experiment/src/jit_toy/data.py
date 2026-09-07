"""Data exercises: all sampling, linear algebra, and indexing are yours.

Batch convention: each row is one sample. Tensor creation starts on CPU with an
explicit Generator and float32 dtype; the trainer later moves observed batches.
P belongs to the data/evaluation boundary, never to the neural network.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig
from .exercises import ExerciseNotImplemented


def sample_spiral(count: int, turns: float, radius: float, *, generator: torch.Generator) -> Tensor:
    """D01: Return CPU float32 intrinsic points [count, 2].

    Practice: rand, scalar/tensor arithmetic, sin, cos, stack, shape inspection.
    Recipe (our choice): draw u uniformly in [0,1), set angle=2*pi*turns*u
    and radial distance=radius*u, then convert polar to Cartesian coordinates.
    Use one shared u for radius and angle: independent draws produce a disk.
    Stack coordinates on the feature axis; concatenating vectors gives [2*N].
    Keep operations vectorized and use the supplied generator for every draw.
    Do not normalize each ambient coordinate later or add ambient data jitter:
    that would change the embedded-manifold experiment. First plot these points.
    Acceptance: [N,2], finite, radius bounded, identical fresh seeds repeat.
    """
    raise ExerciseNotImplemented("D01", "sample the intrinsic spiral")


def make_projection(observed_dim: int, intrinsic_dim: int, *, generator: torch.Generator) -> Tensor:
    """D02: Return one fixed CPU float32 P [D,d] with orthonormal columns.

    Practice: randn, torch.linalg.qr(mode='reduced'), transpose, eye, allclose.
    Start from a random [D,d] Gaussian matrix (D>=d). Decide which QR output
    has the required shape; do not normalize columns independently because
    unit length does not imply mutual orthogonality. Verify P.T @ P = I_d.
    D=2 is valid: a rotation/reflection is still an invertible embedding.
    Generate once per dimension/seed and save the actual P with checkpoints;
    never redraw it per batch, prediction type, or visualization.
    Acceptance: orthogonality within float32 tolerance, full column rank.
    """
    raise ExerciseNotImplemented("D02", "construct the fixed orthogonal projection")


def embed_points(intrinsic: Tensor, projection: Tensor) -> Tensor:
    """D03: Map row batches [N,d] to observed [N,D].

    Practice: matmul/@, .T, device/dtype agreement, dimension assertions.
    The paper writes column vectors x=P*x_hat. Derive the transposed equation
    for row batches before coding. Do not create [N,D,d] intermediates or use
    an elementwise product. No rescaling by sqrt(D): P is an isometry.
    Acceptance: preserves pairwise distances and squared vector norms.
    """
    raise ExerciseNotImplemented("D03", "embed row-batched intrinsic points")


def project_points(observed: Tensor, projection: Tensor) -> Tensor:
    """D04: Project [N,D] back to [N,d], for evaluation ONLY.

    Practice: matrix shapes, transpose reasoning, approximate equality.
    Derive the row-batch form of P.T*x. Test a round trip with D03.
    For generated off-subspace samples this drops perpendicular components;
    a good-looking projected scatter plot alone is not a sufficient metric.
    Never call this in the model, denoising loss, or ODE integrator.
    """
    raise ExerciseNotImplemented("D04", "project observed samples for evaluation")


class ObservedDataset(Dataset):
    """D05: A map-style dataset exposing only observed vectors to the trainer."""

    def __init__(self, observed: Tensor) -> None:
        """Validate rank two, floating dtype, CPU storage; retain [N,D] tensor.

        Practice: shape/ndim, tensor ownership, optional detach/clone reasoning.
        No labels, intrinsic coordinates, or P in __getitem__'s return value.
        """
        raise ExerciseNotImplemented("D05", "store the observed dataset")

    def __len__(self) -> int:
        """D05: Return sample count, not total number of tensor elements."""
        raise ExerciseNotImplemented("D05", "report dataset length")

    def __getitem__(self, index: int) -> Tensor:
        """D05: Integer indexing should return [D]; collation creates [B,D].

        Practice the difference between x[index], x[index:index+1], and advanced
        indexing in a scratch notebook. Do not implement batching here.
        """
        raise ExerciseNotImplemented("D05", "index one observed sample")


@dataclass(frozen=True)
class DataBundle:
    """Keep metadata outside Dataset/model; all fields initially live on CPU."""

    train: ObservedDataset
    reference_intrinsic: Tensor  # independent held-out [N_eval,2]
    projection: Tensor  # [D,2], fixed for the entire run


def build_data(config: ExperimentConfig) -> DataBundle:
    """D06: Compose D01-D05 into training data and held-out visualization data.

    Use separate CPU Generators: seed for training points, seed+1 for reference
    points, projection_seed for P. This keeps the intrinsic dataset identical
    across D and prediction choices. Embedding consumes no new randomness.
    Call sample_spiral twice with separate streams; do not split by slicing
    an angle-sorted spiral. P is shared by train and evaluation.
    Later datasets can implement the same [N,2] contract; update config choices
    and dispatch here, without changing the trainer or model interfaces.
    """
    raise ExerciseNotImplemented("D06", "assemble reproducible data splits")


def make_loader(dataset: ObservedDataset, batch_size: int, *, generator: torch.Generator) -> DataLoader:
    """D07: Build shuffled batches using PyTorch DataLoader.

    Practice: DataLoader, default collation, shuffle, generator, drop_last.
    Baseline choices: num_workers=0 (easy debugging on macOS), drop_last=False,
    no pinned memory. A final short batch must work everywhere downstream.
    Use a loader-specific generator (seed+2), so shuffling cannot change P.
    Later add worker seeding/pinned memory as explicit performance experiments.
    Acceptance: visit each example once per epoch, retain the short last batch.
    """
    raise ExerciseNotImplemented("D07", "batch and shuffle observed data")
