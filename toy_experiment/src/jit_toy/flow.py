"""Schedule, target conversion and loss exercises (paper Eqs. 1-3, Table 1)."""

from dataclasses import dataclass

import torch
from torch import Tensor

from .exercises import ExerciseNotImplemented


@dataclass(frozen=True)
class FlowBatch:
    """One fixed training example batch; callers must not mutate its tensors.

    Frozen fields prevent rebinding, but do not make tensor storage immutable.
    Keeping the batch explicit lets you test an update without drawing noise.
    """

    z: Tensor  # [B,D]
    t: Tensor  # [B,1], broadcasts only across D
    target_velocity: Tensor  # [B,D]


def sample_times(
    batch_size: int,
    *,
    mean: float,
    standard_deviation: float,
    margin: float,
    generator: torch.Generator,
) -> Tensor:
    """F01: CPU float32 logit-normal times [B,1].

    Practice randn, sigmoid, affine transforms, clamp, broadcasting.
    Reference: denoiser.py::Denoiser.sample_t and its forward time reshape.
    The normal variable has the supplied mean and standard_deviation;
    sigmoid maps it to time. Restrict to [margin, 1-margin] for finite
    denominators. The caller passes config.time_mean, time_std and time_eps;
    this function does not need the rest of the experiment configuration.
    Contract: batch_size>0, finite mean, standard_deviation>0 and finite,
    and 0<margin<0.5. ExperimentConfig validates these settings at the boundary.
    Clamping is an explicit practical change to the ideal time distribution.
    Return a column, never [B]: [B] can silently broadcast along D when B=D.
    Consume the supplied random generator; its seed and lifetime belong to
    the caller. Do not create or reseed a generator inside this function.
    """
    raise ExerciseNotImplemented("F01", "sample logit-normal time columns")


def make_flow_batch(clean: Tensor, t: Tensor, noise: Tensor) -> FlowBatch:
    """F02: Construct z_t and target velocity from explicit inputs.

    Practice broadcasting, elementwise arithmetic, shape validation.
    Use paper Eq. 1, z_t=t*x+(1-t)*eps, and Eq. 2, v=x-eps.
    Reference: denoiser.py::Denoiser.forward, adapted from [B,C,H,W] to [B,D].
    JiT computes (x-z)/(1-t).clamp_min(t_eps) for its target. In our strictly
    interior interval the denominator never needs clamping, so x-eps is
    algebraically equivalent (up to rounding) and avoids cancellation.
    clean and noise must both be [B,D]; t must be [B,1] with matching device
    and dtype. Noise is FULL ambient standard Gaussian, never embedded 2D noise.
    Receive noise as an argument to make fixed-input algebra checks easy.
    Acceptance: interpolation endpoints, shape, and exact target sign.
    """
    raise ExerciseNotImplemented("F02", "construct the noisy interpolation and velocity target")


def to_velocity(raw: Tensor, z: Tensor, t: Tensor, prediction: str) -> Tensor:
    """F03: Convert the direct [B,D] network output to velocity (Table 1 row 3).

    x output: v_hat=(raw-z)/(1-t).
    eps output: v_hat=(z-raw)/t.
    v output: v_hat=raw.
    Practice branching over semantic modes, broadcast division, autograd.
    Callers guarantee time_eps<=t<=1-time_eps, including during sampling.
    Validate mode/time/shape instead of silently accepting an unknown target.
    Do not secretly clamp one denominator here: use the shared time policy.
    Keep this transformation OUTSIDE the model. Preserve the graph through raw.
    Reference: denoiser.py::Denoiser.forward/_forward_sample for x conversion;
    eps/v modes are additions from Table 1, not present in that implementation.
    Acceptance: true x, eps, and v all recover x-eps for the same interior t;
    test B=1 and B=D to catch unintended broadcasting.
    """
    raise ExerciseNotImplemented("F03", "convert direct predictions to velocity")


def velocity_loss(predicted: Tensor, target: Tensor) -> Tensor:
    """F04: Return one scalar differentiable velocity-space squared loss.

    Practice subtraction, square, explicit reduction axes, scalar tensors.
    Reference: denoiser.py::Denoiser.forward ends with mean over all non-batch
    axes, then mean over batch. Follow that convention here: mean across D,
    then batch. This is Eq. 3's squared Euclidean norm divided by D; that
    constant changes gradient scale, so keep it fixed across prediction modes.
    Report the optimized mean as loss and loss*D as loss_sum for clarity.
    Reject shape mismatches
    before broadcasting. No extra weighting, detach, .item(), or sqrt here.
    Acceptance: equality to a hand-calculated tiny example; finite gradients.
    """
    raise ExerciseNotImplemented("F04", "reduce the common velocity loss")
