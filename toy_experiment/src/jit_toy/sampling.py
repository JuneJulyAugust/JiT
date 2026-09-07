"""Inference-time ODE exercises, deliberately independent of the projection."""

import torch
from torch import Tensor, nn

from .config import ExperimentConfig
from .exercises import ExerciseNotImplemented


def ode_step(model: nn.Module, z: Tensor, t: Tensor, t_next: Tensor,
             prediction: str, solver: str) -> Tensor:
    """S01: One Euler or Heun step in observed D-dimensional space.

    Inputs: z [B,D], t/t_next [B,1] inside the configured interior interval.
    Practice model calls, broadcast step sizes, reuse of F03, non-inplace ops.
    Reference: denoiser.py::Denoiser._euler_step and _heun_step. Their update
    algebra carries over directly; replace _forward_sample with model + F03.
    Euler uses current velocity and delta_t. Heun first predicts a provisional
    next z with Euler, evaluates velocity at (provisional_z,t_next), then uses
    the average of the two velocities to update the ORIGINAL z. Do not update
    from provisional_z a second time or average raw x/eps predictions.
    Reject unknown solver names. No P, latent projection, or fresh noise here.
    Acceptance: a constant-velocity test model advances exactly delta_t*v;
    check Heun on a simple nonconstant field and compare with its known ODE.
    """
    raise ExerciseNotImplemented("S01", "implement Euler and Heun ODE updates")


def sample(model: nn.Module, config: ExperimentConfig, *, generator: torch.Generator) -> Tensor:
    """S02: Generate eval_size observed points [N,D] with no gradient graph.

    Practice eval(), torch.inference_mode(), linspace, expand, batching, cat.
    Reference: Denoiser.generate builds a steps+1 grid and disables gradients;
    sample_jit.py::main demonstrates portable device checks/inference_mode.
    Difference: JiT integrates 0->1 with clamped x conversion and final Euler.
    Our interior grid permits Heun on every interval and also supports eps,
    whose conversion is singular at JiT's starting time of zero.
    Use the supplied CPU Generator (evaluation seed=seed+1) for standard
    Gaussian [N,D] initial points, then transfer to model device. Integrate
    from time_eps to 1-time_eps with sampling_steps intervals for ALL modes.
    This treats Gaussian noise at time_eps as an approximation to z_time_eps
    and returns a near-data endpoint; never claim exact integration from 0 to 1.
    Create steps+1 grid entries; turn scalar grid times into [B,1] columns
    with expand (expanded views share memory, so do not mutate them in place).
    Call S01 on every interval. Batching is optional initially; chunking later
    should preserve total sample count and order. Return finite detached CPU
    float32 samples; restore the model's previous training flag after sampling,
    including if an error occurs. Do not decode with P inside the solver.
    Acceptance: shape/dtype/device, no grad_fn, fixed seeds reproduce results.
    """
    raise ExerciseNotImplemented("S02", "integrate the learned velocity field")
