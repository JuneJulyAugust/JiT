"""Training exercises. No optimizer or backward pass is supplied for you."""

from typing import Callable, Iterable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .config import ExperimentConfig
from .exercises import ExerciseNotImplemented
from .flow import FlowBatch


def build_optimizer(parameters: Iterable[Tensor], learning_rate: float) -> Optimizer:
    """T01: Create an AdamW optimizer over the supplied model parameters.

    Practice parameter iterables, requires_grad, torch.optim.AdamW. Reference:
    main_jit.py::main uses AdamW with betas=(0.9,0.95). Follow those betas,
    use the supplied learning_rate, and explicitly set weight_decay=0.0 as JiT
    does by default (AdamW's own default is nonzero). With zero decay there is
    no need for util.misc.add_weight_decay's bias/norm parameter grouping.
    The caller moves the model to its device before passing model.parameters().
    This iterable can be a one-shot generator: do not exhaust it to count
    parameters and then hand the exhausted generator to AdamW. Parameter
    registration checks belong to M01; no experiment configuration is needed.
    """
    raise ExerciseNotImplemented("T01", "create the optimizer")


def train_step(
    model: nn.Module,
    optimizer: Optimizer,
    batch: FlowBatch,
    prediction: str,
) -> dict[str, float]:
    """T02: Update model/optimizer from one fixed batch on the model device.

    Practice train(), zero_grad(set_to_none=True), forward,
    scalar loss.backward(), optimizer.step(), detach(), item(), isfinite.
    Reference: engine_jit.py::train_one_epoch for the update order. Data is
    already float32: do not copy its image-specific /255 and [-1,1] transform.
    1. Clear old gradients and enable training mode.
    2. Read z [B,D], t [B,1] and target_velocity [B,D] from batch. Its tensors
       already share the model's device/dtype. Do not mutate them or draw noise.
    3. Call model(z,t), convert using F03/prediction, reduce with F04.
    4. Check loss is finite, run backward, check gradients are finite, then
       step the optimizer. Gradient checks must follow backward.
       Never detach raw predictions or wrap forward/loss in no_grad.
    5. Return Python floats {loss: ..., loss_sum: ...} after the optimizer step.
       Retaining graph tensors in a metrics list causes memory growth.
    No exponential moving average, mixed precision, clipping or schedulers;
    add these as separate recorded extensions once this works.
    Acceptance: overfit a fixed noisy minibatch; parameters actually change,
    gradients do not accumulate between calls, every prediction mode works.
    """
    raise ExerciseNotImplemented("T02", "implement one differentiable training update")


def train(
    model: nn.Module,
    loader: Iterable[Tensor],
    optimizer: Optimizer,
    config: ExperimentConfig,
    *,
    generator: torch.Generator,
    on_metrics: Callable[[int, dict[str, float]], None],
) -> dict[str, float]:
    """T03: Run train_steps updates; report each completed step to on_metrics.

    Practice iter/next, StopIteration, tensor.to and random-generator state.
    The caller supplies the optimizer and a CPU flow generator seeded once.
    loader must be nonempty and re-iterable, as DataLoader is; a one-shot
    iterator cannot restart an epoch. Restart on StopIteration without caching
    batches via itertools.cycle or reseeding the sampler each step.
    Move each clean [B,D] batch to config.device. Draw [B,1] times with F01
    using the three time settings, and full [B,D] standard Gaussian noise on
    CPU with the supplied generator. Transfer both to clean's device/dtype,
    then call F02 and T02. Use the actual B, including for the short last batch.
    After each successful update call on_metrics(step, metrics), with steps
    numbered 1 through train_steps and a fresh dictionary of Python floats.
    File creation, logging intervals and log formatting belong to the caller;
    an in-memory callback suffices to check this loop. Let callback errors
    propagate so a failed write cannot be mistaken for a completed run.
    Return final scalar metrics. Keep data projection/intrinsic points out of
    this signature; they have no role in optimization.
    Acceptance: exact update count even across epoch boundaries; a short final
    batch trains, callback records are ordered, a failed update emits no record.
    Resume with optimizer/random-generator state is a later extension.
    """
    raise ExerciseNotImplemented("T03", "implement the training loop and metric reporting")
