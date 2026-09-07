"""Training exercises. No optimizer or backward pass is supplied for you."""

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .exercises import ExerciseNotImplemented


def build_optimizer(model: nn.Module, config: ExperimentConfig) -> Optimizer:
    """T01: Create an AdamW optimizer over the model's registered parameters.

    Practice model.parameters(), requires_grad, torch.optim.AdamW. Reference:
    main_jit.py::main uses AdamW with betas=(0.9,0.95). Follow those betas,
    use learning_rate from config, and explicitly set weight_decay=0.0 as JiT
    does by default (AdamW's own default is nonzero). With zero decay there is
    no need for util.misc.add_weight_decay's bias/norm parameter grouping.
    Construct after moving the model to the device. Confirm a nonempty
    parameter list; missing parameters usually mean M01 used ordinary lists.
    """
    raise ExerciseNotImplemented("T01", "create the optimizer")


def train_step(model: nn.Module, optimizer: Optimizer, clean: Tensor,
               config: ExperimentConfig, *, generator: torch.Generator) -> dict[str, float]:
    """T02: One complete update; input clean [B,D] is already on model device.

    Practice train(), zero_grad(set_to_none=True), randn, .to(), forward,
    scalar loss.backward(), optimizer.step(), detach(), item(), isfinite.
    Reference: engine_jit.py::train_one_epoch for the update order. Data is
    already float32: do not copy its image-specific /255 and [-1,1] transform.
    1. Clear old gradients and enable training mode.
    2. Draw times via F01 and FULL [B,D] noise on CPU with the supplied flow
       generator; transfer both to clean's device/dtype. CPU draws make the
       baseline generator policy portable to MPS. Use clean.shape[0], not the
       configured batch_size: the last batch may be smaller.
    3. Build F02, call model(z,t), convert using F03, reduce with F04.
    4. Check loss and gradients are finite. Backpropagate before stepping.
       Never detach raw predictions or wrap forward/loss in no_grad.
    5. Return Python floats {loss: ..., loss_sum: ...} AFTER backward.
       Retaining graph tensors in a metrics list causes memory growth.
    No EMA, mixed precision, gradient clipping or schedulers in the baseline;
    add these as separate recorded extensions once this works.
    Acceptance: overfit a fixed noisy minibatch; parameters actually change,
    gradients do not accumulate between calls, every prediction mode works.
    """
    raise ExerciseNotImplemented("T02", "implement one differentiable training update")


def train(model: nn.Module, loader: DataLoader, config: ExperimentConfig,
          metrics_path: Path) -> dict[str, float]:
    """T03: Run exactly train_steps optimizer updates and stream metrics JSONL.

    Practice iter/next, StopIteration, tensor.to, device management, RNG state.
    Build T01 once and a CPU flow Generator seeded with seed+3 once. Restart
    the loader iterator on exhaustion; do not cache batches via itertools.cycle
    or reseed the sampler each step. Move each observed batch to the device.
    Call T02, append step/loss/loss_sum to metrics_path at log_every and
    the last step. Use `from loguru import logger` and direct logger.info(...)
    calls so logging.py reports YOUR source filename and line number.
    Log single-line messages and scalar statistics, not huge tensors.
    Create metrics with exclusive mode to avoid overwriting an old run.
    Return final scalar metrics. Keep data projection/intrinsic points out of
    this signature; they have no role in optimization.
    Acceptance: exact update count even across epoch boundaries; a short final
    batch trains, JSONL parses, a failed update does not report success.
    Resume with optimizer/RNG state is a later extension (see DESIGN.md).
    """
    raise ExerciseNotImplemented("T03", "implement the training loop and metric recording")
