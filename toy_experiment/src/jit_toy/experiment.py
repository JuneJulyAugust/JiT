"""Integration seams: wire your completed exercises together here."""

from pathlib import Path

from .config import ExperimentConfig
from .exercises import ExerciseNotImplemented


def run_training(config: ExperimentConfig, output: Path) -> None:
    """E01: Compose data, model, trainer and an inference checkpoint.

    After D/M/F/T exercises pass:
    Reference main_jit.py::main for data/model/device/optimizer lifecycle and
    util/misc.py::save_model for state_dict checkpointing. This smaller schema
    is independent of image-model checkpoints and deliberately excludes EMA.
    - Validate CPU/MPS/CUDA availability; seed model initialization with seed.
      Use float32 baseline. Fail clearly for unavailable requested devices.
    - Create output with exist_ok=False so old runs are never overwritten.
      Save resolved config.json and environment.json (Python/torch/package
      versions and device). A failed run may retain partial logs, but must
      not contain a success marker or a checkpoint presented as complete.
    - build_data(config); make_loader with its independent seed+2 Generator.
    - Construct ToyMLP(D,H,L), move it to requested device, call train().
    - Save checkpoint.pt containing ONLY tensors and primitive containers:
      schema_version=1, config dict, model_state, projection, reference_intrinsic.
      Save model tensors on CPU; keep the original trained model on its device.
      Practice state_dict, detach/cpu, torch.save; use a temp path then rename.
    - Log completion only after save succeeds. Checkpoint is for inference,
      NOT training resume; supporting resume requires optimizer and RNG states.
    P and intrinsic reference are metadata; neither is passed to the model.
    """
    raise ExerciseNotImplemented("E01", "wire the training experiment and checkpoint")


def run_sampling(checkpoint: Path, output: Path, device: str) -> None:
    """E02: Reload a completed run and produce a samples artifact.

    Practice torch.load(map_location='cpu', weights_only=True), state_dict,
    load_state_dict(strict=True), seeded Generators, tensor metadata validation.
    Reference sample_jit.py::main for strict portable checkpoint loading.
    Validate schema and reconstruct ExperimentConfig from checkpoint values;
    only override device for portability. Instantiate matching ToyMLP, load
    weights, move to device, call S02. Use checkpoint's exact P for D04/V01.
    Save a tensor/primitive dict to output with schema_version=1, config,
    observed [N,D], generated_intrinsic [N,2], reference_intrinsic [N_ref,2],
    projection [D,2], and residual float. Refuse an existing output path.
    Never regenerate P, train data or weights when restoring a checkpoint.
    Acceptance: save/reload gives identical raw model outputs on fixed inputs;
    sampling and plotting work without the original process's memory state.
    """
    raise ExerciseNotImplemented("E02", "load a checkpoint and save generated samples")


def run_plotting(samples: Path, output: Path) -> None:
    """E03: Validate the samples artifact and call V02.

    Use CPU, weights_only=True when loading; validate schema, [N,2] shapes,
    finite values and config metadata. Derive plot title from saved D/mode/seed.
    Refuse an existing output file; create its parent directory if necessary.
    Call plot_comparison(reference_intrinsic, generated_intrinsic, ...).
    Helpful errors should explain how to install the optional [plot] extra.
    """
    raise ExerciseNotImplemented("E03", "load sample artifacts and make the comparison plot")
