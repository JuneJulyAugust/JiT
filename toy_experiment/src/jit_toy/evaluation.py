"""Evaluation exercises: projected quality and off-subspace error."""

from pathlib import Path

from torch import Tensor

from .exercises import ExerciseNotImplemented


def subspace_residual(observed: Tensor, projection: Tensor) -> float:
    """V01: Mean squared Euclidean distance to the embedded linear subspace.

    Practice compose D04+D03, subtraction, square/sum/mean, detach/item.
    Project generated rows into 2D, embed them back into D, subtract that
    component from original rows, sum squared residuals per row, average rows.
    Avoid materializing a dense [D,D] projector; two thin products suffice.
    This measures distance to span(P), NOT distance to the spiral itself.
    D=2 residual is approximately zero even for bad samples, so inspect both
    scatter shape/coverage and residual. Acceptance: embedded reference gives
    near zero; adding a known perpendicular component at D>2 increases error.
    """
    raise ExerciseNotImplemented("V01", "measure off-subspace sample energy")


def plot_comparison(reference: Tensor, generated: Tensor, output: Path, *, title: str) -> None:
    """V02: Save side-by-side intrinsic scatter plots to the requested path.

    Inputs already [N,2] CPU tensors. Practice detach().cpu().numpy() at the
    visualization boundary only. Import matplotlib here (optional dependency),
    select Agg before pyplot for headless use, close the figure after saving.
    Use equal aspect ratio, matching axis limits, small translucent points,
    ground-truth/generated titles and recorded D/mode/seed. Include all finite
    samples in limits or explicitly annotate clipped/outlier counts: never
    hide catastrophic failures to match Figure 2's attractive framing.
    Optional later: assemble a 4x4 Figure 2 comparison from 12 saved runs,
    with dimensions as rows and reference/x/eps/v as columns.
    """
    raise ExerciseNotImplemented("V02", "render reference and generated scatter plots")
