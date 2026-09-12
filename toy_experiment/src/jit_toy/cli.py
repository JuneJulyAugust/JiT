"""Working Click commands. Numerical commands stop at explicit exercise stubs."""

from functools import wraps
import json
from pathlib import Path
from typing import Optional

import click
from loguru import logger

from . import __version__
from .config import experiment_matrix, load_config
from .exercises import ExerciseNotImplemented
from .logging import configure_logging


def exercise_errors(function):
    """Translate only intentional exercise failures; real bugs keep tracebacks."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ExerciseNotImplemented as error:
            raise click.ClickException(str(error)) from error
    return wrapped


def read_config(path: Optional[Path]):
    try:
        return load_config(path)
    except (ValueError, OSError) as error:
        raise click.ClickException(f"Invalid configuration: {error}") from error


@click.group()
@click.version_option(version=__version__)
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), default="INFO")
@click.option("--log-file", type=click.Path(path_type=Path, dir_okay=False), default=None)
def main(log_level: str, log_file: Optional[Path]) -> None:
    """Learn Section 3.3 by completing the numbered PyTorch exercises."""
    configure_logging(log_level, log_file)


@main.command()
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--suite", is_flag=True, help="Print all 12 dimension/prediction configurations.")
def plan(config: Optional[Path], suite: bool) -> None:
    """Print resolved JSON configuration; no tensors or run artifacts are created."""
    base = read_config(config)
    result = [case.to_dict() for case in experiment_matrix(base)] if suite else base.to_dict()
    logger.info("Validated {} configuration(s); numerical exercises remain to be implemented", 12 if suite else 1)
    click.echo(json.dumps(result, indent=2, allow_nan=False))


@main.command()
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", required=True, type=click.Path(path_type=Path, file_okay=False))
@exercise_errors
def train(config: Optional[Path], output: Path) -> None:
    """Train one configuration after completing the exercises (currently a stub)."""
    resolved = read_config(config)
    if output.exists():
        raise click.ClickException(f"Output already exists: {output}; choose a new run directory")
    from .experiment import run_training
    logger.info("Requested training D={} prediction={} output={}", resolved.observed_dim, resolved.prediction, output)
    run_training(resolved, output)


@main.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--device", type=click.Choice(["cpu", "mps", "cuda"]), default="cpu", show_default=True)
@exercise_errors
def sample(checkpoint: Path, output: Path, device: str) -> None:
    """Generate points from a checkpoint after S/E exercises (currently a stub)."""
    if output.exists():
        raise click.ClickException(f"Output already exists: {output}")
    from .experiment import run_sampling
    logger.info("Requested sampling checkpoint={} device={}", checkpoint, device)
    run_sampling(checkpoint, output, device)


@main.command()
@click.option("--samples", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", required=True, type=click.Path(dir_okay=False, path_type=Path))
@exercise_errors
def plot(samples: Path, output: Path) -> None:
    """Plot saved points after V/E exercises (currently a stub)."""
    if output.exists():
        raise click.ClickException(f"Output already exists: {output}")
    from .experiment import run_plotting
    logger.info("Requested plot samples={} output={}", samples, output)
    run_plotting(samples, output)
