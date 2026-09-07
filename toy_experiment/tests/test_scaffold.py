"""Working infrastructure only. Add numerical tests as you complete TODO.md."""

from dataclasses import replace
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from click.testing import CliRunner
from loguru import logger

from jit_toy.cli import main
from jit_toy.config import ExperimentConfig, experiment_matrix, load_config
from jit_toy.exercises import ExerciseNotImplemented
from jit_toy.logging import configure_logging


class ConfigTests(unittest.TestCase):
    def test_suite_covers_all_cases_with_matched_settings(self):
        base = ExperimentConfig(seed=42, projection_seed=71, train_steps=17)
        cases = experiment_matrix(base)
        self.assertEqual({(c.observed_dim, c.prediction) for c in cases},
                         {(d, p) for d in (2, 8, 16, 512) for p in ("x", "eps", "v")})
        self.assertEqual(len(cases), 12)
        for case in cases:
            self.assertEqual(replace(case, observed_dim=base.observed_dim, prediction=base.prediction), base)

    def test_invalid_configs_fail_early(self):
        for values in (
            {"observed_dim": 1}, {"intrinsic_dim": 3}, {"prediction": "noise"},
            {"batch_size": True}, {"train_steps": 1.5}, {"seed": -1},
            {"time_eps": 0.5}, {"time_eps": 0}, {"time_std": float("nan")},
            {"time_mean": "oops"}, {"learning_rate": 0}, {"solver": "other"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ExperimentConfig(**values)

    def test_json_partial_overrides_and_unknown_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"prediction":"eps", "seed":7}')
            config = load_config(path)
            self.assertEqual(config.prediction, "eps")
            self.assertEqual(config.seed, 7)
            self.assertEqual(config.hidden_dim, 256)
            path.write_text('{"lerning_rate": 0.1}')
            with self.assertRaisesRegex(ValueError, "unknown configuration"):
                load_config(path)
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                load_config(path)


class CLITests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.addCleanup(logger.remove)

    def test_help_and_plan_do_not_import_torch(self):
        script = (
            "import sys; from click.testing import CliRunner; from jit_toy.cli import main; "
            "r=CliRunner(); a=r.invoke(main,['--help']); b=r.invoke(main,['plan','--suite']); "
            "assert a.exit_code==b.exit_code==0; assert 'torch' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_plan_stdout_is_json_and_logs_have_source_location(self):
        # A subprocess checks real stream separation across Click versions;
        # older CliRunner releases combine stderr with stdout by default.
        result = subprocess.run([sys.executable, "-m", "jit_toy", "plan", "--suite"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads(result.stdout)), 12)
        self.assertRegex(result.stderr, r"cli\.py:\d+\) \| Validated 12")

    def test_bad_config_produces_user_facing_error(self):
        with self.runner.isolated_filesystem():
            Path("bad.json").write_text('{"observed_dim":0}')
            result = self.runner.invoke(main, ["plan", "--config", "bad.json"])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Invalid configuration", result.output)

    def test_training_reports_exercise_without_creating_run(self):
        # Stub the integration boundary so this remains a CLI test even after
        # the learner implements E01. It does not assert E01 stays unfinished.
        with self.runner.isolated_filesystem(), patch(
            "jit_toy.experiment.run_training", side_effect=ExerciseNotImplemented("E01", "test boundary")
        ):
            result = self.runner.invoke(main, ["train", "--output", "new-run"])
            self.assertEqual(result.exit_code, 1, result.output)
            self.assertIn("E01", result.output)
            self.assertIn("TODO.md", result.output)
            self.assertFalse(Path("new-run").exists())

    def test_existing_training_output_is_preserved(self):
        with self.runner.isolated_filesystem():
            Path("old-run").mkdir()
            sentinel = Path("old-run/important.txt")
            sentinel.write_text("keep")
            result = self.runner.invoke(main, ["train", "--output", "old-run"])
            self.assertEqual(result.exit_code, 1)
            self.assertIn("already exists", result.output)
            self.assertEqual(sentinel.read_text(), "keep")

    def test_sample_and_plot_have_explicit_exercise_boundaries(self):
        with self.runner.isolated_filesystem():
            Path("input.pt").touch()
            for command, option, exercise in (("sample", "--checkpoint", "E02"), ("plot", "--samples", "E03")):
                target = "run_sampling" if command == "sample" else "run_plotting"
                with self.subTest(command=command), patch(
                    f"jit_toy.experiment.{target}", side_effect=ExerciseNotImplemented(exercise, "test boundary")
                ):
                    result = self.runner.invoke(main, [command, option, "input.pt", "--output", "new-output"])
                    self.assertEqual(result.exit_code, 1)
                    self.assertIn(exercise, result.output)
                    self.assertFalse(Path("new-output").exists())

    def test_file_logger_reports_actual_call_site_once(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "trace.log"
            configure_logging(log_file=log_path)
            configure_logging(log_file=log_path)  # Reconfiguration must not duplicate sinks.
            line = inspect.currentframe().f_lineno + 1
            logger.info("one marker")
            logger.remove()
            content = log_path.read_text()
            self.assertEqual(content.count("one marker"), 1)
            self.assertIn(f"{Path(__file__).resolve()}:{line}", content)


if __name__ == "__main__":
    unittest.main()
