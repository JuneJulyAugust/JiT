"""Run with: .venv/bin/python -m unittest discover -s tests -v"""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DemoCLITest(unittest.TestCase):
    def run_demo(self, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / 'sample_jit.py'), *args],
            text=True, capture_output=True, cwd=ROOT,
        )

    def test_help_without_model_dependencies(self):
        result = self.run_demo('--help')
        self.assertEqual(result.returncode, 0)
        self.assertIn('--checkpoint', result.stdout)

    def test_missing_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_demo('--checkpoint', str(Path(directory) / 'missing.pth'))
        self.assertEqual(result.returncode, 2)
        self.assertIn('Checkpoint not found', result.stderr)

    def test_invalid_settings_fail_before_loading_weights(self):
        with tempfile.NamedTemporaryFile(suffix='.pth') as checkpoint:
            for option, value, message in [
                ('--steps', '0', '--steps must be positive'),
                ('--class-ids', '-1', 'class indices'),
                ('--class-ids', '1000', 'class indices'),
            ]:
                with self.subTest(option=option, value=value):
                    result = self.run_demo('--checkpoint', checkpoint.name, option, value)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(message, result.stderr)

    def test_existing_output_is_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'checkpoint.pth'
            checkpoint.touch()
            result = self.run_demo('--checkpoint', str(checkpoint), '--output-dir', directory)
            self.assertEqual(result.returncode, 2)
            self.assertIn('must be empty or new', result.stderr)
            self.assertTrue(checkpoint.exists())


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('einops')
                     and importlib.util.find_spec('numpy'), 'PyTorch/einops/NumPy required')
class PortableModelTest(unittest.TestCase):
    def test_cpu_attention(self):
        os.environ['TORCHDYNAMO_DISABLE'] = '1'
        import torch
        from model_jit import scaled_dot_product_attention
        query, key, value = [torch.randn(1, 2, 4, 8) for _ in range(3)]
        expected = torch.softmax(query @ key.transpose(-2, -1) / 8 ** 0.5, dim=-1) @ value
        torch.testing.assert_close(scaled_dot_product_attention(query, key, value), expected)

    def test_rotary_buffers_move_without_changing_checkpoint_keys(self):
        import torch
        from util.model_util import VisionRotaryEmbeddingFast
        for context in [0, 2]:
            with self.subTest(context=context):
                rope = VisionRotaryEmbeddingFast(dim=4, pt_seq_len=2, num_cls_token=context)
                self.assertEqual(dict(rope.state_dict()), {})
                self.assertIn('freqs_cos', dict(rope.named_buffers()))
                rope.to(dtype=torch.float64)
                self.assertEqual(rope.freqs_cos.dtype, torch.float64)
                result = rope(torch.ones(1, 1, 4 + context, 8, dtype=torch.float64))
                self.assertTrue(torch.isfinite(result).all())


if __name__ == '__main__':
    unittest.main()
