"""Safe launcher checks that never start Portal 2 or install packages."""

from __future__ import annotations

# Python standard library: invoke launchers safely and inspect their exit codes.
import os
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LauncherTest(unittest.TestCase):
    def run_launcher(self, name: str, *args: str, env: dict[str, str] | None = None):
        return subprocess.run(
            [str(ROOT / name), *args], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=20, check=False,
        )

    def test_model_launchers_require_explicit_checkpoint(self):
        launchers = ["run_model.sh", "run_model_ssh.sh"]
        if shutil.which("fish"):
            launchers.append("run_model.fish")
        for launcher in launchers:
            with self.subTest(launcher=launcher):
                result = self.run_launcher(launcher)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("--checkpoint PATH", result.stderr)

    def test_model_launcher_rejects_missing_checkpoint_file(self):
        result = self.run_launcher("run_model.sh", "--checkpoint", "missing-checkpoint.pt")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Checkpoint not found", result.stderr)

    def test_model_launcher_help_needs_no_checkpoint(self):
        result = self.run_launcher("run_model.sh", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--checkpoint", result.stdout)

    def test_hammer_launcher_fails_cleanly_for_unconfigured_path(self):
        env = dict(os.environ, SHIONN_HAMMER_DIR="/nonexistent/shionn-hammer")
        result = self.run_launcher("hammerpp-home.sh", env=env)
        self.assertEqual(result.returncode, 1)
        self.assertIn("SHIONN_HAMMER_DIR", result.stderr)

    def test_installer_help_is_non_mutating(self):
        result = self.run_launcher("install_dependencies.sh", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--permissions", result.stdout)
