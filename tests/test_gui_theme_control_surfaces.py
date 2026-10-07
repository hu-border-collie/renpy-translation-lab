"""Isolated palette-render checks for themed GUI control surfaces."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


class GuiThemeControlSurfaceTests(unittest.TestCase):
    def test_opposite_system_palettes_in_isolated_qt_process(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["QT_QPA_PLATFORM"] = "offscreen"
        environment["RTL_DISABLE_LITELLM_WARMUP"] = "1"
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "unittest",
                "tests.gui_theme_surface_probe",
                "-v",
            ],
            cwd=repository_root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"isolated Qt probe failed:\n{result.stdout}\n{result.stderr}",
        )
        self.assertIn("Ran 4 tests", result.stderr)
        self.assertIn("OK", result.stderr)


if __name__ == "__main__":
    unittest.main()
