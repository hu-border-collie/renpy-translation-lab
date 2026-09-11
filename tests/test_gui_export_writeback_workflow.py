"""Pure argument-builder tests for the GUI export executor."""
from __future__ import annotations

import unittest

from gui_qt.translation_workflow import build_export_writeback_cli_args


class GuiExportWritebackWorkflowTests(unittest.TestCase):
    def test_export_only_args_use_shared_apply_cli(self):
        args = build_export_writeback_cli_args(
            "manifest.json",
            mode="export-only",
            export_root="exports/demo",
        )
        self.assertEqual(
            args,
            [
                "apply",
                "manifest.json",
                "--export-only",
                "exports/demo",
                "--output",
                "json",
                "--non-interactive",
            ],
        )

    def test_apply_export_args_use_export_dir(self):
        args = build_export_writeback_cli_args(
            "manifest.json",
            mode="apply-export",
            export_root="exports/demo",
        )
        self.assertEqual(args[2], "--export-dir")
        self.assertIn("exports/demo", args)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            build_export_writeback_cli_args(
                "manifest.json",
                mode="copy",
                export_root="exports/demo",
            )


if __name__ == "__main__":
    unittest.main()
