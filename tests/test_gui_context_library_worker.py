"""Context-library status collection must stay off the GUI thread."""

from __future__ import annotations

import unittest
from unittest import mock

try:
    from gui_qt.context_library_worker import collect_context_library_status
except ImportError as exc:
    collect_context_library_status = None  # type: ignore[assignment]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(
    collect_context_library_status is None,
    f"GUI dependencies are unavailable: {IMPORT_ERROR}",
)
class ContextLibraryStatusCollectionTests(unittest.TestCase):
    def test_collects_fingerprint_and_status_for_one_project(self) -> None:
        status = {"overall_status": "published", "injectable": True}
        with (
            mock.patch(
                "gemini_translate_batch.compute_current_project_analysis_fingerprint",
                return_value="fp-1",
            ) as fingerprint,
            mock.patch(
                "project_analysis.collect_project_analysis_status",
                return_value=status,
            ) as collect_status,
            mock.patch(
                "project_analysis.format_status_label",
                return_value="已启用",
            ),
            mock.patch(
                "gui_qt.bootstrap_report.read_batch_context_flags",
                return_value={"rag_enabled": True},
            ) as read_flags,
        ):
            config = {"batch": {"rag": {"enabled": True}}}
            result = collect_context_library_status("C:/Games/Demo/work", config)

        self.assertEqual(result.base_dir, "C:/Games/Demo/work")
        self.assertEqual(result.live_fingerprint, "fp-1")
        self.assertEqual(result.status, status)
        self.assertEqual(result.label, "已启用")
        self.assertTrue(result.context_flags["rag_enabled"])
        read_flags.assert_called_once_with(
            config,
            game_root="C:/Games/Demo/work",
        )
        fingerprint.assert_called_once_with("C:/Games/Demo/work")
        collect_status.assert_called_once_with(
            base_dir="C:/Games/Demo/work",
            expected_source_fingerprint="fp-1",
        )

    def test_collection_failure_returns_readable_result(self) -> None:
        with mock.patch(
            "gemini_translate_batch.compute_current_project_analysis_fingerprint",
            side_effect=OSError("broken store"),
        ):
            result = collect_context_library_status("C:/Games/Demo/work")

        self.assertIsNone(result.status)
        self.assertIn("读取失败", result.label)
        self.assertIn("broken store", result.error)


if __name__ == "__main__":
    unittest.main()
