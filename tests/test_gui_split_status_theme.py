import unittest

try:
    from gui_qt.app import MainWindow
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(MainWindow is None, f"GUI dependencies are unavailable: {IMPORT_ERROR}")
class GuiSplitStatusThemeTests(unittest.TestCase):
    def setUp(self):
        self.window = MainWindow.__new__(MainWindow)
        self.window._qt_app = None

    def test_effective_theme_is_dark_follows_explicit_preference(self):
        self.window._theme_preference = "dark"
        self.assertTrue(self.window._effective_theme_is_dark())
        self.window._theme_preference = "light"
        self.assertFalse(self.window._effective_theme_is_dark())

    def test_split_status_row_colors_switch_with_theme(self):
        self.window._theme_preference = "light"
        light = self.window._split_status_row_colors("applied")
        self.window._theme_preference = "dark"
        dark = self.window._split_status_row_colors("applied")
        self.assertNotEqual(light, dark)

    def test_refresh_split_status_table_after_theme_change_updates_rows(self):
        from gui_qt.split_batch import SplitManifestEntry

        entry = SplitManifestEntry(
            index=1,
            total=2,
            manifest_path="/tmp/part01/manifest.json",
            display_name="part01",
            job_name="",
            job_state="",
            safety_level="",
            item_count=10,
            chunk_count=1,
            applied=True,
            has_result=True,
            selectable=False,
            needs_submit=False,
            needs_status=False,
            status_label="已写回",
            status_kind="applied",
        )
        calls: list[str] = []
        self.window._split_status_entries = [entry]
        self.window._split_status_selected_manifest_path = entry.manifest_path
        self.window._configure_split_status_table_hover_palette = (
            lambda: calls.append("palette")
        )
        self.window._update_split_status_selection_ui = (
            lambda path: calls.append(f"ui:{path}")
        )

        self.window._refresh_split_status_table_after_theme_change()

        self.assertEqual(
            calls,
            ["palette", f"ui:{entry.manifest_path}"],
        )

    def test_refresh_split_status_table_after_theme_change_skips_empty_entries(self):
        calls: list[str] = []
        self.window._split_status_entries = []
        self.window._configure_split_status_table_hover_palette = (
            lambda: calls.append("palette")
        )
        self.window._update_split_status_selection_ui = (
            lambda _path: calls.append("ui")
        )

        self.window._refresh_split_status_table_after_theme_change()

        self.assertEqual(calls, ["palette"])


if __name__ == "__main__":
    unittest.main()