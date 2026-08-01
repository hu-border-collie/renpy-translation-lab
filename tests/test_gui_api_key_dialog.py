import unittest

from gui_qt.api_key_helpers import commit_pending_key, mask_api_key

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.api_key_dialog import ApiKeyDialog
except ImportError as exc:
    QApplication = None
    ApiKeyDialog = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support


class GuiApiKeyHelperTests(unittest.TestCase):
    def test_mask_api_key_hides_middle_and_keeps_suffix(self):
        self.assertEqual(mask_api_key("abcdefghijklmnop"), "********mnop")
        self.assertEqual(mask_api_key("ab"), "************")
        self.assertEqual(mask_api_key("   "), "(空)")

    def test_commit_pending_key_keeps_existing_list_when_input_empty(self):
        keys, error = commit_pending_key(["existing"], "   ")

        self.assertIsNone(error)
        self.assertEqual(keys, ["existing"])

    def test_commit_pending_key_appends_new_value(self):
        keys, error = commit_pending_key(["existing"], "  new-key  ")

        self.assertIsNone(error)
        self.assertEqual(keys, ["existing", "new-key"])

    def test_commit_pending_key_rejects_duplicate(self):
        keys, error = commit_pending_key(["existing"], "existing")

        self.assertEqual(error, "duplicate")
        self.assertEqual(keys, ["existing"])

    def test_commit_pending_key_detects_whitespace_variant_duplicate(self):
        keys, error = commit_pending_key(["key123 "], "key123")

        self.assertEqual(error, "duplicate")
        self.assertEqual(keys, ["key123 "])


@gui_test_support.skip_unless_gui(ApiKeyDialog is None, IMPORT_ERROR)
class GuiApiKeyDialogActiveKeyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_active_key_marker_and_selection(self):
        dialog = ApiKeyDialog(
            None,
            keys=["alpha-key-1111", "beta-key-2222"],
            active_index=0,
            support_active_key=True,
            title="管理测试 Key",
        )
        try:
            self.assertIn("当前使用", dialog.key_list.item(0).text())
            self.assertNotIn("当前使用", dialog.key_list.item(1).text())
            dialog.key_list.setCurrentRow(1)
            dialog._on_set_active_selected()
            self.assertEqual(dialog.result_active_index(), 1)
            self.assertIn("当前使用", dialog.key_list.item(1).text())
            dialog._on_remove_selected()
            self.assertEqual(dialog.result_keys(), ["alpha-key-1111"])
            self.assertEqual(dialog.result_active_index(), 0)
        finally:
            dialog.close()
            dialog.deleteLater()


if __name__ == "__main__":
    unittest.main()