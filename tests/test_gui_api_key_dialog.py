import unittest

from gui_qt.api_key_dialog import commit_pending_key, mask_api_key


class GuiApiKeyDialogTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()