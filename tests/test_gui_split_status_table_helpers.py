import unittest

from gui_qt.split_status_table_helpers import (
    split_action_item_payload,
    SPLIT_ACTION_BUTTON_HEIGHT,
    SPLIT_ACTION_BUTTON_MIN_WIDTH,
    is_split_action_column,
    split_action_button_colors,
    split_action_button_rect,
)


class GuiSplitStatusTableHelperTests(unittest.TestCase):
    def test_is_split_action_column(self):
        self.assertTrue(is_split_action_column(5))
        self.assertTrue(is_split_action_column(11))
        self.assertFalse(is_split_action_column(4))
        self.assertFalse(is_split_action_column(6))

    def test_split_action_button_rect_centers_button(self):
        left, top, width, height = split_action_button_rect(132.0, 44.0)
        self.assertEqual(width, float(SPLIT_ACTION_BUTTON_MIN_WIDTH))
        self.assertEqual(height, float(SPLIT_ACTION_BUTTON_HEIGHT))
        self.assertGreater(left, 0.0)
        self.assertGreater(top, 0.0)

    def test_split_action_button_rect_clamps_to_small_cells(self):
        left, top, width, height = split_action_button_rect(40.0, 20.0)
        self.assertLessEqual(width, 24.0)
        self.assertLessEqual(height, 8.0)
        self.assertGreaterEqual(left, 0.0)
        self.assertGreaterEqual(top, 0.0)

    def test_split_action_button_colors_support_light_and_dark(self):
        light_normal = split_action_button_colors(dark=False, state="normal")
        dark_hover = split_action_button_colors(dark=True, state="hover")
        self.assertNotEqual(light_normal, dark_hover)

    def test_split_action_item_payload_only_for_selectable_entries(self):
        self.assertIsNone(
            split_action_item_payload(
                selectable=False,
                manifest_path=r"C:\pkg\manifest.json",
                part_label="part01/03",
            )
        )
        payload = split_action_item_payload(
            selectable=True,
            manifest_path=r"C:\pkg\manifest.json",
            part_label="part01/03",
        )
        self.assertEqual(payload["manifest_path"], r"C:\pkg\manifest.json")
        self.assertEqual(payload["part_label"], "part01/03")


if __name__ == "__main__":
    unittest.main()