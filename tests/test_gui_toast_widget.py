import unittest

try:
    from gui_qt.toast_widget import ToastNotification, ToastStyle
except ImportError as exc:
    ToastNotification = None  # type: ignore[assignment]
    ToastStyle = None  # type: ignore[assignment]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(ToastNotification is None, f"GUI dependencies are unavailable: {IMPORT_ERROR}")
class ToastWidgetTests(unittest.TestCase):
    def test_normalize_message_strips_leading_status_glyphs(self):
        self.assertEqual(
            ToastNotification._normalize_message("✓ 设置已成功保存"),
            "设置已成功保存",
        )
        self.assertEqual(
            ToastNotification._normalize_message("✔设置已成功保存"),
            "设置已成功保存",
        )
        self.assertEqual(
            ToastNotification._normalize_message("⚠ 需要注意"),
            "需要注意",
        )
        self.assertEqual(
            ToastNotification._normalize_message("⚠️ 需要注意"),
            "需要注意",
        )
        self.assertEqual(
            ToastNotification._normalize_message("✘ 失败"),
            "失败",
        )
        self.assertEqual(
            ToastNotification._normalize_message("设置已成功保存"),
            "设置已成功保存",
        )
        self.assertEqual(
            ToastNotification._normalize_message("XML 解析完成"),
            "XML 解析完成",
        )
        # Ordinary punctuation / math prefixes must not be treated as status icons.
        self.assertEqual(
            ToastNotification._normalize_message("!important notice"),
            "!important notice",
        )
        self.assertEqual(
            ToastNotification._normalize_message("× 3 retries"),
            "× 3 retries",
        )
        # Glyph-only input strips to empty (widget still shows the style icon).
        self.assertEqual(
            ToastNotification._normalize_message("✓"),
            "",
        )
        self.assertEqual(
            ToastNotification._normalize_message("✔  "),
            "",
        )

    def test_icon_text_defined_for_all_styles(self):
        from gui_qt import toast_widget as mod

        for style in ToastStyle:
            self.assertIn(style, mod._ICON_TEXT)
            self.assertTrue(mod._ICON_TEXT[style])


if __name__ == "__main__":
    unittest.main()
