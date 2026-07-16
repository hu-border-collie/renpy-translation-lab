import contextlib
import io
import unittest

import gemini_translate
import gemini_translate_batch
from project_version import __version__
from scripts import build_source_release


class ReleaseVersionTests(unittest.TestCase):
    def test_version_is_semantic_version(self):
        self.assertRegex(__version__, r"^\d+\.\d+\.\d+$")

    def test_sync_cli_reports_project_version(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            gemini_translate.main(["--version"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn(__version__, output.getvalue())

    def test_batch_cli_reports_project_version(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            gemini_translate_batch.main(["--version"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn(__version__, output.getvalue())

    def test_release_builder_parses_lfs_pointer(self):
        pointer = (
            b"version https://git-lfs.github.com/spec/v1\n"
            b"oid sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n"
            b"size 123\n"
        )
        self.assertEqual(
            build_source_release._parse_lfs_pointer(pointer),
            ("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", 123),
        )


if __name__ == "__main__":
    unittest.main()
