import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import download_gui_fonts


class DownloadGuiFontsTests(unittest.TestCase):
    def test_verify_accepts_matching_digest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "font.ttf"
            path.write_bytes(b"font data")
            expected = hashlib.sha256(b"font data").hexdigest()
            download_gui_fonts._verify(path, expected, "test font")

    def test_verify_rejects_mismatched_digest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "font.ttf"
            path.write_bytes(b"font data")
            with self.assertRaises(download_gui_fonts.FontInstallError):
                download_gui_fonts._verify(path, "0" * 64, "test font")

    def test_archive_tool_requires_supported_extractor(self):
        with mock.patch("scripts.download_gui_fonts.shutil.which", return_value=None):
            with self.assertRaises(download_gui_fonts.FontInstallError):
                download_gui_fonts._archive_tool()


if __name__ == "__main__":
    unittest.main()
