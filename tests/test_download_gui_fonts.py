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
            with self.assertRaises(download_gui_fonts.FontInstallError) as ctx:
                download_gui_fonts._archive_tool()
        message = str(ctx.exception)
        self.assertIn("bsdtar", message)
        self.assertIn("libarchive-tools", message)

    def test_tool_supports_rar_accepts_bsdtar_and_libarchive(self):
        with mock.patch(
            "scripts.download_gui_fonts._tool_version_text",
            return_value="bsdtar 3.6.2 - libarchive 3.6.2 zlib/...",
        ):
            self.assertTrue(download_gui_fonts._tool_supports_rar("/usr/bin/bsdtar"))
            self.assertTrue(download_gui_fonts._tool_supports_rar("/usr/bin/tar"))

    def test_tool_supports_rar_rejects_gnu_tar(self):
        with mock.patch(
            "scripts.download_gui_fonts._tool_version_text",
            return_value="tar (GNU tar) 1.34\nCopyright (C) 2021 Free Software Foundation, Inc.",
        ):
            self.assertFalse(download_gui_fonts._tool_supports_rar("/usr/bin/tar"))

    def test_archive_tool_rejects_gnu_tar_only_environment(self):
        def fake_which(name):
            if name == "tar":
                return "/usr/bin/tar"
            return None

        with (
            mock.patch("scripts.download_gui_fonts.shutil.which", side_effect=fake_which),
            mock.patch(
                "scripts.download_gui_fonts._tool_version_text",
                return_value="tar (GNU tar) 1.34",
            ),
        ):
            with self.assertRaises(download_gui_fonts.FontInstallError) as ctx:
                download_gui_fonts._archive_tool()
        message = str(ctx.exception)
        self.assertIn("RAR", message)
        self.assertIn("/usr/bin/tar", message)
        self.assertIn("libarchive-tools", message)

    def test_archive_tool_prefers_bsdtar_over_gnu_tar(self):
        def fake_which(name):
            return {
                "bsdtar": "/usr/bin/bsdtar",
                "tar": "/usr/bin/tar",
            }.get(name)

        def fake_version(executable):
            if executable.endswith("bsdtar"):
                return "bsdtar 3.6.2 - libarchive 3.6.2"
            return "tar (GNU tar) 1.34"

        with (
            mock.patch("scripts.download_gui_fonts.shutil.which", side_effect=fake_which),
            mock.patch(
                "scripts.download_gui_fonts._tool_version_text",
                side_effect=fake_version,
            ),
        ):
            self.assertEqual(download_gui_fonts._archive_tool(), "/usr/bin/bsdtar")

    def test_archive_tool_accepts_macos_libarchive_tar(self):
        def fake_which(name):
            if name == "tar":
                return "/usr/bin/tar"
            return None

        with (
            mock.patch("scripts.download_gui_fonts.shutil.which", side_effect=fake_which),
            mock.patch(
                "scripts.download_gui_fonts._tool_version_text",
                return_value="bsdtar 3.5.3 - libarchive 3.5.3 zlib/1.2.11",
            ),
        ):
            self.assertEqual(download_gui_fonts._archive_tool(), "/usr/bin/tar")


if __name__ == "__main__":
    unittest.main()
