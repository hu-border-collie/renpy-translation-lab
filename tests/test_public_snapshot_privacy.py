"""Enforce generic privacy boundaries without storing private marker values."""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "agent-tools",
}


def _public_text_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode == 0:
        candidates = (
            REPO_ROOT / item.decode("utf-8", errors="surrogateescape")
            for item in completed.stdout.split(b"\0")
            if item
        )
    else:
        candidates = REPO_ROOT.rglob("*")
    files: list[Path] = []
    for path in candidates:
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            sample = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in sample[:4096]:
            continue
        files.append(path)
    return files


class PublicSnapshotPrivacyTests(unittest.TestCase):
    def test_archived_prototype_has_no_windows_absolute_paths(self) -> None:
        archive_root = REPO_ROOT / "docs" / "archive" / "prototypes"
        windows_absolute_path = re.compile(r"(?i)(?<![A-Za-z0-9])[a-z]:[\\/]")
        findings: list[str] = []
        for path in _public_text_files():
            if archive_root not in path.parents:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if windows_absolute_path.search(text):
                findings.append(str(path.relative_to(REPO_ROOT)))
        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
