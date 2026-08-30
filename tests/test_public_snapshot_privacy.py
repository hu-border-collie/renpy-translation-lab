"""Enforce generic privacy boundaries without storing private marker values."""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOTYPE_ROOT = REPO_ROOT / "docs" / "archive" / "prototypes"
SENSITIVE_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "translation_plan_minimal"
CALIBRATION_BASELINE = REPO_ROOT / "docs" / "plans" / "quality_calibration_baseline.md"

PERSONAL_HOME_PATH = re.compile(
    r"(?ix)"
    r"(?:[a-z]:[\\/]+users[\\/]+"
    r"(?!user(?:name)?(?:[\\/]|$)|runneradmin(?:[\\/]|$)|runner~1(?:[\\/]|$)))"
    r"|(?:/(?:users|home)/(?!(?:user|username)(?:/|$)))"
    r"|(?:/private" r"/var/)"
)
WINDOWS_ABSOLUTE_PATH = re.compile(r"(?i)(?<![A-Za-z0-9])[a-z]:[\\/]")
UNC_PATH = re.compile(r"(?<![\\])\\\\[^\\\s]+[\\][^\\\s]+")
PROJECT_DIRECTORY = re.compile(r"\bGame_[A-Za-z0-9][A-Za-z0-9_-]*\b")


def _tracked_text_files() -> list[Path]:
    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AssertionError("git ls-files is required for the privacy scan") from exc

    files: list[Path] = []
    for item in completed.stdout.split(b"\0"):
        if not item:
            continue
        path = REPO_ROOT / item.decode("utf-8", errors="surrogateescape")
        if not path.is_file():
            continue
        try:
            sample = path.read_bytes()
        except OSError as exc:
            raise AssertionError(f"cannot read tracked file: {path}") from exc
        if b"\x00" not in sample[:4096]:
            files.append(path)
    return files


class PublicSnapshotPrivacyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tracked_text_files = _tracked_text_files()

    def test_tracked_text_has_no_personal_home_paths(self) -> None:
        findings: list[str] = []
        for path in self.tracked_text_files:
            text = path.read_text(encoding="utf-8", errors="replace")
            if PERSONAL_HOME_PATH.search(text):
                findings.append(str(path.relative_to(REPO_ROOT)))
        self.assertEqual(findings, [])

    def test_archived_prototypes_have_no_absolute_paths(self) -> None:
        findings: list[str] = []
        for path in self.tracked_text_files:
            if PROTOTYPE_ROOT not in path.parents:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if WINDOWS_ABSOLUTE_PATH.search(text) or UNC_PATH.search(text):
                findings.append(str(path.relative_to(REPO_ROOT)))
        self.assertEqual(findings, [])

    def test_sensitive_docs_and_fixtures_have_no_project_directories(self) -> None:
        findings: list[str] = []
        for path in self.tracked_text_files:
            in_sensitive_scope = (
                SENSITIVE_FIXTURE_ROOT in path.parents
                or PROTOTYPE_ROOT in path.parents
                or path == CALIBRATION_BASELINE
            )
            if not in_sensitive_scope:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if PROJECT_DIRECTORY.search(text):
                findings.append(str(path.relative_to(REPO_ROOT)))
        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
