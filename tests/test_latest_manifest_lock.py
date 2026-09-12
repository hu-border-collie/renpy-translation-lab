"""Shared latest-manifest writer lock and GUI/CLI coordination tests."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import atomic_io
import gemini_translate_batch as batch_mod
from gui_qt.project_state import ProjectState


class LatestManifestSharedWriterTests(unittest.TestCase):
    def test_physical_writer_uses_latest_lock_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            latest = Path(tmp) / "latest_manifest.txt"
            seen: list[str] = []

            @contextmanager
            def fake_lock(lock_path, **_kwargs):
                seen.append(str(lock_path))
                yield {}

            with mock.patch.object(atomic_io, "exclusive_file_lock", fake_lock):
                atomic_io.write_latest_manifest_locked(latest, "target")
            self.assertEqual(seen, [f"{latest}.lock"])
            self.assertEqual(latest.read_text(encoding="utf-8"), "target")

    def test_cli_wrapper_delegates_to_shared_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            latest = str(Path(tmp) / "latest_manifest.txt")
            with (
                mock.patch.object(batch_mod, "LATEST_MANIFEST_FILE", latest),
                mock.patch.object(batch_mod, "ensure_batch_dirs"),
                mock.patch.object(
                    batch_mod,
                    "write_latest_manifest_locked",
                ) as writer,
            ):
                batch_mod.remember_latest_manifest("demo/manifest.json")
            writer.assert_called_once_with(
                latest,
                "demo/manifest.json",
                timeout=batch_mod._LATEST_MANIFEST_LOCK_TIMEOUT,
                stale_after=batch_mod._LATEST_MANIFEST_LOCK_STALE_AFTER,
            )

    def test_gui_wrapper_delegates_to_shared_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp) / "batch_jobs"

            class DummyState:
                def get_logs_dir(self):
                    return logs_dir

            state = DummyState()
            with mock.patch(
                "gui_qt.project_state.write_latest_manifest_locked"
            ) as writer:
                state.remember_latest_manifest_path = types.MethodType(
                    ProjectState.remember_latest_manifest_path,
                    state,
                )
                state.remember_latest_manifest_path("demo/manifest.json")
            writer.assert_called_once_with(
                logs_dir / "latest_manifest.txt",
                "demo/manifest.json",
            )

    def test_gui_writer_waits_for_holder_and_wins_after_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = root / "latest_manifest.txt"
            lock = atomic_io.latest_manifest_lock_path(latest)
            marker = root / "held"
            script = (
                "import sys, time\n"
                "from pathlib import Path\n"
                "import atomic_io\n"
                "lock, latest, marker = sys.argv[1:4]\n"
                "with atomic_io.exclusive_file_lock(lock, timeout=5.0):\n"
                "    Path(marker).write_text('held')\n"
                "    time.sleep(0.6)\n"
                "    atomic_io.atomic_write_text(latest, 'holder-B')\n"
            )
            holder = subprocess.Popen(
                [sys.executable, "-c", script, str(lock), str(latest), str(marker)],
                cwd=str(Path(__file__).resolve().parents[1]),
            )
            try:
                deadline = time.time() + 5.0
                while time.time() < deadline and not marker.exists():
                    time.sleep(0.02)
                self.assertTrue(marker.exists())

                class DummyState:
                    def get_logs_dir(self):
                        return latest.parent

                state = DummyState()
                state.remember_latest_manifest_path = types.MethodType(
                    ProjectState.remember_latest_manifest_path,
                    state,
                )
                started = time.monotonic()
                state.remember_latest_manifest_path("gui-writer-C")
                elapsed = time.monotonic() - started
                self.assertGreaterEqual(elapsed, 0.35)
                self.assertEqual(latest.read_text(encoding="utf-8"), "gui-writer-C")
            finally:
                holder.wait(timeout=10.0)

    def test_gui_writer_before_cli_cas_returns_retained_newer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = root / "latest_manifest.txt"
            latest.write_text("writer-A", encoding="utf-8")
            done = root / "gui-done"
            script = (
                "import sys, types\n"
                "from pathlib import Path\n"
                "from gui_qt.project_state import ProjectState\n"
                "latest, done = Path(sys.argv[1]), Path(sys.argv[2])\n"
                "class Dummy:\n"
                "    def get_logs_dir(self):\n"
                "        return latest.parent\n"
                "state = Dummy()\n"
                "state.remember_latest_manifest_path = types.MethodType(\n"
                "    ProjectState.remember_latest_manifest_path, state)\n"
                "state.remember_latest_manifest_path('gui-writer-B')\n"
                "done.write_text('done')\n"
            )
            process = subprocess.Popen(
                [sys.executable, "-c", script, str(latest), str(done)],
                cwd=str(Path(__file__).resolve().parents[1]),
            )
            try:
                deadline = time.time() + 5.0
                while time.time() < deadline and not done.exists():
                    if process.poll() is not None:
                        break
                    time.sleep(0.02)
                self.assertTrue(done.exists())
                with (
                    mock.patch.object(batch_mod, "LATEST_MANIFEST_FILE", str(latest)),
                    mock.patch.object(batch_mod, "ensure_batch_dirs"),
                ):
                    retained = batch_mod.remember_latest_manifest_if_unchanged(
                        "writer-A",
                        "writer-C",
                    )
                self.assertEqual(retained["status"], "retained_newer")
                self.assertEqual(
                    latest.read_text(encoding="utf-8"),
                    "gui-writer-B",
                )
                with (
                    mock.patch.object(batch_mod, "LATEST_MANIFEST_FILE", str(latest)),
                    mock.patch.object(batch_mod, "ensure_batch_dirs"),
                ):
                    advanced = batch_mod.remember_latest_manifest_if_unchanged(
                        "gui-writer-B",
                        "writer-D",
                    )
                self.assertEqual(advanced["status"], "advanced")
                self.assertEqual(latest.read_text(encoding="utf-8"), "writer-D")
            finally:
                process.wait(timeout=10.0)

    def test_no_direct_latest_manifest_writes_outside_shared_service(self):
        repo_root = Path(__file__).resolve().parents[1]
        allowed = {"atomic_io.py"}
        offenders: list[str] = []
        write_pattern = re.compile(r"\.write_text\s*\(|atomic_write_text\s*\(")
        shared_markers = (
            "write_latest_manifest_locked",
            "compare_and_swap_latest_manifest_locked",
        )
        for path in repo_root.rglob("*.py"):
            relative = path.relative_to(repo_root).as_posix()
            if relative.startswith(("tests/", "logs/", "__pycache__/")):
                continue
            if relative in allowed:
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                continue
            for index, line in enumerate(lines):
                if not write_pattern.search(line):
                    continue
                window = "\n".join(lines[max(0, index - 3) : index + 1])
                if "latest" not in window.lower() and "LATEST_MANIFEST_FILE" not in window:
                    continue
                if any(marker in window for marker in shared_markers):
                    continue
                offenders.append(f"{relative}:{index + 1}: {line.strip()}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
