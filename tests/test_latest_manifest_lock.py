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


_SHARED_WRITER_MARKERS = (
    "write_latest_manifest_locked",
    "compare_and_swap_latest_manifest_locked",
)
_DIRECT_LATEST_MUTATION_PATTERNS = (
    re.compile(r"\.write_text\s*\("),
    re.compile(r"\.write_bytes\s*\("),
    re.compile(r"\batomic_write_text\s*\("),
    re.compile(r"\batomic_write\s*\("),
    re.compile(r"\bopen\s*\([^)]*[\"'][wax]"),
    re.compile(r"\.open\s*\([^)]*[\"'][wax]"),
    re.compile(r"\bos\.(?:replace|rename)\s*\("),
    re.compile(r"\bshutil\.(?:copy|copyfile|move)\s*\("),
    re.compile(r"\.replace\s*\("),
    re.compile(r"\.rename\s*\("),
    re.compile(r"\.unlink\s*\("),
)


def _direct_latest_write_offenders(relative_path: str, source: str) -> list[str]:
    """Return limited-pattern offenders for a source file.

    The scan intentionally covers representative direct-write idioms rather
    than proving that no arbitrary writer can exist; see the contract's known
    limitations.
    """

    lines = source.splitlines()
    offenders: list[str] = []
    for index, line in enumerate(lines):
        if not any(mutation.search(line) for mutation in _DIRECT_LATEST_MUTATION_PATTERNS):
            continue
        window = "\n".join(lines[max(0, index - 3) : index + 1])
        if "latest" not in window.lower() and "LATEST_MANIFEST_FILE" not in window:
            continue
        # Only the mutating statement itself may claim to be the shared
        # service; a nearby shared-service call must not hide a separate
        # direct write on the next line.
        if any(marker in line for marker in _SHARED_WRITER_MARKERS):
            continue
        offenders.append(f"{relative_path}:{index + 1}: {line.strip()}")
    return offenders


def _collect_direct_latest_write_offenders(repo_root: Path) -> list[str]:
    allowed = {"atomic_io.py"}
    offenders: list[str] = []
    for path in repo_root.rglob("*.py"):
        relative = path.relative_to(repo_root).as_posix()
        if relative.startswith(("tests/", "logs/", "__pycache__/")):
            continue
        if relative in allowed:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        offenders.extend(_direct_latest_write_offenders(relative, source))
    return offenders


class LatestManifestSharedWriterTests(unittest.TestCase):
    def test_physical_writer_uses_latest_lock_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            latest = Path(tmp) / "latest_manifest.txt"
            seen: list[str] = []

            @contextmanager
            def fake_lock(lock_path, **_kwargs):
                seen.append(str(lock_path))
                yield {}

            with mock.patch.object(atomic_io, "_latest_manifest_file_lock", fake_lock):
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
                "with atomic_io._latest_manifest_file_lock(lock, timeout=5.0):\n"
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

    def test_latest_service_uses_manual_recovery_lock_for_both_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            latest = Path(tmp) / "latest_manifest.txt"
            latest.write_text("writer-A", encoding="utf-8")
            with mock.patch.object(
                atomic_io,
                "_latest_manifest_file_lock",
                wraps=atomic_io._latest_manifest_file_lock,
            ) as lock:
                atomic_io.write_latest_manifest_locked(
                    latest,
                    "writer-B",
                    timeout=0.1,
                )
                atomic_io.compare_and_swap_latest_manifest_locked(
                    latest,
                    "writer-B",
                    "writer-C",
                    timeout=0.1,
                )
            self.assertEqual(lock.call_count, 2)
            for call in lock.call_args_list:
                self.assertEqual(
                    call.args[0],
                    atomic_io.latest_manifest_lock_path(latest),
                )
                self.assertEqual(call.kwargs["timeout"], 0.1)

    def test_dead_aged_latest_lock_is_never_preempted(self):
        with tempfile.TemporaryDirectory() as tmp:
            latest = Path(tmp) / "latest_manifest.txt"
            latest.write_text("writer-A", encoding="utf-8")
            lock = Path(atomic_io.latest_manifest_lock_path(latest))
            child = subprocess.Popen([sys.executable, "-c", "pass"])
            child.wait(timeout=5.0)
            lock.write_text(
                json.dumps(
                    {
                        "pid": child.pid,
                        "token": "dead-owner-token",
                        "created_at": time.time() - 30.0,
                    }
                ),
                encoding="utf-8",
            )
            old = time.time() - 30.0
            os.utime(lock, (old, old))
            for label, operation in (
                (
                    "write",
                    lambda: atomic_io.write_latest_manifest_locked(
                        latest,
                        "writer-B",
                        timeout=0.35,
                    ),
                ),
                (
                    "cas",
                    lambda: atomic_io.compare_and_swap_latest_manifest_locked(
                        latest,
                        "writer-A",
                        "writer-B",
                        timeout=0.35,
                    ),
                ),
            ):
                with self.subTest(operation=label):
                    with self.assertRaises(atomic_io.AtomicFileLockTimeoutError):
                        operation()
                    self.assertEqual(
                        latest.read_text(encoding="utf-8"),
                        "writer-A",
                    )
                    self.assertEqual(
                        json.loads(lock.read_text(encoding="utf-8"))["token"],
                        "dead-owner-token",
                    )

    def test_competing_latest_reapers_never_delete_or_double_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = root / "latest_manifest.txt"
            latest.write_text("writer-A", encoding="utf-8")
            lock = Path(atomic_io.latest_manifest_lock_path(latest))
            child = subprocess.Popen([sys.executable, "-c", "pass"])
            child.wait(timeout=5.0)
            lock.write_text(
                json.dumps(
                    {
                        "pid": child.pid,
                        "token": "dead-reaper-victim",
                        "created_at": time.time() - 30.0,
                    }
                ),
                encoding="utf-8",
            )
            old = time.time() - 30.0
            os.utime(lock, (old, old))
            start = root / "start"
            scripts = []
            for index in range(2):
                marker = root / f"marker-{index}"
                script = (
                    "import sys, time\n"
                    "from pathlib import Path\n"
                    "import atomic_io\n"
                    "latest, marker, start = sys.argv[1:4]\n"
                    "while not Path(start).exists():\n"
                    "    time.sleep(0.01)\n"
                    "try:\n"
                    "    atomic_io.write_latest_manifest_locked(\n"
                    "        latest, 'reaper-%s', timeout=0.6)\n"
                    "except atomic_io.AtomicFileLockTimeoutError:\n"
                    "    Path(marker).write_text('timeout')\n"
                    "else:\n"
                    "    Path(marker).write_text('stole')\n"
                ) % index
                scripts.append(
                    subprocess.Popen(
                        [sys.executable, "-c", script, str(latest), str(marker), str(start)],
                        cwd=str(Path(__file__).resolve().parents[1]),
                    )
                )
            start.write_text("go", encoding="utf-8")
            for process in scripts:
                self.assertEqual(process.wait(timeout=10.0), 0)
            self.assertEqual(
                [ (root / f"marker-{i}").read_text(encoding="utf-8") for i in range(2) ],
                ["timeout", "timeout"],
            )
            self.assertEqual(latest.read_text(encoding="utf-8"), "writer-A")
            self.assertEqual(
                json.loads(lock.read_text(encoding="utf-8"))["token"],
                "dead-reaper-victim",
            )

    def test_limited_pattern_scan_catches_representative_bypasses(self):
        bypasses = (
            'open(latest, "w", encoding="utf-8").write("x")',
            "Path(latest).write_text('x', encoding='utf-8')",
            "os.replace(temp_path, latest)",
            "os.rename(temp_path, LATEST_MANIFEST_FILE)",
            "temp_path.replace(latest)",
            "shutil.copyfile(temp_path, latest)",
            "atomic_write_text(latest, 'x')",
        )
        for snippet in bypasses:
            with self.subTest(snippet=snippet):
                self.assertEqual(
                    len(_direct_latest_write_offenders("demo.py", snippet)),
                    1,
                )
        adjacent_bypass = (
            "written = atomic_io.write_latest_manifest_locked(latest, target)\n"
            "open(latest, 'w', encoding='utf-8').write('x')"
        )
        self.assertEqual(
            len(_direct_latest_write_offenders("demo.py", adjacent_bypass)),
            1,
        )
        safe_snippets = (
            "current = latest.read_text(encoding='utf-8')",
            "with open(latest, 'r', encoding='utf-8') as handle:\n    handle.read()",
            "written = atomic_io.write_latest_manifest_locked(latest, target)",
        )
        for snippet in safe_snippets:
            with self.subTest(snippet=snippet):
                self.assertEqual(
                    _direct_latest_write_offenders("demo.py", snippet),
                    [],
                )

    def test_no_direct_latest_manifest_writes_outside_shared_service(self):
        repo_root = Path(__file__).resolve().parents[1]
        self.assertEqual(_collect_direct_latest_write_offenders(repo_root), [])


if __name__ == "__main__":
    unittest.main()
