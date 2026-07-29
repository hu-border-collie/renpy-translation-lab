import unittest

from PySide6.QtCore import QProcess

from gui_qt.cli_runner import CliRunner


class _FakeProcess:
    def __init__(self, *, stdout=(), stderr=()):
        self.stdout_chunks = list(stdout)
        self.stderr_chunks = list(stderr)

    def readAllStandardOutput(self):
        return self.stdout_chunks.pop(0) if self.stdout_chunks else b""

    def readAllStandardError(self):
        return self.stderr_chunks.pop(0) if self.stderr_chunks else b""


class CliRunnerChannelTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.stdout_lines = []
        self.stderr_lines = []
        self.all_lines = []
        self.finished_codes = []
        self.runner.stdout_line_ready.connect(self.stdout_lines.append)
        self.runner.stderr_line_ready.connect(self.stderr_lines.append)
        self.runner.line_ready.connect(self.all_lines.append)
        self.runner.finished.connect(self.finished_codes.append)

    def test_channels_stay_separate_across_chunks_and_final_tail(self):
        self.runner._proc = _FakeProcess(
            stdout=(b'{\r', b'\n  "ok": true\r\n', b'}'),
            stderr=(b'progress 1\r\npartial diagnostic',),
        )

        self.runner._on_stdout_ready()
        self.runner._on_stdout_ready()
        self.runner._on_stdout_ready()
        self.runner._on_stderr_ready()
        self.runner._on_finished(0, QProcess.ExitStatus.NormalExit)

        self.assertEqual(self.stdout_lines, ['{', '  "ok": true', '}'])
        self.assertEqual(self.stderr_lines, ['progress 1', 'partial diagnostic'])
        self.assertEqual(self.all_lines, self.stdout_lines[:-1] + ['progress 1', '}','partial diagnostic'])
        self.assertEqual(self.finished_codes, [0])
        self.assertIsNone(self.runner._proc)

    def test_finished_drains_unread_process_bytes_before_flushing(self):
        self.runner._proc = _FakeProcess(
            stdout=(b'{"status":"completed"}',),
            stderr=(b'final diagnostic',),
        )

        self.runner._on_finished(7, QProcess.ExitStatus.NormalExit)

        self.assertEqual(self.stdout_lines, ['{"status":"completed"}'])
        self.assertEqual(self.stderr_lines, ['final diagnostic'])
        self.assertEqual(self.finished_codes, [7])


if __name__ == "__main__":
    unittest.main()
