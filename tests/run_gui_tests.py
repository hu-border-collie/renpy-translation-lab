"""Discover ``test_gui_*`` unittest modules for GUI CI."""
from __future__ import annotations

import os
import sys
import unittest

from test_runner_common import ensure_tests_on_path, parse_runner_args, run_discovered_suite


def build_suite() -> unittest.TestSuite:
    _, directory = ensure_tests_on_path()
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for path in sorted(directory.glob("test_gui_*.py")):
        suite.addTests(loader.loadTestsFromName(path.stem))
    return suite


def main(
    argv: list[str] | None = None,
    *,
    shutdown_runtime: bool = True,
) -> int:
    args = parse_runner_args(__doc__ or "", argv)
    ensure_tests_on_path()
    from gui_test_support import (
        guarded_gui_test_environment,
        guarded_test_result_class,
        shutdown_gui_test_runtime,
    )

    guard = None
    with guarded_gui_test_environment() as guard:
        exit_code = run_discovered_suite(
            build_suite(),
            quiet=args.quiet,
            verbose=args.verbose,
            resultclass=guarded_test_result_class(guard),
        )
    unexpected_dialogs = bool(guard and guard.rejected_dialogs)
    runtime_stopped = shutdown_gui_test_runtime() if shutdown_runtime else True
    return 1 if unexpected_dialogs or not runtime_stopped else exit_code


def _terminate_process(exit_code: int) -> None:
    """Exit after flushing output without running unstable Qt finalizers."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(exit_code))


if __name__ == "__main__":
    # The process is about to hard-exit, so calling back into Qt after the test
    # report only adds an offscreen-platform teardown crash window on Linux.
    _terminate_process(main(shutdown_runtime=False))
