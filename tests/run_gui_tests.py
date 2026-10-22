"""Discover GUI and settings-widget unittest modules for guarded GUI CI."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.support.runner_support import (
    ensure_tests_on_path,
    is_gui_test_module,
    iter_test_modules,
    parse_runner_args,
    run_discovered_suite,
    test_module_name,
)


def build_suite() -> unittest.TestSuite:
    _, directory = ensure_tests_on_path()
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for path in iter_test_modules(directory):
        if is_gui_test_module(path):
            suite.addTests(loader.loadTestsFromName(test_module_name(path, directory)))
    return suite


def main(
    argv: list[str] | None = None,
    *,
    shutdown_runtime: bool = True,
) -> int:
    """Run the GUI suite; ``shutdown_runtime=False`` is hard-exit mode.

    In hard-exit mode the caller is expected to call :func:`_terminate_process`
    immediately, so no Qt pool teardown is performed at all.
    """
    args = parse_runner_args(__doc__ or "", argv)
    ensure_tests_on_path()
    from tests.support.gui_test_support import (
        guarded_gui_test_environment,
        guarded_test_result_class,
        shutdown_gui_test_runtime,
    )

    guard = None
    with guarded_gui_test_environment(process_events=shutdown_runtime) as guard:
        exit_code = run_discovered_suite(
            build_suite(),
            quiet=args.quiet,
            verbose=args.verbose,
            resultclass=guarded_test_result_class(guard),
        )
    unexpected_dialogs = bool(guard and guard.rejected_dialogs)
    if shutdown_runtime:
        runtime_stopped = shutdown_gui_test_runtime()
    else:
        # Script entrypoint mode hard-exits immediately after ``main`` returns.
        # Skip even the pool-only Qt teardown here: offscreen Linux runners can
        # segfault after all tests passed while QThreadPool shuts down.
        runtime_stopped = True
    return 1 if unexpected_dialogs or not runtime_stopped else exit_code


def _terminate_process(exit_code: int) -> None:
    """Exit after flushing output without running unstable Qt finalizers."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(exit_code))


if __name__ == "__main__":
    # Verify that the global pool stopped, but skip widget finalization before
    # the hard exit to avoid the offscreen-platform teardown crash on Linux.
    _terminate_process(main(shutdown_runtime=False))
