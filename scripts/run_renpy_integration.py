"""Run the repository's minimal fixture through a real Ren'Py SDK."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "renpy_smoke"
REQUIRED_FIXTURE_FILES = (
    "game/options.rpy",
    "game/screens.rpy",
    "game/script.rpy",
    "game/testcases.rpy",
)


class RenPyIntegrationError(RuntimeError):
    """Raised when the SDK fixture integration cannot complete safely."""


def resolve_launcher(sdk_dir: Path, platform_name: str | None = None) -> list[str]:
    sdk_dir = sdk_dir.resolve()
    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        python_exe = sdk_dir / "lib" / "py3-windows-x86_64" / "python.exe"
        renpy_py = sdk_dir / "renpy.py"
        missing = [path for path in (python_exe, renpy_py) if not path.is_file()]
        if missing:
            raise RenPyIntegrationError(
                "Ren'Py SDK is missing the Windows launcher files: "
                + ", ".join(str(path) for path in missing)
            )
        return [str(python_exe), str(renpy_py)]

    renpy_sh = sdk_dir / "renpy.sh"
    if not renpy_sh.is_file():
        raise RenPyIntegrationError(
            f"Ren'Py SDK is missing the POSIX launcher: {renpy_sh}"
        )
    return [str(renpy_sh)]


def validate_fixture(fixture_dir: Path) -> None:
    missing = [
        relative
        for relative in REQUIRED_FIXTURE_FILES
        if not (fixture_dir / relative).is_file()
    ]
    if missing:
        raise RenPyIntegrationError(
            "Ren'Py integration fixture is incomplete; missing: " + ", ".join(missing)
        )


def validate_generated_template(project_dir: Path, language: str) -> Path:
    language_dir = project_dir / "game" / "tl" / language
    candidates = sorted(language_dir.rglob("*.rpy")) if language_dir.is_dir() else []
    if not candidates:
        raise RenPyIntegrationError(
            f"Ren'Py did not generate any translation scripts under {language_dir}."
        )

    combined = "\n".join(path.read_text(encoding="utf-8-sig") for path in candidates)
    if f"translate {language}" not in combined:
        raise RenPyIntegrationError(
            f"Generated scripts do not contain a translate {language} block."
        )
    if "Hello from the integration fixture." not in combined:
        raise RenPyIntegrationError(
            "Generated scripts do not contain the fixture dialogue source text."
        )
    return language_dir


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    print("Running:", subprocess.list2cmdline(command))
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.returncode:
        raise RenPyIntegrationError(
            f"Ren'Py command failed with exit code {completed.returncode}: "
            + subprocess.list2cmdline(command)
        )
    return completed


def run_integration(
    sdk_dir: Path,
    fixture_dir: Path = DEFAULT_FIXTURE,
    *,
    language: str = "schinese",
    testcase: str = "smoke",
    command_timeout: int = 180,
    temp_parent: Path | None = None,
    platform_name: str | None = None,
    command_runner=run_command,
) -> None:
    sdk_dir = sdk_dir.resolve()
    fixture_dir = fixture_dir.resolve()
    launcher = resolve_launcher(sdk_dir, platform_name)
    validate_fixture(fixture_dir)

    if temp_parent is not None:
        temp_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="rtl-renpy-integration-",
        dir=temp_parent,
    ) as temporary:
        temporary_dir = Path(temporary)
        project_dir = temporary_dir / "project"
        savedir = temporary_dir / "saves"
        shutil.copytree(fixture_dir, project_dir)
        savedir.mkdir()

        env = os.environ.copy()
        env.setdefault("SDL_AUDIODRIVER", "dummy")

        def invoke(command: str, *arguments: str) -> None:
            command_runner(
                launcher
                + [
                    "--savedir",
                    str(savedir),
                    str(project_dir),
                    command,
                    *arguments,
                ],
                cwd=sdk_dir,
                env=env,
                timeout=command_timeout,
            )

        invoke("translate", language)
        language_dir = validate_generated_template(project_dir, language)
        print(f"Generated translation template: {language_dir}")

        invoke("lint", "--error-code")
        invoke("test", testcase, "--report-detailed")

    print("Ren'Py fixture integration completed successfully.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", required=True, type=Path, help="Ren'Py SDK directory")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE,
        help="fixture project directory",
    )
    parser.add_argument("--language", default="schinese")
    parser.add_argument("--testcase", default="smoke")
    parser.add_argument("--command-timeout", type=int, default=180)
    parser.add_argument("--temp-parent", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_integration(
            args.sdk,
            args.fixture,
            language=args.language,
            testcase=args.testcase,
            command_timeout=args.command_timeout,
            temp_parent=args.temp_parent,
        )
    except (OSError, subprocess.TimeoutExpired, RenPyIntegrationError) as exc:
        print(f"Ren'Py integration failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
