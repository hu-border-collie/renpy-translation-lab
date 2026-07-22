"""Install a fixed, tool-maintained Ren'Py SDK build from the official host.

Download is always opt-in (CLI flag or GUI confirmation). Arbitrary URLs are not
accepted. Extraction rejects path traversal, absolute paths, and link members.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

from translator_runtime import is_renpy_sdk_dir

# Single recommended stable build maintained by this tool (official renpy.org).
RECOMMENDED_VERSION = "8.5.3"
RECOMMENDED_ARCHIVE_NAME = f"renpy-{RECOMMENDED_VERSION}-sdk.zip"
RECOMMENDED_FOLDER_NAME = f"renpy-{RECOMMENDED_VERSION}-sdk"
RECOMMENDED_URL = (
    f"https://www.renpy.org/dl/{RECOMMENDED_VERSION}/{RECOMMENDED_ARCHIVE_NAME}"
)
# Official checksums.txt for renpy-8.5.3-sdk.zip (sha256 section).
RECOMMENDED_SHA256 = (
    "ff57648f9c04f27e381c48af6d8e3ee3cdec296bed4d3831f47f09b0a71b505e"
)
# Approximate download size for UI (~155 MiB on renpy.org release page).
RECOMMENDED_SIZE_BYTES = 155 * 1024 * 1024
RECOMMENDED_SOURCE_LABEL = "官方 renpy.org"

USER_AGENT = "renpy-translation-lab/sdk-installer"
DEFAULT_TIMEOUT_SEC = 600.0

ProgressCallback = Callable[[str, int, int], None]
CancelCheck = Callable[[], bool]


class SdkInstallError(RuntimeError):
    """Fatal SDK install failure (user-visible message)."""


class SdkCancelled(Exception):
    """Raised when the user cancels download or extract."""


@dataclass(frozen=True)
class RecommendedSdk:
    version: str
    url: str
    sha256: str
    archive_name: str
    folder_name: str
    size_bytes: int
    source_label: str

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SdkInstallResult:
    ok: bool
    message: str
    sdk_dir: Path | None = None
    reused_existing: bool = False
    persisted_config: bool = False
    cancelled: bool = False


def recommended_sdk() -> RecommendedSdk:
    return RecommendedSdk(
        version=RECOMMENDED_VERSION,
        url=RECOMMENDED_URL,
        sha256=RECOMMENDED_SHA256,
        archive_name=RECOMMENDED_ARCHIVE_NAME,
        folder_name=RECOMMENDED_FOLDER_NAME,
        size_bytes=RECOMMENDED_SIZE_BYTES,
        source_label=RECOMMENDED_SOURCE_LABEL,
    )


def format_size_mib(size_bytes: int) -> str:
    mib = size_bytes / (1024 * 1024)
    return f"{mib:.0f} MiB"


def default_sdk_target(workspace: Path | str | None = None) -> Path:
    """Default install directory: ``<workspace>/renpy-<ver>-sdk`` or CWD-based."""
    if workspace is not None and str(workspace).strip():
        base = Path(workspace).expanduser()
        try:
            base = base.resolve(strict=False)
        except (OSError, RuntimeError):
            base = base.absolute()
    else:
        base = Path.cwd()
    return base / RECOMMENDED_FOLDER_NAME


def tool_package_root() -> Path:
    return Path(__file__).resolve().parent


def translator_config_path(tool_root: Path | None = None) -> Path:
    root = tool_root if tool_root is not None else tool_package_root()
    return Path(root) / "translator_config.json"


def save_renpy_sdk_dir(
    sdk_dir: Path | str,
    config_path: Path | None = None,
) -> Path:
    """Write ``prepare.renpy_sdk_dir`` into translator_config.json."""
    path = Path(config_path) if config_path is not None else translator_config_path()
    resolved = Path(sdk_dir).expanduser().resolve()
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8-sig") or "{}")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise SdkInstallError(
                f"无法更新 prepare.renpy_sdk_dir：读取配置失败 {path}: {exc}"
            ) from exc
        if not isinstance(loaded, dict):
            raise SdkInstallError(
                f"无法更新 prepare.renpy_sdk_dir：{path} 根必须是 JSON object。"
            )
        data = loaded
    prepare = data.get("prepare")
    if prepare is None:
        prepare = {}
        data["prepare"] = prepare
    if not isinstance(prepare, dict):
        raise SdkInstallError(
            f"无法更新 prepare.renpy_sdk_dir：prepare 必须是 JSON object（{path}）。"
        )
    prepare["renpy_sdk_dir"] = resolved.as_posix()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str, *, label: str = "SDK 归档") -> None:
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise SdkInstallError(
            f"{label} SHA-256 校验失败：期望 {expected}，实际 {actual}"
        )


def _check_cancel(should_cancel: CancelCheck | None) -> None:
    if should_cancel is not None and should_cancel():
        raise SdkCancelled("SDK 安装已取消。")


def _safe_zip_member_path(member_name: str, staging: Path) -> Path:
    """Return destination path under *staging* or raise on unsafe members."""
    name = member_name.replace("\\", "/")
    if not name or name.endswith("/"):
        # Directory entries handled by mkdir parents.
        rel = name.rstrip("/")
        if not rel:
            raise SdkInstallError("归档包含空路径成员。")
    else:
        rel = name

    if rel.startswith("/") or (len(rel) > 1 and rel[1] == ":"):
        raise SdkInstallError(f"拒绝绝对路径归档成员：{member_name}")
    parts = Path(rel).parts
    if any(part in ("", ".", "..") or part == ".." for part in parts):
        if ".." in parts:
            raise SdkInstallError(f"拒绝路径穿越归档成员：{member_name}")
    # Drop empty / . components only.
    clean_parts = [p for p in parts if p not in ("", ".")]
    if ".." in clean_parts:
        raise SdkInstallError(f"拒绝路径穿越归档成员：{member_name}")
    if not clean_parts:
        raise SdkInstallError(f"拒绝无效归档成员：{member_name}")

    dest = staging.joinpath(*clean_parts)
    try:
        dest_resolved = dest.resolve(strict=False)
        staging_resolved = staging.resolve(strict=False)
        dest_resolved.relative_to(staging_resolved)
    except ValueError as exc:
        raise SdkInstallError(f"拒绝逃逸归档成员：{member_name}") from exc
    except (OSError, RuntimeError) as exc:
        raise SdkInstallError(f"无法解析归档成员路径：{member_name}: {exc}") from exc
    return dest


def extract_sdk_zip(
    archive: Path,
    staging: Path,
    *,
    should_cancel: CancelCheck | None = None,
    progress: ProgressCallback | None = None,
) -> Path:
    """Safely extract *archive* into *staging*; return the SDK root (has renpy.py)."""
    staging.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            infos = zf.infolist()
            total = max(1, len(infos))
            for index, info in enumerate(infos, start=1):
                _check_cancel(should_cancel)
                if progress is not None:
                    progress("extract", index, total)
                # ZipInfo.is_symlink is 3.13+; also reject mode bits for links.
                is_symlink = bool(getattr(info, "is_symlink", lambda: False)())
                if not is_symlink:
                    # Unix symlink: external_attr high bits 0o120000
                    mode = (info.external_attr >> 16) & 0o170000
                    if mode == 0o120000:
                        is_symlink = True
                if is_symlink:
                    raise SdkInstallError(f"拒绝符号链接归档成员：{info.filename}")
                if info.filename.endswith("/") or info.is_dir():
                    dest_dir = _safe_zip_member_path(info.filename, staging)
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    continue
                dest = _safe_zip_member_path(info.filename, staging)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out)
    except zipfile.BadZipFile as exc:
        raise SdkInstallError(f"无效的 SDK zip：{exc}") from exc

    sdk_root = find_sdk_root(staging)
    if sdk_root is None:
        raise SdkInstallError("解压后未找到包含 renpy.py 的 SDK 根目录。")
    return sdk_root


def find_sdk_root(tree: Path) -> Path | None:
    """Locate a directory under *tree* that is a valid Ren'Py SDK root."""
    if is_renpy_sdk_dir(str(tree)):
        return tree
    # Prefer shallow matches (official zip has one top-level folder).
    try:
        children = sorted(tree.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return None
    for child in children:
        if child.is_dir() and is_renpy_sdk_dir(str(child)):
            return child
    for child in children:
        if child.is_dir():
            found = find_sdk_root(child)
            if found is not None:
                return found
    return None


def download_to_file(
    url: str,
    destination: Path,
    *,
    expected_sha256: str,
    timeout: float = DEFAULT_TIMEOUT_SEC,
    should_cancel: CancelCheck | None = None,
    progress: ProgressCallback | None = None,
    opener: Callable[..., Any] | None = None,
) -> None:
    """Download *url* to *destination* and verify SHA-256.

    *opener* is an optional ``urlopen``-compatible callable for tests.
    """
    if url != RECOMMENDED_URL:
        # Hard allow-list: only the fixed official URL.
        raise SdkInstallError(
            f"仅允许下载工具维护的官方推荐 SDK：{RECOMMENDED_URL}（拒绝 {url}）"
        )

    open_fn = opener or urlopen
    request = Request(url, headers={"User-Agent": USER_AGENT})
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    try:
        with open_fn(request, timeout=timeout) as response:
            total = -1
            try:
                length = response.headers.get("Content-Length")
                if length:
                    total = int(length)
            except (TypeError, ValueError, AttributeError):
                total = -1
            written = 0
            with tmp.open("wb") as out:
                while True:
                    _check_cancel(should_cancel)
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    written += len(chunk)
                    if progress is not None:
                        progress("download", written, total if total > 0 else max(written, 1))
        verify_sha256(tmp, expected_sha256, label="下载的 SDK 归档")
        os.replace(tmp, destination)
    except SdkCancelled:
        tmp.unlink(missing_ok=True)
        raise
    except SdkInstallError:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise SdkInstallError(f"下载 SDK 失败：{url}\n{exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)


def _target_conflict_message(target: Path) -> str:
    return (
        f"目标目录已存在且不是有效的 Ren'Py SDK，拒绝覆盖：{target}\n"
        "请换一个空目录，或手动删除冲突目录后再试。"
    )


def install_recommended_sdk(
    target_dir: Path | str,
    *,
    persist_config: bool = True,
    config_path: Path | None = None,
    timeout: float = DEFAULT_TIMEOUT_SEC,
    should_cancel: CancelCheck | None = None,
    progress: ProgressCallback | None = None,
    opener: Callable[..., Any] | None = None,
) -> SdkInstallResult:
    """Download and install the recommended SDK from the official URL only.

    If *target_dir* already is a valid SDK, reuse it (no overwrite). Offline
    tests should use :func:`install_from_archive` with a local zip.
    """
    target = Path(target_dir).expanduser()
    try:
        target = target.resolve(strict=False)
    except (OSError, RuntimeError):
        target = target.absolute()

    if is_renpy_sdk_dir(str(target)):
        persisted = False
        if persist_config:
            try:
                save_renpy_sdk_dir(target, config_path)
                persisted = True
            except SdkInstallError as exc:
                return SdkInstallResult(
                    ok=False,
                    message=f"已找到有效 SDK，但写入配置失败：{exc}",
                    sdk_dir=target,
                    reused_existing=True,
                    persisted_config=False,
                )
        return SdkInstallResult(
            ok=True,
            message=f"复用已有有效 SDK：{target}",
            sdk_dir=target,
            reused_existing=True,
            persisted_config=persisted,
        )

    if target.exists():
        # Non-empty or any existing path that is not a valid SDK → conflict.
        try:
            if target.is_file() or any(target.iterdir()):
                return SdkInstallResult(
                    ok=False,
                    message=_target_conflict_message(target),
                )
        except OSError as exc:
            return SdkInstallResult(
                ok=False,
                message=f"无法检查目标目录：{target}: {exc}",
            )

    spec = recommended_sdk()
    work_root: Path | None = None
    try:
        work_root = Path(tempfile.mkdtemp(prefix="renpy-sdk-install-"))
        archive_path = work_root / spec.archive_name
        staging = work_root / "staging"
        staging.mkdir(parents=True, exist_ok=True)

        download_to_file(
            spec.url,
            archive_path,
            expected_sha256=spec.sha256,
            timeout=timeout,
            should_cancel=should_cancel,
            progress=progress,
            opener=opener,
        )

        sdk_root = extract_sdk_zip(
            archive_path,
            staging,
            should_cancel=should_cancel,
            progress=progress,
        )
        _check_cancel(should_cancel)

        # Place at target: move the sdk_root contents/folder into place.
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.is_dir() and not any(target.iterdir()):
            target.rmdir()
        # Move the discovered SDK root to the final target path.
        if sdk_root.resolve() != target:
            shutil.move(str(sdk_root), str(target))

        if not is_renpy_sdk_dir(str(target)):
            raise SdkInstallError(f"安装后校验失败，未找到 renpy.py：{target}")

        persisted = False
        if persist_config:
            save_renpy_sdk_dir(target, config_path)
            persisted = True

        return SdkInstallResult(
            ok=True,
            message=f"已安装 Ren'Py SDK {spec.version} → {target}",
            sdk_dir=target,
            reused_existing=False,
            persisted_config=persisted,
        )
    except SdkCancelled as exc:
        return SdkInstallResult(ok=False, message=str(exc), cancelled=True)
    except SdkInstallError as exc:
        return SdkInstallResult(ok=False, message=str(exc))
    except (OSError, ValueError) as exc:
        return SdkInstallResult(ok=False, message=f"SDK 安装失败：{exc}")
    finally:
        if work_root is not None:
            shutil.rmtree(work_root, ignore_errors=True)


def install_from_archive(
    archive: Path | str,
    target_dir: Path | str,
    *,
    expected_sha256: str,
    persist_config: bool = False,
    config_path: Path | None = None,
    should_cancel: CancelCheck | None = None,
    progress: ProgressCallback | None = None,
) -> SdkInstallResult:
    """Install from a local archive (tests / offline). Still verifies SHA-256."""
    target = Path(target_dir).expanduser()
    try:
        target = target.resolve(strict=False)
    except (OSError, RuntimeError):
        target = target.absolute()

    if is_renpy_sdk_dir(str(target)):
        return SdkInstallResult(
            ok=True,
            message=f"复用已有有效 SDK：{target}",
            sdk_dir=target,
            reused_existing=True,
        )
    if target.exists():
        try:
            if target.is_file() or any(target.iterdir()):
                return SdkInstallResult(ok=False, message=_target_conflict_message(target))
        except OSError as exc:
            return SdkInstallResult(ok=False, message=str(exc))

    archive_path = Path(archive)
    work_root: Path | None = None
    try:
        verify_sha256(archive_path, expected_sha256, label="SDK 归档")
        work_root = Path(tempfile.mkdtemp(prefix="renpy-sdk-extract-"))
        staging = work_root / "staging"
        sdk_root = extract_sdk_zip(
            archive_path,
            staging,
            should_cancel=should_cancel,
            progress=progress,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.is_dir() and not any(target.iterdir()):
            target.rmdir()
        shutil.move(str(sdk_root), str(target))
        if not is_renpy_sdk_dir(str(target)):
            raise SdkInstallError(f"安装后校验失败：{target}")
        persisted = False
        if persist_config:
            save_renpy_sdk_dir(target, config_path)
            persisted = True
        return SdkInstallResult(
            ok=True,
            message=f"已从本地归档安装 SDK → {target}",
            sdk_dir=target,
            persisted_config=persisted,
        )
    except SdkCancelled as exc:
        return SdkInstallResult(ok=False, message=str(exc), cancelled=True)
    except SdkInstallError as exc:
        return SdkInstallResult(ok=False, message=str(exc))
    except (OSError, ValueError) as exc:
        return SdkInstallResult(ok=False, message=f"SDK 安装失败：{exc}")
    finally:
        if work_root is not None:
            shutil.rmtree(work_root, ignore_errors=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install the tool-maintained recommended Ren'Py SDK (official source only)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show", help="Show the recommended SDK version and URL")
    show.add_argument("--json", action="store_true", dest="as_json")

    install = sub.add_parser(
        "install",
        help="Download and install the recommended SDK (explicit network use)",
    )
    install.add_argument(
        "--target",
        type=Path,
        default=None,
        help=f"Install directory (default: <workspace or cwd>/{RECOMMENDED_FOLDER_NAME})",
    )
    install.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Workspace root used to derive the default target directory",
    )
    install.add_argument(
        "--persist-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write prepare.renpy_sdk_dir into translator_config.json (default: yes)",
    )
    install.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SEC,
        help=f"Download timeout seconds (default {DEFAULT_TIMEOUT_SEC:.0f})",
    )
    install.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    spec = recommended_sdk()

    if args.command == "show":
        if getattr(args, "as_json", False):
            print(json.dumps(spec.public_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"version: {spec.version}")
            print(f"source:  {spec.source_label}")
            print(f"url:     {spec.url}")
            print(f"sha256:  {spec.sha256}")
            print(f"size:    ~{format_size_mib(spec.size_bytes)}")
            print(f"folder:  {spec.folder_name}")
        return 0

    if args.command == "install":
        target = args.target
        if target is None:
            target = default_sdk_target(args.workspace)
        print(
            f"将从 {spec.source_label} 下载 Ren'Py SDK {spec.version}\n"
            f"  URL:    {spec.url}\n"
            f"  大小:   约 {format_size_mib(spec.size_bytes)}\n"
            f"  SHA256: {spec.sha256}\n"
            f"  目标:   {target}",
            flush=True,
        )

        def _progress(phase: str, current: int, total: int) -> None:
            if phase == "download" and total > 0:
                pct = min(100, int(100 * current / total))
                print(f"\r下载中… {pct}% ({current}/{total})", end="", flush=True)
            elif phase == "extract":
                print(f"\r解压中… {current}/{total}", end="", flush=True)

        result = install_recommended_sdk(
            target,
            persist_config=bool(args.persist_config),
            timeout=float(args.timeout),
            progress=_progress,
        )
        print()
        if getattr(args, "as_json", False):
            payload = {
                "ok": result.ok,
                "message": result.message,
                "sdk_dir": str(result.sdk_dir) if result.sdk_dir else None,
                "reused_existing": result.reused_existing,
                "persisted_config": result.persisted_config,
                "cancelled": result.cancelled,
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            stream = sys.stdout if result.ok else sys.stderr
            print(result.message, file=stream)
        return 0 if result.ok else 1

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
