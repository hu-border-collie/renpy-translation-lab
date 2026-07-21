"""Ingest a single game directory or zip into workspace Game_* layout.

Creates::

    Game_<Name>/
    ├─ original/   # full install root or game tree
    ├─ work/       # empty skeleton
    └─ build/      # empty skeleton

Then registers the project in games_registry.json. Does not move the source,
does not bootstrap work/game, and does not generate TL templates.
"""
from __future__ import annotations

import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import games_registry as registry

ProgressCallback = Callable[[int, int, str], None]
CancelCheck = Callable[[], bool]

# Soft cap against zip bombs (bytes of uncompressed members).
DEFAULT_MAX_ZIP_UNCOMPRESSED_BYTES = 50 * 1024 * 1024 * 1024  # 50 GiB
MAX_RENPY_DETECT_DEPTH = 4
ILLEGAL_FOLDER_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
# Collapse non-alphanumeric (keep CJK and other letters via \w with UNICODE).
SLUG_NON_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)
STAGING_DIR_NAME = ".ingest_staging"


@dataclass
class IngestResult:
    ok: bool
    message: str
    project_path: str = ""
    project_id: str = ""
    project_root: Path | None = None
    game_name: str = ""
    folder_name: str = ""
    files_copied: int = 0
    cancelled: bool = False


def suggest_game_name(source: Path) -> str:
    """Human-readable game title prefilled from a directory or zip path."""
    stem = source.stem if source.is_file() else source.name
    text = stem.strip()
    if text.lower().endswith(".zip"):
        text = text[:-4]
    # Light cleanup: strip trailing version-ish segments when separated by - or _
    # Keep it conservative — user can edit.
    text = re.sub(r"[\s._-]+v?\d+(\.\d+){1,3}([._-]\w+)?$", "", text, flags=re.IGNORECASE)
    text = text.strip(" ._-\t")
    return text or stem.strip() or "Game"


def game_name_to_folder(game_name: str) -> str:
    """Map a user-facing game name to the final Game_* folder name.

    Returns empty string when the name cannot produce a valid folder.
    """
    raw = (game_name or "").strip()
    if not raw:
        return ""
    # If user typed Game_Foo, treat Foo as the slug base.
    if raw.startswith("Game_") and len(raw) > 5:
        raw = raw[5:]
    # Replace illegal path characters with space before slugifying.
    cleaned = ILLEGAL_FOLDER_CHARS_RE.sub(" ", raw)
    cleaned = cleaned.strip(" ._-\t")
    if not cleaned:
        return ""
    # Prefer CamelCase join for space-separated Latin words; otherwise underscore.
    parts = [p for p in re.split(r"[\s._-]+", cleaned) if p]
    if not parts:
        return ""
    if all(re.fullmatch(r"[A-Za-z0-9]+", p) for p in parts):
        slug = "".join(p[:1].upper() + p[1:] for p in parts)
    else:
        slug = "_".join(parts)
        slug = SLUG_NON_WORD_RE.sub("_", slug).strip("_")
    if not slug or slug in {".", ".."}:
        return ""
    if slug in registry.WORKSPACE_SKIP_DIR_NAMES:
        return ""
    folder = f"Game_{slug}"
    if folder == registry.ADASTRA_UNIVERSE_DIR:
        return ""
    return folder


def validate_game_name(game_name: str) -> str:
    """Return an error message if game_name is unusable, else empty string."""
    if not (game_name or "").strip():
        return "请输入游戏名称。"
    folder = game_name_to_folder(game_name)
    if not folder:
        return "游戏名称无法生成有效的 Game_* 目录名（请去掉非法字符或换一个名字）。"
    return ""


def folder_conflict_message(workspace_root: Path, folder_name: str) -> str:
    """Return error text if folder or registry path already exists."""
    if not folder_name:
        return "最终目录名为空。"
    target = workspace_root / folder_name
    if target.exists():
        return f"目标目录已存在：{folder_name}。请换一个游戏名称。"
    registry_path = registry.default_registry_path(workspace_root)
    if registry_path.is_file():
        data = registry.load_registry(registry_path)
        existing = {
            str(p.get("path") or "").replace("\\", "/").strip("/")
            for p in data.get("projects", [])
            if isinstance(p, dict)
        }
        if folder_name.replace("\\", "/").strip("/") in existing:
            return f"总表中已有路径 {folder_name}。请换一个游戏名称。"
    return ""


def _is_cancelled(should_cancel: CancelCheck | None) -> bool:
    return bool(should_cancel and should_cancel())


def _emit(
    on_progress: ProgressCallback | None,
    current: int,
    total: int,
    label: str,
) -> None:
    if on_progress is not None:
        on_progress(current, total, label)


def _looks_like_renpy_game_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "options.rpy").is_file():
        return True
    try:
        return any(path.glob("*.rpa"))
    except OSError:
        return False


def _looks_like_renpy_install_root(path: Path) -> bool:
    """True if path has a game/ child that looks like Ren'Py content."""
    game_dir = path / "game"
    return _looks_like_renpy_game_dir(game_dir)


def detect_renpy_install_root(tree_root: Path, *, max_depth: int = MAX_RENPY_DETECT_DEPTH) -> Path | None:
    """Find a Ren'Py install root under tree_root (directory that contains game/)."""
    if not tree_root.is_dir():
        return None
    if _looks_like_renpy_install_root(tree_root):
        return tree_root

    # Single wrapper folder common in zip extracts.
    try:
        children = [c for c in tree_root.iterdir() if c.is_dir() and not c.name.startswith(".")]
    except OSError:
        return None
    if len(children) == 1 and _looks_like_renpy_install_root(children[0]):
        return children[0]

    best: Path | None = None
    best_depth = max_depth + 1

    def walk(current: Path, depth: int) -> None:
        nonlocal best, best_depth
        if depth > max_depth or best is not None and depth >= best_depth:
            return
        if _looks_like_renpy_install_root(current):
            if depth < best_depth:
                best = current
                best_depth = depth
            return
        try:
            for child in current.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    walk(child, depth + 1)
        except OSError:
            return

    walk(tree_root, 0)
    return best


def resolve_copy_mapping(source_root: Path) -> tuple[Path, Path]:
    """Return (source_content_root, relative_dest_under_original).

    relative_dest is Path('.') meaning copy into original/, or Path('game')
    meaning nest under original/game/.
    """
    if _looks_like_renpy_install_root(source_root):
        return source_root, Path(".")
    if _looks_like_renpy_game_dir(source_root):
        return source_root, Path("game")
    install = detect_renpy_install_root(source_root)
    if install is not None:
        return install, Path(".")
    # Fallback: copy whole tree into original/ as-is.
    return source_root, Path(".")


def _safe_zip_relpath(raw_name: str) -> str:
    if isinstance(raw_name, bytes):
        try:
            raw_name = raw_name.decode("utf-8")
        except UnicodeDecodeError:
            raw_name = raw_name.decode("latin-1", errors="replace")
    rel = str(raw_name).replace("\\", "/").strip().lstrip("/")
    parts: list[str] = []
    for part in rel.split("/"):
        if not part or part == ".":
            continue
        if part == "..":
            return ""
        parts.append(part)
    if not parts:
        return ""
    return "/".join(parts)


def _path_is_within(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
        return True
    except (OSError, ValueError):
        return False


def _count_files(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            total += 1
    return total


def _copy_tree(
    src: Path,
    dst: Path,
    *,
    on_progress: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
    progress_label: str = "复制",
) -> int:
    """Copy directory tree file-by-file; return files copied."""
    if not src.is_dir():
        raise FileNotFoundError(f"源目录不存在：{src}")
    total = max(_count_files(src), 1)
    copied = 0
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        if _is_cancelled(should_cancel):
            raise InterruptedError("操作已取消")
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
        if copied == 1 or copied == total or copied % 50 == 0:
            _emit(on_progress, copied, total, f"{progress_label} {copied}/{total}")
    return copied


def _extract_zip_to_staging(
    zip_path: Path,
    staging_root: Path,
    *,
    max_uncompressed_bytes: int = DEFAULT_MAX_ZIP_UNCOMPRESSED_BYTES,
    on_progress: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> int:
    if not zip_path.is_file():
        raise FileNotFoundError(f"找不到 zip：{zip_path}")
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)

    extracted = 0
    total_bytes = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [info for info in zf.infolist() if not info.is_dir()]
        total = max(len(members), 1)
        for info in members:
            if _is_cancelled(should_cancel):
                raise InterruptedError("操作已取消")
            rel = _safe_zip_relpath(info.filename)
            if not rel:
                continue
            out_path = staging_root / Path(rel)
            if not _path_is_within(staging_root, out_path):
                continue
            total_bytes += max(int(info.file_size), 0)
            if total_bytes > max_uncompressed_bytes:
                raise ValueError(
                    f"zip 未压缩体积超过上限（{max_uncompressed_bytes} 字节），已中止。"
                )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
            if extracted == 1 or extracted == total or extracted % 50 == 0:
                _emit(on_progress, extracted, total, f"解压 {extracted}/{total}")
    return extracted


def _create_skeleton(project_root: Path) -> None:
    (project_root / "original").mkdir(parents=True, exist_ok=True)
    (project_root / "work").mkdir(parents=True, exist_ok=True)
    (project_root / "build").mkdir(parents=True, exist_ok=True)


def materialize_from_directory(
    source_dir: Path,
    project_root: Path,
    *,
    on_progress: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> tuple[int, str]:
    """Copy source into project_root/original with correct nesting.

    Returns (files_copied, note).
    """
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"源目录不存在：{source_dir}")
    content_root, dest_rel = resolve_copy_mapping(source_dir)
    original = project_root / "original"
    target = original / dest_rel if dest_rel != Path(".") else original
    note = ""
    if not _looks_like_renpy_install_root(content_root) and not _looks_like_renpy_game_dir(
        content_root
    ):
        note = "未检测到 Ren'Py game/，内容已原样放入 original/，请人工确认。"
    files = _copy_tree(
        content_root,
        target,
        on_progress=on_progress,
        should_cancel=should_cancel,
        progress_label="复制",
    )
    return files, note


def materialize_from_zip(
    zip_path: Path,
    project_root: Path,
    *,
    on_progress: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
    max_uncompressed_bytes: int = DEFAULT_MAX_ZIP_UNCOMPRESSED_BYTES,
) -> tuple[int, str]:
    staging = project_root / STAGING_DIR_NAME
    try:
        _extract_zip_to_staging(
            zip_path,
            staging,
            max_uncompressed_bytes=max_uncompressed_bytes,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
        content_root, dest_rel = resolve_copy_mapping(staging)
        original = project_root / "original"
        # Prefer moving/extract content into original cleanly.
        if dest_rel == Path(".") and content_root == staging:
            # Move staging contents into original
            original.mkdir(parents=True, exist_ok=True)
            files = 0
            for child in list(staging.iterdir()):
                if _is_cancelled(should_cancel):
                    raise InterruptedError("操作已取消")
                dest = original / child.name
                if child.is_dir():
                    files += _copy_tree(
                        child,
                        dest,
                        on_progress=on_progress,
                        should_cancel=should_cancel,
                        progress_label="整理",
                    )
                else:
                    shutil.copy2(child, dest)
                    files += 1
            note = ""
            if not _looks_like_renpy_install_root(original) and not any(
                _looks_like_renpy_game_dir(p) for p in original.iterdir() if p.is_dir()
            ):
                note = "未检测到 Ren'Py game/，内容已原样放入 original/，请人工确认。"
            return files, note

        target = original / dest_rel if dest_rel != Path(".") else original
        files = _copy_tree(
            content_root,
            target,
            on_progress=on_progress,
            should_cancel=should_cancel,
            progress_label="整理",
        )
        note = ""
        if not _looks_like_renpy_install_root(content_root) and not _looks_like_renpy_game_dir(
            content_root
        ):
            note = "未检测到 Ren'Py game/，内容已原样放入 original/，请人工确认。"
        return files, note
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def ingest_game(
    *,
    source: Path | str,
    workspace_root: Path | str,
    registry_path: Path | str | None = None,
    game_name: str | None = None,
    refresh: bool = True,
    mode: str = registry.REFRESH_MODE_LITE,
    on_progress: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
    max_zip_uncompressed_bytes: int = DEFAULT_MAX_ZIP_UNCOMPRESSED_BYTES,
) -> IngestResult:
    """Copy source into Game_*/original/work/build and register the project."""
    workspace = Path(workspace_root).resolve()
    source_path = Path(source).expanduser().resolve()
    reg_path = Path(registry_path) if registry_path else registry.default_registry_path(workspace)

    if not source_path.exists():
        return IngestResult(False, f"找不到源路径：{source_path}")

    is_zip = source_path.is_file() and source_path.suffix.lower() == ".zip"
    if source_path.is_file() and not is_zip:
        return IngestResult(False, "仅支持游戏目录或 .zip 压缩包。")
    if not source_path.is_dir() and not is_zip:
        return IngestResult(False, f"源路径无效：{source_path}")

    resolved_name = (game_name if game_name is not None else suggest_game_name(source_path)).strip()
    name_error = validate_game_name(resolved_name)
    if name_error:
        return IngestResult(False, name_error, game_name=resolved_name)

    folder_name = game_name_to_folder(resolved_name)
    conflict = folder_conflict_message(workspace, folder_name)
    if conflict:
        return IngestResult(
            False,
            conflict,
            game_name=resolved_name,
            folder_name=folder_name,
        )

    project_root = workspace / folder_name
    if project_root.exists():
        return IngestResult(
            False,
            f"目标目录已存在：{folder_name}。请换一个游戏名称。",
            game_name=resolved_name,
            folder_name=folder_name,
        )

    if _is_cancelled(should_cancel):
        return IngestResult(False, "操作已取消。", cancelled=True, game_name=resolved_name)

    project_root.mkdir(parents=True, exist_ok=False)
    try:
        _create_skeleton(project_root)
        _emit(on_progress, 0, 1, "准备目录")

        if is_zip:
            files_copied, note = materialize_from_zip(
                source_path,
                project_root,
                on_progress=on_progress,
                should_cancel=should_cancel,
                max_uncompressed_bytes=max_zip_uncompressed_bytes,
            )
        else:
            files_copied, note = materialize_from_directory(
                source_path,
                project_root,
                on_progress=on_progress,
                should_cancel=should_cancel,
            )

        if _is_cancelled(should_cancel):
            raise InterruptedError("操作已取消")

        _emit(on_progress, 1, 1, "登记项目")
        if reg_path.is_file():
            data = registry.load_registry(reg_path)
        else:
            data = registry.empty_registry(workspace)
        data["workspace_root"] = workspace.as_posix()

        project = registry.make_project_from_discovered_path(workspace, folder_name)
        project["name"] = resolved_name
        project["id"] = registry.ensure_unique_project_id(data, str(project.get("id") or ""))
        data.setdefault("projects", []).append(project)

        if refresh:
            registry.refresh_project(
                data,
                str(project["id"]),
                workspace_root=workspace,
                mode=mode if mode in registry.REFRESH_MODES else registry.REFRESH_MODE_LITE,
            )

        data["update_summary"] = f"一键整理新增 {folder_name}"
        registry.save_registry(reg_path, data)

        message = (
            f"已整理并登记：{folder_name}（游戏名称：{resolved_name}，"
            f"复制 {files_copied} 个文件）。work/ 仍为空，需要时请运行 bootstrap-work。"
        )
        if note:
            message = f"{message} {note}"

        return IngestResult(
            True,
            message,
            project_path=folder_name,
            project_id=str(project["id"]),
            project_root=project_root,
            game_name=resolved_name,
            folder_name=folder_name,
            files_copied=files_copied,
        )
    except InterruptedError:
        shutil.rmtree(project_root, ignore_errors=True)
        return IngestResult(
            False,
            "操作已取消。",
            cancelled=True,
            game_name=resolved_name,
            folder_name=folder_name,
        )
    except Exception as exc:
        shutil.rmtree(project_root, ignore_errors=True)
        return IngestResult(
            False,
            f"整理失败：{exc}",
            game_name=resolved_name,
            folder_name=folder_name,
        )
