"""Workspace game registry: structured source of truth for GAMES.md.

The registry file (``games_registry.json``) lives at the RenPy workspace root.
``GAMES.md`` is a generated human-readable view — edit the registry, not the table.

Commands::

    python games_registry.py import-md
    python games_registry.py discover
    python games_registry.py refresh --all
    python games_registry.py refresh --project glory_hounds
    python games_registry.py render-md
    python games_registry.py record-batch --project glory_hounds --manifest path/to/manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

ProgressCallback = Callable[[int, int, str], None]
CancelCheck = Callable[[], bool]

REFRESH_MODE_LITE = "lite"
REFRESH_MODE_DEEP = "deep"
REFRESH_MODES = frozenset({REFRESH_MODE_LITE, REFRESH_MODE_DEEP})

SCHEMA_VERSION = 1
REGISTRY_FILENAME = "games_registry.json"
GAMES_MD_FILENAME = "GAMES.md"

PLAY_STATUSES = frozenset({"未玩", "进行中", "已玩完", "弃置", "待确认"})
TRANSLATION_STATUSES = frozenset(
    {
        "未开始",
        "待提取",
        "待反编译",
        "待翻译",
        "翻译中",
        "待润色",
        "已完成",
        "待确认",
    }
)
ENGINES = frozenset({"renpy", "unity", "tyrano", "other"})
STATUS_SOURCES = frozenset({"manual", "doctor", "batch", "scan"})
WORKSPACE_SKIP_DIR_NAMES = frozenset(
    {
        "renpy-translation-lab",
        ".git",
        ".github",
        ".venv",
        "venv",
        "__pycache__",
    }
)
ADASTRA_UNIVERSE_DIR = "Game_Adastra_Universe"

GAMES_MD_HEADER = """# 游戏状态总表

<!-- generated from games_registry.json; do not edit the table below -->

{updated_line}

本表只记录已经纳入整理结构的项目：

- 顶层 `Game_*` 项目。
- `Game_Adastra_Universe/` 下已经拆分整理的三部作品。

暂不记录顶层压缩包、刚解压但尚未纳入 `Game_*` 结构的目录、工具链、SDK、课程 demo 或日志目录。

## 状态口径

- 游玩状态：`未玩` / `进行中` / `已玩完` / `弃置` / `待确认`
- 翻译状态：`未开始` / `待提取` / `待反编译` / `待翻译` / `翻译中` / `待润色` / `已完成` / `待确认`
- 当前版本：优先取当前基线发行版的 `build_info.json` 或 `options.rpy`；没有明确游戏版本时再参考归档包名、启动器名或构建目录名。`script_version.txt` 是 Ren'Py 引擎版本，不作为游戏版本。
- 术语提取：`glossary.json` / `macro_setting.md` / `extracted_keywords/` / 系列 `shared/` 术语表 / 对话提取批次，均记入备注。

## 项目状态

| 项目 | 路径 | 当前版本 | 目录状态 | 游玩状态 | 翻译状态 | 备注 / 下一步 |
|---|---|---|---|---|---|---|
"""

TABLE_ROW_RE = re.compile(
    r"^\|\s*(?P<name>[^|]+?)\s*\|\s*`(?P<path>[^`]+)`\s*\|\s*(?P<version>[^|]*?)\s*\|\s*(?P<layout>[^|]*?)\s*\|\s*(?P<play>[^|]*?)\s*\|\s*(?P<translation>[^|]*?)\s*\|\s*(?P<notes>[^|]*?)\s*\|$"
)
VERSION_CONFIG_RE = re.compile(
    r"""config\.version\s*=\s*['"]([^'"]+)['"]""",
    re.IGNORECASE,
)
TRANSLATE_BLOCK_RE = re.compile(r"^translate\s+\S+\s+(?!strings\b)(?!python\b)\S+\s*:")
TL_NEW_LINE_RE = re.compile(r'^\s*new\s+"')
TL_OLD_LINE_RE = re.compile(r'^\s*old\s+"')
TL_COMMENT_SOURCE_RE = re.compile(r"^\s*#\s+")


def default_workspace_root() -> Path:
    tool_root = Path(__file__).resolve().parent
    return tool_root.parent


def default_registry_path(workspace: Path | None = None) -> Path:
    root = workspace or default_workspace_root()
    return root / REGISTRY_FILENAME


def default_games_md_path(workspace: Path | None = None) -> Path:
    root = workspace or default_workspace_root()
    return root / GAMES_MD_FILENAME


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def format_updated_line(registry: dict[str, Any]) -> str:
    updated_at = registry.get("updated_at", "")
    summary = registry.get("update_summary", "").strip()
    if updated_at:
        date_part = updated_at[:10]
        if summary:
            return f"更新时间：{date_part}（{summary}）"
        return f"更新时间：{date_part}"
    return "更新时间：（尚未生成）"


def slugify_project_id(path: str, name: str = "") -> str:
    normalized = path.replace("\\", "/").strip("/")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()
    if slug:
        return slug
    fallback = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return fallback or "project"


def normalize_translation_status(raw: str) -> str:
    text = raw.strip()
    if not text:
        return "待确认"
    base = re.sub(r"（[^）]*）", "", text).strip()
    if base in TRANSLATION_STATUSES:
        return text
    return "待确认"


def normalize_play_status(raw: str) -> str:
    text = raw.strip()
    if text in PLAY_STATUSES:
        return text
    return "待确认"


def humanize_project_name(rel_path: str) -> str:
    leaf = Path(rel_path.replace("\\", "/")).name
    if leaf.startswith("Game_"):
        leaf = leaf[5:]
    return leaf.replace("_", " ").strip() or rel_path


def iter_workspace_project_paths(workspace_root: Path) -> list[str]:
    """Return relative paths for workspace folders that look like game projects."""
    paths: list[str] = []
    if not workspace_root.is_dir():
        return paths

    for child in sorted(workspace_root.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir() or child.name in WORKSPACE_SKIP_DIR_NAMES:
            continue
        if child.name.startswith("Game_") and child.name != ADASTRA_UNIVERSE_DIR:
            paths.append(child.name)

    adastra_root = workspace_root / ADASTRA_UNIVERSE_DIR
    if adastra_root.is_dir():
        for child in sorted(adastra_root.iterdir(), key=lambda item: item.name.lower()):
            if child.is_dir() and not child.name.startswith("."):
                paths.append(f"{ADASTRA_UNIVERSE_DIR}/{child.name}")
    return paths


def ensure_unique_project_id(registry: dict[str, Any], base_id: str) -> str:
    existing = {
        str(project.get("id") or "")
        for project in registry.get("projects", [])
        if isinstance(project, dict)
    }
    candidate = base_id or "project"
    if candidate not in existing:
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in existing:
        suffix += 1
    return f"{candidate}_{suffix}"


def make_project_from_discovered_path(workspace_root: Path, rel_path: str) -> dict[str, Any]:
    normalized_path = rel_path.replace("\\", "/").strip("/")
    project_root = workspace_root / Path(normalized_path)
    engine, in_pipeline = infer_engine(project_root)
    version, version_source = detect_game_version(project_root)
    base_id = slugify_project_id(normalized_path)
    return {
        "id": base_id,
        "name": humanize_project_name(normalized_path),
        "path": normalized_path,
        "version": version or "待确认",
        "version_source": version_source,
        "layout_status": "",
        "play_status": "待确认",
        "translation_status": "待确认",
        "translation_status_source": "manual",
        "notes": "",
        "engine": engine,
        "in_renpy_pipeline": in_pipeline,
        "auto": {},
    }


def discover_new_project_paths(workspace_root: Path, registry: dict[str, Any]) -> list[str]:
    existing = {
        str(project.get("path") or "").replace("\\", "/").strip("/")
        for project in registry.get("projects", [])
        if isinstance(project, dict) and project.get("path")
    }
    return [
        rel_path
        for rel_path in iter_workspace_project_paths(workspace_root)
        if rel_path.replace("\\", "/").strip("/") not in existing
    ]


def resolve_layout_status(project: dict[str, Any]) -> str:
    layout = str(project.get("layout_status") or "").strip()
    if layout:
        return layout
    auto = project.get("auto") if isinstance(project.get("auto"), dict) else {}
    return str(auto.get("doctor_layout") or "").strip()


def sync_layout_status_from_auto(project: dict[str, Any]) -> None:
    auto = project.get("auto") if isinstance(project.get("auto"), dict) else {}
    doctor_layout = str(auto.get("doctor_layout") or "").strip()
    if doctor_layout:
        project["layout_status"] = doctor_layout


def infer_engine(project_root: Path) -> tuple[str, bool]:
    """Return (engine, in_renpy_pipeline)."""
    if (project_root / "work" / "tl").is_dir() and not (project_root / "work" / "game").is_dir():
        return "tyrano", False

    game_dirs = [
        project_root / "work" / "game",
        project_root / "original" / "game",
        project_root / "game",
    ]
    for game_dir in game_dirs:
        if not game_dir.is_dir():
            continue
        if (game_dir / "options.rpy").is_file() or any(game_dir.glob("*.rpa")):
            return "renpy", True

    if (project_root / "original").is_dir():
        original_children = list((project_root / "original").iterdir())
        if any(child.name.endswith("_Data") for child in original_children):
            return "unity", False
        if any(child.suffix.lower() == ".exe" for child in original_children):
            unity_markers = list((project_root / "original").rglob("UnityPlayer.dll"))
            if unity_markers:
                return "unity", False

    notes_hints = ("TyranoScript", "NW.js", "package.nw", "非 Ren'Py")
    return "other", False


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except OSError:
        return ""


def detect_game_version(project_root: Path) -> tuple[str, str]:
    candidates = [
        project_root / "original" / "game" / "build_info.json",
        project_root / "work" / "game" / "build_info.json",
        project_root / "build" / "build_info.json",
        project_root / "original" / "game" / "options.rpy",
        project_root / "work" / "game" / "options.rpy",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if candidate.suffix == ".json":
            try:
                data = json.loads(candidate.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                for key in ("version", "game_version", "build_version"):
                    value = data.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip(), "detected"
        else:
            match = VERSION_CONFIG_RE.search(_read_text(candidate))
            if match and match.group(1).strip():
                return match.group(1).strip(), "detected"
    return "", "manual"


def collect_tl_counts(tl_dir: Path) -> dict[str, int]:
    counts = {
        "rpy_files": 0,
        "translate_blocks": 0,
        "old_lines": 0,
        "new_lines": 0,
        "commented_original_lines": 0,
        "translated_new_lines": 0,
    }
    if not tl_dir.is_dir():
        return counts

    for path in tl_dir.rglob("*.rpy"):
        counts["rpy_files"] += 1
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except OSError:
            continue
        for line in lines:
            if TRANSLATE_BLOCK_RE.match(line):
                counts["translate_blocks"] += 1
            if TL_OLD_LINE_RE.match(line):
                counts["old_lines"] += 1
            if TL_NEW_LINE_RE.match(line):
                counts["new_lines"] += 1
                if re.search(r"[\u4e00-\u9fff]", line):
                    counts["translated_new_lines"] += 1
            if TL_COMMENT_SOURCE_RE.match(line):
                counts["commented_original_lines"] += 1
    return counts


def _relative_asset(project_root: Path, asset_path: Path) -> str:
    try:
        return asset_path.relative_to(project_root).as_posix()
    except ValueError:
        return asset_path.as_posix()


def _has_active_batch_job(work_dir: Path) -> bool:
    jobs_dir = work_dir / "logs" / "batch_jobs"
    if not jobs_dir.is_dir():
        return False
    for child in jobs_dir.iterdir():
        if not child.is_dir():
            continue
        manifest = child / "manifest.json"
        if not manifest.is_file():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        status = str(data.get("status") or data.get("job_status") or "").lower()
        if status in {"submitted", "running", "in_progress", "pending", "processing"}:
            return True
    return False


def infer_layout_status_lite(auto: dict[str, Any]) -> tuple[str, str]:
    """Approximate doctor layout/mode from filesystem scan only."""
    if not auto.get("in_renpy_pipeline", True):
        return "", ""

    work_exists = bool(auto.get("work_exists"))
    work_empty = bool(auto.get("work_empty"))
    has_tl = int(auto.get("tl_rpy_files") or 0) > 0
    has_original = bool(auto.get("original_game_exists"))

    if not work_exists:
        if has_original or has_tl:
            return "switch_to_work", "existing_tl_only" if has_tl else "can_generate_template"
        return "failed", "blocked_missing_template"

    if has_tl:
        return "ready", "existing_tl_only"

    if work_empty:
        if has_original or int(auto.get("original_editable_rpy_count") or 0) > 0:
            return "attention", "can_generate_template"
        if int(auto.get("original_rpa_count") or 0) > 0:
            return "attention", "blocked_missing_template"
        return "failed", "blocked_missing_template"

    if has_original:
        return "attention", "can_generate_template"
    return "attention", "blocked_missing_template"


def scan_project_auto(
    workspace_root: Path,
    project: dict[str, Any],
    *,
    deep: bool = False,
) -> dict[str, Any]:
    project_path = project["path"]
    project_root = workspace_root / Path(project_path.replace("\\", "/"))
    work_dir = project_root / "work"
    original_game = project_root / "original" / "game"

    engine, in_renpy_pipeline = infer_engine(project_root)
    if project.get("engine"):
        engine = project["engine"]
    if "in_renpy_pipeline" in project:
        in_renpy_pipeline = bool(project["in_renpy_pipeline"])

    try:
        from translator_runtime import resolve_effective_game_root

        game_root = Path(resolve_effective_game_root(str(project_root)))
    except Exception:
        game_root = work_dir if work_dir.is_dir() else project_root

    tl_candidates = [
        game_root / "game" / "tl" / "schinese",
        game_root / "tl" / "schinese",
        project_root / "work" / "game" / "tl" / "schinese",
        project_root / "work" / "tl" / "schinese",
        project_root / "original" / "game" / "tl" / "schinese",
    ]
    tl_dir = next((path for path in tl_candidates if path.is_dir()), tl_candidates[0])

    counts = collect_tl_counts(tl_dir)
    glossary = game_root / "glossary.json"
    macro_setting = game_root / "macro_setting.md"

    pending_tasks = max(counts["commented_original_lines"] - counts["translated_new_lines"], 0)
    if counts["commented_original_lines"] <= 0 and counts["new_lines"] > 0:
        pending_tasks = max(counts["new_lines"] - counts["translated_new_lines"], 0)

    baseline = counts["commented_original_lines"] or counts["translate_blocks"] or counts["new_lines"]
    translated_pct = None
    if baseline > 0:
        translated = baseline - pending_tasks
        translated_pct = round(max(translated, 0) * 100.0 / baseline, 1)

    editable_py_count = 0
    rpa_count = 0
    if original_game.is_dir():
        editable_py_count = len(list(original_game.rglob("*.rpy")))
        rpa_count = len(list(original_game.glob("*.rpa")))

    auto: dict[str, Any] = {
        "last_refresh_at": utc_now_iso(),
        "engine": engine,
        "in_renpy_pipeline": in_renpy_pipeline,
        "project_root": project_root.as_posix(),
        "game_root": game_root.as_posix(),
        "work_exists": work_dir.is_dir(),
        "work_empty": not work_dir.is_dir() or not any(work_dir.iterdir()),
        "original_game_exists": original_game.is_dir(),
        "original_editable_rpy_count": editable_py_count,
        "original_rpa_count": rpa_count,
        "tl_rpy_files": counts["rpy_files"],
        "translate_blocks": counts["translate_blocks"],
        "pending_tasks": pending_tasks,
        "dialogue_translated_pct": translated_pct,
        "glossary": _relative_asset(project_root, glossary) if glossary.is_file() else "",
        "macro_setting": _relative_asset(project_root, macro_setting) if macro_setting.is_file() else "",
        "has_active_batch": _has_active_batch_job(work_dir if work_dir.is_dir() else game_root),
    }

    refresh_mode = REFRESH_MODE_DEEP if deep else REFRESH_MODE_LITE
    auto["refresh_mode"] = refresh_mode

    doctor_layout = ""
    doctor_mode = ""
    if in_renpy_pipeline and game_root.is_dir():
        if deep:
            doctor_layout, doctor_mode = _doctor_layout_snapshot(str(game_root), counts["rpy_files"] > 0)
        else:
            doctor_layout, doctor_mode = infer_layout_status_lite(auto)
    auto["doctor_layout"] = doctor_layout
    auto["doctor_mode"] = doctor_mode
    return auto


@contextmanager
def _temporary_game_root(work_dir: str) -> Iterator[None]:
    import translator_runtime as runtime

    with runtime.locked_runtime_state():
        previous = runtime.BASE_DIR
        runtime._apply_game_root(work_dir)
        try:
            yield
        finally:
            runtime._apply_game_root(previous)


def _doctor_layout_snapshot(game_root: str, has_tl: bool) -> tuple[str, str]:
    try:
        import gemini_translate_batch as batch_mod
    except Exception as exc:
        warnings.warn(
            f"doctor module import failed for {game_root}: {exc}",
            stacklevel=2,
        )
        return "", ""

    try:
        with _temporary_game_root(game_root):
            report = batch_mod.collect_doctor_report()
    except Exception as exc:
        warnings.warn(
            f"doctor report collection failed for {game_root}: {exc}",
            stacklevel=2,
        )
        return "", ""

    return str(report.get("layout_status") or ""), str(report.get("mode") or "")


def suggest_translation_status(project: dict[str, Any], auto: dict[str, Any]) -> str:
    if not auto.get("in_renpy_pipeline", True):
        return project.get("translation_status") or "待确认"

    manual = project.get("translation_status", "")
    if project.get("translation_status_source") == "manual" and manual:
        return manual

    if auto.get("has_active_batch"):
        return "翻译中"

    if not auto.get("work_exists") or auto.get("work_empty"):
        if auto.get("original_rpa_count") and auto.get("original_editable_rpy_count", 0) == 0:
            return "待反编译"
        return "未开始"

    if auto.get("tl_rpy_files", 0) <= 0:
        if auto.get("original_game_exists"):
            return "待提取"
        return "未开始"

    pending = int(auto.get("pending_tasks") or 0)
    pct = auto.get("dialogue_translated_pct")
    if pending <= 0:
        return "已完成"
    if isinstance(pct, (int, float)):
        if pct >= 95:
            return "待润色"
        if pct <= 1:
            return "待翻译"
    if pending >= 50:
        return "待翻译"
    return "待润色"


def parse_games_md_table(content: str) -> list[dict[str, Any]]:
    projects: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for line in content.splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        match = TABLE_ROW_RE.match(line)
        if not match:
            continue
        path = match.group("path").strip()
        if path in seen_paths:
            continue
        seen_paths.add(path)

        name = match.group("name").strip()
        projects.append(
            {
                "id": slugify_project_id(path, name),
                "name": name,
                "path": path,
                "version": match.group("version").strip(),
                "version_source": "manual",
                "layout_status": match.group("layout").strip(),
                "play_status": match.group("play").strip() or "待确认",
                "translation_status": normalize_translation_status(match.group("translation")),
                "translation_status_source": "manual",
                "notes": match.group("notes").strip(),
                "auto": {},
            }
        )
    return projects


def empty_registry(workspace_root: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "workspace_root": workspace_root.as_posix(),
        "updated_at": utc_now_iso(),
        "update_summary": "",
        "projects": [],
    }


def load_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return empty_registry(path.parent)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Registry root must be an object: {path}")
    data.setdefault("schema_version", SCHEMA_VERSION)
    data.setdefault("projects", [])
    return data


def save_registry(path: Path, registry: dict[str, Any]) -> None:
    registry["schema_version"] = SCHEMA_VERSION
    registry["updated_at"] = utc_now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def find_project(registry: dict[str, Any], project_id: str) -> dict[str, Any] | None:
    for project in registry.get("projects", []):
        if project.get("id") == project_id:
            return project
    return None


def import_from_games_md(
    *,
    md_path: Path,
    registry_path: Path,
    workspace_root: Path | None = None,
    merge: bool = False,
) -> dict[str, Any]:
    workspace = workspace_root or md_path.parent
    content = md_path.read_text(encoding="utf-8-sig")
    projects = parse_games_md_table(content)

    for project in projects:
        project_root = workspace / project["path"]
        engine, in_pipeline = infer_engine(project_root)
        project["engine"] = engine
        project["in_renpy_pipeline"] = in_pipeline

    if merge and registry_path.is_file():
        registry = load_registry(registry_path)
        registry["workspace_root"] = workspace.as_posix()
        added, updated = _merge_projects_by_path(registry, projects)
        registry["update_summary"] = (
            f"从 {md_path.name} 合并导入：新增 {added} 个，更新 {updated} 个"
        )
    else:
        registry = empty_registry(workspace)
        registry["projects"] = projects
        registry["update_summary"] = f"从 {md_path.name} 导入 {len(projects)} 个项目"

    save_registry(registry_path, registry)
    return registry


def _merge_projects_by_path(
    registry: dict[str, Any],
    incoming_projects: list[dict[str, Any]],
) -> tuple[int, int]:
    existing_by_path: dict[str, dict[str, Any]] = {}
    for project in registry.get("projects", []):
        if isinstance(project, dict) and project.get("path"):
            existing_by_path[str(project["path"]).replace("\\", "/").strip("/")] = project

    added = 0
    updated = 0
    for project in incoming_projects:
        path_key = str(project["path"]).replace("\\", "/").strip("/")
        current = existing_by_path.get(path_key)
        if current is None:
            project["id"] = ensure_unique_project_id(registry, str(project.get("id") or ""))
            registry.setdefault("projects", []).append(project)
            existing_by_path[path_key] = project
            added += 1
            continue

        for field in (
            "name",
            "version",
            "version_source",
            "layout_status",
            "play_status",
            "translation_status",
            "translation_status_source",
            "notes",
            "engine",
            "in_renpy_pipeline",
        ):
            value = project.get(field)
            if value not in (None, ""):
                current[field] = value
        updated += 1
    return added, updated


def merge_discovered_projects(
    registry: dict[str, Any],
    *,
    workspace_root: Path,
    refresh_new: bool = True,
    mode: str = REFRESH_MODE_LITE,
) -> tuple[int, list[str]]:
    new_paths = discover_new_project_paths(workspace_root, registry)
    if not new_paths:
        return 0, []

    added_paths: list[str] = []
    for rel_path in new_paths:
        project = make_project_from_discovered_path(workspace_root, rel_path)
        project["id"] = ensure_unique_project_id(registry, str(project.get("id") or ""))
        registry.setdefault("projects", []).append(project)
        added_paths.append(rel_path)
        if refresh_new:
            refresh_project(
                registry,
                str(project["id"]),
                workspace_root=workspace_root,
                mode=mode,
            )

    registry["update_summary"] = f"扫描工作区新增 {len(added_paths)} 个项目"
    return len(added_paths), added_paths


def update_project_manual_fields(
    registry: dict[str, Any],
    project_id: str,
    *,
    play_status: str | None = None,
    translation_status: str | None = None,
    notes: str | None = None,
) -> dict[str, Any] | None:
    project = find_project(registry, project_id)
    if project is None:
        return None

    if play_status is not None:
        project["play_status"] = normalize_play_status(play_status)
    if translation_status is not None:
        project["translation_status"] = normalize_translation_status(translation_status)
        project["translation_status_source"] = "manual"
    if notes is not None:
        project["notes"] = notes.strip()
    return project


def refresh_project(
    registry: dict[str, Any],
    project_id: str,
    *,
    workspace_root: Path,
    mode: str = REFRESH_MODE_LITE,
) -> dict[str, Any] | None:
    project = find_project(registry, project_id)
    if project is None:
        return None

    deep = mode == REFRESH_MODE_DEEP
    auto = scan_project_auto(workspace_root, project, deep=deep)
    project["auto"] = auto

    if project.get("version_source") != "manual" or not project.get("version"):
        detected_version, source = detect_game_version(workspace_root / project["path"])
        if detected_version:
            project["version"] = detected_version
            project["version_source"] = source

    if project.get("translation_status_source") != "manual":
        project["translation_status"] = suggest_translation_status(project, auto)
        project["translation_status_source"] = "doctor" if deep else "scan"

    sync_layout_status_from_auto(project)
    return project


def refresh_all(
    registry: dict[str, Any],
    *,
    workspace_root: Path,
    mode: str = REFRESH_MODE_LITE,
    on_progress: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> tuple[int, bool]:
    projects = [
        project
        for project in registry.get("projects", [])
        if isinstance(project, dict) and project.get("id")
    ]
    total = len(projects)
    count = 0
    mode_label = "深度" if mode == REFRESH_MODE_DEEP else "快速"
    for index, project in enumerate(projects, start=1):
        if should_cancel and should_cancel():
            registry["update_summary"] = f"{mode_label}刷新已停止；已完成 {count}/{total} 个项目"
            return count, True
        project_id = str(project["id"])
        if on_progress is not None:
            on_progress(index, total, str(project.get("name") or project_id))
        refresh_project(registry, project_id, workspace_root=workspace_root, mode=mode)
        count += 1
    registry["update_summary"] = f"已{mode_label}刷新 {count} 个项目自动状态"
    return count, False


def render_translation_status(project: dict[str, Any]) -> str:
    status = project.get("translation_status", "待确认")
    auto = project.get("auto") or {}
    suffix = ""
    if "增量" in status or status.endswith("）"):
        return status
    if auto.get("has_active_batch") and status != "翻译中":
        suffix = "（有活跃批次）"
    return f"{status}{suffix}" if suffix else status


def render_notes(project: dict[str, Any]) -> str:
    notes = (project.get("notes") or "").strip()
    auto = project.get("auto") or {}
    extras: list[str] = []

    batch_summary = auto.get("last_batch_summary")
    if isinstance(batch_summary, str) and batch_summary.strip():
        if batch_summary.strip() not in notes:
            extras.append(batch_summary.strip())

    if auto.get("last_refresh_at") and not notes:
        pct = auto.get("dialogue_translated_pct")
        if isinstance(pct, (int, float)):
            extras.append(f"对话约 {pct}% 已译（自动扫描）。")

    if extras:
        merged = notes
        for extra in extras:
            if extra not in merged:
                merged = f"{merged}\n{extra}".strip() if merged else extra
        return " ".join(part for part in merged.splitlines() if part.strip())
    return " ".join(part for part in notes.splitlines() if part.strip())


def _escape_md_table_cell(text: str) -> str:
    return str(text).replace("|", "\\|")


def render_games_md(registry: dict[str, Any]) -> str:
    updated_line = format_updated_line(registry)
    lines = [GAMES_MD_HEADER.format(updated_line=updated_line).rstrip("\n")]

    projects = sorted(registry.get("projects", []), key=lambda item: item.get("name", "").lower())
    for project in projects:
        name = _escape_md_table_cell(project.get("name", ""))
        path = _escape_md_table_cell(project.get("path", ""))
        version = _escape_md_table_cell(project.get("version") or "待确认")
        layout = _escape_md_table_cell(resolve_layout_status(project))
        play = _escape_md_table_cell(project.get("play_status", "待确认"))
        translation = _escape_md_table_cell(render_translation_status(project))
        notes = _escape_md_table_cell(render_notes(project))
        lines.append(
            f"| {name} | `{path}` | {version} | {layout} | {play} | {translation} | {notes} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_games_md(registry: dict[str, Any], md_path: Path) -> None:
    md_path.write_text(render_games_md(registry), encoding="utf-8")


def record_batch(
    registry: dict[str, Any],
    *,
    project_id: str,
    manifest_path: Path,
) -> dict[str, Any] | None:
    project = find_project(registry, project_id)
    if project is None:
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read manifest: {manifest_path}") from exc

    auto = project.setdefault("auto", {})
    batch_id = manifest_path.parent.name
    applied = manifest.get("apply_summary") or manifest.get("check_summary") or {}

    def _first_present(*values: object) -> object | None:
        for value in values:
            if value is not None:
                return value
        return None

    files = _first_present(
        applied.get("files_changed"),
        applied.get("files"),
        manifest.get("file_count"),
    )
    lines = _first_present(
        applied.get("lines_changed"),
        applied.get("lines"),
        manifest.get("line_count"),
    )

    summary_parts = [f"Batch `{batch_id}`"]
    if files is not None:
        summary_parts.append(f"{files} 文件")
    if lines is not None:
        summary_parts.append(f"{lines} 行")
    status = str(manifest.get("status") or "")
    if status:
        summary_parts.append(status)

    auto["last_batch_id"] = batch_id
    auto["last_batch_manifest"] = manifest_path.as_posix()
    auto["last_batch_at"] = utc_now_iso()
    auto["last_batch_summary"] = "，".join(str(part) for part in summary_parts if part)

    if project.get("translation_status_source") != "manual":
        project["translation_status"] = "翻译中" if status.lower() in {"submitted", "running"} else project.get(
            "translation_status",
            "待润色",
        )
        project["translation_status_source"] = "batch"
    return project


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage workspace game registry and GAMES.md")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="RenPy workspace root (default: parent of renpy-translation-lab)",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help=f"Registry JSON path (default: <workspace>/{REGISTRY_FILENAME})",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import-md", help="Import projects from GAMES.md into the registry")
    import_parser.add_argument("--md", type=Path, default=None, help="Source GAMES.md path")
    import_parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge into an existing registry by project path instead of replacing it",
    )

    discover_parser = subparsers.add_parser(
        "discover",
        help="Scan workspace for Game_* folders and add any missing projects",
    )
    discover_parser.add_argument(
        "--deep",
        action="store_true",
        help="Run a refresh on each newly discovered project",
    )

    refresh_parser = subparsers.add_parser("refresh", help="Refresh auto-detected project fields")
    refresh_group = refresh_parser.add_mutually_exclusive_group(required=True)
    refresh_group.add_argument("--all", action="store_true", help="Refresh every project")
    refresh_group.add_argument("--project", dest="project_id", help="Refresh one project id")
    refresh_parser.add_argument(
        "--deep",
        action="store_true",
        help="Run full doctor scan per project (slow; default is lite filesystem scan)",
    )

    subparsers.add_parser("render-md", help="Render GAMES.md from the registry")

    record_parser = subparsers.add_parser("record-batch", help="Record batch manifest metadata for a project")
    record_parser.add_argument("--project", dest="project_id", required=True)
    record_parser.add_argument("--manifest", type=Path, required=True)

    show_parser = subparsers.add_parser("show", help="Print registry or one project as JSON")
    show_parser.add_argument("--project", dest="project_id", default=None)

    return parser


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    workspace = args.workspace or default_workspace_root()
    registry_path = args.registry or default_registry_path(workspace)
    md_path = default_games_md_path(workspace)
    return workspace, registry_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    workspace, registry_path, md_path = resolve_paths(args)

    if args.command == "import-md":
        source_md = args.md or md_path
        if not source_md.is_file():
            print(f"GAMES.md not found: {source_md}", file=sys.stderr)
            return 1
        registry = import_from_games_md(
            md_path=source_md,
            registry_path=registry_path,
            workspace_root=workspace,
            merge=getattr(args, "merge", False),
        )
        print(f"Imported {len(registry.get('projects', []))} projects -> {registry_path}")
        return 0

    if args.command == "discover":
        registry = load_registry(registry_path) if registry_path.is_file() else empty_registry(workspace)
        registry["workspace_root"] = workspace.as_posix()
        mode = REFRESH_MODE_DEEP if getattr(args, "deep", False) else REFRESH_MODE_LITE
        added_count, added_paths = merge_discovered_projects(
            registry,
            workspace_root=workspace,
            refresh_new=True,
            mode=mode,
        )
        save_registry(registry_path, registry)
        if added_count:
            print(f"Discovered {added_count} project(s) -> {registry_path}")
            for rel_path in added_paths:
                print(f"  + {rel_path}")
        else:
            print("No new workspace projects discovered.")
        return 0

    registry = load_registry(registry_path)
    if not registry.get("workspace_root"):
        registry["workspace_root"] = workspace.as_posix()

    if args.command == "refresh":
        mode = REFRESH_MODE_DEEP if getattr(args, "deep", False) else REFRESH_MODE_LITE

        def _cli_progress(current: int, total: int, name: str) -> None:
            print(f"[{current}/{total}] {name}", flush=True)

        if args.all:
            count, _cancelled = refresh_all(
                registry,
                workspace_root=workspace,
                mode=mode,
                on_progress=_cli_progress,
            )
            save_registry(registry_path, registry)
            print(f"Refreshed {count} projects ({mode}) -> {registry_path}")
        else:
            if mode == REFRESH_MODE_DEEP:
                _cli_progress(1, 1, args.project_id)
            project = refresh_project(
                registry,
                args.project_id,
                workspace_root=workspace,
                mode=mode,
            )
            if project is None:
                print(f"Unknown project id: {args.project_id}", file=sys.stderr)
                return 1
            save_registry(registry_path, registry)
            print(f"Refreshed project {args.project_id} ({mode}) -> {registry_path}")
        return 0

    if args.command == "render-md":
        registry["update_summary"] = registry.get("update_summary") or "由 games_registry 生成"
        save_registry(registry_path, registry)
        write_games_md(registry, md_path)
        print(f"Rendered {md_path}")
        return 0

    if args.command == "record-batch":
        try:
            project = record_batch(
                registry,
                project_id=args.project_id,
                manifest_path=args.manifest,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if project is None:
            print(f"Unknown project id: {args.project_id}", file=sys.stderr)
            return 1
        save_registry(registry_path, registry)
        print(f"Recorded batch for {args.project_id} -> {registry_path}")
        return 0

    if args.command == "show":
        if args.project_id:
            project = find_project(registry, args.project_id)
            if project is None:
                print(f"Unknown project id: {args.project_id}", file=sys.stderr)
                return 1
            payload = project
        else:
            payload = registry
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())