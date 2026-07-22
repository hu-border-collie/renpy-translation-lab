"""Workspace game registry: structured source of truth for GAMES.md.

The registry file (``games_registry.json``) lives at the RenPy workspace root.
``GAMES.md`` is a generated human-readable view — edit the registry, not the table.

Commands::

    python games_registry.py setup --workspace path/to/workspace
    python games_registry.py import-md
    python games_registry.py discover
    python games_registry.py ingest --source path/to/game_or.zip
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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
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

> **请勿手改下方表格。** 真源是同目录的 `games_registry.json`（或 GUI「设置 → 工作区」）。  
> 改完后执行：`python renpy-translation-lab/games_registry.py render-md` 重新生成本文件。  
> 若已误改本表，用 `import-md --merge` 拉回 JSON，再 `render-md`。

本表只记录已经纳入整理结构的项目：

- 顶层 `Game_*` 项目。
- `Game_Adastra_Universe/` 下已经拆分整理的三部作品。

暂不记录顶层压缩包、刚解压但尚未纳入 `Game_*` 结构的目录、工具链、SDK、课程 demo 或日志目录。

## 状态口径

- 游玩状态：`未玩` / `进行中` / `已玩完` / `弃置` / `待确认`
- 翻译状态：`未开始` / `待提取` / `待反编译` / `待翻译` / `翻译中` / `待润色` / `已完成` / `待确认`
- 当前版本：优先取 `build_info.json`（含 Ren'Py 常见的 `game/cache/build_info.json`）或 `options.rpy` 的 `config.version`。`script_version.txt` 是 Ren'Py 引擎版本，不作为游戏版本。
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


def tool_package_root() -> Path:
    """Directory that contains games_registry.py (the tool / repo root)."""
    return Path(__file__).resolve().parent


def default_workspace_root() -> Path | None:
    """Return the configured workspace, or None when unset.

    Workspace is **never** inferred from the tool install path (e.g. parent of
    ``renpy-translation-lab``). Callers must pass ``--workspace``, load
    ``workspace_root`` from ``translator_config.json``, or require an explicit
    GUI selection.
    """
    return load_configured_workspace_root()


def translator_config_path(tool_root: Path | None = None) -> Path:
    root = tool_root if tool_root is not None else tool_package_root()
    return Path(root) / "translator_config.json"


def load_configured_workspace_root(config_path: Path | None = None) -> Path | None:
    """Load absolute workspace path from translator_config.json, if present."""
    path = Path(config_path) if config_path is not None else translator_config_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig") or "{}")
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    raw = data.get("workspace_root")
    if not isinstance(raw, str) or not raw.strip():
        return None
    # Normalize at load so relative values do not depend on process CWD later.
    return Path(raw.strip()).expanduser().resolve()


def save_configured_workspace_root(
    workspace: Path | str,
    config_path: Path | None = None,
) -> Path:
    """Persist workspace_root into translator_config.json; return resolved path.

    Refuses to clobber an unreadable existing config (does not rewrite as a
    one-key object). Callers should treat parse/OS errors as hard failures.
    """
    path = Path(config_path) if config_path is not None else translator_config_path()
    resolved = Path(workspace).expanduser().resolve()
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8-sig") or "{}")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot update workspace_root: failed to read {path}: {exc}"
            ) from exc
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Cannot update workspace_root: {path} root must be a JSON object."
            )
        data = loaded
    data["workspace_root"] = resolved.as_posix()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved


WORKSPACE_REQUIRED_MESSAGE = (
    "工作区未设置。请使用 --workspace 指定目录，"
    "或在 translator_config.json 中设置 workspace_root。"
)


def require_workspace_root(workspace: Path | str | None = None) -> Path:
    """Resolve an explicit or configured workspace, or raise ValueError."""
    if workspace is not None and str(workspace).strip():
        return Path(workspace).expanduser().resolve()
    configured = load_configured_workspace_root()
    if configured is not None:
        return configured.expanduser().resolve()
    raise ValueError(WORKSPACE_REQUIRED_MESSAGE)


def default_registry_path(workspace: Path | None = None) -> Path:
    root = workspace if workspace is not None else default_workspace_root()
    if root is None:
        raise ValueError(WORKSPACE_REQUIRED_MESSAGE)
    return Path(root) / REGISTRY_FILENAME


def default_games_md_path(workspace: Path | None = None) -> Path:
    root = workspace if workspace is not None else default_workspace_root()
    if root is None:
        raise ValueError(WORKSPACE_REQUIRED_MESSAGE)
    return Path(root) / GAMES_MD_FILENAME


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


# Machine codes stay English (doctor / compare); UI and GAMES.md show Chinese.
LAYOUT_STATUS_LABELS: dict[str, str] = {
    "ready": "就绪",
    "attention": "需关注",
    "switch_to_work": "建议使用 work",
    "failed": "不可用",
    "non_renpy": "非 Ren'Py",
}
# Reverse map for Chinese short labels and a few legacy free-form phrases.
_LAYOUT_STATUS_ALIASES: dict[str, str] = {
    **{label: code for code, label in LAYOUT_STATUS_LABELS.items()},
    "待确认": "",
    "已整理": "ready",
    "已建 work": "attention",
    "Unity 包": "non_renpy",
}
DOCTOR_MODE_LABELS: dict[str, str] = {
    "existing_tl_only": "已有 TL 模板",
    "can_generate_template": "可生成模板",
    "blocked_missing_template": "缺少模板且无法生成",
}


def canonicalize_layout_status(raw: str) -> str:
    """Map free-form Chinese notes / labels to a short machine code when possible.

    Returns empty string when unknown / placeholder. Unknown free-form text that
    cannot be classified is returned unchanged so callers can still display it.
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    if text in LAYOUT_STATUS_LABELS:
        return text
    if text in _LAYOUT_STATUS_ALIASES:
        return _LAYOUT_STATUS_ALIASES[text]

    compact = text.replace("`", "").replace(" ", "")
    lower = text.lower()

    if any(
        marker in lower
        for marker in (
            "非 ren",
            "non ren",
            "unity",
            "tyrano",
            "nw.js",
            "nwjs",
            "package.nw",
        )
    ):
        return "non_renpy"
    if "switch_to_work" in lower or "建议使用 work" in text or "请改用 work" in text:
        return "switch_to_work"
    if text == "ready" or text == "就绪" or "检查通过" in text:
        return "ready"
    if text == "failed" or text == "不可用" or "检查失败" in text or "损坏" in text:
        return "failed"
    if text == "attention" or text == "需关注" or "需处理" in text:
        return "attention"
    # Legacy long notes like「已建 original/work/build；…」
    if "已建" in compact or "已整理" in compact or "已建original" in compact.lower():
        if any(marker in lower for marker in ("非 ren", "unity", "tyrano", "nw")):
            return "non_renpy"
        return "attention"
    return text


def resolve_layout_status(project: dict[str, Any]) -> str:
    """Raw layout value for storage / doctor compare (code or legacy free-form)."""
    layout = str(project.get("layout_status") or "").strip()
    if layout:
        return layout
    auto = project.get("auto") if isinstance(project.get("auto"), dict) else {}
    return str(auto.get("doctor_layout") or "").strip()


def format_layout_status_label(raw: str) -> str:
    """Human-readable layout status for tables and GAMES.md (always short Chinese)."""
    text = str(raw or "").strip()
    if not text:
        return "待确认"
    code = canonicalize_layout_status(text)
    if code in LAYOUT_STATUS_LABELS:
        return LAYOUT_STATUS_LABELS[code]
    if not code:
        return "待确认"
    # Unknown free-form: keep short; never show multi-sentence legacy blobs in UI.
    if len(code) > 16 or "；" in code or ";" in code:
        return "需关注"
    return code


def resolve_doctor_mode(project: dict[str, Any]) -> str:
    """Raw doctor mode code (English machine token)."""
    auto = project.get("auto") if isinstance(project.get("auto"), dict) else {}
    return str(auto.get("doctor_mode") or "").strip()


def format_doctor_mode_label(raw: str) -> str:
    """Human-readable doctor / template mode for UI."""
    text = str(raw or "").strip()
    if not text:
        return "—"
    return DOCTOR_MODE_LABELS.get(text, text)


def sync_layout_status_from_auto(project: dict[str, Any]) -> None:
    auto = project.get("auto") if isinstance(project.get("auto"), dict) else {}
    doctor_layout = str(auto.get("doctor_layout") or "").strip()
    if doctor_layout:
        project["layout_status"] = doctor_layout
        return

    # Non-Ren'Py (or doctor skipped): prefer a short machine code over free-form notes.
    in_pipeline = project.get("in_renpy_pipeline")
    if in_pipeline is None:
        in_pipeline = auto.get("in_renpy_pipeline", True)
    engine = str(project.get("engine") or auto.get("engine") or "").lower()
    if not in_pipeline or engine in {"unity", "tyrano", "other"}:
        project["layout_status"] = "non_renpy"
        return

    current = str(project.get("layout_status") or "").strip()
    code = canonicalize_layout_status(current)
    if code in LAYOUT_STATUS_LABELS:
        project["layout_status"] = code


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
    # Ren'Py commonly writes build_info under game/cache/ (not always game/ root).
    candidates = [
        project_root / "original" / "game" / "build_info.json",
        project_root / "original" / "game" / "cache" / "build_info.json",
        project_root / "work" / "game" / "build_info.json",
        project_root / "work" / "game" / "cache" / "build_info.json",
        project_root / "build" / "build_info.json",
        project_root / "build" / "game" / "cache" / "build_info.json",
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
    # Not "manual": an empty detect must remain re-tryable on later refresh.
    # (Older rows used version="待确认" + source="manual", which blocked updates.)
    return "", "pending"


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
    """Point runtime paths at ``work_dir`` for the block, then fully restore.

    Snapshots the full :class:`~translator_runtime.RuntimeConfig` so temporary
    path overrides cannot leak into the caller's process globals
    (issue #216 phase 2).
    """
    import translator_runtime as runtime

    with runtime.locked_runtime_state():
        previous = runtime.snapshot_runtime_config()
        # Derive a job-scoped config with the temporary work root while keeping
        # the caller's language/tl_subdir and other project knobs.
        job = previous.copy()
        normalized = runtime._canonical_abs_path(work_dir)
        job.env_game_root = normalized
        job.base_dir = normalized
        job.tl_dir = runtime._canonical_abs_path(
            os.path.join(normalized, job.tl_subdir or runtime.DEFAULT_TL_SUBDIR)
        )
        job.work_game_dir = runtime._canonical_abs_path(
            os.path.join(normalized, runtime.WORK_GAME_SUBDIR)
        )
        try:
            runtime.apply_runtime_config(job)
            yield
        finally:
            runtime.apply_runtime_config(previous)


def _doctor_layout_snapshot_entry(game_root: str, result_queue) -> None:
    """Top-level spawn entry so deep refresh does not hold the GUI process GIL."""
    try:
        import gemini_translate_batch as batch_mod
    except Exception as exc:
        result_queue.put(("__error__", f"doctor module import failed for {game_root}: {exc}"))
        return
    try:
        with _temporary_game_root(game_root):
            report = batch_mod.collect_doctor_report()
        result_queue.put(
            (
                str(report.get("layout_status") or ""),
                str(report.get("mode") or ""),
            )
        )
    except Exception as exc:
        result_queue.put(
            ("__error__", f"doctor report collection failed for {game_root}: {exc}")
        )


def _doctor_layout_snapshot_inprocess(game_root: str) -> tuple[str, str]:
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


def _doctor_layout_snapshot(game_root: str, has_tl: bool) -> tuple[str, str]:
    """Run doctor layout/mode probe, preferring a child process for GUI hosts.

    Deep registry refresh previously called ``collect_doctor_report`` inside a
    QThread in the GUI process, which starves the event loop via the GIL. Spawn
    isolation keeps Settings browseable during depth refresh.
    """
    del has_tl  # retained for call-site compatibility
    try:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue(1)
        proc = ctx.Process(
            target=_doctor_layout_snapshot_entry,
            args=(game_root, result_queue),
            daemon=True,
        )
        proc.start()
        proc.join(timeout=900)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1)
            warnings.warn(
                f"doctor layout snapshot timed out for {game_root}",
                stacklevel=2,
            )
            return "", ""
        try:
            payload = result_queue.get(timeout=1.0)
        except Exception:
            warnings.warn(
                f"doctor layout snapshot returned no result for {game_root}",
                stacklevel=2,
            )
            return "", ""
        if (
            isinstance(payload, tuple)
            and len(payload) == 2
            and payload[0] == "__error__"
        ):
            warnings.warn(str(payload[1]), stacklevel=2)
            return "", ""
        if isinstance(payload, tuple) and len(payload) == 2:
            return str(payload[0] or ""), str(payload[1] or "")
        return "", ""
    except Exception as exc:
        warnings.warn(
            f"doctor layout snapshot subprocess failed for {game_root}: {exc}; "
            "falling back to in-process doctor",
            stacklevel=2,
        )
        return _doctor_layout_snapshot_inprocess(game_root)


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


# ---------------------------------------------------------------------------
# Workspace create / attach (plan → apply). No Qt. Does not auto-run prepare.
# ---------------------------------------------------------------------------

_WRITE_PROBE_NAME = ".renpy_tl_lab_write_test"


class WorkspaceScene(str, Enum):
    """High-level classification of a candidate workspace path."""

    MISSING_PATH = "missing_path"
    NOT_DIRECTORY = "not_directory"
    NOT_WRITABLE = "not_writable"
    EMPTY = "empty"
    REGISTRY_OK = "registry_ok"
    REGISTRY_CORRUPT = "registry_corrupt"
    GAMES_MD_ONLY = "games_md_only"
    GAME_DIRS_ONLY = "game_dirs_only"
    MIXED = "mixed"


@dataclass(frozen=True)
class WorkspaceSetupPlan:
    """Read-only preview of workspace create / attach actions."""

    workspace: Path
    scene: WorkspaceScene
    ok: bool
    error_message: str = ""
    path_exists: bool = False
    is_dir: bool = False
    is_writable: bool = False
    registry_path: Path = field(default_factory=Path)
    registry_exists: bool = False
    registry_valid: bool = False
    registry_project_count: int = 0
    games_md_path: Path = field(default_factory=Path)
    games_md_exists: bool = False
    games_md_row_count: int = 0
    games_md_parse_ok: bool = True
    game_dir_paths: tuple[str, ...] = ()
    undiscovered_paths: tuple[str, ...] = ()
    will_create_directory: bool = False
    will_create_empty_registry: bool = False
    will_attach_registry: bool = False
    suggest_import_md: bool = False
    suggest_discover: bool = False
    suggest_render_md: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkspaceSetupOptions:
    """User choices for :func:`apply_workspace_setup`."""

    import_md: bool = False
    import_md_merge: bool = True
    discover: bool = False
    refresh_new: bool = True
    refresh_mode: str = REFRESH_MODE_LITE
    render_md: bool = False
    persist_workspace_root: bool = True
    config_path: Path | None = None
    create_directory: bool = False


@dataclass(frozen=True)
class WorkspaceSetupResult:
    """Outcome of applying a workspace setup plan."""

    ok: bool
    message: str
    workspace: Path | None = None
    created_registry: bool = False
    imported_md: bool = False
    discovered_count: int = 0
    rendered_md: bool = False
    persisted_workspace_root: bool = False
    project_count: int = 0
    scene: str = ""


def normalize_workspace_path(path: Path | str) -> Path:
    """Expand user and resolve when possible (works for not-yet-created paths)."""
    candidate = Path(path).expanduser()
    try:
        return candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        return candidate.absolute()


def try_read_registry(path: Path) -> tuple[str, dict[str, Any] | None, str]:
    """Inspect a registry file without synthesizing an empty document.

    Returns ``(status, data, error_message)`` where status is one of
    ``missing`` | ``ok`` | ``corrupt``. Never writes to disk.
    """
    if not path.is_file():
        return "missing", None, ""
    try:
        raw = path.read_text(encoding="utf-8-sig")
        data = json.loads(raw if raw.strip() else "{}")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return "corrupt", None, f"无法解析 {path.name}：{exc}"
    if not isinstance(data, dict):
        return "corrupt", None, f"Registry root must be an object: {path}"
    data.setdefault("schema_version", SCHEMA_VERSION)
    if "projects" not in data:
        data["projects"] = []
    if not isinstance(data.get("projects"), list):
        return "corrupt", None, f"{path.name} 的 projects 必须是数组。"
    return "ok", data, ""


def dir_is_writable(path: Path) -> bool:
    """Return True when ``path`` is a directory that accepts new files."""
    if not path.is_dir():
        return False
    probe = path / _WRITE_PROBE_NAME
    try:
        # Unique suffix reduces collision if two processes probe together.
        probe = path / f"{_WRITE_PROBE_NAME}.{os.getpid()}"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def _project_count(registry: dict[str, Any] | None) -> int:
    if not registry:
        return 0
    projects = registry.get("projects")
    if not isinstance(projects, list):
        return 0
    return len(projects)


def _shell_registry_for_discover(
    workspace: Path,
    registry_data: dict[str, Any] | None,
) -> dict[str, Any]:
    """Minimal in-memory registry for discover path comparison (not written)."""
    if registry_data is not None:
        return registry_data
    return empty_registry(workspace)


def plan_workspace_setup(path: Path | str) -> WorkspaceSetupPlan:
    """Read-only classify a path and suggest create / attach actions.

    Does not write registry, GAMES.md, or ``translator_config.json``.
    A write probe file may be created and deleted to test writability.
    """
    workspace = normalize_workspace_path(path)
    registry_path = workspace / REGISTRY_FILENAME
    md_path = workspace / GAMES_MD_FILENAME
    notes: list[str] = []

    path_exists = workspace.exists()
    is_dir = workspace.is_dir() if path_exists else False
    is_writable = False
    will_create_directory = not path_exists

    if not path_exists:
        notes.append("目录尚不存在；确认后可创建。")
        return WorkspaceSetupPlan(
            workspace=workspace,
            scene=WorkspaceScene.MISSING_PATH,
            ok=True,
            path_exists=False,
            is_dir=False,
            is_writable=False,
            registry_path=registry_path,
            games_md_path=md_path,
            will_create_directory=True,
            will_create_empty_registry=True,
            suggest_import_md=False,
            suggest_discover=False,
            notes=tuple(notes),
        )

    if not is_dir:
        return WorkspaceSetupPlan(
            workspace=workspace,
            scene=WorkspaceScene.NOT_DIRECTORY,
            ok=False,
            error_message=f"目标不是目录：{workspace}",
            path_exists=True,
            is_dir=False,
            registry_path=registry_path,
            games_md_path=md_path,
            notes=("目标路径存在但不是目录。",),
        )

    is_writable = dir_is_writable(workspace)
    if not is_writable:
        return WorkspaceSetupPlan(
            workspace=workspace,
            scene=WorkspaceScene.NOT_WRITABLE,
            ok=False,
            error_message=f"目录不可写：{workspace}",
            path_exists=True,
            is_dir=True,
            is_writable=False,
            registry_path=registry_path,
            games_md_path=md_path,
            notes=("无法在该目录创建文件，未写入 workspace_root。",),
        )

    reg_status, reg_data, reg_error = try_read_registry(registry_path)
    if reg_status == "corrupt":
        return WorkspaceSetupPlan(
            workspace=workspace,
            scene=WorkspaceScene.REGISTRY_CORRUPT,
            ok=False,
            error_message=reg_error or f"{REGISTRY_FILENAME} 损坏，拒绝覆盖。",
            path_exists=True,
            is_dir=True,
            is_writable=True,
            registry_path=registry_path,
            registry_exists=True,
            registry_valid=False,
            games_md_path=md_path,
            games_md_exists=md_path.is_file(),
            notes=(
                f"检测到损坏的 {REGISTRY_FILENAME}，不会重写为空表。",
                "请手动修复或移走该文件后再接入。",
            ),
        )

    registry_exists = reg_status == "ok"
    registry_valid = registry_exists
    registry_project_count = _project_count(reg_data)

    games_md_exists = md_path.is_file()
    games_md_row_count = 0
    games_md_parse_ok = True
    if games_md_exists:
        try:
            content = md_path.read_text(encoding="utf-8-sig")
            games_md_row_count = len(parse_games_md_table(content))
        except (OSError, UnicodeError) as exc:
            games_md_parse_ok = False
            notes.append(f"读取 GAMES.md 失败：{exc}")

    game_dir_paths = tuple(iter_workspace_project_paths(workspace))
    shell = _shell_registry_for_discover(workspace, reg_data)
    undiscovered = tuple(discover_new_project_paths(workspace, shell))

    has_registry = registry_exists
    has_md = games_md_exists
    has_games = bool(game_dir_paths)
    kind_count = sum((has_registry, has_md, has_games))

    if kind_count == 0:
        scene = WorkspaceScene.EMPTY
    elif kind_count >= 2:
        scene = WorkspaceScene.MIXED
    elif has_registry:
        scene = WorkspaceScene.REGISTRY_OK
    elif has_md:
        scene = WorkspaceScene.GAMES_MD_ONLY
    else:
        scene = WorkspaceScene.GAME_DIRS_ONLY

    if scene == WorkspaceScene.GAMES_MD_ONLY and not games_md_parse_ok:
        return WorkspaceSetupPlan(
            workspace=workspace,
            scene=scene,
            ok=False,
            error_message="GAMES.md 无法读取，且没有可用的 games_registry.json。",
            path_exists=True,
            is_dir=True,
            is_writable=True,
            registry_path=registry_path,
            registry_exists=False,
            games_md_path=md_path,
            games_md_exists=True,
            games_md_parse_ok=False,
            game_dir_paths=game_dir_paths,
            notes=tuple(notes) or ("GAMES.md 存在但无法解析。",),
        )

    # Default option suggestions (GUI/CLI may override).
    suggest_import_md = False
    if has_md and games_md_parse_ok:
        if scene == WorkspaceScene.GAMES_MD_ONLY:
            suggest_import_md = True
        elif scene == WorkspaceScene.MIXED and (not has_registry or registry_project_count == 0):
            suggest_import_md = True

    suggest_discover = bool(undiscovered)
    suggest_render_md = False

    will_create_empty_registry = not has_registry and not suggest_import_md
    # If only MD and we suggest import, empty registry is not needed.
    if not has_registry and suggest_import_md:
        will_create_empty_registry = False
    if not has_registry and not has_md:
        will_create_empty_registry = True

    will_attach_registry = has_registry

    if scene == WorkspaceScene.EMPTY:
        notes.append("空目录：将创建最小 games_registry.json。")
    elif scene == WorkspaceScene.REGISTRY_OK:
        notes.append(
            f"已有合法总表（{registry_project_count} 个项目）；接入时不覆盖现有记录。"
        )
    elif scene == WorkspaceScene.GAMES_MD_ONLY:
        notes.append(f"仅有 GAMES.md（约 {games_md_row_count} 行）；建议导入初始化总表。")
    elif scene == WorkspaceScene.GAME_DIRS_ONLY:
        notes.append(
            f"发现 {len(game_dir_paths)} 个 Game_* 目录；建议扫描登记。"
        )
    elif scene == WorkspaceScene.MIXED:
        notes.append("检测到总表 / GAMES.md / Game_* 混合内容；可预览后合并或扫描。")

    if undiscovered:
        notes.append(f"尚未登记的 Game_*：{len(undiscovered)} 个。")
    if has_md and not suggest_import_md and has_registry:
        notes.append("已有总表时导入 GAMES.md 将按路径合并，不会整表替换。")

    return WorkspaceSetupPlan(
        workspace=workspace,
        scene=scene,
        ok=True,
        path_exists=True,
        is_dir=True,
        is_writable=True,
        registry_path=registry_path,
        registry_exists=has_registry,
        registry_valid=registry_valid,
        registry_project_count=registry_project_count,
        games_md_path=md_path,
        games_md_exists=has_md,
        games_md_row_count=games_md_row_count,
        games_md_parse_ok=games_md_parse_ok,
        game_dir_paths=game_dir_paths,
        undiscovered_paths=undiscovered,
        will_create_directory=False,
        will_create_empty_registry=will_create_empty_registry,
        will_attach_registry=will_attach_registry,
        suggest_import_md=suggest_import_md,
        suggest_discover=suggest_discover,
        suggest_render_md=suggest_render_md,
        notes=tuple(notes),
    )


def options_from_plan(
    plan: WorkspaceSetupPlan,
    *,
    import_md: bool | None = None,
    discover: bool | None = None,
    render_md: bool | None = None,
    create_directory: bool = False,
    persist_workspace_root: bool = True,
    config_path: Path | None = None,
    refresh_new: bool = True,
) -> WorkspaceSetupOptions:
    """Build apply options, filling defaults from plan suggestions.

    ``create_directory`` is never implied: callers must opt in explicitly even
    when the plan reports ``will_create_directory``.
    """
    return WorkspaceSetupOptions(
        import_md=plan.suggest_import_md if import_md is None else import_md,
        import_md_merge=True,
        discover=plan.suggest_discover if discover is None else discover,
        refresh_new=refresh_new,
        refresh_mode=REFRESH_MODE_LITE,
        render_md=bool(render_md) if render_md is not None else plan.suggest_render_md,
        persist_workspace_root=persist_workspace_root,
        config_path=config_path,
        create_directory=bool(create_directory),
    )


def apply_workspace_setup(
    plan: WorkspaceSetupPlan,
    options: WorkspaceSetupOptions | None = None,
) -> WorkspaceSetupResult:
    """Apply a confirmed workspace setup plan.

    Never rewrites a corrupt registry. Persists ``workspace_root`` only when
    ``options.persist_workspace_root`` is true and required writes succeed.
    """
    opts = options if options is not None else options_from_plan(plan)

    if not plan.ok:
        return WorkspaceSetupResult(
            ok=False,
            message=plan.error_message or "工作区计划不可用，已取消。",
            scene=plan.scene.value,
        )

    # Re-inspect to avoid acting on a stale plan (TOCTOU).
    fresh = plan_workspace_setup(plan.workspace)
    if fresh.scene == WorkspaceScene.REGISTRY_CORRUPT or (
        not fresh.ok and fresh.scene != WorkspaceScene.MISSING_PATH
    ):
        return WorkspaceSetupResult(
            ok=False,
            message=fresh.error_message or "工作区状态已变化，拒绝写入。",
            workspace=fresh.workspace,
            scene=fresh.scene.value,
        )
    if fresh.scene == WorkspaceScene.MISSING_PATH and not opts.create_directory:
        return WorkspaceSetupResult(
            ok=False,
            message=f"目录不存在，且未指定创建：{fresh.workspace}",
            workspace=fresh.workspace,
            scene=fresh.scene.value,
        )

    workspace = fresh.workspace
    registry_path = fresh.registry_path
    md_path = fresh.games_md_path

    if opts.create_directory and not workspace.exists():
        try:
            workspace.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return WorkspaceSetupResult(
                ok=False,
                message=f"无法创建目录：{exc}",
                workspace=workspace,
                scene=fresh.scene.value,
            )

    if not workspace.is_dir():
        return WorkspaceSetupResult(
            ok=False,
            message=f"目标不是目录：{workspace}",
            workspace=workspace,
            scene=WorkspaceScene.NOT_DIRECTORY.value,
        )

    if not dir_is_writable(workspace):
        return WorkspaceSetupResult(
            ok=False,
            message=f"目录不可写：{workspace}",
            workspace=workspace,
            scene=WorkspaceScene.NOT_WRITABLE.value,
        )

    # Refuse to touch corrupt registry after mkdir/writable checks.
    reg_status, reg_data, reg_error = try_read_registry(registry_path)
    if reg_status == "corrupt":
        return WorkspaceSetupResult(
            ok=False,
            message=reg_error or f"{REGISTRY_FILENAME} 损坏，拒绝写入。",
            workspace=workspace,
            scene=WorkspaceScene.REGISTRY_CORRUPT.value,
        )

    created_registry = False
    imported_md = False
    discovered_count = 0
    rendered_md = False
    data: dict[str, Any]

    try:
        if reg_status == "missing":
            if opts.import_md and md_path.is_file():
                data = import_from_games_md(
                    md_path=md_path,
                    registry_path=registry_path,
                    workspace_root=workspace,
                    merge=False,
                )
                created_registry = True
                imported_md = True
            else:
                data = empty_registry(workspace)
                data["workspace_root"] = workspace.as_posix()
                data["update_summary"] = "工作区初始化：空总表"
                save_registry(registry_path, data)
                created_registry = True
        else:
            assert reg_data is not None
            data = reg_data
            data["workspace_root"] = workspace.as_posix()
            if opts.import_md and md_path.is_file():
                # Existing registry: always merge; never wipe projects.
                data = import_from_games_md(
                    md_path=md_path,
                    registry_path=registry_path,
                    workspace_root=workspace,
                    merge=True,
                )
                imported_md = True
            else:
                # Attach only: refresh workspace_root stamp without touching projects.
                save_registry(registry_path, data)

        if opts.discover:
            status_after, data_after, err_after = try_read_registry(registry_path)
            if status_after != "ok" or data_after is None:
                return WorkspaceSetupResult(
                    ok=False,
                    message=err_after or "扫描前无法读取总表。",
                    workspace=workspace,
                    created_registry=created_registry,
                    imported_md=imported_md,
                    scene=fresh.scene.value,
                )
            data = data_after
            data["workspace_root"] = workspace.as_posix()
            mode = (
                opts.refresh_mode
                if opts.refresh_mode in REFRESH_MODES
                else REFRESH_MODE_LITE
            )
            discovered_count, _paths = merge_discovered_projects(
                data,
                workspace_root=workspace,
                refresh_new=opts.refresh_new,
                mode=mode,
            )
            save_registry(registry_path, data)

        if opts.render_md:
            status_md, data_md, err_md = try_read_registry(registry_path)
            if status_md != "ok" or data_md is None:
                return WorkspaceSetupResult(
                    ok=False,
                    message=err_md or "生成 GAMES.md 前无法读取总表。",
                    workspace=workspace,
                    created_registry=created_registry,
                    imported_md=imported_md,
                    discovered_count=discovered_count,
                    scene=fresh.scene.value,
                )
            data_md["workspace_root"] = workspace.as_posix()
            data_md["update_summary"] = (
                data_md.get("update_summary") or "由工作区接入向导生成"
            )
            write_games_md(data_md, md_path)
            rendered_md = True
            data = data_md

    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return WorkspaceSetupResult(
            ok=False,
            message=f"写入工作区失败：{exc}",
            workspace=workspace,
            created_registry=created_registry,
            imported_md=imported_md,
            discovered_count=discovered_count,
            rendered_md=rendered_md,
            scene=fresh.scene.value,
        )

    final_status, final_data, final_err = try_read_registry(registry_path)
    if final_status != "ok" or final_data is None:
        return WorkspaceSetupResult(
            ok=False,
            message=final_err or "写入后无法验证总表。",
            workspace=workspace,
            created_registry=created_registry,
            imported_md=imported_md,
            discovered_count=discovered_count,
            rendered_md=rendered_md,
            scene=fresh.scene.value,
        )

    project_count = _project_count(final_data)
    persisted = False
    if opts.persist_workspace_root:
        try:
            save_configured_workspace_root(workspace, opts.config_path)
            persisted = True
        except ValueError as exc:
            # Registry already written; report partial success clearly.
            parts = [
                f"工作区总表已就绪（{project_count} 个项目）",
                f"但未能写入 workspace_root：{exc}",
            ]
            return WorkspaceSetupResult(
                ok=False,
                message="；".join(parts),
                workspace=workspace,
                created_registry=created_registry,
                imported_md=imported_md,
                discovered_count=discovered_count,
                rendered_md=rendered_md,
                persisted_workspace_root=False,
                project_count=project_count,
                scene=fresh.scene.value,
            )

    summary_bits = [f"工作区已接入：{workspace}"]
    if created_registry and not imported_md:
        summary_bits.append("已创建空总表")
    if imported_md:
        summary_bits.append("已导入 GAMES.md")
    if discovered_count:
        summary_bits.append(f"新登记 {discovered_count} 个项目")
    if rendered_md:
        summary_bits.append("已生成 GAMES.md")
    summary_bits.append(f"共 {project_count} 个项目")
    if persisted:
        summary_bits.append("已写入 workspace_root")

    return WorkspaceSetupResult(
        ok=True,
        message="；".join(summary_bits) + "。",
        workspace=workspace,
        created_registry=created_registry,
        imported_md=imported_md,
        discovered_count=discovered_count,
        rendered_md=rendered_md,
        persisted_workspace_root=persisted,
        project_count=project_count,
        scene=fresh.scene.value,
    )


def plan_to_public_dict(plan: WorkspaceSetupPlan) -> dict[str, Any]:
    """JSON-serializable view of a setup plan (paths as strings)."""
    payload = asdict(plan)
    payload["workspace"] = str(plan.workspace)
    payload["registry_path"] = str(plan.registry_path)
    payload["games_md_path"] = str(plan.games_md_path)
    payload["scene"] = plan.scene.value
    return payload


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


def get_registry_preferences(registry: dict[str, Any]) -> dict[str, Any]:
    preferences = registry.get("preferences")
    return dict(preferences) if isinstance(preferences, dict) else {}


def set_registry_preference(registry: dict[str, Any], key: str, value: Any) -> None:
    preferences = registry.setdefault("preferences", {})
    if not isinstance(preferences, dict):
        preferences = {}
        registry["preferences"] = preferences
    preferences[key] = value


def remove_project(registry: dict[str, Any], project_id: str) -> dict[str, Any] | None:
    projects = registry.get("projects")
    if not isinstance(projects, list):
        return None
    for index, project in enumerate(projects):
        if isinstance(project, dict) and project.get("id") == project_id:
            return projects.pop(index)
    return None


def update_project_manual_fields(
    registry: dict[str, Any],
    project_id: str,
    *,
    name: str | None = None,
    play_status: str | None = None,
    translation_status: str | None = None,
    notes: str | None = None,
) -> dict[str, Any] | None:
    project = find_project(registry, project_id)
    if project is None:
        return None

    if name is not None:
        text = name.strip()
        if text:
            project["name"] = text
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

    current_version = str(project.get("version") or "").strip()
    version_is_placeholder = current_version in {"", "待确认"}
    # Only skip re-detect when the user explicitly set a real manual version.
    if project.get("version_source") != "manual" or version_is_placeholder:
        detected_version, source = detect_game_version(workspace_root / project["path"])
        if detected_version:
            project["version"] = detected_version
            project["version_source"] = source
        elif version_is_placeholder:
            project["version"] = "待确认"
            if project.get("version_source") != "manual":
                project["version_source"] = source or "pending"

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
        layout = _escape_md_table_cell(
            format_layout_status_label(resolve_layout_status(project))
        )
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
        help=(
            "RenPy workspace root (required unless workspace_root is set in "
            "translator_config.json; never defaults to the tool parent directory)"
        ),
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

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Copy a game directory or .zip into Game_*/original/work/build and register it",
    )
    ingest_parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to an unpacked game directory or a .zip archive",
    )
    ingest_parser.add_argument(
        "--name",
        "--game-name",
        dest="game_name",
        default=None,
        help="Game display name (drives final Game_* folder; default: from source name)",
    )
    ingest_parser.add_argument(
        "--deep",
        action="store_true",
        help="Run a deep doctor refresh after registering (default: lite filesystem scan)",
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

    setup_parser = subparsers.add_parser(
        "setup",
        help=(
            "Create or attach a workspace (preview with --dry-run); "
            "initializes games_registry.json without running project prepare"
        ),
    )
    setup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan only; do not write registry, GAMES.md, or config",
    )
    setup_parser.add_argument(
        "--import-md",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Import GAMES.md into the registry (default: plan suggestion)",
    )
    setup_parser.add_argument(
        "--discover",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Scan and register new Game_* folders (default: plan suggestion)",
    )
    setup_parser.add_argument(
        "--render-md",
        action="store_true",
        help="Regenerate GAMES.md from the registry after setup",
    )
    setup_parser.add_argument(
        "--create-directory",
        action="store_true",
        help="Create the workspace directory when it does not exist",
    )
    setup_parser.add_argument(
        "--persist-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write workspace_root into translator_config.json (default: yes)",
    )
    setup_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print plan or result as JSON",
    )

    return parser


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    try:
        workspace = require_workspace_root(args.workspace)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    registry_path = args.registry or default_registry_path(workspace)
    md_path = default_games_md_path(workspace)
    return workspace, registry_path, md_path


def _print_setup_plan(plan: WorkspaceSetupPlan, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(plan_to_public_dict(plan), ensure_ascii=False, indent=2))
        return
    print(f"workspace: {plan.workspace}")
    print(f"scene: {plan.scene.value}")
    print(f"ok: {plan.ok}")
    if plan.error_message:
        print(f"error: {plan.error_message}")
    print(f"registry: exists={plan.registry_exists} valid={plan.registry_valid} "
          f"projects={plan.registry_project_count}")
    print(f"GAMES.md: exists={plan.games_md_exists} rows={plan.games_md_row_count}")
    print(f"Game_*: {len(plan.game_dir_paths)} "
          f"(undiscovered={len(plan.undiscovered_paths)})")
    if plan.undiscovered_paths:
        for rel in plan.undiscovered_paths[:20]:
            print(f"  + {rel}")
        if len(plan.undiscovered_paths) > 20:
            print(f"  … 共 {len(plan.undiscovered_paths)} 个")
    print(
        "suggest: "
        f"import_md={plan.suggest_import_md} "
        f"discover={plan.suggest_discover} "
        f"render_md={plan.suggest_render_md} "
        f"create_dir={plan.will_create_directory} "
        f"create_empty_registry={plan.will_create_empty_registry}"
    )
    for note in plan.notes:
        print(f"- {note}")


def cmd_setup(args: argparse.Namespace) -> int:
    """Create or attach a workspace; require explicit --workspace."""
    if args.workspace is None or not str(args.workspace).strip():
        print(
            "setup 需要显式 --workspace <path>（不会使用隐式推断）。",
            file=sys.stderr,
        )
        return 2

    plan = plan_workspace_setup(args.workspace)
    if getattr(args, "dry_run", False):
        _print_setup_plan(plan, as_json=bool(getattr(args, "as_json", False)))
        return 0 if plan.ok else 1

    if not plan.ok:
        _print_setup_plan(plan, as_json=bool(getattr(args, "as_json", False)))
        return 1

    options = options_from_plan(
        plan,
        import_md=getattr(args, "import_md", None),
        discover=getattr(args, "discover", None),
        render_md=True if getattr(args, "render_md", False) else None,
        create_directory=bool(getattr(args, "create_directory", False)),
        persist_workspace_root=bool(getattr(args, "persist_config", True)),
    )

    result = apply_workspace_setup(plan, options)
    if getattr(args, "as_json", False):
        payload = {
            "ok": result.ok,
            "message": result.message,
            "workspace": str(result.workspace) if result.workspace else None,
            "created_registry": result.created_registry,
            "imported_md": result.imported_md,
            "discovered_count": result.discovered_count,
            "rendered_md": result.rendered_md,
            "persisted_workspace_root": result.persisted_workspace_root,
            "project_count": result.project_count,
            "scene": result.scene,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        stream = sys.stdout if result.ok else sys.stderr
        print(result.message, file=stream)
    return 0 if result.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "setup":
        return cmd_setup(args)

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

    if args.command == "ingest":
        import game_ingest

        mode = REFRESH_MODE_DEEP if getattr(args, "deep", False) else REFRESH_MODE_LITE

        def _ingest_progress(current: int, total: int, name: str) -> None:
            print(f"[{current}/{total}] {name}", flush=True)

        result = game_ingest.ingest_game(
            source=args.source,
            workspace_root=workspace,
            registry_path=registry_path,
            game_name=getattr(args, "game_name", None),
            refresh=True,
            mode=mode,
            on_progress=_ingest_progress,
        )
        if not result.ok:
            print(result.message, file=sys.stderr)
            return 1
        print(result.message)
        if result.folder_name:
            print(f"  folder: {result.folder_name}")
        if result.project_id:
            print(f"  id: {result.project_id}")
        if result.game_name:
            print(f"  name: {result.game_name}")
        print(f"  registry: {registry_path}")
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