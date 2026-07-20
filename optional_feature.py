"""Optional feature dependency status and install command helpers.

Status is derived from package metadata and importability checks in the active
environment — never from a persisted "enabled" flag. Probe paths use
``importlib.util.find_spec`` (and package metadata) so long-lived processes do
not load heavy native libraries such as NumPy or SciPy.
"""

from __future__ import annotations

import importlib.util
import shlex
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent


class FeatureInstallState(str, Enum):
    NOT_INSTALLED = "not_installed"
    PARTIALLY_INSTALLED = "partially_installed"
    INSTALLED = "installed"
    UPDATE_AVAILABLE = "update_available"
    INSTALLING = "installing"
    FAILED = "failed"


@dataclass(frozen=True)
class PackageRequirement:
    """A distribution required by an optional feature.

    ``version`` may be empty for presence-only transitive runtime deps (e.g.
    SciPy pulled in by scikit-learn). Those still participate in install
    completeness probes but never produce an "update available" pin mismatch.
    """

    distribution: str
    version: str
    import_names: tuple[str, ...] = ()

    def metadata_present(self) -> bool:
        try:
            metadata.version(self.distribution)
        except metadata.PackageNotFoundError:
            return False
        return True

    def installed_version(self) -> str:
        try:
            return str(metadata.version(self.distribution) or "").strip()
        except metadata.PackageNotFoundError:
            return ""

    def import_names_available(self) -> bool:
        """Return True when every required top-level module can be found.

        Uses ``find_spec`` only — does not import native extensions into the
        current process.
        """
        names = self.import_names or _default_import_names(self.distribution)
        for name in names:
            try:
                if importlib.util.find_spec(name) is None:
                    return False
            except (ImportError, ModuleNotFoundError, ValueError):
                # Broken/partial installs can raise while resolving the spec.
                return False
        return True

    def is_present(self) -> bool:
        """True only when both distribution metadata and import modules exist."""
        return self.metadata_present() and self.import_names_available()


@dataclass(frozen=True)
class OptionalFeatureSpec:
    """Declarative definition of an optional installable capability."""

    feature_id: str
    display_name: str
    packages: tuple[PackageRequirement, ...]
    lock_relative_path: str
    requirements_relative_path: str
    purpose: str = ""
    components: tuple[str, ...] = ()
    docs_relative_path: str = ""

    def lock_path(self, repo_root: Path = REPO_ROOT) -> Path:
        return (Path(repo_root) / self.lock_relative_path).resolve()

    def requirements_path(self, repo_root: Path = REPO_ROOT) -> Path:
        return (Path(repo_root) / self.requirements_relative_path).resolve()


@dataclass(frozen=True)
class FeatureStatus:
    feature_id: str
    state: FeatureInstallState
    installed_versions: dict[str, str]
    missing: tuple[str, ...]
    outdated: tuple[str, ...]
    message: str
    action_label: str


def _parse_pinned_requirements(path: Path) -> tuple[PackageRequirement, ...]:
    if not path.is_file():
        return ()
    packages: list[PackageRequirement] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        if "==" not in line:
            continue
        name, version = line.split("==", 1)
        distribution = name.strip()
        packages.append(
            PackageRequirement(
                distribution=distribution,
                version=version.strip(),
                import_names=_default_import_names(distribution),
            )
        )
    return tuple(packages)


def _default_import_names(distribution: str) -> tuple[str, ...]:
    normalized = distribution.strip().lower().replace("_", "-")
    mapping = {
        "scikit-learn": ("sklearn",),
        "pillow": ("PIL",),
        "google-genai": ("google.genai",),
        "pyside6": ("PySide6",),
        "scipy": ("scipy",),
        "numpy": ("numpy",),
        "matplotlib": ("matplotlib",),
    }
    if normalized in mapping:
        return mapping[normalized]
    return (distribution.replace("-", "_"),)


def relation_analyzer_feature(repo_root: Path = REPO_ROOT) -> OptionalFeatureSpec:
    requirements_path = Path(repo_root) / "requirements-relation-analyzer.txt"
    packages = list(_parse_pinned_requirements(requirements_path))
    # SciPy is required at runtime (scikit-learn / plotting) even though it is
    # only a transitive lock entry. Probe it so a broken/partial env is
    # reported as repairable rather than "已启用".
    if not any(pkg.distribution.lower().replace("_", "-") == "scipy" for pkg in packages):
        packages.append(
            PackageRequirement(
                distribution="scipy",
                version="",
                import_names=("scipy",),
            )
        )
    package_tuple = tuple(packages)
    return OptionalFeatureSpec(
        feature_id="relation_analyzer",
        display_name="关系分析器",
        packages=package_tuple,
        lock_relative_path="requirements-lock/py311-relation-analyzer.txt",
        requirements_relative_path="requirements-relation-analyzer.txt",
        purpose="从 Ren'Py TL 目录提取人物关系与语义相似度图（独立 CLI：extract_relations.py）。",
        components=tuple(pkg.distribution for pkg in package_tuple),
        docs_relative_path="docs/relation_analysis.md",
    )


def litellm_lock_relative_path() -> str:
    """Return the committed platform lock for the current OS, if any."""
    if sys.platform == "win32":
        return "requirements-lock/py311-windows-litellm.txt"
    if sys.platform.startswith("linux"):
        return "requirements-lock/py311-linux-litellm.txt"
    return ""


def litellm_feature(repo_root: Path = REPO_ROOT) -> OptionalFeatureSpec:
    requirements_path = Path(repo_root) / "requirements-litellm.txt"
    packages = _parse_pinned_requirements(requirements_path)
    lock_relative = litellm_lock_relative_path()
    lock_path = Path(repo_root) / lock_relative if lock_relative else None
    if lock_path is not None and not lock_path.is_file():
        lock_relative = ""
    return OptionalFeatureSpec(
        feature_id="litellm",
        display_name="LiteLLM",
        packages=packages,
        lock_relative_path=lock_relative,
        requirements_relative_path="requirements-litellm.txt",
        purpose="可选同步翻译后端（OpenAI 兼容等 provider）。",
        components=tuple(pkg.distribution for pkg in packages),
        docs_relative_path="docs/dependencies.md",
    )


def probe_feature(
    feature: OptionalFeatureSpec,
    *,
    installing: bool = False,
    last_failed: bool = False,
) -> FeatureStatus:
    """Derive install state without importing heavy native modules."""
    if installing:
        return FeatureStatus(
            feature_id=feature.feature_id,
            state=FeatureInstallState.INSTALLING,
            installed_versions=_installed_versions(feature.packages),
            missing=_missing_packages(feature.packages),
            outdated=(),
            message=f"正在安装或更新 {feature.display_name}…",
            action_label="正在安装…",
        )

    installed_versions = _installed_versions(feature.packages)
    missing = _missing_packages(feature.packages)
    outdated = _outdated_packages(feature.packages, installed_versions)

    if not feature.packages:
        state = FeatureInstallState.NOT_INSTALLED
        message = f"{feature.display_name} 依赖定义不可用。"
        action_label = "安装并启用"
    elif len(missing) == len(feature.packages):
        state = FeatureInstallState.NOT_INSTALLED
        message = f"{feature.display_name} 尚未安装。"
        action_label = "安装并启用"
    elif missing:
        state = FeatureInstallState.PARTIALLY_INSTALLED
        message = (
            f"{feature.display_name} 安装不完整，缺少："
            + "、".join(missing)
            + "。"
        )
        action_label = "修复安装"
    elif outdated:
        state = FeatureInstallState.UPDATE_AVAILABLE
        message = (
            f"{feature.display_name} 已安装，但版本与仓库固定要求不一致："
            + "、".join(outdated)
            + "。"
        )
        action_label = "更新"
    else:
        state = FeatureInstallState.INSTALLED
        message = f"{feature.display_name} 已安装并可用。"
        action_label = "已启用"

    if last_failed and state in {
        FeatureInstallState.NOT_INSTALLED,
        FeatureInstallState.PARTIALLY_INSTALLED,
        FeatureInstallState.UPDATE_AVAILABLE,
    }:
        state = FeatureInstallState.FAILED
        message = f"{feature.display_name} 最近一次安装失败。{message}"
        if action_label == "已启用":
            action_label = "修复安装"

    return FeatureStatus(
        feature_id=feature.feature_id,
        state=state,
        installed_versions=installed_versions,
        missing=missing,
        outdated=outdated,
        message=message,
        action_label=action_label,
    )


def _installed_versions(packages: Iterable[PackageRequirement]) -> dict[str, str]:
    result: dict[str, str] = {}
    for package in packages:
        version = package.installed_version()
        if version:
            result[package.distribution] = version
    return result


def _missing_packages(packages: Iterable[PackageRequirement]) -> tuple[str, ...]:
    missing: list[str] = []
    for package in packages:
        if package.is_present():
            continue
        # Prefer a precise label when only the import surface is broken.
        if package.metadata_present() and not package.import_names_available():
            names = package.import_names or _default_import_names(package.distribution)
            missing.append(f"{package.distribution}（模块不可用：{', '.join(names)}）")
        else:
            missing.append(package.distribution)
    return tuple(missing)


def _outdated_packages(
    packages: Iterable[PackageRequirement],
    installed_versions: dict[str, str],
) -> tuple[str, ...]:
    outdated: list[str] = []
    for package in packages:
        if not package.version:
            # Presence-only transitive requirements (e.g. scipy).
            continue
        current = installed_versions.get(package.distribution, "")
        if not current:
            continue
        if _normalize_version(current) != _normalize_version(package.version):
            outdated.append(
                f"{package.distribution} {current}→{package.version}"
            )
    return tuple(outdated)


def _normalize_version(value: str) -> str:
    return str(value or "").strip().split("+", 1)[0]


def format_shell_command(parts: Iterable[str]) -> str:
    """Format an argv list as a pasteable shell command for the current OS."""
    argv = [str(part) for part in parts]
    if sys.platform == "win32":
        return subprocess.list2cmdline(argv)
    return " ".join(shlex.quote(part) for part in argv)


def hash_checked_install_command(
    feature: OptionalFeatureSpec,
    *,
    python_executable: str = "python",
    repo_root: Path = REPO_ROOT,
) -> str:
    """Return the supported reproducible install command for Python 3.11 locks."""
    lock_path = feature.lock_path(repo_root)
    if lock_path.is_file():
        return format_shell_command(
            [
                python_executable,
                "-m",
                "pip",
                "install",
                "--require-hashes",
                "-r",
                str(lock_path),
            ]
        )
    requirements_path = feature.requirements_path(repo_root)
    return format_shell_command(
        [
            python_executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_path),
        ]
    )


def development_install_command(
    feature: OptionalFeatureSpec,
    *,
    python_executable: str = "python",
    repo_root: Path = REPO_ROOT,
) -> str:
    requirements_path = feature.requirements_path(repo_root)
    return format_shell_command(
        [
            python_executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_path),
        ]
    )


def missing_feature_cli_message(
    feature: OptionalFeatureSpec,
    *,
    python_executable: str = "python",
    repo_root: Path = REPO_ROOT,
    detail: str = "",
) -> str:
    """Actionable no-traceback message for missing optional dependencies."""
    command = hash_checked_install_command(
        feature,
        python_executable=python_executable,
        repo_root=repo_root,
    )
    dev_command = development_install_command(
        feature,
        python_executable=python_executable,
        repo_root=repo_root,
    )
    lines = [
        f"❌ 缺少可选功能「{feature.display_name}」的依赖。",
    ]
    if detail:
        lines.append(detail)
    lines.extend(
        [
            "请安装后再运行（推荐 Python 3.11 哈希锁）：",
            f"  {command}",
            "或使用直接依赖入口：",
            f"  {dev_command}",
            "请勿依赖本工具静默安装依赖。",
        ]
    )
    return "\n".join(lines)


def ensure_relation_analyzer_dependencies(
    *,
    python_executable: str = "python",
    repo_root: Path = REPO_ROOT,
) -> None:
    """Exit cleanly when analyzer packages are missing or incomplete."""
    feature = relation_analyzer_feature(repo_root)
    status = probe_feature(feature)
    if status.state in {
        FeatureInstallState.INSTALLED,
        FeatureInstallState.UPDATE_AVAILABLE,
    }:
        return
    detail = ""
    if status.missing:
        detail = "缺少：" + "、".join(status.missing) + "。"
    raise SystemExit(
        missing_feature_cli_message(
            feature,
            python_executable=python_executable,
            repo_root=repo_root,
            detail=detail,
        )
    ) from None
