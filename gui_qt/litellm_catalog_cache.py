"""User-level cache for LiteLLM provider/model catalogs and UI selections.

This file intentionally stores no credentials.  Runtime configuration remains
in ``translator_config.json`` and provider secrets remain in the operating
system credential store.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Callable, Mapping

from atomic_io import atomic_write_json


CACHE_SCHEMA_VERSION = 1
CATALOG_CACHE_MAX_AGE_DAYS = 30
_MAX_CATALOG_ITEMS = 10_000
_MAX_PROVIDER_LENGTH = 200
_MAX_MODEL_LENGTH = 1_000


def default_litellm_catalog_cache_path() -> Path:
    """Return a per-user, non-project path for LiteLLM GUI state."""
    if sys.platform == "win32":
        root = Path(
            os.environ.get("LOCALAPPDATA")
            or (Path.home() / "AppData" / "Local")
        )
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(
            os.environ.get("XDG_STATE_HOME")
            or (Path.home() / ".local" / "state")
        )
    return root / "renpy-translation-lab" / "litellm_catalog_cache.json"


def _can_create_under(directory: Path) -> bool:
    """Return whether *directory* can actually be written right now.

    Walks up to the nearest existing ancestor and attempts a tiny real write
    probe.  ``os.access`` alone is not reliable under sandbox filter
    drivers: it only simulates the ACL (which can still grant the current
    user write rights) and may report writable even though an actual write
    is blocked, leaving later atomic writes hanging.  A real ``os.open``
    fails fast on read-only filesystems and sandbox-blocked directories
    alike, and the probe file is removed immediately.
    """
    current = directory
    while not current.exists():
        parent = current.parent
        if parent == current:
            return False
        current = parent
    if not current.is_dir():
        return False
    probe = current / f".renpy-cache-probe-{os.getpid()}"
    try:
        fd = os.open(probe, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600)
        os.close(fd)
    except OSError:
        return False
    try:
        os.remove(probe)
    except OSError:
        pass
    return True


def _fallback_litellm_cache_path() -> Path | None:
    """Return a writable fallback for the LiteLLM GUI cache, if available."""
    candidates: list[Path] = []
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        candidates.append(
            Path(runtime_dir) / "renpy-translation-lab" / "litellm_catalog_cache.json"
        )
    candidates.append(
        Path(tempfile.gettempdir())
        / "renpy-translation-lab"
        / "litellm_catalog_cache.json"
    )
    for candidate in candidates:
        if _can_create_under(candidate.parent):
            return candidate
    return None


def _clean_text(value: object, *, limit: int) -> str:
    text = str(value or "").strip()
    if not text or len(text) > limit or any(char in text for char in "\r\n\0"):
        return ""
    return text


def _clean_values(
    values: object,
    *,
    limit: int,
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    cleaned = {
        text
        for value in values[:_MAX_CATALOG_ITEMS]
        if (text := _clean_text(value, limit=limit))
    }
    return tuple(sorted(cleaned, key=str.casefold))


@dataclass(frozen=True)
class CatalogSnapshot:
    values: tuple[str, ...] = ()
    source: str = ""
    fetched_at: str = ""
    litellm_version: str = ""


def catalog_snapshot_warning(
    snapshot: CatalogSnapshot,
    *,
    current_litellm_version: str = "",
    now: datetime | None = None,
) -> str:
    """Describe stale or incompatible cache metadata without discarding data."""
    if not snapshot.values:
        return ""

    warnings: list[str] = []
    raw_fetched_at = snapshot.fetched_at.strip()
    if raw_fetched_at:
        try:
            fetched_at = datetime.fromisoformat(raw_fetched_at.replace("Z", "+00:00"))
        except ValueError:
            warnings.append("缓存更新时间无效，请联网刷新。")
        else:
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=timezone.utc)
            current = now or datetime.now(timezone.utc)
            if current.tzinfo is None:
                current = current.replace(tzinfo=timezone.utc)
            if current.astimezone(timezone.utc) - fetched_at.astimezone(timezone.utc) > timedelta(
                days=CATALOG_CACHE_MAX_AGE_DAYS
            ):
                warnings.append(f"缓存已超过 {CATALOG_CACHE_MAX_AGE_DAYS} 天，请联网刷新。")
    if (
        snapshot.litellm_version
        and current_litellm_version
        and snapshot.litellm_version != current_litellm_version
    ):
        warnings.append(
            f"缓存适用于 LiteLLM {snapshot.litellm_version}，当前为 {current_litellm_version}。"
        )
    return " ".join(warnings)

class LiteLLMCatalogCache:
    """Load and atomically persist sanitized LiteLLM GUI catalog state."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.fallback_reason = ""
        if path is None:
            default_path = default_litellm_catalog_cache_path()
            if _can_create_under(default_path.parent):
                self.path = default_path
            else:
                fallback = _fallback_litellm_cache_path()
                if fallback is not None:
                    self.path = fallback
                    self.fallback_reason = (
                        "默认 LiteLLM 用户目录缓存不可写，已回退到临时目录："
                        f"{fallback.parent}"
                    )
                else:
                    self.path = default_path
        else:
            self.path = Path(path)
        self._now = now or (lambda: datetime.now(timezone.utc))
        self.load_error = ""
        self._selected_provider = ""
        self._selected_models: dict[str, str] = {}
        self._providers = CatalogSnapshot()
        self._models: dict[str, CatalogSnapshot] = {}
        self._load()

    @property
    def selected_provider(self) -> str:
        return self._selected_provider

    @property
    def providers(self) -> CatalogSnapshot:
        return self._providers

    def selected_model(self, provider: str) -> str:
        provider = _clean_text(provider, limit=_MAX_PROVIDER_LENGTH).lower()
        return self._selected_models.get(provider, "")

    def models(self, provider: str) -> CatalogSnapshot:
        provider = _clean_text(provider, limit=_MAX_PROVIDER_LENGTH).lower()
        return self._models.get(provider, CatalogSnapshot())

    def select_provider(self, provider: str) -> None:
        """Lowercase and atomically persist the provider; OSError may propagate."""
        self._selected_provider = _clean_text(
            provider, limit=_MAX_PROVIDER_LENGTH
        ).lower()
        self._save()

    def select_model(self, provider: str, model: str) -> None:
        """Persist a model atomically, removing it when empty; OSError may propagate."""
        provider = _clean_text(provider, limit=_MAX_PROVIDER_LENGTH).lower()
        model = _clean_text(model, limit=_MAX_MODEL_LENGTH)
        if not provider:
            return
        if model:
            self._selected_models[provider] = model
        else:
            self._selected_models.pop(provider, None)
        self._save()

    def update_providers(
        self,
        providers: tuple[str, ...] | list[str],
        *,
        source: str,
        litellm_version: str = "",
    ) -> None:
        """Normalize and atomically persist providers; OSError may propagate."""
        self._providers = CatalogSnapshot(
            values=tuple(
                value.lower()
                for value in _clean_values(
                    providers,
                    limit=_MAX_PROVIDER_LENGTH,
                )
            ),
            source=_clean_text(source, limit=100),
            fetched_at=self._timestamp(),
            litellm_version=_clean_text(litellm_version, limit=100),
        )
        self._save()

    def update_models(
        self,
        provider: str,
        models: tuple[str, ...] | list[str],
        *,
        source: str,
        litellm_version: str = "",
    ) -> None:
        """Normalize and atomically persist provider models; OSError may propagate."""
        provider = _clean_text(provider, limit=_MAX_PROVIDER_LENGTH).lower()
        if not provider:
            raise ValueError("Provider 不能为空。")
        self._models[provider] = CatalogSnapshot(
            values=_clean_values(models, limit=_MAX_MODEL_LENGTH),
            source=_clean_text(source, limit=100),
            fetched_at=self._timestamp(),
            litellm_version=_clean_text(litellm_version, limit=100),
        )
        self._save()

    def remove_provider(self, provider: str) -> None:
        """Drop every cached artifact for *provider* (models, selection).

        Used when a custom provider is deleted so its id cannot resurface in
        dropdowns or restores from the persistent cache. OSError may propagate.
        """
        provider = _clean_text(provider, limit=_MAX_PROVIDER_LENGTH).lower()
        if not provider:
            return
        self._models.pop(provider, None)
        self._selected_models.pop(provider, None)
        if self._selected_provider == provider:
            self._selected_provider = ""
        self._save()

    def _timestamp(self) -> str:
        value = self._now()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._apply_payload(payload)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            self.load_error = f"LiteLLM 目录缓存无效，已忽略：{exc}"

    def _apply_payload(self, payload: object) -> None:
        if not isinstance(payload, Mapping):
            raise ValueError("缓存根节点必须是对象")
        if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError("不支持的缓存版本")

        self._selected_provider = _clean_text(
            payload.get("selected_provider"),
            limit=_MAX_PROVIDER_LENGTH,
        ).lower()
        raw_selected = payload.get("selected_models")
        if isinstance(raw_selected, Mapping):
            for raw_provider, raw_model in raw_selected.items():
                provider = _clean_text(
                    raw_provider,
                    limit=_MAX_PROVIDER_LENGTH,
                ).lower()
                model = _clean_text(raw_model, limit=_MAX_MODEL_LENGTH)
                if provider and model:
                    self._selected_models[provider] = model

        self._providers = self._snapshot_from_payload(
            payload.get("providers"),
            value_limit=_MAX_PROVIDER_LENGTH,
            lowercase=True,
        )
        raw_models = payload.get("models")
        if isinstance(raw_models, Mapping):
            for raw_provider, raw_snapshot in raw_models.items():
                provider = _clean_text(
                    raw_provider,
                    limit=_MAX_PROVIDER_LENGTH,
                ).lower()
                if not provider:
                    continue
                snapshot = self._snapshot_from_payload(
                    raw_snapshot,
                    value_limit=_MAX_MODEL_LENGTH,
                )
                if snapshot.values:
                    self._models[provider] = snapshot

    @staticmethod
    def _snapshot_from_payload(
        payload: object,
        *,
        value_limit: int,
        lowercase: bool = False,
    ) -> CatalogSnapshot:
        if not isinstance(payload, Mapping):
            return CatalogSnapshot()
        values = _clean_values(payload.get("values"), limit=value_limit)
        if lowercase:
            values = tuple(value.lower() for value in values)
        return CatalogSnapshot(
            values=values,
            source=_clean_text(payload.get("source"), limit=100),
            fetched_at=_clean_text(payload.get("fetched_at"), limit=100),
            litellm_version=_clean_text(payload.get("litellm_version"), limit=100),
        )

    @staticmethod
    def _snapshot_payload(snapshot: CatalogSnapshot) -> dict[str, object]:
        return {
            "values": list(snapshot.values),
            "source": snapshot.source,
            "fetched_at": snapshot.fetched_at,
            "litellm_version": snapshot.litellm_version,
        }

    def _save(self) -> None:
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "selected_provider": self._selected_provider,
            "selected_models": dict(sorted(self._selected_models.items())),
            "providers": self._snapshot_payload(self._providers),
            "models": {
                provider: self._snapshot_payload(snapshot)
                for provider, snapshot in sorted(self._models.items())
            },
        }
        atomic_write_json(self.path, payload, ensure_ascii=False, indent=2)
