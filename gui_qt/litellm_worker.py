"""Background workers for operations that may import or call LiteLLM."""

from __future__ import annotations

import json
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PySide6.QtCore import QThread, Signal

from litellm_provider_config import (
    LITELLM_CATALOG_URL,
    LITELLM_PYPI_URL,
    build_native_catalog_headers,
    installed_litellm_version,
    latest_compatible_litellm_version,
    models_for_provider,
    models_from_native_catalog_payload,
    models_from_remote_catalog,
    native_catalog_endpoint,
)
from litellm_sync_backend import LiteLLMSyncBackend
from sync_model_backend import SyncGenerationRequest


CONNECTION_TEST_TIMEOUT_SECONDS = 30
CATALOG_TIMEOUT_SECONDS = 30


def _connection_error_message(exc: Exception) -> str:
    category = str(getattr(exc, "category", "provider_error") or "provider_error")
    details = {
        "authentication": "身份验证失败，请检查供应商密钥。",
        "rate_limit": "供应商限流或配额不足，请稍后重试。",
        "service_unavailable": "供应商服务暂时不可用或请求超时。",
        "missing_dependency": "LiteLLM 尚未正确安装。",
        "provider_error": "请求失败，请检查模型、API Base 和网络。",
    }
    return f"连接失败 [{category}]: {details.get(category, details['provider_error'])}"


def _http_error_message(exc: HTTPError, label: str) -> str:
    code = int(getattr(exc, "code", 0) or 0)
    if code in {401, 403}:
        return f"{label} 身份验证失败（HTTP {code}），请检查 API Key。"
    if code == 404:
        return f"{label} 模型列表接口不可用（HTTP 404）。"
    if code == 429:
        return f"{label} 限流（HTTP 429），请稍后重试。"
    return f"{label} 请求失败（HTTP {code}）。"


class LiteLLMModelCatalogWorker(QThread):
    completed = Signal(object, object, object)

    def __init__(self, provider: str, api_key: str = "", parent=None) -> None:
        super().__init__(parent)
        self.provider = str(provider or "").strip().lower()
        self.api_key = str(api_key or "").strip()
        # Network I/O already yields the GIL; keep the worker low-priority so
        # the settings form stays snappy while the catalog request is in flight.
        self.setPriority(QThread.Priority.LowPriority)

    def _fetch_litellm_catalog(self) -> tuple[str, ...]:
        request = Request(
            LITELLM_CATALOG_URL,
            headers={"User-Agent": "renpy-translation-lab"},
        )
        with urlopen(request, timeout=CATALOG_TIMEOUT_SECONDS) as response:
            catalog = json.load(response)
        if not isinstance(catalog, dict):
            raise ValueError("LiteLLM 官方目录格式无效")
        models = models_from_remote_catalog(self.provider, catalog)
        if not models:
            raise ValueError(f"LiteLLM 目录中没有 {self.provider} 文本模型")
        return models

    def _fetch_native_catalog(self) -> tuple[tuple[str, ...], str]:
        endpoint = native_catalog_endpoint(self.provider)
        if endpoint is None:
            raise ValueError(f"未配置 {self.provider} 官方模型列表")
        if endpoint.require_key and not self.api_key:
            raise ValueError(f"请先保存 {endpoint.label} API Key，再刷新官方模型列表")

        headers = build_native_catalog_headers(endpoint, self.api_key)
        request = Request(endpoint.url, headers=headers)
        try:
            with urlopen(request, timeout=CATALOG_TIMEOUT_SECONDS) as response:
                payload = json.load(response)
        except HTTPError as exc:
            raise RuntimeError(_http_error_message(exc, endpoint.label)) from exc

        models = models_from_native_catalog_payload(endpoint, payload)
        if not models:
            raise ValueError(f"{endpoint.label} 未返回可用文本模型")
        return models, endpoint.source

    def run(self) -> None:
        online_errors: list[str] = []
        try:
            endpoint = native_catalog_endpoint(self.provider)
            if endpoint is not None:
                # Prefer each provider's own live catalog. LiteLLM's pricing table
                # is only a subset and lags behind official model releases.
                try:
                    models, source = self._fetch_native_catalog()
                    self.completed.emit(models, source, None)
                    return
                except Exception as native_exc:
                    online_errors.append(f"{endpoint.label}：{native_exc}")
                    try:
                        models = self._fetch_litellm_catalog()
                        warning = (
                            f"{endpoint.label} 官方列表失败，已改用 LiteLLM 子集目录："
                            f"{native_exc}"
                        )
                        self.completed.emit(models, "online", warning)
                        return
                    except Exception as litellm_exc:
                        online_errors.append(f"LiteLLM：{litellm_exc}")
                        raise RuntimeError("；".join(online_errors)) from litellm_exc

            models = self._fetch_litellm_catalog()
            self.completed.emit(models, "online", None)
        except Exception as exc:
            try:
                models = models_for_provider(self.provider)
            except Exception as local_exc:
                self.completed.emit((), "", f"联网失败：{exc}；本地读取失败：{local_exc}")
                return
            self.completed.emit(models, "local", f"联网失败，已改用本地目录：{exc}")


class LiteLLMVersionWorker(QThread):
    completed = Signal(str, str, str, str, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setPriority(QThread.Priority.LowPriority)

    def run(self) -> None:
        installed = ""
        try:
            installed = installed_litellm_version()
            request = Request(
                LITELLM_PYPI_URL,
                headers={"User-Agent": "renpy-translation-lab"},
            )
            with urlopen(request, timeout=CATALOG_TIMEOUT_SECONDS) as response:
                payload = json.load(response)
            latest = str(payload.get("info", {}).get("version", "")).strip()
            if not latest:
                raise ValueError("PyPI 未返回最新版本")
            releases = payload.get("releases", {})
            if not isinstance(releases, dict):
                raise ValueError("PyPI 未返回版本兼容信息")
            compatible = latest_compatible_litellm_version(releases, sys.version_info[:3])
            requires_python = str(payload.get("info", {}).get("requires_python") or "")
            self.completed.emit(installed, latest, compatible, requires_python, None)
        except Exception as exc:
            self.completed.emit(installed, "", "", "", str(exc))


class LiteLLMConnectionTestWorker(QThread):
    completed = Signal(bool, str)

    def __init__(self, model: str, api_key: str = "", parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.api_key = api_key
        self.setPriority(QThread.Priority.LowPriority)

    def run(self) -> None:
        try:
            result = LiteLLMSyncBackend(api_key=self.api_key or None).generate(
                SyncGenerationRequest(
                    model=self.model,
                    contents="Reply with OK.",
                    config={
                        "max_output_tokens": 8,
                        "temperature": 0,
                        "timeout": CONNECTION_TEST_TIMEOUT_SECONDS,
                    },
                )
            )
            text = result.response_text.strip()
            self.completed.emit(True, f"连接成功。模型返回：{text[:80] or '（空响应）'}")
        except Exception as exc:
            self.completed.emit(False, _connection_error_message(exc))
