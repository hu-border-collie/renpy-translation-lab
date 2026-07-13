"""Background workers for operations that may import or call LiteLLM."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from litellm_provider_config import models_for_provider
from litellm_sync_backend import LiteLLMSyncBackend
from sync_model_backend import SyncGenerationRequest


CONNECTION_TEST_TIMEOUT_SECONDS = 30


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


class LiteLLMModelCatalogWorker(QThread):
    completed = Signal(object, object)

    def __init__(self, provider: str, parent=None) -> None:
        super().__init__(parent)
        self.provider = provider

    def run(self) -> None:
        try:
            self.completed.emit(models_for_provider(self.provider), None)
        except Exception as exc:
            self.completed.emit((), str(exc))


class LiteLLMConnectionTestWorker(QThread):
    completed = Signal(bool, str)

    def __init__(self, model: str, api_key: str = "", parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.api_key = api_key

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
