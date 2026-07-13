"""Background workers for operations that may import or call LiteLLM."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from litellm_provider_config import models_for_provider
from litellm_sync_backend import LiteLLMSyncBackend
from sync_model_backend import SyncGenerationRequest


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
                    config={"max_output_tokens": 8, "temperature": 0},
                )
            )
            text = result.response_text.strip()
            self.completed.emit(True, f"连接成功。模型返回：{text[:80] or '（空响应）'}")
        except Exception as exc:
            message = str(exc)
            if self.api_key:
                message = message.replace(self.api_key, "******")
            self.completed.emit(False, f"连接失败：{message}")
