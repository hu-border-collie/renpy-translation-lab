import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from gui_qt.litellm_catalog_cache import (
    CatalogSnapshot,
    LiteLLMCatalogCache,
    catalog_snapshot_warning,
)


class LiteLLMCatalogCacheTests(unittest.TestCase):
    def test_fresh_cache_has_no_provider_or_model_selection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = LiteLLMCatalogCache(Path(temp_dir) / "catalog.json")

            self.assertEqual(cache.selected_provider, "")
            self.assertEqual(cache.providers.values, ())
            self.assertEqual(cache.models("openai").values, ())
            self.assertEqual(cache.selected_model("openai"), "")

    def test_catalogs_and_per_provider_selections_survive_reload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "catalog.json"
            def now():
                return datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)

            cache = LiteLLMCatalogCache(path, now=now)
            cache.update_providers(
                ["anthropic", "openai"],
                source="online",
                litellm_version="1.83.7",
            )
            cache.update_models(
                "openai",
                ["openai/gpt-current"],
                source="openai",
                litellm_version="1.83.7",
            )
            cache.select_provider("openai")
            cache.select_model("openai", "openai/gpt-current")

            restored = LiteLLMCatalogCache(path)

            self.assertEqual(restored.selected_provider, "openai")
            self.assertEqual(restored.providers.values, ("anthropic", "openai"))
            self.assertEqual(
                restored.models("openai").values,
                ("openai/gpt-current",),
            )
            self.assertEqual(
                restored.selected_model("openai"),
                "openai/gpt-current",
            )
            self.assertEqual(
                restored.models("openai").fetched_at,
                "2026-07-28T12:00:00Z",
            )

    def test_remove_provider_drops_models_selection_and_persists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "catalog.json"
            cache = LiteLLMCatalogCache(path)
            cache.update_models(
                "opencode-go",
                ["opencode-go/gpt-4o-mini"],
                source="opencode-go",
            )
            cache.select_provider("opencode-go")
            cache.select_model("opencode-go", "opencode-go/gpt-4o-mini")

            cache.remove_provider("opencode-go")

            self.assertEqual(cache.models("opencode-go").values, ())
            self.assertEqual(cache.selected_model("opencode-go"), "")
            self.assertEqual(cache.selected_provider, "")
            restored = LiteLLMCatalogCache(path)
            self.assertEqual(restored.models("opencode-go").values, ())
            self.assertEqual(restored.selected_provider, "")

    def test_corrupt_or_wrong_version_cache_is_ignored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "catalog.json"
            path.write_text("{not-json", encoding="utf-8")
            corrupt = LiteLLMCatalogCache(path)
            self.assertEqual(corrupt.providers.values, ())
            self.assertIn("缓存无效", corrupt.load_error)

            path.write_text(
                json.dumps({"schema_version": 999}),
                encoding="utf-8",
            )
            wrong_version = LiteLLMCatalogCache(path)
            self.assertEqual(wrong_version.providers.values, ())
            self.assertIn("不支持的缓存版本", wrong_version.load_error)

    def test_unknown_and_sensitive_fields_are_not_rewritten(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "catalog.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "selected_provider": "openai",
                        "selected_models": {},
                        "providers": {
                            "values": ["openai"],
                            "source": "online",
                            "api_key": "stored-secret",
                        },
                        "models": {},
                        "authorization": "Bearer stored-secret",
                    }
                ),
                encoding="utf-8",
            )
            cache = LiteLLMCatalogCache(path)
            cache.select_provider("openai")

            rewritten = path.read_text(encoding="utf-8")
            self.assertNotIn("stored-secret", rewritten)
            self.assertNotIn("authorization", rewritten.lower())
            self.assertNotIn("api_key", rewritten.lower())


    def test_stale_invalid_or_old_version_cache_is_warned_but_retained(self):
        snapshot = CatalogSnapshot(
            values=("openai/gpt-current",),
            fetched_at="2026-01-01T00:00:00Z",
            litellm_version="1.0.0",
        )
        warning = catalog_snapshot_warning(
            snapshot,
            current_litellm_version="2.0.0",
            now=datetime(2026, 7, 29, tzinfo=timezone.utc),
        )
        self.assertIn("超过 30 天", warning)
        self.assertIn("当前为 2.0.0", warning)
        self.assertEqual(snapshot.values, ("openai/gpt-current",))

        invalid = CatalogSnapshot(values=("openai/gpt",), fetched_at="not-a-date")
        self.assertIn(
            "更新时间无效",
            catalog_snapshot_warning(invalid),
        )

    def test_default_cache_falls_back_when_default_parent_is_not_writable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            blocker = Path(temp_dir) / "not-a-directory"
            blocker.write_text("x", encoding="utf-8")
            default_path = blocker / "litellm_catalog_cache.json"
            fallback_dir = Path(temp_dir) / "fallback"

            with (
                mock.patch(
                    "gui_qt.litellm_catalog_cache.default_litellm_catalog_cache_path",
                    return_value=default_path,
                ),
                mock.patch(
                    "gui_qt.litellm_catalog_cache.tempfile.gettempdir",
                    return_value=str(fallback_dir),
                ),
            ):
                cache = LiteLLMCatalogCache()

            self.assertNotEqual(cache.path, default_path)
            self.assertEqual(
                cache.path,
                fallback_dir / "renpy-translation-lab" / "litellm_catalog_cache.json",
            )
            self.assertIn("回退到临时目录", cache.fallback_reason)

            cache.select_provider("openai")
            self.assertTrue(cache.path.is_file())
            self.assertEqual(
                LiteLLMCatalogCache(cache.path).selected_provider,
                "openai",
            )


if __name__ == "__main__":
    unittest.main()
