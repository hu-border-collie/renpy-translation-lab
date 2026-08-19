"""Stable GUI operation identities must be order-insensitive and comparable."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path

from gui_qt.operation_identity import (
    canonical_digest,
    context_library_config_digest,
    is_current_identity,
    litellm_connection_identity,
)


@dataclass(frozen=True)
class _ProviderStub:
    id: str
    base_url: str


class OperationIdentityTests(unittest.TestCase):
    def test_canonical_digest_ignores_dict_key_order_and_list_vs_tuple(self) -> None:
        left = {"batch": {"rag": True, "source_index": False}, "items": ("a", "b")}
        right = {"items": ["a", "b"], "batch": {"source_index": False, "rag": True}}
        self.assertEqual(canonical_digest(left), canonical_digest(right))

    def test_canonical_digest_normalizes_non_finite_floats(self) -> None:
        self.assertEqual(
            canonical_digest({"n": float("nan")}),
            canonical_digest({"n": "NaN"}),
        )
        self.assertEqual(
            canonical_digest({"n": float("inf")}),
            canonical_digest({"n": "Infinity"}),
        )
        self.assertEqual(
            canonical_digest({"n": float("-inf")}),
            canonical_digest({"n": "-Infinity"}),
        )
        self.assertNotEqual(
            canonical_digest({"n": float("nan")}),
            canonical_digest({"n": float("inf")}),
        )

    def test_canonical_digest_changes_when_values_change(self) -> None:
        self.assertNotEqual(
            canonical_digest({"rag": True}),
            canonical_digest({"rag": False}),
        )

    def test_canonical_digest_normalizes_paths_and_dataclasses(self) -> None:
        root = Path("C:/Games/Demo")
        left = {"root": root, "provider": _ProviderStub("go", "https://x")}
        right = {
            "root": str(root),
            "provider": {"id": "go", "base_url": "https://x"},
        }
        self.assertEqual(canonical_digest(left), canonical_digest(right))

    def test_empty_result_identity_is_treated_as_unspecified(self) -> None:
        self.assertTrue(is_current_identity("", "anything"))
        self.assertTrue(is_current_identity("abc", "abc"))
        self.assertFalse(is_current_identity("abc", "xyz"))

    def test_context_library_digest_follows_effective_flags_not_raw_config(self) -> None:
        enabled = {"batch": {"rag": {"enabled": True}}, "theme": "dark"}
        disabled = {"batch": {"rag": {"enabled": False}}, "theme": "dark"}
        theme_only = {"batch": {"rag": {"enabled": True}}, "theme": "light"}
        self.assertNotEqual(
            context_library_config_digest(enabled),
            context_library_config_digest(disabled),
        )
        self.assertEqual(
            context_library_config_digest(enabled),
            context_library_config_digest(theme_only),
        )
        same_config = {"theme": "dark"}
        self.assertNotEqual(
            context_library_config_digest(
                same_config,
                context_flags={"rag_enabled": True},
            ),
            context_library_config_digest(
                same_config,
                context_flags={"rag_enabled": False},
            ),
        )

    def test_connection_identity_binds_provider_model_and_selected_endpoint(self) -> None:
        first = litellm_connection_identity(
            provider="go",
            model="go/gpt-test",
            custom_providers={"go": _ProviderStub("go", "https://a")},
        )
        same = litellm_connection_identity(
            provider="GO",
            model="go/gpt-test",
            custom_providers={"go": _ProviderStub("go", "https://a")},
        )
        other_model = litellm_connection_identity(
            provider="go",
            model="go/gpt-other",
            custom_providers={"go": _ProviderStub("go", "https://a")},
        )
        other_endpoint = litellm_connection_identity(
            provider="go",
            model="go/gpt-test",
            custom_providers={"go": _ProviderStub("go", "https://b")},
        )
        unused_row = litellm_connection_identity(
            provider="go",
            model="go/gpt-test",
            custom_providers={
                "go": _ProviderStub("go", "https://a"),
                "other": _ProviderStub("other", "https://z"),
            },
        )
        builtin_ignores_registry = litellm_connection_identity(
            provider="openai",
            model="openai/gpt-test",
            custom_providers={"go": _ProviderStub("go", "https://a")},
        )
        builtin_other_registry = litellm_connection_identity(
            provider="openai",
            model="openai/gpt-test",
            custom_providers={"go": _ProviderStub("go", "https://b")},
        )
        self.assertEqual(first, same)
        self.assertEqual(first, unused_row)
        self.assertEqual(builtin_ignores_registry, builtin_other_registry)
        self.assertNotEqual(first, other_model)
        self.assertNotEqual(first, other_endpoint)


if __name__ == "__main__":
    unittest.main()
