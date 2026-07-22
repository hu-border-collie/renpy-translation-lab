# -*- coding: utf-8 -*-
import unittest

import gemini_model_catalog as catalog


class GeminiModelCatalogTests(unittest.TestCase):
    def test_normalize_model_names_dedupes_and_strips(self):
        self.assertEqual(
            catalog.normalize_model_names(
                [" gemini-a ", "gemini-a", "", None, "gemini-b", 123]
            ),
            ["gemini-a", "gemini-b", "123"],
        )
        self.assertEqual(catalog.normalize_model_names("solo-model"), ["solo-model"])
        self.assertEqual(catalog.normalize_model_names(None), [])

    def test_default_rotation_puts_recommended_first(self):
        models = catalog.default_model_rotation_list()
        self.assertEqual(models[0], catalog.DEFAULT_GEMINI_TRANSLATION_MODEL)
        for name in catalog.BUILTIN_GEMINI_TRANSLATION_MODELS:
            self.assertIn(name, models)

    def test_resolve_merges_builtins_catalog_and_selected(self):
        config = {
            "model_catalog": {
                "gemini": ["gemini-custom-a", "gemini-3.6-flash", ""],
                "gemini_embedding": ["gemini-embedding-custom"],
            }
        }
        translation = catalog.resolve_gemini_translation_models(
            config,
            extra_selected=["gemini-selected", "gemini-custom-a"],
        )
        self.assertEqual(translation[0], catalog.BUILTIN_GEMINI_TRANSLATION_MODELS[0])
        self.assertIn("gemini-custom-a", translation)
        self.assertIn("gemini-selected", translation)
        # Builtins keep their relative order; catalog extras come after.
        self.assertLess(
            translation.index("gemini-3.6-flash"),
            translation.index("gemini-custom-a"),
        )

        embedding = catalog.resolve_gemini_embedding_models(
            config,
            extra_selected=["gemini-embedding-selected"],
        )
        self.assertIn("gemini-embedding-001", embedding)
        self.assertIn("gemini-embedding-custom", embedding)
        self.assertIn("gemini-embedding-selected", embedding)

    def test_write_model_catalog_extras_only_persists_non_builtins(self):
        config: dict = {
            "model_catalog": {
                "gemini": ["old-extra"],
                "gemini_embedding": ["old-emb"],
            }
        }
        catalog.write_model_catalog_extras(
            config,
            translation_models=[
                catalog.DEFAULT_GEMINI_TRANSLATION_MODEL,
                "gemini-user-extra",
            ],
            embedding_models=[
                catalog.DEFAULT_GEMINI_EMBEDDING_MODEL,
                "gemini-emb-extra",
            ],
        )
        self.assertEqual(config["model_catalog"]["gemini"], ["gemini-user-extra"])
        self.assertEqual(
            config["model_catalog"]["gemini_embedding"],
            ["gemini-emb-extra"],
        )

    def test_write_model_catalog_extras_removes_empty_section(self):
        config: dict = {
            "model_catalog": {
                "gemini": ["temporary"],
                "gemini_embedding": ["temporary-emb"],
            }
        }
        catalog.write_model_catalog_extras(
            config,
            translation_models=list(catalog.BUILTIN_GEMINI_TRANSLATION_MODELS),
            embedding_models=list(catalog.BUILTIN_GEMINI_EMBEDDING_MODELS),
        )
        self.assertNotIn("model_catalog", config)


if __name__ == "__main__":
    unittest.main()
