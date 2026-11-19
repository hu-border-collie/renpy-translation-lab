import unittest

from gui_qt.settings.settings_schema import (
    ADVANCED_SETTING_FIELD_BY_KEY,
    ADVANCED_SETTING_FIELDS,
    CORE_ADVANCED_SETTING_FIELDS,
    FULL_COVERAGE_SETTING_FIELDS,
    SettingField,
    apply_advanced_settings,
    filter_gemini_rotation_models,
    normalize_for_write,
    read_advanced_settings,
    read_setting,
    recommended_advanced_settings,
    resolve_project_analysis_flags_for_save,
    validate_advanced_settings,
    validate_value,
)


class GuiSettingsSchemaTests(unittest.TestCase):
    def test_quality_policy_json_round_trip_uses_runtime_reader(self):
        import json
        from quality_review import translation_quality as quality

        config = {
            'game_root': 'C:/Game/work',
            'batch': {
                'quality_gate': {
                    'rules': {'suspicious_english_residue': 'warning'},
                    'allowed_latin_tokens': ['RenPy'],
                    'language_allowed_latin_tokens': ['Alice'],
                    'typography_exempt_latin_tokens': ['HP'],
                    'unrelated_key': {'keep': True},
                },
                'unrelated_batch': {'keep': True},
            },
        }
        values = read_advanced_settings(config)
        self.assertEqual(values['batch_quality_gate'], config['batch']['quality_gate'])
        edited = dict(values['batch_quality_gate'])
        edited['typography_exempt_latin_tokens'] = ['Alice']
        values['batch_quality_gate'] = json.dumps(edited, ensure_ascii=False)
        saved = apply_advanced_settings(config, values)
        reloaded = read_advanced_settings(saved)['batch_quality_gate']
        self.assertEqual(reloaded, edited)
        self.assertEqual(saved['batch']['unrelated_batch'], {'keep': True})
        policy = quality.load_policy_from_config(saved)
        self.assertEqual(policy['language_allowed_latin_tokens'], ['Alice'])
        self.assertEqual(policy['typography_exempt_latin_tokens'], ['Alice'])

    def test_missing_quality_policy_gui_default_does_not_materialize_new_fields(self):
        config = {'game_root': 'C:/Game/work'}
        values = read_advanced_settings(config)
        default = values['batch_quality_gate']
        self.assertNotIn('language_allowed_latin_tokens', default)
        self.assertNotIn('typography_exempt_latin_tokens', default)
        saved = apply_advanced_settings(config, values)
        self.assertNotIn('batch', saved)
        edited = dict(default)
        edited['rules'] = {'ascii_ellipsis': 'warning'}
        values['batch_quality_gate'] = edited
        saved = apply_advanced_settings(config, values)
        self.assertNotIn('language_allowed_latin_tokens', saved['batch']['quality_gate'])
        self.assertNotIn('typography_exempt_latin_tokens', saved['batch']['quality_gate'])

    def test_user_visible_setting_titles_use_chinese_copy(self):
        expected_labels = {
            "sync_chunk_size": "同步分块条数",
            "sync_timeout_seconds": "同步单请求超时",
            "batch_chunk_size": "批量分块条数",
            "batch_retry_chunk_size": "补译分块条数",
            "keyword_chunk_size": "关键词分块条数",
            "keyword_max_candidates_per_chunk": "每分块候选上限",
            "revision_chunk_size": "订正分块条数",
            "batch_safety_settings": "安全设置",
        }
        for key, expected_label in expected_labels.items():
            with self.subTest(key=key):
                field = ADVANCED_SETTING_FIELD_BY_KEY[key]
                self.assertEqual(field.label, expected_label)
                self.assertNotIn("chunk", field.description)

    def test_project_analysis_save_flags_preserve_saved_values_without_widgets(self):
        flags = resolve_project_analysis_flags_for_save(
            {
                "project_analysis_enabled": True,
                "project_analysis_inject_enabled": False,
            },
            {},
            {},
        )

        self.assertEqual(
            flags,
            {
                "project_analysis_enabled": True,
                "project_analysis_inject_enabled": False,
            },
        )

    def test_project_analysis_save_flags_use_present_ui_values(self):
        flags = resolve_project_analysis_flags_for_save(
            {
                "project_analysis_enabled": True,
                "project_analysis_inject_enabled": False,
            },
            {
                "batch_project_analysis_enabled": False,
                "batch_project_analysis_inject_published_brief": True,
            },
            {
                "batch_project_analysis_enabled": False,
                "batch_project_analysis_inject_published_brief": True,
            },
        )

        self.assertEqual(
            flags,
            {
                "project_analysis_enabled": False,
                "project_analysis_inject_enabled": True,
            },
        )

    def test_read_advanced_settings_uses_defaults_for_missing_config(self):
        values = read_advanced_settings({})

        self.assertEqual(values["sync_chunk_size"], 60)
        self.assertEqual(values["sync_max_source_chars"], 18000)
        self.assertEqual(values["sync_story_memory_max_context_chars"], 1200)
        self.assertEqual(values["sync_timeout_seconds"], 120)
        self.assertEqual(values["sync_context_before"], 30)
        self.assertEqual(values["sync_context_after"], 10)
        self.assertEqual(values["sync_macro_setting_file"], "macro_setting.md")
        self.assertEqual(values["batch_chunk_size"], 60)
        self.assertEqual(values["batch_temperature"], 0.2)
        self.assertEqual(values["batch_source_index_min_similarity"], 0.72)
        self.assertEqual(values["context_storage_game_dir_name"], "translation_context")
        self.assertEqual(values["batch_rag_store_dir"], "")

    def test_read_advanced_settings_coerces_valid_existing_values(self):
        values = read_advanced_settings(
            {
                "sync": {
                    "chunk_size": "7",
                    "timeout_seconds": "45",
                    "context_before": "12",
                    "context_after": "0",
                    "macro_setting_file": "project_style.md",
                    "rag": {
                        "enabled": True,
                        "min_similarity": "0.5",
                        "store_dir": " logs/sync_rag ",
                    },
                },
                "batch": {
                    "temperature": "0.6",
                    "source_index": {"top_k": "3"},
                    "story_memory": {"include_scene_summary": False},
                },
            }
        )

        self.assertEqual(values["sync_chunk_size"], 7)
        self.assertEqual(values["sync_timeout_seconds"], 45)
        self.assertEqual(values["sync_context_before"], 12)
        self.assertEqual(values["sync_context_after"], 0)
        self.assertEqual(values["sync_macro_setting_file"], "project_style.md")
        self.assertTrue(values["sync_rag_enabled"])
        self.assertEqual(values["sync_rag_min_similarity"], 0.5)
        self.assertEqual(values["sync_rag_store_dir"], "logs/sync_rag")
        self.assertEqual(values["batch_temperature"], 0.6)
        self.assertEqual(values["batch_source_index_top_k"], 3)
        self.assertFalse(values["batch_story_memory_include_scene_summary"])

    def test_validate_advanced_settings_rejects_out_of_range_values(self):
        values = recommended_advanced_settings()
        values["batch_chunk_size"] = 0
        values["batch_temperature"] = 2.5
        values["sync_rag_min_similarity"] = -0.1
        values["sync_timeout_seconds"] = 601
        values["context_storage_game_dir_name"] = ""

        errors = validate_advanced_settings(values)

        self.assertIn("batch_chunk_size", errors)
        self.assertIn("batch_temperature", errors)
        self.assertIn("sync_rag_min_similarity", errors)
        self.assertIn("sync_timeout_seconds", errors)
        self.assertIn("context_storage_game_dir_name", errors)

    def test_list_and_json_fields_round_trip_from_text(self):
        config = {"game_root": "C:/Game/work", "batch": {}}
        values = recommended_advanced_settings()
        values["game_root"] = "C:/Game/work"
        values["include_files"] = "chapter01.rpy\nchapter02.rpy"
        values["include_prefixes"] = "routes/common, routes/extra"
        values["batch_safety_settings"] = '[{"category":"sexually_explicit","threshold":"BLOCK_NONE"}]'
        values["prepare_unpack_command"] = '["python", "unpack.py"]'
        values["batch_macro_setting"] = "Use a concise voice.\nKeep honorifics."

        saved = apply_advanced_settings(config, values)

        self.assertEqual(saved["include_files"], ["chapter01.rpy", "chapter02.rpy"])
        self.assertEqual(saved["include_prefixes"], ["routes/common", "routes/extra"])
        self.assertEqual(
            saved["batch"]["safety_settings"],
            [{"category": "sexually_explicit", "threshold": "BLOCK_NONE"}],
        )
        self.assertEqual(saved["prepare"]["unpack_command"], ["python", "unpack.py"])
        self.assertEqual(saved["batch"]["macro_setting"], "Use a concise voice.\nKeep honorifics.")

    def test_invalid_json_field_reports_error(self):
        values = recommended_advanced_settings()
        values["game_root"] = "C:/Game/work"
        values["batch_safety_settings"] = '[{"bad"'

        errors = validate_advanced_settings(values)

        self.assertIn("batch_safety_settings", errors)
        self.assertIn("有效 JSON", errors["batch_safety_settings"])

    def test_validate_advanced_settings_allows_empty_paths(self):
        values = recommended_advanced_settings()
        values["batch_rag_store_dir"] = ""
        values["batch_source_index_store_dir"] = ""
        values["batch_story_memory_graph_file"] = ""

        errors = validate_advanced_settings(values)

        self.assertNotIn("batch_rag_store_dir", errors)
        self.assertNotIn("batch_source_index_store_dir", errors)
        self.assertNotIn("batch_story_memory_graph_file", errors)

    def test_apply_advanced_settings_preserves_unknown_fields(self):
        config = {
            "game_root": "C:/Game/work",
            "sync": {"model": "gemini-sync", "custom": 1},
            "batch": {
                "model": "gemini-batch",
                "rag": {"enabled": True, "unknown_rag": "keep"},
                "source_index": {"enabled": True},
            },
            "unknown_top": {"keep": True},
        }
        values = recommended_advanced_settings()
        values["game_root"] = "C:/Game/work"
        values["sync_chunk_size"] = 12
        values["batch_temperature"] = 0.4
        values["batch_rag_top_k_history"] = 9
        values["batch_source_index_store_dir"] = "C:/ctx/source"

        saved = apply_advanced_settings(config, values)

        self.assertIs(saved, config)
        self.assertEqual(saved["game_root"], "C:/Game/work")
        self.assertEqual(saved["sync"]["model"], "gemini-sync")
        self.assertEqual(saved["sync"]["custom"], 1)
        self.assertEqual(saved["batch"]["model"], "gemini-batch")
        self.assertTrue(saved["batch"]["rag"]["enabled"])
        self.assertEqual(saved["batch"]["rag"]["unknown_rag"], "keep")
        self.assertTrue(saved["batch"]["source_index"]["enabled"])
        self.assertEqual(saved["unknown_top"], {"keep": True})
        self.assertEqual(saved["sync"]["chunk_size"], 12)
        self.assertEqual(saved["batch"]["temperature"], 0.4)
        self.assertEqual(saved["batch"]["rag"]["top_k_history"], 9)
        self.assertEqual(saved["batch"]["source_index"]["store_dir"], "C:/ctx/source")

    def test_field_registry_exposes_expected_paths(self):
        self.assertEqual(
            ADVANCED_SETTING_FIELD_BY_KEY["batch_source_index_char_limit"].path,
            ("batch", "source_index", "char_limit"),
        )
        self.assertEqual(
            ADVANCED_SETTING_FIELD_BY_KEY["sync_story_memory_graph_file"].path,
            ("sync", "story_memory", "graph_file"),
        )
        self.assertEqual(
            ADVANCED_SETTING_FIELD_BY_KEY["sync_context_before"].path,
            ("sync", "context_before"),
        )
        self.assertEqual(
            ADVANCED_SETTING_FIELD_BY_KEY["sync_context_after"].path,
            ("sync", "context_after"),
        )
        self.assertEqual(
            ADVANCED_SETTING_FIELD_BY_KEY["sync_macro_setting_file"].path,
            ("sync", "macro_setting_file"),
        )
        self.assertEqual(
            ADVANCED_SETTING_FIELD_BY_KEY["model_rotation_models"].kind,
            "gemini_model_list",
        )

    def test_gemini_model_list_rejects_unknown_ids(self):
        with self.assertRaises(ValueError):
            filter_gemini_rotation_models(
                ["gemini-3.1-flash-lite", "totally-fake-model"],
                reject_unknown=True,
            )
        self.assertEqual(
            filter_gemini_rotation_models(
                ["gemini-3.1-flash-lite", "totally-fake-model"],
                reject_unknown=False,
            ),
            ["gemini-3.1-flash-lite"],
        )

    def test_catalog_gemini_lists_strip_builtins_on_write(self):
        config = {"game_root": "C:/Game/work"}
        values = recommended_advanced_settings()
        values["game_root"] = "C:/Game/work"
        values["catalog_gemini_models"] = [
            "gemini-3.1-flash-lite",
            "gemini-custom-extra",
        ]
        values["catalog_gemini_embedding_models"] = [
            "gemini-embedding-001",
            "gemini-embedding-custom",
        ]
        saved = apply_advanced_settings(config, values)
        self.assertEqual(saved["model_catalog"]["gemini"], ["gemini-custom-extra"])
        self.assertEqual(
            saved["model_catalog"]["gemini_embedding"],
            ["gemini-embedding-custom"],
        )
        self.assertEqual(
            ADVANCED_SETTING_FIELD_BY_KEY["catalog_gemini_models"].kind,
            "gemini_catalog_list",
        )

    def test_rotation_accepts_catalog_entry_added_in_same_save(self):
        """New catalog extras must be valid rotation picks in one apply pass."""
        config = {"game_root": "C:/Game/work"}
        values = recommended_advanced_settings()
        values["game_root"] = "C:/Game/work"
        values["catalog_gemini_models"] = ["gemini-brand-new-extra"]
        values["model_rotation_enabled"] = True
        values["model_rotation_models"] = [
            "gemini-3.1-flash-lite",
            "gemini-brand-new-extra",
        ]
        # Must not reject brand-new-extra as unknown against the old empty catalog.
        errors = validate_advanced_settings(values, translator_config=config)
        self.assertNotIn("model_rotation_models", errors)
        saved = apply_advanced_settings(config, values)
        self.assertEqual(saved["model_catalog"]["gemini"], ["gemini-brand-new-extra"])
        self.assertEqual(
            saved["rotation"]["model"]["models"],
            ["gemini-3.1-flash-lite", "gemini-brand-new-extra"],
        )

    def test_apply_advanced_settings_skips_untouched_defaults_and_partial_keys(self):
        config = {"game_root": "C:/Game/work"}
        values = read_advanced_settings(config)
        saved = apply_advanced_settings(config, values)
        self.assertEqual(saved, {"game_root": "C:/Game/work"})

        partial_saved = apply_advanced_settings(config, {"sync_chunk_size": 12})
        self.assertEqual(
            partial_saved,
            {"game_root": "C:/Game/work", "sync": {"chunk_size": 12}},
        )

        # Updating an existing key back to its default writes the reset value.
        reset_saved = apply_advanced_settings(partial_saved, {"sync_chunk_size": 60})
        self.assertEqual(reset_saved["sync"]["chunk_size"], 60)

    def test_read_setting_preserves_invalid_values_and_blocks_overwrite(self):
        config = {
            "engine": "godot",
            "rpgmaker_mv": {"display_fields": ["database"]},
            "rotation": {
                "api_key": {"enabled": "yes"},
                "model": {"models": ["unknown-model"]},
            },
            "sync": {
                "chunk_size": "abc",
                "max_source_chars": 0,
                "rag": {"embedding_backend": "vertex"},
            },
            "batch": {
                "temperature": 2.5,
                "safety_settings": '[{"bad"',
                "rag": {"embedding_backend": "local"},
            },
            "context_storage": {"game_dir_name": ""},
        }
        values = read_advanced_settings(config)
        self.assertEqual(values["engine"], "godot")
        self.assertEqual(values["mv_display_fields"], ["database"])
        self.assertEqual(values["api_key_rotation_enabled"], "yes")
        self.assertEqual(values["model_rotation_models"], ["unknown-model"])
        self.assertEqual(values["sync_chunk_size"], "abc")
        self.assertEqual(values["sync_max_source_chars"], 0)
        self.assertEqual(values["sync_rag_embedding_backend"], "vertex")
        self.assertEqual(values["batch_temperature"], 2.5)
        self.assertEqual(values["batch_safety_settings"], '[{"bad"')
        self.assertEqual(values["batch_rag_embedding_backend"], "local")
        self.assertEqual(values["context_storage_game_dir_name"], "")
        self.assertEqual(
            read_setting({"sync": {"chunk_size": True}}, ADVANCED_SETTING_FIELD_BY_KEY["sync_chunk_size"]),
            True,
        )

        errors = validate_advanced_settings(values, translator_config=config)
        for key in (
            "engine",
            "mv_display_fields",
            "api_key_rotation_enabled",
            "model_rotation_models",
            "sync_chunk_size",
            "sync_max_source_chars",
            "sync_rag_embedding_backend",
            "batch_temperature",
            "batch_safety_settings",
            "batch_rag_embedding_backend",
            "context_storage_game_dir_name",
        ):
            self.assertIn(key, errors)
        with self.assertRaises(ValueError):
            apply_advanced_settings(config, values)

    def test_declarative_choices_validation(self):
        for key, valid_value, invalid_value in (
            ("engine", "rpgmaker_mv", "godot"),
            ("sync_rag_embedding_backend", "openai_compatible", "azure"),
            ("batch_rag_embedding_backend", "gemini", "local"),
            ("mv_display_fields", ["system_terms", "map_display_name"], ["database"]),
        ):
            field = ADVANCED_SETTING_FIELD_BY_KEY[key]
            self.assertIsNotNone(field.choices)
            self.assertEqual(validate_value(field, valid_value), "")
            self.assertTrue(validate_value(field, invalid_value))

        custom = SettingField(
            "custom_mode",
            ("custom", "mode"),
            "自定义模式",
            "",
            "str",
            "alpha",
            "测试",
            choices=("alpha", "beta"),
        )
        self.assertEqual(validate_value(custom, "beta"), "")
        self.assertIn("alpha 或 beta", validate_value(custom, "gamma"))

    def test_settings_schema_defaults_match_runtime_constants(self):
        from context_engine import embedding_runtime
        from engine_adapters import project_config, rpgmaker_mv_fields
        from quality_review import translation_core, translation_plan

        self.assertEqual(
            ADVANCED_SETTING_FIELDS,
            FULL_COVERAGE_SETTING_FIELDS + CORE_ADVANCED_SETTING_FIELDS,
        )
        self.assertEqual(ADVANCED_SETTING_FIELD_BY_KEY["engine"].choices, project_config.SUPPORTED_ENGINES)
        self.assertEqual(ADVANCED_SETTING_FIELD_BY_KEY["mv_display_fields"].choices, rpgmaker_mv_fields.DISPLAY_FIELDS)
        self.assertEqual(
            ADVANCED_SETTING_FIELD_BY_KEY["sync_rag_embedding_backend"].choices,
            embedding_runtime.SUPPORTED_EMBEDDING_BACKENDS,
        )
        self.assertEqual(
            ADVANCED_SETTING_FIELD_BY_KEY["batch_rag_embedding_backend"].choices,
            embedding_runtime.SUPPORTED_EMBEDDING_BACKENDS,
        )
        self.assertEqual(
            ADVANCED_SETTING_FIELD_BY_KEY["sync_chunk_size"].default,
            translation_core.CANONICAL_CHUNK_MAX_ITEMS,
        )
        self.assertEqual(
            ADVANCED_SETTING_FIELD_BY_KEY["sync_max_source_chars"].default,
            translation_core.CANONICAL_CHUNK_MAX_CHARS,
        )
        self.assertEqual(
            ADVANCED_SETTING_FIELD_BY_KEY["sync_max_output_tokens"].default,
            translation_core.DEFAULT_SYNC_MAX_OUTPUT_TOKENS,
        )
        self.assertEqual(
            ADVANCED_SETTING_FIELD_BY_KEY["batch_max_output_tokens"].default,
            translation_core.DEFAULT_BATCH_MAX_OUTPUT_TOKENS,
        )
        self.assertEqual(
            ADVANCED_SETTING_FIELD_BY_KEY["batch_temperature"].default,
            translation_plan.CANONICAL_TEMPERATURE,
        )

    def test_embedding_backend_aliases_normalize_and_validate(self):
        cases = (
            ("google_genai", "gemini"),
            ("google-ai", "gemini"),
            ("openai-compatible", "openai_compatible"),
            ("litellm", "openai_compatible"),
        )
        for key, section in (
            ("sync_rag_embedding_backend", "sync"),
            ("batch_rag_embedding_backend", "batch"),
        ):
            field = ADVANCED_SETTING_FIELD_BY_KEY[key]
            for raw_alias, canonical in cases:
                with self.subTest(key=key, raw_alias=raw_alias):
                    self.assertEqual(validate_value(field, raw_alias), "")
                    self.assertEqual(normalize_for_write(field, raw_alias), canonical)
                    config = {section: {"rag": {"embedding_backend": raw_alias}}}
                    self.assertEqual(read_setting(config, field), canonical)
                    # Preserves valid alias until changed, and writes canonical value on change.
                    unchanged = apply_advanced_settings(
                        {section: {"rag": {"embedding_backend": raw_alias}}},
                        {key: canonical},
                    )
                    self.assertEqual(unchanged[section]["rag"]["embedding_backend"], raw_alias)
                    normalized_save = apply_advanced_settings({}, {key: raw_alias})
                    if canonical == field.default:
                        self.assertEqual(normalized_save, {})
                    else:
                        self.assertEqual(normalized_save[section]["rag"]["embedding_backend"], canonical)

    def test_legacy_override_forces_canonical_write_when_resetting_to_default(self):
        override_cases = (
            ("batch_max_source_chars", "batch", "target_chars", "max_source_chars", 9000, 18000),
            ("batch_retry_max_source_chars", "batch", "retry_target_chars", "retry_max_source_chars", 2000, 4000),
            ("sync_max_source_chars", "sync", "target_chars", "max_source_chars", 9000, 18000),
        )
        for key, section, legacy_key, canonical_key, legacy_val, default_val in override_cases:
            with self.subTest(key=key):
                field = ADVANCED_SETTING_FIELD_BY_KEY[key]
                legacy_cfg = {section: {legacy_key: legacy_val}}
                self.assertEqual(read_setting(legacy_cfg, field), legacy_val)

                # Saving the schema default must write the canonical override so runtime no
                # longer falls back to the stale legacy value.
                saved = apply_advanced_settings(
                    {section: {legacy_key: legacy_val}},
                    {key: default_val},
                )
                self.assertEqual(saved[section][canonical_key], default_val)
                self.assertEqual(saved[section][legacy_key], legacy_val)

                # If legacy already equals the saved value, do not write a duplicate key.
                matching_legacy_cfg = {section: {legacy_key: default_val}}
                untouched = apply_advanced_settings(matching_legacy_cfg, {key: default_val})
                self.assertNotIn(canonical_key, untouched[section])


if __name__ == "__main__":
    unittest.main()
