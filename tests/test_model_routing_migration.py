from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import config_store
import model_config_migration as cli
from model_profile import ModelRoutingConfigError
from model_routing_migration import preview_migration
from model_routing_migration_store import (
    fingerprint, migrate_config_file, preview_config_file, rollback_config_file,
)
from model_routing_reader import read_routing_plan, read_embedding_settings, section_custom_providers


FIXTURES = Path(__file__).parent / "fixtures" / "model_routing_legacy"


def fixture(name="gemini_batch"):
    return json.loads((FIXTURES / (name + ".json")).read_text(encoding="utf-8"))


class MigrationPlanTests(unittest.TestCase):
    def test_migrated_sync_profiles_match_actual_runtime_model_selection(self):
        import translator_runtime as runtime
        for name in ("gemini_batch", "gemini_sync", "litellm_builtin", "litellm_custom"):
            config = fixture(name)
            with runtime.runtime_config_scope(runtime.default_runtime_config()):
                with redirect_stdout(io.StringIO()), patch("litellm_provider_config._installed_litellm_module", return_value=None):
                    runtime.load_sync_translation_settings(config)
                plan = read_routing_plan(preview_migration(config).config, legacy_execution="sync")
                profile = plan.profiles[plan.routes["translation"].profile_id]
                self.assertEqual(profile.model, runtime.MODELS[0])
                self.assertEqual(list(profile.models), runtime.MODELS)
                self.assertEqual(profile.adapter, runtime.SYNC_BACKEND)

    def test_gui_diagnostics_templates_match_cli_parser(self):
        from gui_qt.diagnostics_context import build_cli_commands
        commands = build_cli_commands(python_exe="python", batch_script_path="/tool/gemini_translate_batch.py",
                                      manifest_path="/run/manifest.json", manifest={})
        selected = [c for c in commands if "model_config_migration.py" in c.command]
        self.assertEqual(len(selected), 3)
        self.assertTrue(any("--stage-only" in c.command for c in selected))
        self.assertTrue(all("<CONFIG_COPY>" in c.command and "--json" in c.command for c in selected))

    def test_four_legacy_shapes_preserve_inputs_and_are_idempotent(self):
        for name in ("gemini_batch", "gemini_sync", "litellm_builtin", "litellm_custom"):
            with self.subTest(name=name):
                config = fixture(name)
                config["future"] = {"a": [1, {"b": "保留"}]}
                before = copy.deepcopy(config)
                result = preview_migration(config)
                self.assertEqual(config, before)
                self.assertEqual({k: result.config[k] for k in before}, before)
                self.assertEqual(preview_migration(config).config, result.config)
                repeat = preview_migration(result.config)
                self.assertEqual(repeat.status, "already_current")
                self.assertEqual(repeat.config, result.config)

    def test_empty_or_omitted_stage_models_keep_profiles_and_strategies(self):
        for name in ("example_defaults", "litellm_builtin", "gemini_sync", "litellm_custom"):
            config = fixture(name)
            result = preview_migration(config)
            self.assertEqual(result.status, "ready")
            self.assertEqual(result.config["model_routing"]["routes"], {
                "project_analysis": {"profile_id": "legacy-sync", "strategy": "sync"},
                "final_review": {"profile_id": "legacy-batch", "strategy": "gemini_batch"},
            })
            for mode in ("sync", "gemini_batch"):
                old = read_routing_plan(config, legacy_execution=mode)
                new = read_routing_plan(result.config, legacy_execution=mode)
                for stage in ("project_analysis", "final_review"):
                    self.assertEqual(old.routes[stage].strategy, new.routes[stage].strategy)
                    self.assertEqual(old.profiles[old.routes[stage].profile_id].model,
                                     new.profiles[new.routes[stage].profile_id].model)

    def test_embedding_names_do_not_shadow_builtin_generation_providers(self):
        for provider, model in (("openai", "gpt-4"), ("openrouter", "model-a"), ("anthropic", "claude-test")):
            with self.subTest(provider=provider):
                config = fixture("litellm_builtin")
                config["sync"].update(model=f"{provider}/{model}", models=[f"{provider}/{model}"])
                for scope in ("sync", "batch"):
                    config[scope]["rag"] = {
                        "embedding_backend": "openai_compatible", "embedding_provider": provider,
                        "embedding_endpoint": f"https://{scope}.example.test/v1",
                        "embedding_model": "embed-a", "embedding_api_key_env": "EMBED_KEY",
                    }
                result = preview_migration(config)
                self.assertEqual(result.status, "ready")
                self.assertNotIn(provider, section_custom_providers(result.config["model_routing"]))
                for mode, scope in (("sync", "sync"), ("gemini_batch", "batch")):
                    self.assertEqual(read_embedding_settings(config, execution=mode),
                                     read_embedding_settings(result.config, execution=mode))

    def test_embedding_names_do_not_replace_custom_generation_connection(self):
        config = fixture("litellm_custom")
        migrated = preview_migration(config).config
        connection = section_custom_providers(migrated["model_routing"])["acme-compatible"]
        self.assertEqual(connection.base_url, "https://models.example.test/v1")
        self.assertEqual(connection.api_key_env, "ACME_API_KEY")
        embedding = read_embedding_settings(migrated, execution="sync")
        self.assertEqual(embedding.endpoint, "https://embeddings.example.test/v1")
        self.assertEqual(embedding.api_key_env, "ACME_EMBEDDING_API_KEY")

    def test_explicit_stage_routes_keep_legacy_transports(self):
        migrated = preview_migration(fixture()).config
        for mode in ("sync", "gemini_batch"):
            plan = read_routing_plan(migrated, legacy_execution=mode)
            self.assertEqual(plan.routes["translation"].strategy.value, mode)
            self.assertEqual(plan.routes["project_analysis"].strategy.value, "sync")
            self.assertEqual(plan.routes["final_review"].strategy.value, "gemini_batch")
        default = read_routing_plan(migrated)
        self.assertEqual(default.routes["translation"].profile_id, "legacy-batch")

    def test_new_source_never_reads_retained_legacy_fields(self):
        migrated = preview_migration(fixture("litellm_custom")).config
        migrated["sync"] = {"backend": "broken", "model": "wrong", "custom_litellm_providers": 123}
        migrated["batch"] = {"model": "wrong"}
        plan = read_routing_plan(migrated, legacy_execution="sync")
        profile = plan.profiles[plan.routes["translation"].profile_id]
        self.assertEqual(profile.model, "acme-compatible/model-a")
        self.assertEqual(profile.credential_ref.env_name, "ACME_API_KEY")
        self.assertEqual(profile.base_url, "https://models.example.test/v1")

    def test_invalid_new_section_never_falls_back(self):
        for value in (None, {}, {"schema_version": 2}, []):
            config = fixture()
            config["model_routing"] = value
            with self.subTest(value=value), self.assertRaises(ModelRoutingConfigError):
                read_routing_plan(config)

    def test_legacy_reader_uses_existing_profiles(self):
        for name in ("gemini_batch", "gemini_sync", "litellm_builtin", "litellm_custom"):
            plan = read_routing_plan(fixture(name), legacy_execution="sync")
            self.assertEqual(plan.routes["translation"].profile_id, "primary")

    def test_legacy_entrypoint_references_are_validated(self):
        for bad in ([], {}, None, "missing"):
            config = preview_migration(fixture()).config
            config["model_routing"]["legacy_entrypoints"]["sync_profile_id"] = bad
            with self.subTest(bad=bad), self.assertRaises(ModelRoutingConfigError):
                read_routing_plan(config)

    def test_custom_provider_metadata_including_dormant_connection_is_kept(self):
        config = fixture("litellm_custom")
        config["sync"]["model"] = "openrouter/model-a"
        config["sync"]["custom_litellm_providers"][0]["requires_key"] = False
        section = preview_migration(config).config["model_routing"]
        provider = section["providers"]["litellm-acme-compatible"]
        self.assertEqual(provider["credential_ref"]["kind"], "none")
        self.assertEqual(provider["models_url"], "https://models.example.test/v1/models")

    def test_embedding_connections_are_frozen_separately(self):
        config = fixture()
        config["sync"]["rag"] = {
            "embedding_backend": "openai_compatible", "embedding_provider": "acme",
            "embedding_model": "embed-a", "embedding_endpoint": "https://example.test/v1",
            "embedding_api_key_env": "EMBED_KEY", "output_dimensionality": 1024,
            "future_policy": "keep",
        }
        section = preview_migration(config).config["model_routing"]
        self.assertEqual(section["providers"]["embedding-sync"]["credential_ref"],
                         {"kind": "env", "name": "EMBED_KEY"})
        sync = section["profiles"]["legacy-sync-embedding"]
        self.assertEqual(sync["params"]["backend"], "openai_compatible")
        self.assertEqual(sync["params"]["output_dimension"], 1024)
        self.assertEqual(sync["model"], "embed-a")
        self.assertEqual(section["profiles"]["legacy-batch-embedding"]["model"], "gemini-embedding-001")
        migrated = preview_migration(config).config
        migrated["sync"]["rag"] = {"embedding_model": "wrong"}
        settings = read_embedding_settings(migrated, execution="sync")
        self.assertEqual(settings.model, "embed-a")
        self.assertEqual(settings.api_key_env, "EMBED_KEY")
        self.assertEqual(settings.endpoint, "https://example.test/v1")
        self.assertNotIn("endpoint", sync["params"])

    def test_sensitive_unknown_fields_and_credential_urls_are_refused(self):
        for edit in (
            lambda c: c.update(future={"Authorization": "private-value"}),
            lambda c: c["sync"].update(api_key="private-value"),
            lambda c: c["sync"]["custom_litellm_providers"][0].update(base_url="https://user:private-value@example.test/v1"),
        ):
            config = fixture("litellm_custom")
            edit(config)
            with self.assertRaises(ValueError):
                preview_migration(config)

    def test_ambiguous_or_invalid_legacy_models_refuse_instead_of_switching(self):
        edits = [
            lambda c: c["sync"].update(models="gemini-test"),
            lambda c: c["sync"].update(models=["gemini-other"]),
            lambda c: c["sync"].update(model=[]),
            lambda c: c["batch"].update(model="openrouter/test"),
            lambda c: c.update(rotation={"model": {"enabled": True}}),
            lambda c: c["sync"].update(backend="bad"),
        ]
        for edit in edits:
            config = fixture()
            edit(config)
            with self.assertRaises(ValueError):
                preview_migration(config)

    def test_missing_sync_model_does_not_inherit_different_batch_default(self):
        with self.assertRaisesRegex(ValueError, "Ambiguous legacy defaults"):
            preview_migration({"batch": {"model": "gemini-other"}})

    def test_default_configuration_and_game_batch_override(self):
        self.assertEqual(preview_migration({}).status, "ready")
        config = {"sync": {"model": "gemini-3.1-flash-lite"}}
        migrated = preview_migration(config, game_config={"batch_model": "gemini-other"}).config
        self.assertEqual(migrated["model_routing"]["profiles"]["legacy-batch"]["model"], "gemini-other")

    def test_offline_import_does_not_load_optional_sdks_or_gui(self):
        source = "import sys; import model_routing_migration; assert not any(x in sys.modules for x in ('litellm','PySide6','google.genai'))"
        subprocess.run([sys.executable, "-c", source], check=True, capture_output=True)


class MigrationTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "translator_config.json"
        self.original = b"\xef\xbb\xbf" + json.dumps(fixture(), indent=2).replace("\n", "\r\n").encode()
        self.path.write_bytes(self.original)

    def migrate(self):
        return migrate_config_file(self.path, expected_fingerprint=fingerprint(self.original))

    def test_preview_has_no_side_effects(self):
        report = preview_config_file(self.path)
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["source_fingerprint"], fingerprint(self.original))
        self.assertEqual(list(self.root.iterdir()), [self.path])
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_file_migration_requires_explicit_batch_model(self):
        self.path.write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, "explicit batch.model"):
            preview_config_file(self.path)
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_round_trip_preserves_original_bytes_and_migration_format(self):
        report = self.migrate()
        migrated = self.path.read_bytes()
        self.assertTrue(migrated.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b"\n", migrated.replace(b"\r\n", b""))
        backup = Path(report["backup_path"])
        self.assertEqual(backup.read_bytes(), self.original)
        self.assertEqual(report["target_fingerprint"], fingerprint(migrated))
        rollback_config_file(self.path, report_path=Path(report["report_path"]))
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(backup.read_bytes(), self.original)

    def test_second_migration_preserves_bytes_and_creates_no_backup(self):
        self.migrate()
        before = self.path.read_bytes()
        files = set(self.root.iterdir())
        report = migrate_config_file(self.path, expected_fingerprint=fingerprint(before))
        self.assertEqual(report["status"], "already_current")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(set(self.root.iterdir()), files)

    def test_stale_preview_is_refused_without_backup(self):
        self.path.write_bytes(self.original + b" ")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.migrate()
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_rollback_refuses_changed_config_and_corrupt_backup(self):
        report = self.migrate()
        migrated = self.path.read_bytes()
        self.path.write_bytes(migrated + b" ")
        with self.assertRaisesRegex(ValueError, "changed"):
            rollback_config_file(self.path, report_path=Path(report["report_path"]))
        self.path.write_bytes(migrated)
        Path(report["backup_path"]).write_bytes(self.original + b" ")
        with self.assertRaisesRegex(ValueError, "backup fingerprint"):
            rollback_config_file(self.path, report_path=Path(report["report_path"]))
        self.assertEqual(self.path.read_bytes(), migrated)

    def test_failed_replace_keeps_config_and_durable_recovery_evidence(self):
        with patch("config_store.os.replace", side_effect=OSError("injected")):
            with self.assertRaises(OSError):
                self.migrate()
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(len(list(self.root.glob("*.bak"))), 1)
        self.assertEqual(len(list(self.root.glob("*.migration-*.json"))), 1)
        self.assertFalse(list(self.root.glob("*.tmp")))
        self.assertFalse(list(self.root.glob("*.write-lock")))

    def test_interruption_after_commit_leaves_usable_rollback_report(self):
        replace = config_store.os.replace
        def commit_then_interrupt(*args):
            replace(*args)
            raise KeyboardInterrupt
        with patch("config_store.os.replace", side_effect=commit_then_interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.migrate()
        report_path, = self.root.glob("*.migration-*.json")
        rollback_config_file(self.path, report_path=report_path)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_fsync_failure_does_not_replace_source(self):
        with patch("config_store.os.fsync", side_effect=OSError("injected")):
            with self.assertRaises(OSError):
                self.migrate()
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(list(self.root.iterdir()), [self.path])

    @unittest.skipUnless(os.name == "nt", "Windows DACL preservation")
    def test_windows_dacl_is_preserved_for_backup_report_and_replacement(self):
        import ctypes
        from ctypes import wintypes
        api = ctypes.WinDLL("advapi32", use_last_error=True)
        get_security = api.GetFileSecurityW
        get_security.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p,
                                wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        get_security.restype = wintypes.BOOL
        get_dacl = api.GetSecurityDescriptorDacl
        get_dacl.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.BOOL),
                            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.BOOL)]
        get_dacl.restype = wintypes.BOOL
        def dacl(path):
            size = wintypes.DWORD()
            get_security(str(path), 4, None, 0, ctypes.byref(size))
            buffer = ctypes.create_string_buffer(size.value)
            self.assertTrue(get_security(str(path), 4, buffer, size, ctypes.byref(size)))
            present, defaulted = wintypes.BOOL(), wintypes.BOOL()
            pointer = ctypes.c_void_p()
            self.assertTrue(get_dacl(buffer, ctypes.byref(present), ctypes.byref(pointer), ctypes.byref(defaulted)))
            self.assertTrue(present.value and pointer.value)
            # ACL header: revision byte, reserved byte, WORD AclSize.
            length = ctypes.c_ushort.from_address(pointer.value + 2).value
            return ctypes.string_at(pointer, length)
        original_acl = dacl(self.path)
        report = self.migrate()
        for path in (self.path, Path(report["backup_path"]), Path(report["report_path"])):
            self.assertEqual(dacl(path), original_acl)

    def test_backup_and_report_failures_never_replace_config(self):
        from model_routing_migration_store import write_private_artifact
        for fail_at in (1, 2):
            count = 0
            def injected(*args, **kwargs):
                nonlocal count
                count += 1
                if count == fail_at:
                    raise OSError("injected")
                return write_private_artifact(*args, **kwargs)
            with patch("model_routing_migration_store.write_private_artifact", side_effect=injected):
                with self.assertRaises(OSError):
                    self.migrate()
            self.assertEqual(self.path.read_bytes(), self.original)

    def test_source_change_during_preparation_is_not_overwritten(self):
        from model_routing_migration_store import replace_config_bytes
        def concurrent(*args, **kwargs):
            self.path.write_bytes(self.original + b" ")
            return replace_config_bytes(*args, **kwargs)
        with patch("model_routing_migration_store.replace_config_bytes", side_effect=concurrent):
            with self.assertRaisesRegex(ValueError, "changed"):
                self.migrate()
        self.assertEqual(self.path.read_bytes(), self.original + b" ")

    def test_gui_and_migration_share_exclusive_write_lock(self):
        with config_store.config_write_lock(self.path):
            with self.assertRaisesRegex(config_store.ConfigWriteLockError, "锁文件"):
                self.migrate()
            with self.assertRaisesRegex(config_store.ConfigWriteLockError, "锁文件"):
                config_store.write_json_object(self.path, {})
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_abandoned_lock_recovers_for_gui_and_migration(self):
        lock = self.path.with_name(self.path.name + ".write-lock")
        for write in (lambda: config_store.write_json_object(self.path, fixture()), self.migrate):
            self.path.write_bytes(self.original)
            lock.write_text("12345", encoding="ascii")  # Pre-token lock format.
            old = time.time() - 301
            os.utime(lock, (old, old))
            write()
            self.assertFalse(lock.exists())

    def test_lock_cleanup_keeps_replacement_owner_and_tolerates_missing_lock(self):
        lock = self.path.with_name(self.path.name + ".write-lock")
        replacement = '{"pid": 12345, "token": "different-owner"}'
        with config_store.config_write_lock(self.path):
            lock.write_text(replacement, encoding="utf-8")
        self.assertEqual(lock.read_text(encoding="utf-8"), replacement)
        lock.unlink()
        with config_store.config_write_lock(self.path):
            lock.unlink()
        self.assertFalse(lock.exists())

    def test_cli_lock_timeout_is_classified_without_exposing_path(self):
        with config_store.config_write_lock(self.path), redirect_stdout(output := io.StringIO()):
            code = cli.main(["migrate", "--config", str(self.path), "--stage-only",
                             "--expected-fingerprint", fingerprint(self.original), "--json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["reason"], "config_locked")
        self.assertNotIn(str(self.path), output.getvalue())
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_malformed_json_is_not_rewritten(self):
        for raw in (b'{"sync":{},"sync":{}}', b'[]', b'{"a":NaN}', b'{broken', b''):
            self.path.write_bytes(raw)
            with self.assertRaises(ValueError):
                migrate_config_file(self.path, expected_fingerprint=fingerprint(raw))
            self.assertEqual(self.path.read_bytes(), raw)
            self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_symlink_is_refused(self):
        link = self.root / "link.json"
        try:
            link.symlink_to(self.path)
        except OSError:
            self.skipTest("symlink creation unavailable")
        with self.assertRaisesRegex(ValueError, "symlink"):
            preview_config_file(link)

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    def test_permissions_are_not_broadened(self):
        self.path.chmod(0o600)
        report = self.migrate()
        for p in (self.path, Path(report["backup_path"]), Path(report["report_path"])):
            self.assertEqual(p.stat().st_mode & 0o777, 0o600)

    def test_permission_copy_failure_never_writes_payload_or_replaces_source(self):
        with patch("config_store._copy_access_mode", side_effect=OSError("injected")):
            with self.assertRaises(OSError):
                self.migrate()
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_report_cannot_escape_backup_directory(self):
        report = self.migrate()
        report_path = Path(report["report_path"])
        report["backup_name"] = "../elsewhere.bak"
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "Invalid migration report"):
            rollback_config_file(self.path, report_path=report_path)

    def test_cli_preview_migrate_rollback_json(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = cli.main(["preview", "--config", str(self.path), "--json"])
        self.assertEqual(code, 0)
        preview = json.loads(output.getvalue())
        output = io.StringIO()
        with redirect_stdout(output):
            code = cli.main(["migrate", "--config", str(self.path), "--expected-fingerprint",
                             preview["source_fingerprint"], "--stage-only", "--json"])
        self.assertEqual(code, 0)
        report = json.loads(output.getvalue())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["rollback", "--config", str(self.path), "--report", report["report_path"], "--json"]), 0)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_cli_refusal_does_not_echo_provider_credentials(self):
        config = fixture("litellm_custom")
        config["sync"]["custom_litellm_providers"][0]["base_url"] = "https://user:private-value@example.test"
        self.path.write_text(json.dumps(config))
        output = io.StringIO()
        with redirect_stdout(output):
            code = cli.main(["preview", "--config", str(self.path), "--json"])
        self.assertEqual(code, 2)
        self.assertNotIn("private-value", output.getvalue())
        self.assertEqual(json.loads(output.getvalue())["status"], "refused")

    def test_cli_stale_preview_has_actionable_safe_reason(self):
        with redirect_stdout(output := io.StringIO()):
            code = cli.main(["migrate", "--config", str(self.path), "--stage-only",
                             "--expected-fingerprint", "wrong", "--json"])
        self.assertEqual(code, 2)
        report = json.loads(output.getvalue())
        self.assertEqual(report["reason"], "stale_preview")
        self.assertIn("重新预览", report["next_action"])


if __name__ == "__main__":
    unittest.main()
