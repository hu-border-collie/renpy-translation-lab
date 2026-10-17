"""Sync stage dispatch must validate the selected provider's credentials."""

import contextlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import gemini_translate_batch as batch
import model_profile
import sync_request
import translator_runtime as runtime


class StageReached(Exception):
    pass


class SyncStageCredentialTests(unittest.TestCase):
    def test_embedding_bootstrap_reaches_non_gemini_backend_without_gemini_key(self):
        parser = batch.build_arg_parser()
        for command, entrypoint in (
            ('bootstrap-rag', 'bootstrap_rag_store'),
            ('bootstrap-source-index', 'bootstrap_source_index'),
        ):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as tmp:
                keys = Path(tmp) / 'api_keys.json'
                keys.write_text('{"api_keys": []}', encoding='utf-8')

                def reached(*_args, **_kwargs):
                    self.assertEqual(runtime.API_KEYS, [])
                    raise StageReached

                with (
                    runtime.runtime_config_scope(runtime.default_runtime_config()),
                    mock.patch.dict(os.environ, {}, clear=True),
                    mock.patch.object(runtime, 'CONFIG_FILE', str(keys)),
                    mock.patch.object(batch, 'initialize_batch_logging'),
                    mock.patch.object(runtime, 'load_translator_settings'),
                    mock.patch.object(runtime, 'load_glossary'),
                    mock.patch.object(batch, 'load_batch_settings'),
                    mock.patch.object(batch, 'print_banner'),
                    mock.patch.object(batch, 'current_batch_embedding_settings', return_value=SimpleNamespace(backend='openai_compatible')),
                    mock.patch.object(batch, entrypoint, side_effect=reached),
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaises(StageReached),
                ):
                    batch.dispatch_command(parser, parser.parse_args([command, '--skip-prepare']))

    def test_gemini_embedding_bootstrap_requires_key_before_entering_store(self):
        parser = batch.build_arg_parser()
        for command, entrypoint in (
            ('bootstrap-rag', 'bootstrap_rag_store'),
            ('bootstrap-source-index', 'bootstrap_source_index'),
        ):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as tmp:
                keys = Path(tmp) / 'api_keys.json'
                keys.write_text('{"api_keys": []}', encoding='utf-8')
                with (
                    runtime.runtime_config_scope(runtime.default_runtime_config()),
                    mock.patch.dict(os.environ, {}, clear=True),
                    mock.patch.object(runtime, 'CONFIG_FILE', str(keys)),
                    mock.patch.object(batch, 'initialize_batch_logging'),
                    mock.patch.object(runtime, 'load_translator_settings'),
                    mock.patch.object(runtime, 'load_glossary'),
                    mock.patch.object(batch, 'load_batch_settings'),
                    mock.patch.object(batch, 'current_batch_embedding_settings', return_value=SimpleNamespace(backend='gemini')),
                    mock.patch.object(batch, entrypoint) as bootstrap,
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    batch.dispatch_command(parser, parser.parse_args([command, '--skip-prepare']))
                bootstrap.assert_not_called()

    def test_sync_stages_reach_provider_dispatch_without_gemini_credentials(self):
        cases = (
            (["sync-keywords"], "sync_keyword_candidates"),
            (["sync-revisions"], "sync_revisions"),
            (["project-analysis-generate"], "freeze_runtime_routing_plan"),
            (["final-review-run-sync", "campaign.json"], "run_final_review_run_sync"),
        )
        parser = batch.build_arg_parser()
        for argv, entrypoint in cases:
            with self.subTest(command=argv[0]), tempfile.TemporaryDirectory() as tmp:
                keys = Path(tmp) / "api_keys.json"
                keys.write_text('{"api_keys": []}', encoding="utf-8")

                def reached(*_args, **_kwargs):
                    self.assertEqual(runtime.API_KEYS, [])
                    raise StageReached

                with (
                    runtime.runtime_config_scope(runtime.default_runtime_config()),
                    mock.patch.dict(os.environ, {}, clear=True),
                    mock.patch.object(runtime, "CONFIG_FILE", str(keys)),
                    mock.patch.object(batch, "initialize_batch_logging"),
                    mock.patch.object(runtime, "load_translator_settings"),
                    mock.patch.object(runtime, "load_glossary"),
                    mock.patch.object(batch, "load_batch_settings"),
                    mock.patch.object(batch, "print_banner"),
                    mock.patch.object(batch, entrypoint, side_effect=reached),
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaises(StageReached),
                ):
                    batch.dispatch_command(parser, parser.parse_args(argv))

    def test_gemini_sync_request_still_requires_its_own_credentials(self):
        plan = model_profile.resolve_routing_plan({
            "sync": {"backend": "gemini", "model": "gemini-3.1-flash-lite"},
        })
        with (
            runtime.runtime_config_scope(runtime.default_runtime_config()),
            mock.patch.object(model_profile, "build_sync_backend") as backend,
            self.assertRaises(SystemExit),
        ):
            sync_request.run_sync_request({}, plan.routes["translation"], plan)
        backend.assert_not_called()

    def test_remote_batch_still_rejects_missing_gemini_credentials(self):
        parser = batch.build_arg_parser()
        with tempfile.TemporaryDirectory() as tmp:
            keys = Path(tmp) / "api_keys.json"
            keys.write_text('{"api_keys": []}', encoding="utf-8")
            with (
                runtime.runtime_config_scope(runtime.default_runtime_config()),
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(runtime, "CONFIG_FILE", str(keys)),
                mock.patch.object(batch, "initialize_batch_logging"),
                mock.patch.object(batch, "show_status") as status,
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                batch.dispatch_command(parser, parser.parse_args(["status", "manifest.json"]))
            status.assert_not_called()


if __name__ == "__main__":
    unittest.main()
