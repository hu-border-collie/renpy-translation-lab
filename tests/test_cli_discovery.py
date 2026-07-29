import contextlib
import io
import json
import unittest
from unittest import mock

import cli_discovery
import gemini_translate_batch as batch


class CliDiscoveryTests(unittest.TestCase):
    def test_capabilities_lists_live_commands_without_loading_project_config(self):
        stdout = io.StringIO()

        with (
            mock.patch.object(batch.legacy, "load_config") as load_config,
            mock.patch.object(batch.legacy, "load_translator_settings") as load_settings,
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = batch.main(["capabilities"])

        payload = json.loads(stdout.getvalue())
        names = {item["name"] for item in payload["commands"]}
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["type"], "capabilities")
        self.assertEqual(payload["command_count"], len(names))
        self.assertIn("capabilities", names)
        self.assertIn("schema", names)
        self.assertNotIn("commands", names)
        capabilities = {
            item["name"]: item for item in payload["commands"]
        }["capabilities"]
        self.assertTrue(capabilities["machine_output"])
        self.assertTrue(capabilities["supports_json"])
        load_config.assert_not_called()
        load_settings.assert_not_called()

    def test_capabilities_derive_agent_support_from_current_parser(self):
        parser = batch.build_arg_parser()

        payload = cli_discovery.capabilities(
            parser,
            cli_version="test",
            machine_output_commands=batch.MACHINE_OUTPUT_COMMANDS,
            explicit_target_commands=batch.EXPLICIT_TARGET_COMMANDS,
            result_schema_version=batch.cli_contract.CLI_SCHEMA_VERSION,
        )

        commands = {item["name"]: item for item in payload["commands"]}
        self.assertTrue(commands["status"]["supports_json"])
        self.assertTrue(commands["status"]["supports_strict_exit_codes"])
        self.assertTrue(commands["status"]["supports_non_interactive"])
        self.assertTrue(commands["status"]["requires_explicit_target_in_agent_mode"])
        self.assertFalse(commands["build"]["requires_explicit_target_in_agent_mode"])
        self.assertFalse(commands["split"]["supports_json"])

    def test_schema_exports_live_argument_contract(self):
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            exit_code = batch.main(["schema", "status"])

        payload = json.loads(stdout.getvalue())
        arguments = {item["name"]: item for item in payload["arguments"]}
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["type"], "command_schema")
        self.assertEqual(payload["command"], "status")
        self.assertEqual(arguments["target"]["kind"], "positional")
        self.assertEqual(arguments["target"]["nargs"], "?")
        self.assertEqual(arguments["output"]["choices"], ["text", "json"])
        self.assertEqual(arguments["non_interactive"]["value_type"], "boolean")

    def test_schema_preserves_required_and_repeatable_arguments(self):
        parser = batch.build_arg_parser()

        ingest = cli_discovery.command_schema(
            parser,
            "project-analysis-ingest-keywords",
        )
        structure = cli_discovery.command_schema(
            parser,
            "project-analysis-build-structure",
        )

        ingest_args = {item["name"]: item for item in ingest["arguments"]}
        structure_args = {item["name"]: item for item in structure["arguments"]}
        self.assertTrue(ingest_args["summary_jsonl"]["required"])
        self.assertTrue(structure_args["script_root"]["repeatable"])


if __name__ == "__main__":
    unittest.main()
