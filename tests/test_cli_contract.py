import io
import json
import unittest
from pathlib import Path

import cli_contract


class CliContractTests(unittest.TestCase):
    def test_success_envelope_has_stable_shape_and_json_safe_values(self):
        envelope = cli_contract.success_envelope(
            "check",
            status="safe",
            result={
                "path": Path("manifest.json"),
                "values": {2, 1},
                "frozen_values": frozenset({4, 3}),
            },
            artifacts={"manifest": Path("manifest.json")},
            warnings=["note"],
        )

        self.assertEqual(envelope["schema_version"], 1)
        self.assertTrue(envelope["ok"])
        self.assertEqual(envelope["status"], "safe")
        self.assertEqual(envelope["result"]["path"], "manifest.json")
        self.assertEqual(envelope["result"]["frozen_values"], [3, 4])
        self.assertEqual(envelope["result"]["values"], [1, 2])
        self.assertIsNone(envelope["error"])

    def test_error_envelope_exposes_programmatic_error_fields(self):
        envelope = cli_contract.error_envelope(
            "apply",
            code="COMMAND_REFUSED",
            message="unsafe",
            suggested_action="run_check_again",
            details={"reason": "stale"},
        )

        self.assertFalse(envelope["ok"])
        self.assertEqual(envelope["status"], "failed")
        self.assertEqual(envelope["error"]["code"], "COMMAND_REFUSED")
        self.assertFalse(envelope["error"]["retryable"])
        self.assertEqual(envelope["error"]["suggested_action"], "run_check_again")

    def test_write_json_envelope_writes_one_parseable_document(self):
        stream = io.StringIO()
        envelope = cli_contract.success_envelope("doctor", status="ready")

        cli_contract.write_json_envelope(envelope, stream)

        self.assertEqual(json.loads(stream.getvalue()), envelope)
        self.assertTrue(stream.getvalue().endswith("\n"))


if __name__ == "__main__":
    unittest.main()
