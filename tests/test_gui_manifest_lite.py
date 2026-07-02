import json
import tempfile
import unittest
from pathlib import Path

from gui_qt.manifest_lite import (
    read_manifest_index_fields,
    read_manifest_lite,
)


class GuiManifestLiteTests(unittest.TestCase):
    def test_read_manifest_lite_skips_heavy_chunks_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            payload = {
                "mode": "translation",
                "base_dir": "C:/game/work",
                "job_name": "batches/demo",
                "job_state": "JOB_STATE_SUCCEEDED",
                "summary": {"item_count": 12, "chunk_count": 3, "file_count": 2},
                "files": {"a.rpy": {"path": "C:/game/work/a.rpy"}},
                "chunks": [{"key": f"chunk-{index}", "payload": "x" * 96} for index in range(4000)],
                "last_check_summary": {"safety_level": "safe", "pending_lines": 12},
                "applied_at": "2026-06-30T12:00:00",
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            lite = read_manifest_lite(path)

            self.assertEqual(lite.get("mode"), "translation")
            self.assertEqual(lite.get("job_state"), "JOB_STATE_SUCCEEDED")
            self.assertEqual(lite.get("summary", {}).get("item_count"), 12)
            self.assertEqual(lite.get("last_check_summary", {}).get("safety_level"), "safe")
            self.assertEqual(lite.get("applied_at"), "2026-06-30T12:00:00")
            self.assertNotIn("chunks", lite)

    def test_read_manifest_index_fields_reads_only_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "mode": "keyword_extraction",
                        "base_dir": "C:/game/work",
                        "chunks": [{"key": "ignored"}],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            fields = read_manifest_index_fields(path)

            self.assertEqual(fields.get("mode"), "keyword_extraction")
            self.assertEqual(fields.get("base_dir"), "C:/game/work")
            self.assertNotIn("chunks", fields)


if __name__ == "__main__":
    unittest.main()