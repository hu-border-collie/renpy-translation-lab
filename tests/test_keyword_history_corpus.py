import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gemini_translate_batch as batch


class KeywordHistoryCorpusScanTests(unittest.TestCase):
    def test_final_digest_is_collected_after_corpus_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "script.rpy"
            source_path.write_text('translate schinese start:\n    old "Cart"\n    new "车"\n', encoding="utf-8")
            manifest = {
                "base_dir": str(root),
                "files": {"script.rpy": {"path": "script.rpy"}},
            }
            events = []

            def collect_digests(_file_path_map):
                events.append("digest")
                return {"script.rpy": "before" if events.count("digest") == 1 else "after"}

            def collect_jobs(**_kwargs):
                events.append("jobs")
                return [{"file_rel_path": "script.rpy", "items": []}]

            def build_corpus(_jobs):
                events.append("build")
                return ([{"source": "Cart", "current_translation": "车"}], [])

            with (
                mock.patch.object(
                    batch.revision_corpus,
                    "collect_file_digests",
                    side_effect=collect_digests,
                ),
                mock.patch.object(
                    batch,
                    "collect_revision_file_jobs",
                    side_effect=collect_jobs,
                ),
                mock.patch.object(
                    batch.revision_corpus,
                    "build_corpus_items",
                    side_effect=build_corpus,
                ),
            ):
                result = batch.collect_keyword_history_corpus(manifest)

        self.assertEqual(events, ["digest", "jobs", "build", "digest"])
        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["source_changed_during_scan"])
        self.assertEqual(result["items"], [{"source": "Cart", "current_translation": "车"}])


if __name__ == "__main__":
    unittest.main()
