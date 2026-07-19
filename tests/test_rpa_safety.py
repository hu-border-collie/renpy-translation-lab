import pickle
import tempfile
import unittest
import zlib
from dataclasses import replace
from pathlib import Path

import translator_runtime as runtime
from relation_analyzer import common
from rpa_safety import (
    DEFAULT_RPA_LIMITS,
    RpaExtractionBudget,
    RpaResourceLimitError,
)


class RpaSafetyTests(unittest.TestCase):
    def _write_rpa3(self, root, raw_index, data=b"", name="archive.rpa"):
        archive = root / name
        offset = 34 + len(data)
        header = b"RPA-3.0 %016x %08x\n" % (offset, 0)
        archive.write_bytes(header + data + zlib.compress(pickle.dumps(raw_index, protocol=4)))
        return archive

    def _write_rpa2(self, root, raw_index, data=b""):
        archive = root / "archive-v2.rpa"
        offset = 25 + len(data)
        header = b"RPA-2.0 %016x\n" % offset
        archive.write_bytes(header + data + zlib.compress(pickle.dumps(raw_index, protocol=4)))
        return archive

    def test_runtime_rejects_compressed_index_over_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = self._write_rpa3(root, {"game/script.rpy": [(0, 0, b"")]})
            limits = replace(DEFAULT_RPA_LIMITS, max_compressed_index_bytes=1)

            with self.assertRaisesRegex(RpaResourceLimitError, "compressed index"):
                runtime._read_rpa_index(str(archive), limits=limits)

    def test_runtime_rejects_decompressed_index_over_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = self._write_rpa3(
                root,
                {"game/script.rpy": [(0, 0, b"A" * 1024)]},
            )
            limits = replace(DEFAULT_RPA_LIMITS, max_decompressed_index_bytes=128)

            with self.assertRaisesRegex(RpaResourceLimitError, "decompressed index"):
                runtime._read_rpa_index(str(archive), limits=limits)

    def test_analyzer_rejects_decompressed_index_over_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = self._write_rpa3(
                root,
                {"portrait.png": [(0, 0, b"A" * 1024)]},
            )
            limits = replace(DEFAULT_RPA_LIMITS, max_decompressed_index_bytes=128)

            with self.assertRaisesRegex(RpaResourceLimitError, "decompressed index"):
                common.read_rpa_index(str(archive), limits=limits)

    def test_both_parsers_reject_chunk_outside_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = self._write_rpa3(root, {"game/script.rpy": [(10_000_000, 5, b"")]})

            with self.assertRaisesRegex(RuntimeError, "outside archive"):
                runtime._read_rpa_index(str(archive))
            with self.assertRaisesRegex(RuntimeError, "outside archive"):
                common.read_rpa_index(str(archive))

    def test_runtime_rejects_negative_chunk_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = ((-1, 0, b""), (0, -1, b""))
            for index, chunk in enumerate(samples):
                archive = self._write_rpa3(
                    root,
                    {"game/script.rpy": [chunk]},
                    name=f"negative-{index}.rpa",
                )
                with self.subTest(chunk=chunk):
                    with self.assertRaisesRegex(RuntimeError, "negative offset or length"):
                        runtime._read_rpa_index(str(archive))

    def test_runtime_rejects_index_cardinality_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries_archive = self._write_rpa3(
                root,
                {"a.rpy": [], "b.rpy": []},
                name="entries.rpa",
            )
            chunks_archive = self._write_rpa3(
                root,
                {"a.rpy": [(0, 0), (0, 0)]},
                name="chunks.rpa",
            )
            total_chunks_archive = self._write_rpa3(
                root,
                {"a.rpy": [(0, 0)], "b.rpy": [(0, 0)]},
                name="total-chunks.rpa",
            )

            with self.assertRaisesRegex(RpaResourceLimitError, "index entries"):
                runtime._read_rpa_index(
                    str(entries_archive),
                    limits=replace(DEFAULT_RPA_LIMITS, max_entries=1),
                )
            with self.assertRaisesRegex(RpaResourceLimitError, "chunks for one member"):
                runtime._read_rpa_index(
                    str(chunks_archive),
                    limits=replace(DEFAULT_RPA_LIMITS, max_chunks_per_member=1),
                )
            with self.assertRaisesRegex(RpaResourceLimitError, "total chunks"):
                runtime._read_rpa_index(
                    str(total_chunks_archive),
                    limits=replace(DEFAULT_RPA_LIMITS, max_total_chunks=1),
                )

    def test_analyzer_rejects_member_output_over_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = self._write_rpa3(root, {"portrait.png": [(0, 0, b"A" * 16)]})
            limits = replace(DEFAULT_RPA_LIMITS, max_member_output_bytes=8)

            self.assertIn("portrait.png", common.read_rpa_index(str(archive)))
            with self.assertRaisesRegex(RpaResourceLimitError, "member output"):
                common.read_rpa_index(str(archive), limits=limits)

    def test_runtime_rejects_total_extraction_over_limit_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = self._write_rpa3(
                root,
                {
                    "game/a.rpy": [(0, 0, b"A" * 8)],
                    "game/b.rpy": [(0, 0, b"B" * 8)],
                },
            )
            target = root / "target"
            limits = replace(DEFAULT_RPA_LIMITS, max_total_extraction_bytes=12)

            with self.assertRaisesRegex(RpaResourceLimitError, "total extraction"):
                runtime._extract_rpa_scripts(str(archive), str(target), limits=limits)

            self.assertFalse((target / "game" / "a.rpy").exists())
            self.assertFalse((target / "game" / "b.rpy").exists())

    def test_runtime_shares_total_extraction_budget_across_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._write_rpa3(
                root,
                {"game/a.rpy": [(0, 0, b"A" * 8)]},
                name="first.rpa",
            )
            second = self._write_rpa3(
                root,
                {"game/b.rpy": [(0, 0, b"B" * 8)]},
                name="second.rpa",
            )
            target = root / "target"
            budget = RpaExtractionBudget(12)

            self.assertEqual(
                runtime._extract_rpa_scripts(
                    str(first),
                    str(target),
                    extraction_budget=budget,
                ),
                1,
            )
            with self.assertRaisesRegex(RpaResourceLimitError, "total extraction"):
                runtime._extract_rpa_scripts(
                    str(second),
                    str(target),
                    extraction_budget=budget,
                )

            self.assertEqual((target / "game" / "a.rpy").read_bytes(), b"A" * 8)
            self.assertFalse((target / "game" / "b.rpy").exists())

    def test_valid_member_remains_readable_by_both_entry_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = b"hello"
            archive = self._write_rpa3(
                root,
                {"portrait.png": [(34, len(data), b"prefix-")]},
                data=data,
            )

            runtime_index = runtime._read_rpa_index(str(archive))
            analyzer_index = common.read_rpa_index(str(archive))

            self.assertEqual(runtime_index["portrait.png"], [(34, 5, b"prefix-")])
            self.assertEqual(analyzer_index["portrait.png"], [(34, 5, b"prefix-")])
            self.assertEqual(common.read_rpa_member(str(archive), "portrait.png"), b"prefix-hello")

    def test_valid_rpa2_member_remains_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = b"hello-v2"
            archive = self._write_rpa2(
                root,
                {"portrait.png": [(25, len(data), b"prefix-")]},
                data=data,
            )

            runtime_index = runtime._read_rpa_index(str(archive))
            analyzer_index = common.read_rpa_index(str(archive))

            self.assertEqual(runtime_index["portrait.png"], [(25, 8, b"prefix-")])
            self.assertEqual(analyzer_index["portrait.png"], [(25, 8, b"prefix-")])
            self.assertEqual(
                common.read_rpa_member(str(archive), "portrait.png"),
                b"prefix-hello-v2",
            )


if __name__ == "__main__":
    unittest.main()
