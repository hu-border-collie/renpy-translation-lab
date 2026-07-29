# -*- coding: utf-8 -*-
"""P0 characterization tests for the current Ren'Py parsing boundary.

These tests intentionally exercise existing helpers directly. P0 does not
introduce an adapter implementation; P1 must preserve these outputs while it
makes parse failures and unsupported candidates observable.
"""

import os
import tempfile
import unittest

import gemini_translate_batch as batch
import translation_core
import translator_runtime as runtime


class TestEngineAdapterP0Characterization(unittest.TestCase):
    def test_scan_preserves_source_identity_speaker_and_block_occurrence(self):
        lines = [
            'define e = Character(_("Eileen"))\n',
            "translate schinese chapter:\n",
            '    # e "Hello {player}!"\n',
            '    e "你好，[player]！"\n',
            '    # "Choose wisely"\n',
            '    "Choose wisely"\n',
            "translate schinese strings:\n",
            '    old "Start game"\n',
            '    new "开始游戏"\n',
            "translate schinese chapter:\n",
            '    # e "Hello {player}!"\n',
            '    e "Hello {player}!"\n',
        ]

        mapping = runtime.scan_all_translation_units(lines, "script.rpy")
        first_chapter_id = translation_core.build_identity_v2(
            "script.rpy", "chapter", 1, "Hello {player}!"
        )
        narration_id = translation_core.build_identity_v2(
            "script.rpy", "chapter", 2, "Choose wisely"
        )
        string_id = translation_core.build_identity_v2(
            "script.rpy", "strings", 1, "Start game"
        )
        repeated_chapter_id = translation_core.build_identity_v2(
            "script.rpy",
            "chapter",
            1,
            "Hello {player}!",
            block_occurrence=2,
        )

        self.assertEqual(
            set(mapping),
            {
                first_chapter_id,
                narration_id,
                string_id,
                repeated_chapter_id,
            },
        )
        self.assertEqual(mapping[first_chapter_id][0::3], (3, "你好，[player]！"))
        self.assertEqual(mapping[narration_id][0::3], (5, "Choose wisely"))
        self.assertEqual(mapping[string_id][0::3], (8, "开始游戏"))
        self.assertEqual(
            mapping[repeated_chapter_id][0::3],
            (11, "Hello {player}!"),
        )
        self.assertFalse(any("Eileen" in item_id for item_id in mapping))

        tasks, progress = runtime.collect_tasks_with_progress(lines)
        self.assertEqual(progress, {"translated_count": 2})
        self.assertEqual(
            [
                (
                    task["text"],
                    task["block_name"],
                    task["block_index"],
                    task["block_occurrence"],
                    task.get("speaker_id", ""),
                    task.get("speaker_name", ""),
                )
                for task in tasks
            ],
            [
                ("Choose wisely", "chapter", 2, 1, "", ""),
                (
                    "Hello {player}!",
                    "chapter",
                    1,
                    2,
                    "e",
                    "Eileen",
                ),
            ],
        )

    def test_identity_is_stable_across_line_drift_but_not_duplicate_blocks(self):
        original = [
            "translate schinese chapter:\n",
            '    # "A stable line"\n',
            '    "A stable line"\n',
            "translate schinese chapter:\n",
            '    # "A stable line"\n',
            '    "A stable line"\n',
        ]
        drifted = [
            "\n",
            "init python:\n",
            "    pass\n",
            "\n",
            *original,
        ]

        original_map = runtime.scan_all_translation_units(original, "script.rpy")
        drifted_map = runtime.scan_all_translation_units(drifted, "script.rpy")

        self.assertEqual(set(original_map), set(drifted_map))
        self.assertEqual(len(original_map), 2)
        first_id = translation_core.build_identity_v2(
            "script.rpy", "chapter", 1, "A stable line"
        )
        second_id = translation_core.build_identity_v2(
            "script.rpy",
            "chapter",
            1,
            "A stable line",
            block_occurrence=2,
        )
        self.assertEqual(set(original_map), {first_id, second_id})
        self.assertEqual(original_map[first_id][0], 2)
        self.assertEqual(drifted_map[first_id][0], 6)
        self.assertEqual(original_map[second_id][0], 5)
        self.assertEqual(drifted_map[second_id][0], 9)

    def test_v2_relocation_updates_live_span_and_reports_missing_identity(self):
        file_rel_path = "chapter.rpy"
        live_lines = [
            "\n",
            "init python:\n",
            "    pass\n",
            "translate schinese chapter:\n",
            '    # e "Hello there"\n',
            '    e "Hello there"\n',
        ]
        item_id = translation_core.build_identity_v2(
            file_rel_path, "chapter", 1, "Hello there"
        )
        item = {
            "id": item_id,
            "text": "Hello there",
            "line": 2,
            "line_number": 3,
            "start": 4,
            "end": 17,
        }
        unresolved = {
            "id": "chapter.rpy:missing:1:deadbeef",
            "text": "Missing line",
            "line": 0,
            "line_number": 1,
            "start": 0,
            "end": 0,
        }

        with tempfile.TemporaryDirectory() as tmp:
            file_path = os.path.join(tmp, file_rel_path)
            with open(file_path, "w", encoding="utf-8") as handle:
                handle.writelines(live_lines)
            manifest = {
                "version": 2,
                "manifest_version": 2,
                "files": {file_rel_path: {"path": file_path}},
            }
            chunk = {
                "file_rel_path": file_rel_path,
                "items": [item, unresolved],
            }

            missing = batch.relocate_v2_chunk_items(
                manifest,
                chunk,
                {},
                translation_core.MODE_TRANSLATION,
            )

        token_start = live_lines[5].index('"')
        token_end = live_lines[5].rindex('"') + 1
        self.assertEqual(
            (item["line"], item["line_number"], item["start"], item["end"]),
            (5, 6, token_start, token_end),
        )
        self.assertEqual(missing, [unresolved])
        self.assertEqual(
            (
                unresolved["line"],
                unresolved["line_number"],
                unresolved["start"],
                unresolved["end"],
            ),
            (0, 1, 0, 0),
        )

    def test_render_applies_multiple_spans_with_current_quote_escaping(self):
        source = '    e "First line" + \'Second line\'\n'
        first_start = source.index('"First line"')
        first_end = first_start + len('"First line"')
        second_start = source.index("'Second line'")
        second_end = second_start + len("'Second line'")
        lines = [source]
        replacements = {
            0: [
                (first_start, first_end, '第一 "行"', "", '"'),
                (second_start, second_end, "第二 '行'", "", "'"),
            ]
        }

        rendered = runtime.render_replacement_lines(lines, replacements)

        self.assertEqual(lines, [source])
        self.assertEqual(
            rendered,
            ['    e "第一 \\"行\\"" + \'第二 \\\'行\\\'\'\n'],
        )


if __name__ == "__main__":
    unittest.main()
