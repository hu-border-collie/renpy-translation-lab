"""Regressions for Ren'Py say lines that carry a ``with`` clause (#518).

The generated TL comment keeps the say statement's trailing clause
(``# m "Noooooo!" with vpunch``). The marker must still pair with the
translated line, the three coverage views must agree on the span, and no
adapter occurrence may reach the project snapshot with an empty ``unit_id``.
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import engine_adapters.versioning as versioning
import gemini_translate_batch as batch
import review_index
import revision_corpus
import translation_core
import translator_runtime as runtime
from engine_adapters import GameVersion, build_project_snapshot
from engine_adapters.contracts import (
    Occurrence,
    OpaqueLocator,
    ProjectDiscoveryRequest,
    ValidatedTranslation,
)
from engine_adapters.renpy import RenPyAdapter, build_translation_snapshot
from engine_adapters.reuse import TranslationInput, build_translation_records
from engine_adapters.writeback import WritebackPlanError, render_writeback_plan

SCRIPT = (
    "translate schinese start_abc123:\n"
    "\n"
    '    # m "Noooooo!" with vpunch\n'
    '    m "不——！" with vpunch\n'
    "\n"
    "translate schinese start_def456:\n"
    "\n"
    '    # m "Thanks!"\n'
    '    m "谢谢！"\n'
    "\n"
    "translate schinese start_ghi789:\n"
    "\n"
    '    # m "Ouch!" with hpunch\n'
    '    m "哎哟！" with hpunch\n'
    "\n"
    "translate schinese start_pending:\n"
    "\n"
    '    # m "Watch out!" with vpunch\n'
    '    m "" with vpunch\n'
    "\n"
    "translate schinese start_unmarked:\n"
    "\n"
    '    m "无标记对照"\n'
)

WITH_CLAUSE_LINES = (3, 8, 13)  # 0-based line indexes of the translated pairs
PENDING_LINE = 18
UNMARKED_LINE = 22


class RenPyWithClauseIdentityTests(unittest.TestCase):
    def make_project(self, text: str):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        tl_dir = root / "game" / "tl" / "schinese"
        tl_dir.mkdir(parents=True)
        script = tl_dir / "script.rpy"
        script.write_text(text, encoding="utf-8")
        return root, tl_dir, script

    def snapshot_for(self, text: str):
        root, tl_dir, _script = self.make_project(text)
        return build_translation_snapshot(
            RenPyAdapter(legacy_module=runtime),
            ProjectDiscoveryRequest(
                project_root=str(root),
                localization_root=str(tl_dir),
                target_language="schinese",
            ),
        )

    def test_marker_regex_keeps_say_clauses_out_of_source(self):
        self.assertIs(batch.REPAIR_LINE_COMMENT_RE, runtime.TL_COMMENT_SOURCE_RE)
        for regex in (runtime.TL_COMMENT_SOURCE_RE, batch.REPAIR_LINE_COMMENT_RE):
            with self.subTest(regex=regex.pattern):
                for line, expected in (
                    ('    # m "Noooooo!" with vpunch', "Noooooo!"),
                    ('    # m "Ouch!" with hpunch', "Ouch!"),
                    ('    # m "Wait" nointeract', "Wait"),
                    ('    # m "Got it" id confirm_1', "Got it"),
                    ('    # m "Got it" with vpunch id confirm_1', "Got it"),
                    ('    # m "Fade" with Dissolve(0.5)', "Fade"),
                    ('    # m "Fade" with MoveTransition(0.4, enter=Dissolve(0.2))', "Fade"),
                    ('    # m "Fade" with store.my_trans', "Fade"),
                ):
                    match = regex.match(line)
                    self.assertIsNotNone(match)
                    self.assertEqual(match.group("text"), expected)
                    self.assertTrue(match.group("suffix"))
                self.assertEqual(
                    regex.match('    # m "Noooooo!" with vpunch').group("prefix").strip(),
                    "m",
                )
                self.assertEqual(
                    regex.match('    # m "Noooooo!" with vpunch').group("suffix").strip(),
                    "with vpunch",
                )

                marker_only = regex.match('    # m "Thanks!"')
                self.assertIsNotNone(marker_only)
                self.assertEqual(marker_only.group("text"), "Thanks!")
                self.assertIsNone(marker_only.group("suffix"))

                # Escaped quotes inside the source still close on the last quote.
                escaped = regex.match('    # m "say \\"hi\\"" with vpunch')
                self.assertIsNotNone(escaped)
                self.assertEqual(escaped.group("text"), 'say \\"hi\\"')

                # Unrelated trailing text is still not a source marker.
                self.assertIsNone(regex.match('    # m "text" without a clause'))
                self.assertIsNone(regex.match('    # TODO: fix "this" with better wording'))
                self.assertIsNone(regex.match('    # TODO: fix "this" with better'))
                marker = regex.match('    # m "old" with vpunch')
                self.assertFalse(
                    runtime.tl_source_marker_matches_target(
                        marker, '    m "新译" with hpunch'
                    )
                )
                # The recognition regex remains greedy; the shared token
                # parser resolves the two independent source/target spans.
                double = regex.match('    # "Who" "text"')
                self.assertEqual(double.group("text"), 'Who" "text')
                pairs = runtime.paired_tl_comment_literals(
                    double, '    "谁" "正文"'
                )
                self.assertEqual(
                    [(pair["source"], pair["translation"]) for pair in pairs],
                    [("Who", "谁"), ("text", "正文")],
                )

    def test_clause_match_accepts_crlf_lines(self):
        marker = runtime.TL_COMMENT_SOURCE_RE.match(
            '    # m "Noooooo!" with vpunch\r'
        )
        self.assertIsNotNone(marker)
        self.assertTrue(marker.group("suffix").endswith("\r"))
        self.assertTrue(
            runtime.tl_source_marker_matches_target(
                marker, '    m "不——！" with vpunch\r\n'
            )
        )

    def test_other_say_clauses_pair_like_with(self):
        for clause in (
            "nointeract", "id confirm_1",
            "with MoveTransition(0.4, enter=Dissolve(0.2))",
            "with store.my_trans",
        ):
            with self.subTest(clause=clause):
                text = (
                    "translate schinese clause_1:\n"
                    "\n"
                    f'    # m "Got it!" {clause}\n'
                    f'    m "明白了！" {clause}\n'
                )
                lines = text.splitlines(keepends=True)
                self.assertEqual(len(runtime.collect_translation_entries_from_lines(lines)), 1)
                self.assertEqual(
                    len(batch.collect_translation_entries_from_lines(lines, "script.rpy")), 1
                )
                snapshot = self.snapshot_for(text)
                self.assertEqual(len(snapshot.occurrences), 1)
                occurrence = snapshot.occurrences[0]
                self.assertIn(":clause_1:1:", occurrence.unit.id)
                self.assertEqual(occurrence.unit.source_text, "Got it!")
                self.assertEqual(occurrence.unit.current_translation, "明白了！")

    def test_free_text_comment_is_not_paired_as_source_marker(self):
        for comment in (
            '# TODO: fix "this" with better wording',
            '# use "dissolve" with care',
        ):
            with self.subTest(comment=comment):
                text = (
                    "translate schinese note_1:\n"
                    "\n"
                    f"    {comment}\n"
                    '    m "已译"\n'
                )
                lines = text.splitlines(keepends=True)
                self.assertEqual(runtime.collect_translation_entries_from_lines(lines), [])
                self.assertEqual(batch.collect_translation_entries_from_lines(lines, "script.rpy"), [])
                snapshot = self.snapshot_for(text)
                self.assertEqual(len(snapshot.occurrences), 1)
                candidate = next(
                    candidate
                    for candidate in snapshot.inventory.candidates
                    if candidate.unit is not None and candidate.unit.display_line_number == 4
                )
                self.assertTrue(candidate.evidence.get("source_marker_missing"))
                self.assertEqual(candidate.evidence.get("identity_source"), "locator_fallback")

    def test_clause_tokens_ignore_spacing_and_target_trailing_comment(self):
        text = (
            "translate schinese clause_spacing:\n"
            "\n"
            '    # m "Wait!" with MoveTransition(0.4,enter=Dissolve(0.2))\n'
            '    m "等等！" with MoveTransition(0.4, enter=Dissolve(0.2)) # note\n'
        )
        lines = text.splitlines(keepends=True)
        self.assertEqual(len(runtime.collect_translation_entries_from_lines(lines)), 1)
        self.assertEqual(len(batch.collect_translation_entries_from_lines(lines, "script.rpy")), 1)
        snapshot = self.snapshot_for(text)
        self.assertEqual(len(snapshot.occurrences), 1)
        self.assertEqual(snapshot.occurrences[0].unit.source_text, "Wait!")

    def test_changed_clause_keeps_missing_marker_evidence(self):
        text = (
            "translate schinese changed_clause:\n"
            "\n"
            '    # m "Wait!" with vpunch\n'
            '    m "等等！" with hpunch\n'
        )
        lines = text.splitlines(keepends=True)
        self.assertEqual(runtime.collect_translation_entries_from_lines(lines), [])
        self.assertEqual(batch.collect_translation_entries_from_lines(lines, "script.rpy"), [])
        snapshot = self.snapshot_for(text)
        candidate = next(
            candidate for candidate in snapshot.inventory.candidates
            if candidate.unit is not None and candidate.unit.display_line_number == 4
        )
        self.assertTrue(candidate.evidence.get("source_marker_missing"))
        self.assertEqual(candidate.evidence.get("identity_source"), "locator_fallback")

    def test_prose_comment_between_marker_and_target_does_not_hide_marker(self):
        text = (
            "translate schinese note_between:\n"
            "\n"
            '    # m "Noooooo!" with vpunch\n'
            '    # see "note" with care\n'
            '    m "不——！" with vpunch\n'
        )
        lines = text.splitlines(keepends=True)
        self.assertEqual(runtime.find_source_text_for_translation_line(lines, 4), "Noooooo!")
        snapshot = self.snapshot_for(text)
        self.assertEqual(len(snapshot.occurrences), 1)
        self.assertEqual(snapshot.occurrences[0].unit.source_text, "Noooooo!")

    def test_with_clause_lines_keep_identity_across_views(self):
        root, tl_dir, script = self.make_project(SCRIPT)
        lines = SCRIPT.splitlines(keepends=True)
        rel_path = "script.rpy"

        identity_mapping = runtime.scan_all_translation_units(lines, rel_path)
        identity_lines = {value[0] for value in identity_mapping.values()}
        self.assertEqual(
            identity_lines,
            set(WITH_CLAUSE_LINES) | {PENDING_LINE},
        )

        tasks = runtime.collect_tasks(lines)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["line"], PENDING_LINE)
        self.assertEqual(tasks[0]["text"], "Watch out!")
        self.assertTrue(tasks[0]["id"])

        adapter_snapshot = build_translation_snapshot(
            RenPyAdapter(legacy_module=runtime),
            ProjectDiscoveryRequest(
                project_root=str(root),
                localization_root=str(tl_dir),
                target_language="schinese",
            ),
        )
        occurrences = list(adapter_snapshot.occurrences)
        self.assertEqual(len(occurrences), 5)
        self.assertTrue(all(occurrence.unit.id for occurrence in occurrences))

        by_line = {
            occurrence.unit.display_line_number: occurrence
            for occurrence in occurrences
        }
        for line_index in WITH_CLAUSE_LINES:
            occurrence = by_line[line_index + 1]
            self.assertIn(
                occurrence.unit.id,
                set(identity_mapping),
            )
            self.assertTrue(occurrence.unit.source_text)
            self.assertTrue(occurrence.unit.metadata.get("live_catalog_text"))

        snapshot = build_project_snapshot(
            adapter_snapshot,
            GameVersion(version_id="v1"),
        )
        self.assertTrue(all(record.unit_id for record in snapshot.occurrences))

        jobs = batch.collect_revision_file_jobs(
            file_paths=[(rel_path, str(script))]
        )
        corpus_items = [item for job in jobs for item in job["items"]]
        corpus_ids = {item["id"] for item in corpus_items}
        translated_occurrences = [
            by_line[line_index + 1] for line_index in WITH_CLAUSE_LINES
        ]
        self.assertEqual(
            {occurrence.unit.id for occurrence in translated_occurrences},
            corpus_ids,
        )

        # The empty target is a pending task, not a revision candidate: the
        # corpus intentionally keeps it out while adapter/task views include it.
        entries = batch.collect_translation_entries_from_lines(
            SCRIPT.split("\n"),
            file_rel_path=rel_path,
        )
        pending_entries = [
            entry for entry in entries if entry.get("line_number") == PENDING_LINE + 1
        ]
        self.assertEqual(len(pending_entries), 1)
        self.assertEqual(pending_entries[0]["source"], "Watch out!")
        self.assertEqual(pending_entries[0]["translation"], "")
        self.assertFalse(batch.should_include_revision_entry(pending_entries[0]))

    def test_records_and_review_index_attach_with_clause_lines(self):
        root, tl_dir, script = self.make_project(SCRIPT)
        scan = build_translation_snapshot(
            RenPyAdapter(legacy_module=runtime),
            ProjectDiscoveryRequest(
                project_root=str(root),
                localization_root=str(tl_dir),
                target_language="schinese",
            ),
        )
        snapshot = build_project_snapshot(scan, GameVersion(version_id="v1"))
        translated_lines = {line + 1 for line in WITH_CLAUSE_LINES}
        translated = {
            occurrence.unit.id: occurrence
            for occurrence in scan.occurrences
            if occurrence.unit.display_line_number in translated_lines
        }
        self.assertEqual(len(translated), len(WITH_CLAUSE_LINES))
        record_set = build_translation_records(
            snapshot,
            [
                TranslationInput(
                    unit_id=occurrence.unit.id,
                    source_text=occurrence.unit.source_text,
                    translation_text=str(
                        occurrence.unit.metadata.get("live_catalog_text") or ""
                    ),
                    origin="human_confirmed",
                )
                for occurrence in translated.values()
            ],
        )
        jobs = batch.collect_revision_file_jobs(
            file_paths=[("script.rpy", str(script))]
        )
        digests = {"script.rpy": hashlib.sha256(script.read_bytes()).hexdigest()}
        corpus = root / "corpus"
        revision_corpus.export_revision_corpus(
            str(corpus),
            jobs,
            project_slug="demo",
            game_root=str(root),
            tl_dir=str(tl_dir),
            tl_subdir="schinese",
            source_digests_before=digests,
            source_digests_after=digests,
            source_digests_scanned=digests,
        )
        bundle = review_index.load_corpus_bundle(corpus)
        entries, diagnostics = review_index.build_index_entries(
            bundle["rows"],
            bundle["manifest"],
            records=[record.to_dict() for record in record_set.records],
        )
        self.assertEqual(diagnostics, [])
        attached = {
            entry["translation_record"]["unit_id"]
            for entry in entries
            if entry["translation_record"] is not None
        }
        self.assertEqual(attached, set(translated))

    def test_with_clause_marker_decodes_escaped_source(self):
        text = (
            "translate schinese escaped_1:\n"
            "\n"
            '    # m "Say \\"hi\\"\\nnow" with vpunch\n'
            '    m "说\\"你好\\"\\n现在" with vpunch\n'
        )
        snapshot = self.snapshot_for(text)
        self.assertEqual(len(snapshot.occurrences), 1)
        occurrence = snapshot.occurrences[0]
        self.assertIn(":escaped_1:1:", occurrence.unit.id)
        self.assertEqual(occurrence.unit.source_text, 'Say "hi"\nnow')
        self.assertEqual(occurrence.unit.current_translation, '说"你好"\n现在')

    def test_double_string_say_excludes_only_empty_body_from_revision(self):
        text = (
            "translate schinese pending_say:\n"
            '    # "Guard" "Wait here." with hpunch\n'
            '    "守卫" "" with hpunch\n'
        )
        root, tl_dir, script = self.make_project(text)
        snapshot = build_translation_snapshot(
            RenPyAdapter(legacy_module=runtime),
            ProjectDiscoveryRequest(
                project_root=str(root), localization_root=str(tl_dir),
                target_language="schinese",
            ),
        )
        self.assertEqual(
            [
                (item.unit.source_text, item.unit.current_translation)
                for item in snapshot.occurrences
            ],
            [("Guard", "守卫"), ("Wait here.", "")],
        )
        entries = batch.collect_translation_entries_from_lines(text.splitlines(), "script.rpy")
        self.assertEqual(
            [(item["source"], item["translation"]) for item in entries],
            [("Guard", "守卫"), ("Wait here.", "")],
        )
        self.assertEqual(
            [batch.should_include_revision_entry(item) for item in entries],
            [True, False],
        )
        jobs = batch.collect_revision_file_jobs(file_paths=[("script.rpy", str(script))])
        self.assertEqual([item["source"] for item in jobs[0]["items"]], ["Guard"])
        self.assertEqual(len(runtime.collect_tasks(text.splitlines(keepends=True))), 1)

    def test_double_string_say_shape_mismatch_reports_unpaired_marker(self):
        text = (
            "translate schinese malformed_say:\n"
            '    # "Guard" "Wait here."\n'
            '    "守卫"\n'
        )
        snapshot = self.snapshot_for(text)
        self.assertEqual(
            batch.collect_translation_entries_from_lines(text.splitlines(), "script.rpy"),
            [],
        )
        self.assertEqual(
            runtime.scan_all_translation_units(
                text.splitlines(keepends=True), "script.rpy",
                mode=translation_core.MODE_REVISION,
            ),
            {},
        )
        self.assertTrue(
            any(
                candidate.classification == "parse_error"
                and "renpy.source_marker_unpaired" in candidate.reason_codes
                for candidate in snapshot.inventory.candidates
            )
        )

    def test_double_string_say_writeback_changes_only_selected_literal(self):
        text = (
            "translate schinese selected_say:\n"
            '    # "Guard" "Hello, traveler." with vpunch\n'
            '    "守卫" "你好，旅人。" with vpunch\n'
        )
        root, tl_dir, script = self.make_project(text)
        adapter = RenPyAdapter(legacy_module=runtime)
        request = ProjectDiscoveryRequest(
            project_root=str(root), localization_root=str(tl_dir),
            target_language="schinese",
        )
        snapshot = build_translation_snapshot(adapter, request)
        name, body = snapshot.occurrences
        self.assertEqual(
            (name.unit.source_text, body.unit.source_text),
            ("Guard", "Hello, traveler."),
        )
        validation = adapter.validate_translation(body, "欢迎，旅人。")
        self.assertEqual(validation.status, "pass")
        plan = adapter.build_writeback_plan(
            snapshot.project,
            (ValidatedTranslation(body, "欢迎，旅人。", validation),),
            snapshot.project.source_documents,
        )
        self.assertEqual(len(plan.operations), 1)
        self.assertEqual(plan.operations[0].kind, "text_span_replace")
        rendered = "".join(
            render_writeback_plan(plan, snapshot.project.source_documents)["script.rpy"]
        )
        self.assertEqual(
            rendered.replace("\r\n", "\n"),
            text.replace('"你好，旅人。" with vpunch', '"欢迎，旅人。" with vpunch'),
        )
        self.assertEqual(script.read_text(encoding="utf-8"), text)

        script.write_text(text.replace("你好，旅人。", "已变动。"), encoding="utf-8")
        live = adapter.discover_project(request)
        with self.assertRaises(WritebackPlanError):
            render_writeback_plan(plan, live.source_documents)

    def test_unmarked_target_uses_locator_fallback_identity(self):
        snapshot = self.snapshot_for(SCRIPT)
        candidate = next(
            candidate
            for candidate in snapshot.inventory.candidates
            if candidate.unit is not None
            and candidate.unit.display_line_number == UNMARKED_LINE + 1
        )
        self.assertEqual(candidate.classification, "already_translated")
        self.assertTrue(candidate.unit.id)
        self.assertEqual(
            candidate.unit.id,
            "script.rpy:start_unmarked:loc:23:6",
        )
        self.assertEqual(
            candidate.evidence.get("identity_source"),
            "locator_fallback",
        )
        self.assertTrue(candidate.evidence.get("source_marker_missing"))
        self.assertEqual(candidate.evidence.get("identity_v2"), candidate.unit.id)

    def test_empty_unit_id_fails_with_stable_code_and_locator(self):
        unit = translation_core.TranslationUnit(
            id="",
            mode=translation_core.MODE_TRANSLATION,
            text="broken",
            source="broken",
            current_translation="坏",
            file_rel_path="script.rpy",
            line=11,
            line_number=12,
            start=4,
            end=10,
        )
        occurrence = Occurrence(
            occurrence_id="occ1:broken",
            engine="renpy",
            project_snapshot_fingerprint="fingerprint",
            content_fingerprint="content",
            candidate_id="candidate-1",
            locator=OpaqueLocator(
                engine="renpy",
                locator_schema_version=1,
                locator={"file_rel_path": "script.rpy", "line_hint": 12},
            ),
            unit=unit,
        )
        with self.assertRaises(
            versioning.SnapshotOccurrenceIdentityError
        ) as caught:
            versioning.build_unit_occurrence_records([occurrence])
        error = caught.exception
        self.assertEqual(error.code, "ADAPTER_UNIT_ID_MISSING")
        self.assertEqual(error.file_rel_path, "script.rpy")
        self.assertEqual(error.line_number, 12)
        self.assertEqual(error.occurrence_id, "occ1:broken")
        self.assertIn("script.rpy:12", str(error))

    def test_snapshot_identity_error_is_a_stable_cli_envelope(self):
        error = versioning.SnapshotOccurrenceIdentityError(
            file_rel_path="script.rpy",
            line_number=12,
            occurrence_id="occ1:broken",
        )
        stdout, diagnostics = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(
                batch,
                "run_project_snapshot_export",
                side_effect=error,
            ),
            mock.patch.object(batch.legacy, "load_translator_settings"),
            mock.patch.object(batch, "load_batch_settings"),
            redirect_stdout(stdout),
            redirect_stderr(diagnostics),
        ):
            exit_code = batch.main(
                [
                    "export-project-snapshot",
                    "--version-id",
                    "1.0",
                    "--output",
                    "json",
                    "--strict-exit-codes",
                ]
            )
        self.assertEqual(exit_code, 5)
        envelope = json.loads(stdout.getvalue())
        self.assertEqual(envelope["status"], "failed")
        self.assertEqual(envelope["error"]["code"], "ADAPTER_UNIT_ID_MISSING")
        self.assertEqual(
            envelope["error"]["suggested_action"],
            "fix_source_identity_and_reexport_snapshot",
        )
        self.assertEqual(envelope["error"]["details"]["file_rel_path"], "script.rpy")
        self.assertEqual(envelope["error"]["details"]["line_number"], 12)


if __name__ == "__main__":
    unittest.main()
