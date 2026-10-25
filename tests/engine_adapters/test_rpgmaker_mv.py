"""Independent original MV narrative positions, byte output and refusal paths."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from engine_adapters import (
    EngineAdapter, InventoryPolicy, MvSupportWindow, ProjectDiscoveryRequest,
    RpgMakerMvAdapter, VALIDATION_SCHEMA_VERSION, ValidatedTranslation, render_writeback_plan,
)
from engine_adapters.coverage import build_coverage_report, digest_json
from engine_adapters.rpgmaker_mv import CORE_FILES
from engine_adapters.writeback import WritebackPlanError
from execution.sync_translation_preview import _deserialize_writeback_plan
from foundation import atomic_io
from quality_review import structure_protection, translation_plan
from quality_review.translation_core import TranslationUnit


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "rpgmaker_mv_narrative"


def at(data, path):
    for component in path:
        data = data[component]
    return data


class MvAdapterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        # Discovery expands Windows 8.3 aliases. Transaction targets and their
        # preimage keys must use the same canonical spelling.
        self.root = Path(temporary.name).resolve()
        self.source, self.target = self.root / "source", self.root / "target"
        shutil.copytree(FIXTURE / "source", self.source)
        # These are original inert marker bytes, not distributed engine/plugin code.
        materials = {}
        for relative in CORE_FILES:
            materials[relative] = f"// Original test marker: {relative}\n".encode()
        materials["js/rpg_core.js"] += b"Utils.RPGMAKER_NAME = 'MV';\n"
        materials["js/plugins.js"] = (
            b'// Original registry fixture\nvar $plugins = '
            b'[{"name":"GALV_MessageStyles","status":true,"parameters":{}}];\n'
        )
        materials["js/plugins/GALV_MessageStyles.js"] = b"// Original inert 2.8 profile marker.\n"
        for relative, content in materials.items():
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (self.source / "img").mkdir()
        (self.source / "img" / "original.asset").write_bytes(b"original non-JSON resource\0")
        shutil.copytree(self.source, self.target)
        self.window = MvSupportWindow(
            {relative: hashlib.sha256(content).hexdigest() for relative, content in materials.items()},
            message_styles=True,
        )
        self.adapter = RpgMakerMvAdapter(self.window)
        self.request = ProjectDiscoveryRequest(str(self.source), str(self.target), "zh-CN")
        self.positions = json.loads((FIXTURE / "positions.json").read_text(encoding="utf-8"))

    def snapshot(self):
        project = self.adapter.discover_project(self.request)
        inventory = self.adapter.inventory_candidates(project, InventoryPolicy())
        report = self.report(project, inventory)
        project = replace(project, coverage_digest=report.coverage_digest,
                          coverage_review_digest="original-fixture-independent-position-review")
        selected = [row.candidate_id for row in inventory.candidates
                    if row.classification == "translatable"]
        occurrences = self.adapter.extract_occurrences(project, inventory, selected)
        return project, inventory, report, occurrences

    def report(self, project, inventory):
        return build_coverage_report(
            project, inventory, self.adapter.audit_extraction(project, inventory),
            adapter_behavior_digest=self.adapter.behavior_digest(),
        )

    def submissions(self, occurrences):
        by_position = {(row["file"], tuple(row["path"])): row for row in self.positions}
        submitted = []
        for occurrence in occurrences:
            locator = occurrence.locator.locator
            row = by_position[locator["file_rel_path"], tuple(locator["json_path"])]
            self.assertEqual(occurrence.unit.source, row["source"])
            self.assertEqual(locator.get("mirror_paths", []), row["mirrors"])
            translated = row["translation"]
            submitted.append(ValidatedTranslation(
                occurrence, translated, self.adapter.validate_translation(occurrence, translated)))
        return submitted

    def plan(self, project, occurrences, **kwargs):
        return self.adapter.build_writeback_plan(
            project, self.submissions(occurrences),
            kwargs.pop("live_sources", project.source_documents),
            live_targets=kwargs.pop("live_targets", project.target_documents), **kwargs)

    def alter(self, relative, mutate, *, target=False):
        root = self.target if target else self.source
        path = root / relative
        data = json.loads(path.read_text(encoding="utf-8"))
        mutate(data)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def alter_both(self, relative, mutate):
        self.alter(relative, mutate)
        (self.target / relative).write_bytes((self.source / relative).read_bytes())

    @staticmethod
    def commands(data):
        return data["events"][1]["pages"][0]["list"]

    def test_protocol_explicit_window_and_capabilities(self):
        self.assertIsInstance(self.adapter, EngineAdapter)
        self.assertEqual(self.adapter.capabilities().declarative_writeback,
                         ("json_source_string_set",))
        self.assertFalse(self.adapter.capabilities().native_catalog)
        with self.assertRaises(TypeError):
            RpgMakerMvAdapter()
        with self.assertRaisesRegex(ValueError, "window.incomplete"):
            MvSupportWindow({})
        with self.assertRaises(TypeError):
            self.window.material_sha256["js/rpg_core.js"] = "0" * 64

    def test_independent_positions_and_explicit_scope_gaps(self):
        project, inventory, report, occurrences = self.snapshot()
        self.assertEqual(len(occurrences), 14)
        self.assertEqual(report.catalog_provenance["narrative_physical_positions"], 21)
        self.assertEqual(report.coverage_status, "attention")
        self.assertEqual(report.classification_counts["unsupported"], 3)
        self.assertEqual(sum(row.reason_codes == ("rpgmaker_mv.empty_message",)
                             for row in inventory.candidates), 1)
        self.assertEqual(sum(row.reason_codes == ("rpgmaker_mv.choice_mirror",)
                             for row in inventory.candidates), 6)
        self.assertFalse(report.invariant_errors)
        self.submissions(occurrences)
        self.assertEqual(project.catalog_provenance["runtime_acceptance"], "unverified")

    def test_runtime_normalized_cancel_index_preserves_raw_choice_settings(self):
        self.alter_both('data/Map001.json', lambda data: self.commands(data)[3]['parameters'].__setitem__(1, 2))
        project, inventory, report, occurrences = self.snapshot()
        self.assertNotEqual(report.coverage_status, 'block')
        self.assertEqual(len(occurrences), 14)
        choice = next(row for row in occurrences if row.locator.locator.get('choice_index') == 0)
        self.assertEqual(choice.locator.locator['choice_settings'][0], 2)
        rendered = render_writeback_plan(self.plan(project, occurrences), project.source_documents,
            live_targets=project.target_documents)
        self.assertEqual(self.commands(json.loads(''.join(rendered['data/Map001.json'])))[3]['parameters'][1], 2)

    def test_normalized_cancel_can_have_no_403_without_weakening_other_choice_checks(self):
        def mutate(data):
            commands = self.commands(data)
            commands[3]['parameters'][1] = 2
            del commands[10:13]
        self.alter_both('data/Map001.json', mutate)
        project, inventory, report, occurrences = self.snapshot()
        self.assertNotEqual(report.coverage_status, 'block')
        self.assertEqual(len(occurrences), 13)
        for invalid in (-3, True, 1.5):
            self.alter_both('data/Map001.json', lambda data: self.commands(data)[3]['parameters'].__setitem__(1, invalid))
            project = self.adapter.discover_project(self.request)
            inventory = self.adapter.inventory_candidates(project, InventoryPolicy())
            self.assertEqual(self.report(project, inventory).coverage_status, 'block')

    def test_handwritten_output_twenty_physical_targets_and_other_bytes(self):
        project, _, _, occurrences = self.snapshot()
        plan = self.plan(project, occurrences)
        self.assertEqual(plan.writeback_plan_schema_version, 2)
        self.assertEqual(len(plan.operations), 20)
        rendered = render_writeback_plan(plan, project.source_documents,
                                        live_targets=project.target_documents)
        self.assertEqual(set(rendered), {"data/Map001.json", "data/Map002.json",
                                        "data/CommonEvents.json"})
        for relative, lines in rendered.items():
            self.assertEqual("".join(lines).encode("utf-8"),
                             (FIXTURE / "expected" / relative).read_bytes())
        self.assertIn("1.234567890123456789", "".join(rendered["data/Map001.json"]))
        self.assertEqual((self.target / "img/original.asset").read_bytes(),
                         (self.source / "img/original.asset").read_bytes())

    def test_bom_and_crlf_survive_adapter_and_shared_renderer(self):
        relative = "data/Map001.json"
        raw = b"\xef\xbb\xbf" + (FIXTURE / "source" / relative).read_bytes().replace(b"\n", b"\r\n")
        for root in (self.source, self.target):
            (root / relative).write_bytes(raw)
        project, _, _, occurrences = self.snapshot()
        rendered = render_writeback_plan(self.plan(project, occurrences), project.source_documents,
                                        live_targets=project.target_documents)
        expected = b"\xef\xbb\xbf" + (FIXTURE / "expected" / relative).read_bytes().replace(b"\n", b"\r\n")
        self.assertEqual("".join(rendered[relative]).encode("utf-8"), expected)

    def test_persisted_plan_and_existing_transaction_only_modify_copy(self):
        project, _, _, occurrences = self.snapshot()
        plan = _deserialize_writeback_plan(json.loads(json.dumps(self.plan(project, occurrences).to_dict())))
        rendered = render_writeback_plan(plan, project.source_documents,
                                        live_targets=project.target_documents)
        atomic_io.atomic_write_many_lines(
            [(self.target / relative, lines) for relative, lines in rendered.items()],
            expected_preimages={Path(doc.file_path): doc.sha256 for doc in project.target_documents},
            journal_path=self.target / "original-test-transaction.json",
        )
        for relative in rendered:
            self.assertEqual((self.target / relative).read_bytes(),
                             (FIXTURE / "expected" / relative).read_bytes())
            self.assertEqual((self.source / relative).read_bytes(),
                             (FIXTURE / "source" / relative).read_bytes())
        self.assertEqual((self.target / "data/Actors.json").read_bytes(),
                         (self.source / "data/Actors.json").read_bytes())

    def test_repeat_source_text_stays_distinct_and_has_message_context(self):
        _, _, _, occurrences = self.snapshot()
        duplicate = [row for row in occurrences if row.unit.text == "Same words."]
        self.assertEqual(len(duplicate), 2)
        self.assertNotEqual(duplicate[0].occurrence_id, duplicate[1].occurrence_id)
        self.assertNotEqual(duplicate[0].unit.id, duplicate[1].unit.id)
        continuation = next(row for row in occurrences if row.unit.text.startswith("A second"))
        self.assertEqual(continuation.locator.locator["message_line_index"], 1)
        self.assertEqual(len(continuation.unit.metadata["mv_context"]["message_lines"]), 2)

    def test_common_event_records_both_call_sites_but_one_physical_write(self):
        project, _, _, occurrences = self.snapshot()
        common = [row for row in occurrences if row.unit.file_rel_path == "data/CommonEvents.json"]
        self.assertEqual(len(common), 1)
        self.assertEqual(len(common[0].locator.locator["common_event_call_sites"]), 2)
        self.assertEqual(len(self.plan(project, common).operations), 1)

    def test_partial_choice_always_includes_its_independent_mirror(self):
        project, _, _, occurrences = self.snapshot()
        choice = next(row for row in occurrences if row.unit.text == "Wait")
        plan = self.plan(project, [choice])
        self.assertEqual(len(plan.operations), 2)
        self.assertEqual({operation.target_json_path for operation in plan.operations},
                         {tuple(choice.locator.locator["json_path"]),
                          tuple(choice.locator.locator["mirror_paths"][0])})

    def test_nested_choice_links_stay_with_the_correct_indent(self):
        nested = [
            {"code":102, "indent":1, "parameters":[["Inspect","Return"],-1,0,2,0]},
            {"code":402, "indent":1, "parameters":[0,"Inspect"]},
            {"code":230, "indent":2, "parameters":[1]},
            {"code":402, "indent":1, "parameters":[1,"Return"]},
            {"code":230, "indent":2, "parameters":[2]},
            {"code":404, "indent":1, "parameters":[]},
        ]
        self.alter_both("data/Map002.json",
                        lambda data: self.commands(data).__setitem__(slice(4, 5), nested))
        project, _, _, occurrences = self.snapshot()
        occurrence = next(row for row in occurrences if row.unit.text == "Inspect")
        translated = "查看"
        plan = self.adapter.build_writeback_plan(
            project, [ValidatedTranslation(occurrence, translated,
                self.adapter.validate_translation(occurrence, translated))],
            project.source_documents, live_targets=project.target_documents)
        self.assertEqual({row.target_json_path for row in plan.operations}, {
            ("events",1,"pages",0,"list",4,"parameters",0,0),
            ("events",1,"pages",0,"list",5,"parameters",1),
        })
        rendered = render_writeback_plan(plan, project.source_documents,
                                        live_targets=project.target_documents)
        commands = self.commands(json.loads("".join(rendered["data/Map002.json"])))
        self.assertEqual(commands[2]["parameters"][0], ["Yes", "No"])
        self.assertEqual(commands[3]["parameters"][1], "Yes")
        self.assertEqual(commands[5]["parameters"][1], "查看")

    def test_translated_target_never_becomes_new_source(self):
        for path in (FIXTURE / "expected/data").iterdir():
            (self.target / "data" / path.name).write_bytes(path.read_bytes())
        project, _, _, occurrences = self.snapshot()
        self.assertEqual(occurrences[0].unit.source, occurrences[0].unit.text)
        self.assertTrue(any(row.unit.current_translation != row.unit.source for row in occurrences))
        rendered = render_writeback_plan(self.plan(project, occurrences), project.source_documents,
                                        live_targets=project.target_documents)
        for relative, lines in rendered.items():
            self.assertEqual("".join(lines).encode(), (self.target / relative).read_bytes())

    def test_source_and_target_roots_cannot_overlap_or_omit_language(self):
        for request in (
            replace(self.request, localization_root=str(self.source)),
            replace(self.request, localization_root=str(self.source / "nested")),
            replace(self.request, project_root=str(self.target.parent)),
            replace(self.request, target_language=""),
            replace(self.request, include_files=("data/Map001.json",)),
        ):
            with self.subTest(request=request), self.assertRaises(WritebackPlanError):
                self.adapter.discover_project(request)

    def test_core_plugin_registry_and_plugin_bytes_are_pinned_on_both_sides(self):
        for relative in ("js/rpg_core.js", "js/plugins.js", "js/plugins/GALV_MessageStyles.js"):
            for root in (self.source, self.target):
                with self.subTest(relative=relative, side=root.name):
                    path = root / relative
                    original = path.read_bytes()
                    path.write_bytes(original + b"// drift\n")
                    with self.assertRaisesRegex(WritebackPlanError, "material changed"):
                        self.adapter.discover_project(self.request)
                    path.write_bytes(original)

    def test_mz_and_unbound_plugin_profiles_are_rejected(self):
        (self.source / "js/rmmz_core.js").write_bytes(b"// original MZ marker")
        with self.assertRaisesRegex(WritebackPlanError, "MZ"):
            self.adapter.discover_project(self.request)
        (self.source / "js/rmmz_core.js").unlink()
        adapter = RpgMakerMvAdapter(MvSupportWindow(dict(self.window.material_sha256)))
        with self.assertRaisesRegex(WritebackPlanError, "profile mismatch"):
            adapter.discover_project(self.request)

    def test_missing_common_events_are_not_silently_counted_as_zero(self):
        (self.source / "data/CommonEvents.json").unlink()
        with self.assertRaisesRegex(WritebackPlanError, "zero coverage"):
            self.adapter.discover_project(self.request)

    def test_non_target_fields_and_array_structure_cannot_change_in_target(self):
        for mutate in (
            lambda data: data.update(width=13),
            lambda data: self.commands(data)[0]["parameters"].__setitem__(0, "OtherFaces"),
            lambda data: self.commands(data)[3]["parameters"].__setitem__(1, -1),
            lambda data: data["events"][1]["pages"].reverse(),
            lambda data: self.commands(data).insert(0, {"code":0, "indent":0, "parameters":[]}),
        ):
            with self.subTest(mutate=mutate):
                original = (self.target / "data/Map001.json").read_bytes()
                self.alter("data/Map001.json", mutate, target=True)
                with self.assertRaisesRegex(WritebackPlanError, "structure changed"):
                    self.adapter.discover_project(self.request)
                (self.target / "data/Map001.json").write_bytes(original)

    def test_distinct_high_precision_numbers_do_not_collapse_during_target_comparison(self):
        path = self.target / "data/Map001.json"
        path.write_bytes(path.read_bytes().replace(
            b"1.234567890123456789", b"1.234567890123456788"))
        with self.assertRaisesRegex(WritebackPlanError, "structure changed"):
            self.adapter.discover_project(self.request)

    def test_independent_coverage_detects_missing_duplicate_and_hidden_candidates(self):
        project, inventory, _, _ = self.snapshot()
        first = next(row for row in inventory.candidates if row.classification == "translatable")
        excluded = replace(first, classification="explicitly_excluded",
                           translation_scope="exclude", analysis_scope="exclude")
        unsupported = replace(first, classification="unsupported",
                              translation_scope="unknown", analysis_scope="unknown")
        for candidates in (
            tuple(row for row in inventory.candidates if row is not first),
            (*inventory.candidates, first),
            tuple(excluded if row is first else row for row in inventory.candidates),
            tuple(unsupported if row is first else row for row in inventory.candidates),
        ):
            with self.subTest(count=len(candidates)):
                altered = replace(inventory, candidates=candidates)
                report = self.report(project, altered)
                self.assertEqual(report.coverage_status, "block")
                self.assertIn("rpgmaker_mv.coverage.mismatch", report.reason_counts)
                with self.assertRaises(WritebackPlanError):
                    self.adapter.extract_occurrences(project, altered, [first.candidate_id])

    def test_changed_source_and_target_invalidate_old_plans(self):
        project, _, _, occurrences = self.snapshot()
        plan = self.plan(project, occurrences)
        for target in (False, True):
            root = self.target if target else self.source
            path = root / "data/Map001.json"
            original = path.read_bytes()
            path.write_bytes(original.replace(b"Hello, traveler.", b"Hello, visitor."))
            live = self.adapter.read_live_documents(project, target=target)
            with self.subTest(target=target), self.assertRaises(WritebackPlanError):
                self.plan(project, occurrences, **{"live_targets" if target else "live_sources": live})
            with self.assertRaises(WritebackPlanError):
                render_writeback_plan(plan, project.source_documents if target else live,
                                      live_targets=live if target else project.target_documents)
            path.write_bytes(original)

    def test_new_data_file_and_runtime_material_drift_are_not_ignored(self):
        project, inventory, _, occurrences = self.snapshot()
        (self.source / "data/Map003.json").write_bytes((self.source / "data/Map002.json").read_bytes())
        self.assertEqual(self.report(project, inventory).coverage_status, "block")
        live = self.adapter.read_live_documents(project)
        result = self.adapter.relocate_occurrences(project, occurrences, live)
        self.assertFalse(result.occurrences)
        self.assertEqual(len(result.unresolved_occurrence_ids), 14)
        (self.source / "data/Map003.json").unlink()
        (self.target / "js/plugins.js").write_bytes(b"// drift")
        self.assertEqual(self.report(project, inventory).coverage_status, "block")

    def test_same_text_after_command_or_page_movement_is_not_relocated(self):
        project, _, _, occurrences = self.snapshot()
        for mutate in (
            lambda data: self.commands(data).insert(0, {"code":230, "indent":0, "parameters":[1]}),
            lambda data: data["events"][1]["pages"].reverse(),
        ):
            original = (self.source / "data/Map001.json").read_bytes()
            self.alter("data/Map001.json", mutate)
            result = self.adapter.relocate_occurrences(
                project, occurrences, self.adapter.read_live_documents(project))
            self.assertFalse(result.occurrences)
            self.assertEqual(len(result.unresolved_occurrence_ids), 14)
            (self.source / "data/Map001.json").write_bytes(original)

    def test_exact_relocation_and_altered_identity_refusal(self):
        project, _, _, occurrences = self.snapshot()
        result = self.adapter.relocate_occurrences(project, occurrences, project.source_documents)
        self.assertEqual(result.occurrences, occurrences)
        self.assertFalse(result.unresolved_occurrence_ids)
        changed = replace(occurrences[0], content_fingerprint="forged")
        result = self.adapter.relocate_occurrences(project, [changed], project.source_documents)
        self.assertEqual(result.unresolved_occurrence_ids, (changed.occurrence_id,))

    def test_source_only_missing_coverage_and_forged_validation_are_refused(self):
        project, _, _, occurrences = self.snapshot()
        submitted = self.submissions(occurrences)
        with self.assertRaises(WritebackPlanError):
            self.adapter.build_writeback_plan(project, submitted, project.source_documents)
        for bad_project in (replace(project, coverage_digest=""),
                            replace(project, coverage_review_digest=""),
                            replace(project, coverage_digest="old")):
            with self.subTest(project=bad_project.coverage_digest), self.assertRaises(WritebackPlanError):
                self.plan(bad_project, occurrences)
        forged = replace(submitted[0], validation=replace(submitted[0].validation,
                                                         translation_digest="old"))
        with self.assertRaises(WritebackPlanError):
            self.adapter.build_writeback_plan(project, [forged], project.source_documents,
                                             live_targets=project.target_documents)

    def test_duplicate_submission_and_modified_mirror_locator_are_refused(self):
        project, inventory, _, occurrences = self.snapshot()
        with self.assertRaises(WritebackPlanError):
            self.adapter.extract_occurrences(project, inventory, [occurrences[0].candidate_id] * 2)
        with self.assertRaises(WritebackPlanError):
            self.plan(project, [occurrences[0], occurrences[0]])
        choice = next(row for row in occurrences if row.unit.text == "Wait")
        address = {**choice.locator.locator, "mirror_paths": []}
        changed = replace(choice, locator=replace(choice.locator, locator=address))
        with self.assertRaises(WritebackPlanError):
            self.adapter.build_writeback_plan(
                project, [ValidatedTranslation(changed, "等一等",
                    self.adapter.validate_translation(changed, "等一等"))],
                project.source_documents, live_targets=project.target_documents)

    def test_snapshot_metadata_lies_and_duplicate_documents_are_refused(self):
        project, _, _, occurrences = self.snapshot()
        for live in (
            (*project.source_documents, project.source_documents[0]),
            (replace(project.source_documents[0], content=b"forged"), *project.source_documents[1:]),
        ):
            with self.subTest(count=len(live)), self.assertRaises(WritebackPlanError):
                self.plan(project, occurrences, live_sources=live)

    def test_invalid_event_shapes_and_choice_relationships_block_extraction(self):
        path = self.source / "data/Map001.json"
        original = path.read_bytes()
        mutations = [
            lambda rows: rows[1].__setitem__("code", True),
            lambda rows: rows[1]["parameters"].__setitem__(0, None),
            lambda rows: rows[1].__setitem__("indent", True),
            lambda rows: rows[0]["parameters"].append("MZ speaker"),
            lambda rows: rows[4]["parameters"].__setitem__(1, "Wrong mirror"),
            lambda rows: rows[4]["parameters"].__setitem__(0, 7),
            lambda rows: rows[7]["parameters"].__setitem__(0, 0),
            lambda rows: rows[3]["parameters"].__setitem__(1, -3),
            lambda rows: rows[3]["parameters"].__setitem__(2, True),
            lambda rows: rows.__delitem__(10),
            lambda rows: rows.__delitem__(13),
            lambda rows: rows.__delitem__(0),
            lambda rows: rows.__setitem__(1, None),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                path.write_bytes(original)
                self.alter_both("data/Map001.json", lambda data: mutate(self.commands(data)))
                project = self.adapter.discover_project(self.request)
                inventory = self.adapter.inventory_candidates(project, InventoryPolicy())
                self.assertEqual(self.report(project, inventory).coverage_status, "block")
                with self.assertRaises(WritebackPlanError):
                    self.adapter.extract_occurrences(project, inventory, [])
        path.write_bytes(original)
        (self.target / "data/Map001.json").write_bytes(original)

    def test_invalid_json_duplicate_keys_and_surrogates_are_reported(self):
        for content in (b'{"events":', b'{"events":[],"events":[]}',
                        b'{"events":[],"name":"\\ud800"}', b'{"events":[],"width":NaN}'):
            with self.subTest(content=content):
                (self.source / "data/Map001.json").write_bytes(content)
                (self.target / "data/Map001.json").write_bytes(content)
                project = self.adapter.discover_project(self.request)
                inventory = self.adapter.inventory_candidates(project, InventoryPolicy())
                self.assertEqual(self.report(project, inventory).coverage_status, "block")

    def test_unknown_controls_and_popup_on_continuation_cannot_be_extracted(self):
        path = self.source / "data/Map001.json"
        original = path.read_bytes()
        for text in (r"\Unknown[1]Word", r"\c[bad]Word", r"\Ctext",
                     r"\pop[0]Wrong continuation"):
            with self.subTest(text=text):
                path.write_bytes(original)
                self.alter_both("data/Map001.json", lambda data: self.commands(data)[2]["parameters"].__setitem__(0, text))
                project = self.adapter.discover_project(self.request)
                inventory = self.adapter.inventory_candidates(project, InventoryPolicy())
                self.assertGreater(self.report(project, inventory).classification_counts["unknown"], 0)
                with self.assertRaises(WritebackPlanError):
                    self.adapter.extract_occurrences(project, inventory, [])

    def test_translation_controls_case_order_parameters_and_ownership_are_exact(self):
        _, _, _, occurrences = self.snapshot()
        first = next(row for row in occurrences if row.unit.text.startswith(r"\pop"))
        valid = r"\pop[0]\c[2]你好，旅人。\C[0]"
        self.assertEqual(self.adapter.validate_translation(first, valid).status, "pass")
        for text in (
            "", " ", valid.replace(r"\c[2]", r"\C[2]"),
            valid.replace(r"\c[2]", r"\c[3]"), valid.replace(r"\pop[0]", r"\pop[1]"),
            "前缀" + valid, valid + "\n", valid + r"\x[1]",
            r"\pop[0]\C[0]你好，旅人。\c[2]",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.adapter.validate_translation(first, text).status, "block")

    def test_invalid_utf8_translation_returns_a_serializable_block(self):
        _, _, _, occurrences = self.snapshot()
        occurrence = next(row for row in occurrences if row.unit.text == "Same words.")
        for encoded in (r'"\ud800"', r'"\udfff"', r'"before\ud800after"',
                        r'"\ud800\ud800"'):
            with self.subTest(encoded=encoded):
                text = json.loads(encoded)
                result = self.adapter.validate_translation(occurrence, text)
                self.assertEqual(result.status, "block")
                self.assertEqual(result.reason_codes, ("rpgmaker_mv.translation.encoding",))
                self.assertRegex(result.translation_digest, r"^[0-9a-f]{64}$")
                json.dumps(result.to_dict(), ensure_ascii=False).encode("utf-8")

    def test_invalid_utf8_digest_preserves_distinct_input_codepoints(self):
        _, _, _, occurrences = self.snapshot()
        occurrence = next(row for row in occurrences if row.unit.text == "Same words.")
        texts = (
            chr(0xD800), chr(0xDFFF), chr(0xD800) * 2,
            chr(0xD800) + chr(0xDC00) + chr(0xD800),
            chr(0x10000) + chr(0xD800),
        )
        digests = [self.adapter.validate_translation(occurrence, text).translation_digest
                   for text in texts]
        self.assertEqual(len(set(digests)), len(texts))
        self.assertEqual(self.adapter.validate_translation(occurrence, texts[0]).translation_digest,
                         digests[0])
        for repaired in ("\ufffd", r"\ud800"):
            self.assertNotIn(self.adapter.validate_translation(occurrence, repaired).translation_digest,
                             digests)

    def test_valid_utf8_translation_digest_keeps_the_existing_contract(self):
        _, _, _, occurrences = self.snapshot()
        occurrence = next(row for row in occurrences if row.unit.text == "Same words.")
        for text in ("同一句话。", "旅人🌿"):
            with self.subTest(text=text):
                result = self.adapter.validate_translation(occurrence, text)
                self.assertEqual(result.status, "pass")
                self.assertEqual(result.translation_digest, digest_json({
                    "validation_schema_version": VALIDATION_SCHEMA_VERSION, "translation": text,
                }))

    def test_invalid_utf8_and_forged_pass_cannot_authorize_writeback(self):
        project, _, _, occurrences = self.snapshot()
        occurrence = next(row for row in occurrences if row.unit.text == "Same words.")
        text = json.loads(r'"\ud800"')
        blocked = self.adapter.validate_translation(occurrence, text)
        for validation in (blocked, replace(blocked, status="pass", reason_codes=())):
            with self.subTest(status=validation.status), self.assertRaises(WritebackPlanError) as error:
                self.adapter.build_writeback_plan(
                    project, [ValidatedTranslation(occurrence, text, validation)],
                    project.source_documents, live_targets=project.target_documents)
            self.assertEqual(error.exception.reason_code, "rpgmaker_mv.validation.stale")
        for document in project.target_documents:
            self.assertEqual(Path(document.file_path).read_bytes(), document.content)

    def test_mv_batch_route_remains_explicitly_unsupported(self):
        with self.assertRaisesRegex(ValueError, "durable Sync"):
            translation_plan.build_translation_plan([], execution_strategy="gemini_batch",
                                                    source_identity={"engine":"rpgmaker_mv"})

    def test_late_target_edit_is_still_guarded_by_existing_atomic_transaction(self):
        project, _, _, occurrences = self.snapshot()
        plan = self.plan(project, occurrences)
        rendered = render_writeback_plan(plan, project.source_documents,
                                        live_targets=project.target_documents)
        path = self.target / "data/Map001.json"
        changed = path.read_bytes() + b" "
        path.write_bytes(changed)
        with self.assertRaises(atomic_io.AtomicWritePreimageConflict):
            atomic_io.atomic_write_many_lines(
                [(self.target / relative, lines) for relative, lines in rendered.items()],
                expected_preimages={Path(doc.file_path): doc.sha256 for doc in project.target_documents},
                journal_path=self.target / "original-test-transaction.json")
        self.assertEqual(path.read_bytes(), changed)

    def test_adapter_plan_failure_rolls_back_all_original_target_bytes(self):
        project, _, _, occurrences = self.snapshot()
        plan = self.plan(project, occurrences)
        rendered = render_writeback_plan(plan, project.source_documents,
                                        live_targets=project.target_documents)
        real_replace = atomic_io.os.replace
        failed = False

        def fail_second(source, destination):
            nonlocal failed
            if (not failed and Path(destination) == self.target / "data/Map002.json"
                    and str(source).endswith(".txn.tmp")):
                failed = True
                raise OSError("original MV fixture failed write")
            return real_replace(source, destination)

        with mock.patch.object(atomic_io.os, "replace", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "fixture failed write"):
                atomic_io.atomic_write_many_lines(
                    [(self.target / relative, lines) for relative, lines in rendered.items()],
                    expected_preimages={Path(doc.file_path): doc.sha256 for doc in project.target_documents},
                    journal_path=self.target / "original-test-transaction.json")
        for doc in project.target_documents:
            self.assertEqual(Path(doc.file_path).read_bytes(), doc.content)

    def test_adapter_plan_crash_journal_recovers_bytes_before_retry(self):
        project, _, _, occurrences = self.snapshot()
        rendered = render_writeback_plan(self.plan(project, occurrences), project.source_documents,
                                        live_targets=project.target_documents)
        real_replace = atomic_io.os.replace
        journal = self.target / "original-test-transaction.json"

        def crash_second(source, destination):
            if (Path(destination) == self.target / "data/Map002.json"
                    and str(source).endswith(".txn.tmp")):
                raise SystemExit("original MV fixture crash")
            return real_replace(source, destination)

        with mock.patch.object(atomic_io.os, "replace", side_effect=crash_second):
            with self.assertRaises(SystemExit):
                atomic_io.atomic_write_many_lines(
                    [(self.target / relative, lines) for relative, lines in rendered.items()],
                    expected_preimages={Path(doc.file_path): doc.sha256 for doc in project.target_documents},
                    journal_path=journal)
        self.assertTrue(journal.exists())
        self.assertTrue(atomic_io.recover_atomic_write_transaction(
            journal, expected_transaction_kind="apply"))
        for doc in project.target_documents:
            self.assertEqual(Path(doc.file_path).read_bytes(), doc.content)


class MvProtectionTests(unittest.TestCase):
    def test_canonical_units_retain_popup_profile_in_shared_model_view(self):
        source = r"\pop[0]Hello"
        unit = TranslationUnit(id="original-line", text=source,
                               metadata={"mv_message_styles": True})
        views, bundle = structure_protection.model_units(
            [unit], engine="rpgmaker_mv", request_id="original-request")
        self.assertEqual(unit.text, source)
        self.assertNotIn(r"\pop[0]", views[0].text)
        mapping = bundle["items"][unit.id]
        self.assertTrue(mapping["mv_message_styles"])
        self.assertEqual(structure_protection.restore(
            views[0].text.replace("Hello", "你好"), mapping, source=source,
            engine="rpgmaker_mv", request_id="original-request", item_id=unit.id,
            mv_message_styles=True), r"\pop[0]你好")

    def test_core_tokens_keep_case_arguments_and_order_in_shared_protection(self):
        source = r"\V[1]\n[2]\P[3]\g\I[4]\c[5]\C[0]Hello\\. \{large\}\$\. \|\!\>\<\^"
        args = dict(engine="rpgmaker_mv", request_id="original-fixture", item_id="line")
        view, mapping = structure_protection.protect(source, **args)
        self.assertNotIn(r"\V[1]", view)
        translated = view.replace("Hello", "你好")
        self.assertEqual(structure_protection.restore(translated, mapping, source=source, **args),
                         source.replace("Hello", "你好"))
        for damaged in (translated.replace(mapping["entries"][0]["marker"], ""),
                        translated + mapping["entries"][0]["marker"]):
            with self.assertRaises(ValueError):
                structure_protection.restore(damaged, mapping, source=source, **args)

    def test_popup_profile_is_explicit_and_mapping_bound(self):
        source = r"\pop[-1]Hello \c[1]there\C[0]"
        args = dict(engine="rpgmaker_mv", request_id="original-fixture", item_id="line")
        with self.assertRaises(ValueError):
            structure_protection.protect(source, **args)
        view, mapping = structure_protection.protect(source, **args, mv_message_styles=True)
        restored = structure_protection.restore(view.replace("Hello", "你好"), mapping,
                                                source=source, **args, mv_message_styles=True)
        self.assertEqual(restored, source.replace("Hello", "你好"))
        with self.assertRaises(ValueError):
            structure_protection.restore(view, mapping, source=source, **args)
        with self.assertRaisesRegex(ValueError, "popup_position"):
            structure_protection.restore("前缀" + view, mapping, source=source, **args,
                                         mv_message_styles=True)

    def test_unknown_malformed_and_unregistered_popup_forms_are_rejected(self):
        for source in (r"\Unknown[1]", r"\V[x]", r"\V[\V[1]]", r"\pop[a1]",
                       r"\pop[1,2]", r"\C[1", "\\", r"\q", "\\\u00a0q"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                structure_protection.protect(source, engine="rpgmaker_mv", request_id="r",
                                             item_id="i", mv_message_styles=True)

    def test_backslash_ascii_space_is_preserved_as_an_exact_token(self):
        source = 'Hello\\ world.'
        args = dict(engine='rpgmaker_mv', request_id='original-fixture', item_id='line')
        view, mapping = structure_protection.protect(source, **args)
        self.assertEqual([entry['value'] for entry in mapping['entries']], ['\\ '])
        translated = view.replace('Hello', '你好').replace('world.', '世界。')
        self.assertEqual(structure_protection.restore(translated, mapping, source=source, **args), '你好\\ 世界。')
        with self.assertRaises(ValueError):
            structure_protection.restore(translated.replace(mapping['entries'][0]['marker'], ''),
                                         mapping, source=source, **args)


if __name__ == "__main__":
    unittest.main()
