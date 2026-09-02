# -*- coding: utf-8 -*-
"""TyranoScript V600+ hybrid adapter tests (#265 P5 / #399)."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path

from engine_adapters.coverage import build_coverage_report, digest_json
from engine_adapters.contracts import (
    InventoryPolicy,
    ProjectDiscoveryRequest,
    ValidatedTranslation,
)
from engine_adapters.tyrano import (
    ADAPTER_VERSION,
    TyranoAdapter,
    _merged_tag_registry,
    _read_keep_space_setting,
    build_translation_snapshot,
    parse_tyrano_scenario,
)
from engine_adapters.writeback import WritebackPlanError, render_writeback_plan

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tyranoscript_v600"


class TyranoAdapterFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = TyranoAdapter()
        cls.snapshot = build_translation_snapshot(
            cls.adapter,
            ProjectDiscoveryRequest(
                project_root=str(FIXTURE_DIR),
                localization_root=str(FIXTURE_DIR / "data" / "others" / "lang"),
                target_language="ch",
            ),
        )

    def test_capabilities_are_hybrid_with_catalog_writeback(self):
        capabilities = self.adapter.capabilities()
        self.assertEqual(capabilities.engine, "tyrano")
        self.assertEqual(capabilities.adapter_version, ADAPTER_VERSION)
        self.assertTrue(capabilities.source_inventory)
        self.assertTrue(capabilities.native_catalog)
        self.assertTrue(capabilities.native_catalog_required_for_writeback)
        self.assertEqual(capabilities.declarative_writeback, ("json_catalog_set",))
        self.assertEqual(capabilities.relocation, False)

    def test_discovery_scans_ks_and_fingerprint_behavior_config(self):
        project = self.snapshot.project
        docs = {document.file_rel_path for document in project.source_documents}
        self.assertEqual(
            docs,
            {
                "data/scenario/scene1.ks",
                "data/scenario/choices.ks",
                "data/scenario/broken.ks",
                "data/system/Config.tjs",
                "data/others/lang/ch.json",
            },
        )
        self.assertTrue(project.source_fingerprint)
        self.assertTrue(project.project_snapshot_fingerprint)
        self.assertEqual(project.catalog_provenance["target_language"], "ch")
        self.assertEqual(
            project.catalog_provenance["catalog_rel_path"],
            "data/others/lang/ch.json",
        )

    def test_inventory_classification_contract(self):
        inventory = self.snapshot.inventory
        counts = Counter(candidate.classification for candidate in inventory.candidates)
        self.assertEqual(
            counts,
            Counter(
                {
                    "explicitly_excluded": 17,
                    "already_translated": 12,
                    "parse_error": 4,
                    "unsupported": 2,
                    "translatable": 1,
                    "unknown": 1,
                }
            ),
        )
        # No candidate is silently dropped: every parser node / comment line is
        # inventoried exactly once.
        self.assertEqual(len(inventory.candidates), 37)
        self.assertEqual(len({candidate.candidate_id for candidate in inventory.candidates}), 37)

    def test_inventory_matches_golden_inventory_contract(self):
        golden = json.loads(
            (FIXTURE_DIR / "expected" / "inventory.json").read_text(encoding="utf-8")
        )
        candidates = list(self.snapshot.inventory.candidates)
        by_golden_key = {}
        for item in golden["candidates"]:
            file_rel_path = item["file_rel_path"]
            key = (file_rel_path, item["line"], item["node_index"])
            by_golden_key[key] = item

        for candidate in candidates:
            locator = candidate.locator.locator
            scenario = locator["scenario"]
            node_index = locator.get("node_index")
            key = (scenario, locator["line"], node_index)
            self.assertIn(key, by_golden_key, candidate.candidate_id)
            expected = by_golden_key[key]
            self.assertEqual(
                candidate.classification,
                expected["classification"],
                key,
            )
            self.assertEqual(
                candidate.structure_kind,
                expected["structure_kind"],
                key,
            )
            self.assertEqual(
                tuple(candidate.reason_codes),
                tuple(expected["reason_codes"]),
                key,
            )
            if expected.get("catalog"):
                self.assertIsNotNone(candidate.catalog_link, key)
                self.assertEqual(
                    candidate.catalog_link["path"],
                    expected["catalog"]["path"],
                    key,
                )
                self.assertEqual(
                    candidate.catalog_link["translation"],
                    expected["catalog"]["translation"],
                    key,
                )
            else:
                self.assertIsNone(candidate.catalog_link, key)

            source_value = locator.get("parser_value")
            if node_index is None and candidate.structure_kind == "comment":
                lines = (FIXTURE_DIR / "data" / "scenario" / scenario).read_text(
                    encoding="utf-8"
                ).splitlines()
                source_value = lines[locator["line"]].strip()
            self.assertEqual(source_value, expected["source_value"], key)

        self.assertEqual(len(candidates), len(golden["candidates"]))

    def test_candidate_catalog_links_resolve_to_live_catalog(self):
        catalog = json.loads((FIXTURE_DIR / "data" / "others" / "lang" / "ch.json").read_text(encoding="utf-8"))
        for candidate in self.snapshot.inventory.candidates:
            link = candidate.catalog_link
            if link is None:
                continue
            value = catalog
            for part in link["path"]:
                self.assertIn(part, value, candidate.candidate_id)
                value = value[part]
            self.assertEqual(link["translation"], value, candidate.candidate_id)

    def test_malformed_lines_are_parse_errors_not_translatable(self):
        parse_errors = [
            candidate
            for candidate in self.snapshot.inventory.candidates
            if candidate.classification == "parse_error"
        ]
        self.assertEqual(len(parse_errors), 4)
        for candidate in parse_errors:
            self.assertNotEqual(candidate.classification, "translatable")
            self.assertNotEqual(candidate.classification, "already_translated")
            self.assertEqual(
                candidate.locator.locator["file_rel_path"],
                "data/scenario/broken.ks",
            )

    def test_escaped_quote_is_catalog_matched(self):
        candidates = [
            candidate
            for candidate in self.snapshot.inventory.candidates
            if candidate.locator.locator.get("name") == "ptext"
            and candidate.raw_excerpt == "It's fine"
        ]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].classification, "already_translated")
        self.assertEqual(candidates[0].catalog_link["translation"], "没问题")

    def test_iscript_text_is_explicitly_excluded(self):
        choices = [
            candidate
            for candidate in self.snapshot.inventory.candidates
            if candidate.locator.locator["file_rel_path"] == "data/scenario/choices.ks"
        ]
        # ``Candidate`` has no ``line`` property; use locator line.
        iscript_text = [
            candidate
            for candidate in choices
            if candidate.locator.locator["line"] in (14, 15)
            and candidate.structure_kind == "text"
        ]
        self.assertEqual(len(iscript_text), 2)
        self.assertTrue(
            all(candidate.classification == "explicitly_excluded" for candidate in iscript_text)
        )
        self.assertTrue(
            all("tyrano.iscript_content" in candidate.reason_codes for candidate in iscript_text)
        )

    def test_unknown_macro_and_dynamic_parameter_are_not_writable(self):
        unknown = [
            candidate
            for candidate in self.snapshot.inventory.candidates
            if candidate.classification == "unknown"
        ]
        self.assertEqual(len(unknown), 1)
        self.assertEqual(
            unknown[0].reason_codes,
            ("tyrano.unregistered_macro_invocation",),
        )
        dynamic = [
            candidate
            for candidate in self.snapshot.inventory.candidates
            if candidate.classification == "unsupported"
            and "tyrano.dynamic_parameter_expression" in candidate.reason_codes
        ]
        self.assertEqual(len(dynamic), 1)
        self.assertEqual(dynamic[0].raw_excerpt, "&sf.button_label")

    def test_coverage_status_is_block_without_review_overrides(self):
        report = self.snapshot.report
        self.assertEqual(report.coverage_status, "block")
        self.assertEqual(report.catalog_freshness, "unknown")
        self.assertFalse(report.invariant_errors)
        self.assertEqual(report.engine, "tyrano")
        self.assertTrue(report.coverage_digest)

    def test_extract_occurrences_only_returns_extractable_candidates(self):
        inventory = self.snapshot.inventory
        approved = [
            candidate.candidate_id
            for candidate in inventory.candidates
            if candidate.classification in {"translatable", "already_translated"}
        ]
        self.assertEqual(len(approved), 13)
        occurrences = self.adapter.extract_occurrences(
            self.snapshot.project,
            inventory,
            approved,
        )
        self.assertEqual(len(occurrences), 13)
        self.assertEqual(
            {occurrence.locator.engine for occurrence in occurrences},
            {"tyrano"},
        )
        for occurrence in occurrences:
            self.assertTrue(occurrence.unit.id)
            self.assertTrue(occurrence.unit.source)
            self.assertTrue(occurrence.content_fingerprint)

        with self.assertRaises(ValueError):
            self.adapter.extract_occurrences(
                self.snapshot.project,
                inventory,
                [approved[0], approved[0]],
            )

    def test_relocation_remains_fail_closed(self):
        with self.assertRaises(NotImplementedError):
            self.adapter.relocate_occurrences(self.snapshot.project, (), ())

    def test_validation_rejects_empty_and_control_characters(self):
        occurrence = self.snapshot.occurrences[0]
        empty = self.adapter.validate_translation(occurrence, "")
        self.assertEqual(empty.status, "block")
        self.assertEqual(empty.reason_codes, ("tyrano.translation.empty",))

        control = self.adapter.validate_translation(occurrence, "译文\n下一行")
        self.assertEqual(control.status, "block")
        self.assertEqual(
            control.reason_codes,
            ("tyrano.translation.control_character",),
        )

        valid = self.adapter.validate_translation(occurrence, "有效译文")
        self.assertEqual(valid.status, "pass")
        self.assertTrue(valid.source_constraints_digest)
        self.assertTrue(valid.translation_digest)

    def _validated(self, occurrence, translated_text):
        return ValidatedTranslation(
            occurrence=occurrence,
            translated_text=translated_text,
            validation=self.adapter.validate_translation(occurrence, translated_text),
        )

    def test_catalog_plan_is_semantic_and_does_not_render_ks_files(self):
        occurrence = next(
            item for item in self.snapshot.occurrences if item.unit.text == "Hello, world!"
        )
        plan = self.adapter.build_writeback_plan(
            self.snapshot.project,
            (self._validated(occurrence, "新的问候"),),
            self.snapshot.project.source_documents,
        )
        self.assertEqual(len(plan.operations), 1)
        operation = plan.operations[0]
        self.assertEqual(operation.kind, "json_catalog_set")
        self.assertEqual(operation.target_rel_path, "data/others/lang/ch.json")
        self.assertEqual(
            operation.target_json_path,
            ("scenes", "scene1.ks", "scenario", "Hello, world!"),
        )
        self.assertEqual((operation.line, operation.start_col, operation.end_col), (-1, -1, -1))

        rendered = render_writeback_plan(
            plan,
            self.snapshot.project.source_documents,
        )
        self.assertEqual(set(rendered), {"data/others/lang/ch.json"})
        catalog = json.loads("".join(rendered["data/others/lang/ch.json"]))
        self.assertEqual(
            catalog["scenes"]["scene1.ks"]["scenario"]["Hello, world!"],
            "新的问候",
        )

    def test_duplicate_occurrences_share_one_catalog_operation(self):
        occurrences = [
            item for item in self.snapshot.occurrences if item.unit.text == "akane"
        ]
        self.assertEqual(len(occurrences), 2)
        plan = self.adapter.build_writeback_plan(
            self.snapshot.project,
            tuple(self._validated(item, "朱音") for item in occurrences),
            self.snapshot.project.source_documents,
        )
        self.assertEqual(len(plan.operations), 1)
        self.assertEqual(plan.operations[0].target_json_path, ("charas", "akane"))

        conflicting = (
            self._validated(occurrences[0], "朱音"),
            self._validated(occurrences[1], "茜"),
        )
        with self.assertRaises(WritebackPlanError) as captured:
            self.adapter.build_writeback_plan(
                self.snapshot.project,
                conflicting,
                self.snapshot.project.source_documents,
            )
        self.assertEqual(
            captured.exception.reason_code,
            "tyrano.writeback.catalog_translation_conflict",
        )

    def test_missing_catalog_row_and_incomplete_snapshot_block_planning(self):
        missing_row = next(
            item
            for item in self.snapshot.occurrences
            if item.unit.text == " text before a "
        )
        with self.assertRaises(WritebackPlanError) as captured:
            self.adapter.build_writeback_plan(
                self.snapshot.project,
                (self._validated(missing_row, " 标签前的文本 "),),
                self.snapshot.project.source_documents,
            )
        self.assertEqual(
            captured.exception.reason_code,
            "tyrano.writeback.catalog_row_missing",
        )

        catalog_rel_path = "data/others/lang/ch.json"
        without_catalog = tuple(
            document
            for document in self.snapshot.project.source_documents
            if document.file_rel_path != catalog_rel_path
        )
        existing = next(
            item for item in self.snapshot.occurrences if item.unit.text == "Hello, world!"
        )
        with self.assertRaises(WritebackPlanError) as captured:
            self.adapter.build_writeback_plan(
                self.snapshot.project,
                (self._validated(existing, "新的问候"),),
                without_catalog,
            )
        self.assertEqual(
            captured.exception.reason_code,
            "tyrano.writeback.source_snapshot_mismatch",
        )

    def test_changed_catalog_snapshot_blocks_render(self):
        occurrence = next(
            item for item in self.snapshot.occurrences if item.unit.text == "Hello, world!"
        )
        plan = self.adapter.build_writeback_plan(
            self.snapshot.project,
            (self._validated(occurrence, "新的问候"),),
            self.snapshot.project.source_documents,
        )
        changed_documents = tuple(
            replace(document, sha256="0" * 64)
            if document.file_rel_path == "data/others/lang/ch.json"
            else document
            for document in self.snapshot.project.source_documents
        )
        with self.assertRaises(WritebackPlanError) as captured:
            render_writeback_plan(plan, changed_documents)
        self.assertEqual(
            captured.exception.reason_code,
            "common.writeback.source_snapshot_mismatch",
        )

    def test_common_renderer_rejects_missing_signed_catalog_path(self):
        occurrence = next(
            item for item in self.snapshot.occurrences if item.unit.text == "Hello, world!"
        )
        plan = self.adapter.build_writeback_plan(
            self.snapshot.project,
            (self._validated(occurrence, "新的问候"),),
            self.snapshot.project.source_documents,
        )
        changed_operation = replace(
            plan.operations[0],
            operation_id="",
            target_json_path=("scenes", "scene1.ks", "scenario", "missing"),
        )
        operation_payload = changed_operation.to_dict()
        operation_payload.pop("operation_id")
        changed_operation = replace(
            changed_operation,
            operation_id="op1:" + digest_json(operation_payload),
        )
        changed_plan = replace(plan, operations=(changed_operation,), plan_digest="")
        plan_payload = changed_plan.to_dict()
        plan_payload.pop("plan_digest")
        changed_plan = replace(changed_plan, plan_digest=digest_json(plan_payload))

        with self.assertRaises(WritebackPlanError) as captured:
            render_writeback_plan(
                changed_plan,
                self.snapshot.project.source_documents,
            )
        self.assertEqual(
            captured.exception.reason_code,
            "common.writeback.catalog_path_missing",
        )


class TyranoAdapterNegativeFixtureTests(unittest.TestCase):
    def test_catalog_writeback_preserves_utf8_bom_and_crlf(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_project(
                root,
                "*start|Start\nHello, world!\n",
                catalog={
                    "scenes": {
                        "sample.ks": {
                            "scenario": {"Hello, world!": "你好"},
                            "tag": {},
                        }
                    }
                },
            )
            catalog_path = root / "data" / "others" / "lang" / "ch.json"
            payload = json.dumps(
                {
                    "scenes": {
                        "sample.ks": {
                            "scenario": {"Hello, world!": "你好"},
                            "tag": {},
                        }
                    }
                },
                ensure_ascii=False,
                indent=2,
            ).replace("\n", "\r\n") + "\r\n"
            catalog_path.write_bytes(b"\xef\xbb\xbf" + payload.encode("utf-8"))

            adapter = TyranoAdapter()
            snapshot = build_translation_snapshot(
                adapter,
                ProjectDiscoveryRequest(
                    project_root=str(root),
                    localization_root=str(catalog_path.parent),
                    target_language="ch",
                ),
            )
            occurrence = snapshot.occurrences[0]
            translated = ValidatedTranslation(
                occurrence=occurrence,
                translated_text="新的问候",
                validation=adapter.validate_translation(occurrence, "新的问候"),
            )
            plan = adapter.build_writeback_plan(
                snapshot.project,
                (translated,),
                snapshot.project.source_documents,
            )
            rendered = "".join(
                render_writeback_plan(plan, snapshot.project.source_documents)[
                    "data/others/lang/ch.json"
                ]
            )
            self.assertTrue(rendered.startswith("\ufeff"))
            self.assertIn("\r\n", rendered)

    def test_missing_catalog_blocks_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario_dir = root / "data" / "scenario"
            scenario_dir.mkdir(parents=True)
            (scenario_dir / "missing_catalog.ks").write_text(
                '; comment\n*start|Start\nHello, world!\n',
                encoding="utf-8",
                newline="\n",
            )
            adapter = TyranoAdapter()
            project = adapter.discover_project(
                ProjectDiscoveryRequest(
                    project_root=str(root),
                    localization_root=str(root / "data" / "others" / "lang"),
                    target_language="ch",
                )
            )
            inventory = adapter.inventory_candidates(project, InventoryPolicy())
            draft = adapter.audit_extraction(project, inventory)
            report = build_coverage_report(
                project,
                inventory,
                draft,
                adapter_behavior_digest=adapter.behavior_digest(),
            )
            self.assertEqual(draft.catalog_freshness, "missing")
            self.assertIn("tyrano.catalog.missing_file", draft.reason_codes)
            self.assertEqual(report.coverage_status, "block")

    def test_missing_project_root_fails_closed(self):
        adapter = TyranoAdapter()
        with self.assertRaises(ValueError):
            adapter.discover_project(
                ProjectDiscoveryRequest(
                    project_root="",
                    localization_root="",
                    target_language="ch",
                )
            )

    def test_include_files_filter(self):
        adapter = TyranoAdapter()
        project = adapter.discover_project(
            ProjectDiscoveryRequest(
                project_root=str(FIXTURE_DIR),
                localization_root=str(FIXTURE_DIR / "data" / "others" / "lang"),
                target_language="ch",
                include_files=("data/scenario/scene1.ks",),
            )
        )
        rel_paths = {document.file_rel_path for document in project.source_documents}
        self.assertIn("data/scenario/scene1.ks", rel_paths)
        self.assertNotIn("data/scenario/choices.ks", rel_paths)

    @staticmethod
    def _make_project(root, scenario_text, *, catalog=None, catalog_text=None):
        scenario_dir = root / "data" / "scenario"
        scenario_dir.mkdir(parents=True)
        (scenario_dir / "sample.ks").write_text(
            scenario_text,
            encoding="utf-8",
            newline="\n",
        )
        if catalog is not None or catalog_text is not None:
            lang_dir = root / "data" / "others" / "lang"
            lang_dir.mkdir(parents=True)
            payload = catalog_text
            if catalog_text is None:
                payload = json.dumps(catalog or {}, ensure_ascii=False)
            (lang_dir / "ch.json").write_text(payload, encoding="utf-8", newline="\n")

    @staticmethod
    def _adapter_report(root):
        adapter = TyranoAdapter()
        project = adapter.discover_project(
            ProjectDiscoveryRequest(
                project_root=str(root),
                localization_root=str(root / "data" / "others" / "lang"),
                target_language="ch",
            )
        )
        inventory = adapter.inventory_candidates(project, InventoryPolicy())
        draft = adapter.audit_extraction(project, inventory)
        report = build_coverage_report(
            project,
            inventory,
            draft,
            adapter_behavior_digest=adapter.behavior_digest(),
        )
        return draft, report, inventory

    def test_missing_row_blocks_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_project(
                root,
                "*start|Start\nHello, world!\n",
                catalog={"scenes": {"sample.ks": {"scenario": {}, "tag": {}}}},
            )
            draft, report, inventory = self._adapter_report(root)
            self.assertIn("tyrano.catalog.missing_row", draft.reason_codes)
            self.assertEqual(report.coverage_status, "block")

    def test_empty_translation_blocks_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_project(
                root,
                "*start|Start\nHello, world!\n",
                catalog={
                    "scenes": {
                        "sample.ks": {
                            "scenario": {"Hello, world!": ""},
                            "tag": {},
                        }
                    }
                },
            )
            draft, report, _ = self._adapter_report(root)
            self.assertIn("tyrano.catalog.empty_translation", draft.reason_codes)
            self.assertEqual(report.coverage_status, "block")

    def test_missing_scenario_section_blocks_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_project(
                root,
                "*start|Start\nHello, world!\n",
                catalog={"scenes": {}},
            )
            draft, report, _ = self._adapter_report(root)
            self.assertIn("tyrano.catalog.missing_scenario", draft.reason_codes)
            self.assertEqual(report.coverage_status, "block")

    def test_invalid_json_blocks_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_project(
                root,
                "*start|Start\nHello, world!\n",
                catalog_text="[not json",
            )
            draft, report, _ = self._adapter_report(root)
            self.assertIn("tyrano.catalog.invalid_json", draft.reason_codes)
            self.assertEqual(report.coverage_status, "block")

    def test_stale_extra_catalog_row_reports_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_project(
                root,
                "*start|Start\nHello, world!\n",
                catalog={
                    "scenes": {
                        "sample.ks": {
                            "scenario": {
                                "Hello, world!": "你好",
                                "Text removed from source": "幽灵译文",
                            },
                            "tag": {},
                        }
                    }
                },
            )
            draft, report, _ = self._adapter_report(root)
            self.assertIn("tyrano.catalog.stale", draft.reason_codes)
            self.assertEqual(report.coverage_status, "block")

    def test_registered_tag_with_multiple_parameters_gets_one_candidate_each(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_project(
                root,
                '*start|Start\n[ptext text="Hello" hint="World"]\n',
                catalog={
                    "tags": {"ptext": ["text", "hint"]},
                    "scenes": {
                        "sample.ks": {
                            "scenario": {},
                            "tag": {"ptext": {"text": {"Hello": "你好"}, "hint": {"World": "世界"}}},
                        }
                    },
                },
            )
            _, _, inventory = self._adapter_report(root)
            tag_candidates = [
                candidate
                for candidate in inventory.candidates
                if candidate.structure_kind == "tag"
                and candidate.locator.locator.get("name") == "ptext"
            ]
            self.assertEqual(len(tag_candidates), 2, [c.raw_excerpt for c in tag_candidates])
            self.assertEqual(
                {candidate.raw_excerpt for candidate in tag_candidates},
                {"Hello", "World"},
            )
            for candidate in tag_candidates:
                self.assertIn(
                    candidate.locator.locator.get("param_name"),
                    {"text", "hint"},
                )
                self.assertEqual(candidate.classification, "already_translated")

    def test_target_language_path_escape_is_rejected(self):
        adapter = TyranoAdapter()
        with self.assertRaises(ValueError):
            adapter.discover_project(
                ProjectDiscoveryRequest(
                    project_root=str(FIXTURE_DIR),
                    localization_root=str(FIXTURE_DIR / "data" / "others" / "lang"),
                    target_language="../../secret",
                )
            )

    def test_uppercase_ks_extension_is_inventoried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario_dir = root / "data" / "scenario"
            scenario_dir.mkdir(parents=True)
            (scenario_dir / "SCENE.KS").write_text(
                "*start|Start\nHello, world!\n",
                encoding="utf-8",
                newline="\n",
            )
            lang_dir = root / "data" / "others" / "lang"
            lang_dir.mkdir(parents=True)
            (lang_dir / "ch.json").write_text(
                json.dumps(
                    {
                        "scenes": {
                            "SCENE.KS": {
                                "scenario": {"Hello, world!": "你好"},
                                "tag": {},
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            _, report, inventory = self._adapter_report(root)
            self.assertEqual(report.coverage_status, "attention")
            self.assertEqual(len(inventory.candidates), 2)

    def test_keep_space_setting_accepts_optional_semicolon(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data" / "system").mkdir(parents=True)
            (root / "data" / "system" / "Config.tjs").write_text(
                "KeepSpaceInParameterValue = 1;\n",
                encoding="utf-8",
                newline="\n",
            )
            self.assertEqual(_read_keep_space_setting(root), "1")

    def test_default_tag_registry_survives_project_tags(self):
        merged = _merged_tag_registry(
            {"tags": {"mymacro": ["value"]}}
        )
        self.assertEqual(merged["glink"], ("text",))
        self.assertEqual(merged["ptext"], ("text",))
        self.assertEqual(merged["mymacro"], ("value",))

    def test_parse_tyrano_scenario_normalizes_keep_space_level(self):
        lines = ['[ptext layer=0 x=10 text="  Keep  inner " ]']
        result = parse_tyrano_scenario(lines, keep_space="1")
        ptext = next(node for node in result.nodes if node.name == "ptext")
        self.assertEqual(ptext.pm["text"], "Keepinner")
        result2 = parse_tyrano_scenario(lines, keep_space="2")
        ptext2 = next(node for node in result2.nodes if node.name == "ptext")
        self.assertEqual(ptext2.pm["text"], "Keep  inner")
        result3 = parse_tyrano_scenario(lines, keep_space="3")
        ptext3 = next(node for node in result3.nodes if node.name == "ptext")
        self.assertEqual(ptext3.pm["text"], "  Keep  inner ")

        # Legal space padding around ``=`` is not a parse error.
        no_error = parse_tyrano_scenario(['[ptext text = "Hello"]'])
        self.assertEqual(no_error.parse_errors, ())
        self.assertEqual(no_error.nodes[-1].pm["text"], "Hello")

        # Backtick-quoted values preserve interior spaces even at KeepSpace 1.
        backtick = parse_tyrano_scenario(["[ptext text=` keep  inner `]"], keep_space="1")
        self.assertEqual(backtick.nodes[-1].pm["text"], "keep  inner")


if __name__ == "__main__":
    unittest.main()
