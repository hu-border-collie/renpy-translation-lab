# -*- coding: utf-8 -*-
"""Coverage reason-code contract tests for the Ren'Py and Tyrano adapters."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
import re
import tempfile
import unittest

from engine_adapters.contracts import (
    CandidateInventory,
    CoverageReportDraft,
    ProjectDiscoveryRequest,
)
from engine_adapters.coverage import (
    CANDIDATE_REASON_CODES,
    REPORT_REASON_CODES,
    build_coverage_report,
)
from engine_adapters.renpy import (
    RenPyAdapter,
    build_translation_snapshot as build_renpy_translation_snapshot,
)
from engine_adapters.tyrano import (
    TyranoAdapter,
    build_translation_snapshot as build_tyrano_translation_snapshot,
)
import translator_runtime as runtime


REASON_CODE_PATTERN = re.compile(r"^(renpy|tyrano|coverage|project)\.[a-z][a-z0-9_.]*$")
REASON_COLLECTION_NAMES = {"reason_code", "reason_codes", "reasons", "report_reasons"}
INVENTORY_ROOTS = {"inventory_candidates", "audit_extraction"}

# These functions emit validation/writeback diagnostics, not candidate or
# coverage-report reasons. Keep the exclusion explicit so a new diagnostic
# does not get mistaken for an inventory/audit reason by this test.
EXCLUDED_REASON_FUNCTIONS = {
    "renpy.py": {
        "_validation_reason_codes",
        "validate_translation",
        "build_writeback_plan",
    },
    "tyrano.py": {
        "validate_translation",
        "_catalog_json_path",
        "_catalog_value_at_path",
        "build_writeback_plan",
    },
}

VALIDATION_WRITEBACK_PREFIXES = (
    "renpy.placeholder.",
    "renpy.string_literal.",
    "renpy.tag.",
    "renpy.field.",
    "renpy.percent_token.",
    "tyrano.translation.",
    "tyrano.writeback.",
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tyranoscript_v600"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _literal_reason_values(node: ast.AST) -> list[str]:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and REASON_CODE_PATTERN.fullmatch(node.value)
    ):
        return [node.value]
    if not isinstance(node, (ast.List, ast.Tuple)):
        return []
    return [
        element.value
        for element in node.elts
        if isinstance(element, ast.Constant)
        and isinstance(element.value, str)
        and REASON_CODE_PATTERN.fullmatch(element.value)
    ]


def _function_nodes(tree: ast.AST) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _called_function_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    called: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
    return called


def _reachable_reason_functions(
    tree: ast.AST,
    *,
    excluded: set[str],
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    functions = _function_nodes(tree)
    reachable: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    pending = list(INVENTORY_ROOTS)
    while pending:
        name = pending.pop()
        if name in reachable or name in excluded:
            continue
        function = functions.get(name)
        if function is None:
            continue
        reachable[name] = function
        pending.extend(_called_function_names(function))
    return reachable


def _collect_reason_literals(
    path: Path,
    *,
    excluded: set[str],
) -> tuple[tuple[str, str, int], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = _reachable_reason_functions(tree, excluded=excluded)
    found: set[tuple[str, str, int]] = set()

    for function_name, function in functions.items():
        for node in ast.walk(function):
            values: list[str] = []
            if isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"append", "extend"}
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in REASON_COLLECTION_NAMES
                ):
                    for argument in node.args:
                        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                            values.extend(
                                value
                                for value in [argument.value]
                                if REASON_CODE_PATTERN.fullmatch(value)
                            )
                        else:
                            values.extend(_literal_reason_values(argument))
                for keyword in node.keywords:
                    if keyword.arg in REASON_COLLECTION_NAMES:
                        values.extend(_literal_reason_values(keyword.value))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(
                    isinstance(target, ast.Name) and target.id in REASON_COLLECTION_NAMES
                    for target in targets
                ):
                    values.extend(
                        _literal_reason_values(
                            node.value,
                        )
                    )
            elif isinstance(node, ast.Return):
                values.extend(_literal_reason_values(node.value))

            for value in values:
                found.add((value, function_name, node.lineno))

    return tuple(sorted(found, key=lambda item: (item[0], item[1], item[2])))


class CoverageReasonCodeContractTests(unittest.TestCase):
    @staticmethod
    def _renpy_request(root: Path, localization_root: Path) -> ProjectDiscoveryRequest:
        return ProjectDiscoveryRequest(
            project_root=str(root),
            localization_root=str(localization_root),
            target_language="schinese",
        )

    def _make_renpy_project(self, files: dict[str, str]) -> tuple[tempfile.TemporaryDirectory, Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        localization_root = root / "game" / "tl" / "schinese"
        localization_root.mkdir(parents=True)
        for relative_path, text in files.items():
            target = localization_root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        return temporary, root, localization_root

    @staticmethod
    def _assert_snapshot_reason_codes_are_allowed(
        testcase: unittest.TestCase,
        snapshot,
    ) -> None:
        candidate_codes = {
            code
            for candidate in snapshot.inventory.candidates
            for code in candidate.reason_codes
        }
        report_codes = set(snapshot.report.audit_reason_codes)
        testcase.assertTrue(
            candidate_codes,
            "fixture must exercise at least one candidate reason code",
        )
        testcase.assertTrue(
            candidate_codes <= CANDIDATE_REASON_CODES,
            sorted(candidate_codes - CANDIDATE_REASON_CODES),
        )
        testcase.assertTrue(
            report_codes <= CANDIDATE_REASON_CODES | REPORT_REASON_CODES,
            sorted(report_codes - CANDIDATE_REASON_CODES - REPORT_REASON_CODES),
        )
        testcase.assertEqual(
            snapshot.report.invariant_errors,
            (),
            "fixture must not block only because a reason code is unregistered",
        )
        testcase.assertEqual(
            snapshot.report.reason_counts.get("coverage.inventory.invalid_candidate", 0),
            0,
            snapshot.report.to_dict(),
        )

    def test_unregistered_candidate_reason_blocks_coverage(self):
        temporary, root, localization_root = self._make_renpy_project(
            {"script.rpy": 'translate schinese start:\n    "Hello"\n'}
        )
        self.addCleanup(temporary.cleanup)
        adapter = RenPyAdapter(legacy_module=runtime)
        snapshot = build_renpy_translation_snapshot(
            adapter,
            self._renpy_request(root, localization_root),
        )
        self.assertTrue(snapshot.inventory.candidates)

        invalid_candidate = replace(
            snapshot.inventory.candidates[0],
            reason_codes=("renpy.not_registered_for_test",),
        )
        invalid_inventory: CandidateInventory = replace(
            snapshot.inventory,
            candidates=(invalid_candidate, *snapshot.inventory.candidates[1:]),
        )
        report = build_coverage_report(
            snapshot.project,
            invalid_inventory,
            CoverageReportDraft(source_fingerprint=snapshot.project.source_fingerprint),
            adapter_behavior_digest=adapter.behavior_digest(),
        )

        self.assertTrue(report.invariant_errors)
        self.assertEqual(report.coverage_status, "block")
        self.assertIn(
            "coverage.inventory.invalid_candidate",
            report.reason_counts,
        )

    def test_renpy_fixture_inventory_and_report_reasons_are_allowlisted(self):
        files = {
            "dialogue.rpy": (
                'translate schinese chapter:\n'
                '    # e "Hello {player}!"\n'
                '    e "你好，[player]！"\n'
                '    # "Terry" "Hello there."\n'
                '    "Terry" "你好。"\n'
            ),
            "empty_target.rpy": (
                'translate schinese empty:\n'
                '    # e "Empty target."\n'
                '    e ""\n'
            ),
            "old_new.rpy": (
                'translate schinese strings:\n'
                '    old "Start game"\n'
                '    new "开始游戏"\n'
            ),
            "tokenize_error.rpy": (
                'translate schinese broken:\n'
                '    # "Dangling source"\n'
                '    old "Old without new"\n'
                '    text f"Dynamic {name}"\n'
                '    "unterminated\n'
            ),
        }
        temporary, root, localization_root = self._make_renpy_project(files)
        self.addCleanup(temporary.cleanup)

        snapshot = build_renpy_translation_snapshot(
            RenPyAdapter(legacy_module=runtime),
            self._renpy_request(root, localization_root),
        )
        self._assert_snapshot_reason_codes_are_allowed(self, snapshot)
        candidate_codes = {
            code
            for candidate in snapshot.inventory.candidates
            for code in candidate.reason_codes
        }
        self.assertIn("renpy.speaker_label_sibling_translated", candidate_codes)
        self.assertIn("renpy.empty_target", candidate_codes)
        self.assertIn("renpy.old_new_pair", candidate_codes)
        self.assertIn("renpy.tokenize_error", candidate_codes)

    def test_tyrano_fixture_inventory_and_report_reasons_are_allowlisted(self):
        snapshot = build_tyrano_translation_snapshot(
            TyranoAdapter(),
            ProjectDiscoveryRequest(
                project_root=str(FIXTURE_DIR),
                localization_root=str(FIXTURE_DIR / "data" / "others" / "lang"),
                target_language="ch",
            ),
        )
        self._assert_snapshot_reason_codes_are_allowed(self, snapshot)

    def test_inventory_and_audit_reason_literals_are_allowlisted(self):
        allowed = CANDIDATE_REASON_CODES | REPORT_REASON_CODES
        expected_codes = {
            "renpy.speaker_label_sibling_translated",
            "renpy.catalog.provenance_unknown",
            "tyrano.text_node",
            "tyrano.catalog.missing_file",
        }
        observed: set[str] = set()
        details: list[str] = []

        for filename in ("renpy.py", "tyrano.py"):
            path = REPO_ROOT / "engine_adapters" / filename
            records = _collect_reason_literals(
                path,
                excluded=EXCLUDED_REASON_FUNCTIONS[filename],
            )
            self.assertTrue(records, filename)
            for code, function_name, line_number in records:
                observed.add(code)
                details.append(f"{filename}:{line_number}:{function_name}:{code}")

        self.assertTrue(expected_codes <= observed, sorted(expected_codes - observed))
        self.assertFalse(
            observed - allowed,
            "unregistered inventory/audit reason literals:\n" + "\n".join(details),
        )

    def test_validation_and_writeback_diagnostics_stay_out_of_candidate_allowlist(self):
        self.assertFalse(
            [
                code
                for code in CANDIDATE_REASON_CODES
                if code.startswith(VALIDATION_WRITEBACK_PREFIXES)
            ]
        )


if __name__ == "__main__":
    unittest.main()
