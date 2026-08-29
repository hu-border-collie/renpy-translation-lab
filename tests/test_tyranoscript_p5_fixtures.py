# -*- coding: utf-8 -*-
"""Offline fixture characterization for #265 P5 (TyranoScript V600+).

The fixture directory contains a hand-written TyranoScript project, a
native ``data/others/lang/<lang>.json`` catalog following the official
TyranoStudio V603 shape, frozen parser output from the official V602c
runtime parser, and a hand-curated candidate inventory that the future
``TyranoAdapter`` must satisfy.

These tests are intentionally adapter-free.  They keep the fixture and the
hand-curated expectations internally consistent so the P5 implementation can
start from a trustworthy golden baseline.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tyranoscript_v600"
SCENARIO_DIR = FIXTURE_DIR / "data" / "scenario"
CATALOG_PATH = FIXTURE_DIR / "data" / "others" / "lang" / "ch.json"
PARSER_NODES_PATH = FIXTURE_DIR / "expected" / "parser_nodes.json"
INVENTORY_PATH = FIXTURE_DIR / "expected" / "inventory.json"
NEGATIVE_CASES_PATH = FIXTURE_DIR / "negative_cases.json"

SCENARIO_FILES = ("scene1.ks", "choices.ks", "broken.ks")

CLASSIFICATIONS = frozenset(
    {
        "translatable",
        "already_translated",
        "explicitly_excluded",
        "unsupported",
        "parse_error",
        "unknown",
    }
)

REASON_CODES = frozenset(
    {
        "tyrano.comment",
        "tyrano.engine_control_structure",
        "tyrano.character_definition",
        "tyrano.chara_ptext",
        "tyrano.text_node",
        "tyrano.registered_tag_parameter",
        "tyrano.tag_parameter_not_registered",
        "tyrano.unregistered_macro_invocation",
        "tyrano.dynamic_parameter_expression",
        "tyrano.iscript_boundary_tag",
        "tyrano.iscript_content",
        "tyrano.lang_set_control_tag",
        "tyrano.unterminated_quoted_parameter",
        "tyrano.official_parser_compensated",
        "tyrano.unquoted_parameter_sequence",
        "tyrano.unclosed_inline_tag",
    }
)


def _read_text(rel_path):
    return (FIXTURE_DIR / rel_path).read_text(encoding="utf-8")


def _read_json(rel_path):
    return json.loads((FIXTURE_DIR / rel_path).read_text(encoding="utf-8"))


def _source_lines(file_name):
    return (SCENARIO_DIR / file_name).read_text(encoding="utf-8").splitlines()


def _nodes_by_file():
    return _read_json("expected/parser_nodes.json")["files"]


def _catalog():
    return _read_json("data/others/lang/ch.json")


def _inventory():
    return _read_json("expected/inventory.json")

def _official_comment_line_indexes(lines):
    """Mirror the official parser's full-line comment recognition."""
    indexes = set()
    in_block_comment = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if in_block_comment:
            indexes.add(index)
            if stripped == "*/":
                in_block_comment = False
        elif stripped.startswith(";"):
            indexes.add(index)
        elif stripped == "/*":
            indexes.add(index)
            in_block_comment = True
    return indexes


class FixtureLayoutTests(unittest.TestCase):
    def test_fixture_files_are_utf8_without_crlf(self):
        for path in sorted(FIXTURE_DIR.rglob("*")):
            if not path.is_file():
                continue
            data = path.read_bytes()
            self.assertNotIn(b"\r\n", data, path)
            data.decode("utf-8")

    def test_catalog_target_language_matches_static_lang_set(self):
        catalog_name = CATALOG_PATH.name
        self.assertEqual(catalog_name, "ch.json")
        scene_lines = _source_lines("scene1.ks")
        self.assertIn('[lang_set name="ch" ]', scene_lines)

    def test_config_freezes_official_keep_space_setting(self):
        config = _read_text("data/system/Config.tjs")
        self.assertIn(";KeepSpaceInParameterValue = 2;", config)
        self.assertIn(";projectID = tyrano_p5_fixture;", config)


class ParserNodeCharacterizationTests(unittest.TestCase):
    """Freeze observations from the official V602c runtime parser."""

    def test_parser_nodes_cover_every_scenario_file_with_sequential_indexes(self):
        nodes_by_file = _nodes_by_file()
        self.assertEqual(set(nodes_by_file), set(SCENARIO_FILES))
        for file_name in SCENARIO_FILES:
            lines = _source_lines(file_name)
            nodes = nodes_by_file[file_name]
            self.assertEqual(
                [node["node_index"] for node in nodes],
                list(range(len(nodes))),
                file_name,
            )
            for node in nodes:
                self.assertGreaterEqual(node["line"], 0, node)
                self.assertLess(node["line"], len(lines), node)
                self.assertIsInstance(node["name"], str, node)
                self.assertIsInstance(node["pm"], dict, node)
                self.assertIsInstance(node["val"], str, node)

    def test_official_parser_preserves_quoted_spaces_and_escaped_quote(self):
        nodes_by_file = _nodes_by_file()
        scene1 = {node["node_index"]: node for node in nodes_by_file["scene1.ks"]}
        choices = {node["node_index"]: node for node in nodes_by_file["choices.ks"]}
        self.assertEqual(scene1[8]["pm"]["text"], "Start Game")
        self.assertEqual(choices[6]["pm"]["text"], "It's fine")

    def test_official_parser_splits_text_around_inline_tags(self):
        nodes_by_file = _nodes_by_file()
        scene1_texts = [
            node["pm"]["val"]
            for node in nodes_by_file["scene1.ks"]
            if node["name"] == "text"
        ]
        self.assertEqual(
            scene1_texts,
            ["Hello, world!", "A line with ", " inline markup."],
        )
        choices_line3 = [
            (node["name"], node["pm"].get("val", ""))
            for node in nodes_by_file["choices.ks"]
            if node["line"] == 3
        ]
        self.assertEqual(
            choices_line3,
            [
                ("text", "Plain "),
                ("nw", ""),
                ("text", " text before a "),
                ("glink", ""),
                ("text", " choice."),
            ],
        )

    def test_official_parser_is_permissive_on_broken_lines(self):
        nodes_by_file = _nodes_by_file()
        broken = nodes_by_file["broken.ks"]
        ptext = next(node for node in broken if node["name"] == "ptext")
        unknown_macro = next(
            node for node in broken if node["name"] == "unknown_macro"
        )
        unclosed_text = next(
            node
            for node in broken
            if node["name"] == "text" and node["line"] == 7
        )
        unclosed_glink = next(
            node
            for node in broken
            if node["name"] == "glink" and node["line"] == 7
        )
        # The official parser silently compensates the missing quote.
        self.assertEqual(ptext["pm"]["text"], "Unterminated param")
        # Two unquoted tokens become two empty parameters instead of an error.
        self.assertEqual(unknown_macro["pm"], {"value": "", "unquoted": ""})
        # An unclosed inline tag is still emitted without a warning.
        self.assertEqual(unclosed_text["pm"]["val"], "A line with an unclosed ")
        self.assertEqual(unclosed_glink["pm"]["text"], "oops")
        # Full-line block comments disappear from parser output.
        broken_lines = _source_lines("broken.ks")
        self.assertEqual(broken_lines[10], "text inside block comment")
        self.assertFalse(
            any(node["line"] in (9, 10, 11) for node in broken),
        )

    def test_iscript_text_is_still_parser_visible_but_inventory_excludes_it(self):
        nodes_by_file = _nodes_by_file()
        choices = nodes_by_file["choices.ks"]
        self.assertTrue(any(node["name"] == "iscript" for node in choices))
        self.assertTrue(any(node["name"] == "endscript" for node in choices))
        self.assertEqual(
            [node["pm"]["val"] for node in choices if node["name"] == "text" and node["line"] >= 13],
            [
                "// This JavaScript string is intentionally not a candidate.",
                'var notice = "Not a candidate";',
            ],
        )
        inventory_by_key = {
            (item["file_rel_path"], item["line"]): item
            for item in _inventory()["candidates"]
        }
        for line in (14, 15):
            item = inventory_by_key[("choices.ks", line)]
            self.assertEqual(item["classification"], "explicitly_excluded")
            self.assertIn("tyrano.iscript_content", item["reason_codes"])


class ExpectedInventoryTests(unittest.TestCase):
    def test_inventory_schema_and_reason_code_vocabulary(self):
        inventory = _inventory()
        self.assertEqual(inventory["schema_version"], 1)
        self.assertEqual(inventory["engine"], "tyrano")
        self.assertEqual(inventory["target_language"], "ch")
        self.assertEqual(
            set(inventory["classification_vocabulary"]),
            CLASSIFICATIONS,
        )
        self.assertEqual(set(inventory["reason_code_vocabulary"]), REASON_CODES)
        candidate_ids = [item["candidate_id"] for item in inventory["candidates"]]
        self.assertEqual(len(candidate_ids), len(set(candidate_ids)))
        for item in inventory["candidates"]:
            self.assertIn(item["file_rel_path"], SCENARIO_FILES)
            self.assertIn(item["classification"], CLASSIFICATIONS)
            self.assertTrue(
                set(item["reason_codes"]).issubset(REASON_CODES),
                item,
            )
            self.assertIn(item["structure_kind"], {"comment", "label", "text", "tag", "chara_ptext"})

    def test_every_parser_node_and_comment_line_has_exactly_one_candidate(self):
        nodes_by_file = _nodes_by_file()
        inventory = _inventory()
        by_file = {file_name: [] for file_name in SCENARIO_FILES}
        for item in inventory["candidates"]:
            by_file[item["file_rel_path"]].append(item)

        for file_name in SCENARIO_FILES:
            lines = _source_lines(file_name)
            node_to_candidate = {
                item["node_index"]: item
                for item in by_file[file_name]
                if item["node_index"] is not None
            }
            nodes = nodes_by_file[file_name]
            self.assertEqual(
                set(node_to_candidate),
                {node["node_index"] for node in nodes},
                file_name,
            )
            comment_candidates = [
                item
                for item in by_file[file_name]
                if item["node_index"] is None
            ]
            for item in comment_candidates:
                self.assertEqual(item["structure_kind"], "comment", item)
                self.assertEqual(
                    lines[item["line"]],
                    item["source_value"],
                    item,
                )

    def test_candidate_source_values_match_parser_normalized_values(self):
        nodes_by_file = _nodes_by_file()
        for item in _inventory()["candidates"]:
            node_index = item["node_index"]
            if node_index is None:
                continue
            node = nodes_by_file[item["file_rel_path"]][node_index]
            self.assertEqual(node["node_index"], node_index, item)
            self.assertEqual(node["line"], item["line"], item)
            structure_kind = item["structure_kind"]
            if structure_kind == "text":
                self.assertEqual(node["name"], "text", item)
                self.assertEqual(item["source_value"], node["pm"]["val"], item)
            elif structure_kind == "chara_ptext":
                self.assertEqual(node["name"], "chara_ptext", item)
                self.assertEqual(item["source_value"], node["pm"]["name"], item)
            elif structure_kind == "label":
                self.assertEqual(node["name"], "label", item)
                self.assertEqual(item["source_value"], node["pm"]["val"], item)
            else:
                self.assertEqual(node["name"], item["evidence"]["parser_name"], item)

    def test_catalog_links_point_to_existing_matching_rows(self):
        catalog = _catalog()
        for item in _inventory()["candidates"]:
            link = item.get("catalog")
            if link is None:
                self.assertNotEqual(
                    item["classification"],
                    "already_translated",
                    item,
                )
                continue
            self.assertEqual(item["classification"], "already_translated", item)
            value = catalog
            for part in link["path"]:
                self.assertIn(part, value, item)
                value = value[part]
            # Catalog paths always end with the source key; traversing the
            # final component yields the target-language value.
            self.assertEqual(link["path"][-1], item["source_value"], item)
            self.assertEqual(link["translation"], value, item)

    def test_every_source_catalog_row_is_referenced_by_an_expected_candidate(self):
        catalog = _catalog()
        candidates = _inventory()["candidates"]
        referenced_paths = {
            tuple(item["catalog"]["path"])
            for item in candidates
            if item.get("catalog")
        }
        # Scenario text rows.
        for scene, scene_catalog in catalog["scenes"].items():
            for source_value in scene_catalog["scenario"]:
                self.assertIn(
                    ("scenes", scene, "scenario", source_value),
                    referenced_paths,
                )
            # Registered tag rows.
            for tag_name, params in scene_catalog.get("tag", {}).items():
                for param_name, rows in params.items():
                    for source_value in rows:
                        self.assertIn(
                            ("scenes", scene, "tag", tag_name, param_name, source_value),
                            referenced_paths,
                        )
        # Character rows.
        for chara_name in catalog["charas"]:
            self.assertIn(("charas", chara_name), referenced_paths)

    def test_intentionally_pending_text_row_has_no_catalog_entry(self):
        catalog = _catalog()
        self.assertNotIn(
            " text before a ",
            catalog["scenes"]["choices.ks"]["scenario"],
        )
        pending = next(
            item
            for item in _inventory()["candidates"]
            if item["candidate_id"] == "tyrano:choices.ks:line3:node3:text"
        )
        self.assertEqual(pending["classification"], "translatable")
        self.assertIsNone(pending["catalog"])

    def test_classification_contract_covers_all_required_negative_shapes(self):
        candidates = _inventory()["candidates"]
        def one(candidate_id):
            return next(
                item for item in candidates if item["candidate_id"] == candidate_id
            )

        self.assertEqual(
            one("tyrano:scene1.ks:line9:node6:ruby")["classification"],
            "unsupported",
        )
        self.assertEqual(
            one("tyrano:choices.ks:line9:node8:unknown_macro")["classification"],
            "unknown",
        )
        self.assertEqual(
            one("tyrano:choices.ks:line11:node9:ptext_dynamic")["classification"],
            "unsupported",
        )
        for broken_id in (
            "tyrano:broken.ks:line3:node1:ptext",
            "tyrano:broken.ks:line5:node2:unknown_macro",
            "tyrano:broken.ks:line7:node3:text",
            "tyrano:broken.ks:line7:node4:glink",
        ):
            self.assertEqual(one(broken_id)["classification"], "parse_error")

    def test_catalog_shape_matches_official_studio_v603(self):
        catalog = _catalog()
        self.assertEqual(
            set(catalog),
            {"scenes", "charas", "systems", "tags"},
        )
        self.assertEqual(
            catalog["tags"],
            {"glink": ["text"], "ptext": ["text"], "mymacro": ["value"]},
        )
        self.assertEqual(catalog["systems"], {"go_title": "返回标题？"})
        self.assertEqual(catalog["charas"], {"akane": "茜"})
        # ``broken.ks`` intentionally has no catalog section: every parser node
        # on that file is classified ``parse_error`` and must not be written.
        self.assertEqual(
            set(catalog["scenes"]),
            {"scene1.ks", "choices.ks"},
        )


class NegativeMutationContractTests(unittest.TestCase):
    def test_every_negative_mutation_targets_a_real_baseline_row(self):
        catalog = _catalog()
        cases = _read_json("negative_cases.json")["cases"]
        self.assertTrue(cases)
        for case in cases:
            for mutation in case["mutations"]:
                path = mutation["path"]
                value = catalog
                exists = True
                for part in path:
                    if part not in value:
                        exists = False
                        break
                    value = value[part]
                if mutation["action"] == "remove_path":
                    self.assertTrue(exists, path)
                else:
                    self.assertEqual(mutation["action"], "set_path")
                    self.assertIn("value", mutation)
                    # set_path targets either an existing row (mutate) or a
                    # deliberately absent row (stale injection); both must be
                    # resolvable by future tests through the path parent.
                    if path and path[:-1]:
                        parent = catalog
                        for part in path[:-1]:
                            self.assertIn(part, parent, path)
                            parent = parent[part]


if __name__ == "__main__":
    unittest.main()
