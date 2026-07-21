"""Pure unit tests for EUI-style registry table column presets."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from gui_qt.games_registry_table import (
    REGISTRY_PREF_TABLE_COLUMN_WIDTHS,
    REGISTRY_PREF_TABLE_COLUMN_WIDTHS_LEGACY,
    REGISTRY_TABLE_COLUMN_DEFS,
    REGISTRY_TABLE_PATH_COLUMN,
    clamp_width_for_fit,
    column_headers,
    default_width_map,
    flex_column_index,
    interactive_column_ids,
    migrate_stored_widths,
    min_width_for_column,
    row_cell_values,
    widths_for_persist,
)


class GamesRegistryTablePresetsTests(unittest.TestCase):
    def test_path_is_last_flex_column(self) -> None:
        self.assertEqual(flex_column_index(), len(REGISTRY_TABLE_COLUMN_DEFS) - 1)
        self.assertEqual(REGISTRY_TABLE_PATH_COLUMN, flex_column_index())
        self.assertTrue(REGISTRY_TABLE_COLUMN_DEFS[-1].flex)
        self.assertEqual(REGISTRY_TABLE_COLUMN_DEFS[-1].id, "path")
        self.assertEqual(
            column_headers(),
            ["项目", "目录状态", "版本", "游玩", "翻译", "路径"],
        )
        self.assertEqual(
            interactive_column_ids(),
            ["name", "layout", "version", "play", "translation"],
        )
        self.assertEqual(REGISTRY_PREF_TABLE_COLUMN_WIDTHS, "table_column_widths")
        self.assertEqual(
            REGISTRY_PREF_TABLE_COLUMN_WIDTHS_LEGACY,
            "table_column_width_fractions",
        )

    def test_row_cell_values_follow_column_order(self) -> None:
        row = SimpleNamespace(
            name="Demo",
            layout_status="就绪",
            version="1.0",
            play_status="进行中",
            translation_status="翻译中",
            path="Game_Demo",
        )
        self.assertEqual(
            row_cell_values(row),
            ["Demo", "就绪", "1.0", "进行中", "翻译中", "Game_Demo"],
        )

    def test_min_width_covers_header_and_longest_enum_sample(self) -> None:
        layout = next(c for c in REGISTRY_TABLE_COLUMN_DEFS if c.id == "layout")

        def advance(text: str) -> int:
            # CJK-ish: count chars * 12 to approximate pixel advance.
            return len(text) * 12

        min_w = min_width_for_column(layout, advance)
        longest = max(
            (layout.title, *layout.enum_samples),
            key=lambda s: advance(s),
        )
        self.assertGreaterEqual(min_w, advance(longest) + 28)
        self.assertIn("建议使用 work", layout.enum_samples)

    def test_migrate_id_map_and_legacy_index_map(self) -> None:
        defaults = default_width_map()
        migrated_ids = migrate_stored_widths({"name": 200, "layout": 140, "path": 999})
        self.assertEqual(migrated_ids["name"], 200)
        self.assertEqual(migrated_ids["layout"], 140)
        self.assertNotIn("path", migrated_ids)
        self.assertEqual(migrated_ids["version"], defaults["version"])

        # Legacy index → px (old order: name, path, version, layout, play, translation).
        legacy = migrate_stored_widths(
            {"0": 150, "1": 400, "2": 90, "3": 130, "4": 80, "5": 110}
        )
        self.assertEqual(legacy["name"], 150)
        self.assertEqual(legacy["version"], 90)
        self.assertEqual(legacy["layout"], 130)
        self.assertEqual(legacy["play"], 80)
        self.assertEqual(legacy["translation"], 110)
        self.assertNotIn("path", legacy)

    def test_migrate_legacy_fraction_list(self) -> None:
        fracs = [0.2, 0.3, 0.1, 0.15, 0.1, 0.15]
        migrated = migrate_stored_widths(fracs)
        self.assertEqual(migrated["name"], int(960 * 0.2))
        self.assertEqual(migrated["version"], int(960 * 0.1))
        self.assertNotIn("path", migrated)

    def test_widths_for_persist_drops_flex_and_unknown(self) -> None:
        payload = widths_for_persist(
            {"name": 160, "path": 300, "nope": 1, "layout": 120}
        )
        self.assertEqual(payload, {"name": 160, "layout": 120})

    def test_clamp_width_for_fit_respects_max(self) -> None:
        translation = next(c for c in REGISTRY_TABLE_COLUMN_DEFS if c.id == "translation")
        self.assertEqual(
            clamp_width_for_fit(translation, 500, min_width=80),
            translation.max_width,
        )
        self.assertEqual(
            clamp_width_for_fit(translation, 50, min_width=80),
            80,
        )


if __name__ == "__main__":
    unittest.main()
