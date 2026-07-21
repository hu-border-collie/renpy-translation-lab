"""Tests for single-game ingest into Game_* layout + registry."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import game_ingest
import games_registry as registry


def _write_min_renpy_install(root: Path, *, title: str = "Demo") -> None:
    game = root / "game"
    game.mkdir(parents=True)
    (game / "options.rpy").write_text(
        f'define config.name = _("{title}")\ndefine config.version = "1.0"\n',
        encoding="utf-8",
    )
    (root / "Demo.py").write_text("# launcher stub\n", encoding="utf-8")


class GameIngestNamingTests(unittest.TestCase):
    def test_suggest_game_name_from_zip_stem(self):
        path = Path("Glory Hounds-6.7.zip")
        name = game_ingest.suggest_game_name(path)
        self.assertTrue(name)
        self.assertNotIn(".zip", name.lower())

    def test_game_name_to_folder_latin_words(self):
        self.assertEqual(game_ingest.game_name_to_folder("Glory Hounds"), "Game_GloryHounds")
        self.assertEqual(game_ingest.game_name_to_folder("Custom Title"), "Game_CustomTitle")

    def test_game_name_to_folder_strips_existing_prefix(self):
        self.assertEqual(game_ingest.game_name_to_folder("Game_Foo"), "Game_Foo")
        self.assertNotEqual(game_ingest.game_name_to_folder("Game_Foo"), "Game_Game_Foo")

    def test_game_name_to_folder_empty_and_illegal(self):
        self.assertEqual(game_ingest.game_name_to_folder(""), "")
        self.assertEqual(game_ingest.game_name_to_folder("   "), "")
        err = game_ingest.validate_game_name("")
        self.assertTrue(err)

    def test_game_name_to_folder_cjk(self):
        folder = game_ingest.game_name_to_folder("我的游戏")
        self.assertTrue(folder.startswith("Game_"))
        self.assertIn("我的游戏", folder)


class GameIngestMaterializeTests(unittest.TestCase):
    def test_ingest_directory_install_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            source = Path(tmp) / "raw_game"
            _write_min_renpy_install(source, title="Alpha")

            result = game_ingest.ingest_game(
                source=source,
                workspace_root=workspace,
                game_name="Alpha Game",
            )
            self.assertTrue(result.ok, result.message)
            self.assertEqual(result.folder_name, "Game_AlphaGame")
            project_root = workspace / "Game_AlphaGame"
            self.assertTrue((project_root / "original" / "game" / "options.rpy").is_file())
            self.assertTrue((project_root / "work").is_dir())
            self.assertTrue((project_root / "build").is_dir())
            # work should be empty skeleton (no game/)
            self.assertFalse((project_root / "work" / "game").exists())

            data = registry.load_registry(workspace / registry.REGISTRY_FILENAME)
            self.assertEqual(len(data["projects"]), 1)
            self.assertEqual(data["projects"][0]["name"], "Alpha Game")
            self.assertEqual(data["projects"][0]["path"], "Game_AlphaGame")

    def test_ingest_game_dir_only_nests_under_original_game(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            source = Path(tmp) / "only_game"
            source.mkdir()
            (source / "options.rpy").write_text("define config.version = '1'\n", encoding="utf-8")

            result = game_ingest.ingest_game(
                source=source,
                workspace_root=workspace,
                game_name="Bare",
            )
            self.assertTrue(result.ok, result.message)
            self.assertTrue(
                (workspace / "Game_Bare" / "original" / "game" / "options.rpy").is_file()
            )

    def test_ingest_zip_with_wrapper_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            payload = Path(tmp) / "payload"
            install = payload / "MyGame"
            _write_min_renpy_install(install)
            zip_path = Path(tmp) / "MyGame-1.0.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                for path in install.rglob("*"):
                    if path.is_file():
                        zf.write(path, path.relative_to(payload).as_posix())

            result = game_ingest.ingest_game(
                source=zip_path,
                workspace_root=workspace,
                game_name="My Game",
            )
            self.assertTrue(result.ok, result.message)
            self.assertTrue(
                (workspace / "Game_MyGame" / "original" / "game" / "options.rpy").is_file()
            )

    def test_zip_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            zip_path = Path(tmp) / "evil.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("../evil.txt", "nope")
                zf.writestr("game/options.rpy", "define config.version = '1'\n")

            result = game_ingest.ingest_game(
                source=zip_path,
                workspace_root=workspace,
                game_name="Safe",
            )
            self.assertTrue(result.ok, result.message)
            self.assertFalse((workspace / "evil.txt").exists())
            self.assertFalse((workspace.parent / "evil.txt").exists())

    def test_zip_uncompressed_cap_rejects_oversized_declared_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "big.zip"
            payload = b"x" * 8192
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
                zf.writestr("game/options.rpy", payload.decode("latin-1"))
            staging = Path(tmp) / "staging"
            with self.assertRaises(ValueError) as ctx:
                game_ingest._extract_zip_to_staging(
                    zip_path,
                    staging,
                    max_uncompressed_bytes=1024,
                )
            self.assertIn("未压缩体积超过上限", str(ctx.exception))

    def test_copy_stream_caps_actual_bytes_even_when_declared_was_zero(self):
        """Hard budget counts written bytes (understated ZipInfo.file_size)."""
        src = io.BytesIO(b"y" * 4096)
        dst = io.BytesIO()
        with self.assertRaises(ValueError) as ctx:
            game_ingest._copy_stream_with_byte_cap(
                src,
                dst,
                total_bytes=0,
                max_uncompressed_bytes=1024,
                chunk_size=256,
            )
        self.assertIn("未压缩体积超过上限", str(ctx.exception))
        self.assertLessEqual(len(dst.getvalue()), 1024)

    def test_conflict_when_folder_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            (workspace / "Game_Taken").mkdir()
            source = Path(tmp) / "src"
            _write_min_renpy_install(source)

            result = game_ingest.ingest_game(
                source=source,
                workspace_root=workspace,
                game_name="Taken",
            )
            self.assertFalse(result.ok)
            self.assertIn("已存在", result.message)

    def test_cancel_during_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            source = Path(tmp) / "src"
            _write_min_renpy_install(source)
            for i in range(20):
                (source / "game" / f"file_{i}.rpy").write_text(f"# {i}\n", encoding="utf-8")

            calls = {"n": 0}

            def should_cancel() -> bool:
                calls["n"] += 1
                return calls["n"] > 3

            result = game_ingest.ingest_game(
                source=source,
                workspace_root=workspace,
                game_name="CancelMe",
                should_cancel=should_cancel,
            )
            self.assertFalse(result.ok)
            self.assertTrue(result.cancelled)
            self.assertFalse((workspace / "Game_CancelMe").exists())

    def test_cli_ingest_subcommand(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            source = Path(tmp) / "src"
            _write_min_renpy_install(source)
            code = registry.main(
                [
                    "--workspace",
                    str(workspace),
                    "ingest",
                    "--source",
                    str(source),
                    "--name",
                    "CLI Game",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(
                (workspace / "Game_CLIGame" / "original" / "game" / "options.rpy").is_file()
            )
            data = json.loads((workspace / registry.REGISTRY_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(data["projects"][0]["name"], "CLI Game")


if __name__ == "__main__":
    unittest.main()
