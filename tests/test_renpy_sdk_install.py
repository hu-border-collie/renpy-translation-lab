"""Offline tests for recommended Ren'Py SDK installer."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import renpy_sdk_install as sdk


def _make_sdk_zip(path: Path, *, root_name: str = "renpy-8.5.3-sdk", bad_member: str | None = None) -> str:
    """Write a minimal SDK zip; return its sha256 hex."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        if bad_member:
            zf.writestr(bad_member, b"evil")
        else:
            zf.writestr(f"{root_name}/renpy.py", b"# renpy\n")
            zf.writestr(f"{root_name}/README.txt", b"test\n")
    path.write_bytes(buf.getvalue())
    return sdk.sha256_file(path)


class RenpySdkInstallTests(unittest.TestCase):
    def test_recommended_spec_is_official_fixed(self):
        spec = sdk.recommended_sdk()
        self.assertEqual(spec.version, "8.5.3")
        self.assertTrue(spec.url.startswith("https://www.renpy.org/dl/"))
        self.assertEqual(len(spec.sha256), 64)
        self.assertIn("sdk.zip", spec.archive_name)

    def test_existing_valid_sdk_detects_and_rejects_non_sdk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "empty"
            empty.mkdir()
            self.assertIsNone(sdk.existing_valid_sdk(empty))
            self.assertIsNone(sdk.existing_valid_sdk(None))
            sdk_dir = root / "renpy-8.5.3-sdk"
            sdk_dir.mkdir()
            (sdk_dir / "renpy.py").write_text("# renpy\n", encoding="utf-8")
            found = sdk.existing_valid_sdk(sdk_dir)
            self.assertIsNotNone(found)
            assert found is not None
            self.assertEqual(found.resolve(), sdk_dir.resolve())

    def test_download_rejects_non_allowlisted_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "x.zip"
            with self.assertRaises(sdk.SdkInstallError) as ctx:
                sdk.download_to_file(
                    "https://evil.example/sdk.zip",
                    dest,
                    expected_sha256="0" * 64,
                )
            self.assertIn("仅允许", str(ctx.exception))

    def test_sha256_mismatch_refuses_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "sdk.zip"
            _make_sdk_zip(archive)
            target = root / "out-sdk"
            result = sdk.install_from_archive(
                archive,
                target,
                expected_sha256="0" * 64,
                persist_config=False,
            )
            self.assertFalse(result.ok)
            self.assertIn("SHA-256", result.message)
            self.assertFalse(target.exists())

    def test_path_traversal_member_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "evil.zip"
            digest = _make_sdk_zip(archive, bad_member="../escape.py")
            staging = root / "staging"
            with self.assertRaises(sdk.SdkInstallError) as ctx:
                sdk.extract_sdk_zip(archive, staging)
            self.assertIn("穿越", str(ctx.exception))
            self.assertEqual(digest, sdk.sha256_file(archive))

    def test_absolute_path_member_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "evil.zip"
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                # Zip allows forward-slash absolute-looking names.
                zf.writestr("/tmp/evil.py", b"x")
            archive.write_bytes(buf.getvalue())
            with self.assertRaises(sdk.SdkInstallError):
                sdk.extract_sdk_zip(archive, root / "staging")

    def test_install_from_archive_success_and_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "sdk.zip"
            digest = _make_sdk_zip(archive)
            target = root / "workspace" / "renpy-8.5.3-sdk"
            config = root / "translator_config.json"
            config.write_text(json.dumps({"game_root": "C:/g"}), encoding="utf-8")

            result = sdk.install_from_archive(
                archive,
                target,
                expected_sha256=digest,
                persist_config=True,
                config_path=config,
            )
            self.assertTrue(result.ok, result.message)
            self.assertTrue((target / "renpy.py").is_file())
            self.assertTrue(result.persisted_config)
            data = json.loads(config.read_text(encoding="utf-8"))
            self.assertEqual(data["game_root"], "C:/g")
            self.assertEqual(
                Path(data["prepare"]["renpy_sdk_dir"]).resolve(),
                target.resolve(),
            )

    def test_reuse_existing_valid_sdk(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "renpy-8.5.3-sdk"
            target.mkdir()
            (target / "renpy.py").write_text("# ok\n", encoding="utf-8")
            result = sdk.install_from_archive(
                Path(tmp) / "missing.zip",
                target,
                expected_sha256="0" * 64,
                persist_config=False,
            )
            self.assertTrue(result.ok, result.message)
            self.assertTrue(result.reused_existing)

    def test_conflict_existing_non_sdk_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "sdk.zip"
            digest = _make_sdk_zip(archive)
            target = root / "occupied"
            target.mkdir()
            (target / "notes.txt").write_text("keep\n", encoding="utf-8")
            result = sdk.install_from_archive(
                archive,
                target,
                expected_sha256=digest,
                persist_config=False,
            )
            self.assertFalse(result.ok)
            self.assertIn("拒绝覆盖", result.message)
            self.assertTrue((target / "notes.txt").is_file())
            self.assertFalse((target / "renpy.py").exists())

    def test_cancel_during_extract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "sdk.zip"
            digest = _make_sdk_zip(archive)
            target = root / "sdk"
            calls = {"n": 0}

            def should_cancel() -> bool:
                calls["n"] += 1
                return calls["n"] > 1

            result = sdk.install_from_archive(
                archive,
                target,
                expected_sha256=digest,
                should_cancel=should_cancel,
                persist_config=False,
            )
            self.assertFalse(result.ok)
            self.assertTrue(result.cancelled)

    def test_fake_download_install_recommended(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "payload.zip"
            digest = _make_sdk_zip(archive)
            payload = archive.read_bytes()
            target = root / "renpy-8.5.3-sdk"

            class _Resp:
                headers = {"Content-Length": str(len(payload))}

                def __init__(self):
                    self._bio = io.BytesIO(payload)

                def read(self, n: int = -1) -> bytes:
                    return self._bio.read(n)

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            def fake_urlopen(request, timeout=0):
                self.assertEqual(request.full_url, sdk.RECOMMENDED_URL)
                return _Resp()

            with mock.patch.object(sdk, "RECOMMENDED_SHA256", digest):
                result = sdk.install_recommended_sdk(
                    target,
                    persist_config=False,
                    opener=fake_urlopen,
                )
            self.assertTrue(result.ok, result.message)
            self.assertTrue((target / "renpy.py").is_file())

    def test_cli_show(self):
        code = sdk.main(["show"])
        self.assertEqual(code, 0)

    def test_default_sdk_target_under_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            ws.mkdir()
            target = sdk.default_sdk_target(ws)
            self.assertEqual(target.name, sdk.RECOMMENDED_FOLDER_NAME)
            self.assertEqual(target.parent.resolve(), ws.resolve())

    def test_rejects_install_into_tool_repo(self):
        tool = sdk.tool_package_root()
        inside = tool / "renpy-8.5.3-sdk"
        with self.assertRaises(sdk.SdkInstallError) as ctx:
            sdk.validate_sdk_install_target(inside)
        self.assertIn("工具仓库", str(ctx.exception))
        result = sdk.install_from_archive(
            tool / "missing.zip",
            inside,
            expected_sha256="0" * 64,
            persist_config=False,
        )
        self.assertFalse(result.ok)
        self.assertIn("工具仓库", result.message)

    def test_rejects_install_into_game_star(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Game_Example" / "renpy-sdk"
            target.parent.mkdir()
            with self.assertRaises(sdk.SdkInstallError):
                sdk.validate_sdk_install_target(target)

    def test_rejects_workspace_root_and_game_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            game = workspace / "MyGame" / "work"
            game.mkdir(parents=True)
            workspace.mkdir(exist_ok=True)
            with self.assertRaises(sdk.SdkInstallError) as ctx:
                sdk.validate_sdk_install_target(workspace, workspace_root=workspace)
            self.assertIn("工作区根", str(ctx.exception))
            with self.assertRaises(sdk.SdkInstallError) as ctx2:
                sdk.validate_sdk_install_target(game / "sdk", workspace_root=workspace, game_root=game)
            self.assertIn("游戏项目", str(ctx2.exception))
            # Child under workspace is allowed by path rules.
            ok = sdk.validate_sdk_install_target(
                workspace / sdk.RECOMMENDED_FOLDER_NAME,
                workspace_root=workspace,
                game_root=game,
            )
            self.assertEqual(ok.name, sdk.RECOMMENDED_FOLDER_NAME)

    def test_config_write_failure_removes_placed_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "sdk.zip"
            digest = _make_sdk_zip(archive)
            target = root / "ws" / "renpy-8.5.3-sdk"
            target.parent.mkdir()
            with mock.patch.object(
                sdk,
                "save_renpy_sdk_dir",
                side_effect=sdk.SdkInstallError("config denied"),
            ):
                result = sdk.install_from_archive(
                    archive,
                    target,
                    expected_sha256=digest,
                    workspace_root=target.parent,
                    persist_config=True,
                )
            self.assertFalse(result.ok)
            self.assertIn("config denied", result.message)
            self.assertFalse(target.exists())

    def test_save_renpy_sdk_dir_is_atomic_and_preserves_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "translator_config.json"
            config.write_text(
                json.dumps({"game_root": "C:/g", "prepare": {"language": "schinese"}}),
                encoding="utf-8",
            )
            sdk_dir = Path(tmp) / "renpy-8.5.3-sdk"
            sdk_dir.mkdir()
            (sdk_dir / "renpy.py").write_text("#\n", encoding="utf-8")
            saved = sdk.save_renpy_sdk_dir(sdk_dir, config)
            self.assertEqual(saved.resolve(), sdk_dir.resolve())
            data = json.loads(config.read_text(encoding="utf-8"))
            self.assertEqual(data["game_root"], "C:/g")
            self.assertEqual(data["prepare"]["language"], "schinese")
            self.assertEqual(
                Path(data["prepare"]["renpy_sdk_dir"]).resolve(),
                sdk_dir.resolve(),
            )


if __name__ == "__main__":
    unittest.main()
