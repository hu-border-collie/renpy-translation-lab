import json
import tempfile
import unittest
from pathlib import Path

from scripts import compile_dependency_locks as locks


class DependencyLockTests(unittest.TestCase):
    def test_manifest_and_committed_locks_are_current(self):
        self.assertEqual(locks.verify_manifest(), [])

    def test_manifest_detects_a_manually_changed_hash(self):
        manifest = json.loads(locks.MANIFEST_PATH.read_text(encoding="utf-8"))
        first_lock = next(iter(manifest["locks"]))
        manifest["locks"][first_lock] = "0" * 64
        errors = locks.verify_manifest_payload(manifest)
        self.assertTrue(any("manually edited lock" in error for error in errors))

    def test_source_hash_normalizes_platform_line_endings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "requirements.txt"
            digests = []
            for content in (b"one\ntwo\n", b"one\r\ntwo\r\n", b"one\rtwo\r"):
                path.write_bytes(content)
                digests.append(locks.sha256_text_file(path))
            self.assertEqual(len(set(digests)), 1)

    def test_relation_analyzer_uses_repository_owned_inputs(self):
        relation = (
            locks.REPO_ROOT / "relation_analyzer" / "requirements.txt"
        ).read_text(encoding="utf-8")
        semantic = (
            locks.REPO_ROOT / "relation_analyzer" / "requirements-semantic.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("-r ../requirements-core.txt", relation)
        self.assertIn("-r ../requirements-genai.txt", semantic)

    def test_every_lock_uses_hashes_and_exact_versions(self):
        for relative in locks.expected_locks():
            text = (locks.REPO_ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("--hash=sha256:", text, relative)
            requirement_lines = [
                line for line in text.splitlines()
                if line and not line.startswith((" ", "#", "--"))
            ]
            self.assertTrue(requirement_lines, relative)
            self.assertTrue(
                all("==" in line or " @ " in line for line in requirement_lines),
                relative,
            )

    def test_lock_profiles_keep_optional_dependencies_isolated(self):
        for platform in locks.PLATFORMS:
            cli = (
                locks.REPO_ROOT / locks.lock_relative_path(platform, "cli")
            ).read_text(encoding="utf-8")
            gui = (
                locks.REPO_ROOT / locks.lock_relative_path(platform, "gui")
            ).read_text(encoding="utf-8")
            litellm = (
                locks.REPO_ROOT / locks.lock_relative_path(platform, "litellm")
            ).read_text(encoding="utf-8")
            self.assertNotIn("pyside6==", cli)
            self.assertNotIn("litellm==", cli)
            self.assertIn("pyside6==6.11.1", gui)
            self.assertIn("litellm==1.83.7", litellm)
            self.assertIn("keyring==25.7.0", litellm)

    def test_ci_installs_hash_checked_platform_locks(self):
        workflow = (
            locks.REPO_ROOT / ".github" / "workflows" / "tests.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("compile_dependency_locks.py --check", workflow)
        self.assertIn("--require-hashes", workflow)
        self.assertIn("py311-windows-gui.txt", workflow)
        self.assertIn("py311-linux-gui.txt", workflow)


if __name__ == "__main__":
    unittest.main()
