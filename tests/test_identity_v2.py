# -*- coding: utf-8 -*-
import os
import tempfile
import unittest
import json
import shutil
from pathlib import Path
from unittest import mock

import translation_core
import translator_runtime as runtime
import gemini_translate_batch as batch_mod
from rag_memory import JsonRagStore, hash_text


class TestIdentityV2AndCompatibility(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.old_base_dir = batch_mod.legacy.BASE_DIR
        self.old_tl_dir = batch_mod.legacy.TL_DIR
        self.old_jobs_dir = batch_mod.BATCH_JOBS_DIR
        self.old_rag_enabled = batch_mod.RAG_ENABLED
        self.old_rag_store_dir = batch_mod.RAG_STORE_DIR
        self.old_global_store = batch_mod._RAG_STORE

        batch_mod.legacy.BASE_DIR = self.tmp_dir
        batch_mod.legacy.TL_DIR = os.path.join(self.tmp_dir, "game", "tl", "schinese")
        batch_mod.BATCH_JOBS_DIR = os.path.join(self.tmp_dir, "logs", "batch_jobs")
        batch_mod._RAG_STORE = None
        os.makedirs(batch_mod.legacy.TL_DIR, exist_ok=True)
        os.makedirs(batch_mod.BATCH_JOBS_DIR, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        batch_mod.legacy.BASE_DIR = self.old_base_dir
        batch_mod.legacy.TL_DIR = self.old_tl_dir
        batch_mod.BATCH_JOBS_DIR = self.old_jobs_dir
        batch_mod.RAG_ENABLED = self.old_rag_enabled
        batch_mod.RAG_STORE_DIR = self.old_rag_store_dir
        batch_mod._RAG_STORE = self.old_global_store

    def test_identity_v2_generation_and_stability(self):
        # 验证 ID 的稳定性和唯一性
        id1 = translation_core.build_identity_v2("script.rpy", "chapter01", 1, "Hello World")
        id2 = translation_core.build_identity_v2("script.rpy", "chapter01", 1, "Hello World")
        self.assertEqual(id1, id2)

        # 参数改变应产生不同的 ID
        id3 = translation_core.build_identity_v2("script.rpy", "chapter02", 1, "Hello World")
        self.assertNotEqual(id1, id3)

        id4 = translation_core.build_identity_v2("script.rpy", "chapter01", 2, "Hello World")
        self.assertNotEqual(id1, id4)

        id5 = translation_core.build_identity_v2("script.rpy", "chapter01", 1, "Hello World!")
        self.assertNotEqual(id1, id5)

        # 默认 _global 和斜杠归一化
        id_win = translation_core.build_identity_v2("game\\script.rpy", None, 0, "Test")
        id_unix = translation_core.build_identity_v2("game/script.rpy", "_global", 0, "Test")
        self.assertEqual(id_win, id_unix)

    def test_collect_tasks_with_block_tracking(self):
        # 模拟包含 translate 块的 rpy 文件内容
        lines = [
            "translate schinese chapter1_start:\n",
            "    # \"Line 1\"\n",
            "    \"Line 1\"\n",
            "    # \"Line 2\"\n",
            "    \"Line 2\"\n",
            "translate schinese chapter1_next:\n",
            "    # \"Line 3\"\n",
            "    \"Line 3\"\n",
        ]
        
        # 即使被注释的行不会成为任务，但 block_index 的计数应该与 scan_all_translation_units 对应
        tasks = runtime.collect_tasks(lines)
        self.assertEqual(len(tasks), 3)
        
        self.assertEqual(tasks[0]["block_name"], "chapter1_start")
        self.assertEqual(tasks[0]["block_index"], 1)
        
        self.assertEqual(tasks[1]["block_name"], "chapter1_start")
        self.assertEqual(tasks[1]["block_index"], 2)
        
        self.assertEqual(tasks[2]["block_name"], "chapter1_next")
        self.assertEqual(tasks[2]["block_index"], 1)

    def test_repeated_translate_block_names_do_not_collide(self):
        lines = [
            "translate schinese strings:\n",
            "    old \"Start Game\"\n",
            "    new \"Start Game\"\n",
            "translate schinese strings:\n",
            "    old \"Start Game\"\n",
            "    new \"Start Game\"\n",
        ]

        mapping = runtime.scan_all_translation_units(lines, "script.rpy")
        self.assertEqual(len(mapping), 2)
        ids = sorted(mapping)
        self.assertNotEqual(ids[0], ids[1])
        self.assertTrue(any(":strings:1:" in item_id for item_id in ids))
        self.assertTrue(any(":strings#2:1:" in item_id for item_id in ids))

    def test_non_targets_do_not_shift_translation_identity(self):
        original_lines = [
            "translate schinese start:\n",
            "    # \"Line 1\"\n",
            "    \"Line 1\"\n",
            "    # \"Line 2\"\n",
            "    \"Line 2\"\n",
        ]
        drifted_lines = [
            "translate schinese start:\n",
            "    # \"Line 1\"\n",
            "    \"Line 1\"\n",
            "    \"images/title.png\"\n",
            "    \"已经翻译\"\n",
            "    # \"Line 2\"\n",
            "    \"Line 2\"\n",
        ]

        original = runtime.scan_all_translation_units(original_lines, "script.rpy")
        drifted = runtime.scan_all_translation_units(drifted_lines, "script.rpy")
        self.assertEqual(set(original), set(drifted))
        line2_id = next(item_id for item_id, value in original.items() if value[3] == "Line 2")
        self.assertIn(":start:2:", line2_id)
        self.assertEqual(original[line2_id][0], 4)
        self.assertEqual(drifted[line2_id][0], 6)

    def test_line_drift_resolution(self):
        # 原始文件
        orig_lines = [
            "translate schinese chapter1:\n",
            "    # \"Line 1\"\n",
            "    \"Line 1\"\n",
        ]
        # 在顶部漂移了 2 个空行和 1 行无意义的 python 语句
        drifted_lines = [
            "\n",
            "init python:\n",
            "    pass\n",
            "\n",
            "translate schinese chapter1:\n",
            "    # \"Line 1\"\n",
            "    \"Line 1\"\n",
        ]

        # 1. 扫描原始文件构建的 V2 ID
        mapping_orig = runtime.scan_all_translation_units(orig_lines, "script.rpy")
        self.assertEqual(len(mapping_orig), 1)
        orig_id = list(mapping_orig.keys())[0]
        orig_line, orig_start, orig_end, orig_text = mapping_orig[orig_id]
        self.assertEqual(orig_line, 2)  # 第 3 行 (0-indexed 2)
        self.assertEqual(orig_text, "Line 1")

        # 2. 扫描发生漂移后的文件，ID 应该完全相同，但定位指向了更新后的行号
        mapping_drifted = runtime.scan_all_translation_units(drifted_lines, "script.rpy")
        self.assertEqual(len(mapping_drifted), 1)
        self.assertIn(orig_id, mapping_drifted)
        drifted_line, drifted_start, drifted_end, drifted_text = mapping_drifted[orig_id]
        self.assertEqual(drifted_line, 6)  # 第 7 行 (0-indexed 6)
        self.assertEqual(drifted_text, "Line 1")

    def test_collect_result_actions_compatibility_and_drift_v2(self):
        # 1. 写入漂移后的翻译文件到 TL 目录中
        file_rel_path = "script.rpy"
        file_path = os.path.join(batch_mod.legacy.TL_DIR, file_rel_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 发生漂移了的文件（多出了前面的空行和 python 语句）
        drifted_lines = [
            "\n",
            "init python:\n",
            "    pass\n",
            "\n",
            "translate schinese chapter1:\n",
            "    # \"Hello world\"\n",
            "    \"Hello world\"\n",  # 原文是 "Hello world"，在第 6 行 (0-indexed)
        ]
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(drifted_lines)

        # 2. 构建一个 Manifest v2，里面的 items 的行号是发生漂移之前的（第 2 行）
        # 它的 ID 是 V2 ID
        v2_id = translation_core.build_identity_v2(file_rel_path, "chapter1", 1, "Hello world")
        manifest = {
            "version": 2,
            "manifest_version": 2,
            "core_schema_version": 2,
            "mode": batch_mod.MANIFEST_MODE_TRANSLATION,
            "input_jsonl_path": os.path.join(self.tmp_dir, "requests.jsonl"),
            "result_jsonl_path": os.path.join(self.tmp_dir, "results.jsonl"),
            "settings": {},
            "files": {
                file_rel_path: {
                    "path": file_path,
                    "task_count": 1,
                }
            },
            "chunks": [
                {
                    "key": "chunk_0",
                    "file_rel_path": file_rel_path,
                    "chunk_index": 1,
                    "items": [
                        {
                            "id": v2_id,
                            "text": "Hello world",
                            "line": 2,  # 旧物理行号！
                            "start": 4,
                            "end": 17,
                            "quote": "\"",
                        }
                    ]
                }
            ],
            "_manifest_path": os.path.join(self.tmp_dir, "manifest.json"),
            "_package_dir": self.tmp_dir,
        }

        # 3. 模拟 Batch API 返回的结果（以 V2 ID 为 key）
        results = [
            {
                "key": "chunk_0",
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": json.dumps([{"id": v2_id, "translation": "你好世界"}])
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
        ]
        with open(manifest["result_jsonl_path"], "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")

        # 4. 执行 collect_result_actions，验证行号是否成功被动态修复到第 6 行，并且校验成功
        replacements, _, _, _ = batch_mod.collect_result_actions(manifest, validate_sources=True)
        self.assertIn(file_rel_path, replacements)
        # 应该修正定位并成功映射到第 6 行上！
        self.assertIn(6, replacements[file_rel_path])
        self.assertEqual(len(replacements[file_rel_path][6]), 1)
        action_tuple = replacements[file_rel_path][6][0]
        self.assertEqual(action_tuple[2], "你好世界")

    def test_collect_revision_actions_uses_v2_identity_after_line_drift(self):
        file_rel_path = "script.rpy"
        file_path = os.path.join(batch_mod.legacy.TL_DIR, file_rel_path)

        original_lines = [
            "translate schinese chapter1:\n",
            "    old \"Void Gate\"\n",
            "    new \"虚空门\"\n",
        ]
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(original_lines)

        jobs = batch_mod.collect_revision_file_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(len(jobs[0]["items"]), 1)
        item = dict(jobs[0]["items"][0])
        self.assertIn(":chapter1:1:", item["id"])
        self.assertEqual(item["line"], 2)

        drifted_lines = [
            "\n",
            "init python:\n",
            "    pass\n",
            "\n",
            "translate schinese chapter1:\n",
            "    old \"Void Gate\"\n",
            "    new \"虚空门\"\n",
        ]
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(drifted_lines)

        result_path = os.path.join(self.tmp_dir, "revision_results.jsonl")
        manifest = {
            "version": 2,
            "manifest_version": 2,
            "core_schema_version": 2,
            "mode": batch_mod.MANIFEST_MODE_REVISION,
            "input_jsonl_path": os.path.join(self.tmp_dir, "revision_requests.jsonl"),
            "result_jsonl_path": result_path,
            "settings": {},
            "files": {
                file_rel_path: {
                    "path": file_path,
                    "task_count": 1,
                }
            },
            "chunks": [
                {
                    "key": "rv-0",
                    "file_rel_path": file_rel_path,
                    "chunk_index": 1,
                    "items": [item],
                }
            ],
            "_manifest_path": os.path.join(self.tmp_dir, "revision_manifest.json"),
            "_package_dir": self.tmp_dir,
        }
        response_text = json.dumps([
            {
                "id": item["id"],
                "should_update": True,
                "revised_translation": "虚空之门",
                "reason": "统一术语",
            }
        ], ensure_ascii=False)
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "key": "rv-0",
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": response_text}
                                ]
                            }
                        }
                    ]
                },
            }, ensure_ascii=False) + "\n")

        replacements, _, failures, summary, _ = batch_mod.collect_revision_actions(
            manifest,
            validate_sources=True,
        )
        self.assertEqual(failures, [])
        self.assertEqual(summary["parsed_items"], 1)
        self.assertIn(file_rel_path, replacements)
        self.assertIn(6, replacements[file_rel_path])
        action_tuple = replacements[file_rel_path][6][0]
        self.assertEqual(action_tuple[2], "虚空之门")

    def test_collect_result_actions_compatibility_v1_fallback(self):
        # 测试旧版 V1 Manifest。如果 Manifest v1 发生漂移，我们将不会进行扫描修复，而是继续使用其内部的 line 定位。
        file_rel_path = "script.rpy"
        file_path = os.path.join(batch_mod.legacy.TL_DIR, file_rel_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        drifted_lines = [
            "\n",
            "translate schinese chapter1:\n",
            "    # \"Hello world\"\n",
            "    \"Hello world\"\n",
        ]
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(drifted_lines)

        # 构建 Manifest v1
        v1_id = f"{file_rel_path}:1:4"
        manifest = {
            "version": 1,  # 旧版
            "core_schema_version": 1,
            "mode": batch_mod.MANIFEST_MODE_TRANSLATION,
            "input_jsonl_path": os.path.join(self.tmp_dir, "requests.jsonl"),
            "result_jsonl_path": os.path.join(self.tmp_dir, "results.jsonl"),
            "settings": {},
            "files": {
                file_rel_path: {
                    "path": file_path,
                    "task_count": 1,
                }
            },
            "chunks": [
                {
                    "key": "chunk_0",
                    "file_rel_path": file_rel_path,
                    "chunk_index": 1,
                    "items": [
                        {
                            "id": v1_id,
                            "text": "Hello world",
                            "line": 1,  # 仍然是原本的 1 物理行
                            "start": 4,
                            "end": 17,
                            "quote": "\"",
                        }
                    ]
                }
            ],
            "_manifest_path": os.path.join(self.tmp_dir, "manifest.json"),
            "_package_dir": self.tmp_dir,
        }

        results = [
            {
                "key": "chunk_0",
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": json.dumps([{"id": v1_id, "translation": "你好世界"}])
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
        ]
        with open(manifest["result_jsonl_path"], "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")

        replacements, _, _, _ = batch_mod.collect_result_actions(manifest, validate_sources=False)
        self.assertIn(file_rel_path, replacements)
        # 应该依然使用原本在 manifest 中设定的 1 物理行，不发生位置重定位。
        self.assertIn(1, replacements[file_rel_path])

    def test_rag_store_source_checksum_fallback(self):
        # 实例化临时 RAG Store 并写入 legacy V1 ID 记录
        store_dir = os.path.join(self.tmp_dir, "rag_store")
        batch_mod.RAG_ENABLED = True
        batch_mod.RAG_STORE_DIR = store_dir
        
        store = batch_mod.get_rag_store()
        store.load()

        # 写入一条 V1 格式 ID 的现有记录
        source_text = "Seek the holy sword"
        source_hash = hash_text(source_text)
        legacy_id = "script.rpy:10:4"
        
        legacy_record = {
            "memory_id": legacy_id,
            "source_text": source_text,
            "source_checksum": source_hash,
            "translation": "寻找圣剑",
            "translation_checksum": hash_text("寻找圣剑"),
            "quality_state": "seed",
            "embedding": [0.1] * 768,
            "embedding_model": batch_mod.RAG_EMBEDDING_MODEL,
            "embedding_task_type": batch_mod.RAG_DOCUMENT_TASK_TYPE,
            "embedding_dim": batch_mod.RAG_OUTPUT_DIMENSIONALITY,
            "embedding_text_kind": "source_text",
            "embedding_text_checksum": source_hash,
        }
        store.upsert_history([legacy_record])

        # 模拟我们将要把一个新的 V2 任务（拥有相同的原文和 checksum，但 ID 升级了）同步到 RAG Store
        v2_id = translation_core.build_identity_v2("script.rpy", "chapter2", 2, source_text)
        new_record = {
            "memory_id": v2_id,
            "source_text": source_text,
            "source_checksum": source_hash,
            "translation": "寻找圣剑",
            "translation_checksum": hash_text("寻找圣剑"),
            "quality_state": "seed",
        }

        # 调用 sync_rag_store_for_jobs 进行同步
        file_jobs = [
            {
                "file_rel_path": "script.rpy",
                "file_path": "script.rpy",
            }
        ]
        
        # 这里的 sync_rag_store_for_jobs 内部应该能够命中 legacy_record 并直接复用其 embedding 写入 v2_id 记录中！
        # 如果没有复用的话，它会试图去调用外部 Embedding API 导致测试中抛出异常（或者因为没有配置 mocked API 而报错）。
        # 我们用 mock 阻止外部的 embed_history_records 调用
        with mock.patch("gemini_translate_batch.embed_history_records") as mock_embed:
            mock_embed.return_value = []
            stats = batch_mod.sync_rag_store_for_jobs(
                file_jobs=[],
                scan_all_files=False,
                extra_records=[new_record]
            )
            # 验证确实被调用了，但参数是一个空列表 [] (因为没有需要向外部 API 请求 embedding 的记录)
            mock_embed.assert_called_once_with([])
            self.assertEqual(stats["reused_embeddings"], 1)
            self.assertEqual(stats["embedding_pending"], 0)

        # 验证 RAG Store 中是否成功写入了 V2 的记录
        v2_rec = store.get_history_record(v2_id)
        self.assertIsNotNone(v2_rec)
        self.assertEqual(v2_rec["embedding"], [0.1] * 768)

    def test_doctor_command_warnings(self):
        # 1. 模拟在 logs/batch_jobs 下放一个旧版 Manifest V1
        legacy_pkg = os.path.join(batch_mod.BATCH_JOBS_DIR, "20260611_old_job")
        os.makedirs(legacy_pkg, exist_ok=True)
        legacy_manifest = {
            "version": 1,
            "manifest_version": 1,
        }
        with open(os.path.join(legacy_pkg, "manifest.json"), "w") as f:
            json.dump(legacy_manifest, f)

        # 2. 模拟 RAG Store 并写入旧版键
        store_dir = os.path.join(self.tmp_dir, "rag_store")
        batch_mod.RAG_ENABLED = True
        batch_mod.RAG_STORE_DIR = store_dir
        
        store = batch_mod.get_rag_store()
        store.load()
        
        legacy_record = {
            "memory_id": "script.rpy:10:4",  # 旧版 ID 冒号格式
            "source_text": "Test",
            "source_checksum": hash_text("Test"),
            "translation": "测试",
            "translation_checksum": hash_text("测试"),
            "embedding": [0.1] * 768,  # 需要有效 embedding 才能被 upsert 成功
        }
        store.upsert_history([legacy_record])

        # 3. 运行 collect_doctor_report
        report = batch_mod.collect_doctor_report()
        warnings = report.get("warnings", [])
        
        # 4. 验证警告列表中是否包含 Manifest v1 与 RAG 旧 ID 升级提醒
        self.assertTrue(any("legacy manifest" in w for w in warnings))
        self.assertTrue(any("RAG store contains legacy ID format keys" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
