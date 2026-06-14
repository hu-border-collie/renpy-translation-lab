# Project notes

本文档记录仓库状态、边界、安全注意事项和适用人群。日常使用入口见根目录 `README.md`。

## Environment requirements

- Python 3.11+
- `google-genai`，供同步脚本和 Batch 脚本使用
- 有效的 Gemini API Key
- Ren'Py 项目中的 `game/tl/schinese` 翻译目录

## Current boundaries

目前项目更偏“核心引擎”，暂未重点覆盖：

- 图形界面（GUI）
- Excel / HTML 协作流
- 面向普通用户的零配置体验
- 完整的游戏解包 / 打包一体化发布流程
- 面向超大项目的完整 RAG 生产工作流，例如严格的波次式回灌编排、多阶段调度策略
- 完整的结构化剧情图谱生产工作流，例如自动 seed 生成、Neo4j 可视化导出

## Project status

这是一个仍在持续探索和改进中的个人实验项目。

- 日常实验和更新会先在作者本地工作区中进行；这个公开仓库只同步已经跑通、适合公开发布的版本。
- 它不是已经打磨完成、可直接开箱使用的正式产品。
- 不保证在所有环境下稳定运行。
- 更适合作为思路实现、代码快照和进一步改造的基础。
- 项目开发过程中使用了 AI 辅助生成代码，整体方向、功能取舍、测试验证与集成决策由作者负责。
- 目前不承诺及时处理 issue、兼容性问题或长期更新。
- 当前更推荐使用 Batch 脚本；同步脚本保留用于直接运行、补译、局部修复、smoke test，以及可选的 RAG 滚动记忆验证。
- Batch / RAG 仍是主要验证方向；同步 RAG 更适合小批量即时反馈和局部精修，不是 Batch 吞吐流程的替代品。
- 当前的 RAG 能力更适合“小包验证 + 逐步扩展”，还不应被表述为已经完成的大项目生产级方案。
- Structured Story Memory 目前是可选 MVP，适合作为人工维护剧情上下文的 prompt 增强，不是完整自动图谱系统。

## Safety notes

执行任何会修改项目文件的操作前，请先备份，并优先在副本上测试。

不要把以下内容提交到公开仓库：

- 真实 API key
- 你本地的 `api_keys.json`
- 你本地的 `translator_config.json`
- 你本地的 `glossary.json` / `glossary_*.json`
- 你本地的 `story_graph.json` / `story_graph.seed.json`
- 你本地的 `macro_setting.md`
- 私有游戏脚本
- 本地 batch 结果
- history / rag store
- 日志和缓存

## Intended users

这个仓库更适合下面这类使用者：

- 已经熟悉 Ren'Py 项目目录结构
- 能自行准备 `work/game/tl/schinese`、安装 Ren'Py SDK，或理解 `prepare` 行为
- 能阅读 Python 脚本并按需修改本地配置
- 接受这是实验性工具，而不是稳定打包好的最终产品
