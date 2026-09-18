# #431 LiteLLM 调用点清点与迁移矩阵

> **状态**：S2 快照（`main@dbbe18a`，PR #504 合并后）。本文只记录清点与路线，不删除依赖、
> 不改默认执行路径、不迁移旧配置。
> **关联**：#431、#344、#341、#348。
> **口径**：代码 / 测试证据优先；生产 workspace 与真实 Provider smoke 不在本表。

## 1. 当前调用点

| 调用点 | 代码位置 | 用途 | S2 状态 |
| --- | --- | --- | --- |
| Sync 生成（翻译 / 术语 / 订正 / 项目分析） | `litellm_sync_backend.LiteLLMSyncBackend`；`model_profile.build_sync_backend` | `sync.backend=litellm` 或 `adapter=litellm` 的同步文本生成 | 直连 `openai_compatible` 已覆盖同一合同；LiteLLM 分支保留 |
| Embedding | `translator_runtime` / `gemini_translate_batch` 的 `litellm.embedding` | RAG / Source Index embedding | **仍需 LiteLLM**（#341 范围，S2 不覆盖） |
| 能力探测 / 连接测试 | `model_capability_probe.default_generator`；`gui_qt/litellm_worker.py` | 一次有界请求 + capability 报告 | 走 `build_sync_backend`：直连 profile 已不走 LiteLLM；LiteLLM profile 仍走 |
| Provider 目录 / 模型列表 | `litellm_provider_config` 的 native catalog / model cost map；GUI LiteLLM 设置页 worker | LiteLLM 内置 / 在线模型目录、版本与安装 | **仍需 LiteLLM**；直连 adapter 使用自己的 models 端点，不读 LiteLLM 目录 |
| 自定义 Provider 注册与迁移 | `litellm_provider_config.custom_provider_registry`；`model_routing_migration` | 旧 `sync.custom_litellm_providers` 解析与迁移 | **仍需 LiteLLM 语义**；直连 provider 使用 `model_routing` provider/profile |
| LiteLLM 安装 / 版本 / 可选依赖 gating | `optional_feature.py`、`gui_qt/settings/litellm_page.py`、requirements lock | 仅 LiteLLM 用户需要 | 与直连路径解耦；S2 不改 |
| Provider contract smoke / durable smoke | `scripts/run_provider_contract_smoke.py`、`scripts/run_durable_sync_provider_smoke.py` | 真实连接验证 | 发布前人工；直连 adapter 后续补同类脚本 |

## 2. 迁移分类

### 2.1 已被直连后端覆盖（S1 + S2）

- 同步生成五阶段通过 `build_sync_backend` 统一构造；adapter 为 `openai_compatible` 时
  不 import LiteLLM。
- 连接 / 凭据 / 非敏感 extra headers / 脱敏 URL / 错误分类由共享
  `openai_compatible_connection` 与 `sync_model_backend` 合同提供。
- 模型目录发现、`profiles-probe` 连接诊断已接入直连路径。
- 内置预设（OpenAI / OpenRouter / DeepSeek / xAI / Ollama / 自定义）不依赖 LiteLLM。

### 2.2 仍需 LiteLLM

- **非 OpenAI-compatible 的原生 LiteLLM Provider**：Claude、Bedrock、Vertex 等通过
  LiteLLM 适配的 provider；直连 adapter 只承诺 Chat Completions 兼容端点。
- **Embedding**：`litellm.embedding` 路径；是否改为直连 embedding 属 #341 决策。
- **LiteLLM 官方在线目录 / 模型成本表 / 版本管理 / 可选安装**：服务现有
  `sync.backend=litellm` 与 GUI LiteLLM 页用户。
- **旧配置迁移语义**：`sync.custom_litellm_providers` 与 `adapter=litellm` 的
  provider 前缀模型继续按原语义解析；不能等价迁移时保持旧后端并给出诊断。

### 2.3 暂无实际需求 / 后续决策

- 删除 `requirements-litellm.txt`、锁文件与 CI 安装任务：需等待 2.1 覆盖队列完成、
  2.2 有替代方案或明确不再支持，且需要独立迁移 / 回滚计划。
- LiteLLM remote Batch、供应商目录自动刷新、Anthropic Messages / OpenAI Responses：
  按真实需求另开切片。
- 真实 Provider smoke 矩阵：直连 adapter 至少覆盖 OpenAI / OpenRouter / DeepSeek /
  xAI / Ollama 中可获得的连接；结果记录环境与日期，不把 mocked 测试当作兼容性证明。

## 3. 迁移与回滚原则（后续切片沿用）

1. 直连与 LiteLLM 是 adapter 选择，不是互相重写；配置迁移保留 provider / model /
   credential_ref / params 语义。
2. 不静默切换实际模型、服务地址或执行策略；不可等价迁移时保留旧后端并报告原因。
3. 每步迁移必须可重复执行、有备份与回滚路径；旧配置 fixture 在迁移前后语义一致。
4. 迁移默认完成、真实 smoke 通过且覆盖矩阵确认前，不删除 LiteLLM 依赖与安装路径。
5. 任何默认值变更仍按 #457 的结论单独决策。
