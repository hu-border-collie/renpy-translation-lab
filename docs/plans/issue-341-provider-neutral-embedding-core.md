# Issue #341：Provider-neutral Embedding 纯核心

本阶段只冻结可独立合并的核心合同，不接入生产调用路径，也不改变现有 Gemini 行为。实现位于 `embedding_backend.py`，不依赖 Provider SDK、LiteLLM、PySide6、凭据或网络。

## 核心合同

- `EmbeddingBackend` 是最小同步 protocol：adapter 接收一个已校验的 `EmbeddingBatchRequest`，返回 `EmbeddingBatchResult`，失败时映射为带稳定 `EmbeddingErrorCategory` 的 `EmbeddingBackendError`。
- `EmbeddingTaskType` 只表达跨 Provider 的 `document` / `query` 语义。后续 Gemini adapter 负责映射到 `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`；OpenAI-compatible adapter 可在 API 不支持 task 参数时仅保留该语义用于 identity。
- `EmbeddingIdentity` 固定 `backend`、`provider`、`model`、`task_type` 和 `output_dimension`，并提供包含 schema version 的规范 JSON 与无凭据 SHA-256 fingerprint。store 应完整保存 document identity 的 `to_dict()`，不能只保存 model 名称。
- batch request 固定 identity、输入顺序、有限正 timeout 和安全 metadata；result 绑定 request fingerprint，并校验向量数、每条维度、有限数值与 usage metadata。
- metadata 中任何 credential-shaped key（包括嵌套的 API key、authorization、secret、password、credential ref、单数 token 等）都会在入口被拒绝，不会脱敏后继续参与持久化或指纹。

## Store / query 兼容性

`check_store_query_compatibility(store_identity, query_identity)` 要求：

1. store identity 的 task type 为 `document`；
2. query identity 的 task type 为 `query`；
3. backend、provider、model、output dimension 完全相同。

通过时 action 为 `none`。任何差异都会返回有序、字段级 mismatch code，action 固定为 `rebuild_store`，并明确禁止比较这两组向量。identity 反序列化还会核验持久化 fingerprint，避免字段被修改后仍冒充旧 store。

## 后续接线边界

- Gemini adapter：把现有 `embed_content` 参数和异常映射到本合同；保留当前默认模型、维度和重试行为。adapter 返回后必须调用 `validate_embedding_result`。
- OpenAI-compatible adapter：必须显式取得 embedding model、provider/config identity 和维度，不能从生成模型推断；endpoint、header 和 credential 只留在 adapter 配置，不得进入 request/store metadata。
- Sync RAG / Source Index：构建或更新 store 时写入 document identity；查询前由所选 adapter 生成 query identity并执行兼容检查。若 action 为 `rebuild_store`，跳过检索并把 codes/message 交给 diagnostics，绝不计算相似度。
- `translation_plan.py`：后续 retrieval provider 消费已经通过兼容检查的命中与诊断；本核心不导入 plan，也不组装 prompt，从而避免与 #346 P2/P3 并行工作发生耦合。

尚未包含：Provider adapter、异步/流式 API、重试策略、配置/doctor/GUI、store 迁移、Sync Source Index 与 Published Project Analysis 生产接线。这些应在后续阶段分别落地，并保持旧 store 不被静默迁移或重写。
