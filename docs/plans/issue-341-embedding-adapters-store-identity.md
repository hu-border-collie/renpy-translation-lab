# Issue #341：Embedding adapters 与 store identity

状态：独立可合并切片 · 基线 `origin/main@a7319aa`（2026-08-23）

本切片承接 PR #388 冻结的纯核心合同，新增离线可测的 Provider adapter，
并把完整 document identity 接入现有 JSON RAG / Source Index store。它不改变
当前 Sync/Batch Gemini 生产调用，也不接入 translation plan 或 prompt。

## 文件所有权与接缝

- `embedding_adapters.py`：Gemini 与 OpenAI-compatible/LiteLLM adapter、稳定
  provider endpoint/configuration identity、usage 归一化、封闭错误分类。
- `embedding_backend.py`：安全 metadata 的公共校验入口，以及 persisted store
  identity 缺失/损坏时的 fail-closed compatibility report。
- `rag_memory.py`：`JsonRagStore` / `JsonSourceIndexStore` 的 identity 持久化与
  compatibility-gated search。已有向量的 store 不允许静默补贴或替换 identity。

本切片明确不修改 `translation_plan.py`、translation runtime prompt 接缝、
Published Project Analysis 注入，也不修改 GUI Project Analysis 订正页。

## Adapter 合同

- Adapter 从显式 model、output dimension、provider、endpoint/configuration 构造
  document/query `EmbeddingIdentity`；不得从生成模型推断 embedding model。
- Gemini 将语义 task 映射为 `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`，并把
  timeout 放入 Google GenAI HTTP options。
- OpenAI-compatible adapter 支持 LiteLLM callable 或已配置的
  `embeddings.create` callable。API 不支持 task 参数时，task 仍保留在 identity。
- LiteLLM 的 OpenAI 官方缺省会固定并显式传递 `https://api.openai.com/v1`；
  其它 Provider、环境变量或预配置 transport 隐式提供的 endpoint 无法安全核验，
  必须显式声明，否则 adapter 在构造阶段 fail closed。
- `openai_client` transport 视为已配置 client；adapter 不接受会被 client API
  静默忽略的 `api_key` / `request_headers` 覆盖，调用方必须在 client 上配置它们。
- 所有响应都经过 `EmbeddingBatchResult` 与 `validate_embedding_result`，因此数量、
  维度、非有限数和 request binding 漂移统一归类为 `invalid_response`。
- Provider 异常只按 status/type 分类为稳定错误；公开异常不包含原始错误文本、
  URL、header、响应 body 或错误码。

## Provider identity 与凭据边界

持久化的 `provider` 是 provider id 加规范 endpoint/configuration 的 SHA-256，
不保存原始 endpoint。API key 和 request headers 不进入 identity builder。
credential-shaped configuration key，以及带 userinfo/query/fragment 的 endpoint 或
configuration URL 会直接被拒绝，避免秘密影响 metadata 或 fingerprint。

## Store compatibility 与迁移

新 store 应在写入第一条向量前调用 `set_embedding_identity(document_identity)`。
查询使用 `search_history_compatible` 或 `search_segments_compatible`，它们先按字段
比较 backend/provider/model/task type/dimension，再决定是否计算相似度。

通用 `set_metadata` 不接受 `embedding_identity`，避免绕过上述非空 store 防护。
旧 store 只有纯文本记录而没有任何非空 embedding 时可以安全附加 identity；只要
存在一个非空向量，就必须沿用完全相同的 identity 或显式重建。

旧 store 没有完整 identity，或 identity fingerprint 损坏时，查询返回空命中和
字段级 diagnostics，action 固定为 `rebuild_store`。已有向量的 store 不能通过
`set_embedding_identity` 被静默“升级”；必须由后续生产接线显式重建或使用新目录。

## 后续阶段

1. 增加完整 runtime/config/CLI/GUI/doctor 选择与诊断，并把 adapter 接到 Sync RAG
   和 Batch RAG/Source Index 的构建、查询路径。
2. 接普通 Sync Source Index，只有 compatibility-gated 命中可以交给 #346 冻结的
   retrieval provider；本阶段不改 TranslationPlan provider 接缝。
3. 单独接 fresh Published Project Analysis brief；不与向量 store identity 混为一层。
