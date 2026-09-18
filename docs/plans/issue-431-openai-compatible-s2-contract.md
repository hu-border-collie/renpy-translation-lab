# #431 S2：直连 adapter 的目录发现、诊断与迁移矩阵合同

> **状态**：S2 实现完成（PR 待合并）；验收结论在合并后回填到 #431。
> S1 直连生成合同见
> [issue-431-openai-compatible-contract.md](issue-431-openai-compatible-contract.md)。
> **基线**：`main@dbbe18a`（PR #504 合并点）。
> **关联**：#431、#344、#341、#348。

## 1. 范围

S2 在 S1 直连生成后端之上补齐四块：

1. **模型目录发现**：只读调用 Provider 的 models 端点，作为模型 ID 填写建议；
   目录缺失、失败或未收录时**始终允许手动输入模型 ID**，不作为调用白名单。
2. **连接诊断**：`profiles-probe` 按 ModelProfile 声明的结构化输出模式发起一次
   有界请求，分别报告鉴权、基本生成、结构化输出、reasoning 与 usage。
3. **取消语义边界**：明确 GUI/CLI 当前取消哪些层，以及 `generate_async` 的线程边界。
4. **LiteLLM 迁移矩阵**：清点当前调用点，标注直连已覆盖 / 仍需 LiteLLM / 暂无需求，
   为后续是否移除依赖提供依据；本切片不删除 LiteLLM。

## 2. 模型目录发现

- 端点：`provider.models_url` 优先；否则由 `base_url` 派生 `/models`，保留 query。
- 认证与请求头：复用 `openai_compatible_connection`，与生成路径共用
  `credential_ref`、`extra_headers` 校验与脱敏规则。
- 解析：OpenAI 风格 `{"data":[{"id":"..."}]}`；保留 provider 原始 ID 与斜杠语义，
  去重后排序；不按模态过滤，避免误删可调用模型。
- 输出：`profiles-list-models --profile <ID> --output json` 返回
  `status / profile_id / provider / base_url / models_url / count / models / source`；
  `base_url` / `models_url` 诊断值统一脱敏（去除 userinfo/query/fragment）。
  失败使用稳定机器码（`CREDENTIAL_UNAVAILABLE`、`MODEL_CATALOG_TIMEOUT`、
  `MODEL_CATALOG_RATE_LIMITED`、`MODEL_CATALOG_UNAVAILABLE`、
  `MODEL_CATALOG_UNSUPPORTED`、`MODEL_CATALOG_INVALID`、`MODEL_CATALOG_FAILED`）。
- GUI：profile 编辑区提供「拉取模型列表」与只读目录下拉，选择后填入模型字段；
  模型字段始终可手动编辑。切换 Profile 时清空上一 Provider 的目录缓存，避免跨供应商误填；
  目录错误只提示，不阻止保存或启动。
- 非目标：后台定时刷新、跨 Provider 聚合目录、把目录当白名单、目录失败自动切换模型。

## 3. 连接诊断

- 复用现有 `profiles-probe` / GUI「测试所选 Profile 能力」与 `build_sync_backend`。
- 探测请求使用 strict JSON schema 兼容的 `_translation_schema`
  （每层 `additionalProperties: false`、属性全量 `required`）。
- 传入 ModelProfile 已声明的 `structured_output_mode`，避免用适配器默认值评估
  Provider 的真实结构化输出能力。
- usage 的 reasoning token 同时识别顶层 `reasoning_tokens` 与
  `completion_tokens_details.reasoning_tokens`。
- 仍然恰好一次真实计费请求；密钥不进入报告。

## 4. 取消语义

| 层 | 当前行为 | S2 结论 |
| --- | --- | --- |
| durable Sync CLI | 取消写回 run store，进程/attempt 边界可恢复 | 不变，仍是权威取消路径 |
| GUI 连接测试 / probe worker | `asyncio` task 取消立即返回；`generate_async` 不等待结果 | `CancelledError` 原样传播；底层阻塞 HTTP 可能继续到超时 |
| `generate_async` | `asyncio.to_thread` 包装同步 HTTP | S2 明确线程边界，不做 socket 级 abort；真正中断 HTTP 留后续切片 |
| 同步 CLI | 用户中断进程即终止；无后台持久化副作用 | 不变 |

## 5. LiteLLM 迁移矩阵

详见 [issue-431-litellm-migration-matrix.md](issue-431-litellm-migration-matrix.md)。
S2 只完成清点与文档，不删除依赖、不改默认路径、不迁移旧配置。

## 6. 测试

- 目录：payload 解析、显式/派生 URL、GET 与请求头、query 保留与诊断脱敏、
  鉴权/限流/5xx/无效 JSON/空目录错误分类、connection helper 复用。
- CLI：成功 envelope、非直连 adapter 拒绝、缺失凭据机器码、文本模式。
- 诊断：strict schema 校验、声明模式传入 probe、reasoning token 嵌套识别。
- 取消：`generate_async` 在取消时传播 `CancelledError` 且不返回成功结果。
- GUI：拉取按钮触发 immediate action、目录下拉填充与模型字段写入、目录失败提示。

## 7. 非目标

- 不实现真实 Provider smoke；发布前人工验证另行记录。
- 不实现模型列表缓存/定时刷新，不引入第二套目录服务。
- 不做 socket 级取消、不删除 LiteLLM、不解绑 final_review、不做 Anthropic / Responses。
