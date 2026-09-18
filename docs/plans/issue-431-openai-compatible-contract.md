# #431 S1：直连 OpenAI-compatible Sync 生成合同

> **状态**：S1 已实现（当前分支；真实 smoke、final_review 解绑与 LiteLLM 移除属后续切片）。本文件冻结首期“协议适配器 + 供应商连接 + ModelProfile”的最小合同；
> 实现与界面以当前代码和现行手册为准。
> **基线**：`main@3fe9c79`（PR #503 合并点）。
> **关联**：#431、#344、#348、#457、#265。
> **父 issue**：#344。

## 1. 范围

S1 新增 **`openai_compatible` 协议适配器**，为同步生成路径提供不依赖 LiteLLM 的直连
Chat Completions 后端，并完成最小配置与 GUI 接线：

- 新增 adapter `openai_compatible`，覆盖 provider / profile / 路由解析与能力快照；
- 新增直连 HTTP 后端（标准库实现，不 `import litellm`）；
- Provider 连接支持 `base_url`、`models_url`、`credential_ref` 与 `extra_headers`；
- ModelProfile 支持模型级 `params`（白名单生成参数）与
  `capability_overrides.structured_output.mode`；
- GUI「模型与 Provider」页支持 adapter、extra headers、预设和结构化输出模式；
- 离线合同测试覆盖请求序列化、结构化输出三模式、usage、reasoning / 最终文本区分、
  超时、限流、鉴权、服务不可用与无效响应。

**S1 覆盖范围**：所有通过 `model_profile.build_sync_backend` 的同步生成阶段（初译、术语、
订正、项目分析等）都可在 profile adapter 为 `openai_compatible` 时走直连后端；durable Sync
与普通 Sync 共用同一后端构造入口。

## 2. 职责边界

| 层 | 职责 |
| --- | --- |
| 协议适配器 | 认证方式、请求序列化、响应解析、错误映射；不拥有业务 prompt / schema |
| Provider 连接 | adapter、上游 provider 标识、base URL、模型列表 URL、凭据引用、必要且非敏感的额外请求头 |
| ModelProfile | 绑定连接与模型 ID，承载模型级能力覆盖、结构化输出模式与生成参数 |
| ExecutionStrategy | `sync` / `gemini_batch`；`gemini_batch` 继续只接受 `gemini` adapter |
| 执行器 / 路由 | 继续使用现有 durable Sync、TaskRoute、usage ledger、check/apply 合同 |

ModelProfile 是运行时唯一模型来源；模型目录只用于发现与辅助填写，不是调用白名单。

## 3. 首期迁移矩阵

| 调用点 | 现状 | S1 结果 |
| --- | --- | --- |
| Sync 生成（初译 / 术语 / 订正 / 项目分析） | 经 `LiteLLMSyncBackend` | adapter 为 `openai_compatible` 时改走直连后端；其余 provider 继续 LiteLLM |
| `build_sync_backend` | gemini / litellm 两分支 | 新增 `openai_compatible` 分支 |
| GUI 连接诊断 / 能力探测 | `generate_async` | 直连后端提供 `generate_async` 包装，探测复用同一合同 |
| final_review | 硬限 `gemini_batch` | S1 保持排除；S3 已通过 `final-review-run-sync` 解绑 |
| Embedding | `litellm.embedding` / Gemini | 不在 S1 范围，继续现状 |
| Gemini Batch | Gemini 直连 | 不变 |
| 默认值 | `legacy-batch + gemini_batch` | 不变（#457 维持现状） |
| LiteLLM 依赖 | 生成 / embedding / 目录 / 诊断 | 不删除；S1 只让常用兼容 provider 可绕过 |

## 4. 配置合同

### 4.1 Provider

```json
{
  "label": "OpenAI",
  "adapter": "openai_compatible",
  "provider": "openai",
  "base_url": "https://api.openai.com/v1",
  "models_url": "https://api.openai.com/v1/models",
  "credential_ref": {
    "kind": "keyring",
    "name": "openai",
    "env_name": "OPENAI_API_KEY"
  },
  "extra_headers": {
    "X-Client-Version": "renpy-translation-lab"
  }
}
```

- `base_url` 必填、必须是干净的 `http(s)` URL，不带 userinfo / query / fragment；
  后端在其后拼接 `/chat/completions`（已以 `/chat/completions` 结尾时不再重复）。
- `credential_ref.kind` 支持 `none` / `env` / `keyring`；`api_keys_json` 只属于 Gemini，
  配置校验拒绝。
- `extra_headers` 只允许非敏感请求头；`Authorization` / `api-key` / `token` 等名称会被
  现有敏感键扫描拒绝，凭据必须走 `credential_ref`。请求头值不进入 manifest 诊断输出。
- 内置预设（S1）：`openai`、`openrouter`、`deepseek`、`xai`、`ollama`、`custom`。
  预设只提供连接默认值（base URL、env 名、结构化输出模式建议），不绑定模型白名单。

### 4.2 ModelProfile

```json
{
  "label": "OpenAI GPT",
  "provider_id": "openai",
  "model": "gpt-4.1-mini",
  "models": ["gpt-4.1-mini"],
  "params": {
    "temperature": 0.2,
    "max_output_tokens": 4096,
    "top_p": 1.0
  },
  "capability_overrides": {
    "structured_output": {"mode": "strict_json_schema"}
  }
}
```

- `params` 只接受白名单键：`temperature`、`max_output_tokens`、`top_p`、
  `frequency_penalty`、`presence_penalty`、`seed`、`stop`、`timeout`；
  其他键在配置校验阶段以 `unsupported_generation_param` 拒绝，不做静默忽略。
- 参数优先级：任务 / plan 生成的请求配置作为默认值，ModelProfile `params` 作为模型级
  显式覆盖后写入请求；显式请求级覆盖留给后续切片。
- `capability_overrides.structured_output.mode` 取 `strict_json_schema` / `json_object` /
  `prompt_only_json`；未声明时按适配器默认（`prompt_only_json`）或预设建议解析。
- `models` 只表达同一 provider 内的轮换池；S1 直连后端单次请求只使用当前 `model`，
  不自行跨模型轮换。

## 5. 请求 / 响应合同

### 5.1 请求

- `POST {base_url}/chat/completions`，`Content-Type: application/json`，
  `Accept: application/json`。
- `messages`：`system_instruction`（如有）映射为 `system` 消息，`contents` 为字符串时映射为
  `user` 消息；消息列表形态沿用现有同步请求合同。
- `model` 原样传递，保留斜杠语义（例如 `accounts/...`、`org/model`），不重写 provider 前缀。
- `temperature` / `max_tokens`（由 `max_output_tokens` 映射）/ `top_p` /
  `frequency_penalty` / `presence_penalty` / `seed` / `stop` 按配置传递。
- `timeout` 使用现有 `normalize_sync_timeout_seconds` 边界；HTTP 层以秒为单位。
- `response_format` 按结构化输出模式生成；`prompt_only_json` 不发送 `response_format`。
- 认证：`Authorization: Bearer <resolved key>`；`credential_ref.kind=none` 时不发送。
  `diagnostic_api_key`（GUI / doctor 一次性探测）优先，不写入配置或 manifest。

### 5.2 响应

- `choices[0].message.content` 为最终文本；字符串或 text parts 列表均支持。
- `message.reasoning_content` / `reasoning` 只进入 `output_diagnostics`，不混入最终译文。
- `finish_reason`、`usage` 原样保留，供现有 usage ledger 与预算诊断归一化。
- 合法 JSON 文本写入 `parsed`；解析失败不伪装成功，交由现有结果校验。
- 响应中的 provider 原文不进入异常消息；错误只暴露分类、HTTP 状态与稳定 reason。

### 5.3 错误映射

| 条件 | 分类 |
| --- | --- |
| 401 / 403 | `authentication` |
| 429 | `rate_limit` |
| 408 / socket timeout / `TimeoutError` | `timeout` |
| 500 / 502 / 503 / 504 | `service_unavailable` |
| 400 / 404 / 422 且与结构化输出 / 能力相关 | `unsupported_capability` |
| 响应不是 JSON、缺 `choices`、content 非文本 | `invalid_response` |
| 其余 HTTP / 传输错误 | `provider_error` |

错误对象沿用 `sync_model_backend.SyncBackendError`，由现有重试 / 恢复分类消费；不新增
第二套错误合同。

## 6. 结构化输出模式

| 模式 | 请求行为 | 说明 |
| --- | --- | --- |
| `strict_json_schema` | `response_format={"type":"json_schema","json_schema":{"name":...,"schema":...,"strict":true}}` | 仅当 profile / preset 明确声明时使用 |
| `json_object` | `response_format={"type":"json_object"}` | 兼容 provider 的保守默认候选 |
| `prompt_only_json` | 不发送 `response_format`，schema 仍由 prompt / 结果校验约束 | adapter 默认；不假装严格约束已生效 |

Provider 返回“不支持 response_format”类错误时映射为 `unsupported_capability`，
**不静默降级**为 prompt-only 后宣称成功。

## 7. CLI / GUI 合同

- CLI 继续读取同一 `model_routing` 配置，不新增命令；`profiles-show / profiles-validate`
  展示 adapter、能力来源与校验错误。
- GUI「模型与 Provider」页：
  - provider adapter 下拉加入 `openai_compatible`；
  - provider 编辑增加预设选择与 `extra_headers` JSON 字段；
  - profile 能力区增加结构化输出模式选择（跟随适配器 / 三种模式）；
  - 凭据仍只保存引用，页面不解析、不回显 key。
- 连接诊断与能力探测继续复用 `build_sync_backend`；直连后端提供 `generate_async` 以保持
  现有 GUI worker 的取消 / 超时语义。

## 8. 测试与验收

- 后端离线合同测试：请求 URL / headers / messages / 三种结构化输出 / 参数映射 /
  超时 / 鉴权 / 限流 / 5xx / 无效 JSON / 空 choices / usage / reasoning 与最终文本分离 /
  斜杠模型 ID / 无凭据模式。
- 路由与配置测试：adapter 枚举、`extra_headers` 读取、`params` 白名单与拒绝、
  capability 解析、`build_sync_backend` 分支、legacy entrypoint 不再误拒
  `openai_compatible` profile 的 params。
- GUI 测试：adapter 可选、extra headers 往返、预设填充、结构化模式往返、凭据不回显。
- 不调用真实 provider；真实连接 smoke 属发布前人工验证，不作为本切片 CI 前提。

## 9. 非目标与后续切片

- 不实现 Anthropic Messages / OpenAI Responses；Chat Completions 与 Responses 不混用。
- 不删除 LiteLLM、不改默认主路径、不迁移旧配置行为。
- 不解绑 final_review 的 `gemini_batch` 限制。
- 不实现通用 provider 脚本、任意协议插件、跨 provider 自动 fallback 或负载均衡。
- 不重写上下文、术语、质量、恢复与安全写回核心。
- 后续：真实 provider smoke、内置预设的模型目录发现、Anthropic / Responses 协议、
  final_review 解绑、按迁移矩阵决定是否移除 LiteLLM。
