# #348 P0：Model Routing 配置与迁移合同

> 状态：P0 合同已冻结；尚未接入生产读取、迁移写盘或 GUI。当前运行行为仍以旧
> `sync.*` / `batch.*` 配置为准。

本文冻结 #348 第一阶段的长期配置形状、兼容优先级、迁移输入和 #202/#348
所有权边界。可执行验证器位于 `model_routing_config.py`；schema-v1 示例与四类旧配置
fixture 位于 `tests/fixtures/model_routing_config_v1.json` 和
`tests/fixtures/model_routing_legacy/`。

## 决策

### D1：一个版本化聚合根

新配置只写入 `translator_config.json` 顶层的 `model_routing`：

```json
{
  "model_routing": {
    "schema_version": 1,
    "providers": {
      "gemini-direct": {
        "label": "Google Gemini",
        "adapter": "gemini",
        "provider": "gemini",
        "credential_ref": {
          "kind": "api_keys_json",
          "name": "api_keys",
          "env_name": "GEMINI_API_KEY"
        }
      }
    },
    "profiles": {
      "gemini-main": {
        "label": "Gemini Main",
        "provider_id": "gemini-direct",
        "model": "gemini-3.1-flash-lite",
        "models": [],
        "capability_overrides": {},
        "params": {},
        "embedding_profile_id": ""
      }
    },
    "defaults": {
      "primary_profile_id": "gemini-main",
      "execution_strategy": "sync"
    },
    "routes": {}
  }
}
```

把 schema version 放在聚合根内，可让项目路径、prepare、上下文开关、质量规则等
非模型配置继续独立演进，也避免用整个 `translator_config.json` 的版本变化强制迁移
无关字段。

### D2：Provider 连接与 Model Profile 分离

- `providers` 保存 adapter、上游 Provider 标识、catalog/base URL 和凭据引用。
- `profiles` 保存模型、同 Provider 内轮换、能力覆盖、生成参数和 embedding profile
  引用。
- 一个 Provider 可被多个 profile 复用；编辑连接信息不会要求复制每个模型配置。
- resolver 组合二者生成现有 `model_profile.ModelProfile` 快照，run/manifest 仍只记录
  非敏感引用和实际解析结果。

`adapter` 首期只允许 `gemini` / `litellm`。LiteLLM 内置 Provider 与自定义
OpenAI-compatible Provider 使用同一结构；自定义连接只是在 `base_url`、
`models_url` 和 credential reference 上有额外值。

### D3：默认值与阶段覆盖

- `defaults.primary_profile_id` 是角色指针，不是 profile 自身 ID。
- `defaults.execution_strategy` 是用户显式选择的 `sync` 或 `gemini_batch`。
- `routes` 只保存显式覆盖；缺失的阶段继承两个默认值。
- route 可以只覆盖 `profile_id`、只覆盖 `strategy`，或同时覆盖二者。
- 可配置阶段固定为：初译 `translation`、术语 `keyword`、订正 `revision`、
  项目分析 `project_analysis`、最终审校 `final_review`。
- `gemini_batch` 只能与 `gemini` adapter profile 组合；不支持的组合在启动前拒绝，
  不静默切换 Provider 或模型。

### D4：用户 ID 与内部 slot 分离

Provider/profile ID 使用小写 `[a-z0-9_-]+`。用户 profile 不得使用 `primary`、
`batch` 或 `<stage>_model` / `<stage>_override`，这些名称属于
`model_profile.py` 的旧配置 resolver slot。迁移器必须生成稳定、不冲突且可重复的 ID。

### D5：凭据只保存引用

`credential_ref` 首期允许：

- `api_keys_json`：现有 Gemini key 文件槽位；
- `keyring`：系统安全存储中的 Provider 槽位；
- `env`：环境变量名；
- `none`：无需鉴权的本地 Provider。

`model_routing` 任意层级都禁止 `api_key`、`token`、`secret`、`password`、
`authorization` 等凭据值字段。URL 不得包含用户名、密码、query 或 fragment。
manifest、日志和诊断导出继续只显示 reference 与脱敏结果。

### D6：未知字段必须保留

schema-v1 validator 接受非敏感未知字段。兼容 reader 可以忽略尚不认识的字段，
但 Settings 保存和 migrator 必须在原对象上合并已知修改，不得从 dataclass 重建整段
配置。较新 schema version 必须明确拒绝执行，不能降级后继续运行。

## 新旧配置优先级

迁移期采用单一事实来源，不混合拼装：

1. `model_routing` 不存在：完整使用旧配置，行为不变；
2. `model_routing` 存在且 schema/引用/能力合法：完整使用新配置；
3. `model_routing` 存在但无效或版本不支持：拒绝启动并给出路径化诊断；不得回退旧配置；
4. 迁移成功后可以暂时保留旧字段用于回滚，但运行时不得按字段逐个从旧配置补值；
5. 已冻结的 run 始终使用自己的 ModelRoutingPlan/RunSnapshot，不受后续配置迁移影响。

这一优先级避免“新配置写错一个字段后悄悄使用旧模型”以及 Sync/Batch 混合来源。

## 旧字段映射

| 旧字段 | schema-v1 目标 | 迁移语义 |
|---|---|---|
| `sync.backend` | Provider + Sync profile | 保持实际 Gemini/LiteLLM adapter |
| `sync.model` | Sync profile `model` | 保持实际同步模型 |
| `sync.models` | Sync profile `models` | 保持同 profile 内轮换顺序 |
| `sync.custom_litellm_providers` | `providers` | 迁移连接元数据和凭据引用，不迁移 key 值 |
| `batch.model` | Gemini Batch profile | 保持实际 Batch 模型和默认策略 |
| `batch.project_analysis.model` | `routes.project_analysis` | 仅非空值生成覆盖 |
| `batch.final_review.model` | `routes.final_review` | 仅非空值生成覆盖 |
| `sync.rag` / `batch.rag` embedding 字段 | embedding profile | 只迁移模型连接；RAG policy 留在原上下文配置 |

初译、术语和订正在旧配置中没有独立模型时继承对应运行的 base profile，不人为生成
重复 route。`batch` 的 chunk、prompt、thinking、pricing、quality gate 等执行/业务参数
不属于 ModelProfile，不迁入 `model_routing`。

现有产品的默认入口是 Gemini Batch，因此旧配置迁移后的 `defaults` 仍指向由
`batch.model` 生成的 Gemini profile，默认策略仍是 `gemini_batch`。旧 `sync-*` 命令
在兼容期显式选择迁移所得的 Sync profile，不从产品默认值反推；这样既保留 GUI
默认入口，也不改变脚本化 Sync 调用的模型。用户之后主动修改主 profile/策略才改变
该默认值。

## 迁移事务合同（P1）

P1 migrator 必须满足：

1. 先解析并验证原 JSON object；无效输入不写盘；
2. 从内存副本构造完整 schema-v1，运行合同验证和 legacy/effective-plan 对比；
3. 在同目录创建带时间戳且权限不放宽的可恢复备份；
4. 临时文件写入、flush 后原子替换，任一步失败都保留原配置；
5. 重复迁移返回 `already_current`，不新增备份、不改变字节或 ID；
6. 保留所有非敏感未知顶层字段及未知旧字段；
7. 不读取、复制或持久化 API Key 值；
8. 输出 migration report：source/target schema、生成的 ID、映射/保留/警告字段、
   备份路径和非敏感 fingerprint；
9. 提供显式 rollback，恢复前校验当前文件与迁移报告，避免覆盖迁移后的用户编辑；
10. GUI 首次发现旧配置时先预览并请求用户确认，不在页面加载时隐式写盘。

## #202 / #348 所有权

| 关注点 | 唯一所有者 | 消费者 |
|---|---|---|
| schema、validator、migrator、rollback | #348 共用核心 | CLI、GUI Settings |
| ModelProfile/strategy/route 解析 | #348 共用核心 | TranslationPlan、Sync/Batch 服务 |
| Settings page contract/coordinator | #202 GUI 层 | #348 Model Profiles 页面 |
| load/collect/validate/save 编排 | #202 coordinator | 各 Settings page adapter |
| 原始 JSON、unknown-field preservation、原子保存 | ProjectState/共用 config store | coordinator、CLI migrator |
| dirty snapshot 与离开保护 | #202 coordinator | 各 Settings 页面 |
| run 状态和恢复 | #347 `SyncRunService` / RunSnapshot | 统一翻译页 |
| check → bound preview → apply | 现有公共安全层 | 统一翻译页 |

完整 GUI 开始前，#202 先提供 page adapter 的 `load/collect/validate` contract；#348
不在新页面中直接复制 `MainWindow._on_save_config`，#202 也不定义第二套模型迁移语义。

## 分阶段验收

- **P0（本文）**：schema、纯 validator、四类旧 fixture、调用链和所有权合同；不改变运行行为。
- **P1**：幂等 migrator、备份/rollback、兼容 reader、legacy/effective-plan 等价测试。
- **P2**：生产 resolver 和服务消费 schema-v1；旧配置兼容入口保留弃用诊断。
- **P3**：统一 CLI/GUI/Settings；复用 #347 服务和 #202 coordinator。
- **P4**：迁移/回滚文档、诊断导出、四类真实 smoke 与 #344 收口。

## P0 非目标

- 不修改 `translator_config.example.json` 的生产示例；它必须在 P1 reader/migrator
  可用后再切换。
- 不改变 CLI 命令、GUI 文案、默认模型或默认 Sync/Batch 策略。
- 不执行自动迁移、写配置备份或真实 Provider 请求。
- 不把 `model_routing` 当作已经可用的用户配置；当前新增 fixture 只用于合同测试。
