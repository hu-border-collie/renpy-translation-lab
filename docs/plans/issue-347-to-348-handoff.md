# #347 → #348 交接：耐久 Sync 产品化接缝

> 状态：#347 实现交接；#348 消费本文合同，不把执行状态机复制到 GUI。

## #347 已冻结的接缝

- `sync_run_service.py` 是产品入口。`SyncRunService` 暴露
  `start/resume/status/cancel/derive`，`build_production_sync_run_service()` 负责接入
  当前 TranslationPlan、Provider adapter、结果导出和 freshness 校验。
- 服务返回 `SyncRunStore.build_snapshot()` 的版本化字典；GUI 不应读取
  `state.sqlite3`、推断请求/attempt 状态或自行修复 lineage。
- 状态枚举、合法转换、错误码和 run 选择规则集中在
  `sync_run_contracts.py`。终态是 `completed/completed_with_errors/failed/cancelled`；
  `outcome_unknown` 属于请求/attempt 事实，不能自动重发。
- CLI 兼容入口是 `sync-start/sync-resume/sync-status/sync-cancel/sync-derive`。
  `--json` 与 `--output json` 使用同一 schema-v1 envelope；严格退出码固定为：
  partial `3`、failed/cancelled `4`、选择/schema/freshness `5`、busy `6`。
- `sync-status` 是纯存储查询，`freshness.status=not_checked`；只有可能继续调用
  Provider 的 `start/resume/derive` 才重建当前执行上下文并校验 freshness。
- 终态不会授权写回。耐久结果必须经过离线 `check <RUN>`，再由
  `apply <RUN>` 消费绑定 preview。`--force` 不绕过 stale check、源快照、质量
  blocker、adapter 或制品哈希校验；重复 apply 是无副作用的 `already_applied`。
- 稳定制品包括 `plan.json`、`run_manifest.json`、`targets.json`、
  `results.jsonl`、`requests.jsonl`、`events.jsonl`、check manifest 和绑定 preview。
  GUI 应展示这些公开制品与 snapshot 字段，不展示数据库内部表。

## #348 所有权

#348 决定统一翻译页面、长期配置结构和迁移、新用户默认值、最终命令别名与产品文案。
接入时应：

1. 直接调用 `SyncRunService`，把关闭本地 worker与不可逆的取消 run 分成两个动作；
2. 从 snapshot 展示进度、`next_action`、错误安全摘要和 artifact 路径；
3. 保留 `outcome_unknown` 的显式风险确认：重试必须同时确认可能重复调用/计费，
   或明确排除 unknown IDs；
4. 复用 `check -> bound preview -> apply`，不在 GUI 建立第二套质量或写回门禁；
5. 配置迁移只生成服务已经支持的 provider/model/profile/policy 输入，不修改已冻结 run；
6. 为 CLI 别名保留现有 `sync-*` 映射、JSON 字段和严格退出码兼容测试。

## 有界 Provider 中断冒烟

以下脚本必须显式确认一次可能计费的请求。每次执行最多调用一个 Provider 一次，
在 Provider 成功但 T3 尚未提交时强制退出，再验证恢复结果为 `outcome_unknown` 且
重复调用数为 0。缺少凭据会安全跳过；输出只含 provider、分类、计数和状态。

```powershell
python scripts/run_durable_sync_provider_smoke.py --provider-class gemini --acknowledge-billable-request I_ACKNOWLEDGE_ONE_BILLABLE_PROVIDER_REQUEST
python scripts/run_durable_sync_provider_smoke.py --provider-class litellm --litellm-provider deepseek --acknowledge-billable-request I_ACKNOWLEDGE_ONE_BILLABLE_PROVIDER_REQUEST
$env:DURABLE_SYNC_CUSTOM_API_KEY = '<custom provider key>'
python scripts/run_durable_sync_provider_smoke.py --provider-class custom --custom-base-url https://example.invalid/v1 --custom-model model-id --acknowledge-billable-request I_ACKNOWLEDGE_ONE_BILLABLE_PROVIDER_REQUEST
```

自动化测试使用假的子进程/Provider 验证同一中断边界，不消耗真实额度；真实三类
Provider 的命令只用于显式 opt-in 的环境验收。

## #348 接入验收清单

- GUI 启动、恢复、查询、取消、派生均只经服务层；没有 SQLite SQL。
- busy、freshness mismatch、unknown 风险确认和四种终态均有 UI/CLI 一致测试。
- 关闭窗口或 worker 不会隐式取消 run；恢复不会重发已 dispatched 的 attempt。
- preview/apply 路径可从 GUI 审查，重复 apply 不重复写文件、进度或 usage。
- 配置迁移、别名和文案变更不改变既有 schema-v1 machine envelope。
