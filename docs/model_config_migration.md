# 模型配置离线迁移（#348 P1）

P1 提供可复用迁移器、兼容 reader 与开发者 CLI，用于在配置副本上验证 schema-v1。
它不会加载密钥、连接模型、执行 prepare 命令或改变游戏文件。

**P2 已接入生产读取。** 配置含合法 `model_routing` 时，新任务使用其中的模型、Provider、
阶段路由与 embedding 连接；保留的旧模型字段不参与补值。缺少新 section 时保持旧行为；
新 section 无效时拒绝执行。已有 manifest 的冻结路由优先于当前配置。

`migrate --stage-only` 仍只修改指定文件，不启动任务、不自动安装配置；若指定生产配置，
新 section 在下次加载时生效。请先在副本上预览和验证，再替换生产配置。
GUI 原有 workflow 调用相同 CLI，因此同样消费新配置；统一 Model Profiles 表单与耐久任务
页面仍属于 P3。本阶段 GUI 模型页显示兼容说明，保存时校验并保留新 section 和未知字段。

现有兼容命令使用 `legacy_entrypoints` 角色指针；项目分析保持 Sync、最终审校保持 Gemini
Batch。跨 Provider 的模型字符串覆盖、Gemini 非 `api_keys_json/api_keys` 凭据槽和非空生成
profile `params` 尚未接入旧命令，会明确拒绝，不能静默忽略。生成请求参数继续保存在
`sync` / `batch` 执行设置中；embedding profile 的 `params` 已按独立连接读取。
P3 再提供统一 profile/strategy 选择和完整参数编辑表面。

生产示例 `translator_config.example.json` 已添加等价的 v1 section，旧字段保留用于回滚，
默认模型和 Sync/Batch 策略不变。
v1 结构示例见 [合同 fixture](../tests/fixtures/model_routing_config_v1.json)。

## 操作

先手动准备配置副本，使用明确路径；工具不默认定位或自动迁移实际配置。

```powershell
python model_config_migration.py preview --config <CONFIG_COPY> --json
python model_config_migration.py migrate --config <CONFIG_COPY> --expected-fingerprint <SOURCE_FINGERPRINT> --stage-only --json
python model_config_migration.py rollback --config <CONFIG_COPY> --report <MIGRATION_REPORT> --json
```

`SOURCE_FINGERPRINT` 必须来自刚才的 preview；`MIGRATION_REPORT` 来自 migrate 返回的
`report_path`。GUI「诊断与运行日志」的任务命令参考也提供这三个模板。

文件迁移要求明确的 `batch.model`，不会读取外部游戏配置的 `batch_model`。纯内存
API `preview_migration(config, game_config=...)` 可由后续服务传入已确认的项目默认值。
所有命令支持 `--help`；成功退出码为 0，拒绝为 2。拒绝结果包含稳定的
`MODEL_CONFIG_MIGRATION_REFUSED`、分类 `reason` 和可操作的 `next_action`，不会回显原配置
或 Provider 异常正文。

## 保持的行为

- 产品默认 profile 为 `legacy-batch`，策略为 `gemini_batch`。
- `legacy_entrypoints` 冻结旧 Sync/Batch 命令的 profile 指针，不从旧字段逐个补值。
- 初译、术语、订正继承该入口的 base；项目分析显式固定 Sync，最终审校显式固定 Batch。
  即使旧阶段没有指定模型，也保存所需的 profile/strategy 覆盖，以保留原行为。
- 未选择的自定义 Provider 也保留连接元数据和凭据引用。
- Sync 和 Batch 的 embedding 分别冻结为 `purpose=embedding` 的 profile，保留原
  backend、endpoint、模型、维度、task type 和 timeout；RAG policy 留在原位置。
  `openai_compatible` embedding 的 adapter 语义保存在 params，不能当作生成路由使用。
- 未知字段和原旧字段的 JSON 值保持不变；迁移不重建整个配置来丢弃未知字段。
- 新 section 存在却无效时，兼容 reader 拒绝，绝不退回旧模型。

迁移前会比较两种执行方式下各阶段的实际模型、轮换、adapter、凭据引用、连接、
能力与策略。测试还对照 `translator_runtime.load_sync_translation_settings` 的真实
模型选择，避免只比较两个互相复制的映射。

若 `sync.models` 首项与 `sync.model` 不一致、缺少明确模型而旧入口默认值可能不同，
或启用了未冻结的隐式轮换池，迁移会拒绝。应先确认实际需要保留的模型选择，再在副本
中明确模型及完整轮换列表，而不是由迁移器替用户选择。

## 文件事务与回滚

1. 严格解析 UTF-8/BOM JSON object；拒绝重复键、非有限数、无效 schema 和凭据值字段。
2. 持有 GUI/迁移共同使用的 `.write-lock`，核对 preview 的源指纹。
3. 创建同目录、带时间戳和随机 ID 的独占 `.bak`，保存完整原始字节。
4. 在替换前持久化迁移报告，包含源/目标指纹、备份路径、ID、映射字段及暂存状态。
5. 临时文件 flush/fsync 后再次核对源字节，再原子替换配置。保留 BOM 和 CRLF/LF 风格。
6. 备份、报告、临时文件写入内容前，保留 POSIX mode 或复制 Windows DACL，避免扩大访问权。

报告中的 `staged` 表示已准备好目标及回滚证据，不独立证明替换已经发生；当前配置的
目标指纹才是提交证据。替换失败可能留下备份/报告，但原配置不受影响。进程在提交后
被终止，报告也已经落盘，仍可回滚。重复迁移合法 v1 返回 `already_current`，不改变
文件字节、不新增备份。

回滚要求报告与配置同目录、备份位于指定目录、原始备份指纹匹配，且当前配置仍等于
迁移时的目标指纹。迁移后哪怕只改了空白，自动回滚也会拒绝；需先人工保留编辑并合并。
成功回滚恢复精确原始字节，保留报告和备份用于审计。

锁只协调本工具和 GUI；外部编辑器不遵守该锁，最终源字节复核也不是操作系统级 CAS。
迁移期间应关闭其他配置写入者。GUI 和迁移共用仓库文件锁协议：竞争时最多等待一秒。
该锁是 kernel 级进程锁（POSIX `flock` / Windows `LockFile`，见 #474）：进程退出或崩溃
时由操作系统自动释放，因此不再按锁文件年龄抢占，也不存在两个恢复者互相删除锁的
read-then-unlink 竞态；锁文件释放后保留，不应手动删除。latest 游标写服务使用独立的
非抢占文件锁，仍保留人工确认后清理遗弃锁的恢复方式。
超时后 GUI 显示中文提示及具体锁路径，CLI 返回 `config_locked`，可等待后重试。事务保证普通 I/O 失败和进程
中断下的恢复边界，不承诺所有文件系统断电场景下的持久性。

## 开发接口与后续

- `model_routing_migration.preview_migration`：纯内存候选与非敏感报告。
- `model_routing_reader.read_routing_plan`：共用 legacy/v1 reader；保留现有 ModelRoutingPlan。
- `model_routing_reader.read_embedding_settings`：独立读取对应路径的 embedding 连接与请求参数；
  新 section 存在时不再读取旧 RAG 连接字段。
- `model_routing_migration_store`：显式 preview/migrate/rollback 文件事务。
- `config_store`：GUI 与迁移共用原始 JSON 保存边界；不引入另一个配置状态源。

P2 复用现有能力/凭据检查、稳定错误分类和冻结任务兼容，并校验 embedding backend。
四类真实 Provider smoke 仍在 P4 验收，不以离线测试替代。P3 由 #202 的页面合同
承载首次迁移预览/确认、dirty/save 和模型设置，禁止在加载设置页时隐式迁移。迁移完成后可在「设置 → 模型与 Provider」统一维护 providers / profiles / 默认值与阶段路由；该页直接编辑 `model_routing` 并保留未知字段。

参见 [配置合同](plans/issue-348-model-routing-config-contract.md)、
[架构概览](architecture.md)、[代码路径](code_paths.md)。

GUI「诊断与运行日志」在空闲、Sync 和 Batch 上下文都显示这三个离线命令模板，
无需先加载 Batch 任务。`<CONFIG_COPY>` 始终是待替换的配置副本占位符。
