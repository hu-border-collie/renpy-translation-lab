# 架构概览

本文描述当前 `main` 的维护者边界。面向使用者的操作步骤以快速开始和工作流文档为准；
规划中的结构以 `docs/plans/` 为准。

## 分层

1. **CLI 与 GUI 入口**
   - `gemini_translate.py`、`gemini_translate_batch.py` 提供 CLI 和机器 JSON 合同。
   - `gui_qt/app.py` 是当前桌面壳；工作台页面与 workflow 负责把用户动作映射到共用服务或 CLI。
2. **项目与配置**
   - `translator_runtime.py` 从代码默认值、`api_keys.json`、环境变量和
     `translator_config.json` 构造运行时快照。
   - `gui_qt/project_state.py` 负责 GUI 的项目路径及 JSON 文件读取/保存。
   - `gui_qt/settings_schema.py` 描述高级设置字段；当前 Settings 总体 load/save/dirty
     编排仍位于 `gui_qt/app.py`。
3. **模型路由与请求计划**
   - `model_profile.py` 把现有 `sync.*` / `batch.*` 只读解析为 provider-neutral
     ModelRoutingPlan。它的 `primary`/`batch` 等 ID 是兼容 slot，不是长期用户 ID。
   - `translation_plan.py` 冻结 source snapshot、模型路由、上下文、prompt/schema 和预算。
   - #348 的 schema-v1 合同位于 `model_routing_config.py`，目前尚未接入生产读取。
4. **执行与耐久状态**
   - CLI 耐久路径：`sync-start` → `sync_run_service.py` / `SyncRunService`；状态机和机器错误位于
     `sync_run_contracts.py`，SQLite 细节不暴露给 GUI。
   - GUI Sync 页仍由 `gui_qt/sync_translation_workflow.py` 启动 `gemini_translate.py`，
     不持久化可恢复 run；统一页接入 #347 服务属于 #348 P3。
   - Gemini Batch 继续使用 Batch 生命周期，但与 Sync 消费相同 TranslationPlan 合同。
5. **检查与写回**
   - 模型结果先规范化并检查，再生成绑定 preview，最后由公共 apply 安全层写回。
   - `--force` 不得绕过 stale check、源快照、结构 blocker 或 adapter 写回计划验证。
6. **引擎边界**
   - Engine adapter 负责 discovery、candidate inventory、coverage、occurrence、relocation、
     translation validation 和声明式 writeback plan。
   - 公共层负责 manifest/result/project identity、实时源复核和原子事务写入。

## 配置事实来源

当前生产事实仍是旧 `translator_config.json`：

- `sync.backend/model/models/custom_litellm_providers` 描述同步模型；
- `batch.model` 及阶段 `model` 描述 Gemini Batch 和少数阶段覆盖；
- GUI 从原始 JSON 加载并在原对象上修改，以保留未建模字段；
- API Key 值只来自 `api_keys.json`、环境变量或系统 keyring，不属于项目配置。

#348 将以版本化 `model_routing` 替代模型相关旧字段。迁移期必须选择完整的新或旧事实
来源，不允许逐字段混合。合同见
[Model Routing 配置与迁移合同](plans/issue-348-model-routing-config-contract.md)。

## GUI Settings 收缩方向

`MainWindow` 当前仍拥有页面构建、load、collect、validate、dirty snapshot、保存和离开保护。
#202 的目标是引入 Settings coordinator 与页面 contract，让页面只处理自己的字段，
同时保持一个原始 JSON config store 和一个保存事务。#348 的模型页面必须接入该 contract，
不能另建配置缓存或保存入口。

具体维护路径见 [代码路径索引](code_paths.md)。
