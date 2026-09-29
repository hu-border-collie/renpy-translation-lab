# 维护复杂度治理路线

> 状态：2026-09-29；第一、二轮工程均已合并并完成关闭，后续仅为候选。
> 第二轮实现基线：[`main@585d608`](https://github.com/hu-border-collie/renpy-translation-lab/commit/585d60829736b4686ed977303c92c87639cba157)。
> 第二轮交付：[PR #534](https://github.com/hu-border-collie/renpy-translation-lab/pull/534)，合并提交 [`e792a03`](https://github.com/hu-border-collie/renpy-translation-lab/commit/e792a03366c3606e3cd92b7eade6444b0a8e81f9)。
> 本文管理方向、优先顺序和停止条件；实现事实见[架构概览](../architecture.md)与[调用链索引](../code_paths.md)。

## 目标与工作方式

保持现有功能和公开行为，逐步减少共享可变状态、对入口脚本的反向依赖及重复实现。
每次只启动一个边界明确的切片；实现 issue 承担本轮范围与验收，完成后可独立关闭。
候选方向必须在上一轮完成后重新核对代码和收益，才决定是否立项；文档顺序不构成交付承诺。

每轮记录三类证据：删除了什么平行实现或兼容分支；哪些依赖方向改变；状态由谁唯一持有。
文件行数可以作辅助统计，不作为目标；搬文件后仍反向导入入口、用回调继续读取全局变量，
或留下新旧两套长期实现，都需要说明尚未解决的依赖。

## 已完成：第一轮

[#528](https://github.com/hu-border-collie/renpy-translation-lab/issues/528) 已于 2026-09-27 完成关闭：

- [PR #529](https://github.com/hu-border-collie/renpy-translation-lab/pull/529)：清理 Settings 动态属性转发、平行状态和重复控件 fallback，迁移相关测试；保留接口的实际调用者已登记在架构文档。
- [PR #530](https://github.com/hu-border-collie/renpy-translation-lab/pull/530)：共享请求执行下移到 `sync_request.py`，service 不再为执行请求导入 CLI；入口保留装配和单向薄转发。

两份 PR 已合并；验收及 CI 证据见 #528 完成记录。该轮解决的是这两个切口，未完成整个 runtime 的状态隔离或主窗口生命周期治理。

## 已完成：第二轮耐久同步请求依赖隔离

执行单：[#531](https://github.com/hu-border-collie/renpy-translation-lab/issues/531)，已于 2026-09-29 按 completed 关闭；PR #534 于 2026-09-28 合并。最新 Tests 工作流全部通过，固定提交独立审查发现的默认 timeout 缺口已修复并复验。完成证据见 issue 与 PR；真实 Provider / 项目验收仍归 #344 / #431。

原有 `sync_request.runtime_dependencies()` 复制部分配置，但客户端工厂与凭据预算/轮换回调
仍读取 `translator_runtime` 的可变状态。已用生产 adapter + fake transport 重现 A 创建后
加载 B，A 请求选用 B 的 Gemini key；这是离线行为复现，不代表已发生真实生产事故。
本轮把默认 service 装配提前至创建时，并使 durable CLI 显式装配捕获自己的 key 选择。
custom providers 深复制；LiteLLM 内部同 attempt 限流换 key 已关闭。没有新增持久配置，
密钥不进入 plan、数据库或可见 repr。跨进程 resume 重新装配，外部 env/keyring 保留按引用解析。
独立审查发现默认 durable policy 的 120 秒覆盖已绑定 timeout；已改为新 run 从 service 绑定值
生成默认 policy，并以真实 start→fake transport 及 derive policy 持久值验证。显式 policy
仍优先，resume 沿用存储的 policy。

本轮只处理 durable Provider 请求链：在生产 service 创建时绑定所需配置与凭据选择，
显式传入和默认装配路径都不因随后加载另一项目而改变已绑定任务。复用现有配置和请求对象，
隔离嵌套可变值及回调捕获状态，保持 frozen routing 与单 attempt 单次 Provider 调用合同。

关键验收是同进程 A/B/A 交错执行：通过生产 adapter 与 fake transport 观察 A 的请求依赖
未被 B 的初始化污染，而非只检查 dataclass 字段。保留禁止导入 CLI 的边界回归。
凭据仅存在于必要内存上下文；恢复时重新装配并执行既有 freshness 检查，不持久化密钥。
环境变量/keyring 等外部凭据源沿用既有合同，不承诺冻结其所有外部变化。

本轮结束条件：#531 的依赖清点、实际调用隔离、行为回归和文档更新完成。
不要求全部全局变量清零，不宣称整个进程已支持并发多项目；非 durable 入口及轮换行为保留现状。

## 候选后续：按实际收益另行选择

| 候选 | 启动依据 | 首个切片的约束 |
| --- | --- | --- |
| 主窗口任务生命周期 | 同一任务的启动、输出、结束、取消与清理仍需跨多个窗口分支修改 | 选一个真实 workflow，复用已有生命周期设施；主窗口保留展示与顶层装配，验证 stale result、项目切换和 shutdown |
| 其余 CLI 反向依赖 | 某个实际服务仍必须导入整个 CLI 才能独立调用或测试 | 一次迁移一个使用链，依赖下移并迁调用者，删除旧平行实现；不单为移动函数新建通用框架 |
| 其余全局状态与缓存 | 仍有可复现的初始化顺序依赖、跨任务污染或反复修改共享状态的维护成本 | 明确任务/项目/进程所有权和缓存失效，再选择局部迁移；不预先换库或重做全部持久化 |

当前不为这三项批量创建实现 issue。每个候选都需具体调用点、收益、保留行为、删除清单和结束条件。
若清点后收益不足，允许继续暂缓或取消；已验证共用的写回、身份与质量检查继续复用。

## 验证、维护与停止条件

- 每轮 PR 同时说明减少的依赖/分支/状态副本及保留理由，更新对应 issue 和本文状态。
- 依照 [AGENTS.md](../../AGENTS.md) 与 [CONTRIBUTING.md](../../CONTRIBUTING.md) 做受影响行为回归；保留断言意义，避免测试迫使生产代码长期模拟旧结构。
- 不改变公开 CLI、配置与产物合同，不削弱 `check -> apply`、源快照或事务写回保证。必要外部兼容集中在边界，并记录调用者。
- 真实 Provider/项目验证继续由 [#344](https://github.com/hu-border-collie/renpy-translation-lab/issues/344)、[#431](https://github.com/hu-border-collie/renpy-translation-lab/issues/431) 等既有任务承接；本路线的离线验证不能代替它们。
- [#510](https://github.com/hu-border-collie/renpy-translation-lab/issues/510) 负责全量计划状态整理；本文只管理维护性路线，不重新安排其他功能队列。
- 一轮验收通过即可关闭，后续扩展不自动成为关闭条件。没有足够收益证据时暂停下一轮；不以“项目所有复杂度消失”为无限期完成标准。
