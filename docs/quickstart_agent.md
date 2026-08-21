# Agent / CLI 快速开始

文档地图：[docs/README.md](README.md)

本文面向**使用本工具完成翻译任务**的自动化 Agent。若任务是修改 Ren'Py Translation Lab 本身，请改读根目录 [AGENTS.md](../AGENTS.md) 和 [CONTRIBUTING.md](../CONTRIBUTING.md)。

CLI 是自动化操作的事实来源。必要时可以使用 GUI 做可视验证，但不要依赖点击界面完成可重复的无人值守流程。

## 操作原则

- 使用 Python 3.11 或更高版本，在仓库根目录运行命令。
- 先检查当前项目、配置和 Git 状态；只在游戏副本或已有备份的目录中工作。
- 先运行只读的 `doctor`，不要直接提交任务。
- 用 `python gemini_translate_batch.py --help` 和子命令 `--help` 读取当前参数，不要根据其他项目猜命令。
- 记录 `build` 输出的确切 manifest 路径，并在后续命令中显式传入；不要依赖“最新任务”推断。
- 只有 `check` 对当前 manifest/results 返回 `writeback_gate.decision=allow` 时才执行 `apply`。
- 不要用 `--force` 规避安全判断；它只处理有限的重复/恢复场景，不能绕过 stale check、源快照校验或 `block`。

## 机器可读输出

普通用户推荐使用 GUI；Agent、脚本和 CI 调用 Batch CLI 时，核心流程命令支持 `--output json`：

```powershell
python gemini_translate_batch.py doctor --output json
python gemini_translate_batch.py build --output json
python gemini_translate_batch.py submit <manifest> --output json
python gemini_translate_batch.py status <manifest> --output json
python gemini_translate_batch.py download <manifest> --output json
python gemini_translate_batch.py check <manifest> --output json
python gemini_translate_batch.py apply <manifest> --output json
```

P3/P4 的版本资产与译文复用命令也使用同一 envelope：

```powershell
python gemini_translate_batch.py export-project-snapshot --version-id <GAME_VERSION> --output json
python gemini_translate_batch.py reconcile-project-snapshots <base-snapshot> <target-snapshot> --output json
python gemini_translate_batch.py build-translation-records <snapshot> <manifest> --output json
python gemini_translate_batch.py build-reuse-candidates <base-snapshot> <target-snapshot> <reconciliation> <records> --output json
python gemini_translate_batch.py import-reuse-decisions <reuse-report> <decisions> --output json
python gemini_translate_batch.py export-reuse-results <reuse-report> <manifest> --output json
```

JSON 模式的 stdout 只包含一个 JSON 文档；banner、进度、warning、prepare 子进程输出和原有文本摘要会实时写入 stderr，完整 Batch 控制台日志仍会落盘。成功结果使用版本化 envelope：

```json
{
  "schema_version": 1,
  "command": "check",
  "ok": true,
  "status": "ready",
  "result": {},
  "artifacts": {},
  "warnings": [],
  "error": null
}
```

其中：

- `status` 表示业务状态，例如 Batch job state、检查结论 `ready / ready_with_warnings / blocked`
  （兼容旧字段 `safety_level` 的 `safe / warn / block`）、reconciliation
  的 `ready / attention` 或 `applied`；
- `result` 是命令摘要，`artifacts` 给出 manifest、results、检查报告等产物路径；
- 命令拒绝执行时 `ok=false`，`error.code` 与 `error.message` 用于程序判断和诊断；
- 默认退出码仍保持兼容。Agent 可同时传入 `--output json --strict-exit-codes`，让业务状态映射为稳定退出码；严格模式只与 JSON 输出组合使用。
- 无论是否启用严格退出码，都必须读取 `result.check.writeback_gate.decision`；只有它为
  `allow` 才能继续 `apply`。`quality_gate.decision=needs_review` 表示存在质量报警，默认不阻止
  `apply`，但需要在写回后继续人工处理。

当 argparse 在生成参数对象前就失败时，只要原始参数包含精确的 `--output json` 或 `--output=json`，stdout 仍会返回一个 schema v1 错误 envelope（`error.code=ARGUMENT_PARSE_ERROR`，退出码 `2`），原生 usage 和诊断保留在 stderr。此阶段尚未解析出完整参数，因此不会使用 `--output-file` 或 `--fields`；未能可靠识别 JSON 意图的最早期语法错误继续使用普通 argparse 文本。原始参数扫描遇到 `--` 后停止，其后的内容只作为 positional 数据处理。

严格退出码约定：

| 退出码 | 含义 | 常见场景 |
|---:|---|---|
| `0` | 命令成功，可继续或继续轮询 | `check` 返回 `ready`、job pending/running、无待处理工作 |
| `1` | 未分类的内部错误，默认不可重试 | 意外异常或未知 SDK 错误 |
| `2` | 命令行用法错误 | 参数缺失、严格模式未配合 JSON 输出 |
| `3` | 命令完成，但需要 Agent 处理 | `check` 返回 `ready_with_warnings` 或旧 `warn`，reconciliation 返回 `attention` |
| `4` | 被门禁阻止或进入终止失败状态 | `block`、doctor blocked、job failed/cancelled |
| `5` | 输入、配置或状态已失效 | stale check、manifest/results 漂移、前置产物缺失 |
| `6` | 远端临时错误，可稍后重试 | rate limit、quota、timeout、service unavailable |

严格模式的兜底错误 envelope 可能使用 `STALE_STATE`、`PRECONDITION_FAILED`、`COMMAND_BLOCKED`、`REMOTE_RETRYABLE`、`COMMAND_REFUSED` 或 `INTERNAL_ERROR`。核心前置条件优先返回稳定错误码：manifest 读取与身份问题使用 `MANIFEST_NOT_FOUND`、`INVALID_MANIFEST_JSON`、`INVALID_MANIFEST_ENCODING`、`INVALID_MANIFEST_SHAPE`、`MANIFEST_UNREADABLE`、`INVALID_MANIFEST_PATH`、`MANIFEST_PROJECT_IDENTITY_MISSING`、`MANIFEST_PROJECT_MISMATCH`、`MANIFEST_MODE_MISMATCH` 或 `MANIFEST_PACKAGE_DIR_MISSING`；Batch 输入问题使用 `BATCH_INPUT_NOT_FOUND` 或 `INVALID_BATCH_INPUT_JSON`；apply 门禁使用 `APPLY_CHECK_REQUIRED`、`STALE_CHECK_CONTRACT`、`STALE_CHECK_FINGERPRINT` 或 `UNSAFE_CHECK_STATUS`；未知 `fail_apply_preflight` reason 回落到 `APPLY_PREFLIGHT_FAILED`。

除 `UNSAFE_CHECK_STATUS` 按安全门禁退出 `4` 外，上述前置条件错误在严格模式下退出 `5`。同时读取 `retryable`、`suggested_action` 和权威的 `details.semantic_exit_code`，不要解析 `message` 文本。未启用严格模式时保持 schema v1 的兼容退出行为。

模型任务还会在 prepare、创建 package 或发送请求前校验本次阶段实际使用的
ModelProfile 与 ExecutionStrategy。拒绝时使用以下稳定合同：

| `error.code` | 含义 | `suggested_action` |
|---|---|---|
| `MODEL_PROFILE_INVALID` | adapter、模型格式、阶段路由或能力覆盖无效 | `fix_translator_config` |
| `MODEL_ROUTE_CAPABILITY_MISSING` | profile 不支持所选执行方式 | `choose_supported_strategy_or_profile` |
| `MODEL_PROFILE_CREDENTIAL_REF_MISSING` | env/keyring 凭据引用无法解析 | `inspect_configuration_and_artifacts` |

三者均为不可自动重试的配置错误，严格模式退出 `5`。`details.issues` 包含本次活动阶段的完整问题列表；不要因失败而切换到另一个 Provider 或模型。
如果配置在 plan 构建前就无效（例如 `sync.backend` 拼写错误），任务入口同样归一为
`MODEL_PROFILE_INVALID`，不会向 Agent 暴露裸 `ValueError`。

`doctor --output json` 的 `result.model_routing.status` 只有 `ok` 或 `attention`，表示只读诊断，**不会**把 doctor 退出码改成 blocked。Agent 不要看到 `"status": "attention"` 就拒绝启动同步翻译；真正阻断启动的是活动阶段的上述错误码。未使用的终审 / Batch 配置无效只会进入 `attention` 与 warnings。


## 严格非交互与显式 manifest

七个核心命令支持 `--non-interactive`。当前核心流程本身不会读取 stdin；该选项进一步禁止 manifest 消费命令使用隐藏 target：

```powershell
python gemini_translate_batch.py status <manifest> --output json --non-interactive --strict-exit-codes
```

在 `submit / status / download / check / apply` 中，`--non-interactive` 要求显式传入 manifest 路径或 package 目录：

- 不再读取 `latest_manifest.txt` 或扫描最新 package；
- `submit` 不再因 target 为空而隐式执行 build；
- 缺少 target 时返回 `error.code=EXPLICIT_TARGET_REQUIRED`、`suggested_action=pass_manifest_path`；
- 同时启用 `--strict-exit-codes` 时退出码为 `5`。

如果只想禁止 target 回退、但不需要声明完整非交互契约，可使用 `--require-explicit-target`。`doctor` 与 `build` 本来不消费 manifest，因此这两个命令在非交互模式下不要求 target。

默认模式保持兼容：未传这两个新选项时，现有 latest-manifest 与 submit-build 回退仍然可用。Agent 应优先使用 `--output json --non-interactive --strict-exit-codes`，并始终显式传递同一 manifest。

## 输出裁剪与文件输出

上述七个核心 JSON 命令与两个版本资产命令还支持三个可组合选项：

```powershell
python gemini_translate_batch.py status <manifest> --output json --compact
python gemini_translate_batch.py check <manifest> --output json --fields command status result.check.writeback_gate result.check.quality_gate artifacts.manifest
python gemini_translate_batch.py status <manifest> --output json --output-file .\status.json
```

- `--compact` 移除缩进和多余空格，但仍输出一个以换行结尾的合法 JSON 文档；
- `--fields` 使用点路径投影结果，保留嵌套结构；可一次传入多个路径、用逗号分隔，或重复传入该选项；
- 请求的可选路径不存在时会被省略，命令不会因此失败；列表不支持按下标继续投影，应选择整个列表字段；
- 空路径、连续点等非法字段路径会在 workflow 执行前返回 `error.code=INVALID_FIELD_PATH` 和退出码 `2`；
- `--output-file` 将最终 JSON 原子写入指定路径，并保持 stdout 为空；父目录会按需创建；
- 文件中的 `artifacts.output_file` 记录绝对输出路径，除非 `--fields` 主动将它裁掉；
- 三个选项都必须显式配合 `--output json`，不会改变文本模式。

字段投影只改变展示形状，不改变命令执行结果或严格退出码。例如 `check` 返回 `warn` 时，即使只输出 `status`，配合 `--strict-exit-codes` 仍退出 `3`。需要稳定 envelope 时不要使用 `--fields`；需要最小上下文时应至少选择 `command / ok / status / error` 以及当前动作所需的 `result`、`artifacts` 路径。

`--output-file` 适合把结果交给后续脚本或避免终端上下文膨胀：

```powershell
python gemini_translate_batch.py check <manifest> `
  --output json --non-interactive --strict-exit-codes `
  --compact --output-file .\check-result.json
```

成功写入时，命令的业务退出码保持不变；调用方随后读取自己传入的文件路径。CLI 会在 workflow 执行前探测明显不可写的目标，避免已知失败路径触发业务副作用。

若文件无法创建或原子替换，诊断写入 stderr，完整的 `OUTPUT_FILE_WRITE_FAILED` envelope 改为回退到 stdout（不应用 `--fields` 投影）；严格模式退出 `5`，兼容模式退出 `1`。调用方必须同时检查 `error.details.workflow_started` 与 `error.details.command_completed`：前者为 `false` 表示前置探测失败、workflow 未执行；前者为 `true` 而后者为 `false` 表示 workflow 已启动但以错误结束；两者都为 `true` 表示 workflow 已成功完成后结果文件落盘失败，写操作可能已经生效。后两种情况可结合 `original_ok`、`original_status` 或 `original_error_code` 判断原命令结果。

## 机器发现


Agent 可在不加载项目配置、不读取 API Key、也不触发 workflow 的情况下查询当前 CLI：

```powershell
python gemini_translate_batch.py capabilities
python gemini_translate_batch.py schema status
```

两个命令都直接向 stdout 输出 `schema_version=1` 的 JSON：

- `capabilities` 返回 CLI 版本、结果契约版本、完整命令索引，以及每个命令是否支持 JSON、严格退出码、非交互和显式 target；
- `schema <command>` 返回该命令当前的 positional/options、类型、required、repeatable、choices、默认值和帮助文本；
- `capabilities.commands` 也声明各命令是否支持 compact、字段投影和文件输出；`schema` 给出这些选项的实时 argparse 形状；
- schema 直接从现行 argparse 定义生成，`--help` 仍是人类阅读的事实来源，不维护第二份手写命令表。

`capabilities` 与 `schema` 本身就是 JSON 命令，也接受 `--compact / --fields / --output-file`，但不需要也不提供冗余的 `--output json`。

没有单独提供 `commands` 命令，因为 `capabilities.commands` 已覆盖同一用途。输出裁剪是公开参数，不存在 discovery schema 之外的隐藏 Agent 行为。
没有 `--output json` 时仍使用原有人类可读文本。当前结构化模式承诺覆盖上面的七个
核心命令和两个版本资产命令；其他子命令以各自 `--help` 和落盘产物为准。

## 1. 安装核心依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python gemini_translate_batch.py --version
python gemini_translate_batch.py --help
```

Linux / macOS 激活命令为 `source .venv/bin/activate`。CLI 不需要安装 `requirements-gui.txt`。

## 2. 准备配置与项目

默认本地文件为：

| 位置 | 文件 | 用途 |
|---|---|---|
| 工具仓库根目录 | `api_keys.json` | Gemini API Key；也可使用 `GEMINI_API_KEY`、`GEMINI_API_KEY_2`、`GEMINI_API_KEY_3` 环境变量 |
| 工具仓库根目录 | `translator_config.json` | 当前 `game_root`、模型、过滤器和运行参数 |
| 当前游戏 `work` 目录 | `glossary.json` | 项目术语，可选 |
| 当前游戏 `work` 目录 | `macro_setting.md` | 角色口吻和世界观约束，可选 |
| 当前游戏 `work` 目录 | `project_context_settings.json` | 当前项目的 Batch RAG / 原文索引开关，可选 |

首次配置时，可从仓库中的 `*.example.*` 文件复制；如果目标文件已存在，先读取并保留现有值，不要直接覆盖。`translator_config.json.game_root` 应指向游戏的 `work` 目录，例如：

```json
{
  "game_root": "C:/games/Game_Example/work"
}
```

`api_keys.json` 保存在本机不等于请求内容留在本机。Gemini 调用会把认证信息、待译文本、提示词和启用的上下文发送给 Google；LiteLLM 同步调用会发送给所选 Provider。本项目没有自建模型中转服务。处理敏感文本前应核对供应商当前条款，并确保对发送的游戏内容拥有必要权限；详细边界见 [同步翻译工作流](sync_workflow.md#gemini-与-litellm-数据边界)。

典型项目结构：

```text
Game_Example/
├─ original/
├─ work/
│  └─ game/
│     └─ tl/
│        └─ <language>/
└─ build/
```

如果只有 `original/game` 且 `work` 不存在或为空，可运行：

```powershell
python gemini_translate_batch.py bootstrap-work
```

这只创建工作副本，不生成 TL 模板，也不调用 Gemini。生成模板所需的 Ren'Py SDK 和 prepare 配置见 [安装与本地配置](setup.md)。不要运行来源不明的 `prepare.unpack_command` 或 `prepare.template_command`；`translator_config.json` 是可执行的本地配置。

## 3. 只读预检

```powershell
python gemini_translate_batch.py doctor
```

`doctor` 不调用 Gemini，也不写回 `.rpy`。确认输出中的 `game_root`、`tl_subdir`、目标语言和待译数量符合预期。遇到阻塞项时先修复；可选建议不应被误判为强制失败。

## 4. 执行安全 Batch 流程

先构建本地任务包：

```powershell
python gemini_translate_batch.py build
```

从输出中取得 manifest，例如 `logs/batch_jobs/<package>/manifest.json`。后续始终使用同一个确切路径：

```powershell
python gemini_translate_batch.py submit logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py status logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py download logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py check logs/batch_jobs/<package>/manifest.json
```

执行约束：

- 提交前按任务要求设置并确认成本上限；参数以 `submit --help` 为准。
- `status` 显示云端任务成功后才运行 `download`；仍在运行时等待并再次查询，不要重复提交。
- `check` 是干跑校验，不修改 `.rpy`；它会输出 `writeback_gate` / `quality_gate`（文本模式仍保留
  `Safety status` 行），并把结构失败报告写入 `check_failures.jsonl`、质量报警写入 `quality_findings.jsonl`。
- `writeback_gate.decision=deny` 时停止写回，阅读 `check_failures.jsonl` 及命令输出，再按
  [Batch 工作流与安全检查](batch_workflows.md) 使用 retry、repair 或 revision 流程。
- `quality_gate.decision=needs_review` 不阻止写回，但必须在写回后按规则、文件与严重程度处理报警。

只有 `writeback_gate.decision=allow` 时才执行。`allow` 是**结构性写回安全**结论，不是译文质量验收：

```powershell
python gemini_translate_batch.py apply logs/batch_jobs/<package>/manifest.json
```

`apply` 写回前会再次验证 manifest、results 和当前源文本；任一方漂移都应视为需要重新检查，而不是绕过。

## 5. 完成与交付

- 检查 `apply` 摘要和目标 `.rpy` diff。
- 在 Ren'Py 中运行 lint 或项目既有 smoke test。
- 按 `quality_findings.jsonl` 运行机械质量检查，并对错译、术语、语气、反讽和上下文一致性进行人工/LLM 通读；不能把 `writeback_gate=allow` 报告成“译文质量合格”。
- 报告使用的 manifest、最终安全等级、写回结果和仍未处理的失败项。
- 不提交 `api_keys.json`、`translator_config.json`、私有游戏脚本、`logs/` 或 Batch 结果到公开仓库。

完整子命令和恢复流程见 [Batch 工作流与安全检查](batch_workflows.md)；配置、SDK 与目标语言见 [安装与本地配置](setup.md)。
