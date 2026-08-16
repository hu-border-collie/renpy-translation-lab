# GUI 测试在 Codex 沙箱环境卡住：调查报告

> 结论先行：**不是测试代码问题，也不是机器负载问题，而是 Codex 沙箱的写限制**。
> GUI 测试在运行时会真实写入 `%LOCALAPPDATA%\renpy-translation-lab\litellm_catalog_cache.json`
> （LiteLLM 模型选择缓存，属 GUI 正常运行行为）。该路径不在 Codex 沙箱的可写范围内，
> 写入被拦截后表现为挂起或 `PermissionError`，导致整个 GUI 测试套件卡死。

## 现象

在 Codex 桌面应用（Windows）内运行 GUI 全量测试时：

1. `python -B tests/run_gui_tests.py -q` 超过 600 秒仍不结束；
2. shell 超时后 Python 进程**不会自动退出**，残留在后台持续占用 CPU；
3. 反复运行都卡在同一位置，看似死循环。

CLI 全量测试（1113 项，约 47 秒）与改动相关的少量 GUI 测试不受影响。

## 时间线

| 时间 | 事件 |
|---|---|
| 2026-08-13 23:35 | 第一次 GUI 全量运行，600 秒超时后进程残留（PID 45836） |
| 2026-08-14 01:04 | 误以为进程未清理干净，再次后台启动（PID 44072），造成 CPU 争抢 |
| 2026-08-14 01:06 | 终止两个残留进程；用户报告系统卡顿 |
| 2026-08-14 | faulthandler 定位卡点；对照实验确认根因 |

## 调查方法

### 1. 堆栈快照定位卡点

写了一个带 `faulthandler.dump_traceback_later(20, repeat=True)` 的诊断 runner，
每 20 秒 dump 一次运行中堆栈。连续 4 次快照完全一致，指向：

```text
tempfile._mkstemp_inner
  → atomic_io.atomic_write
  → gui_qt/litellm_catalog_cache._save
  → app.py _save_litellm_cache（设置页构建时保存模型选择）
  → tests/test_gui_control_layout_audit.py:157
```

测试逻辑本身没有死循环——它卡在操作系统创建临时文件这一步。

### 2. 对照实验排除机器负载

同一个 Python 进程内做最小写入探针：

| 路径 | 结果 |
|---|---|
| 系统 TEMP 目录 | `mkstemp + fsync + replace` 约 **3 毫秒**，正常 |
| `%LOCALAPPDATA%\renpy-translation-lab\` | 直接 `open()` 报 `PermissionError`；`mkstemp` 挂起 120 秒 |

磁盘 IO 正常，排除 iCloud / OneDrive 负载是主因。

### 3. 权限与身份确认

- `whoami /groups`：当前进程属于 `CodexSandboxUsers` 组（沙箱身份）；
- `icacls`：`%LOCALAPPDATA%\renpy-translation-lab` 的 ACL 中
  `CodexSandboxUsers` 只有 `RX`（读 + 执行），**没有写权限**；
- Codex 会话的可写范围只有工作区目录（`workspace-write` 模式），AppData 不在其中。

证据链闭合：沙箱内进程写沙箱外路径 → 被拦截 → 挂起/拒绝 → 套件卡死。

## 影响范围

- 在 Codex 沙箱内运行 GUI 全量测试**必然**卡在
  `test_gui_control_layout_audit.py::test_settings_sections_controls_no_clip_or_overlap`
  附近的 cache 写入上；同一测试文件的其他用例以及任何构建设置页并触发
  LiteLLM cache 保存的用例都可能受影响。
- CLI 测试不受影响（写 TEMP / 测试内临时目录，均在沙箱允许范围）。
- CI 的 GUI 测试 job 在 Linux offscreen 环境运行，不受此问题影响。
- 与 #338 的代码改动无关：在写 cache 之前的位置并无本次改动介入。

## WSL2 只读 home 环境（实测补充）

- 与 Codex 沙箱的差异：当前 WSL2 环境根文件系统为只读（`ro`），
  仅仓库目录是单独的 `rw` 挂载，`/tmp` 可写；默认缓存路径
  `~/.local/state/renpy-translation-lab/litellm_catalog_cache.json` 同样不可写。
- 但 `atomic_write_text` 会**立即**失败并返回 `OSError 30 (Read-only file system)`，
  不会像 Codex 沙箱那样挂起；`gui_qt/app.py` 的 `_save_litellm_cache()` 会捕获
  该 `OSError` 并仅记录日志，因此 GUI 测试可以继续跑完。
- 实测（PySide6 6.11.1 + `QT_QPA_PLATFORM=offscreen`）：
  `test_settings_sections_controls_no_clip_or_overlap` 约 0.5 秒通过；
  整个 `tests/test_gui_control_layout_audit.py` 3 个用例 1.136 秒全部通过，
  无进程残留。
- 注意：当前环境未安装 `litellm` 时，该测试不会触发缓存 `_save`（埋点调用次数为 0），
  所以“测试通过”不能证明缓存路径可写；需要直接写入探针验证。
- 规避：仅需当前 WSL 会话缓存时，将 `XDG_STATE_HOME` 指到可写目录，例如
  `XDG_STATE_HOME=/tmp/renpy-translation-lab-state`；实测同一写入路径约 0.79 毫秒成功。
  需要跨 WSL 重启持久化时，将 `XDG_STATE_HOME` 指到可写且持久的挂载点。
- 代码已增加自动回退：默认缓存目录不可写时，LiteLLM 缓存会自动写到系统临时目录，
  并在 GUI 日志/状态栏提示；因此不再需要手动设置 `XDG_STATE_HOME` 也能避免 `OSError`，
  但临时目录在重启后不会保留。
- 回退探测基于对最近可写祖先目录的**真实写入检查**（而非仅 ACL 模拟），因此同一修复
  也覆盖 Windows Codex 沙箱的写入拦截场景（见 #360）。

## 规避与修复

### 在 Codex 会话内

- 自 #369 起，GUI 运行时的 LiteLLM 缓存会在默认目录不可写时自动回退到系统临时目录，
  沙箱内跑 GUI 全量测试不再卡在缓存写入上；如仍想缩小范围，可只跑改动相关文件（如
  `python -m unittest tests.test_gui_settings_schema tests.test_gui_sync_translation_report`）。
- 确需排除运行时环境因素时，可在沙箱外运行（需要用户批准 escalation），例如直接在本机终端执行
  `python -B tests/run_gui_tests.py -q`。

### 长期建议（可选）

- 代码已实现自动回退（默认目录不可写时改用系统临时目录），沙箱与只读 home 场景都
  不再依赖修改 Codex 沙箱配置；若仍希望缓存跨重启持久化，可在 Codex 沙箱配置中把
  `%LOCALAPPDATA%\renpy-translation-lab` 加入可写范围；
- 注意：shell 命令超时**不会杀死残留的 Python 测试进程**，排查时先
  `Get-Process | Where-Object { $_.ProcessName -like '*python*' }` 确认无残留再重跑。
- 测试隔离提醒：GUI 测试共享用户级 LiteLLM 缓存（默认目录不可写时位于系统临时目录），
  若缓存中存在历史 provider/model 选择，`test_workspace_action_bar_uses_immediate_save_context`
  可能偶发判定“有未保存的更改”而失败；清理缓存（如删除 `%TEMP%\renpy-translation-lab`）后
  全量可稳定通过。该问题在正常 Windows 环境同样存在，属于测试共享状态的既有隔离缺陷。
