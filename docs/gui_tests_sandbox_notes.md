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

## 规避与修复

### 在 Codex 会话内

- 不跑 GUI 全量测试；改为只跑改动相关文件（如
  `python -m unittest tests.test_gui_settings_schema tests.test_gui_sync_translation_report`）。
- 确需全量时，在沙箱外运行（需要用户批准 escalation），例如直接在本机终端执行
  `python -B tests/run_gui_tests.py -q`。

### 长期建议（可选）

- GUI 测试对 `%LOCALAPPDATA%` 的写入属于正常运行时行为，测试套件本身不需要改；
- 若希望沙箱内也能跑 GUI 测试，需要在 Codex 沙箱配置中把
  `%LOCALAPPDATA%\renpy-translation-lab` 加入可写范围；
- 注意：shell 命令超时**不会杀死残留的 Python 测试进程**，排查时先
  `Get-Process | Where-Object { $_.ProcessName -like '*python*' }` 确认无残留再重跑。
