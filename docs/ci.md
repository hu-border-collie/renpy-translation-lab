# CI 与定时集成检查

## Pull request 门禁

`.github/workflows/tests.yml` 在 pull request 和 `main` push 上运行：

- Windows / Linux unittest；
- 不安装 PySide6 的 CLI-only 测试；
- Linux offscreen GUI 测试。

这些检查不访问供应商 API，也不下载 Ren'Py SDK，必须保持确定、快速并作为合并门禁。

## Ren'Py 真实 SDK 集成

`.github/workflows/renpy-integration.yml` 每周定时运行，也可从 Actions 手动触发。它不会在普通 pull request 上运行，因为官方 SDK 约 146 MiB，并且 Ren'Py 8.5 的自动测试会经过真实 GUI / OpenGL 渲染路径。

工作流固定使用 Ren'Py 8.5.2：

- 下载：`https://www.renpy.org/dl/8.5.2/renpy-8.5.2-sdk.tar.bz2`
- SHA-256：`cf9ed145e5b32521a4b2caddb4cd3073c64259ac51e1f7aab94a8a8ff72b55c4`
- 官方校验来源：`https://www.renpy.org/dl/8.5.2/checksums.txt`

集成 runner 会把 `tests/fixtures/renpy_smoke/` 复制到临时目录，然后依次：

1. 生成 `schinese` 翻译模板并验证产物；
2. 用 `lint --error-code` 检查脚本；
3. 运行 `test smoke --report-detailed`，真实启动引擎并走完最小翻译 fixture。

所有生成的 `.rpyc`、`game/tl/`、save 和日志都留在临时目录，不修改仓库 fixture。

本地已有 SDK 时可复现：

```powershell
python -B scripts/run_renpy_integration.py --sdk C:\RenPy_Workspace\renpy-8.5.2-sdk
```

Linux 无桌面环境时，在 Xvfb 下执行同一 runner：

```bash
xvfb-run -a python -B scripts/run_renpy_integration.py --sdk /path/to/renpy-8.5.2-sdk
```

SDK 缺失、模板没有生成、lint 失败或 testcase 失败都会返回非零退出码和明确诊断。
