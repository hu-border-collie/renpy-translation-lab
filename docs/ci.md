# CI 与定时集成检查

## Pull request 门禁

`.github/workflows/tests.yml` 在 pull request 和 `main` push 上运行。各任务使用的依赖锁如下：

| 任务 | runner | 哈希锁 / 检查 |
|---|---|---|
| `dependency-locks` | Linux | 离线校验 `manifest.json`、输入摘要、锁摘要与生成器元数据 |
| `litellm-lock-install` | Linux / Windows | 分别安装 `py311-linux-litellm.txt` / `py311-windows-litellm.txt` |
| `unittest` | Linux / Windows | 安装 `py311-gui.txt` 并运行完整 unittest |
| `cli-without-gui` | Linux | 安装 `py311-cli.txt`，确认没有 PySide6 并运行 CLI 测试 |
| `gui` | Linux | 安装 `py311-gui.txt` 并运行 offscreen GUI 测试 |

所有安装均使用 `pip --require-hashes`。这些检查不访问供应商 API，也不下载 Ren'Py SDK，必须保持确定、快速并作为合并门禁。

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

## 外部 provider 契约 smoke

`.github/workflows/provider-contract-smoke.yml` 每周定时运行，也可手动触发；它不在普通 pull request 上运行。Linux runner 使用 `pip --require-hashes` 安装 `requirements-lock/py311-linux-litellm.txt`，然后覆盖 Gemini 直连，以及生产 LiteLLM provider 目录中的 OpenAI、Anthropic、OpenRouter、DeepSeek 和 xAI。Windows LiteLLM 锁只在 pull request 的 `litellm-lock-install` 中验证安装，不执行真实 provider 请求。Ollama 依赖本地服务，不属于外部凭据 smoke。

仓库可按需配置以下 Actions secrets：

- `PROVIDER_SMOKE_GEMINI_API_KEY`
- `PROVIDER_SMOKE_OPENAI_API_KEY`
- `PROVIDER_SMOKE_ANTHROPIC_API_KEY`
- `PROVIDER_SMOKE_OPENROUTER_API_KEY`
- `PROVIDER_SMOKE_DEEPSEEK_API_KEY`
- `PROVIDER_SMOKE_XAI_API_KEY`

未配置的 provider 会打印 `SKIP` 并成功结束；至少配置一个 secret 时，对应 provider 会经过生产 adapter 发出一次请求，验证返回文本仍可解析为约定 JSON。任何配置过的 provider 失败都会打印 provider 名称和 `authentication`、`rate_limit`、`service_unavailable`、`invalid_response` 或 `provider_error` 分类，并让工作流失败。

每个 provider 的硬限制为 1 次请求、64 个输出 token、30 秒客户端超时；单次保守规划成本上限为 0.01 USD，六个 provider 全部配置时每轮估算不超过 0.06 USD。实际价格以 provider 当期计费为准。smoke 不读取或写入游戏项目文件。

本地可使用 provider 原生环境变量运行，例如：

```powershell
$env:OPENAI_API_KEY = "..."
python -B scripts/run_provider_contract_smoke.py --provider openai
Remove-Item Env:OPENAI_API_KEY
```
