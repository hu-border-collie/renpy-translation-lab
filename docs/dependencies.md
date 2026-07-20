# 依赖输入与哈希锁

## 所有权

仓库把人工维护的直接依赖与生成的传递依赖分开：

| 输入 | 所有者 / 用途 |
|---|---|
| `requirements-genai.txt` | Google GenAI SDK（主翻译路径） |
| `requirements.txt` | 核心 CLI 组合入口（不含关系分析器） |
| `requirements-gui.txt` | 可选 PySide6 GUI |
| `requirements-litellm.txt` | 可选 LiteLLM 同步后端 |
| `requirements-relation-analyzer.txt` | 可选关系 / 语义分析科学栈 |
| `relation_analyzer/requirements*.txt` | 只引用上述权威输入，不复制版本 |

直接依赖更新应单独提交，说明升级原因并重新生成全部锁；不要直接编辑 `requirements-lock/*.txt` 或 `manifest.json`。

关系分析器是可选功能：普通 CLI / GUI / LiteLLM 安装**不包含** NumPy、Matplotlib、scikit-learn/SciPy、Pillow。语义模式在安装分析器后，再组合已有 GenAI 依赖即可。

## 可复现安装

哈希锁以 Python 3.11、x86-64 Windows 与 glibc 2.34+ Linux 为受验证目标。Linux 下限来自 PySide6 6.11.1 提供的 wheel。选择平台与功能 profile：

```powershell
# Windows/Linux CLI（翻译主路径，含 google-genai）
python -m pip install --require-hashes -r requirements-lock/py311-cli.txt

# Windows/Linux GUI
python -m pip install --require-hashes -r requirements-lock/py311-gui.txt

# Windows CLI + LiteLLM
python -m pip install --require-hashes -r requirements-lock/py311-windows-litellm.txt

# 可选：关系分析器（relation 模式）
python -m pip install --require-hashes -r requirements-lock/py311-relation-analyzer.txt

# 可选：语义分析 = 分析器 + GenAI（分两次哈希安装）
python -m pip install --require-hashes -r requirements-lock/py311-relation-analyzer.txt
python -m pip install --require-hashes -r requirements-lock/py311-cli.txt
```

Linux LiteLLM 使用 `py311-linux-litellm.txt`。CLI / GUI / 关系分析器的通用锁由 uv universal resolution 生成；只有依赖树确实不同的 LiteLLM 保留平台锁。其他 Python 版本仍可从直接 requirements 安装，但不属于提交锁覆盖的可复现环境。

开发入口（无哈希）：

```powershell
pip install -r requirements.txt
pip install -r requirements-gui.txt
pip install -r requirements-litellm.txt
pip install -r requirements-relation-analyzer.txt
# 或 semantic 组合：
pip install -r relation_analyzer/requirements-semantic.txt
```

GUI「设置 → 扩展」也可在当前解释器中安装 / 修复 / 更新关系分析器（使用提交的 Python 3.11 哈希锁与 `--require-hashes`）。

## 生成与校验

锁生成器固定为 `uv 0.11.29`：

```powershell
python -m pip install uv==0.11.29
python scripts/compile_dependency_locks.py --upgrade
python scripts/compile_dependency_locks.py --check
```

普通重生成不传 `--upgrade`，会尽量保留现有传递版本；有意更新依赖时才使用 `--upgrade`。生成器为 CLI / GUI / 关系分析器生成 Python 3.11 universal lock，并分别解析 Windows/Linux LiteLLM profile，为每个发行文件写入 SHA-256。`.gitattributes` 强制依赖输入、锁和生成器使用 LF，确保 Windows/Linux checkout 的字节摘要一致。`requirements-lock/manifest.json` 同时记录该规则、所有直接输入和锁文件的摘要，因此 CI 能在不访问网络的情况下发现：

- 直接 requirements 已修改但锁未重生成；
- 生成锁被手工修改；
- 生成器版本、Python 基线或 profile 集合发生变化。

CI 覆盖锁文件的方式如下；完整任务映射见 `docs/ci.md`：

- `dependency-locks` 离线检查 manifest、输入、生成器元数据与全部锁是否同步；
- `unittest` 在 Windows/Linux 安装通用 `py311-gui.txt`，`cli-without-gui` 与 `gui` 在 Linux 分别安装 `py311-cli.txt` 和 `py311-gui.txt`；
- `cli-without-gui` 额外断言分析器专用发行包未安装；
- `relation-analyzer` 安装 `py311-relation-analyzer.txt`（并组合 CLI 锁以覆盖 semantic 所需 GenAI）后运行分析器测试；
- `litellm-lock-install` 在 Windows/Linux 分别安装对应 LiteLLM 平台锁；
- 定时 `provider-contract-smoke` 安装 Linux LiteLLM 锁后才执行真实 provider 请求。

上述安装均使用 `pip --require-hashes`；pull request 中的 LiteLLM 任务只验证干净环境安装，不访问 provider API。

## 依赖漏洞审计

PR 的 `quality` 任务会对全部提交的哈希锁运行 `pip-audit`（见 [CI 与定时集成检查](ci.md)）。已知、已接受的例外写在 `quality/pip-audit-exceptions.json`；每条需说明原因与复查日期。清除例外的正确方式是在独立依赖升级 PR 中抬高 pin 并重生成锁，而不是在应用功能 PR 里顺手改版本。
