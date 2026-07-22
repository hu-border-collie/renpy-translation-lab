# GUI 快速开始

文档地图：[docs/README.md](README.md)

本文面向第一次通过图形工作台使用 Ren'Py Translation Lab 的用户，目标是从源码安装到完成一次**经过安全检查的 Batch 翻译**。完整界面说明见 [GUI 工作台](gui_workbench.md)，配置字段说明见 [安装与本地配置](setup.md)。

## 开始前

准备以下内容：

- Python 3.11 或更高版本；
- 一份待处理游戏的**副本或备份**；
- 可用的 Gemini API Key；
- 游戏的 `work` 目录，通常包含 `game/tl/<language>/`；如果只有 `original/game`，也可以稍后通过 GUI 准备 `work`；
- 如需从原始脚本生成 TL 模板，准备 Ren'Py SDK 路径。

本项目以源码形式运行，目前没有零配置安装包，也不负责游戏解包或重打包。

## 1. 安装并启动

在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-gui.txt
python -m gui_qt
```

Linux / macOS 激活命令为 `source .venv/bin/activate`。未安装推荐字体不会阻止 GUI 启动；界面会回退到系统字体。

## 2. 选择项目

打开左侧「项目与环境」，在顶部执行以下任一操作：

- 点击「指定本地目录…」，选择当前游戏的 `work` 目录；如果尚无 `work`，选择包含 `original/game` 的项目根目录；
- 或点击「切换项目」，到「设置 → 项目列表」使用 **「创建 / 接入工作区…」**：选择目录 → 预览总表初始化/接入选项 → 确认。成功后再从列表「切换到此项目」。

如果选择的项目根目录下已经存在 `work`，GUI 会提示并切换到该目录。若尚无 `work`，但项目中已有 `original/game`，点击「准备工作目录」创建 `work/game` 副本；这一步不会生成 TL 模板，也不会调用 Gemini。接入工作区本身也**不会**自动准备 `work` 或下载 Ren'Py SDK。

典型目录如下：

```text
Game_Example/
├─ original/
├─ work/
│  └─ game/
│     └─ tl/
│        └─ <language>/
└─ build/
```

## 3. 配置密钥和模型

打开左侧「设置」：

1. 在「密钥」中点击「管理 API Key」，添加并保存 Gemini API Key。密钥会写入本地 `api_keys.json`，不会上传或代理。
2. 在「模型」中确认同步与 Batch 模型。第一次使用可以保留推荐值。
3. 在「项目」中确认翻译目录、术语表和准备流程；需要生成或刷新 TL 模板时，再配置 Ren'Py SDK。
4. 修改模型、项目、上下文或高级设置后，点击底部「保存设置」。API Key 管理对话框会单独保存密钥，不依赖这个按钮。

如果项目需要专有术语或角色口吻，可在 `work` 目录准备 `glossary.json` 和可选的 `macro_setting.md`。没有它们仍可运行，但上下文质量可能较低。

## 4. 运行环境检查

回到「项目与环境」，点击「环境检查」。该检查不会调用 Gemini，也不会写回 `.rpy`。

- 有阻塞项：按摘要修正项目路径、TL 模板、SDK 或配置，然后重新检查。
- 只有可选建议：可以按项目需要处理，不等同于阻塞。
- 需要生成 TL 模板：先在「设置 → 项目」配置 Ren'Py SDK，再按界面提示生成模板。

环境检查通过前，批量翻译和同步翻译不会启动。

## 5. 完成一次 Batch 翻译

1. 打开左侧「批量翻译」。
2. 点击「开始翻译」，按界面提示确认任务与费用边界。
3. 等待流程依次完成准备、提交、云端执行、下载结果和安全校验。中途离开或重启后，可使用「继续翻译」或「查询云端状态」。
4. 在写回区域查看检查结论：
   - 「可写回」：确认正在操作游戏副本或已有备份，再点击「写回翻译」；
   - 「需处理」或「禁止写回」：展开「问题处理」，先修补、补译或重新检查，不要尝试强行写回。

底层安全流程为：

```text
doctor -> build -> submit -> status -> download -> check -> apply
```

只有最近一次 `check` 与当前任务结果匹配且结论为 `safe` 时，GUI 才允许执行 `apply`。

## 6. 写回后

- 检查目标语言目录中的改动，并在 Ren'Py 中运行 lint 或实际启动游戏验证。
- 保留任务记录和失败报告，直到确认项目可正常运行。
- 不要把 API Key、私有游戏脚本、Batch 结果或日志提交到公开仓库。

遇到问题时，可在页眉打开「运行日志」，查看任务上下文、可复制命令和原始 CLI 输出。高级恢复流程见 [Batch 工作流与安全检查](batch_workflows.md)。
