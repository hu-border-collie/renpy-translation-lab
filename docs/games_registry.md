# 工作区项目总表（Games Registry）

文档地图：[docs/README.md](README.md)

本工具维护 Ren'Py **工作区**里所有游戏项目的结构化总表，并据此生成人类可读的 `GAMES.md`。它与单项目翻译流程（`translator_config.json` + 工作台环境检查）互补，不负责替代 `doctor` 或批量写回安全检查。

## 定位

| 能力 | 负责方 |
| --- | --- |
| 浏览 / 切换工作区内多个项目 | **Games Registry**（本功能） |
| 检查**当前**项目能否开始翻译 | 「项目与环境」的环境检查（`collect_doctor_report`） |
| 批量翻译、写回、安全检查 | 现有 CLI + GUI 主流程 |

Registry 回答的是：「工作区里有哪些项目？各自路径、版本、游玩 / 翻译状态大致如何？」  
环境检查回答的是：「**我现在选中的这个项目**能不能开工？记忆库、待译、建议是什么？」

两者底层在**深度模式**下都会调用 `collect_doctor_report()`，但用途和数据范围不同，**不能互相替代**。

## 文件位置

默认路径（`renpy-translation-lab` 的上一级为工作区根目录）：

```text
RenPy_Workspace/
├─ games_registry.json    # 结构化真源（请编辑此文件或其 GUI 刷新结果）
├─ GAMES.md               # 由 registry 生成的 Markdown 总表（勿手改表格区）
├─ Game_Example/
│  ├─ original/
│  └─ work/
└─ renpy-translation-lab/
   └─ games_registry.py  # CLI 入口
```

仓库内提供 `games_registry.example.json` 作为字段示例；**实际数据**应放在工作区根目录，不纳入本仓库版本控制。

## 数据模型要点

每条 `projects[]` 记录通常包含：

- **人工字段**：`name`、`path`、`version`、`play_status`、`translation_status`、`notes` 等
- **自动字段**（写在 `auto` 下）：最近刷新时间、TL 文件数、待译条数、对话完成比例、是否有活跃批次等
- **`translation_status_source`**：标记翻译状态由谁维护
  - `manual`：人工维护；**快速 / 深度刷新都不会覆盖** `translation_status`
  - `scan`：快速刷新根据磁盘扫描推断
  - `doctor`：深度刷新结合 `collect_doctor_report` 的 layout / mode 推断
  - `batch`：写回后 `record-batch` 更新

游玩状态（`play_status`）可在 GUI「项目详情」或 JSON 中维护；刷新不会改写。

## GAMES.md 与 JSON 的同步策略

| 方向 | 操作 | 结果 |
| --- | --- | --- |
| **JSON → MD** | GUI「同步 GAMES.md」或 `render-md` | 用 `games_registry.json` **覆盖** `GAMES.md` 表格区（含生成标记） |
| **MD → JSON** | GUI「从 GAMES.md 导入」或 `import-md` | 解析 MD 表格写入 JSON；已有 registry 时可用 `--merge` **按路径合并** |
| **日常真源** | 刷新、写回 `record-batch`、GUI 编辑 | 一律写入 `games_registry.json` |

合并导入时：同路径项目的 `name` / 版本 / 状态 / 备注等以 GAMES.md 为准；**未出现在 MD 中的 JSON 项目会保留**。合并后若要让 Markdown 与 JSON 一致，请再执行「同步 GAMES.md」。

手改 `GAMES.md` 后不要用「同步 GAMES.md」（会覆盖你的修改）；应使用「从 GAMES.md 导入」拉回 JSON。

## CLI 用法

在 `renpy-translation-lab` 目录下：

```powershell
# 从现有 GAMES.md 初始化 / 合并导入（首次迁移时常用）
python games_registry.py import-md

# 按路径合并到已有 registry（不覆盖未出现在 GAMES.md 中的项目）
python games_registry.py import-md --merge

# 扫描工作区 Game_* 目录，登记尚未出现在 registry 中的项目
python games_registry.py discover

# 快速刷新全部项目（默认：扫磁盘 + TL 行数，不跑 doctor）
python games_registry.py refresh --all

# 深度刷新单个项目（含 doctor，较慢）
python games_registry.py refresh --project game_example --deep

# 由 registry 重新生成 GAMES.md
python games_registry.py render-md

# 记录一次批量写回 manifest（一般由 GUI 写回后自动调用）
python games_registry.py record-batch --project game_example --manifest path/to/manifest.json

# 查看 JSON
python games_registry.py show
python games_registry.py show --project game_example
```

可选参数：

- `--workspace <path>`：工作区根目录（默认：工具目录的上一级）
- `--registry <path>`：自定义 `games_registry.json` 路径

## 刷新模式

| 模式 | CLI | GUI | 行为 |
| --- | --- | --- | --- |
| **快速** | 默认 | 扫描模式 → 快速 | 扫 `work` / `original` / TL 目录与行数；用启发式推断 layout，**不**调用 doctor |
| **深度** | `--deep` | 扫描模式 → 深度 | 对每个项目调用 `collect_doctor_report()`，写入 `auto.doctor_layout` / `auto.doctor_mode` |

深度刷新全部项目可能耗时数分钟；GUI 在后台线程执行，可用「停止」按钮在项目之间协作式取消，已完成项目的结果会保存。

## 图形界面

安装 GUI 依赖后，在图形界面 **设置 → 工作区** 分区中管理总表（原工作台「工作区项目…」弹窗已并入此处）：

- 表格列：项目、路径、版本、**目录状态**、游玩、翻译
- 悬停 tooltip：自动扫描摘要、翻译状态来源、备注等
- **从 GAMES.md 导入**：首次接入或按路径合并更新总表（无需先跑 CLI）
- **扫描新项目**：发现工作区里尚未登记的 `Game_*` 目录并加入总表（默认快速刷新）
- **同步 GAMES.md**：由当前 `games_registry.json` 重新生成 Markdown 总表
- **刷新当前 / 刷新全部** + **扫描模式**（快速 / 深度）
- **停止**：刷新进行中可用；会等待当前项目的 doctor 完成后再停
- **切换到此项目**：将工作台 `game_root` 切到选中行（存在 `work/` 时自动解析）
- **项目详情**：可改显示名称、游玩 / 翻译 / 备注；只读展示目录状态、Doctor 模式、最近刷新时间
- **删除项目**：仅从总表移除记录，不删除磁盘上的 `Game_*` 目录
- **搜索 / 筛选 / 排序**：按关键词、引擎、翻译状态过滤，并按名称 / 路径 / 翻译状态 / 最近刷新排序
- **打开分区时自动扫描新项目**：可勾选；偏好写入 `games_registry.json` 的 `preferences`
- **刷新成功后**会询问是否同步 `GAMES.md`（与写回后行为一致）
- 刷新时表格**仍可滚动**；切换与刷新按钮会暂时禁用
- 若存在未登记的 `Game_*` 目录，状态栏会提示可点击「扫描新项目」

### 与写回流程的联动

批量翻译 **写回成功** 后，GUI 会：

1. 对当前 `game_root` 匹配 registry 中的项目，执行 `record-batch`
2. 对该项做一次**快速**刷新更新 `auto`
3. 询问是否据此 **同步 `GAMES.md`**

若当前项目不在总表中，会跳过 registry 记录（不影响写回本身）。

### 环境检查（项目与环境）

「项目与环境」的 **环境检查** 在后台对**当前选中项目**运行 `collect_doctor_report()`，并渲染完整摘要（warnings、记忆库、待译、建议等），用于解锁「开始翻译」。

这与 registry 深度刷新**共用同一 doctor 引擎**，但：

- 环境检查只看当前 `translator_config.json` 的 `game_root`
- 深度刷新按 registry 逐个项目切换路径，并会同步更新 layout / mode、版本探测结果，以及 `translation_status` / `translation_status_source`（非 manual 时）

在总表里深度刷过，**不等于**可以跳过工作台环境检查。

### 总表与环境检查的 layout 对比

工作台 **环境检查** 完成后，若当前项目在 `games_registry.json` 中有记录，会自动对比：

- 总表中的 `layout_status` / `auto.doctor_mode`（通常来自上次深度或快速刷新）
- 本次 `collect_doctor_report()` 的 `layout_status` / `mode`

结果会写入环境检查摘要与日志（`[总表对比]`）。若不一致，界面只提示**总表记录与本次检查不同**并引导到 **设置 → 工作区** 刷新；`layout` / `mode` 等机器字段只写日志，不面向用户展示。

打开 **设置 → 工作区** 且当前项目刚跑过环境检查时，详情面板的「总表对比」行会显示是否一致。这**不能替代**完整环境检查，只用于发现总表是否过期。

## 推荐工作流

**首次接入已有 `GAMES.md`：**

```text
GUI **设置 → 工作区** → 从 GAMES.md 导入 → （可选）扫描新项目 / 刷新全部 → 同步 GAMES.md
```

也可使用 CLI：`import-md` → 检查 `games_registry.json` → （可选）`discover` / `refresh --all` → `render-md`

**日常维护：**

```text
GUI 切换项目 → 环境检查 → 翻译 / 写回
                ↓ 写回后可选同步 GAMES.md
偶尔：**设置 → 工作区** → 扫描新项目 / 快速刷新全部 / 深度刷新单个疑难项
```

**编辑游玩 / 备注 / 人工翻译状态：**

在 GUI「项目详情」面板保存，或直接改 `games_registry.json`，再「同步 GAMES.md」或 `render-md`。

## 测试

与本功能相关的自动化测试：

```powershell
python -m pytest tests/test_games_registry.py tests/test_gui_games_registry*.py -q
```

## 已知限制

- 停止刷新只能在**项目之间**生效，无法打断单个项目正在进行的 doctor
- 快速刷新对 layout 的判断是启发式的，不如深度模式准确
- 「扫描新项目」只识别顶层 `Game_*` 与 `Game_Adastra_Universe/` 下已整理子目录，不会自动登记压缩包或未纳入 `Game_*` 结构的目录
- 未勾选「打开时自动扫描」时，不会自动写入新发现的项目；需手动点「扫描新项目」或运行 `discover`
- GUI 可改显示名称，但**不能**通过对话框变更项目路径（`path`）或移动磁盘目录