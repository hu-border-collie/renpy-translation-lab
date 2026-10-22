# #509 Translator++ 单版本往返：实测结果与接入决定

> **状态（2026-10-22 完成补测）**：已完成真实 GUI 导入、保存 `.trans`、无修改/人工译文导出、独立 null 保存/重开/导出、staging 缺失与恢复重试、一次命令位置漂移及畸形/缺失源文件测试。固定 **TPP 8.8.9 → RMMVJS parser 2.19 → MV 编辑器/core 1.5.1**。限定译文与 null 补测均精确命中 19 个字符串位置，部分原生和 HTML5 运行有证据；无修改全资源导出改写插件文件生成注释，前插注释后的 context 译文迁移缺一处。**本轮 spike 的验收证据已齐，决定保留 TPP 独立使用，生产 bridge 暂缓；记录交付与合并状态另行跟踪，不将上游缺陷写成通过。**
> **仓库基线**：`main@d7c9763d9fa8fd004d0527c937a591ce6499a97a`，开始时工作区干净、远端 main 同值；开始时读取的 [#509](https://github.com/hu-border-collie/renpy-translation-lab/issues/509) 为 OPEN、无评论。未修改线上 issue。
> **范围**：原创夹具、独立核对和限定往返实验；未开发 bridge、`.trans` reader/writer、Adapter、服务或产品 GUI，未调用翻译 Provider。交付仅包含本报告、原创数据覆盖层和研究更新。

## 1. 版本、来源和许可

现行入口为[接入评估](translatorpp_integration_assessment.md)、[能力矩阵](visual_novel_localization_matrix.md)、[生态研究](multi_engine_localization_ecosystem_research.md)。只读核对过未合并的 #272 提交 `7e2cbd8a7f190ea300a5caf85c930644370cbbbc`，没有合并或 cherry-pick。

| 对象 | 本次实际证据 | 边界 |
| --- | --- | --- |
| Translator++ | [官方下载页](https://dreamsavior.net/download/)指向官方 [8.8.9 Release](https://github.com/dreamsavior/TranslatorPlusPlus/releases/tag/8.8.9)；完整取得 `Translator++_8.8.9_x64_.51895b12.7z`，221,133,957 字节，SHA-256 与 Release API digest 一致 | 应用根 package 为 8.8.9；About 未核对，不能用 EXE 的 NW.js 版本 0.104.0 代替应用版本 |
| MV/MZ parser | 新建工程 GUI 为 `RPG Maker XP, VX, Ace, MV & MZ Parser Ver. 3.3`（rmrgss）；检测 HTML 入口为 rmmv 后，实际日志明确显示 **RMMVJS Parser 2.19**，保存工程的 parser/parserVersion 为 rmmvjs/2.19 | 前端 3.3 与实际 MV parser 2.19 分开记；不将前端版本当成导出 parser 版本 |
| 展开 | 使用已有 7-Zip 24.09，检查成员路径后展开；16,371 个文件/目录条目，文件大小合计 977,593,978 字节 | 未分发第三方文件；Windows tar 无法读取加密头的失败已保留 |
| MV 来源 | [官方日文 Windows 试用页](https://www.rpgmakerofficial.com/support/trial/)提供 `https://cdn.tkool.jp/updata/RPGMV_J.zip`；完整 1,873,840,514 字节，1,716 项 CRC 全通过 | 首次传输提前结束、续传超时均保留；后用验证 Content-Range 的有界分段下载完成 |
| 实际引擎窗口 | 用户完成安装和试用提示后，Agent 在编辑器 About 截图及可访问文本中确认 **Version 1.5.1**；实际 `NewData/js/rpg_core.js` 的 `Utils.RPGMAKER_VERSION` 同为 **1.5.1**；新工程 `Game.rpgproject` 为 `RPGMV 1.5.1` | 本轮固定日文 Windows 试用 **MV 编辑器/core 1.5.1**；不沿用旧建议 1.6.3，未运行更新器，不混入 MZ |
| 环境 | Windows 平台 `10.0.26300`、Python 3.14.0；Computer Use 能接管 MV 主窗口 | CIM OS/Windows 应用事件日志查询受限；不编造 edition、事件日志或故障根因 |

实取文件 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| TPP 完整 7z | `2389b58a1315870ca4bf3ee7b2eff06d65b2b635a5c56a9b3b0a59b1c1d9131c` |
| TPP EXE | `def49ba8a2cf245d2a214fad42689ca53ff06e2ede4a082f950e64c3f56f7d3e` |
| TPP 根 package.json | `f69f252139fc9c87b826fae03972aa0d3f4923afd9132214dfdbd67ddf36e5c0` |
| TPP parser package.json | `7627b0cfcfa1ab680b847572bc49808538e7541bb763820fe3f9ff3c67fdd834` |
| TPP rmmv.js | `eede42aff1f78227cc1c116fc6ce76f31bc4037481f19dc8a244b3039c606cc7` |
| MV 完整 ZIP | `07fc32b1b2943a7eb1612b717b0aee9f2a9d870a70e7fc566fc5276354f061a7` |
| MV Setup.exe | `802e672448edcbf0f4463844920d71ba6261b69620dfd25a20f81de23321093a` |
| MV Setup-1.bin | `72733f43459802037bac5c56fbdc898ca9faaa21420b367b35a7c88fbce550c3` |
| 实际 rpg_core.js | `e9d52fbce06360f00cd3a61a22eaf1c60790c72b6c2c803ac5af9c9be268e7c7` |

其余五个 `rpg_*.js`、编辑器 EXE 和本地许可页 hash 记录在忽略目录 `private-evidence/mv-installed-baseline-hashes.json`；完整工程基线还覆盖 libs、字体/CSS、图像/音频等所有文件。

MV 官方试用条款写明首次启动起 30 天、可制作游戏，游戏分发仍受条件约束；本地安装的 `Help/page/02.html` 已读取。用户提供的首次提示截图显示剩余 30 天，之后已进入编辑器；未输入密钥、购买或绕过试用。编辑器、core、NW.js、默认数据库、字体和素材仅保留本地，不纳入仓库 MIT 许可。[官方试用条款](https://www.rpgmakerofficial.com/support/trial/)、[MV 使用与游戏分发条款](https://rpgmakerofficial.com/product/MV_Help/page/02.html)

TPP 根 package 声明 GPL，但 `licenseScope` 排除 addons、modules、PHP、node_modules 等路径。实际新建入口 rmrgss 3.3 的随包 `LICENSE.txt` 为 Apache 2.0 附 Commons Clause，包含出售限制；实际委派的 rmmv 2.19 manifest 未声明独立 license。没有据根许可推定所有组件同一许可，没有复制或分发其实现，也未作商业许可结论。公开目录只有原创事件、测试文本和说明。rmrgss package、RMUtil.js、rmrgss.module.js、LICENSE.txt 的 hash 单列于本地 `actual-parser-3.3-files.json`；前三者分别为 `a9762eb43348bbcda51c71a58239318b129dec1e3fce7ee0b90f9a503f05fb06`、`b151369755b8094cb39fe0cf264c83e1f2a077adc6596ece53b1793cc968a984`、`877dd42ff6f098b8ab651150568256b6741251d15350ec307469a56a75f9256d`。

## 2. 启动失败与恢复边界

用户已授权下载、安装和首次运行；不再重复征求同一范围的开工许可。此前 MV 安装器临时目录错误及无可接管窗口已恢复，**不是当前 MV 阻塞**。

| 路径 | 实际观察 | 判定 |
| --- | --- | --- |
| MV 安装初期 | 临时目录创建 `is-*.tmp` 报错误 5；独立进程 TEMP/TMP 尝试仍未得到向导。随后用户完成正常安装，提供主界面，Agent 接管并核对版本 | 保留失败；不再描述为尚未安装 |
| TPP Computer Use 启动 | 两次 `sky.launch_app` 超时；普通 Start-Process 也曾挂起，已取消本轮挂起启动命令 | 没有主界面成功证据 |
| TPP 直接进程 | 立即退出，`0x80000003`；日志为 `crashpad_win.cc:91`、`CreateDirectoryAndGetError(metrics_path, nullptr)` 失败 | 启动初始化失败，尚未加载项目 |
| 正常资源管理器双击 | 找到并双击真实 EXE，出现 SmartScreen；由用户自行处理并明确回复“我点过运行了” | Agent 未代点安全提示；放行后仍无 TPP 进程/主窗口，20:15 日志再报 metrics 目录失败 |
| 工作区用户数据目录 | 创建普通 `.lnk`，目标仍为原 EXE，仅增加 `--user-data-dir` 指向独立 `tpp-profile/UserData`；经资源管理器启动 | metrics 文件实际生成，但随即退出，`sandbox/policy/sandbox.cc:67`、`result == SBOX_ALL_OK` 失败 |
| 获准的正常进程复核 | 同 EXE、同用户数据目录再作有界启动复核，20:18 立即退出，同码同错误；fresh 窗口列表无 TPP | 停止重复启动；根因未确诊，未关闭 sandbox 或修改系统安全设置 |

`--user-data-dir` 是 NW.js 支持的数据/缓存/崩溃目录选项；本轮只更换这个目录，没有增加禁用保护的参数。日志中其他 Chromium 参数来自归档自带 manifest，已和本轮参数区分。[NW.js 命令行文档](https://nwjs.readthedocs.io/en/nw20/References/Command%20Line%20Options/)

这些是工具启动失败，**不是已经证明的 TPP parser、导出或 staging 缺陷**。当时停止要求用户重复打开原 EXE；后续根因调查及已验证的新启动路径见 §2.2。不得用私有接口、手造 `.trans` 或换用未记录的 runtime 冒充正常 GUI 实验。

### 2.2 2026-10-22：根因与成功对照

对启动作有界 Win32 调试，只观察 API 返回值、进程令牌和临时对象，不绕过断言、不改 EXE 文件、安全设置或现有对象权限。得到以下证据：

| 检查 | 实际结果 | 解释 |
| --- | --- | --- |
| 默认执行环境 | AppContainer=true；临时窗口站创建返回错误 5 | 默认工具环境自身存在限制，但不能据此解释普通用户启动的全部失败 |
| 获准普通用户环境 | AppContainer=false、中完整性；复制 Chromium 所用 DACL/权限创建窗口站和桌面均成功，创建后立即关闭 | 对照环境能执行该操作，不支持“系统已无法创建任何隔离桌面”的泛化判断 |
| TPP 失败进程 | 非 AppContainer、无受限 token、无线程模拟；读取 WinSta0 安全描述符成功；两个 `CreateWindowStationW` 调用（原请求及降权限回退）均为 NULL/错误 5 | 错误发生在创建窗口站，尚未进入 CreateDesktop 或工程载入 |
| 进程完整性差异 | 对照进程 `S-1-16-8192`（Medium）；TPP `S-1-16-4096`（Low），用户及登录会话组相同；将 TPP 实际安全描述符复制到对照进程调用可成功 | 完整性级别差异已实测；普通 token 不等于中完整性 token |
| 文件来源 | 原 `tools/Translator++` 目录与 EXE 的 `icacls` 均显示继承的 `Mandatory Label\Low Mandatory Level:(I)(NW)` | Windows 按用户/可执行文件两者较低的完整性级别创建进程；放行 SmartScreen 不会移除该标签 |
| 成功对照 | 用获准普通用户环境，将同一已核对的官方 7z 重新解到普通用户 Temp 下的新独立目录，不复制原 ACL；新 EXE 未继承 Low 标签。普通启动后存活、实际 token 为 Medium，Computer Use 捕获主界面及 `Translator++ Ver.8.8.9` | 启动问题已恢复；原低完整性目录仍保留，不能继续以其 EXE 作成功路径 |

Windows 对低完整性可执行文件的进程创建规则见 [Microsoft MIC 文档](https://learn.microsoft.com/en-us/windows/win32/secauthz/mandatory-integrity-control)。Chromium 的 [CreateAltWindowStation 实现](https://raw.githubusercontent.com/chromium/chromium/141.0.7390.37/sandbox/win/src/window.cc)与本轮捕获到的权限请求和错误回退一致。初始 metrics 目录错误也符合低完整性进程无法写普通用户数据目录的限制；本轮没有另行追踪那次目录 API，因而将其作为同一根因的解释，不冒充额外 API 级复现。

新旧 EXE、根 package、parser package 和 rmmv.js 四份内容 hash 全部相同；没有换 NW.js 或 parser。成功路径是**在正常完整性目录重新解压同一包并正常运行**，没有执行 `--no-sandbox`、提权运行 TPP 或修改文件标签。最初由哪个历史进程设置工作区标签没有审计；目前确认的是继承关系及其运行后果。

新目录及 UserData 的完整路径仅保存在本地 `private-evidence/tpp-clean-extraction-location.json` 与 checkpoint；目录位于用户临时区，后续清理可能移除，恢复前须检查存在及 hash。`private-evidence/tpp-startup-root-cause-and-success.json` 保存中完整性/同 hash/窗口证据；原日志、失败 trace 和原目录保留。一次新解压后误按无前缀路径核对 EXE 已纠正为归档实际 `Translator++/` 子目录；启动脚本的 WorkingDirectory 参数绑定错误也有保留，NW.js 随后仍从 EXE 包目录正常加载，本轮不以命令无报错代替实际窗口证据。

这解决启动阻塞。随后导入、选列、context、staging 和导出实测见 §4；About 未单独核对，应用版本以实际标题、保存工程的 editorVersion 及包 hash 相互印证。保存工程的 `appVersion="0.9"` 不是本轮应用发行版本。

## 3. 原创工程与独立分母

使用 MV New Project 正常创建《The Night Signal》，关闭工程后写入六份本轮原创数据/配置，再由编辑器复开确认。完整原件有 **1,100 文件、403,838,024 字节**，均留本地；随后建立独立 `input-copy` 和 `runtime-original`，并为各导出建立专用 `runtime-*` 副本。TPP 处理 input-copy，运行只用专用 runtime 副本，导出分别到新目录，原件不参与运行或导出。

公开的[原创数据覆盖层](issue-509-materials/README.md)提供 [Map001.json](issue-509-materials/Map001.json)、[CommonEvents.json](issue-509-materials/CommonEvents.json)、[制作材料](issue-509-materials/storyboard.json)和[实际 occurrence 清单](issue-509-materials/occurrences.json)。复现需合法 MV 1.5.1 安装提供引擎/素材；不是可独立启动的完整游戏包。System、Actors、Items 和无插件设置的复现步骤在覆盖层说明中，不提交默认第三方数据库。

实际剧情为一个 17×13 地图、两个事件各两页、一个公共事件、三个选项和取消分支。变量 1 初置 7，actor 1 为 Mira，敲铃调用公共事件后显示 8；switch 1 和自开关 A 控制两组第二页。item 1 的原创 marker `UNSELECTED_RESOURCE_509_KEEP` 是整文件未选守卫。

| 独立统计 | 实测值 | 不可混同的对象 |
| --- | --- | --- |
| 401 文本行 | 19 | 屏幕自动换行不等于新增 401 |
| 102 选项元素 | 3 | 与 402 分支标签分别记账 |
| 402 标签镜像 | 3 | 不能因原文相同而丢弃 |
| 全部文本参数位置 | **25** | 非 TPP 网格行或 context 数 |
| 主要出现位置 / 唯一原文 | **22 / 19** | 主要位置不含三个标签镜像 |
| message 块 / 制作块 | 15 / 16 | 一个制作块为 choices |
| TPP 网格 / context | 地图 17 行、19 context；公共事件 3 行、3 context | 含 4 个元数据 context；18 个剧情 context 对应 15 个 message 块和 3 个选项，不等于 25 个参数位置 |

`The lamp is still on.` 出现在 event 1 两页、event 2 与公共事件四处。实测三个语境分别译为门口、值班室、钟楼，event 2 保留原文。event 2 与 event 1 同地图，使用空默认译文和 context override 保留原文，不能称为按事件取消资源勾选；Items 另作整文件未选守卫。两列冲突、空串和独立 null 的不同证据状态见 §4.2。

修正版 **v3** 冻结文件：

| 制品 | SHA-256 |
| --- | --- |
| 全树基线 `original-baseline-v3.json` | `33d872d754f7515332de0b3bd4fc11e213b14060483f5aca226bbdbac3e5d632` |
| Map001.json | `5dd860bbe3f88b34c8348bcc003e1f36ee58bc9866c442fd4405aea637627c66` |
| CommonEvents.json | `19d8a108fe88d95f8ca41b3b1cf374f52c7c6962db5abdd043addfe2a5d47f6d` |

独立清单记录文件/完整参数路径/code/indent；全树 hash 覆盖数值、类型、数组顺序、地图瓦片/坐标、页面条件、命令及资源。20:19 再核对原件全树与 v3 基线一致，input-copy 对原件亦逐文件一致；这是源完整性验证，**不是无修改 TPP 往返**。

### 已保留的夹具构造失败

- v1 的空 `plugins.js` 用自写注释替换生成头，编辑器拒绝加载。保留原文件/基线后恢复生成的两行注释头和空插件数组，复开成功。
- v2 的字面路径只处理了 JSON 转义，MV 消息层把它显示为 `C:Gate.`。保留旧地图、材料和基线后，在引擎消息文本层双写反斜杠；v3 实际显示正确的字面路径。

两次都是 **TPP 导入前**的原创夹具制作修正，没有抹除旧基线，也不归因为 TPP 缺陷。后续若再改源数据须新建基线，不继承 v3 检查。

## 4. 真实 GUI 往返

### 4.1 导入、工程制品和无修改导出

正常新建工程，选择前端 parser 3.3，在页面输入独立 `input-copy/index.html`，检测为 rmmv；Include JavaScript plugins 关闭。勾选 Create a development copy 并填写独立目标，但实际委派的 MV 路径没有建立该目标，使用应用自己的 `data/staging/TlmfZwbpBo`。仅将“请求设置”和“实际生成位置”分别记录，不据勾选状态声称开发副本已创建。

实际导入日志显示 parser 2.19、539 ms，解析 15 份 JSON 并建立 19 文件的 staging，包含 `initial.json` 和 `data/data`。随后通过 GUI Save As 保存 `unchanged.trans`，另存 `translated.trans` 后才编辑。所有 `.trans` 都由应用生成；离线核对只读这些实际制品，不调用私有接口或制造工程文件。

| 制品 | 字节 / SHA-256 | 含义 |
| --- | --- | --- |
| unchanged.trans | 34,175 / `258fbb1d02f9a675ee9b5ff68bdd98d2bd055979c4c9725ef2aa45d4e08c663b` | 正常保存的无修改工程，后续保留 |
| translated.trans | 35,458 / `0e32577b7ec8b728c71de1727eb170eac2955756120bba4f7bb7323591248044` | 实际 GUI 人工译文输入；不含模型生成或付费调用 |

无修改导出前，先从原件建立完整独立输出副本并核对相同。GUI 选择 Export to a folder；未勾选任何资源，实际导出了全部 15 份 JSON。设置为 Ignore tag、无自动折行规则、Maximum dialog line=4；实际日志完成 155 ms。

独立全树比较：无新增/删除文件，15 份 JSON 仅格式变化，**零 JSON 语义变化**；唯一其他变化为 `js/plugins.js`：生成者注释由 RPG Maker 改为 Translator++，CRLF 改为 LF，移除结尾换行，空 `$plugins` 数组保留。该文件由 87 字节变为 84 字节，hash 从 `479f200fdb746f0dae49f40bbc1c06af8dfbac92d78b6ef18b51b9e152ec3a2e` 变为 `46df17724f1f47291d63cc6da1618feb7bf73d4d5887e5710ecfdcf092cbfe83`。

这是本例可解释的注释/序列化副作用，**严格非目标文件保护检查仍为 fail**。未修改输出或放宽审计以制造 pass；后续 MV 编辑器复开和原生标题/地图/首句运行成功，另记为运行证据。不能把“能运行”改写成“逐文件未变”。

### 4.2 人工译文、选列和 context

在 GUI 的 Initial 列填写构造译文；表格粘贴正常分配多行单元格及两列候选，保存后再独立读回确认。文件选择仅勾选 Map001 和 CommonEvents，其他设置与无修改导出相同；导出到另一份完整新副本，日志完成 47 ms。

| 用例 | 保存/导出实测 | 结论边界 |
| --- | --- | --- |
| 同地图重复句 | 地图中同句合并一行，保留三个 context；默认译文留空，两处 override 为“门口”和“值班室”，event 2 context 为空 | 三处分别导出为两种中文和原文，没有串用；不能据此确认其他版本或更新后的身份稳定 |
| 公共事件同句 | CommonEvents 的独立行设为“钟楼” | 跨文件同句得到第三种译文 |
| 两列冲突 | Initial=`左列候选`，Machine translation=`右列候选`，其余列为空 | 图表分支实际采用右列候选；未验证第三/四译文列或其他选列组合 |
| 空字符串 | bell/leave 两行的前两译文列为空字符串 | 对应对白保留原文，没有导出为空对白 |
| 独立 null | 最初 Delete 仍保存 `""`；后续在独立工程中通过编辑未使用列并撤销，取得非空原文行上的 null，重开/再保存后仍保留 | 该行与其余空串共同回退原文，19 处预期译文不变，详见 §4.5；追加空白行不计为此证据 |
| 多行/引号/路径 | 两行和四行消息保持各行；引号、引擎层双写路径反斜杠保持 | 没有因序列化或折行改变命令数量、顺序与缩进 |
| 控制码 | `\V[1]`、`\N[1]`、`\C[2]`/`\C[0]`、`\I[1]`、`\.`、`\|` 保留 | 运行观察另列；未知控制码、脚本、插件方言仍排除 |
| 选项镜像 | 三个 102 元素和三个 402 分支标签均更新 | 分支索引及取消类型不变 |
| 未选整文件 | Items 及其他未勾选文件逐字节不变；本次 plugins.js 也未改变 | 即使日志仍写 exporting plugins.js，也按实际文件比较判定 |

独立预期由原创制作材料和源命令树构造；两列 probe 的判定使用实际右列结果，未预设列优先级。全树检查恰好命中 **19 个显式字符串叶位置**，无其他 JSON 语义、非 JSON、文件增删变化，严格检查 **pass**。其余 6 个清单参数仍为原文（bell/leave/cancel 及第二事件三处）。这不是“全游戏翻译率 100%”，也不代表候选文字语义质量合格。

独立对账读取实际 101 header 与后续 401，核对聚合行完整原文；选择项同时核对 102 索引与 402 镜像。原始 25 个参数均可解释，无未匹配项。TPP 地图 17 行/19 context、公共事件 3 行/3 context，其中 4 个 context 是地图显示名、两个事件名和公共事件名；余下 18 个是剧情 context，覆盖 19 条 401 和 3 个 102 元素。三个 402 标签没有独立 parser context，按分支镜像另外记账。两行/四行聚合消息和同文合行均不能按“一行=一个参数”换算。

### 4.3 staging 缺失、恢复与源文件负例

仅备份并暂时重命名本轮 `TlmfZwbpBo`，其他缓存未动；复开真实 `translated.trans`，GUI 明确警告 staging 不存在、可以编辑但不能导出/注入。随后实际向空隔离目录导出，日志报 CommonEvents 路径 `ENOENT`，目标目录零文件；同时仍显示 All process finished / Finished，并另有“目标目录无文件”错误文字。未将完成条当成功，也未观察成已有游戏副本被清空。

原缓存恢复后，19 文件 hash 与备份全部相同；同一译文工程、同一选项向新完整副本重试，日志完成 67 ms。严格 19 处替换检查 pass，输出全树 hash 与第一次译文导出一致。原失败目录和日志保留，恢复不靠复制成功产物冒充重跑。

进一步使用官方 Update Project from Source，分别构造两个负例。该功能固定读取工程已记录的 input-copy 路径，没有先弹选择新源的窗口，因此只临时改变这个可恢复输入副本，保留原件和精确备份；每次新缓存生成后立即恢复，最终全树核对 input-copy 与原件相同。

| 负例 | 实际结果 | 不可推导的结论 |
| --- | --- | --- |
| Map001.json 截断为无效 JSON | GUI 更新显示 Done；保存后的工程没有 Map001 条目，新缓存保留原始无效文件；捕获日志没有专门解释该畸形 JSON 的遗漏报告 | 不是该地图已提取成功；未导出/运行这个不完整工程，不泛化为所有错误都被忽略 |
| CommonEvents.json 缺失 | GUI 更新显示 Done；新生成 initial.json 含正常地图、无 CommonEvents；捕获日志未明确报告该源文件遗漏 | 不是零遗漏或公共事件已处理；没有以旧制品补齐缺失文件冒充 parser 成功 |

本轮另外有可访问性索引失效、原生文件对话框控件不可点击、截图被其他窗口遮挡等操作失败。使用实际已聚焦文件名字段、重取窗口和新截图恢复；这些自动化工具失败不归为 TPP parser 缺陷。一次备份脚本的编码名称错误已改正，缓存始终保留于重命名位置，之后复制备份并验证恢复；没有丢失数据。

### 4.4 单条命令前插的真实漂移

先另存独立实验工程，做一次源内容不变的 Update Project 对照：新建 staging `L9A8YkpDJf`，实际两处 context 译文保留；序列化移除了部分空参数/行说明字段，不能据 JSON 字段全等要求判定游戏行为。

然后只在 event 1/page 0/list 1 前插一条 code 108 注释，消息内容保持不变。Update Project 生成新 staging `VYdZMyXngt`，另存实际 `drift-updated.trans`。新注释自身成为网格第 3 行，灯光句移到第 4 行；按原文和完整 context 重新核对，不能沿用旧网格行号。

- 第一处灯光 header 从 list 1 移到 list 2，context 路径随之改变，原有“门口”译文未迁移；未变的第二页路径仍保留“值班室”译文。
- 实际导出到从漂移源建立的新副本后，预期 19 处替换只有 **18 处**生效。缺失位置为 `/events/1/pages/0/list/3/parameters/0`，保留英文；注释本身、其余文字和结构均符合本例源内容，没有观察到译文错写到另一个 context。
- 更新/导出日志均显示完成，所捕获界面未提供这处译文迁移的专门 conflict/unresolved 报告。该例证明“原文没变”不足以保证 context 译文完整迁移，**不证明所有上游流程都缺少诊断**。

`input-copy` 的临时注释随后精确恢复；原件、先前成功译文工程及导出未改变。未测试复制同文后新增歧义、换页/移动事件、重复/冲突 ID 或所有源漂移组合，这些继续保留 unknown。

### 4.5 独立 null 补测与重开导出

为补齐空值验收，从真实 `translated.trans` 经 GUI Save As 新建 `null-roundtrip.trans`，不覆盖先前制品。在 Map001 的第 11 行（原文 `I will ring once.`）选择尚未使用的 Best translation 列，输入临时标记 `ISSUE509_UNDO_TO_NULL`，再按 Ctrl+Z 撤销、Ctrl+S 保存。离线只读核对该行实际保存为：

```json
["I will ring once.", "", "", null, ""]
```

null 位于 Better translation（零基列索引 3）；不能因输入动作发生在 Best translation 就写成 null 位于该列。对应 parser context 为 `Map001/events/1/pages/0/list/16/message/noPicture/bottom`，源/输出对白参数是 `/events/1/pages/0/list/17/parameters/0`。原文、context、命令树均未修改，追加空白行不参与此判定，未手改 `.trans` 或调用应用内部 API。

通过 Open 重新选择该工程并再次保存，null 仍保留。首次保存为 35,446 字节，SHA-256 `493fa2ba649fbfdb4d9e6e2c1977da1a87b6d24abaa1a274496d606c6f5b2a7b`；重开再保存后的 SHA-256 为 `92fb5b6bfd5088aa5977f5f60bff5e851995b7884c3df2285b6154eb264066e2`。两阶段制品 hash 分开记录，不要求应用保存元数据字节不变。

重新仅勾选 Map001、CommonEvents，以相同 Ignore tag、无折行规则、最大 4 行设置，向从原件复制的全新 `exports/null-roundtrip` 导出。实际 GUI 日志完成 66 ms。使用补测前已声明的同一份 19 处预期核对，严格检查 **pass**，没有其他结构/文件变化；null 行和对照空串行均保留原文，没有空白对白或临时标记泄漏。

输出全树 hash 还与首次译文导出逐文件相同，因此可关联 §5 中已有译文运行证据；没有把本轮文件相同冒充重新执行了原生/HTML5 游戏。此例覆盖 null 与空串混合、所有候选均为空的回退，不宣称证明“null 覆盖非空译文”、所有列排列或其他版本。

## 5. 运行、质量与净收益

所有运行都使用独立 runtime 副本，未运行或写回原件；只使用正常按键/点击，不调用引擎内部函数跳过剧情。

| 环境与副本 | 已观察到的运行证据 | 未验证边界 |
| --- | --- | --- |
| HTML5 原文 v3 | 新游戏/地图，双行引号及正确路径、Count 7/Mira、颜色/图标/等待、三选项、敲铃公共事件计数 8、取消后第二页四行、第二事件两页 | v3 原文 Look/Leave 直接分支和存读档未补跑，Look 仅在 v2 观察过 |
| HTML5 译文输出 | 三种灯光语境、两行中文路径、变量 7/Mira、颜色/图标及控制消息，三个中文选项，图表采用右列候选、敲铃公共事件译文和计数 8、取消后第二页译文/四行，第二事件两页仍英文 | 等待时长未逐帧量化；未测试所有菜单、操作系统或字体环境 |
| HTML5 译文存读档 | 通过菜单保存至空测试槽，重载页面后 Continue 显示该槽，载入恢复原位置及第二事件第二页 | 浏览器本地存储，不等同原生存档验收 |
| 原生无修改输出 | MV 1.5.1 编辑器正常复开；自带 NW.js 测试游戏 URL 确认为该 runtime 副本，标题、新游戏、地图、首句英文出现 | 没有完整遍历所有原生路线 |
| 原生译文输出 | 编辑器复开、实际 URL 为译文副本；首句中文、两行引号/字面路径、7/Mira、颜色/图标/控制消息和三中文选项；选择 Leave 后进入对应英文分支 | 完整 Leave 末句未留到新截图；原生第二页、公共事件、存读档仍未完整重跑，不能借用 HTML5 结果填满 |

原生截图为 `native-unchanged-*`、`native-translated-*`；浏览器证据为 `html5-translated-*`，均留本地忽略目录。一次原生路径截图被 TPP 遮挡，标成 `-occluded` 后另拍真实前台路径画面，不将遮挡图当通过证据。初期原生应用访问超时已恢复，本轮已有实际原生画面。

本机所观察中文没有缺字或超出对话框，两行与四行显示完整；实际字体 fallback 未追踪到具体字体文件，不能据本机成功承诺其他机器。构造译文只用于机械验证：“左/右列候选”是故意的 probe，其他原文保留和语境改写也不是完整汉化；语义、角色语气和最终质量仍需人工审稿，不作质量完成结论。

**决定：缩小为 TPP 独立工具使用，暂不立项生产文件 bridge。** 本例已证明它能提取、编辑并正确导出受限 MV 1.5.1 叙事数据，人工操作也可保存 context 和复用工程；因此不否定独立使用价值。但源漂移丢失译文、畸形/缺失文件的独立 coverage 需求、资源选择副作用和 staging 恢复，使本项目仍须承担身份/快照、选列与 context、遗漏、结构及导出验收，不只是读写表格。

机器阶段日志为毫秒级，不代表端到端工时。实际流程含两次工具启动恢复、原生对话框接管、独立源/导出/运行副本管理、context 手工填写、保存/读回、负例和恢复。实验跨晚间及翌晨并有中断，未连续计量主动工时，不能把墙钟差当人工耗时，也没有与原生 Adapter 或纯 TPP 完成同口径 A/B，因此不给节省百分比或净开发量数字。

后续最小范围是保留这套原创覆盖层、精确参数清单与结果复现；只有另行决定推进 bridge 时，再补歧义/冲突和更完整运行证据，并设计快照/context 迁移的显式 unresolved 合同。独立 null 已在 §4.5 补齐；其余未测项作为本轮边界，不暗中增加 #509 的验收范围。不能因本轮 19 处导出 pass 直接启动 reader/writer 或 MCP，也不扩展 #508 首期 Ren'Py 范围。

## 6. 原 issue 八项状态

| 验收项 | 当前状态 | 证据或缺口 |
| --- | --- | --- |
| 1. 版本/来源/parser/许可可追溯 | 本轮路径已记录 | MV About/core 1.5.1、TPP 标题/editorVersion/hash、前端3.3与实际2.19、配置与许可边界；TPP About未单独核对 |
| 2. 原创 fixture、独立清单、复现 | 原创覆盖层与清单已提供 | 25 参数及合法组装说明；完整 runtime/素材不分发 |
| 3. 无修改往返 | 文本/JSON结构语义保持；严格文件检查 fail | 15 JSON仅格式变化，插件注释/换行副作用已明确；原生复开/首句运行另记 |
| 4. 人工列/context/空值/转义 | 本轮所列用例通过 | 19 处预期导出 pass；独立非空原文行 null 保存/重开/导出已补齐，混合空值回退原文 |
| 5. 同文/未选资源及恢复 | 本轮固定源通过；更新暴露缺口 | 同文无误关联、未选守卫、staging失败及恢复成功；漂移后缺一处译文 |
| 6. coverage、运行、质量分别报告 | 已分列，运行覆盖仍有缺口 | 25参数逐项对账；原生与HTML5、机械输出与语义质量分别报告 |
| 7. 继续/收缩/停止决定 | 已形成收缩决定 | 保留TPP独立使用，生产bridge暂缓，无未经计量收益数字 |
| 8. 三份研究同步及文档验证 | 同步本轮实测边界 | 不提升引擎支持评级；最终本地链接和diff检查单列 |

八项所需的实验、差分、运行边界、决定和文档记录已具备，可按研究 spike 交付；线上 issue 的关闭随交付合并跟踪。无修改插件注释变化有解释，严格文件审计仍为 fail；源漂移缺译仍为失败证据。完成实验验收不等于这些上游行为通过了生产安全门禁。

## 7. 本地证据与复现检查点

私有根目录为 gitignored 的 `artifacts/issue-509-2026-10-21/`；`checkpoint.json` 记录实际目录、hash、缓存恢复状态与下一步。下载包、第三方实现/许可、完整游戏、`.trans`、原始日志和含默认素材的截图均不加入 Git；公开只有原创覆盖层及本报告。

| 证据文件（私有根目录内） | 内容 |
| --- | --- |
| original-baseline-v3.json | 原件全树与独立原始分母 |
| private-evidence/tpp-gui-import-settings.json、tpp-import-ui-log.txt | 实际设置、委派 parser 和缓存位置 |
| unchanged-post-export-audit.json | 无修改语义与非目标插件变化 |
| private-evidence/translated-trans-input-receipt.json、translated-expected-changes.json | 实际保存值和独立声明的19处字符串预期 |
| private-evidence/translated-post-export-expected-audit.json | 限定译文严格比较 pass |
| private-evidence/source-parser-occurrence-reconciliation.json | 源参数、101组、102及402镜像对账 |
| private-evidence/tpp-missing-staging-*-ui.txt、staging-recovery-receipt.json | 实际警告、失败和缓存恢复 |
| private-evidence/recovered-post-export-expected-audit.json | 恢复后重试 pass，全树同首次译文输出 |
| private-evidence/drift-export-desired-audit.json、drift-export-receipt.json | 源漂移期望19/实际18及确切缺口 |
| private-evidence/malformed-source-receipt.json、missing-source-receipt.json | 畸形/缺失源被排除的实际工程证据 |
| private-evidence/input-copy-after-negative-tests-audit.json | 所有临时源改动恢复后，完整副本与原件一致 |
| private-evidence/null-trans-input-receipt.json、tpp-null-reopened-ui.txt | 非空原文行的真实 null、重开/再保存和两个阶段 hash |
| private-evidence/null-post-export-expected-audit.json、null-export-receipt.json、tpp-null-export-ui.txt | null 补测实际导出、19处严格预期及全树同首次译文输出 |

一次性 `audit_engine_copy.py` 是本实验独立的只读核对，不是公共产品接口；严格 JSON 比较覆盖数值/类型、数组顺序和长度、键集合、命令缩进与所有未预期的叶变化，全树 hash 覆盖其他资源。8 个合成自检只证明核对器自身；真实导出检查单独留档。未为纯文档/材料改动运行无关的完整 CLI/GUI 门禁。

最终检查通过：原件 1,100 文件与冻结 v3 全树一致，恢复后的 input-copy 同值；公开两份数据覆盖层逐字节一致，25 参数清单与独立扫描逐项相同。89 份 tracked/新增 Markdown 的 558 个本地文件目标无断链，`git diff --check` 通过。结果写入本地 checkpoint；本轮临时 localhost 服务已停止，浏览器视口已恢复默认，TPP 保留有效的 null 补测工程。公开交付不包含完整游戏、应用工程或私有运行证据，也没有实现产品接入。
