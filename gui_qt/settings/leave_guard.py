"""Qt-free unsaved-settings leave-guard copy (#202 Phase D).

The coordinator owns when to prompt and which copy to use. ``MainWindow``
still shows the Qt dialog and runs the unique save transaction.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SettingsLeaveGuardPrompt:
    """One save / discard / cancel prompt for leaving unsaved Settings."""

    kind: str
    title: str
    text: str
    informative: str
    save_label: str
    discard_label: str
    cancel_label: str


SETTINGS_LEAVE_GUARD_PROMPTS: dict[str, SettingsLeaveGuardPrompt] = {
    "workflow": SettingsLeaveGuardPrompt(
        kind="workflow",
        title="设置尚未保存",
        text="设置页有未保存的更改。",
        informative=(
            "当前任务会读取已保存的 translator_config.json；"
            "未保存的更改不会生效。"
        ),
        save_label="保存并继续",
        discard_label="不保存继续",
        cancel_label="取消",
    ),
    "registry_switch": SettingsLeaveGuardPrompt(
        kind="registry_switch",
        title="设置尚未保存",
        text="设置页有未保存的更改。",
        informative="切换工作区项目会重新加载设置，未保存的更改将丢失。",
        save_label="保存并切换",
        discard_label="不保存切换",
        cancel_label="取消",
    ),
    "close": SettingsLeaveGuardPrompt(
        kind="close",
        title="设置尚未保存",
        text="设置页有未保存的更改。",
        informative=(
            "直接关闭窗口会丢失尚未写入 translator_config.json 的修改。"
            "可先保存、放弃更改后退出，或取消以继续编辑。"
        ),
        save_label="保存并退出",
        discard_label="不保存退出",
        cancel_label="取消",
    ),
    "leave_tab": SettingsLeaveGuardPrompt(
        kind="leave_tab",
        title="设置尚未保存",
        text="设置页有未保存的更改。",
        informative="离开前可以保存设置，或留在设置页继续检查。",
        save_label="保存并离开",
        discard_label="不保存离开",
        cancel_label="留在设置页",
    ),
}


def settings_leave_guard_prompt(kind: str) -> SettingsLeaveGuardPrompt:
    try:
        return SETTINGS_LEAVE_GUARD_PROMPTS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown settings leave-guard kind: {kind!r}") from exc
