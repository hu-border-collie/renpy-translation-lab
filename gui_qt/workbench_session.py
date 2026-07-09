"""Per-work-mode session state for the workbench (GUI IA P1a / #160, P1c / #162).

Switching the left nav must freeze the previous mode's runtime state and restore
it when the user returns — not globally clear workflow / writeback / snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkbenchModeSession:
    """Session bag for one WorkMode while the user works on another page."""

    workflow: Any | None = None
    workflow_step_output_lines: list[str] = field(default_factory=list)
    writeback_manifest_path: str = ""
    completed_manifest_snapshot: dict[str, object] | None = None
    viewing_completed_manifest: bool = False
    keyword_merge_candidates_path: str = ""
    # Workbench status-tab index (batch: 0=准备, 1=执行, 2=结果). Default 执行.
    stage_index: int = 1
    # UI snapshots so non-resume / offline modes do not fall back to blank idle.
    workflow_status: str = ""
    workflow_heading: str = ""
    workflow_message: str = ""
    workflow_facts: list[str] = field(default_factory=list)
    writeback_summary: Any | None = None  # WritebackSummary | None

    def is_empty(self) -> bool:
        return (
            self.workflow is None
            and not self.workflow_step_output_lines
            and not self.writeback_manifest_path
            and self.completed_manifest_snapshot is None
            and not self.viewing_completed_manifest
            and not self.keyword_merge_candidates_path
            and not self.has_workflow_ui()
            and not self.workflow_facts
            and self.writeback_summary is None
        )

    def has_workflow_ui(self) -> bool:
        return bool(
            self.workflow_status
            or self.workflow_heading
            or self.workflow_message
            or self.workflow_facts
        )
