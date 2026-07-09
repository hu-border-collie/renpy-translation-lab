"""Per-work-mode session state for the workbench (GUI IA P1a / #160).

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
    # Optional UI snapshot keys (status strings) for tests / future stage index.
    stage_index: int = 0

    def is_empty(self) -> bool:
        return (
            self.workflow is None
            and not self.workflow_step_output_lines
            and not self.writeback_manifest_path
            and self.completed_manifest_snapshot is None
            and not self.viewing_completed_manifest
            and not self.keyword_merge_candidates_path
        )
