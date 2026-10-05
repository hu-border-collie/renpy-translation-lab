"""Presentation-only projection of existing revision workflow/service state."""
from __future__ import annotations

from .user_copy import REVISION_GUIDANCE_COPY
from .work_modes import WorkMode


def revision_guidance(
    *, mode: WorkMode, running: bool, stopped: bool, workflow_status: str,
    writeback_status: str, can_apply: bool, has_preview: bool,
    proposal: dict[str, object] | None, reviewing: bool, has_corpus: bool,
    resume_available: bool, findings_available: bool = False,
) -> tuple[str, str, str]:
    """Return display key/title/next step; never authorize a workflow action.

    Inputs come from the owning coordinator and services, independently of
    widget visibility. Active/error states take precedence over older artifacts.
    """
    if running:
        key = "stopping" if stopped else "running"
    elif workflow_status == "stale":
        key = "stale"
    elif workflow_status == "stopped":
        key = "stopped"
    elif workflow_status == "failed":
        key = "failed"
    elif writeback_status in {"failed", "unknown", "stale"}:
        key = "stale" if writeback_status == "stale" else "failed"
    elif writeback_status == "applied":
        key = "applied"
    elif can_apply:
        key = "ready"
    elif has_preview:
        key = "preview"
    elif findings_available:
        key = "final_select"
    elif proposal is not None:
        key = (
            "select" if proposal.get("session_status") == "ready"
            and int(proposal.get("selectable_count") or 0) > 0 else "invalid"
        )
    elif reviewing:
        key = "review"
    elif has_corpus:
        key = "corpus"
    elif workflow_status == "waiting" or (workflow_status == "ready" and resume_available):
        key = "waiting"
    else:
        key = {
            WorkMode.SYNC_REVISION: "sync_empty",
            WorkMode.FINAL_REVIEW: "final_empty",
        }.get(mode, "empty")
    title, message = REVISION_GUIDANCE_COPY[key]
    if key == "waiting" and not resume_available:
        message = REVISION_GUIDANCE_COPY["waiting_unavailable"]
    return key, title, message
