"""Durable Sync run workflow state for the GUI (#348 P3).

The workbench never reads ``state.sqlite3`` itself. Every step invokes the
existing ``gemini_translate_batch.py`` machine interface and renders the
schema-v1 envelope returned by #347's service layer. Closing the local worker
never cancels a run: recovery is an explicit ``sync-status`` / ``sync-resume``
action, while an irreversible ``sync-cancel`` is a separate user action.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cli_contract

from .batch_workflow_support import machine_output_args
from .translation_workflow import WorkflowStep, WorkflowUpdate
from .user_copy import (
    DURABLE_SYNC_RUN_COPY,
    format_quality_gate_fact,
    format_safety_fact,
)

DURABLE_SYNC_SCRIPT = "gemini_translate_batch.py"

#: Run statuses whose persisted results may be checked and previewed.
CHECKABLE_RUN_STATUSES = frozenset({"completed", "completed_with_errors"})

#: Run statuses that stop the durable executor.
TERMINAL_RUN_STATUSES = frozenset(
    {"completed", "completed_with_errors", "failed", "cancelled"}
)

STEP_COPY: dict[str, tuple[str, str]] = {
    "sync-start": (
        "正在启动耐久同步翻译",
        "正在构建冻结翻译计划并创建可恢复的同步运行。",
    ),
    "sync-status": (
        "正在查询同步任务状态",
        "正在读取耐久运行的进度与下一步动作，不会调用模型。",
    ),
    "sync-resume": (
        "正在继续同步翻译",
        "正在从已持久化的进度恢复；已提交的请求不会重复发送。",
    ),
    "sync-cancel": (
        "正在取消同步任务",
        "正在请求停止后续调度；已经成功的结果仍会保留在运行记录中。",
    ),
    "sync-derive": (
        "正在派生新的同步任务",
        "正在基于已有成功结果创建可继续的新运行。",
    ),
    "check": (
        "正在检查同步翻译结果",
        "正在离线校验结果、结构与质量门禁，并生成绑定预览。",
    ),
    "apply": (
        "正在写回同步翻译预览",
        "正在复核源快照与预览绑定，然后原子写回项目脚本。",
    ),
}


def parse_machine_envelope(output: str) -> dict[str, Any] | None:
    """Return the schema-v1 envelope contained in command stdout, if valid."""
    try:
        return cli_contract.parse_result_envelope(output)
    except ValueError:
        return None


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def snapshot_of(envelope: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the durable run snapshot carried by a machine envelope."""
    if not isinstance(envelope, Mapping) or not envelope.get("ok"):
        return {}
    return _as_mapping(envelope.get("result"))


def run_status_of(snapshot: Mapping[str, Any] | None) -> str:
    return str((snapshot or {}).get("run_status") or "").strip()


def durable_sync_facts(
    snapshot: Mapping[str, Any] | None,
    *,
    run_dir: str = "",
) -> list[str]:
    """Render the public snapshot fields the GUI is allowed to show."""
    snapshot = _as_mapping(snapshot)
    if not snapshot:
        return []
    progress = _as_mapping(snapshot.get("progress"))
    requests = _as_mapping(progress.get("requests"))
    items = _as_mapping(progress.get("items"))
    usage = _as_mapping(progress.get("usage"))
    facts: list[str] = []
    run_id = str(snapshot.get("run_id") or "").strip()
    if run_id:
        facts.append(f"{DURABLE_SYNC_RUN_COPY['run_label']}：{run_id}")
    if run_dir:
        facts.append(f"{DURABLE_SYNC_RUN_COPY['run_dir_label']}：{run_dir}")
    status_label = DURABLE_SYNC_RUN_COPY["status_labels"].get(
        run_status_of(snapshot),
        run_status_of(snapshot) or DURABLE_SYNC_RUN_COPY["status_unknown"],
    )
    facts.append(f"{DURABLE_SYNC_RUN_COPY['status_label']}：{status_label}")
    expected = _as_int(items.get("expected"))
    accepted = _as_int(items.get("accepted"))
    unresolved = _as_int(items.get("unresolved"))
    if expected or accepted or unresolved:
        facts.append(
            f"条目：已接受 {accepted}/{expected}，未解决 {unresolved}"
        )
    total = _as_int(requests.get("total"))
    if total:
        succeeded = _as_int(requests.get("succeeded"))
        pending = _as_int(requests.get("pending"))
        in_flight = _as_int(requests.get("in_flight"))
        facts.append(
            f"请求：完成 {succeeded}/{total}，等待 {pending}，进行中 {in_flight}"
        )
    retryable_failed = _as_int(requests.get("retryable_failed"))
    terminal_failed = _as_int(requests.get("terminal_failed"))
    if retryable_failed:
        facts.append(f"可重试失败请求：{retryable_failed}")
    if terminal_failed:
        facts.append(f"无法解决请求：{terminal_failed}")
    outcome_unknown = _as_int(requests.get("outcome_unknown"))
    if outcome_unknown:
        facts.append(
            f"{DURABLE_SYNC_RUN_COPY['outcome_unknown_fact']}：{outcome_unknown}"
        )
    cancellation = _as_mapping(snapshot.get("cancellation"))
    if cancellation.get("requested"):
        facts.append(DURABLE_SYNC_RUN_COPY["cancel_requested_fact"])
    tokens = usage.get("total_tokens")
    if tokens is not None:
        facts.append(f"已知 Token：{_as_int(tokens)}")
    estimated = usage.get("estimated_cost")
    if estimated is not None:
        facts.append(f"预估成本：{estimated}")
    actual = usage.get("actual_cost")
    if actual is not None:
        facts.append(f"已知成本：{actual}")
    next_action = str(snapshot.get("next_action") or "").strip()
    if next_action:
        label = DURABLE_SYNC_RUN_COPY["next_action_labels"].get(
            next_action, next_action
        )
        facts.append(f"{DURABLE_SYNC_RUN_COPY['next_action_label']}：{label}")
    return facts


class SyncTranslationWorkflow:
    """Plan one durable Sync run across explicit, user-visible machine steps."""

    def __init__(
        self,
        pending_steps: list[str] | tuple[str, ...] = (),
        *,
        operation: str = "",
        run_id: str = "",
        profile_id: str = "",
        retry_unknown: bool = False,
        exclude_unknown: bool = False,
    ) -> None:
        self.manifest_path = ""
        self.run_id = str(run_id or "")
        self.profile_id = str(profile_id or "").strip()
        self.run_dir = ""
        self.run_snapshot: dict[str, Any] | None = None
        self.preview_ready = False
        #: Path of the checked manifest that bound the writeable preview.
        self.check_manifest = ""
        self.operation = str(operation or "")
        self._retry_unknown = bool(retry_unknown)
        self._exclude_unknown = bool(exclude_unknown)
        self._pending_steps = list(pending_steps)

    # -- constructors -------------------------------------------------

    @classmethod
    def start_new(cls, profile_id: str = "") -> "SyncTranslationWorkflow":
        """Start a fresh durable run and check its results when terminal."""
        return cls(["sync-start"], operation="start", profile_id=profile_id)

    @classmethod
    def query_latest(cls) -> "SyncTranslationWorkflow":
        """Query the latest durable run without resuming or spending quota."""
        return cls(["sync-status"], operation="query")

    @classmethod
    def resume_latest(cls) -> "SyncTranslationWorkflow":
        """Query the latest run, then resume or check it when appropriate."""
        return cls(["sync-status"], operation="resume")

    @classmethod
    def resume_run(cls, run_id: str) -> "SyncTranslationWorkflow":
        return cls(["sync-resume"], operation="resume", run_id=run_id)

    @classmethod
    def cancel_run(cls, run_id: str) -> "SyncTranslationWorkflow":
        return cls(["sync-cancel"], operation="cancel", run_id=run_id)

    @classmethod
    def derive_run(
        cls,
        run_id: str,
        *,
        retry_unknown: bool = False,
        exclude_unknown: bool = False,
    ) -> "SyncTranslationWorkflow":
        return cls(
            ["sync-derive"],
            operation="derive",
            run_id=run_id,
            retry_unknown=retry_unknown,
            exclude_unknown=exclude_unknown,
        )

    @classmethod
    def check_run(cls, run_id: str) -> "SyncTranslationWorkflow":
        return cls(["check"], operation="check", run_id=run_id)

    @classmethod
    def apply_run(cls, run_id: str) -> "SyncTranslationWorkflow":
        return cls(["apply"], operation="apply", run_id=run_id)

    # -- step planning -------------------------------------------------

    def current_step(self) -> WorkflowStep | None:
        if not self._pending_steps:
            return None
        key = self._pending_steps[0]
        heading, message = STEP_COPY.get(key, ("正在运行同步任务", ""))
        return WorkflowStep(
            key=key,
            args=self._step_args(key),
            heading=heading,
            message=message,
            script_basename=DURABLE_SYNC_SCRIPT,
        )

    def _step_args(self, key: str) -> list[str]:
        if key == "sync-start":
            args = ["sync-start"]
            if self.profile_id:
                args.extend(["--profile", self.profile_id])
        elif key == "sync-status":
            args = ["sync-status", self.run_id] if self.run_id else [
                "sync-status",
                "--latest",
            ]
        elif key == "sync-resume":
            args = ["sync-resume", self.run_id]
        elif key == "sync-cancel":
            args = ["sync-cancel", self.run_id]
        elif key == "sync-derive":
            args = ["sync-derive", self.run_id]
            if self._exclude_unknown:
                args.append("--exclude-unknown")
            elif self._retry_unknown:
                args.extend(["--retry-unknown", "--ack-duplicate-billing-risk"])
        elif key == "check":
            args = ["check", self.run_id]
        elif key == "apply":
            args = ["apply", self.run_id]
        else:  # pragma: no cover - constructors own the step vocabulary
            raise ValueError(f"Unknown durable Sync step: {key}")
        return machine_output_args(args)

    # -- step completion -----------------------------------------------

    def complete_current_step(self, exit_code: int, output: str) -> WorkflowUpdate:
        if not self._pending_steps:
            return WorkflowUpdate(
                status="failed",
                heading=DURABLE_SYNC_RUN_COPY["anomaly_heading"],
                message=DURABLE_SYNC_RUN_COPY["anomaly_message"],
                facts=[],
            )
        key = self._pending_steps.pop(0)
        envelope = parse_machine_envelope(output)
        if envelope is None:
            return self._interrupted_update(key, exit_code, output)
        if not envelope.get("ok"):
            return self._error_update(key, envelope)
        handler = getattr(self, f"_complete_{key.replace('-', '_')}")
        return handler(envelope)

    def _interrupted_update(
        self,
        key: str,
        exit_code: int,
        output: str,
    ) -> WorkflowUpdate:
        self._pending_steps.clear()
        facts = [f"退出码：{exit_code}"] if exit_code else []
        if output.strip():
            facts.append(DURABLE_SYNC_RUN_COPY["log_hint_fact"])
        if key in {"sync-start", "sync-resume"} and not self.run_id:
            # The worker died before any snapshot was rendered. Locate the
            # persisted run with a read-only status query so the user can
            # resume or cancel it instead of guessing a run id.
            self.operation = "recover"
            self._pending_steps.append("sync-status")
            return WorkflowUpdate(
                status="waiting",
                heading=DURABLE_SYNC_RUN_COPY["interrupted_locate_heading"],
                message=DURABLE_SYNC_RUN_COPY["interrupted_locate_message"],
                facts=facts,
                should_continue=True,
                timeline_step_key="sync-status",
            )
        message = (
            DURABLE_SYNC_RUN_COPY["worker_stopped_message"]
            if key in {"sync-start", "sync-resume"}
            else DURABLE_SYNC_RUN_COPY["command_interrupted_message"]
        )
        return WorkflowUpdate(
            status="failed",
            heading=DURABLE_SYNC_RUN_COPY["interrupted_heading"],
            message=message,
            facts=facts,
        )

    def _error_update(
        self,
        key: str,
        envelope: Mapping[str, Any],
    ) -> WorkflowUpdate:
        self._pending_steps.clear()
        error = _as_mapping(envelope.get("error"))
        code = str(error.get("code") or "").strip()
        message = DURABLE_SYNC_RUN_COPY["error_messages"].get(code)
        if not message:
            message = DURABLE_SYNC_RUN_COPY["error_fallback"].format(code=code or "UNKNOWN")
        facts = [f"错误码：{code}"] if code else []
        if key == "sync-status" and code == "SYNC_RUN_NOT_FOUND":
            return WorkflowUpdate(
                status="warning",
                heading=DURABLE_SYNC_RUN_COPY["no_run_heading"],
                message=message,
                facts=facts,
            )
        return WorkflowUpdate(
            status="failed",
            heading=DURABLE_SYNC_RUN_COPY["failed_heading"],
            message=message,
            facts=facts,
        )

    def _record_snapshot(self, envelope: Mapping[str, Any]) -> dict[str, Any]:
        snapshot = snapshot_of(envelope)
        self.run_snapshot = snapshot
        run_id = str(snapshot.get("run_id") or "").strip()
        if run_id:
            self.run_id = run_id
        artifacts = _as_mapping(envelope.get("artifacts"))
        run_dir = str(artifacts.get("run_dir") or "").strip()
        if run_dir:
            self.run_dir = run_dir
        return snapshot

    def _queue_check_after_run(self) -> None:
        if "check" not in self._pending_steps:
            self._pending_steps.insert(0, "check")

    def _complete_sync_start(self, envelope: Mapping[str, Any]) -> WorkflowUpdate:
        return self._complete_run_step(envelope, heading_key="started")

    def _complete_sync_resume(self, envelope: Mapping[str, Any]) -> WorkflowUpdate:
        return self._complete_run_step(envelope, heading_key="resumed")

    def _complete_run_step(
        self,
        envelope: Mapping[str, Any],
        *,
        heading_key: str,
    ) -> WorkflowUpdate:
        snapshot = self._record_snapshot(envelope)
        status = run_status_of(snapshot)
        facts = durable_sync_facts(snapshot, run_dir=self.run_dir)
        if status in CHECKABLE_RUN_STATUSES:
            self._queue_check_after_run()
            return WorkflowUpdate(
                status="running",
                heading=DURABLE_SYNC_RUN_COPY["terminal_heading"],
                message=DURABLE_SYNC_RUN_COPY["terminal_message"],
                facts=facts,
                should_continue=True,
                timeline_step_key="check",
            )
        if status == "cancelled":
            return WorkflowUpdate(
                status="warning",
                heading=DURABLE_SYNC_RUN_COPY["cancelled_heading"],
                message=DURABLE_SYNC_RUN_COPY["cancelled_message"],
                facts=facts,
                timeline_step_key="sync-cancel",
            )
        if status == "failed":
            return WorkflowUpdate(
                status="failed",
                heading=DURABLE_SYNC_RUN_COPY["failed_heading"],
                message=DURABLE_SYNC_RUN_COPY["run_failed_message"],
                facts=facts,
            )
        if status == "cancel_requested":
            return WorkflowUpdate(
                status="waiting",
                heading=DURABLE_SYNC_RUN_COPY["cancel_wait_heading"],
                message=DURABLE_SYNC_RUN_COPY["cancel_wait_message"],
                facts=facts,
            )
        return WorkflowUpdate(
            status="waiting",
            heading=DURABLE_SYNC_RUN_COPY[heading_key],
            message=DURABLE_SYNC_RUN_COPY["run_pending_message"],
            facts=facts,
        )

    def _complete_sync_status(self, envelope: Mapping[str, Any]) -> WorkflowUpdate:
        snapshot = self._record_snapshot(envelope)
        facts = durable_sync_facts(snapshot, run_dir=self.run_dir)
        next_action = str(snapshot.get("next_action") or "").strip()
        status = run_status_of(snapshot)
        if self.operation == "recover":
            if next_action == "derive":
                message = DURABLE_SYNC_RUN_COPY["recover_located_derive_message"]
            elif status in TERMINAL_RUN_STATUSES:
                message = DURABLE_SYNC_RUN_COPY["recover_located_terminal_message"]
            else:
                message = DURABLE_SYNC_RUN_COPY["recover_located_message"]
            return WorkflowUpdate(
                status="warning",
                heading=DURABLE_SYNC_RUN_COPY["recover_located_heading"],
                message=message,
                facts=facts,
            )
        if self.operation == "resume":
            if next_action == "resume":
                self._pending_steps.insert(0, "sync-resume")
                if "check" not in self._pending_steps:
                    self._pending_steps.insert(1, "check")
                return WorkflowUpdate(
                    status="running",
                    heading=DURABLE_SYNC_RUN_COPY["recover_heading"],
                    message=DURABLE_SYNC_RUN_COPY["recover_message"],
                    facts=facts,
                    should_continue=True,
                    timeline_step_key="sync-resume",
                )
            if next_action == "check":
                self._queue_check_after_run()
                return WorkflowUpdate(
                    status="running",
                    heading=DURABLE_SYNC_RUN_COPY["terminal_heading"],
                    message=DURABLE_SYNC_RUN_COPY["terminal_message"],
                    facts=facts,
                    should_continue=True,
                    timeline_step_key="check",
                )
        if status in CHECKABLE_RUN_STATUSES and next_action == "check":
            return WorkflowUpdate(
                status="done",
                heading=DURABLE_SYNC_RUN_COPY["completed_heading"],
                message=DURABLE_SYNC_RUN_COPY["completed_message"],
                facts=facts,
            )
        if next_action == "derive":
            return WorkflowUpdate(
                status="warning",
                heading=DURABLE_SYNC_RUN_COPY["derive_heading"],
                message=DURABLE_SYNC_RUN_COPY["derive_message"],
                facts=facts,
            )
        if next_action == "wait_cancel":
            return WorkflowUpdate(
                status="waiting",
                heading=DURABLE_SYNC_RUN_COPY["cancel_wait_heading"],
                message=DURABLE_SYNC_RUN_COPY["cancel_wait_message"],
                facts=facts,
            )
        return WorkflowUpdate(
            status="waiting",
            heading=DURABLE_SYNC_RUN_COPY["pending_heading"],
            message=DURABLE_SYNC_RUN_COPY["pending_message"],
            facts=facts,
        )

    def _complete_sync_cancel(self, envelope: Mapping[str, Any]) -> WorkflowUpdate:
        snapshot = self._record_snapshot(envelope)
        facts = durable_sync_facts(snapshot, run_dir=self.run_dir)
        status = run_status_of(snapshot)
        if status == "cancelled":
            return WorkflowUpdate(
                status="warning",
                heading=DURABLE_SYNC_RUN_COPY["cancelled_heading"],
                message=DURABLE_SYNC_RUN_COPY["cancel_done_message"],
                facts=facts,
            )
        if status == "cancel_requested":
            return WorkflowUpdate(
                status="waiting",
                heading=DURABLE_SYNC_RUN_COPY["cancel_wait_heading"],
                message=DURABLE_SYNC_RUN_COPY["cancel_wait_message"],
                facts=facts,
                timeline_step_key="sync-cancel",
            )
        if status in CHECKABLE_RUN_STATUSES:
            return WorkflowUpdate(
                status="warning",
                heading=DURABLE_SYNC_RUN_COPY["cancel_missed_heading"],
                message=DURABLE_SYNC_RUN_COPY["cancel_missed_message"],
                facts=facts,
            )
        return WorkflowUpdate(
            status="waiting",
            heading=DURABLE_SYNC_RUN_COPY["cancel_wait_heading"],
            message=DURABLE_SYNC_RUN_COPY["cancel_unknown_message"],
            facts=facts,
        )

    def _complete_sync_derive(self, envelope: Mapping[str, Any]) -> WorkflowUpdate:
        snapshot = self._record_snapshot(envelope)
        facts = durable_sync_facts(snapshot, run_dir=self.run_dir)
        status = run_status_of(snapshot)
        if status in CHECKABLE_RUN_STATUSES:
            self._queue_check_after_run()
            return WorkflowUpdate(
                status="running",
                heading=DURABLE_SYNC_RUN_COPY["derived_heading"],
                message=DURABLE_SYNC_RUN_COPY["derived_message"],
                facts=facts,
                should_continue=True,
                timeline_step_key="check",
            )
        return WorkflowUpdate(
            status="waiting",
            heading=DURABLE_SYNC_RUN_COPY["derived_heading"],
            message=DURABLE_SYNC_RUN_COPY["run_pending_message"],
            facts=facts,
        )

    def _complete_check(self, envelope: Mapping[str, Any]) -> WorkflowUpdate:
        result = _as_mapping(envelope.get("result"))
        check = _as_mapping(result.get("check"))
        gate = _as_mapping(check.get("writeback_gate"))
        decision = str(gate.get("decision") or "").strip()
        safety = str(check.get("safety_level") or "")
        quality_gate = _as_mapping(check.get("quality_gate"))
        check_status = str(check.get("check_status") or envelope.get("status") or "")
        artifacts = _as_mapping(envelope.get("artifacts"))
        self.check_manifest = str(artifacts.get("manifest") or "").strip()
        facts = []
        if safety:
            facts.append(format_safety_fact(safety, prefix="结构检查"))
        if quality_gate:
            facts.append(format_quality_gate_fact(quality_gate))
        if self.run_id:
            facts.append(f"{DURABLE_SYNC_RUN_COPY['run_label']}：{self.run_id}")
        if self.check_manifest:
            facts.append(f"{DURABLE_SYNC_RUN_COPY['check_manifest_label']}：{self.check_manifest}")
        self.preview_ready = decision == "allow"
        if self.preview_ready:
            status = "warning" if check_status == "ready_with_warnings" else "done"
            return WorkflowUpdate(
                status=status,
                heading=DURABLE_SYNC_RUN_COPY["check_ready_heading"],
                message=DURABLE_SYNC_RUN_COPY["check_ready_message"],
                facts=facts,
            )
        reason = str(gate.get("reason") or gate.get("message") or "").strip()
        message = DURABLE_SYNC_RUN_COPY["check_blocked_message"]
        if reason:
            message = f"{message}（{reason}）"
        return WorkflowUpdate(
            status="warning",
            heading=DURABLE_SYNC_RUN_COPY["check_blocked_heading"],
            message=message,
            facts=facts,
        )

    def _complete_apply(self, envelope: Mapping[str, Any]) -> WorkflowUpdate:
        result = _as_mapping(envelope.get("result"))
        apply_result = _as_mapping(result.get("apply"))
        last_result = str(apply_result.get("last_apply_result") or "").strip()
        applied_files = apply_result.get("applied_files")
        facts: list[str] = []
        if isinstance(applied_files, list):
            facts.append(f"已写回文件：{len(applied_files)} 个")
        if last_result:
            facts.append(f"写回结果：{last_result}")
        heading = (
            DURABLE_SYNC_RUN_COPY["already_applied_heading"]
            if last_result == "already_applied"
            else DURABLE_SYNC_RUN_COPY["applied_heading"]
        )
        message = (
            DURABLE_SYNC_RUN_COPY["already_applied_message"]
            if last_result == "already_applied"
            else DURABLE_SYNC_RUN_COPY["applied_message"]
        )
        self.preview_ready = False
        return WorkflowUpdate(
            status="done",
            heading=heading,
            message=message,
            facts=facts,
        )
