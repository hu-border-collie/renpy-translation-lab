"""Read-only planning for resuming a Batch translation package."""
from execution import batch_submit_recovery

TERMINAL_FAILURE_STATES = frozenset({
    'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED',
})


def next_action(manifest: dict, *, package_dir: str | None = None) -> str:
    """Choose one next step using the manifest and local submit journal."""
    if manifest.get('applied_at'):
        return ''
    summary = manifest.get('last_check_summary')
    if manifest.get('retry_of_manifest') and isinstance(summary, dict) and summary:
        gate = summary.get('writeback_gate')
        decision = gate.get('decision') if isinstance(gate, dict) else None
        if decision == 'deny' or (
            not decision and summary.get('safety_level') in ('warn', 'block')
        ):
            return ''
        return 'check'
    if summary or manifest.get('last_check'):
        return ''
    state = str(manifest.get('job_state') or '')
    if state in TERMINAL_FAILURE_STATES:
        return ''
    if state == 'RESULTS_MERGED':
        return 'check'
    if state == 'JOB_STATE_SUCCEEDED':
        return 'download'
    if manifest.get('job_name'):
        return 'status'
    if manifest.get('submit_disabled'):
        return ''
    uncertain = batch_submit_recovery.get_uncertain_submit_state(
        manifest, package_dir=package_dir,
    ) or {}
    if uncertain.get('kind') == 'job_created_uncommitted':
        return 'recover-submit'
    if uncertain.get('kind') == 'upload_pending_job_create':
        return 'resume-submit'
    return 'submit'
