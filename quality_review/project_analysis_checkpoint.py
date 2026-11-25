"""Private, request-bound response receipts for a project-analysis generation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from foundation.atomic_io import atomic_write_json
from providers.sync_model_backend import SyncGenerationResult
from quality_review.project_analysis import ProjectAnalysisError, stable_json_sha256


class AnalysisGenerationCheckpoint:
    """Keep returned responses separate from the draft artifacts they produce.

    The caller serializes generation against the same store. Each input
    binding has a pending generation that a normal invocation can resume.
    """

    def __init__(self, root: Path, *, force: bool, completed_run_id: str):
        self.pending = root / "pending.json"
        state = None if force else self._read(self.pending)
        if state is not None and (
            not isinstance(state.get("run_id"), str)
            or not re.fullmatch(r"[a-f0-9]{32}", state["run_id"])
            or type(state.get("force")) is not bool
            or state.get("binding") != root.name
        ):
            raise ProjectAnalysisError("Project Analysis generation checkpoint has an invalid identity.")
        if state is None or state["run_id"] == completed_run_id:
            state = {"run_id": uuid4().hex, "force": bool(force), "binding": root.name}
            self._write(self.pending, state)
        self.run_id: str = state["run_id"]
        self.force: bool = state["force"]
        self.directory = root / self.run_id

    @staticmethod
    def _read(path: Path) -> dict | None:
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            payload = envelope["payload"]
            if (
                envelope["version"] != 1 or not isinstance(payload, dict)
                or envelope["sha256"] != stable_json_sha256(payload)
            ):
                raise ValueError("checkpoint digest mismatch")
            return payload
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ProjectAnalysisError("Unreadable Project Analysis generation checkpoint.") from exc

    @staticmethod
    def _write(path: Path, payload: dict) -> None:
        try:
            atomic_write_json(path, {
                "version": 1,
                "payload": payload,
                "sha256": stable_json_sha256(payload),
            })
        except (OSError, ValueError, TypeError) as exc:
            raise ProjectAnalysisError(
                "Could not save the Project Analysis checkpoint. Generation stopped."
            ) from exc

    def response(self, key: str) -> tuple[str, SyncGenerationResult] | None:
        """Read only a usable response for this exact request in the current run."""
        record = self._read(self.directory / f"{key}.json")
        if record is None:
            return None
        try:
            if (record["key"] != key or record["run_id"] != self.run_id
                    or not isinstance(record["receipt_id"], str) or not record["receipt_id"]):
                raise ValueError("checkpoint identity mismatch")
            result = SyncGenerationResult(**record["result"])
            if not isinstance(result.response_text, str) or not result.response_text.strip():
                raise ValueError("checkpoint is missing response text")
        except (ValueError, TypeError, KeyError) as exc:
            raise ProjectAnalysisError("Invalid Project Analysis response checkpoint.") from exc
        return record["receipt_id"], result

    def save_response(self, key: str, result: SyncGenerationResult) -> str:
        """Persist a returned response and its usage before validation can raise."""
        receipt_id = uuid4().hex
        usable_text = str(result.response_text or "").strip()
        filename = f"{key}.json" if usable_text else f"{key}.invalid-{receipt_id}.json"
        self._write(self.directory / filename, {
            "run_id": self.run_id,
            "key": key,
            "receipt_id": receipt_id,
            "result": asdict(result),
        })
        return receipt_id

    def prepared_drafts(self) -> dict | None:
        """Read verified drafts after an interruption during their publication."""
        return self._read(self.directory / "drafts.json")

    def save_prepared_drafts(self, payload: dict) -> None:
        self._write(self.directory / "drafts.json", payload)

    def finish(self) -> None:
        """Discard this run's private receipts once the draft manifest is saved."""
        try:
            self.pending.unlink(missing_ok=True)
            for path in self.directory.glob("*.json"):
                path.unlink(missing_ok=True)
            if self.directory.is_dir():
                self.directory.rmdir()
            self.pending.parent.rmdir()
        except OSError:
            # The saved manifest identifies this run if cleanup is interrupted.
            pass
