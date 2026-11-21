"""Background jobs: one active job per run, a small worker pool."""

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime

from swe_agent.api.models import JobView
from swe_agent.logging_setup import redact_secrets

logger = logging.getLogger(__name__)

ACTIVE = ("queued", "running")
_KEEP_JOBS = 500


class JobBusy(Exception):
    """A job for this run is already queued or running."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class _Job:
    action: str
    state: str = "queued"
    error: str = ""
    queued_at: str = field(default_factory=_now)
    started_at: str = ""
    finished_at: str = ""


class JobManager:
    def __init__(self, workers: int = 1) -> None:
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="agent-job")
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()

    def _snapshot(self, job: _Job) -> JobView:
        return JobView(
            state=job.state,  # type: ignore[arg-type]
            action=job.action,
            error=job.error,
            queued_at=job.queued_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )

    def submit(self, run_id: str, action: str, work: Callable[[], object]) -> JobView:
        """Queue work for a run. Raises JobBusy if the run already has an active job."""
        with self._lock:
            current = self._jobs.get(run_id)
            if current is not None and current.state in ACTIVE:
                raise JobBusy(f"Run {run_id} is busy ({current.action}).")
            if len(self._jobs) > _KEEP_JOBS:  # forget the oldest finished jobs
                finished = [k for k, j in self._jobs.items() if j.state not in ACTIVE]
                for key in finished[:100]:
                    del self._jobs[key]
            job = _Job(action=action)
            self._jobs[run_id] = job
            view = self._snapshot(job)
        self._pool.submit(self._run, run_id, job, work)
        return view

    def _run(self, run_id: str, job: _Job, work: Callable[[], object]) -> None:
        with self._lock:
            job.state, job.started_at = "running", _now()
        try:
            work()
        except Exception as exc:  # noqa: BLE001 (a job must never crash the worker thread)
            logger.exception("job failed run_id=%s action=%s", run_id, job.action)
            with self._lock:
                job.state = "error"
                job.error = redact_secrets(f"{type(exc).__name__}: {exc}")[:500]
                job.finished_at = _now()
            return
        with self._lock:
            job.state, job.finished_at = "done", _now()

    def view(self, run_id: str) -> JobView:
        with self._lock:
            job = self._jobs.get(run_id)
            return self._snapshot(job) if job else JobView()

    def busy(self, run_id: str) -> bool:
        return self.view(run_id).state in ACTIVE

    def shutdown(self) -> None:
        """Stop accepting work. A running job is not killed; it stops at its next checkpoint."""
        self._pool.shutdown(wait=False, cancel_futures=True)