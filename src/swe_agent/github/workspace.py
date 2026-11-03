"""One isolated folder per agent run."""

from __future__ import annotations

import logging
import os
import re
import shutil
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path

from swe_agent.config import Settings, get_settings

logger = logging.getLogger(__name__)

_RUN_ID_RE = re.compile(r"[a-f0-9]{12}")


class WorkspaceError(Exception):
    """Something is wrong with a workspace (bad id, already exists...)."""


@dataclass(frozen=True)
class Workspace:
    run_id: str
    root: Path

    @property
    def repo_dir(self) -> Path:
        return self.root / "repo"


def _on_remove_error(func, path, _exc_info) -> None:
    """Windows keeps .git files read-only. Make them writable, then retry."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


class WorkspaceManager:
    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self.base: Path = Path(settings.workspace_dir).resolve()
        self.base.mkdir(parents=True, exist_ok=True)

    def _path_for(self, run_id: str) -> Path:
        if not _RUN_ID_RE.fullmatch(run_id):
            raise WorkspaceError(f"Invalid run id: {run_id!r}")
        path = (self.base / run_id).resolve()
        if path.parent != self.base:  # belt and braces
            raise WorkspaceError("Workspace path escapes the workspace folder.")
        return path

    def create(self, run_id: str | None = None) -> Workspace:
        run_id = run_id or uuid.uuid4().hex[:12]
        root = self._path_for(run_id)
        try:
            root.mkdir(exist_ok=False)
        except FileExistsError as exc:
            raise WorkspaceError(f"Workspace {run_id} already exists.") from exc
        logger.info("workspace created run_id=%s", run_id)
        return Workspace(run_id=run_id, root=root)

    def get(self, run_id: str) -> Workspace:
        root = self._path_for(run_id)
        if not root.is_dir():
            raise WorkspaceError(f"Workspace {run_id} does not exist.")
        return Workspace(run_id=run_id, root=root)

    def delete(self, run_id: str) -> None:
        root = self._path_for(run_id)
        if root.exists():
            shutil.rmtree(root, onerror=_on_remove_error)
            logger.info("workspace deleted run_id=%s", run_id)