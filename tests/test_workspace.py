import os
import stat

import pytest

from swe_agent.config import Settings
from swe_agent.github.workspace import WorkspaceError, WorkspaceManager


def make_manager(tmp_path):
    return WorkspaceManager(Settings(_env_file=None, workspace_dir=tmp_path / "ws"))


def test_create_makes_folder_inside_base(tmp_path):
    manager = make_manager(tmp_path)
    ws = manager.create()
    assert ws.root.is_dir()
    assert ws.root.parent == manager.base
    assert len(ws.run_id) == 12


@pytest.mark.parametrize("bad", ["../x", "..", "abc", "A" * 12, "../../Windows", "a/b"])
def test_bad_run_ids_are_rejected(tmp_path, bad):
    manager = make_manager(tmp_path)
    with pytest.raises(WorkspaceError):
        manager.create(bad)
    with pytest.raises(WorkspaceError):
        manager.delete(bad)


def test_duplicate_run_id_fails(tmp_path):
    manager = make_manager(tmp_path)
    manager.create("aaaaaaaaaaaa")
    with pytest.raises(WorkspaceError):
        manager.create("aaaaaaaaaaaa")


def test_delete_removes_read_only_files(tmp_path):
    manager = make_manager(tmp_path)
    ws = manager.create()
    locked = ws.repo_dir / ".git" / "objects" / "pack.idx"
    locked.parent.mkdir(parents=True)
    locked.write_text("data")
    os.chmod(locked, stat.S_IREAD)  # like Windows git files
    manager.delete(ws.run_id)
    assert not ws.root.exists()


def test_get_missing_workspace_fails(tmp_path):
    with pytest.raises(WorkspaceError):
        make_manager(tmp_path).get("bbbbbbbbbbbb")