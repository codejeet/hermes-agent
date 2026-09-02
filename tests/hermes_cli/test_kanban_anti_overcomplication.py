"""Hard completion gate for the Kanban anti-overcomplication pass."""

import json

import pytest

from hermes_cli import kanban_db as kb


def _claimed_worktree_task(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Implement a simpler client flow",
            assignee="test-worker",
            workspace_kind="worktree",
            workspace_path=str(tmp_path / "worktree"),
        )
        assert kb.claim_task(conn, task_id)
        run_id = kb.get_task(conn, task_id).current_run_id
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return task_id, run_id


def test_worktree_worker_cannot_complete_without_check(monkeypatch, tmp_path):
    task_id, _ = _claimed_worktree_task(monkeypatch, tmp_path)
    from tools import kanban_tools as kt

    response = json.loads(kt._handle_complete({"summary": "implemented and tested"}))

    assert "error" in response
    assert "anti-overcomplication" in response["error"].lower()
    with kb.connect() as conn:
        assert kb.get_task(conn, task_id).status == "running"


def test_worktree_worker_check_is_persisted_in_handoff(monkeypatch, tmp_path):
    task_id, _ = _claimed_worktree_task(monkeypatch, tmp_path)
    from tools import kanban_tools as kt

    response = json.loads(
        kt._handle_complete(
            {
                "summary": "implemented, simplified, and tested",
                "anti_overcomplication": {
                    "checked": True,
                    "removed": ["redundant approval modal", "unrequested PII masking"],
                    "note": "Kept tenant authorization because it prevents cross-client access.",
                },
            }
        )
    )

    assert response["ok"] is True
    with kb.connect() as conn:
        run = kb.latest_run(conn, task_id)
        assert run.metadata["anti_overcomplication"] == {
            "checked": True,
            "removed": ["redundant approval modal", "unrequested PII masking"],
            "note": "Kept tenant authorization because it prevents cross-client access.",
        }


def test_db_gate_blocks_worker_completion_bypass(monkeypatch, tmp_path):
    task_id, run_id = _claimed_worktree_task(monkeypatch, tmp_path)

    with kb.connect() as conn:
        with pytest.raises(kb.AntiOvercomplicationRequiredError):
            kb.complete_task(
                conn,
                task_id,
                summary="attempted CLI/import bypass",
                expected_run_id=run_id,
            )
        assert kb.get_task(conn, task_id).status == "running"


def test_scratch_research_task_does_not_require_implementation_check(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Summarize three papers", workspace_kind="scratch")
        assert kb.claim_task(conn, task_id)
        run_id = kb.get_task(conn, task_id).current_run_id
        assert kb.complete_task(
            conn,
            task_id,
            summary="three papers summarized",
            expected_run_id=run_id,
        )
        assert kb.get_task(conn, task_id).status == "done"
