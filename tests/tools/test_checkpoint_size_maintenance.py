"""Size maintenance must not repeatedly block otherwise valid checkpoints."""

import pytest

import tools.checkpoint_manager as checkpoints


@pytest.fixture
def overfull_store(tmp_path, monkeypatch):
    store = tmp_path / "store"
    store.mkdir()
    state = {"count": 2, "now": 100.0, "calls": []}
    ref = "refs/hermes/project"

    def run_git(args, *_args, **_kwargs):
        state["calls"].append(args)
        if args[0] == "for-each-ref":
            return True, ref, ""
        if args[:2] == ["rev-list", "--count"]:
            return True, str(state["count"]), ""
        if args[:2] == ["rev-list", "--reverse"]:
            return True, "old\nlatest", ""
        if args[0] == "rev-parse":
            return True, "tree", ""
        if args[0] == "log":
            return True, "checkpoint", ""
        if args[0] == "commit-tree":
            return True, "retained", ""
        if args[0] == "update-ref":
            if state.get("update_fails"):
                return False, "", "cannot lock ref"
            state["count"] = 1
        return True, "", ""

    monkeypatch.setattr(checkpoints, "_run_git", run_git)
    monkeypatch.setattr(checkpoints, "_dir_size_bytes", lambda _: 2 * 1024 * 1024)
    monkeypatch.setattr(checkpoints.time, "monotonic", lambda: state["now"])
    return store, state


def test_size_pruning_stops_when_no_more_history_can_be_removed(overfull_store):
    store, state = overfull_store
    manager = checkpoints.CheckpointManager(enabled=True, max_total_size_mb=1)

    manager._enforce_size_cap(store)

    assert state["count"] == 1
    counts = [call for call in state["calls"] if call[:2] == ["rev-list", "--count"]]
    # One pass trims history; the next sees the retained minimum and stops.
    assert len(counts) == 2
    assert len([call for call in state["calls"] if call[0] == "gc"]) == 1


def test_retained_minimum_does_not_trigger_useless_gc(overfull_store):
    store, state = overfull_store
    state["count"] = 1
    manager = checkpoints.CheckpointManager(enabled=True, max_total_size_mb=1)

    manager._enforce_size_cap(store)

    assert state["count"] == 1
    assert not any(call[0] in {"update-ref", "reflog", "gc"} for call in state["calls"])


def test_failed_ref_update_is_not_treated_as_pruning_progress(overfull_store):
    store, state = overfull_store
    state["update_fails"] = True
    manager = checkpoints.CheckpointManager(enabled=True, max_total_size_mb=1)

    manager._enforce_size_cap(store)

    assert state["count"] == 2
    assert len([call for call in state["calls"] if call[0] == "update-ref"]) == 1
    assert not any(call[0] == "gc" for call in state["calls"])


def test_overfull_store_cleanup_is_throttled_but_eventually_retried(overfull_store):
    store, state = overfull_store
    manager = checkpoints.CheckpointManager(enabled=True, max_total_size_mb=1)
    manager._enforce_size_cap(store)
    first_calls = len(state["calls"])

    # A subsequent edit adds history again; it must not immediately rescan
    # every project or garbage-collect the shared store again.
    state["count"] = 2
    state["now"] += 10
    manager.new_turn()
    manager._enforce_size_cap(store)
    assert len(state["calls"]) == first_calls
    assert state["count"] == 2

    state["now"] += 600
    manager._enforce_size_cap(store)
    assert len(state["calls"]) > first_calls
    assert state["count"] == 1


def test_checkpoint_and_rollback_still_work_during_maintenance_cooldown(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", tmp_path / "checkpoints")
    # Model a store whose retained minimum remains above the configured cap,
    # while letting all Git snapshot/restore operations run on disposable data.
    monkeypatch.setattr(checkpoints, "_dir_size_bytes", lambda _: 2 * 1024 * 1024)
    monkeypatch.setattr(checkpoints.time, "monotonic", lambda: 100.0)
    project = tmp_path / "project"
    project.mkdir()
    source = project / "main.py"
    source.write_text("original\n")
    manager = checkpoints.CheckpointManager(enabled=True, max_total_size_mb=1)

    assert manager.ensure_checkpoint(str(project), "initial")
    original = manager.list_checkpoints(str(project))[0]["hash"]
    source.write_text("changed\n")
    manager.new_turn()
    assert manager.ensure_checkpoint(str(project), "changed")
    assert len(manager.list_checkpoints(str(project))) == 2

    source.write_text("before rollback\n")
    restored = manager.restore(str(project), original)
    assert restored["success"]
    assert source.read_text() == "original\n"
    undo = manager.list_checkpoints(str(project))[0]
    assert undo["reason"].startswith("pre-rollback snapshot")
    assert manager.restore(str(project), undo["hash"])["success"]
    assert source.read_text() == "before rollback\n"
