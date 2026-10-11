"""Host cancellation must fence LCM publication as well as SessionDB writes."""

import copy
import importlib
import sys
import threading
import time
from pathlib import Path
from types import ModuleType

import pytest

from agent.auxiliary_client import _aux_interrupt_cancel_requested
from agent.conversation_compression import (
    CompressionCommitFence,
    _supported_compression_kwargs,
    compress_context,
    run_compress_context_with_progress_timeout,
)
from hermes_state import SessionDB


# This integration suite exercises the installed third-party engine alongside
# the real host. Register its package without importing its plugin entrypoint
# or adding its root to sys.path, where tools.py would shadow Hermes tools.
_plugin_dir = Path.home() / ".hermes" / "plugins" / "hermes-lcm"
if not (_plugin_dir / "engine.py").is_file():
    pytest.skip("installed hermes-lcm integration unavailable", allow_module_level=True)
_plugin = ModuleType("hermes_lcm")
_plugin.__path__ = [str(_plugin_dir)]
sys.modules.setdefault("hermes_lcm", _plugin)
engine_module = importlib.import_module("hermes_lcm.engine")
LCMConfig = importlib.import_module("hermes_lcm.config").LCMConfig
LCMEngine = engine_module.LCMEngine


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    from run_agent import AIAgent

    db = SessionDB(tmp_path / "state.db")
    db.create_session("guard-session", source="cli")
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        enabled_toolsets=[],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_db=db,
        session_id="guard-session",
    )
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            fresh_tail_count=2,
            leaf_chunk_tokens=120,
            dynamic_leaf_chunk_enabled=True,
            dynamic_leaf_chunk_max=240,
            dynamic_leaf_chunk_max_passes=1,
            incremental_max_depth=0,
        ),
        hermes_home=str(home),
    )
    engine.on_session_start("guard-session", platform="cli", context_length=200_000)
    engine.threshold_tokens = 1
    agent.context_compressor = engine
    agent._compression_feasibility_checked = True
    agent._cached_system_prompt = "system"
    agent.compression_in_place = True
    monkeypatch.setattr(agent, "_build_system_prompt", lambda *_args, **_kwargs: "system")
    monkeypatch.setattr(
        engine_module, "summarize_with_escalation", lambda **_kwargs: ("Earlier facts retained.", 1)
    )
    messages = [{"role": "system", "content": "system"}] + [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"Fact {index}: " + "detail " * 35,
        }
        for index in range(12)
    ]
    for message in messages:
        db.append_message("guard-session", message["role"], message["content"])
    try:
        yield agent, engine, db, messages
    finally:
        agent.close()
        engine.shutdown()
        db.close()


def _compress(agent, messages, fence=None):
    return compress_context(
        agent, copy.deepcopy(messages), "system", approx_tokens=10_000, commit_fence=fence
    )


def _frontier(engine):
    return engine._lifecycle.get_by_conversation(engine._conversation_id).current_frontier_store_id


def _thread_call(fn):
    result = {}
    done = threading.Event()

    def run():
        try:
            result["value"] = fn()
        except BaseException as exc:
            result["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, done, result


def test_timed_out_summary_cannot_publish_late_and_retry_succeeds(runtime, monkeypatch, caplog):
    agent, engine, db, messages = runtime
    summary_started = threading.Event()
    release_summary = threading.Event()
    worker_done = threading.Event()
    worker_errors = []

    def summarize(**_kwargs):
        summary_started.set()
        assert release_summary.wait(10)
        return "Earlier facts retained.", 1

    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)

    def worker(fence):
        try:
            return _compress(agent, messages, fence)
        except BaseException as exc:
            worker_errors.append(exc)
            raise
        finally:
            worker_done.set()

    try:
        returned, _ = run_compress_context_with_progress_timeout(
            worker=worker,
            messages=messages,
            system_prompt_fallback="system",
            idle_timeout_seconds=0.3,
            total_ceiling_seconds=0.6,
        )
        assert summary_started.wait(5)
        assert returned is messages
        assert engine._dag.get_session_nodes("guard-session") == []
    finally:
        release_summary.set()
        assert worker_done.wait(10)

    assert worker_errors == []
    assert engine._dag.get_session_nodes("guard-session") == []
    assert _frontier(engine) == 0
    assert engine._last_compacted_store_id == 0
    assert len(db.get_messages_as_conversation("guard-session")) == len(messages)
    assert db.get_compression_lock_holder("guard-session") is None
    assert any(
        '"failure_class":"commit_fence_cancelled"' in record.getMessage()
        for record in caplog.records
    )

    retried, _ = _compress(agent, messages, CompressionCommitFence())
    assert len(engine._dag.get_session_nodes("guard-session")) == 1
    assert _frontier(engine) > 0
    assert len(retried) < len(messages)
    assert agent._last_compression_attempt_in_place is True


def test_hard_stop_before_publication_preserves_frontier(runtime, monkeypatch):
    agent, engine, db, messages = runtime

    def summarize(**_kwargs):
        agent._hard_interrupt_requested.set()
        return "Earlier facts retained.", 1

    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)
    fence = CompressionCommitFence()
    returned, _ = _compress(agent, messages, fence)

    assert returned == messages
    assert engine._dag.get_session_nodes("guard-session") == []
    assert _frontier(engine) == 0
    assert not fence.commit_in_flight
    assert db.get_compression_lock_holder("guard-session") is None


def test_admitted_publication_defers_hard_stop_through_host_handoff(runtime, monkeypatch):
    agent, engine, db, messages = runtime
    publication_started = threading.Event()
    release_publication = threading.Event()
    fence = CompressionCommitFence()
    publish = engine._dag.publish_node_and_advance_frontier
    cancellation_checks = []

    def blocked_publish(*args, **kwargs):
        publication_started.set()
        assert release_publication.wait(10)
        return publish(*args, **kwargs)

    def check_condensation_cancel(**_kwargs):
        cancellation_checks.append(_aux_interrupt_cancel_requested())

    monkeypatch.setattr(engine._dag, "publish_node_and_advance_frontier", blocked_publish)
    monkeypatch.setattr(engine, "_maybe_condense", check_condensation_cancel)
    thread, done, result = _thread_call(lambda: _compress(agent, messages, fence))
    try:
        assert publication_started.wait(5)
        assert fence.commit_in_flight
        assert fence.try_cancel_before_commit() is None
        agent._hard_interrupt_requested.set()
        fence.revoke_commit_admission()
    finally:
        release_publication.set()
        assert done.wait(10)
        thread.join(1)

    assert "error" not in result
    compressed, _ = result["value"]
    assert cancellation_checks == [False]
    assert len(compressed) < len(messages)
    assert agent._last_compression_attempt_in_place is True
    assert len(db.get_messages_as_conversation("guard-session")) == len(compressed)
    assert len(engine._dag.get_session_nodes("guard-session")) == 1
    assert not fence.commit_in_flight
    assert fence.try_cancel_before_commit() is False
    assert db.get_compression_lock_holder("guard-session") is None


def test_publication_exception_releases_fence_and_lease(runtime, monkeypatch):
    agent, engine, db, messages = runtime
    publish = engine._dag.publish_node_and_advance_frontier

    def fail_publication(*_args, **_kwargs):
        raise RuntimeError("publication failed before mutation")

    monkeypatch.setattr(engine._dag, "publish_node_and_advance_frontier", fail_publication)
    fence = CompressionCommitFence()
    with pytest.raises(RuntimeError, match="publication failed before mutation"):
        _compress(agent, messages, fence)

    assert not fence.commit_in_flight
    assert fence.try_cancel_before_commit() is False
    assert _frontier(engine) == 0
    assert db.get_compression_lock_holder("guard-session") is None
    monkeypatch.setattr(engine._dag, "publish_node_and_advance_frontier", publish)
    compressed, _ = _compress(agent, messages, CompressionCommitFence())
    assert len(compressed) < len(messages)


def test_no_fence_keeps_direct_compression_behavior(runtime):
    agent, engine, _db, messages = runtime
    compressed, _ = _compress(agent, messages)
    assert len(compressed) < len(messages)
    assert len(engine._dag.get_session_nodes("guard-session")) == 1


def test_optional_condensation_cannot_start_with_exhausted_attempt_budget(runtime, monkeypatch):
    agent, engine, _db, messages = runtime
    engine._config.summary_timeout_ms = 0
    engine._config.incremental_max_depth = 2
    engine._config.condensation_fanin = 1
    depths = []

    def summarize(**kwargs):
        depths.append(kwargs["depth"])
        return "Earlier facts retained.", 1

    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)
    _compress(agent, messages, CompressionCommitFence())

    assert depths == [0]


def test_condensation_uses_remaining_deadline(runtime, monkeypatch):
    _agent, engine, _db, messages = runtime
    engine.compress(copy.deepcopy(messages))
    engine._config.incremental_max_depth = 1
    engine._config.condensation_fanin = 1
    timeouts = []

    def summarize(**kwargs):
        timeouts.append(kwargs["timeout"])
        return "Condensed facts.", 1

    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)
    engine._maybe_condense(deadline=time.monotonic() + 5)

    assert len(timeouts) == 1
    assert 0 < timeouts[0] <= 5
    assert any(node.depth == 1 for node in engine._dag.get_session_nodes("guard-session"))


def test_condensation_deadline_expiry_does_not_abort_admitted_handoff(runtime, monkeypatch):
    agent, engine, db, messages = runtime
    engine._config.incremental_max_depth = 1
    engine._config.condensation_fanin = 1

    def deadline_expired(*_args, **_kwargs):
        raise TimeoutError("condensation budget expired before dispatch")

    monkeypatch.setattr(engine, "_condense_summary_nodes", deadline_expired)
    compressed, _ = _compress(agent, messages, CompressionCommitFence())

    assert len(compressed) < len(messages)
    assert agent._last_compression_attempt_in_place is True
    assert len(db.get_messages_as_conversation("guard-session")) == len(compressed)
    assert len(engine._dag.get_session_nodes("guard-session")) == 1


def test_exception_after_publication_releases_fence_without_deleting_raw_history(runtime, monkeypatch):
    agent, engine, db, messages = runtime

    def fail_assembly(*_args, **_kwargs):
        raise RuntimeError("assembly failed after publication")

    monkeypatch.setattr(engine, "_assemble_context", fail_assembly)
    fence = CompressionCommitFence()
    with pytest.raises(RuntimeError, match="assembly failed after publication"):
        _compress(agent, messages, fence)

    # This guard coordinates cancellation, not transactions across two DBs.
    # Unexpected later exceptions preserve durable partial progress and raw
    # history, while releasing the ownership needed for explicit recovery.
    assert len(engine._dag.get_session_nodes("guard-session")) == 1
    assert len(engine._store.get_session_messages("guard-session")) == len(messages)
    assert len(db.get_messages_as_conversation("guard-session")) == len(messages)
    assert not fence.commit_in_flight
    assert fence.try_cancel_before_commit() is False
    assert db.get_compression_lock_holder("guard-session") is None


def test_preexisting_unadopted_publication_requires_explicit_recovery(runtime):
    agent, engine, db, messages = runtime
    # Reproduce the pre-fix state: LCM committed a compacted prefix, but the
    # host never adopted it. Recreating an agent must not silently reset that
    # durable frontier or treat its unchanged transcript as new raw history.
    engine.compress(copy.deepcopy(messages))
    frontier = _frontier(engine)
    replacement = engine.clone_for_agent()
    replacement.on_session_start("guard-session", platform="cli", context_length=200_000)
    replacement.threshold_tokens = 1
    agent.context_compressor = replacement
    try:
        assert _frontier(replacement) == frontier
        with pytest.raises(RuntimeError, match="lacks complete raw store lineage"):
            _compress(agent, messages, CompressionCommitFence())
        assert _frontier(replacement) == frontier
        assert len(replacement._dag.get_session_nodes("guard-session")) == 1
        assert len(db.get_messages_as_conversation("guard-session")) == len(messages)
    finally:
        agent.context_compressor = engine
        replacement.shutdown()


def test_commit_guard_is_only_forwarded_to_compatible_engine_signatures():
    def legacy(messages, current_tokens=None):
        return messages

    def compatible(messages, current_tokens=None, commit_guard=None):
        return messages

    guard = lambda: None
    kwargs = dict(
        current_tokens=100, focus_topic=None, force=False,
        memory_context="", commit_guard=guard,
    )
    assert _supported_compression_kwargs(legacy, **kwargs) == {"current_tokens": 100}
    assert _supported_compression_kwargs(compatible, **kwargs)["commit_guard"] is guard
