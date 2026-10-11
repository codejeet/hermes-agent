"""A verified SessionDB branch recovers unadopted LCM publication."""

import copy
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore, build_session_key


_plugin_dir = Path.home() / ".hermes" / "plugins" / "hermes-lcm"
if not (_plugin_dir / "engine.py").is_file():
    pytest.skip("installed hermes-lcm integration unavailable", allow_module_level=True)
_plugin = ModuleType("hermes_lcm")
_plugin.__path__ = [str(_plugin_dir)]
sys.modules.setdefault("hermes_lcm", _plugin)
engine_module = importlib.import_module("hermes_lcm.engine")
LCMConfig = importlib.import_module("hermes_lcm.config").LCMConfig


def test_discord_branch_recovers_same_conversation_without_losing_history(
    tmp_path, monkeypatch
):
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = SessionSource(
        platform=Platform.DISCORD, chat_id="test-chat", user_id="test-user"
    )
    session_key = build_session_key(source)
    sessions = SessionStore(home / "sessions", GatewayConfig())
    db = sessions._db
    assert db is not None
    parent_id = sessions.get_or_create_session(source).session_id
    messages = [{"role": "system", "content": "Keep the complete conversation."}]
    for index in range(12):
        messages.append({
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"Fact {index}: " + "detail " * 35,
        })
    messages[1]["api_content"] = "Exact original provider bytes for fact zero."
    messages[2].update({
        "tool_calls": [
            {
                "id": "call_recovery",
                "type": "function",
                "function": {
                    "name": "terminal",
                    "arguments": '{"command":"printf test"}',
                },
            }
        ],
        "reasoning": "Synthetic reasoning metadata.",
        "reasoning_content": "Synthetic reasoning content.",
        "reasoning_details": [{"type": "reasoning.text", "text": "Synthetic detail."}],
        "codex_reasoning_items": [{"type": "reasoning", "id": "synthetic-reasoning"}],
        "codex_message_items": [{"type": "message", "id": "synthetic-message"}],
        "finish_reason": "tool_calls",
    })
    messages[3:3] = [
        {
            "role": "tool",
            "content": "Synthetic tool result.",
            "tool_call_id": "call_recovery",
            "tool_name": "terminal",
            "effect_disposition": "performed",
            "observed": True,
            "display_kind": "text",
            "display_metadata": {"synthetic": True},
        },
        {"role": "assistant", "content": "Inspection finished."},
    ]
    db.append_message(parent_id, "user", "Archived parent history must also survive.")
    db.archive_and_compact(parent_id, messages)
    original_history = sessions.load_transcript(parent_id)
    original_rows = db.get_messages(parent_id)
    original_all_rows = db.get_messages(parent_id, include_inactive=True)
    assert len(original_all_rows) > len(original_rows)
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=2,
        leaf_chunk_tokens=120,
        dynamic_leaf_chunk_enabled=True,
        dynamic_leaf_chunk_max=240,
        dynamic_leaf_chunk_max_passes=1,
        incremental_max_depth=0,
    )
    monkeypatch.setattr(
        engine_module,
        "summarize_with_escalation",
        lambda **_kwargs: ("Earlier facts retained.", 1),
    )
    engines = []

    def new_engine(session_id):
        engine = engine_module.LCMEngine(config=config, hermes_home=str(home))
        engines.append(engine)
        engine.on_session_start(
            session_id,
            platform="discord",
            conversation_id=session_key,
            context_length=200_000,
        )
        engine.threshold_tokens = 1
        return engine

    try:
        parent = new_engine(parent_id)
        parent.compress(copy.deepcopy(original_history))
        parent_nodes = copy.deepcopy(parent._dag.get_session_nodes(parent_id))
        parent_raw = copy.deepcopy(parent._store.get_session_messages(parent_id))
        assert len(parent_nodes) == 1
        assert sessions.load_transcript(parent_id) == original_history
        rebound = new_engine(parent_id)
        with pytest.raises(RuntimeError, match="lacks complete raw store lineage"):
            rebound.compress(copy.deepcopy(original_history))

        # Use the same supported storage/routing APIs as /branch, but verify
        # the complete copy before switching. The slash handler intentionally
        # tolerates partial copies and normalizes its model-history projection.
        parent_session = db.get_session(parent_id)
        child_id = "recovery-session"
        child_config = json.loads(parent_session["model_config"] or "{}")
        child_config["_branched_from"] = parent_id
        db.create_session(
            child_id,
            source=parent_session["source"],
            model=parent_session["model"],
            system_prompt=parent_session["system_prompt"],
            parent_session_id=parent_id,
            model_config=child_config,
        )
        copied_rows = copy.deepcopy(original_rows)
        # get_messages decodes content/tool_calls, but these provider sidecars
        # remain stored JSON strings. The batch writer expects their objects.
        for row in copied_rows:
            for field in (
                "reasoning_details",
                "codex_reasoning_items",
                "codex_message_items",
                "codex_compaction_items",
            ):
                if isinstance(row.get(field), str):
                    row[field] = json.loads(row[field])
        assert db.append_messages_batch(child_id, copied_rows, chunk_rows=500) == len(
            original_rows
        )
        child = db.get_session(child_id)
        assert child["parent_session_id"] == parent_id
        assert json.loads(child["model_config"])["_branched_from"] == parent_id

        child_rows = db.get_messages(child_id)
        assert len(child_rows) == len(original_rows)
        fields = set(original_rows[0]) - {"id", "session_id"}
        for index, (child_row, parent_row) in enumerate(zip(child_rows, original_rows)):
            for key in fields:
                assert child_row.get(key) == parent_row.get(key), (index, key)
        assert db.get_messages(parent_id) == original_rows
        assert parent._dag.get_session_nodes(parent_id) == parent_nodes
        assert sessions.get_or_create_session(source).session_id == parent_id
        assert sessions.switch_session(session_key, child_id).session_id == child_id

        recovery = new_engine(child_id)
        assert recovery._conversation_id == session_key
        assert recovery._last_compacted_store_id == 0
        child_history = sessions.load_transcript(child_id)
        compressed = recovery.compress(copy.deepcopy(child_history))

        assert len(compressed) < len(child_history)
        assert len(recovery._dag.get_session_nodes(child_id)) == 1
        assert recovery._dag.get_session_nodes(parent_id) == parent_nodes
        assert recovery._store.get_session_messages(parent_id) == parent_raw
        assert db.get_messages(parent_id) == original_rows
        assert db.get_messages(parent_id, include_inactive=True) == original_all_rows
        assert sessions.get_or_create_session(source).session_id == child_id
    finally:
        for engine in reversed(engines):
            engine.shutdown()
        db.close()
