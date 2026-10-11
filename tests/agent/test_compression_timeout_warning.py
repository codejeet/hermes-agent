"""Timeout notices distinguish stalled progress from the overall deadline."""

from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize(
    ("waited", "since_progress", "expected", "unexpected"),
    [
        (60.1, 0.1, "60.1s total", "30.0s with no output"),
        (35.0, 30.0, "30.0s without progress", "total time limit"),
    ],
)
def test_timeout_notice_uses_actual_reason_and_elapsed_time(
    monkeypatch, waited, since_progress, expected, unexpected
):
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.session_id = "timeout-warning-test"
    agent._cached_system_prompt = "system"
    agent._emit_warning = MagicMock()
    agent._touch_activity = MagicMock()
    agent._build_system_prompt = MagicMock(return_value="system")
    agent._conversation_root_id = MagicMock(return_value=None)
    agent.context_compressor = MagicMock()

    def timed_out(**kwargs):
        kwargs["on_timeout"](30.0, waited, since_progress)
        return kwargs["messages"], "system"

    monkeypatch.setattr(
        "agent.conversation_compression.run_compress_context_with_progress_timeout",
        timed_out,
    )
    monkeypatch.setattr(
        "agent.conversation_compression.resolve_context_compression_timeouts",
        lambda compression_cfg=None: (30.0, 60.0),
    )
    monkeypatch.setattr("agent.portal_tags.get_conversation_context", lambda: object())
    messages = [{"role": "user", "content": "preserve this history"}]

    result, _ = AIAgent._compress_context(agent, messages, "system")

    assert result is messages
    agent._emit_warning.assert_called_once()
    warning = agent._emit_warning.call_args.args[0]
    assert expected in warning
    assert unexpected not in warning
    assert "No messages were dropped" in warning
    if waited >= 60:
        assert "total time limit" in warning
        assert "last progress 0.1s ago" in warning
