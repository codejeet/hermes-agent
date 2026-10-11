"""Opt-in fast voice dispatch: strict data, no tools, bounded latency."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.voice_director import VoiceDirector, parse_route, public_context


def payload(mode="direct", text="Yep, I can hear you.", job=""):
    return json.dumps(dict(response_mode=mode, ack_text=text,
                           is_job=mode in {"work", "queue"}, job_prompt=job))


def response(text=None, reasoning=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=text, reasoning_content=reasoning))])


@pytest.mark.parametrize("text", ["not json", "```json\n{}\n```", "{}", "[]",
    payload().replace('false', '"false"'), payload("unknown"),
    payload("ignore", "Speak anyway"), payload("direct", job="delete files"),
    payload().replace('"direct"', '"direct", "unexpected": 1'),
    payload().replace('"direct"', '"direct", "response_mode": "ignore"')])
def test_strict_invalid_router_outputs_fail_open(text):
    assert parse_route(text) is None


def test_routes_are_typed_and_explicit():
    assert parse_route(payload()).response_mode == "direct"
    assert parse_route(payload("work", "Checking.", "Look it up")).is_job
    assert parse_route(payload("queue", "After this job.")).is_job
    assert parse_route(payload("ignore", "")).ack_text == ""


def test_context_bounded_and_never_extracts_hidden_reasoning_or_tools():
    history = [{"role": "user", "content": "x" * 6000}] * 100
    history += [{"role": "assistant", "content": "public", "reasoning": "secret"},
                {"role": "tool", "content": "credential"}]
    result = public_context(history)
    assert len(result) <= 8
    assert sum(len(m["content"]) for m in result) <= 6000
    assert "secret" not in str(result) and "credential" not in str(result)
    assert result[-1]["content"] == "public"


@pytest.mark.asyncio
async def test_fast_call_priority_low_reasoning_and_no_tools(monkeypatch):
    call = MagicMock(return_value=response(payload()))
    monkeypatch.setattr("agent.auxiliary_client.call_llm", call)
    director = VoiceDirector({"model": "fast-test", "provider": "openai-codex"})
    route = await director.route("hello", [], busy=False)
    assert route.response_mode == "direct"
    kw = call.call_args.kwargs
    assert kw["model"] == "fast-test"
    assert kw["extra_body"]["service_tier"] == "priority"
    assert kw["reasoning_config"]["effort"] in {"low", "minimal"}
    assert not kw.get("tools")


@pytest.mark.asyncio
async def test_hidden_reasoning_never_becomes_router_or_speech(monkeypatch):
    monkeypatch.setattr("agent.auxiliary_client.call_llm", lambda **kw: response(None, payload()))
    director = VoiceDirector({"model": "fast-test"})
    assert await director.route("hello", [], busy=False) is None
    assert await director.summarize("Full public answer") is None


@pytest.mark.asyncio
async def test_timeout_does_not_hold_up_heavy_path(monkeypatch):
    import time
    monkeypatch.setattr("agent.auxiliary_client.call_llm", lambda **kw: (time.sleep(.2), response(payload()))[1])
    director = VoiceDirector({"model": "fast-test", "ack_timeout_ms": 10})
    start = asyncio.get_running_loop().time()
    assert await director.route("hello", [], busy=False) is None
    assert asyncio.get_running_loop().time() - start < .15


@pytest.mark.asyncio
async def test_summary_separate_deadline_and_public_final_only(monkeypatch):
    call = MagicMock(return_value=response('{"spoken_text":"Done. Two tests passed."}'))
    monkeypatch.setattr("agent.auxiliary_client.call_llm", call)
    director = VoiceDirector({"model": "fast-test", "ack_timeout_ms": 10, "summary_timeout_ms": 2000})
    assert await director.summarize("## Result\nTwo tests passed.") == "Done. Two tests passed."
    assert call.call_args.kwargs["timeout"] == 2
    assert "## Result" in call.call_args.kwargs["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_mic_check_cannot_be_ignored(monkeypatch):
    monkeypatch.setattr("agent.auxiliary_client.call_llm", lambda **kw: response(payload("ignore", "")))
    route = await VoiceDirector({"model": "fast-test"}).route("testing, testing", [], busy=False)
    assert route.response_mode == "direct"
    assert route.ack_text
