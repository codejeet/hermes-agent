"""Gateway voice director lifecycle regressions; no live providers/config/audio."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.voice_director import VoiceRoute


@pytest.fixture
def voice_config(monkeypatch):
    config = {"voice": {"fast_ack": {"enabled": True, "model": "fast-test"},
                        "job_model": {"model": "heavy-test", "provider": "openai-codex", "reasoning_effort": "high"}}}
    monkeypatch.setattr(run, "_load_gateway_config", lambda: config)
    return config


def event(text="hello", kind=MessageType.VOICE):
    return MessageEvent(text=text, message_type=kind, source=SessionSource(
        platform=Platform.DISCORD, chat_id="123", user_id="u1"))


def runner():
    r = object.__new__(run.GatewayRunner)
    r._voice_mode = {}
    r._send_voice_reply = AsyncMock()
    r._is_session_run_current = lambda *a: True
    a = MagicMock()
    a._pending_messages = {}
    a._send_with_retry = AsyncMock()
    a._should_auto_tts_for_chat.return_value = True
    r.adapters = {Platform.DISCORD: a}
    r.config = MagicMock(group_sessions_per_user=True, thread_sessions_per_user=False)
    return r, a


def test_opt_in_and_voice_off_and_text_scope(voice_config):
    r, _ = runner()
    assert r._voice_director_settings(event().source, MessageType.VOICE)
    assert r._voice_director_settings(event().source, MessageType.TEXT) is None
    r._voice_mode["discord:123"] = "off"
    assert r._voice_director_settings(event().source, MessageType.VOICE) is None
    r._voice_mode.clear()
    voice_config["voice"]["fast_ack"]["enabled"] = False
    assert r._voice_director_settings(event().source, MessageType.VOICE) is None


@pytest.mark.asyncio
async def test_route_original_transcript_authoritative_and_advisory_hidden(voice_config, monkeypatch):
    r, _ = runner()
    e = event("Please check the tests")
    route = VoiceRoute("work", "Checking now.", True, "Run focused tests")
    monkeypatch.setattr("gateway.voice_director.VoiceDirector.route", AsyncMock(return_value=route))
    result = await r._voice_director_route(e, e.text, [], "sk", 1)
    assert result == route
    assert e.text == "Please check the tests"
    assert e._hermes_voice_router_job_prompt == "Run focused tests"
    r._send_voice_reply.assert_awaited_once_with(e, "Checking now.")


@pytest.mark.asyncio
async def test_busy_work_queues_without_interrupt_or_session_write(voice_config, monkeypatch):
    r, a = runner()
    r._is_user_authorized = lambda *_: True
    r._draining = False
    r._running_agents = {"sk": MagicMock()}
    agent = r._running_agents["sk"]
    r.session_store = None
    r._voice_director_history = AsyncMock(return_value=[])
    route = VoiceRoute("queue", "I'll do that next.", True, "after current")
    monkeypatch.setattr("gateway.voice_director.VoiceDirector.route", AsyncMock(return_value=route))
    e = event("after that, check the docs")
    assert await r._voice_director_busy(e, "sk") is True
    assert a._pending_messages["sk"] is e
    assert e._hermes_voice_director_deferred
    agent.interrupt.assert_not_called()
    assert e.text == "after that, check the docs"


@pytest.mark.asyncio
async def test_busy_ignore_not_queued_or_spoken(voice_config, monkeypatch):
    r, a = runner()
    r._voice_director_history = AsyncMock(return_value=[])
    monkeypatch.setattr("gateway.voice_director.VoiceDirector.route", AsyncMock(return_value=VoiceRoute("ignore", "", False, "")))
    assert await r._voice_director_busy(event("background tv"), "sk")
    assert not a._pending_messages
    r._send_voice_reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_busy_direct_answers_now_but_persistence_serialized(voice_config, monkeypatch):
    r, a = runner()
    r._voice_director_history = AsyncMock(return_value=[])
    monkeypatch.setattr("gateway.voice_director.VoiceDirector.route", AsyncMock(return_value=VoiceRoute("direct", "You're welcome!", False, "")))
    e = event("thanks")
    assert await r._voice_director_busy(e, "sk")
    r._send_voice_reply.assert_awaited_once_with(e, "You're welcome!")
    a._send_with_retry.assert_awaited_once()
    assert a._pending_messages["sk"] is e
    assert e._hermes_voice_direct_delivered


@pytest.mark.asyncio
async def test_busy_router_failure_still_queues_not_interrupt(voice_config, monkeypatch):
    r, a = runner()
    r._voice_director_history = AsyncMock(return_value=[])
    monkeypatch.setattr("gateway.voice_director.VoiceDirector.route", AsyncMock(return_value=None))
    e = event("check this next")
    assert await r._voice_director_busy(e, "sk")
    assert a._pending_messages["sk"] is e


@pytest.mark.asyncio
async def test_cancelled_router_never_acknowledges(voice_config, monkeypatch):
    r, _ = runner()
    r._is_session_run_current = lambda *a: False
    monkeypatch.setattr("gateway.voice_director.VoiceDirector.route", AsyncMock(return_value=VoiceRoute("work", "Checking", True, "")))
    e = event()
    await r._voice_director_route(e, e.text, [], "sk", 1)
    r._send_voice_reply.assert_not_awaited()


def test_job_runtime_scoped_priority_and_configured_reasoning(voice_config, monkeypatch):
    r, _ = runner()
    monkeypatch.setattr(run, "_resolve_runtime_agent_kwargs_for_provider", lambda p: {"provider": p, "api_key": "fake", "model": "provider-default"})
    original = {"provider": "other", "api_key": "other"}
    config = r._voice_director_settings(event().source, MessageType.VOICE)
    model, runtime, reasoning = r._voice_job_runtime(config, "normal", original, {"effort": "low"})
    assert model == "heavy-test"
    assert runtime["provider"] == "openai-codex"
    assert reasoning["effort"] == "high"
    assert original["provider"] == "other"
    assert r._voice_job_runtime(None, "normal", original, None) == ("normal", original, None)


@pytest.mark.asyncio
async def test_summary_preserves_full_response_and_owns_tts(voice_config, monkeypatch):
    r, _ = runner()
    e = event()
    e._hermes_voice_director = True
    monkeypatch.setattr("gateway.voice_director.VoiceDirector.summarize", AsyncMock(return_value="Two tests passed."))
    full = "## Tests\nTwo tests passed. Detailed table."
    await r._voice_director_final(e, full, [], "sk", 1)
    assert e._hermes_voice_spoken_text == "Two tests passed."
    assert e._hermes_voice_tts_handled
    r._send_voice_reply.assert_awaited_once_with(e, "Two tests passed.")


@pytest.mark.asyncio
async def test_voice_off_during_summary_prevents_late_speech(voice_config, monkeypatch):
    r, _ = runner()
    e = event()
    e._hermes_voice_director = True
    async def summary(*a):
        r._voice_mode["discord:123"] = "off"
        return "Done"
    monkeypatch.setattr("gateway.voice_director.VoiceDirector.summarize", summary)
    await r._voice_director_final(e, "Full text", [], "sk", 1)
    r._send_voice_reply.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["direct", "ignore", "work"])
async def test_actual_gateway_handler_routes_and_persists_without_duplicate_heavy_call(voice_config, monkeypatch, tmp_path, mode):
    from tests.gateway.test_42039_duplicate_user_message import _bootstrap, _event, _source
    r = _bootstrap(monkeypatch, tmp_path)
    e = _event()
    e.message_type = MessageType.VOICE
    route = VoiceRoute(mode, "Hello!" if mode == "direct" else "", mode == "work", "Check carefully" if mode == "work" else "")
    monkeypatch.setattr("gateway.voice_director.VoiceDirector.route", AsyncMock(return_value=route))
    r._send_voice_reply = AsyncMock()
    r._run_agent = AsyncMock(return_value={"final_response": "Finished work", "messages": [
        {"role": "user", "content": e.text}, {"role": "assistant", "content": "Finished work"}],
        "history_offset": 0, "agent_persisted": False})
    result = await r._handle_message_with_agent(e, _source(), "agent:main:telegram:group:-1001:12345", 1)
    if mode == "work":
        r._run_agent.assert_awaited_once()
        assert r._run_agent.await_args.kwargs["message"] == e.text
        assert "Check carefully" not in r._run_agent.await_args.kwargs["message"]
    else:
        r._run_agent.assert_not_awaited()
    rows = [c.args[1] for c in r.session_store.append_to_transcript.call_args_list if len(c.args) > 1]
    if mode == "direct":
        assert "Hello!" in result
        assert [x["content"] for x in rows if x["role"] in {"user", "assistant"}] == [e.text, "Hello!"]
        assert all(not c.kwargs.get("skip_db", False) for c in r.session_store.append_to_transcript.call_args_list)
    elif mode == "ignore":
        assert result is None
        assert not rows


@pytest.mark.asyncio
async def test_actual_busy_handler_routes_before_interrupt_mode(voice_config, monkeypatch):
    from tests.gateway.test_busy_session_ack import _make_runner, _make_adapter
    r, sentinel = _make_runner()
    e = event("after that, check this")
    a = _make_adapter()
    r.adapters[Platform.DISCORD] = a
    r._running_agents["sk"] = sentinel
    r._voice_mode = {}
    r._voice_director_busy = AsyncMock(return_value=True)
    assert await r._handle_active_session_busy_message(e, "sk")
    r._voice_director_busy.assert_awaited_once_with(e, "sk")


@pytest.mark.asyncio
async def test_send_voice_checks_cancellation_after_synthesis(voice_config, monkeypatch, tmp_path):
    r, a = runner()
    # Exercise the real implementation, not the local helper's AsyncMock.
    e = event()
    live = [True]
    e._hermes_voice_live = lambda: live[0]
    p = tmp_path / "audio.mp3"
    def tts(**kw):
        p.write_bytes(b"audio")
        live[0] = False
        return json.dumps({"success": True, "file_path": str(p)})
    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", tts)
    a.send_voice = AsyncMock()
    await run.GatewayRunner._send_voice_reply(r, e, "Do not speak this late")
    a.send_voice.assert_not_awaited()
    assert not p.exists()


@pytest.mark.asyncio
async def test_failed_tts_does_not_log_successful_director_delivery(voice_config, monkeypatch, caplog):
    r, _ = runner()
    r._send_voice_reply = AsyncMock(return_value=False)
    monkeypatch.setattr("gateway.voice_director.VoiceDirector.route", AsyncMock(return_value=VoiceRoute("work", "Checking", True, "")))
    import logging
    with caplog.at_level(logging.INFO):
        e = event("check this")
        await r._voice_director_route(e, e.text, [], "sk", 1)
    assert "ack_sent" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [MessageType.VOICE, MessageType.TEXT])
async def test_real_agent_constructor_voice_only_priority_and_no_streaming_tts(voice_config, monkeypatch, tmp_path, kind):
    import sys
    from tests.gateway.test_streaming_tts_gateway_regression import _setup_monkeypatches, _make_runner, _make_voice_source, _NoopAgent
    _setup_monkeypatches(monkeypatch, tmp_path)
    monkeypatch.setattr(run, "_load_gateway_config", lambda: voice_config)
    captured = []
    class Agent(_NoopAgent):
        def __init__(self, **kw):
            captured.append(kw)
            super().__init__(**kw)
    monkeypatch.setattr(sys.modules["run_agent"], "AIAgent", Agent)
    monkeypatch.setattr(run, "_resolve_runtime_agent_kwargs_for_provider", lambda p: {"provider": p, "api_key": "fake", "api_mode": "codex_responses"})
    r = _make_runner()
    await r._run_agent(message="check tests", context_prompt="", history=[], source=_make_voice_source(),
                       session_id="s1", session_key="sk", message_type=kind)
    assert len(captured) == 1
    if kind == MessageType.VOICE:
        assert captured[0]["model"] == "heavy-test"
        assert captured[0]["request_overrides"]["service_tier"] == "priority"
        assert captured[0]["reasoning_config"]["effort"] == "high"
        assert captured[0]["enabled_toolsets"] == ["core"]
    else:
        assert captured[0]["model"] == "test-model"
        assert not (captured[0].get("request_overrides") or {}).get("service_tier")


@pytest.mark.asyncio
async def test_direct_health_check_does_not_wait_for_history_compression(voice_config, monkeypatch, tmp_path):
    from tests.gateway.test_42039_duplicate_user_message import _bootstrap, _event, _source
    r = _bootstrap(monkeypatch, tmp_path)
    e = _event()
    e.message_type = MessageType.VOICE
    e.text = "mic check"
    r.session_store.load_transcript.return_value = [{"role": "user", "content": "old"}] * 10
    metadata = AsyncMock(return_value=100000)
    monkeypatch.setattr("agent.model_metadata.get_model_context_length_async", metadata)
    r._run_agent = AsyncMock(return_value={"final_response": "normal", "messages": []})
    result = await r._handle_message_with_agent(e, _source(), "agent:main:telegram:group:-1001:12345", 1)
    assert "hear you" in result
    metadata.assert_not_awaited()


@pytest.mark.asyncio
async def test_adapter_never_double_synthesizes_director_response(monkeypatch):
    from tests.gateway.test_telegram_voice_v0_regressions import _PendingVoiceAdapter, _source
    adapter = _PendingVoiceAdapter()
    adapter._auto_tts_default = True
    e = MessageEvent(text="hi", source=_source(), message_type=MessageType.VOICE)
    e._hermes_voice_tts_handled = True
    adapter._message_handler = AsyncMock(return_value="Full detailed text remains here")
    tts = MagicMock()
    monkeypatch.setattr("tools.tts_tool.check_tts_requirements", lambda: True)
    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", tts)
    await adapter._process_message_background(e, "sk")
    tts.assert_not_called()
    assert any("Full detailed" in item[1] for item in adapter.sent)


@pytest.mark.asyncio
async def test_deferred_followup_not_consumed_by_inband_recursion(voice_config, monkeypatch, tmp_path):
    from tests.gateway.test_streaming_tts_gateway_regression import _setup_monkeypatches, _make_runner, _make_voice_source
    from tests.gateway.test_telegram_voice_v0_regressions import _PendingVoiceAdapter
    _setup_monkeypatches(monkeypatch, tmp_path)
    monkeypatch.setattr(run, "_load_gateway_config", lambda: voice_config)
    monkeypatch.setattr(run, "_resolve_runtime_agent_kwargs_for_provider", lambda p: {"provider": p, "api_key": "fake"})
    r = _make_runner()
    a = _PendingVoiceAdapter()
    r.adapters[Platform.TELEGRAM] = a
    pending = MessageEvent(text="after that, check docs", message_type=MessageType.VOICE, source=_make_voice_source())
    pending._hermes_voice_director_deferred = True
    a._pending_messages["sk"] = pending
    result = await r._run_agent(message="first job", context_prompt="", history=[], source=_make_voice_source(),
                              session_id="s1", session_key="sk", message_type=MessageType.VOICE)
    assert result["final_response"] == "Hello from the agent."
    assert a._pending_messages["sk"] is pending


@pytest.mark.asyncio
@pytest.mark.parametrize("voice_off", [False, True])
async def test_direct_spoken_while_busy_persists_once_on_drain(voice_config, monkeypatch, tmp_path, voice_off):
    from tests.gateway.test_42039_duplicate_user_message import _bootstrap, _event, _source
    r = _bootstrap(monkeypatch, tmp_path)
    e = _event()
    e.message_type = MessageType.VOICE
    e._hermes_voice_route = VoiceRoute("direct", "You're welcome", False, "")
    e._hermes_voice_direct_delivered = True
    if voice_off:
        r._voice_mode["telegram:-1001"] = "off"
    r._run_agent = AsyncMock()
    r._send_voice_reply = AsyncMock()
    assert await r._handle_message_with_agent(e, _source(), "agent:main:telegram:group:-1001:12345", 1) is None
    r._run_agent.assert_not_awaited()
    r._send_voice_reply.assert_not_awaited()
    rows = [c.args[1] for c in r.session_store.append_to_transcript.call_args_list]
    assert [x["content"] for x in rows if x["role"] in {"user", "assistant"}] == [e.text, "You're welcome"]


@pytest.mark.asyncio
async def test_busy_audio_attachments_are_distinct_fifo_turns(voice_config, monkeypatch):
    r, a = runner()
    r._voice_director_history = AsyncMock(return_value=[])
    r._prepare_busy_steer_text = AsyncMock(return_value="transcribed")
    monkeypatch.setattr("gateway.voice_director.VoiceDirector.route", AsyncMock(return_value=VoiceRoute("queue", "", True, "")))
    first, second = event("one"), event("two")
    first.media_urls, second.media_urls = ["/tmp/one.ogg"], ["/tmp/two.ogg"]
    await r._voice_director_busy(first, "sk")
    await r._voice_director_busy(second, "sk")
    assert a._pending_messages["sk"] is first
    assert first.media_urls == ["/tmp/one.ogg"]
    assert r._peek_session_state("sk").conversation.queued_events == [second]


@pytest.mark.asyncio
async def test_router_context_includes_already_spoken_direct_pending_persistence(voice_config, monkeypatch):
    r, a = runner()
    r.session_store = None
    r._lookup_session_id_under_store_lock = lambda *_: None
    prior = event("thanks")
    prior._hermes_voice_route = VoiceRoute("direct", "You're welcome", False, "")
    prior._hermes_voice_direct_delivered = True
    a._pending_messages["sk"] = prior
    history = await r._voice_director_history("sk")
    assert [row["content"] for row in history] == ["thanks", "You're welcome"]
