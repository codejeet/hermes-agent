"""Voice fast-ack/router tests for gateway voice turns."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _ensure_discord_mock():
    """Install a lightweight discord mock when discord.py isn't available."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    discord_mod.opus = SimpleNamespace(is_loaded=lambda: True, load_opus=lambda *_args, **_kwargs: None)
    discord_mod.FFmpegPCMAudio = MagicMock
    discord_mod.PCMVolumeTransformer = MagicMock
    discord_mod.http = SimpleNamespace(Route=MagicMock)

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult, SessionSource


def _make_event(text: str = "hello", message_type=MessageType.VOICE, chat_id="123") -> MessageEvent:
    source = SessionSource(
        chat_id=chat_id,
        user_id="user1",
        platform=Platform.TELEGRAM,
    )
    source.thread_id = None
    event = MessageEvent(text=text, message_type=message_type, source=source)
    event.message_id = "msg42"
    return event


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_db = None
    runner.session_store = MagicMock()
    return runner


class FakeAutoTtsAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, extra={}), Platform.TELEGRAM)
        self._message_handler = AsyncMock()
        self.sent_text = None
        self.played_tts_path = None

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, *args, **kwargs):
        return SendResult(success=True, message_id="sent1")

    async def get_chat_info(self, chat_id: str):
        return {}

    async def _keep_typing(self, chat_id: str, metadata=None, stop_event=None):
        if stop_event is not None:
            await stop_event.wait()

    async def _run_processing_hook(self, *args, **kwargs):
        return None

    async def _send_with_retry(self, **kwargs):
        self.sent_text = kwargs.get("content")
        return SendResult(success=True, message_id="sent1")

    async def play_tts(self, chat_id: str, audio_path: str, metadata=None):
        self.played_tts_path = audio_path
        return SendResult(success=True)


class TestVoiceFastAckRouter:
    def test_parse_voice_fast_ack_router_json_strips_fences_and_coerces_fields(self):
        from gateway.run import GatewayRunner

        raw = """```json
        {"ack_text": 123, "is_job": "yes", "job_prompt": 456, "response_mode": null}
        ```"""

        parsed = GatewayRunner._parse_voice_fast_ack_router_response(raw)

        assert parsed == {
            "ack_text": "123",
            "is_job": True,
            "job_prompt": "456",
            "response_mode": "",
        }

    def test_parse_voice_fast_ack_router_rejects_prefixed_non_json(self):
        from gateway.run import GatewayRunner

        raw = 'Here is the JSON: {"ack_text":"ok","is_job":true,"job_prompt":"","response_mode":"normal"}'

        assert GatewayRunner._parse_voice_fast_ack_router_response(raw) is None

    @pytest.mark.asyncio
    async def test_voice_fast_ack_disabled_skips_router_and_tts(self):
        runner = _make_runner()
        event = _make_event()
        runner._send_voice_reply = AsyncMock(return_value=True)

        with patch("gateway.run._load_gateway_config", return_value={}), \
             patch("gateway.run.call_llm") as call_llm:
            prepared, route = await runner._maybe_prepare_voice_fast_ack(event, "hello", event.source)

        assert prepared == "hello"
        assert route is None
        call_llm.assert_not_called()
        runner._send_voice_reply.assert_not_called()

    @pytest.mark.asyncio
    async def test_voice_fast_ack_uses_configured_aux_model_and_speaks_ack(self):
        runner = _make_runner()
        event = _make_event()
        runner._send_voice_reply = AsyncMock(return_value=True)
        cfg = {
            "voice": {
                "fast_ack": {
                    "enabled": True,
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "secret-key",
                    "base_url": "https://api.example.test/v1",
                    "ack_timeout_ms": 700,
                    "max_tokens": 96,
                }
            }
        }
        router_json = json.dumps({
            "ack_text": "Got it — checking now.",
            "is_job": True,
            "job_prompt": "Check the thing",
            "response_mode": "normal",
        })

        with patch("gateway.run._load_gateway_config", return_value=cfg), \
             patch("gateway.run.call_llm", return_value=object()) as call_llm, \
             patch("gateway.run.extract_content_or_reasoning", return_value=router_json):
            prepared, route = await runner._maybe_prepare_voice_fast_ack(event, "hello", event.source)

        call_llm.assert_called_once()
        kwargs = call_llm.call_args.kwargs
        assert kwargs["provider"] == "openai"
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["base_url"] == "https://api.example.test/v1"
        assert kwargs["api_key"] == "secret-key"
        assert kwargs["max_tokens"] == 96
        assert kwargs["timeout"] == pytest.approx(0.7)
        prompt_text = "\n".join(m["content"] for m in kwargs["messages"])
        assert "strict JSON" in prompt_text
        assert "ack_text" in prompt_text
        runner._send_voice_reply.assert_awaited_once_with(event, "Got it — checking now.")
        assert route["job_prompt"] == "Check the thing"
        assert "Original voice transcript:" in prepared
        assert "hello" in prepared
        assert "Suggested prompt: Check the thing" in prepared

    @pytest.mark.asyncio
    async def test_voice_fast_ack_invalid_json_fails_open_to_original_message(self):
        runner = _make_runner()
        event = _make_event()
        runner._send_voice_reply = AsyncMock(return_value=True)
        cfg = {"voice": {"fast_ack": {"enabled": True, "provider": "openai", "model": "fast"}}}

        with patch("gateway.run._load_gateway_config", return_value=cfg), \
             patch("gateway.run.call_llm", return_value=object()), \
             patch("gateway.run.extract_content_or_reasoning", return_value="not json"):
            prepared, route = await runner._maybe_prepare_voice_fast_ack(event, "hello", event.source)

        assert prepared == "hello"
        assert route is None
        runner._send_voice_reply.assert_not_called()

    @pytest.mark.asyncio
    async def test_voice_fast_ack_tts_failure_falls_back_to_text_ack(self):
        runner = _make_runner()
        event = _make_event()
        adapter = SimpleNamespace(send=AsyncMock(return_value=SendResult(success=True)))
        runner.adapters = {Platform.TELEGRAM: adapter}
        runner._send_voice_reply = AsyncMock(return_value=False)
        cfg = {"voice": {"fast_ack": {"enabled": True, "provider": "openai", "model": "fast"}}}
        router_json = json.dumps({
            "ack_text": "Heard you — working on it.",
            "is_job": True,
            "job_prompt": "",
            "response_mode": "normal",
        })

        with patch("gateway.run._load_gateway_config", return_value=cfg), \
             patch("gateway.run.call_llm", return_value=object()), \
             patch("gateway.run.extract_content_or_reasoning", return_value=router_json):
            prepared, route = await runner._maybe_prepare_voice_fast_ack(event, "hello", event.source)

        assert prepared == "hello"
        assert route["ack_text"] == "Heard you — working on it."
        adapter.send.assert_awaited_once_with("123", "Heard you — working on it.", metadata=None)

    @pytest.mark.asyncio
    async def test_voice_fast_ack_respects_explicit_voice_off(self):
        runner = _make_runner()
        event = _make_event()
        runner._voice_mode = {"telegram:123": "off"}
        runner._send_voice_reply = AsyncMock(return_value=True)
        cfg = {"voice": {"fast_ack": {"enabled": True, "provider": "openai", "model": "fast"}}}

        with patch("gateway.run._load_gateway_config", return_value=cfg), \
             patch("gateway.run.call_llm") as call_llm:
            prepared, route = await runner._maybe_prepare_voice_fast_ack(event, "hello", event.source)

        assert prepared == "hello"
        assert route is None
        call_llm.assert_not_called()
        runner._send_voice_reply.assert_not_called()

    @pytest.mark.asyncio
    async def test_voice_fast_ack_stt_failure_markers_skip_router(self):
        runner = _make_runner()
        event = _make_event()
        runner._send_voice_reply = AsyncMock(return_value=True)
        cfg = {"voice": {"fast_ack": {"enabled": True, "provider": "openai", "model": "fast"}}}

        with patch("gateway.run._load_gateway_config", return_value=cfg), \
             patch("gateway.run.call_llm") as call_llm:
            prepared, route = await runner._maybe_prepare_voice_fast_ack(
                event,
                "🎤 I received your voice message but can't transcribe it — no STT provider is configured.",
                event.source,
            )

        assert "no STT provider" in prepared
        assert route is None
        call_llm.assert_not_called()
        runner._send_voice_reply.assert_not_called()

    def test_build_voice_fast_speech_messages_filters_tool_transcript_for_spoken_progress(self):
        runner = _make_runner()
        cfg = {
            "filtered_thought_fast_speech": {
                "emotion": "curious, focused",
                "max_words": 18,
            }
        }
        events = [
            {"event_type": "tool.started", "tool_name": "read_file/sk-secret-/opt/app/config.yaml", "preview": "/workspace/secret_app.py", "args": {"path": "/workspace/secret_app.py", "api_key": "sk-secret"}},
            {"event_type": "reasoning.available", "tool_name": "_thinking", "preview": "I need to inspect the files line by line and reveal my chain of thought."},
        ]

        messages = runner._build_voice_fast_speech_messages(events, cfg)
        prompt_text = "\n".join(m["content"] for m in messages)

        assert "filtered thought fast speech" in prompt_text.lower()
        assert "Do not reveal private chain-of-thought" in prompt_text
        assert "curious, focused" in prompt_text
        assert "read_file" in prompt_text
        assert "working with local files" in prompt_text
        assert "/workspace/secret_app.py" not in prompt_text
        assert "/opt/app/config.yaml" not in prompt_text
        assert "sk-secret" not in prompt_text
        assert "inspect the files line by line" not in prompt_text

    def test_voice_fast_speech_sanitizer_blocks_secret_or_cot_output(self):
        from gateway.run import GatewayRunner

        assert GatewayRunner._sanitize_voice_fast_speech_text(
            "I found API key sk-1234567890abcdef in the file.",
            max_words=18,
        ) is None
        assert GatewayRunner._sanitize_voice_fast_speech_text(
            "My chain of thought is that we should inspect everything.",
            max_words=18,
        ) is None
        for path_text in (
            "I’m reading /opt/app/config.yaml now.",
            "I’m reading /Users/alice/project/app.py now.",
            "I’m reading C:\\Users\\alice\\project\\app.py now.",
            "I’m reading ./config.yaml now.",
            "I’m reading ~/project/app.py now.",
            "I’m reading src/app.py now.",
        ):
            assert GatewayRunner._sanitize_voice_fast_speech_text(path_text, max_words=18) is None
        assert GatewayRunner._sanitize_voice_fast_speech_text(
            "[curious] I’m checking the project files now.",
            max_words=7,
        ) == "[curious] I’m checking the project files now."

    @pytest.mark.asyncio
    async def test_voice_fast_speech_update_uses_spark_and_speaks_filtered_progress(self):
        runner = _make_runner()
        event = _make_event()
        runner._send_voice_reply = AsyncMock(return_value=True)
        cfg = {
            "voice": {
                "fast_ack": {
                    "enabled": True,
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "ack_timeout_ms": 700,
                    "filtered_thought_fast_speech": {
                        "enabled": True,
                        "max_tokens": 80,
                        "max_words": 16,
                        "emotion": "curious",
                    },
                }
            }
        }
        events = [
            {"event_type": "tool.started", "tool_name": "read_file", "preview": "/workspace/app.py", "args": {"path": "/workspace/app.py"}},
            {"event_type": "tool.completed", "tool_name": "read_file", "duration": 0.2, "is_error": False},
        ]
        speech_json = json.dumps({
            "should_speak": True,
            "speech_text": "[curious] I’m looking through the files to figure this out.",
        })

        with patch("gateway.run._load_gateway_config", return_value=cfg), \
             patch("gateway.run.call_llm", return_value=object()) as call_llm, \
             patch("gateway.run.extract_content_or_reasoning", return_value=speech_json):
            spoken = await runner._maybe_send_voice_fast_speech_update(event, events)

        assert spoken == "[curious] I’m looking through the files to figure this out."
        runner._send_voice_reply.assert_awaited_once_with(event, spoken)
        kwargs = call_llm.call_args.kwargs
        assert kwargs["provider"] == "openai"
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["max_tokens"] == 80
        assert kwargs["timeout"] == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_voice_fast_speech_update_respects_voice_off(self):
        runner = _make_runner()
        event = _make_event()
        runner._voice_mode = {"telegram:123": "off"}
        runner._send_voice_reply = AsyncMock(return_value=True)
        cfg = {
            "voice": {
                "fast_ack": {
                    "enabled": True,
                    "filtered_thought_fast_speech": {"enabled": True},
                }
            }
        }

        with patch("gateway.run._load_gateway_config", return_value=cfg), \
             patch("gateway.run.call_llm") as call_llm:
            spoken = await runner._maybe_send_voice_fast_speech_update(
                event,
                [{"event_type": "tool.started", "tool_name": "read_file", "preview": "/x"}],
            )

        assert spoken is None
        call_llm.assert_not_called()
        runner._send_voice_reply.assert_not_called()

    @pytest.mark.asyncio
    async def test_voice_fast_ack_final_summary_sets_spoken_text_for_base_auto_tts(self):
        runner = _make_runner()
        event = _make_event()
        long_response = "# Result\n\n" + ("Detailed markdown content. " * 80)
        cfg = {
            "voice": {
                "fast_ack": {
                    "enabled": True,
                    "summarize_final": True,
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "max_spoken_summary_seconds": 8,
                }
            }
        }

        with patch("gateway.run._load_gateway_config", return_value=cfg), \
             patch("gateway.run.call_llm", return_value=object()) as call_llm, \
             patch("gateway.run.extract_content_or_reasoning", return_value="Here’s the short spoken version."):
            returned = await runner._maybe_prepare_voice_final_summary(event, long_response)

        assert returned == long_response
        assert event._hermes_voice_spoken_text == "Here’s the short spoken version."
        call_llm.assert_called_once()
        prompt_text = "\n".join(m["content"] for m in call_llm.call_args.kwargs["messages"])
        assert "natural spoken" in prompt_text.lower()

    def test_voice_reply_tts_text_prefers_spoken_summary(self):
        runner = _make_runner()
        event = _make_event()
        event._hermes_voice_spoken_text = "Short spoken summary."

        assert runner._voice_reply_tts_text(event, "# Long detailed markdown") == "Short spoken summary."

    @pytest.mark.asyncio
    async def test_base_auto_tts_uses_fast_ack_spoken_text_when_present(self, tmp_path):
        adapter = FakeAutoTtsAdapter()
        adapter._auto_tts_enabled_chats.add("123")
        long_response = "# Full markdown response\n\n" + ("Lots of detailed text. " * 20)
        adapter._message_handler.return_value = long_response
        event = _make_event()
        event._hermes_voice_spoken_text = "Short spoken summary."
        audio_path = tmp_path / "summary.ogg"
        audio_path.write_bytes(b"fake")

        with patch("tools.tts_tool.check_tts_requirements", return_value=True), \
             patch("tools.tts_tool.text_to_speech_tool", return_value=json.dumps({"success": True, "file_path": str(audio_path)})) as tts, \
             patch.object(Path, "exists", return_value=True), \
             patch("os.remove"):
            await adapter._process_message_background(event, "telegram:123")

        assert tts.call_args.kwargs["text"] == "Short spoken summary."
        assert adapter.played_tts_path == str(audio_path)
        assert adapter.sent_text == long_response.strip()
