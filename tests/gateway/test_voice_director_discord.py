"""Cancellation at Discord's audio boundary, after asynchronous decoding."""
import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from tests.gateway.test_discord_voice_mixer import _make_adapter, vm


@pytest.mark.asyncio
async def test_voice_off_during_decode_never_queues_audio(monkeypatch):
    a = _make_adapter()
    a._voice_clients[111] = MagicMock()
    a._voice_clients[111].is_connected.return_value = True
    a._playback_timeout_for_audio = AsyncMock(return_value=10)
    a._reset_voice_timeout = MagicMock()
    mx = MagicMock()
    a._voice_mixers[111] = mx
    live = [True]
    def decode(path):
        live[0] = False
        return bytes(vm.FRAME_SIZE)
    monkeypatch.setattr(vm, "decode_to_pcm", decode)
    assert not await a.play_in_voice_channel(111, "/tmp/test.mp3", is_current=lambda: live[0])
    mx.play_speech.assert_not_called()


@pytest.mark.asyncio
async def test_stop_during_mixer_playback_clears_speech(monkeypatch):
    a = _make_adapter()
    a._voice_clients[111] = MagicMock()
    a._voice_clients[111].is_connected.return_value = True
    a._playback_timeout_for_audio = AsyncMock(return_value=10)
    a._reset_voice_timeout = MagicMock()
    live = [True]
    class Mixer:
        play_speech = MagicMock()
        stop_speech = MagicMock()
        @property
        def speech_active(self):
            live[0] = False
            return True
    mx = Mixer()
    a._voice_mixers[111] = mx
    monkeypatch.setattr(vm, "decode_to_pcm", lambda p: bytes(vm.FRAME_SIZE))
    assert not await a.play_in_voice_channel(111, "/tmp/test.mp3", is_current=lambda: live[0])
    mx.stop_speech.assert_called_once()


@pytest.mark.asyncio
async def test_task_cancellation_stops_mixer_speech(monkeypatch):
    a = _make_adapter()
    a._voice_clients[111] = MagicMock()
    a._voice_clients[111].is_connected.return_value = True
    a._playback_timeout_for_audio = AsyncMock(return_value=10)
    a._reset_voice_timeout = MagicMock()
    mx = MagicMock(speech_active=True)
    started = asyncio.Event()
    mx.play_speech.side_effect = lambda *args, **kwargs: started.set()
    a._voice_mixers[111] = mx
    monkeypatch.setattr(vm, "decode_to_pcm", lambda p: bytes(vm.FRAME_SIZE))
    task = asyncio.create_task(a.play_in_voice_channel(111, "/tmp/test.mp3", is_current=lambda: True))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    mx.stop_speech.assert_called_once()
