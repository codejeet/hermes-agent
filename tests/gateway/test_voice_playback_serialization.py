import asyncio
from unittest.mock import AsyncMock
import pytest
from tests.gateway.test_discord_voice_mixer import _make_adapter

@pytest.mark.asyncio
async def test_fast_reply_and_job_final_do_not_speak_over_each_other():
    a = _make_adapter()
    active = 0
    overlap = False
    calls = []
    async def play(guild_id, audio_path, **kwargs):
        nonlocal active, overlap
        active += 1
        overlap |= active > 1
        calls.append(audio_path)
        await asyncio.sleep(.01)
        active -= 1
        return True
    a._play_in_voice_channel_unlocked = play
    result = await asyncio.gather(a.play_in_voice_channel(111, 'quick.mp3'), a.play_in_voice_channel(111, 'final.mp3'))
    assert result == [True, True]
    assert calls == ['quick.mp3', 'final.mp3']
    assert not overlap

@pytest.mark.asyncio
async def test_cancelled_waiting_clip_is_not_started():
    a = _make_adapter()
    started, release = asyncio.Event(), asyncio.Event()
    live = [True]
    async def play(guild_id, path, **kwargs):
        started.set()
        await release.wait()
        return True
    a._play_in_voice_channel_unlocked = AsyncMock(side_effect=play)
    first = asyncio.create_task(a.play_in_voice_channel(111, 'first.mp3'))
    try:
        await asyncio.wait_for(started.wait(), .2)
        second = asyncio.create_task(a.play_in_voice_channel(111, 'stale.mp3', is_current=lambda: live[0]))
        await asyncio.sleep(0)
        live[0] = False
        release.set()
        assert await first
        assert not await second
        assert a._play_in_voice_channel_unlocked.await_count == 1
    finally:
        release.set()
        await first
