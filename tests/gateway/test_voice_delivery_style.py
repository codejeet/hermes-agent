"""The director carries the configured spoken style into both fast stages."""
from unittest.mock import AsyncMock
import pytest
from gateway.voice_director import VoiceDirector

@pytest.mark.asyncio
async def test_configured_style_reaches_route_prompt():
    d = VoiceDirector({'spoken_summary_style':'Succinct and casual; natural inflection, no theatrics.'})
    d._call = AsyncMock(return_value='{"response_mode":"direct","ack_text":"Hey!","is_job":false,"job_prompt":""}')
    await d.route('Good morning!', [])
    prompt = d._call.call_args.args[0][0]['content']
    assert 'Succinct and casual; natural inflection, no theatrics.' in prompt

@pytest.mark.asyncio
async def test_configured_style_reaches_summary_prompt():
    d = VoiceDirector({'spoken_summary_style':'Use at most one subtle audio tag, such as [curious].'})
    d._call = AsyncMock(return_value='{"spoken_text":"[curious] Want the details?"}')
    result = await d.summarize('The requested check passed.')
    prompt = d._call.call_args.args[0][0]['content']
    assert 'Use at most one subtle audio tag, such as [curious].' in prompt
    assert result == '[curious] Want the details?'
