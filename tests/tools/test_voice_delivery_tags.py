"""Natural voice inflection tags must survive the Markdown/TTS cleanup."""
from tools.tts_text_normalize import prepare_spoken_text
from tools.tts_tool import _strip_markdown_for_tts


def test_subtle_audio_tag_survives_gateway_and_tool_cleanup():
    text = '[curious] **What are we working on?**'
    assert prepare_spoken_text(_strip_markdown_for_tts(text), max_chars=None) == '[curious] What are we working on?'


def test_eleven_v4_turbo_gets_selected_model_and_tagged_script(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from tools import tts_tool
    captured = {}
    def convert(**kwargs):
        captured.update(kwargs)
        return iter([b'audio-test-bytes'])
    monkeypatch.setattr(tts_tool, '_resolve_provider_key', lambda *args: 'test-key')
    monkeypatch.setattr(tts_tool, '_import_elevenlabs', lambda: lambda **kwargs: SimpleNamespace(text_to_speech=SimpleNamespace(convert=convert)))
    path = tmp_path / 'preview.mp3'
    tts_tool._generate_elevenlabs('[curious] What are we working on?', str(path), {'elevenlabs': {'voice_id':'test-voice','model_id':'eleven_v4_turbo'}})
    assert captured['model_id'] == 'eleven_v4_turbo'
    assert captured['text'] == '[curious] What are we working on?'
    assert path.read_bytes() == b'audio-test-bytes'
