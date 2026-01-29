import os
import threading
from unittest.mock import MagicMock
import pytest
from pydub import AudioSegment

from src.transcribe import (
    _export_to_temp_wav,
    _transcribe_file,
    refine_text,
    transcribe_to_ram,
)


# ============================
# TEST _export_to_temp_wav
# ============================

def test_export_to_temp_wav_creates_file():
    audio = AudioSegment.silent(duration=1000)  # 1 second of silence
    path = _export_to_temp_wav(audio)

    # File must exist and end with .wav
    assert os.path.exists(path)
    assert path.endswith(".wav")

    # Cleaning
    os.remove(path)


# ============================
# TEST refine_text
# ============================

def test_refine_text_calls_punct(monkeypatch):
    class DummyPunct:
        def restore_punctuation(self, text):
            return text.upper()

    dummy = DummyPunct()

    import src.transcribe as tr
    monkeypatch.setattr(tr, "punct", dummy)

    # normal test
    assert refine_text("hello world") == "HELLO WORLD"
    # empty string test
    assert refine_text("   ") == "   "


# ============================
# TEST _transcribe_file CON MOCK
# ============================

def test_transcribe_file_mock():
    class DummySegment:
        def __init__(self):
            self.start = 0
            self.text = "ciao"

    class DummyModel:
        def transcribe(self, path, language="it", **kwargs):
            return [DummySegment()], None

    # Create a temporary audio file
    audio = AudioSegment.silent(duration=500)
    tmp_path = _export_to_temp_wav(audio)

    segments = _transcribe_file(DummyModel(), tmp_path)
    assert len(segments) == 1
    assert segments[0].text == "ciao"

    os.remove(tmp_path)


# ============================
# TEST transcribe_to_ram CON MOCK
# ============================

def test_transcribe_to_ram_mock(monkeypatch):
    audio = AudioSegment.silent(duration=1000)  # 1 second of silence

    # Dummy model
    class DummySegment:
        def __init__(self):
            self.start = 0
            self.text = "transcribed text"

    class DummyModel:
        def transcribe(self, path, language="it", **kwargs):
            return [DummySegment()], None

    monkeypatch.setattr("src.transcribe._transcribe_file", lambda model, path, language="it": [DummySegment()])
    monkeypatch.setattr("src.transcribe.preprocess_audio", lambda path: audio)

    result = transcribe_to_ram(DummyModel(), "fake_path.wav")
    assert "transcribed text" in result
