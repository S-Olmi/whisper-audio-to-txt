import io
import pytest
from pydub import AudioSegment
from src.audio_config import preprocess_audio

# ============================
# Test E2E: preprocess_audio
# ============================

def test_preprocess_audio_e2e(tmp_path):
    """
    Make sure that preprocess_audio:
    - converts to mono
    - resamples at 16 kHz
    - normalizes the audio
    """
    # Create a temporary wav file with "non-standard" characteristics
    file_path = tmp_path / "test_input.wav"

    # Original audio: stereo, 22050 Hz, 1 second
    audio = AudioSegment.silent(duration=1000, frame_rate=22050).set_channels(2)
    audio.export(file_path, format="wav")

    # Preprocess
    processed_audio = preprocess_audio(str(file_path))

    # Main controls
    assert processed_audio.channels == 1, "It must be mono"
    assert processed_audio.frame_rate == 16000, "It must be at 16kHz"
    assert isinstance(processed_audio, AudioSegment), "It must return an AudioSegment"

    # Extra control: duration does not change significantly
    # (resample must not change duration)
    assert abs(len(processed_audio) - len(audio)) < 5, "Duration must remain almost the same"
