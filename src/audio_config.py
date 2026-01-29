from pydub import AudioSegment, effects

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


def preprocess_audio(file_path: str) -> AudioSegment:
    """
    Preprocessing an audio file:
    - Convert to mono (if necessary)
    - Resample to 16 kHz (if necessary)
    - Normalize volume
    Returns an AudioSegment ready for transcription.
    """
    audio = AudioSegment.from_file(file_path)

    audio = _ensure_channels(audio, TARGET_CHANNELS)
    audio = _ensure_samplerate(audio, TARGET_SAMPLE_RATE)
    audio = _normalize(audio)

    return audio


# ============================
# UTILITY FUNCTIONS
# ============================

def _ensure_channels(audio: AudioSegment, channels: int) -> AudioSegment:
    """Converts audio to the required number of channels only when necessary."""
    return audio if audio.channels == channels else audio.set_channels(channels)


def _ensure_samplerate(audio: AudioSegment, sample_rate: int) -> AudioSegment:
    """Resample audio at the requested rate only if different."""
    return audio if audio.frame_rate == sample_rate else audio.set_frame_rate(sample_rate)


def _normalize(audio: AudioSegment) -> AudioSegment:
    """Normalize audio with pydub, a feature isolated for future extensions."""
    return effects.normalize(audio)
