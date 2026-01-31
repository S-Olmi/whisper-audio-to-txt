from faster_whisper import WhisperModel
from pydub import AudioSegment
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import os
import tempfile
import logging

from src.audio_config import preprocess_audio
from src.dedup import deduplicate_chunks

import numpy as np

MAX_EXTEND_MS = 30_000

logger = logging.getLogger(__name__)

# ============================
# MODELS
# ============================

SUPPORTED_PUNCT_LANGS = ["it", "en", "fr", "de"]

def load_whisper_model(flag:bool, model_name, device):
    return WhisperModel(model_name, device=device, compute_type="int8", local_files_only=flag)


def refine_text(text: str, punct_model, language: str) -> str:
    """Apply punctuations if the language is supported."""
    if not text.strip():
        return text

    if language.lower() in SUPPORTED_PUNCT_LANGS and punct_model is not None:
        logger.info(f"Punctuation has been restored for the language: {language}")
        return punct_model.restore_punctuation(text)

    logger.warning(f"Punctuation restoration has been skipped: {language} is not supported for the NLP model.")
    return text


# ============================
# UTILS
# ============================

def _export_to_temp_wav(audio_segment):
    """Saves an AudioSegment to a temporary WAV file and returns the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        audio_segment.export(tmp.name, format="wav")
        return tmp.name
    finally:
        tmp.close()


def _transcribe_file(model, wav_path, language="it"):
    """Single Whisper call with standardized parameters."""
    segments_gen, _ = model.transcribe(
        wav_path,
        language=language,
        beam_size=1,
        vad_filter=False
    )
    return [seg for seg in segments_gen]


# ============================
# DYNAMIC TRANSCRIPTION
# ============================

def transcribe_dynamic(whisper_model, punct_model, audio_path, language,
                       threshold_minutes=30, chunk_duration_sec=300,
                       max_workers=4):

    audio = AudioSegment.from_file(audio_path)
    duration_min = audio.duration_seconds / 60
    logger.info(f"Received audio file: {audio_path} - Duration: {duration_min:.2f} min")

    if duration_min < threshold_minutes:
        logger.info("Use RAM mode for small files.")
        raw_text = transcribe_to_ram(whisper_model, audio_path, language)
    else:
        logger.info(f"Use STREAMING (chunking) mode for large files.")
        raw_text = transcribe_streaming(whisper_model, audio_path, language,
                                    chunk_duration_sec, max_workers)


    return refine_text(raw_text, punct_model, language)


# ============================
# SMALL FILES (RAM MODE)
# ============================

def transcribe_to_ram(whisper_model, audio_path, language):
    print("RAM mode (small file)")

    audio = preprocess_audio(audio_path)
    duration_sec = audio.duration_seconds

    tmp_path = _export_to_temp_wav(audio)

    results = []
    lock = threading.Lock()
    done_event = threading.Event()

    def worker():
        nonlocal results
        segments = _transcribe_file(whisper_model, tmp_path, language)
        with lock:
            results.extend(segments)
        done_event.set()

    thread = threading.Thread(target=worker)
    thread.start()

    pbar = tqdm(total=duration_sec, desc="Transcription", unit="s")

    estimated_rtf = 1.0  # real time: 1 min audio → 1 min transcript
    estimated_time = duration_sec * estimated_rtf

    start_time = time.time()

    while not done_event.is_set():
        elapsed = time.time() - start_time
        # Advance the bar based on the elapsed time
        estimated_progress = min(elapsed / estimated_time * duration_sec, duration_sec)

        pbar.n = estimated_progress
        pbar.refresh()

        time.sleep(0.1)

    thread.join()

    # final correction — 100%
    pbar.n = duration_sec
    pbar.refresh()
    pbar.close()

    try:
        os.remove(tmp_path)
    except:
        pass

    text = " ".join(seg.text for seg in sorted(results, key=lambda s: s.start))
    return text


# ============================
# LARGE FILES (STREAMING MODE)
# ============================

def transcribe_streaming(whisper_model, audio_path, language="it",
                         chunk_duration_sec=300, max_workers=4):

    print("STREAMING mode (large file) with chunking and parallelization")

    audio = preprocess_audio(audio_path)
    duration_ms = len(audio)
    chunk_ms = chunk_duration_sec * 1000

    # ----------------------------
    # SPLIT IN CHUNK
    # ----------------------------
    # chunks = [
    #     audio[start:start + chunk_ms]
    #     for start in range(0, duration_ms, chunk_ms)
    # ]

    chunks = split_audio_on_silence(audio, target_chunk_ms=chunk_ms)

    if len(chunks) <= 1:
        logger.warning("Silence split failed, fallback to fixed-size chunks")
        chunks = [
            audio[i:i + chunk_ms]
            for i in range(0, len(audio), chunk_ms)
        ]

    results = [None] * len(chunks)

    def worker(idx, audio_chunk):
        tmp_path = _export_to_temp_wav(audio_chunk)
        try:
            segments = _transcribe_file(whisper_model, tmp_path, language)
            text = " ".join(seg.text for seg in segments)
            return idx, text
        finally:
            try:
                os.remove(tmp_path)
            except:
                pass

    # ----------------------------
    # PARALLEL WHISPER
    # ----------------------------
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, i, ch): i for i, ch in enumerate(chunks)}

        with tqdm(
                total=len(futures),
                desc="Chunk transcribed",
                unit="chunk",
                smoothing=0.1
        ) as pbar:
            for future in as_completed(futures):
                idx, text = future.result()
                results[idx] = text
                pbar.update(1)

    # ----------------------------
    # DEDUPLICATES OVERLAP CHUNKS
    # ----------------------------
    return deduplicate_chunks(results)




# ============================
# SILENCE CHUNKING
# ============================


def split_audio_on_silence(audio, target_chunk_ms=5*60*1000, min_silence_ms=500):
    """
    Split audio in chunks around silence boundaries.
    - audio: numpy array o AudioSegment
    - target_chunk_ms: desired chunk (e.g. 5 minutes)
    - min_silence_ms: minimum duration of silence to consider it a boundary
    """
    chunks = []
    start = 0
    total_ms = len(audio)

    ref_db = rms_db(audio)

    while start < total_ms:
        end = min(start + target_chunk_ms, total_ms)

        # extend the chunk until the next silence
        while end < total_ms and end - start < target_chunk_ms + MAX_EXTEND_MS:
            window = audio[end:end + min_silence_ms]
            if len(window) == 0:
                break
            if is_silence(window, ref_db):
                break
            end += min_silence_ms // 2  # scan in half window
        if end <= start:
            end = min(start + target_chunk_ms, total_ms)
        chunk = audio[start:end]

        logger.info(
            f"Chunk {len(chunks)}: {(end - start) / 1000:.1f}s"
        )

        chunks.append(chunk)
        start = end
    return chunks





def is_silence(audio_window, ref_db, margin_db=15):
    """
    Returns True if the audio is below the dB threshold for the entire segment.
    """
    if len(audio_window) == 0:
        return True

    samples = np.array(audio_window.get_array_of_samples()).astype(np.float32)

    if len(samples) == 0:
        return True

    if audio_window.channels > 1:
        samples = samples.reshape((-1, audio_window.channels)).mean(axis=1)

    rms = np.sqrt(np.mean(samples ** 2))
    db = 20 * np.log10(rms + 1e-9)

    return db < (ref_db - margin_db)



def rms_db(audio):
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)
    rms = np.sqrt(np.mean(samples ** 2))
    return 20 * np.log10(rms + 1e-9)




