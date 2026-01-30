# Audio Transcription Service (Faster-Whisper)

A local audio transcription service based on **Faster-Whisper** with **automatic punctuation restoration**, optimized for large audio files through a custom chunk-based processing pipeline. It is designed to run locally, with full offline support and intelligent large file handling.

The project focuses on robustness and production-oriented concerns such as memory management, chunk-based parallel inference, text deduplication, and punctuation restoration, delivering clean and structured output ready for downstream use.

---

## Why this project exists

This project was born from a very practical need: consuming audio messages in environments where listening is not possible (e.g. during work hours or meetings).

At the same time, it served as a hands-on exploration of modern speech-to-text pipelines, focusing on robustness, performance on long audio files, and post-processing quality rather than raw model experimentation.

---

## Key Features

- **Dynamic processing strategy**  
  Small audio files are processed entirely in RAM, while large files are automatically split and transcribed using parallel chunking to optimize memory usage and throughput.

- **Text deduplication**  
  N-gram–based post-processing removes repeated segments and Whisper hallucinations (repeated or overlapping phrases) introduced during chunk merging.

- **Punctuation restoration**  
  Integration of *DeepMultilingualPunctuation* applied to selected languages only 

  A local patch to ensure compatibility with `transformers >= 4.30`.

- **Audio preprocessing**  
  All input files are automatically:
  - resampled to 16 kHz  
  - converted to mono  
  - RMS-normalized  
  using **pydub**

- **Security (demonstrative)**
  Bearer Token authentication is implemented **for demonstration purposes only**, to showcase basic API security practices in a local-only environment.

---

## System Architecture

1. **Ingestion**  
   Audio upload and validation via FastAPI.

2. **Pre-processing**  
   Conversion to a normalized `.wav` format.

3. **Inference**  
   Parallel transcription of audio chunks using Faster-Whisper.

4. **Refinement**  
   Deduplication and punctuation restoration.

5. **Delivery**  
   Structured JSON output returned via REST endpoints.

---

## Output Format (Example)

The API returns a structured JSON response:

```json
{
  "filename": "STROIE5483928404.mp3",
  "language": "en",
  "status": "ok",
  "refined_text": "Close your eyes, exhale, feel your body, relax and let go of whatever you're carrying. Today, well, I'm letting go of the worry that I wouldn't get my new contacts in time for..."
}
```

- **filename**: original uploaded filename
- **language**: language code used for transcription (e.g. `it`, `fr`, `en`, `de`, ...)
- **status**: transcription status  
- **refined_text**: cleaned transcription (with punctuation restoration when supported)

---

## Input Example

The input consists of a **real audio file** (any supported format) and an optional `language` query parameter (str, default: `it`)  
There is no fixed duration limit; processing time scales linearly with file length, subject to available system resources.

All audio formats are supported as long as they can be decoded by FFMPEG, since every file is converted to a normalized `.wav` during preprocessing.

---

## Running the Service (Docker)

The service is fully containerized and includes all required system dependencies (including FFMPEG).

### 1. Environment Configuration

```bash
cp .env.example .env
```

Set an `API_TOKEN` inside the `.env` file.  
Even though the service runs locally, the token must be provided when calling the API (via Swagger).

---

### 2. Build and Run

```bash
docker build -t whisper-api .
docker run -d -p 8000:8000 --env-file .env --name whisper-service whisper-api
```

The Docker image already includes the required model weights and runs fully offline.

---

## Running the Service (Python Script)

### 1. Environment Configuration

```bash
cp .env.example .env
```

Set an `API_TOKEN` inside the `.env` file.  
Even though the service runs locally, the token must be provided when calling the API (via Swagger).

---

### 2. Requirements

In order to run the script, the following requirements are needed:

- Python 3.10+
- FFmpeg

To install FFmpeg, simply run in your terminal

```bash
sudo apt-get update && apt-get install -y ffmpeg 
```

To install Python, visit and follow instructions on ``https://www.python.org/downloads/``

---

### 3. Install dependencies and Run

```bash
pip install -r requirements.txt
```

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

On first launch, the application downloads the required Whisper and DeepPunctuation model weights.
Subsequent runs operate fully offline, using the cached weights.

---

## Using the API (Swagger UI)

The service is designed to be used primarily through **Swagger**, as it simplifies authentication, file uploads, and parameter inspection during development and testing.

After starting the container (or running `uvicorn` locally), open:

``http://localhost:8000/docs``


### How to use Swagger:

1. Click **Authorize**
2. Insert your Bearer Token:

   ```
   Bearer YOUR_API_TOKEN
   ```
3. Select the transcription endpoint
4. Upload an audio file
5. Execute the request and inspect the JSON response

This workflow is recommended and reflects how the service was developed and tested.

---

## CPU vs GPU Execution

- The **default configuration runs on CPU**, as the project was developed on a machine without a GPU.
- If a GPU is available, transcription would be significantly faster.
- GPU support may already be partially automatic, depending on the underlying Faster-Whisper configuration.

### Planned improvement

Expose a **CPU / GPU selection option directly in Swagger**, allowing both execution modes to coexist.

---

## Testing

The test suite includes:

- Unit tests for the deduplication logic
- Integration tests for the transcription pipeline

Run tests with:

```bash
pytest tests/
```

---

## Logging

The API logs authentication attempts and useful information both in the terminal and on a file named "whisper_app.log", in the root directory of the project.
The logging level can be changed in the local environment file (e.g. `DEBUG`) for additional details.

## Technical Notes

The project includes a local patch for the `PunctuationModel` class.

The modification fixes the token aggregation logic (`grouped_entities`) to align with current `transformers` standards, avoiding dependency pinning to legacy versions and ensuring forward compatibility with future `transformers` releases.

Punctuation is applied only for the following languages: `it`, `fr`, `de`, `en`

---

## Roadmap

- Simultaneous translation into Italian  
- Asynchronous streaming transcription via WebSocket  
- Explicit CPU / GPU selection via Swagger UI

---

## Limitations

- **Limited punctuation language support**  
  Punctuation restoration is currently available only for `it`, `fr`, `de`, and `en`, due to model constraints.

- **Translation target language constraints**  
  Whisper natively supports translation only into English.  
  Supporting multiple target languages would require loading additional models, significantly increasing memory usage.  
  For this reason, future translation features will focus primarily on Italian.

- **Model size vs accuracy trade-off**  
  The service uses the `whisper-turbo` model instead of the original large-v3 model.  
  This choice prioritizes reasonable transcription times on CPU-only machines at the cost of slightly reduced transcription fidelity.

- **Accent sensitivity**  
  Empirical testing suggests that non-native accents (e.g. French spoken by non-native speakers) may result in occasional word misinterpretations.

- **Disfluencies handling**  
  Vocal fillers such as “uhm” or long pauses may sometimes be transcribed as words.

- **Audio quality dependency**  
  Noisy or compressed audio (e.g. messaging app voice notes) tends to produce more transcription errors compared to clean, studio-quality recordings such as podcasts.

