FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from faster_whisper import WhisperModel; WhisperModel('dropbox-dash/faster-whisper-large-v3-turbo', device='cpu'); from src.deepmultilingualpunctuation import PunctuationModel; PunctuationModel()"

COPY src/ ./src/
COPY main.py .
COPY hf_bootstrap.py .

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]