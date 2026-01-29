import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.transcribe import load_model, transcribe_dynamic, refine_text

import os
from dotenv import load_dotenv
from secrets import compare_digest
import logging


# ============================
# CONFIGURAZIONE
# ============================

app = FastAPI(title="Whisper Private API")
auth_scheme = HTTPBearer()

load_dotenv()  # Carica le variabili dal file .env
API_TOKEN = os.getenv("API_TOKEN")

if not API_TOKEN:
    raise RuntimeError("API_TOKEN not set")

# Caricamento modello una sola volta
model = load_model("large-v3-turbo", device="cpu")

# Definiamo il livello e il formato
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Creiamo i "Handler" (i destinatari dei log)
# 1. FileHandler: scrive su file
file_handler = logging.FileHandler("whisper_app.log")
file_handler.setFormatter(logging.Formatter(log_format))

# 2. StreamHandler: scrive a terminale
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(log_format))

# Configuriamo il logger radice (Root Logger)
logging.basicConfig(
    level=log_level,
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger(__name__)
logger.info("Sistema di logging configurato: output su terminale e su whisper_app.log")

logger = logging.getLogger(__name__)
logger.info("API Whisper avviata correttamente.")


# ============================
# AUTENTICAZIONE
# ============================

def validate_token(
    credentials: HTTPAuthorizationCredentials = Security(auth_scheme),) -> None:
    if credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authentication scheme")

    if not compare_digest(credentials.credentials, API_TOKEN):
        logger.warning(f"Tentativo di accesso con token non valido!")
        raise HTTPException(status_code=401, detail="Invalid token")
    logger.info("Autenticazione riuscita.")


# ============================
# UTILITIES
# ============================

async def save_temp_file(uploaded: UploadFile, suffix=".wav") -> str:
    """
    Saves an uploaded file to a temporary file and returns its path.
    The caller is responsible for deleting the file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await uploaded.read())
        return tmp.name


def remove_file(path: str) -> None:
    """Safely removes a file if it exists."""
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


# ============================
# ENDPOINT
# ============================

@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    _: bool = Depends(validate_token),
):
    # Salvataggio del file temporaneo
    tmp_path = await save_temp_file(file)

    try:
        raw_text = transcribe_dynamic(model, tmp_path, language="it")
        cleaned_text = refine_text(raw_text)

        return {
            "filename": file.filename,
            "status": "ok",
            "refined_text": cleaned_text,
        }

    finally:
        remove_file(tmp_path)


