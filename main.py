import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Security, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.transcribe import load_model, transcribe_dynamic, refine_text

import os
from dotenv import load_dotenv
from secrets import compare_digest
import logging

SUPPORTED_PUNCTUATION_LANGS = ["it", "en", "de", "fr"]


# ============================
# INITIAL SETTINGS
# ============================

app = FastAPI(title="Whisper Private API")
auth_scheme = HTTPBearer()

load_dotenv()  # Load variables from .env file
API_TOKEN = os.getenv("API_TOKEN")

if not API_TOKEN:
    raise RuntimeError("API_TOKEN not set")

# Load model only once
model = load_model("large-v3-turbo", device="cpu")

# Define level and format of the logger
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Handler initialization
# 1. FileHandler: write on file
file_handler = logging.FileHandler("whisper_app.log")
file_handler.setFormatter(logging.Formatter(log_format))

# 2. StreamHandler: write on terminal
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(log_format))

# Root Logger Settings
logging.basicConfig(
    level=log_level,
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger(__name__)
logger.info("Logging system has been set: output both on terminal and on whisper_app.log")

logger = logging.getLogger(__name__)
logger.info("Whisper API  successfully started.")


# ============================
# AUTENTICAZIONE
# ============================

def validate_token(
    credentials: HTTPAuthorizationCredentials = Security(auth_scheme),) -> None:
    if credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authentication scheme")

    if not compare_digest(credentials.credentials, API_TOKEN):
        logger.warning(f"Login attempt with invalid token!")
        raise HTTPException(status_code=401, detail="Invalid token")
    logger.info("Authentication successful.")


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
    language: str = Query("it"),
    _: bool = Depends(validate_token),
):
    # Temporary file save
    tmp_path = await save_temp_file(file)

    try:
        raw_text = transcribe_dynamic(model, tmp_path, language=language)

        if language.lower() in SUPPORTED_PUNCTUATION_LANGS:
            cleaned_text = refine_text(raw_text)
            logger.info(f"Punctuation has been restored for the language: {language}")
        else:
            cleaned_text = raw_text
            logger.warning(f"Punctuation restoration has been skipped: {language} is not supported for the NLP model.")


        return {
            "filename": file.filename,
            "language": language,
            "status": "ok",
            "refined_text": cleaned_text,
        }

    finally:
        remove_file(tmp_path)


