from pathlib import Path
import platform
import os
import subprocess
import sys
import logging


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
# CHECK FOR LOCAL FILES
# ============================


def get_huggingface_cache_dir(model_name):
    """Automatically find HuggingFace cache directory for all OS."""
    # if you set the path as variable on .env
    custom_path = os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE")
    if custom_path:
        return Path(custom_path)

    # check OS
    home = Path.home()
    return home / ".cache" / "huggingface" / "hub" / model_name


def ensure_models():
    hf_bootstrap = Path(__file__).parent / "hf_bootstrap.py"

    if not (get_huggingface_cache_dir("models--dropbox-dash--faster-whisper-large-v3-turbo").exists()
            and get_huggingface_cache_dir("models--oliverguhr--fullstop-punctuation-multilang-large").exists()):
        subprocess.check_call([sys.executable, str(hf_bootstrap)])



ensure_models()

# ============================
# FLAG ONLINE OR OFFLINE MODE
# ============================

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Security, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.transcribe import transcribe_dynamic, load_whisper_model
from src.punctuationmodel import PunctuationModel
from functools import lru_cache

from dotenv import load_dotenv
from secrets import compare_digest

# ============================
# INITIAL SETTINGS
# ============================

app = FastAPI(title="Whisper Private API")
auth_scheme = HTTPBearer()

load_dotenv()  # Load variables from .env file
API_TOKEN = os.getenv("API_TOKEN")

if not API_TOKEN:
    raise RuntimeError("API_TOKEN not set")


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

def hf_cache_to_faster_whisper_model_dir(hf_model_cache_dir: str | Path) -> str:
    """
    Given a HuggingFace model cache directory (models--xxx),
    return the snapshot directory usable by faster-whisper.
    """
    snapshots_dir = hf_model_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        logger.error(f"Snapshots directory not found at {hf_model_cache_dir}")
        raise RuntimeError(f"No snapshots directory found in {hf_model_cache_dir}")

    snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not snapshots:
        logger.error(f"No snapshot found in {snapshots_dir}")
        raise RuntimeError(f"No snapshot found in {snapshots_dir}")

    return str(snapshots[0])


# ============================
# MODELS
# ============================

@lru_cache()
def get_whisper_model():
    """Loads Whisper model only once and keeps it in cache memory.
    If you run the script for the first time, it will download the weights for the model.
    After that, it will use your offline cache"""
    logger.info("Loading Whisper Model...")
    try:
        local_path = get_huggingface_cache_dir("models--dropbox-dash--faster-whisper-large-v3-turbo")
        snapshots_dir = hf_cache_to_faster_whisper_model_dir(local_path)
        logger.info("Whisper: Model weights found locally. No downloads needed.")
        return load_whisper_model(True,snapshots_dir, device="cpu")
    except Exception:
        logger.warning("Whisper: Model weights not found locally. They'll be downloaded")
        return load_whisper_model(False,"dropbox-dash/faster-whisper-large-v3-turbo", device="cpu")


@lru_cache()
def get_punct_model():
    """Loads Punctuation model only once and keeps it in cache memory.
    If you run the script for the first time, it will download the weights for the model.
    After that, it will use your offline cache"""
    model_name = "models--oliverguhr--fullstop-punctuation-multilang-large"
    local_path = get_huggingface_cache_dir(model_name)

    if local_path.exists():
        logger.info("Punctuation: Local directory found. Loading pure offline mode...")
    else:
        logger.info("Punctuation: Model is not present locally. Model weights will be downloaded at first startup of the app.")

    logger.info("Loading Punctuation Model...")
    return PunctuationModel()

# ============================
# ENDPOINT
# ============================

@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    language: str = Query("it", description="ISO Language Code"),
    _: bool = Depends(validate_token),
    whisper_model = Depends(get_whisper_model),
    punct_model = Depends(get_punct_model),
):
    # Temporary file save
    tmp_path = await save_temp_file(file)

    try:
        final_text = transcribe_dynamic(
            whisper_model,
            punct_model,
            tmp_path,
            language=language
        )


        return {
            "filename": file.filename,
            "language": language,
            "status": "ok",
            "refined_text": final_text,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        remove_file(tmp_path)


