import logging
import os
import sys
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_format = "%(asctime)s [%(levelname)s] hf_bootstrap: %(message)s"

logging.basicConfig(
    level=log_level,
    format=log_format,
    handlers=[
        logging.FileHandler("whisper_app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
logger.info("HF models bootstrap starting....")


os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

AutoTokenizer.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")
AutoModel.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")

WhisperModel("dropbox-dash/faster-whisper-large-v3-turbo")


model_name = "facebook/nllb-200-distilled-600M"
AutoTokenizer.from_pretrained(model_name)
AutoModelForSeq2SeqLM.from_pretrained(model_name)

logger.info("HF models bootstrap completed.")