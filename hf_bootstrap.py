import logging
import os
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

logger.info("HF models bootstrap starting....")

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

AutoTokenizer.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")
AutoModel.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")

WhisperModel("dropbox-dash/faster-whisper-large-v3-turbo", device="cpu")

logger.info("HF models bootstrap completed.")