from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.environ.get("PITCHTIP_DATA", "data"))
CLIPS_DIR = DATA_DIR / "clips"
PITCHES_DIR = DATA_DIR / "pitches"
POSE_DIR = DATA_DIR / "pose"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"

USER_AGENT = "Mozilla/5.0 (pitchtip research; personal use)"


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
