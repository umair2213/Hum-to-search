from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SONG_DIR = BASE_DIR / "songs"

SEPARATED_DIR = BASE_DIR / "separated"

QDRANT_PATH = BASE_DIR / "database" / "qdrant"

COLLECTION_NAME = "songs"

VECTOR_SIZE = 768

SUPPORTED_FORMATS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a"
}

DEMUCS_MODEL = "htdemucs"