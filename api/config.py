from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
CODE_DIR = BASE_DIR / "code"
CHECKPOINT_PATH = CODE_DIR / "checkpoints" / "best_model.pt"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac"}
MAX_FILE_SIZE_MB = 100
VALID_TARGETS = {"vocals", "drums", "bass", "other"}
