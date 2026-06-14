import sys
from pathlib import Path

# Add src/ to path so tests can import transcript_pipeline, etc.
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
