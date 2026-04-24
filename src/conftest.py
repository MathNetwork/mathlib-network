import sys
from pathlib import Path

# Add src/ to sys.path so that `analysis`, `plots`, etc. are importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
