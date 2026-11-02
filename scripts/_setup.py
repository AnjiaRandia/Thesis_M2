"""Repository paths and imports shared by the production command-line scripts."""

from pathlib import Path
import sys
import matplotlib

matplotlib.use("Agg")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from hiphases import style

style.apply()
DATA = PROJECT_ROOT / "data"
RESULTS = PROJECT_ROOT / "results"
FIGURES = RESULTS / "figures"
