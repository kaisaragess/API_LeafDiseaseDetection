from pathlib import Path
from typing import List

# Base directory
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Paths to artifacts and results
ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
BUNDLE_PATH: Path = ARTIFACTS_DIR / "rice_leaf_backend_bundle.joblib"
RESULTS_DIR: Path = ARTIFACTS_DIR / "results"

CSV_MACRO_F1: Path = RESULTS_DIR / "ringkasan_95CI_macro_f1.csv"
CSV_STATISTICAL: Path = RESULTS_DIR / "uji_statistik_friedman_wilcoxon_lengkap.csv"
CSV_ROBUSTNESS: Path = RESULTS_DIR / "hasil_robustness_test.csv"

# Preprocessing & Image Configuration
IMG_HEIGHT: int = 384
IMG_WIDTH: int = 128
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Out-of-Distribution (Non-Leaf Guard)
MIN_LEAF_AREA_RATIO: float = 0.08  # 8% threshold

# Upload Constraints
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES: List[str] = ["image/jpeg", "image/png", "image/jpg"]

# Supported Enums
SUPPORTED_PIPELINES: List[str] = ["B0", "B1", "B2", "P1"]
SUPPORTED_CLASSIFIERS: List[str] = [
    "KNN",
    "SVM",
    "RandomForest",
    "ExtraTrees",
    "Ensemble_Voting",
]
DEFAULT_PIPELINE: str = "P1"
DEFAULT_CLASSIFIER: str = "SVM"

# Scientific Disclaimer
DISCLAIMER_TEXT: str = (
    "Sistem ini merupakan purwarupa akademik pada data terbatas (120 citra) "
    "dan bukan alat diagnosis lapangan siap pakai."
)
