# src/utils/config.py
from pathlib import Path

# 1. Get the directory where config.py lives
# 2. Go up 2 levels to get to the project root (fraud-detection/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 3. Define the data path relative to that root
SOURCE_DATA_PATH = PROJECT_ROOT / "data/raw/creditcard.csv"
RANDOM_STATE = 42

MODEL_OUTPUT_PATH = PROJECT_ROOT /"models/final_model.pkl"

MODEL_REPORT_PATH = PROJECT_ROOT /"reports"
