# src/utils_paths.py

import os
import pandas as pd

# -----------------------
# FIND PROJECT ROOT
# -----------------------

def find_project_root():
    """
    เดินขึ้นบนจนเจอโฟลเดอร์ที่มี Data/
    """
    current = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    while True:
        if os.path.isdir(os.path.join(current, "Data")):
            return current

        parent = os.path.dirname(current)
        if parent == current:
            return current

        current = parent


PROJECT_ROOT = find_project_root()

# -----------------------
# PATHS TO PROCESSED DATA
# -----------------------

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "Data", "processed")

FILE_HOME = os.path.join(PROCESSED_DIR, "cleaned_HomeC.csv")
FILE_VALID = os.path.join(PROCESSED_DIR, "cleaned_Energy_Validation.csv")
FILE_UCI = os.path.join(PROCESSED_DIR, "cleaned_UCI_Power.csv")

# -----------------------
# DATA LOADER
# -----------------------

def load_csv(path):
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return None
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False)
    except Exception as e:
        print(f"❌ Error loading CSV {path}: {e}")
        return None
