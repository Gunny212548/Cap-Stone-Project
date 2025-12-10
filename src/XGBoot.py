# ============================================
#   XGBoost.py (Full Working Version)
# ============================================

import os
import json
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add src folder
PROJECT_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_SRC)

# ML libs
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
except:
    print("❌ ERROR: โปรดติดตั้ง xgboost → pip install xgboost")

# Import dataset paths from utils_path
from utils_path import FILE_UCI, load_csv

plt.style.use('ggplot')
pd.options.mode.chained_assignment = None


# ===============================================================
#  FIND PROJECT ROOT (ML/) → result/output + result/figures
# ===============================================================
def find_project_root():
    current = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    while True:
        if os.path.isdir(os.path.join(current, "data")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return current
        current = parent

PROJECT_ROOT = find_project_root()

RESULT_DIR = os.path.join(PROJECT_ROOT, "result")
OUTPUT_DIR = os.path.join(RESULT_DIR, "output")
FIG_DIR = os.path.join(RESULT_DIR, "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("Project Root:", PROJECT_ROOT)
print("Output Folder:", OUTPUT_DIR)
print("Figures Folder:", FIG_DIR)


# ===============================================================
#  SAVE HELPERS
# ===============================================================
def save_text(fn, text):
    path = os.path.join(OUTPUT_DIR, fn)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Saved Text →", path)


def save_json(fn, data):
    path = os.path.join(OUTPUT_DIR, fn)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Saved JSON →", path)


def save_plot(fn):
    path = os.path.join(FIG_DIR, fn)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Plot →", path)


# ===============================================================
#  ROBUST CSV LOADER (เหมือน Gradient_Boosting.py)
# ===============================================================
def robust_load_csv(path):
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print("CSV Error:", e)
        return None

    # Normalize columns
    df.columns = (
        df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("[^a-z0-9_]", "", regex=True)
    )

    # Detect datetime
    time_cols = [c for c in df.columns if "time" in c]

    if len(time_cols) > 0:
        try:
            df[time_cols[0]] = pd.to_datetime(df[time_cols[0]])
            df = df.set_index(time_cols[0])
        except:
            print("⚠ Warning: Cannot parse datetime column.")
    else:
        print("⚠ Warning: No datetime column found.")

    return df


# ===============================================================
#  PART 3 – Forecasting using XGBoost
# ===============================================================
def forecast_xgboost():
    print("\n" + "=" * 60)
    print("🚀 PART 3: Time-Series Forecasting (XGBoost Model)")
    print("=" * 60)

    df = load_csv(FILE_UCI)

    if df is None:
        print("❌ ERROR: Cannot load UCI dataset.")
        return None

    # Target column
    target_col = next((c for c in df.columns if "global_active" in c.lower()), None)
    if target_col is None:
        print("❌ ERROR: Global Active Power column missing.")
        return None

    print(f"📌 Using target column: {target_col}")

    # Resample hourly
    df_h = df[[target_col]].copy()
    df_h[target_col] = pd.to_numeric(df_h[target_col], errors='ignore')
    df_h = df_h.resample("h").mean().dropna()
    df_h.columns = ["y"]

    # Generate lag features
    for lag in [1, 2, 24, 168]:
        df_h[f"lag_{lag}"] = df_h["y"].shift(lag)

    df_h["rolling_mean_24"] = df_h["y"].rolling(24).mean()

    df_h.dropna(inplace=True)

    X = df_h.drop("y", axis=1)
    y = df_h["y"]

    split = int(len(X) * 0.9)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows...")

    # Train model with Early Stopping
    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    preds = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"   🏆 RMSE: {rmse:.4f}")
    print(f"   🏆 MAE:  {mae:.4f}")
    print(f"   🏆 R²:   {r2:.4f}")

    save_text("xgboost_metrics.txt",
              f"rmse={rmse:.6f}\nmae={mae:.6f}\nr2={r2:.6f}")

    # Plot
    plt.figure(figsize=(12, 5))
    L = min(168, len(y_test))

    plt.plot(y_test.index[:L], y_test.values[:L], label="Actual", alpha=0.7)
    plt.plot(y_test.index[:L], preds[:L], label="XGBoost Forecast", linestyle="--", linewidth=2)

    plt.title("XGBoost Forecast – Next 7 Days")
    plt.xlabel("Date")
    plt.ylabel("Power (kW)")
    plt.legend()

    save_plot("xgboost_forecast.png")

    return {"rmse": rmse, "mae": mae, "r2": r2}


# ===============================================================
# MAIN
# ===============================================================
if __name__ == "__main__":
    metrics = forecast_xgboost()
    print("\n🎉 XGBoost Forecast Completed!")
    print(metrics)
