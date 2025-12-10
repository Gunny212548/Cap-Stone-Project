# ============================================
#   Gradient_Boosting.py (Full Working Version)
# ============================================

import os
import json
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to import path
PROJECT_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_SRC)

# ML libs
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, confusion_matrix, precision_recall_curve
)
from sklearn.preprocessing import MinMaxScaler

# Import path utilities
from utils_path import FILE_HOME, FILE_VALID, FILE_UCI, load_csv

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
print("Output:", OUTPUT_DIR)
print("Figures:", FIG_DIR)


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
#  ROBUST CSV LOADER
# ===============================================================
def robust_load_csv(path):
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print("CSV error:", e)
        return None

    df.columns = (
        df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("[^a-z0-9_]", "", regex=True)
    )

    # detect datetime column
    time_cols = [c for c in df.columns if "time" in c]

    if len(time_cols) > 0:
        try:
            df[time_cols[0]] = pd.to_datetime(df[time_cols[0]])
            df = df.set_index(time_cols[0])
        except:
            print("⚠ Cannot parse datetime")
    else:
        print("⚠ No datetime column detected")

    return df


# ===============================================================
#  PART 1 — MAIN SMART HOME ANALYSIS
# ===============================================================
def analyze_smart_home_main(filepath):
    print("\n" + "="*60)
    print("🚀 PART 1: Main Dataset Analysis")
    print("="*60)

    df = robust_load_csv(filepath)
    if df is None:
        return

    # --------------------------
    # Insight: Top energy devices
    # --------------------------
    exclude = [
        'time', 'use', 'house_overall', 'gen', 'summary', 'icon',
        'temperature', 'humidity', 'visibility', 'pressure', 'windspeed',
        'apparenttemperature', 'dewpoint', 'precipintensity', 'cloudcover',
        'windbearing'
    ]

    device_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [float, int]]

    if device_cols:
        total_usage = df[device_cols].sum().sort_values(ascending=False).head(10)

        plt.figure(figsize=(10, 5))
        sns.barplot(
            x=total_usage.values,
            y=total_usage.index,
            hue=total_usage.index,
            dodge=False,
            legend=False,
            palette='magma'
        )
        plt.title("⚡ Top Energy Devices")
        save_plot("top_energy_devices.png")

        print("Top device:", total_usage.index[0])

    # --------------------------
    # Feature Engineering
    # --------------------------
    target_col = next((c for c in ['use', 'house_overall', 'mains'] if c in df.columns), df.columns[0])

    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"lag_{lag}h"] = df[target_col].shift(lag)

    df.dropna(inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove(target_col)

    X = df[numeric_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --------------------------
    # Regression
    # --------------------------
    print("\n[ML 1] Regression ...")

    reg = RandomForestRegressor(
        n_estimators=300,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    reg.fit(X_train, y_train)

    preds = reg.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    print("R2:", r2)
    print("MAE:", mae)

    save_text("regression_metrics.txt", f"R2={r2}\nMAE={mae}")

    # Feature importance
    importance = pd.Series(reg.feature_importances_, index=X.columns).sort_values(ascending=False)
    save_text("regression_feature_importance.txt", importance.head(10).to_string())

    # --------------------------
    # Classification / Anomaly
    # --------------------------
    print("\n[ML 2] Anomaly Detection ...")

    limit = df[target_col].quantile(0.9)
    y_class = (df[target_col] > limit).astype(int)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train_c, y_train_c)

    y_prob = clf.predict_proba(X_test_c)[:, 1]

    prec, rec, thres = precision_recall_curve(y_test_c, y_prob)
    f1 = 2*(prec*rec)/(prec+rec+1e-9)
    best_idx = np.argmax(f1)
    best_th = thres[best_idx]

    y_pred_c = (y_prob >= best_th).astype(int)
    acc = accuracy_score(y_test_c, y_pred_c)

    save_text("classification_metrics.txt", f"Threshold={best_th}\nAccuracy={acc}\nF1={f1[best_idx]}")

    cm = confusion_matrix(y_test_c, y_pred_c)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    save_plot("confusion_matrix.png")

    return df


# ===============================================================
#  PART 2 — VALIDATION
# ===============================================================
def validate_dataset(df_main, df_val):
    print("\n" + "="*40)
    print("🚀 PART 2: Validation Comparison")
    print("="*40)

    if df_main is None or df_val is None:
        print("❌ Missing dataset(s). Cannot validate.")
        return

    # -----------------------------
    # 1) หา target column ของ Main Dataset
    # -----------------------------
    possible_main = ['use', 'house_overall', 'mains']
    target_main = next((c for c in possible_main if c in df_main.columns), None)

    if target_main is None:
        target_main = df_main.select_dtypes(include=[np.number]).columns[0]
        print(f"⚠️ target_main not found. Auto-selected: {target_main}")

    # -----------------------------
    # 2) หา target column ของ Validation Dataset
    # -----------------------------
    possible_val = ['mains', 'mains_power', 'energy_consumption', 'use']
    target_val = next((c for c in possible_val if c in df_val.columns), None)

    if target_val is None:
        target_val = df_val.select_dtypes(include=[np.number]).columns[0]
        print(f"⚠️ target_val not found. Auto-selected: {target_val}")

    # -----------------------------
    # 3) ป้องกันปัญหา index ไม่มีเวลา
    # -----------------------------
    if not isinstance(df_main.index, pd.DatetimeIndex):
        print("❌ df_main has no datetime index!")
        return
    if not isinstance(df_val.index, pd.DatetimeIndex):
        print("❌ df_val has no datetime index!")
        return

    # -----------------------------
    # 4) คำนวณค่าเฉลี่ยรายชั่วโมง
    # -----------------------------
    try:
        p_main = df_main.groupby(df_main.index.hour)[target_main].mean().values.reshape(-1, 1)
        p_val = df_val.groupby(df_val.index.hour)[target_val].mean().values.reshape(-1, 1)
    except Exception as e:
        print("❌ Error during grouping:", e)
        print("df_main columns:", df_main.columns)
        print("df_val columns:", df_val.columns)
        return

    # -----------------------------
    # 5) Normalize และ Plot
    # -----------------------------
    scaler = MinMaxScaler()
    nm = scaler.fit_transform(p_main)
    nv = scaler.fit_transform(p_val)

    plt.figure(figsize=(10, 5))
    plt.plot(nm, label="Main", marker="o")
    plt.plot(nv, label="Validation", linestyle="--", marker="x")
    plt.legend()
    plt.title("Daily Pattern Comparison")

    save_plot("validation_pattern_comparison.png")
    plt.close()

    print("✅ Validation comparison saved!")



# ===============================================================
#  PART 3 — FORECASTING
# ===============================================================
def forecast_uci(df_uci):
    print("\n" + "="*60)
    print("🚀 PART 3: Forecasting")
    print("="*60)

    target_col = next((c for c in df_uci.columns if "global_active" in c.lower()), None)

    df_h = df_uci[[target_col]].copy()
    df_h[target_col] = pd.to_numeric(df_h[target_col], errors="coerce")
    df_h = df_h.resample("h").mean().dropna()
    df_h.columns = ["y"]

    for lag in [1, 24, 168]:
        df_h[f"lag_{lag}"] = df_h["y"].shift(lag)

    df_h.dropna(inplace=True)

    X = df_h.drop("y", axis=1)
    y = df_h["y"]

    split = int(len(X)*0.9)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    save_text("forecast_metrics.txt", f"RMSE={rmse}\nMAE={mae}\nR2={r2}")

    plt.figure(figsize=(12, 5))
    L = min(168, len(preds))
    plt.plot(y_test.index[:L], y_test.values[:L], label="Actual")
    plt.plot(y_test.index[:L], preds[:L], label="Forecast", linestyle="--")
    plt.legend()
    save_plot("forecast_next_7_days.png")

    return {"rmse": rmse, "mae": mae, "r2": r2}


# ===============================================================
#  MAIN EXECUTION
# ===============================================================
if __name__ == "__main__":

    # Load datasets
    df_main = load_csv(FILE_HOME)
    df_valid = load_csv(FILE_VALID)
    df_uci = load_csv(FILE_UCI)

    print("Loaded:", df_main.shape, df_valid.shape, df_uci.shape)

    # PART 1
    analyze_smart_home_main(FILE_HOME)

    # PART 2
    validate_dataset(df_main, df_valid)

    # PART 3
    metrics = forecast_uci(df_uci)

    save_json("summary.json", {"forecast": metrics})
    print("\n🎉 All Tasks Completed Successfully!")
