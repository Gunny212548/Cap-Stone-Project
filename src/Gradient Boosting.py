import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, confusion_matrix, precision_recall_curve
)
from sklearn.preprocessing import MinMaxScaler

plt.style.use('ggplot')
pd.options.mode.chained_assignment = None

# ===========================================================
# 📁 FIND PROJECT ROOT (always E:/ML)
# ===========================================================

def find_project_root():
    """
    ไล่ขึ้นโฟลเดอร์ทีละชั้นจนเจอโฟลเดอร์หลักของโปรเจ็กต์ที่มี cleaned_data / models
    """
    current = os.path.abspath(os.path.dirname(__file__))

    while True:
        if (
            os.path.isdir(os.path.join(current, "cleaned_data")) or
            os.path.isdir(os.path.join(current, "models"))
        ):
            return current

        parent = os.path.dirname(current)
        if parent == current:  # มาถึง root แล้ว
            return current

        current = parent


PROJECT_ROOT = find_project_root()

# ===========================================================
# 📁 CREATE result/output + result/figures
# ===========================================================

RESULT_DIR = os.path.join(PROJECT_ROOT, "result")
OUTPUT_DIR = os.path.join(RESULT_DIR, "output")
FIG_DIR = os.path.join(RESULT_DIR, "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("📂 Project Root =", PROJECT_ROOT)
print("📁 Output Folder =", OUTPUT_DIR)
print("📁 Figures Folder =", FIG_DIR)

# ===========================================================
# 📌 SAVE FUNCTIONS
# ===========================================================

def save_text(filename, text):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("📁 Saved Text →", path)

def save_csv(filename, df):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=True)
    print("📁 Saved CSV →", path)

def save_json(filename, data):
    import json
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print("📁 Saved JSON →", path)

def save_plot(filename):
    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print("📁 Saved Plot →", path)

# ===========================================================
# 📦 DATA CONFIGURATION
# ===========================================================

DATA_FOLDER = 'cleaned_data'
MODEL_FOLDER = 'models'
os.makedirs(MODEL_FOLDER, exist_ok=True)

FILE_MAIN = 'cleaned_HomeC.csv'
FILE_VALID = 'cleaned_Energy_Validation.csv'
FILE_FORECAST = 'cleaned_UCI_Power.csv'

# ===========================================================
# 🛠️ ROBUST CSV LOADER
# ===========================================================

def robust_load_csv(filename):
    path = os.path.join(DATA_FOLDER, filename)
    if not os.path.exists(path):
        print("❌ File Not Found:", filename)
        return None

    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False)
        if df.shape[1] <= 1:
            df = pd.read_csv(path, sep=";", index_col=0, parse_dates=True)
        return df
    except Exception as e:
        print("❌ Error:", e)
        return None

# ===========================================================
# 🏠 PART 1 — MAIN SMART HOME ML ANALYSIS
# ===========================================================

def analyze_smart_home_main(filename):
    print("\n" + "="*60)
    print("🚀 PART 1: Main Smart Home Dataset")
    print("="*60)

    df = robust_load_csv(filename)
    if df is None:
        return None

    # 🔥 DEVICE POWER USAGE INSIGHT
    exclude = [
        'time', 'use', 'house_overall', 'gen', 'summary', 'icon',
        'temperature', 'humidity', 'visibility', 'pressure', 'windspeed',
        'apparenttemperature', 'dewpoint', 'precipintensity', 'cloudcover',
        'windbearing'
    ]

    device_cols = [c for c in df.columns if c.lower() not in exclude and df[c].dtype in [float, int]]

    if device_cols:
        total_usage = df[device_cols].sum().sort_values(ascending=False).head(10)

        plt.figure(figsize=(10, 5))
        sns.barplot(
            x=total_usage.values,
            y=total_usage.index,
            hue=total_usage.index,
            palette="magma",
            dodge=False,
            legend=False
        )
        plt.title("⚡ Top Energy Consuming Devices")
        plt.xlabel("Total Usage (kW)")

        save_plot("top_energy_devices.png")
        plt.close()

    # 🔥 ML FEATURE ENGINEERING
    target_col = next((c for c in ['use', 'House overall', 'mains'] if c in df.columns), df.columns[0])
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    # Lag features improve prediction
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"lag_{lag}h"] = df[target_col].shift(lag)

    df.dropna(inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove(target_col)

    X = df[numeric_cols]
    y = df[target_col]

    # 🔥 REGRESSION MODEL
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    print("   R² =", r2)
    print("   MAE =", mae)

    save_text("regression_metrics.txt", f"R2={r2}\nMAE={mae}")

    feature_imp = pd.Series(reg.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    save_text("regression_feature_importance.txt", feature_imp.to_string())

    # 🔥 ANOMALY DETECTION
    print("\n[ML 2] Anomaly Detection")

    limit = df[target_col].quantile(0.9)
    y_class = (df[target_col] > limit).astype(int)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_c, y_train_c)

    y_proba = clf.predict_proba(X_test_c)[:, 1]
    prec, rec, thres = precision_recall_curve(y_test_c, y_proba)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-9)
    best_idx = np.argmax(f1_scores)

    best_th = thres[best_idx]
    best_f1 = f1_scores[best_idx]
    y_pred_c = (y_proba >= best_th).astype(int)

    acc = accuracy_score(y_test_c, y_pred_c)

    save_text(
        "classification_metrics.txt",
        f"Threshold={best_th}\nAccuracy={acc}\nF1={best_f1}"
    )

    cm = confusion_matrix(y_test_c, y_pred_c)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")

    save_plot("confusion_matrix.png")
    plt.close()

    return df

# ===========================================================
# ⚖️ PART 2 — VALIDATION DATASET COMPARISON
# ===========================================================

def validate_dataset(df_main, filename_valid):
    df_val = robust_load_csv(filename_valid)
    if df_val is None:
        return

    print("\n" + "="*60)
    print("🚀 PART 2: Validation Dataset Comparison")
    print("="*60)

    target_main = next((c for c in ['use', 'House overall'] if c in df_main.columns), df_main.columns[0])
    target_val = next((c for c in ['mains', 'Energy_Consumption', 'use'] if c in df_val.columns), df_val.columns[0])

    p_main = df_main.groupby(df_main.index.hour)[target_main].mean().values.reshape(-1, 1)
    p_val = df_val.groupby(df_val.index.hour)[target_val].mean().values.reshape(-1, 1)

    scaler = MinMaxScaler()
    norm_main = scaler.fit_transform(p_main)
    norm_val = scaler.fit_transform(p_val)

    plt.figure(figsize=(10, 5))
    plt.plot(norm_main, label="Main Dataset", marker="o")
    plt.plot(norm_val, label="Validation Dataset", linestyle="--", marker="x")
    plt.legend()
    plt.title("Daily Pattern Comparison")

    save_plot("validation_pattern_comparison.png")
    plt.close()

# ===========================================================
# 🔮 PART 3 — FORECASTING MODEL
# ===========================================================

def forecast_uci_model(filename):
    print("\n" + "="*60)
    print("🚀 PART 3: Time-Series Forecasting")
    print("="*60)

    df = robust_load_csv(filename)
    if df is None:
        return

    target_col = next((c for c in df.columns if "global_active" in c.lower()), None)
    if not target_col:
        return

    df_h = df[[target_col]].copy()
    df_h[target_col] = pd.to_numeric(df_h[target_col], errors='coerce')

    df_h = df_h.resample("h").mean().dropna()
    df_h.columns = ["y"]

    for i in [1, 24, 168]:
        df_h[f"lag_{i}"] = df_h["y"].shift(i)

    df_h.dropna(inplace=True)

    X = df_h.drop("y", axis=1)
    y = df_h["y"]

    split_idx = int(len(X) * 0.9)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = GradientBoostingRegressor(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    save_text("forecast_metrics.txt", f"RMSE={rmse}\nMAE={mae}\nR2={r2}")

    plt.figure(figsize=(12, 5))
    plt.plot(y_test.index[:168], y_test.values[:168], label="Actual")
    plt.plot(y_test.index[:168], preds[:168], label="Forecast", linestyle="--")
    plt.legend()
    plt.title("Forecast: Next 7 Days")

    save_plot("forecast_next_7_days.png")
    plt.close()

# ===========================================================
# MAIN EXECUTION
# ===========================================================

if __name__ == "__main__":
    df_main = analyze_smart_home_main(FILE_MAIN)

    if df_main is not None:
        validate_dataset(df_main, FILE_VALID)

    forecast_uci_model(FILE_FORECAST)

    print("\n🎉 All tasks completed successfully!")
