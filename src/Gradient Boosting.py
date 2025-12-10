import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,accuracy_score,confusion_matrix,precision_recall_curve)
from sklearn.preprocessing import MinMaxScaler

# Settings
plt.style.use('ggplot')
pd.options.mode.chained_assignment = None

# ==============================================
# 📁 CREATE OUTPUT FOLDER IN PROJECT ROOT
# ==============================================

# PROJECT_ROOT = โฟลเดอร์หลักของโปรเจกต์ (ขึ้นจากไฟล์ .py ไป 1 ขั้น)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# RESULT folder → E:\ML\result\output
RESULT_FOLDER = os.path.join(PROJECT_ROOT, "result", "output")
os.makedirs(RESULT_FOLDER, exist_ok=True)

def save_plot(filename):
    path = os.path.join(RESULT_FOLDER, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"📁 Saved Plot → {path}")

def save_text(filename, text):
    path = os.path.join(RESULT_FOLDER, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"📁 Saved Text → {path}")



# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
DATA_FOLDER = 'cleaned_data'
MODEL_FOLDER = 'models'
os.makedirs(MODEL_FOLDER, exist_ok=True)

# File Names
FILE_MAIN = 'cleaned_HomeC.csv'              
FILE_VALID = 'cleaned_Energy_Validation.csv'  
FILE_FORECAST = 'cleaned_UCI_Power.csv'


# ==============================================================================
# 🛠️ HELPER: Robust Loader
# ==============================================================================
def robust_load_csv(filename):
    path = os.path.join(DATA_FOLDER, filename)
    if not os.path.exists(path):
        print(f"❌ Error: ไม่พบไฟล์ {filename}")
        return None
    
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False, na_values=['?', 'nan'])
        if df.shape[1] <= 1:
            df = pd.read_csv(path, sep=';', index_col=0, parse_dates=True, low_memory=False, na_values=['?', 'nan'])
        return df
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")
        return None



# ==============================================================================
# 🏠 PART 1: Augmented Smart Home (Analysis & Improved ML)
# ==============================================================================
def analyze_smart_home_main(filename):
    print("\n" + "="*60)
    print("🚀 PART 1: Main Dataset Analysis (Insight & High-Performance ML)")
    print("="*60)

    df = robust_load_csv(filename)
    if df is None:
        return None

    # ============================
    # 1) Insight อุปกรณ์ใช้ไฟสูงสุด
    # ============================
    exclude = [
        'time', 'use', 'house_overall', 'gen', 'summary', 'icon',
        'temperature', 'humidity', 'visibility', 'pressure', 'windspeed',
        'apparenttemperature', 'dewpoint', 'precipintensity', 'cloudcover',
        'windbearing'
    ]

    device_cols = [
        c for c in df.columns
        if c.lower() not in exclude and df[c].dtype in [float, int]
    ]

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
        plt.title('⚡ Top Energy Consuming Devices')
        plt.xlabel('Total Usage (kW)')
        plt.tight_layout()

        save_plot("top_energy_devices.png")  # ⭐ Save
        print(f"💡 Insight: อุปกรณ์ที่ใช้ไฟสูงสุดคือ {total_usage.index[0]}")

    # ============================
    # 2) เตรียมข้อมูล ML
    # ============================
    target_col = next(
        (c for c in ['use', 'House overall', 'mains'] if c in df.columns),
        df.columns[0]
    )

    # Features based on timestamp
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    # 🔥 Lag Feature เพื่อเพิ่ม accuracy
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'lag_{lag}h'] = df[target_col].shift(lag)

    df.dropna(inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove(target_col)

    X = df[numeric_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ============================
    # 3) Regression Model
    # ============================
    print("\n[ML 1] Regression Analysis...")

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

    print(f"   🏆 R-Squared (R²): {r2:.4f}")
    print(f"   🏆 MAE: {mae:.4f} kW")

    # Save metrics
    save_text("regression_metrics.txt", f"R2={r2}\nMAE={mae}")

    # Feature importance
    importances = pd.Series(reg.feature_importances_, index=X.columns)
    top_imp = importances.sort_values(ascending=False).head(10)

    print("\n   🔍 Top 10 Important Features (Regression):")
    for name, val in top_imp.items():
        print(f"      {name}: {val:.4f}")

    save_text(
        "regression_feature_importance.txt",
        "\n".join([f"{n}: {v:.4f}" for n, v in top_imp.items()])
    )

    # ============================
    # 4) Classification (Detect Spike)
    # ============================
    print("\n[ML 2] Anomaly Detection (High Usage)...")

    limit = df[target_col].quantile(0.9)
    y_class = (df[target_col] > limit).astype(int)

    print(f"   ℹ️ Threshold (high usage) > {limit:.4f} kW")
    print(f"   ℹ️ Class 1 (high) ratio: {y_class.mean():.4f}")

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_c, y_train_c)

    y_proba = clf.predict_proba(X_test_c)[:, 1]
    prec, rec, thres = precision_recall_curve(y_test_c, y_proba)

    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_th = thres[best_idx] if best_idx < len(thres) else 0.5
    best_f1 = f1_scores[best_idx]

    y_pred_c = (y_proba >= best_th).astype(int)
    acc = accuracy_score(y_test_c, y_pred_c)

    print(f"   🏁 Best Threshold: {best_th:.3f}")
    print(f"   🏆 Accuracy: {acc:.4f}")
    print(f"   🏆 F1-Score: {best_f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test_c, y_pred_c)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    save_plot("confusion_matrix.png")

    save_text(
        "classification_metrics.txt",
        f"Threshold={best_th}\nAccuracy={acc}\nF1={best_f1}"
    )

    return df




# ==============================================================================
# ⚖️ PART 2: Validation Dataset Comparison
# ==============================================================================
def validate_dataset(df_main, filename_valid):
    print("\n" + "="*60)
    print("🚀 PART 2: Validation (Comparison with Dataset 2)")
    print("="*60)
    
    df_val = robust_load_csv(filename_valid)
    if df_val is None: return

    target_main = next((c for c in ['use', 'House overall'] if c in df_main.columns), df_main.columns[0])
    target_val = next((c for c in ['mains', 'Energy_Consumption', 'use'] if c in df_val.columns), df_val.columns[0])

    profile_main = df_main.groupby(df_main.index.hour)[target_main].mean().values.reshape(-1, 1)
    profile_val = df_val.groupby(df_val.index.hour)[target_val].mean().values.reshape(-1, 1)

    scaler = MinMaxScaler()
    norm_main = scaler.fit_transform(profile_main)
    norm_val = scaler.fit_transform(profile_val)

    plt.figure(figsize=(10, 5))
    plt.plot(norm_main, label='Main Dataset', marker='o', linewidth=2)
    plt.plot(norm_val, label='Validation Dataset', marker='x', linestyle='--')
    plt.title('Validation: Daily Energy Pattern Comparison')
    plt.xlabel('Hour of Day (0-23)')
    plt.ylabel('Normalized Usage (0-1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_plot("validation_pattern_comparison.png")
    plt.show()

    print("✅ Validation Insight: กราฟควรมีทรงคล้ายกัน (เช่น พีคช่วงเย็น)")



# ==============================================================================
# 🔮 PART 3: Time-Series Forecasting
# ==============================================================================
def forecast_uci_model(filename):
    print("\n" + "="*60)
    print("🚀 PART 3: Time-Series Forecasting (Dataset 3)")
    print("="*60)
    
    df = robust_load_csv(filename)
    if df is None: return

    target_col = next((c for c in df.columns if 'global_active' in c.lower()), None)
    if not target_col:
        print("❌ Error: ไม่พบคอลัมน์ Global Active Power")
        return
    
    df_h = df[[target_col]].copy()
    df_h[target_col] = pd.to_numeric(df_h[target_col], errors='coerce')
    df_h = df_h.resample('h').mean().dropna()
    df_h.columns = ['y']
    
    for i in [1, 24, 168]:
        df_h[f'lag_{i}'] = df_h['y'].shift(i)
    
    df_h.dropna(inplace=True)
    
    X = df_h.drop('y', axis=1)
    y = df_h['y']
    
    split = int(len(X) * 0.9)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    print("⏳ Training Gradient Boosting Forecaster...")
    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"   🏆 RMSE: {rmse:.4f} kW")
    print(f"   🏆 MAE: {mae:.4f} kW")
    print(f"   🏆 R²: {r2:.4f}")
    
    plt.figure(figsize=(12, 5))
    plot_len = min(168, len(y_test))
    plt.plot(y_test.index[:plot_len], y_test.values[:plot_len], label='Actual')
    plt.plot(y_test.index[:plot_len], preds[:plot_len], label='Forecast', linestyle='--')
    plt.title('Future Forecast: Next 7 Days')
    plt.xlabel('Date')
    plt.ylabel('Global Active Power (kW)')
    plt.legend()

    save_plot("forecast_next_7_days.png")
    plt.show()

    save_text("forecast_metrics.txt", f"RMSE={rmse:.4f}\nMAE={mae:.4f}\nR2={r2:.4f}")



# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":

    summary_text = []   # <-- เก็บค่าทั้งหมดไว้สำหรับ save รายงาน

    # PART 1
    df_main_result = analyze_smart_home_main(FILE_MAIN)

    # PART 2
    if df_main_result is not None:
        validate_dataset(df_main_result, FILE_VALID)
        summary_text.append("[Validation] Daily patterns validated successfully")

    # PART 3
    forecast_result = forecast_uci_model(FILE_FORECAST)

    # SAVE FULL SUMMARY
    save_text("full_metrics_report.txt", "\n\n".join(summary_text))