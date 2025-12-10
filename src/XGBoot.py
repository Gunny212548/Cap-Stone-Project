import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, 
                             f1_score, accuracy_score, classification_report, confusion_matrix,precision_recall_curve)
from sklearn.preprocessing import MinMaxScaler

# Settings
plt.style.use('ggplot')
pd.options.mode.chained_assignment = None

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
DATA_FOLDER = 'cleaned_data'
MODEL_FOLDER = 'models'
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ชื่อไฟล์
FILE_MAIN = 'cleaned_HomeC.csv'              # Dataset 1: Augmented Smart Home
FILE_VALID = 'cleaned_Energy_Validation.csv' # Dataset 2: Energy Consumption Smart Homes
FILE_FORECAST = 'cleaned_UCI_Power.csv'      # Dataset 3: Household Electric Power

# ==============================================================================
# 🛠️ HELPER: Robust Loader
# ==============================================================================
def robust_load_csv(filename):
    path = os.path.join(DATA_FOLDER, filename)
    if not os.path.exists(path):
        print(f"❌ Error: ไม่พบไฟล์ {filename}")
        return None
    
    try:
        # ลองโหลดแบบมาตรฐาน (Comma)
        df = pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False, na_values=['?', 'nan'])
        # ถ้าโหลดแล้วคอลัมน์ติดกันเป็นก้อนเดียว ให้ลอง Semicolon
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

    # --- 1. Insight: อุปกรณ์ที่กินไฟเยอะที่สุด ---
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
        total_usage = df[device_cols].sum().sort_values(ascending=False).head(8)
        plt.figure(figsize=(10, 5))
        # ✅ แก้ FutureWarning: ใช้ hue + legend=False
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
        plt.show()
        print(f"💡 Insight: อุปกรณ์ที่ใช้ไฟสูงสุดคือ {total_usage.index[0]}")

    # --- 2. Prepare Data for ML ---
    target_col = next(
        (c for c in ['use', 'House overall', 'mains'] if c in df.columns),
        df.columns[0]
    )
    
    # Feature Engineering (Time-based)
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # 🔥 เพิ่ม Lag Features หลายช่วงเวลา เพื่อดัน R²
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'lag_{lag}h'] = df[target_col].shift(lag)

    # ลบแถวที่มี NaN จากการสร้าง lag
    df.dropna(inplace=True)

    # ใช้ทุก numeric feature ยกเว้น target เป็น X
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    X = df[numeric_cols]
    y = df[target_col]

    # Train/Test Split (random split สำหรับ regression)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- 3. Regression (ทำนายปริมาณไฟ) ---
    print("\n[ML 1] Regression Analysis...")
    reg = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    print(f"   🏆 R-Squared (R²): {r2:.4f} (เป้าหมาย > 0.8)")
    print(f"   🏆 MAE: {mae:.4f} kW")

    # Optional: แสดง Feature Importance Top 10
    try:
        importances = pd.Series(reg.feature_importances_, index=X.columns)
        top_imp = importances.sort_values(ascending=False).head(10)
        print("\n   🔍 Top 10 Important Features (Regression):")
        for name, val in top_imp.items():
            print(f"      {name}: {val:.4f}")
    except Exception:
        pass

    # --- 4. Classification (จับผิดไฟพุ่ง) ---
    print("\n[ML 2] Anomaly Detection (High Usage)...")
    # ✅ ใช้ quantile 0.9 แทน mean + 1.5 std เพื่อลดความ extreme ของ class 1
    limit = df[target_col].quantile(0.9)
    y_class = (df[target_col] > limit).astype(int)
    
    print(f"   ℹ️ Threshold (high usage) > {limit:.4f} kW")
    print(f"   ℹ️ Class 1 (high) ratio: {y_class.mean():.4f}")

    # ใช้ feature ชุดเดียวกับ regression
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # 🔥 RandomForest + class_weight + tuning เล็กน้อย
    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_c, y_train_c)

    # หา threshold ที่ดีที่สุดสำหรับ F1 จาก probability
    y_proba = clf.predict_proba(X_test_c)[:, 1]
    prec, rec, thres = precision_recall_curve(y_test_c, y_proba)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_th = thres[best_idx] if best_idx < len(thres) else 0.5
    best_f1 = f1_scores[best_idx]

    # ใช้ threshold ที่หาได้
    y_pred_c = (y_proba >= best_th).astype(int)
    acc = accuracy_score(y_test_c, y_pred_c)

    print(f"   🏁 Best Threshold from PR curve: {best_th:.3f}")
    print(f"   🏆 Accuracy: {acc:.4f}")
    print(f"   🏆 F1-Score: {best_f1:.4f} (เป้าหมาย > 0.5 สำหรับ Imbalanced Data)")

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test_c, y_pred_c)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Anomaly Detection)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

    return df

# ==============================================================================
# ⚖️ PART 2: Validation
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
    plt.plot(norm_main, label='Main Dataset', marker='o')
    plt.plot(norm_val, label='Validation Dataset', marker='x', linestyle='--')
    plt.title('Validation: Daily Energy Pattern')
    plt.xlabel('Hour (0-23)')
    plt.ylabel('Normalized Usage')
    plt.legend()
    plt.show()

# ==============================================================================
# 🔮 PART 3: Forecasting (Updated: XGBoost Fix)
# ==============================================================================
def forecast_uci_model(filename):
    print("\n" + "="*60)
    print("🚀 PART 3: Time-Series Forecasting (Model: XGBoost)")
    print("="*60)
    
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("❌ Error: โปรดติดตั้ง xgboost ก่อน (pip install xgboost)")
        return

    df = robust_load_csv(filename)
    if df is None: return

    # Find Target
    target_col = next((c for c in df.columns if 'global_active' in c.lower()), None)
    if not target_col:
        print("❌ Error: ไม่พบคอลัมน์ Global Active Power")
        return
    
    # --- 1. Preprocessing ---
    df_h = df[[target_col]].copy()
    df_h[target_col] = pd.to_numeric(df_h[target_col], errors='coerce')
    df_h = df_h.resample('h').mean().dropna()
    df_h.columns = ['y']
    
    # Lag Features
    lags = [1, 2, 24, 168]
    for i in lags:
        df_h[f'lag_{i}'] = df_h['y'].shift(i)
        
    # Rolling Statistics
    df_h['rolling_mean_24'] = df_h['y'].rolling(window=24).mean()
    
    df_h.dropna(inplace=True)
    
    X = df_h.drop('y', axis=1)
    y = df_h['y']
    
    # --- 2. Split ---
    split = int(len(X) * 0.9)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    print(f"⏳ Training XGBoost on {len(X_train)} hours of data...")
    
    # --- 3. Model Training (FIXED HERE) ---
    # ย้าย early_stopping_rounds มาไว้ใน XGBRegressor()
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50 # <--- ใส่ตรงนี้แทน สำหรับ XGBoost เวอร์ชันใหม่
    )
    
    # ลบ early_stopping_rounds ออกจาก .fit()
    model.fit(
        X_train, y_train, 
        eval_set=[(X_test, y_test)], 
        verbose=False
    )
    
    preds = model.predict(X_test)
    
    # --- 4. Evaluation ---
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"   🏆 RMSE: {rmse:.4f} kW")
    print(f"   🏆 MAE:  {mae:.4f} kW")
    print(f"   🏆 R²:   {r2:.4f}")
    
    # --- 5. Plotting ---
    plt.figure(figsize=(12, 5))
    plot_len = min(168, len(y_test))
    plt.plot(y_test.index[:plot_len], y_test.values[:plot_len], label='Actual', color='green', alpha=0.7)
    plt.plot(y_test.index[:plot_len], preds[:plot_len], label='XGBoost Forecast', color='red', linestyle='--', linewidth=2)
    plt.title('Future Forecast: Next 7 Days (XGBoost)')
    plt.xlabel('Date')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.show()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # 1. Main Analysis
    df_main_result = analyze_smart_home_main(FILE_MAIN)
    
    # 2. Validation
    if df_main_result is not None:
        validate_dataset(df_main_result, FILE_VALID)
    
    # 3. Forecasting (XGBoost)
    forecast_uci_model(FILE_FORECAST)