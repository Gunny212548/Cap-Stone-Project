import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, 
                             f1_score, accuracy_score, classification_report, confusion_matrix)
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

# ชื่อไฟล์ (ตรวจสอบให้ตรงกับในเครื่องของคุณ)
FILE_MAIN = 'cleaned_HomeC.csv'              # Dataset 1: Augmented Smart Home
FILE_VALID = 'cleaned_Energy_Validation.csv' # Dataset 2: Energy Consumption Smart Homes
FILE_FORECAST = 'cleaned_UCI_Power.csv' # Dataset 3: Household Electric Power

# ==============================================================================
# 🛠️ HELPER: Robust Loader (แก้ปัญหาโหลดไฟล์ไม่ติด)
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
    if df is None: return None

    # --- 1. Insight: อุปกรณ์ที่กินไฟเยอะที่สุด ---
    # กรองเอาเฉพาะคอลัมน์อุปกรณ์ (สมมติชื่อตาม Dataset HomeC ปกติ)
    exclude = ['time', 'use', 'house_overall', 'gen', 'summary', 'icon', 
               'temperature', 'humidity', 'visibility', 'pressure', 'windspeed', 
               'apparenttemperature', 'dewpoint', 'precipintensity', 'cloudcover', 'windbearing']
    
    device_cols = [c for c in df.columns if c.lower() not in exclude and df[c].dtype in [float, int]]
    
    if device_cols:
        total_usage = df[device_cols].sum().sort_values(ascending=False).head(8)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=total_usage.values, y=total_usage.index, palette='magma')
        plt.title('⚡ Top Energy Consuming Devices')
        plt.xlabel('Total Usage (kW)')
        plt.tight_layout()
        plt.show() 
        print(f"💡 Insight: อุปกรณ์ที่ใช้ไฟสูงสุดคือ {total_usage.index[0]}")

    # --- 2. Prepare Data for ML ---
    target_col = next((c for c in ['use', 'House overall', 'mains'] if c in df.columns), df.columns[0])
    
    # Feature Engineering
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # 🔥 KEY FIX: เพิ่ม Lag Feature (ค่าไฟชั่วโมงที่แล้ว) เพื่อดัน R2 Score
    df['lag_1h'] = df[target_col].shift(1)
    df.dropna(inplace=True) # ลบแถวแรกทิ้ง

    features = ['hour', 'day_of_week', 'month', 'temperature', 'humidity', 'lag_1h']
    valid_features = [c for c in features if c in df.columns]

    X = df[valid_features]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. Regression (ทำนายปริมาณไฟ) ---
    print("\n[ML 1] Regression Analysis...")
    reg = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    print(f"   🏆 R-Squared (R²): {r2:.4f} (เป้าหมาย > 0.8)")
    print(f"   🏆 MAE: {mae:.4f} kW")
    
    # --- 4. Classification (จับผิดไฟพุ่ง) ---
    print("\n[ML 2] Anomaly Detection (High Usage)...")
    # สร้าง Class: 1=สูงผิดปกติ (Mean + 1.5 Std), 0=ปกติ
    limit = df[target_col].mean() + (1.5 * df[target_col].std())
    y_class = (df[target_col] > limit).astype(int)
    
    print(f"   ℹ️ Threshold > {limit:.2f} kW")
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    
    # 🔥 KEY FIX: ใส่ class_weight='balanced' เพื่อดัน F1-Score
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_train_c, y_train_c)
    y_pred_c = clf.predict(X_test_c)

    f1 = f1_score(y_test_c, y_pred_c)
    acc = accuracy_score(y_test_c, y_pred_c)
    
    print(f"   🏆 Accuracy: {acc:.4f}")
    print(f"   🏆 F1-Score: {f1:.4f} (เป้าหมาย > 0.5 สำหรับ Imbalanced Data)")
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test_c, y_pred_c)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Anomaly Detection)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    return df

# ==============================================================================
# ⚖️ PART 2: Validation (Energy Consumption Smart Homes)
# ==============================================================================
def validate_dataset(df_main, filename_valid):
    print("\n" + "="*60)
    print("🚀 PART 2: Validation (Comparison with Dataset 2)")
    print("="*60)
    
    df_val = robust_load_csv(filename_valid)
    if df_val is None: return

    # Identify Targets
    target_main = next((c for c in ['use', 'House overall'] if c in df_main.columns), df_main.columns[0])
    target_val = next((c for c in ['mains', 'Energy_Consumption', 'use'] if c in df_val.columns), df_val.columns[0])

    # Calculate Hourly Profile
    profile_main = df_main.groupby(df_main.index.hour)[target_main].mean().values.reshape(-1, 1)
    profile_val = df_val.groupby(df_val.index.hour)[target_val].mean().values.reshape(-1, 1)
    
    # Normalize for comparison
    scaler = MinMaxScaler()
    norm_main = scaler.fit_transform(profile_main)
    norm_val = scaler.fit_transform(profile_val)
    
    plt.figure(figsize=(10, 5))
    plt.plot(norm_main, label='Main Dataset (Augmented Home)', marker='o', linewidth=2)
    plt.plot(norm_val, label='Validation Dataset', marker='x', linestyle='--')
    plt.title('Validation: Daily Energy Pattern Comparison')
    plt.xlabel('Hour of Day (0-23)')
    plt.ylabel('Normalized Usage (0-1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("✅ Validation Insight: กราฟควรมีทรงคล้ายกัน (เช่น พีคช่วงเย็น)")

# ==============================================================================
# 🔮 PART 3: Forecasting (Household Electric Power Consumption)
# ==============================================================================
def forecast_uci_model(filename):
    print("\n" + "="*60)
    print("🚀 PART 3: Time-Series Forecasting (Dataset 3)")
    print("="*60)
    
    df = robust_load_csv(filename)
    if df is None: return

    # Find Target
    target_col = next((c for c in df.columns if 'global_active' in c.lower()), None)
    if not target_col:
        print("❌ Error: ไม่พบคอลัมน์ Global Active Power")
        return
    
    # Resample & Lag Features
    df_h = df[[target_col]].copy()
    df_h[target_col] = pd.to_numeric(df_h[target_col], errors='coerce')
    df_h = df_h.resample('h').mean().dropna()
    df_h.columns = ['y']
    
    for i in [1, 24, 168]: # 1 hour, 1 day, 1 week
        df_h[f'lag_{i}'] = df_h['y'].shift(i)
    
    df_h.dropna(inplace=True)
    
    X = df_h.drop('y', axis=1)
    y = df_h['y']
    
    # Train/Test Split
    split = int(len(X) * 0.9)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    print("⏳ Training Gradient Boosting Forecaster...")
    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    
    print(f"   🏆 RMSE: {rmse:.4f} kW")
    print(f"   🏆 MAE: {mae:.4f} kW")
    
    # Forecast Plot
    plt.figure(figsize=(12, 5))
    plot_len = min(168, len(y_test))
    plt.plot(y_test.index[:plot_len], y_test.values[:plot_len], label='Actual', color='green')
    plt.plot(y_test.index[:plot_len], preds[:plot_len], label='Forecast', color='red', linestyle='--')
    plt.title('Future Forecast: Next 7 Days')
    plt.xlabel('Date')
    plt.ylabel('Global Active Power (kW)')
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
    
    # 3. Forecasting
    forecast_uci_model(FILE_FORECAST)