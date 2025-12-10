import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor  # <-- NEW: สำหรับ Neural Network
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, 
                             f1_score, accuracy_score, classification_report, confusion_matrix)
from sklearn.preprocessing import MinMaxScaler, StandardScaler # <-- NEW: สำหรับ Scaling ข้อมูล

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
    if df is None: return None

    # --- 1. Insight: Top Consuming Devices ---
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
    df['lag_1h'] = df[target_col].shift(1) # Lag Feature
    df.dropna(inplace=True)

    features = ['hour', 'day_of_week', 'month', 'temperature', 'humidity', 'lag_1h']
    valid_features = [c for c in features if c in df.columns]

    X = df[valid_features]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. Regression ---
    print("\n[ML 1] Regression Analysis (Random Forest)...")
    reg = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"   🏆 R-Squared (R²): {r2:.4f}")
    print(f"   🏆 MAE: {mae:.4f} kW")
    
    # --- 4. Classification ---
    print("\n[ML 2] Anomaly Detection (High Usage)...")
    limit = df[target_col].mean() + (1.5 * df[target_col].std())
    y_class = (df[target_col] > limit).astype(int)
    
    print(f"   ℹ️ Threshold > {limit:.2f} kW")
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_train_c, y_train_c)
    y_pred_c = clf.predict(X_test_c)

    f1 = f1_score(y_test_c, y_pred_c)
    acc = accuracy_score(y_test_c, y_pred_c)
    print(f"   🏆 Accuracy: {acc:.4f}")
    print(f"   🏆 F1-Score: {f1:.4f}")
    
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
# 🔮 PART 3: Forecasting (Modified with MLPRegressor / Neural Network)
# ==============================================================================
def forecast_uci_model(filename):
    print("\n" + "="*60)
    print("🚀 PART 3: Time-Series Forecasting (Model: MLPRegressor / Neural Network)")
    print("="*60)
    
    # MLPRegressor และ StandardScaler ถูก Import ไว้ด้านบนแล้ว

    df = robust_load_csv(filename)
    if df is None: return

    # Find Target
    target_col = next((c for c in df.columns if 'global_active' in c.lower()), None)
    if not target_col:
        print("❌ Error: ไม่พบคอลัมน์ Global Active Power")
        return
    
    # --- 1. Preprocessing & Feature Engineering ---
    df_h = df[[target_col]].copy()
    df_h[target_col] = pd.to_numeric(df_h[target_col], errors='coerce')
    df_h = df_h.resample('h').mean().dropna() # Resample เป็นรายชั่วโมง
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
    
    # --- 2. Train/Test Split ---
    split = int(len(X) * 0.9) 
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # --- 3. Scaling (CRITICAL for Neural Networks) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"⏳ Training MLPRegressor (Neural Network) on {len(X_train)} hours of data...")
    
    # --- 4. Model Training (MLPRegressor) ---
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50),  # 2 layers: 100 neurons, then 50 neurons
        max_iter=500,                  # อนุญาตให้วนซ้ำได้สูงสุด 500 รอบ
        solver='adam',
        activation='relu',
        random_state=42,
        early_stopping=True,           # ใช้ Early Stopping เพื่อป้องกัน Overfitting
        validation_fraction=0.1        # 10% ของ Training Data เป็น Validation Set
    )
    
    # Training the model on scaled data
    model.fit(X_train_scaled, y_train)
    
    # Predict using scaled test data
    preds = model.predict(X_test_scaled)
    
    # --- 5. Evaluation ---
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"   🏆 RMSE: {rmse:.4f} kW")
    print(f"   🏆 MAE:  {mae:.4f} kW")
    print(f"   🏆 R²:   {r2:.4f}")
    
    # --- 6. Plotting ---
    plt.figure(figsize=(12, 5))
    plot_len = min(168, len(y_test))
    
    # Index สำหรับ Plotting
    plot_index = X_test.index[:plot_len] 
    
    plt.plot(plot_index, y_test[:plot_len], label='Actual', color='green', alpha=0.7)
    plt.plot(plot_index, preds[:plot_len], label='MLP Forecast', color='purple', linestyle='--', linewidth=2)
    plt.title('Future Forecast: Next 7 Days (MLPRegressor)')
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
    
    # 3. Forecasting (MLPRegressor)
    forecast_uci_model(FILE_FORECAST)