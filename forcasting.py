import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Import ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, 
                             f1_score, accuracy_score, classification_report, confusion_matrix)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ตั้งค่าการแสดงผล
plt.style.use('ggplot')
pd.options.mode.chained_assignment = None

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
DATA_FOLDER = 'cleaned_data'
MODEL_FOLDER = 'models'
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ชื่อไฟล์ (แก้ให้ตรงกับของคุณ)
FILE_MAIN = 'cleaned_HomeC.csv'
FILE_VALID = 'cleaned_Energy_Validation.csv'
FILE_FORECAST = 'cleaned_UCI_Power.csv' 

# ==============================================================================
# 🏠 PART 1: Regression Analysis (ทำนายปริมาณการใช้ไฟ)
# Metrics: R2, MAE, MSE, RMSE
# ==============================================================================
def train_regression_model(filename):
    print("\n" + "="*60)
    print("🚀 PART 1: Regression Model (Predicting Energy Amount)")
    print("="*60)

    path = os.path.join(DATA_FOLDER, filename)
    if not os.path.exists(path):
        print(f"❌ Error: ไม่พบไฟล์ {path}")
        return None

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    
    # Feature Engineering
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Features Selection
    target_col = next((c for c in ['use', 'House overall', 'mains'] if c in df.columns), df.columns[0])
    features = ['hour', 'day_of_week', 'month', 'is_weekend', 'temperature', 'humidity']
    valid_features = [c for c in features if c in df.columns]

    X = df[valid_features]
    y = df[target_col]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    print("⏳ Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    # --- 📊 DATA SCIENCE METRICS (REGRESSION) ---
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    print(f"\n🏆 Model Evaluation Results:")
    print(f"   1. R-Squared (R²) : {r2:.4f}  (ยิ่งใกล้ 1 ยิ่งแม่น)")
    print(f"   2. MAE            : {mae:.4f} kW (ค่าความคลาดเคลื่อนเฉลี่ย)")
    print(f"   3. RMSE           : {rmse:.4f} kW (Error ที่ให้ความสำคัญกับค่าที่ผิดเยอะๆ)")

    # Plot Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds, alpha=0.3, color='blue')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Usage')
    plt.ylabel('Predicted Usage')
    plt.title(f'Regression Result: Actual vs Predicted (R2={r2:.2f})')
    plt.show() 
    
    # 

    return df  # ส่งคืนค่าเพื่อไปใช้ต่อ

# ==============================================================================
# 🚨 PART 1.5: Classification Analysis (High Usage Detection)
# Metrics: F1-Score, Accuracy, Confusion Matrix
# ==============================================================================
def train_classification_model(df):
    print("\n" + "="*60)
    print("🚦 PART 1.5: Classification Model (Detecting High Usage)")
    print("   (To calculate F1-Score, we transform this into a classification problem)")
    print("="*60)

    # 1. Create Classification Target (สร้างโจทย์ใหม่)
    # ถ้าใช้ไฟเกินค่าเฉลี่ย + 1 Standard Deviation ให้ถือว่าเป็น "High Usage" (Class 1)
    target_col = next((c for c in ['use', 'House overall', 'mains'] if c in df.columns), df.columns[0])
    threshold = df[target_col].mean() + df[target_col].std()
    
    df['is_high_usage'] = (df[target_col] > threshold).astype(int)
    print(f"ℹ️ Threshold for High Usage: > {threshold:.2f} kW")
    print(f"ℹ️ Class Balance: {df['is_high_usage'].value_counts().to_dict()} (0=Normal, 1=High)")

    features = ['hour', 'day_of_week', 'month', 'temperature', 'humidity']
    valid_features = [c for c in features if c in df.columns]

    X = df[valid_features]
    y = df['is_high_usage']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Classifier
    print("⏳ Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # --- 📊 DATA SCIENCE METRICS (CLASSIFICATION) ---
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n🏆 Classification Evaluation Results:")
    print(f"   1. Accuracy : {acc:.4f} (ทายถูกกี่ %)")
    print(f"   2. F1-Score : {f1:.4f} (ความแม่นยำถัวเฉลี่ย Precision/Recall)")
    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'High'], yticklabels=['Normal', 'High'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix: High Usage Detection')
    plt.show() 
    
    # [Image of confusion matrix]


# ==============================================================================
# 🔮 PART 3: Time-Series Forecasting (Corrected Version)
# ==============================================================================
def train_forecasting_model(filename):
    print("\n" + "="*60)
    print("🚀 PART 3: Time-Series Forecasting (UCI Data)")
    print("="*60)
    
    path = os.path.join(DATA_FOLDER, filename)
    if not os.path.exists(path):
        print(f"❌ Error: ไม่พบไฟล์ {path}")
        return

    print(f"📂 Loading: {filename}")

    # --- 🛠️ 1. โหลดข้อมูลแบบปลอดภัย (Robust Loading) ---
    try:
        # ลองโหลดแบบ Comma (,) ก่อน เพราะเป็นมาตรฐานไฟล์ Clean
        df = pd.read_csv(path, sep=',', index_col=0, parse_dates=True, low_memory=False, na_values=['?', 'nan'])
        
        # ถ้าโหลดแล้วมีแค่ 1 คอลัมน์ แปลว่าอาจจะใช้ separator ผิด ให้ลองใช้ Semicolon (;)
        if df.shape[1] <= 1:
            print("⚠️ Warning: ดูเหมือนรูปแบบไฟล์ไม่ใช่ Comma, กำลังลองใช้ Semicolon (;)...")
            df = pd.read_csv(path, sep=';', index_col=0, parse_dates=True, low_memory=False, na_values=['?', 'nan'])

    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    # --- 🛠️ 2. ตรวจสอบข้อมูลก่อนไปต่อ ---
    print(f"📊 Raw Data Shape: {df.shape}") # ดูขนาดข้อมูลก่อนลบ
    
    if df.empty:
        print("❌ Error: ไฟล์ไม่มีข้อมูล (Empty DataFrame)")
        return

    # ลบค่าว่าง (dropna) ทีหลัง และตรวจสอบว่าเหลือข้อมูลไหม
    df.dropna(inplace=True)
    if df.empty:
        print("❌ Error: ข้อมูลหายหมดหลังจากลบค่าว่าง (ตรวจสอบไฟล์ต้นฉบับว่ามีค่า Null เยอะเกินไปหรือไม่)")
        return

    # --- 🛠️ 3. จัดการ Target Column ---
    # หาคอลัมน์เป้าหมาย (Global_active_power)
    target_col = 'Global_active_power'
    
    # ถ้าหาไม่เจอ ให้ลองหาชื่อที่ใกล้เคียง (Case insensitive)
    if target_col not in df.columns:
        possible_cols = [c for c in df.columns if 'global' in c.lower() and 'active' in c.lower()]
        if possible_cols:
            target_col = possible_cols[0]
            print(f"ℹ️ Auto-detected target column: '{target_col}'")
        else:
            # ถ้ายังไม่เจออีก ให้ใช้คอลัมน์แรกสุดที่เป็นตัวเลข
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                target_col = numeric_cols[0]
                print(f"⚠️ Warning: ไม่พบคอลัมน์เป้าหมาย ใช้คอลัมน์แรกแทน: '{target_col}'")
            else:
                print(f"❌ Error: ไม่พบคอลัมน์ข้อมูลที่เป็นตัวเลขใน {df.columns}")
                return

    print(f"🎯 Target Column: {target_col}")

    # --- ส่วน Feature Engineering และ Modeling (เหมือนเดิม) ---
    df_hourly = df[[target_col]].copy()
    
    # แปลงเป็นตัวเลขให้ชัวร์ (เผื่อมี string ปน)
    df_hourly[target_col] = pd.to_numeric(df_hourly[target_col], errors='coerce')
    
    # Resample เป็นรายชั่วโมง
    df_hourly = df_hourly.resample('H').mean().dropna()
    df_hourly.columns = ['y']
    
    # Lag Features
    for i in [1, 24, 168]: 
        df_hourly[f'lag_{i}'] = df_hourly['y'].shift(i)
    df_hourly.dropna(inplace=True)

    if df_hourly.empty:
         print("❌ Error: ข้อมูลไม่เพียงพอสำหรับสร้าง Lag Features (ต้องการอย่างน้อย 1 สัปดาห์)")
         return

    X = df_hourly.drop('y', axis=1)
    y = df_hourly['y']

    # Split
    split = int(len(X) * 0.9)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print("⏳ Training Gradient Boosting Model...")
    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n🏆 Forecasting Evaluation Results:")
    print(f"   1. RMSE     : {rmse:.4f} kW")
    print(f"   2. MAE      : {mae:.4f} kW")
    print(f"   3. R-Squared: {r2:.4f}")

    # Plot
    plt.figure(figsize=(12, 5))
    plot_len = min(168, len(y_test)) # ป้องกัน error ถ้าข้อมูล test น้อยกว่า 168 ชม.
    plt.plot(y_test.index[:plot_len], y_test.values[:plot_len], label='Actual', color='green')
    plt.plot(y_test.index[:plot_len], preds[:plot_len], label='Forecast', color='red', linestyle='--')
    plt.title('Time Series Forecasting: Actual vs Forecast (1 Week)')
    plt.legend()
    plt.show()


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    # 1. Regression (R2, MAE, MSE)
    df_result = train_regression_model(FILE_MAIN)

    # 2. Classification (F1 Score, Accuracy) -> ใช้ Data เดียวกันกับ Part 1
    if df_result is not None:
        train_classification_model(df_result)

    # 3. Forecasting
    train_forecasting_model(FILE_FORECAST)