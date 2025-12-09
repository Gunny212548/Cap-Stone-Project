import pandas as pd
import numpy as np
import os

# โฟลเดอร์สำหรับเก็บไฟล์ที่ Clean แล้ว
OUTPUT_FOLDER = 'cleaned_data'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ฟังก์ชันจัดรูปแบบชื่อคอลัมน์
def normalize_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df

# ==========================================================
# 🏠 DATASET 1: Augmented Smart Home (HomeC)
# ==========================================================
def clean_dataset_1_homec(file_path):
    print(f"\n--- [1] Processing Dataset 1 (Main): {file_path} ---")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: ไม่พบไฟล์ '{file_path}' (อย่าลืมแตกไฟล์ RAR นะครับ)")
        return None

    try:
        df = pd.read_csv(file_path, low_memory=False)
        
        # ลบหน่วย [kW]
        df.columns = [c.replace(' [kW]', '') for c in df.columns]
        df = normalize_columns(df)
        
        # จัดการเวลา
        time_col = next((c for c in ['time', 'date'] if c in df.columns), None)
        if not time_col: raise ValueError("ไม่พบคอลัมน์เวลา")

        if pd.api.types.is_numeric_dtype(df[time_col]):
            df['datetime'] = pd.to_datetime(df[time_col], unit='s')
        else:
            df['datetime'] = pd.to_datetime(df[time_col], errors='coerce')

        df.set_index('datetime', inplace=True)

        # Resample Hourly
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.resample("1h").mean().ffill().bfill()

        # ลบคอลัมน์ขยะ
        cols_drop = ['year', 'month', 'day', 'hour', 'minute', 'weekofyear', 'time', 'date', 'unnamed_0']
        numeric_df.drop(columns=[c for c in cols_drop if c in numeric_df.columns], inplace=True)

        output_path = os.path.join(OUTPUT_FOLDER, "cleaned_HomeC.csv")
        numeric_df.to_csv(output_path)
        print(f"✅ Saved -> {output_path} | Shape: {numeric_df.shape}")
        return numeric_df

    except Exception as e:
        print(f"❌ Error cleaning Dataset 1: {e}")
        return None

# ==========================================================
# 🔋 DATASET 2: Energy Consumption (ไฟล์ Excel)
# เป้าหมาย: Clean เพื่อใช้ Validate ผลลัพธ์
# ==========================================================
def clean_dataset_2_energy_excel(file_path):
    print(f"\n--- [2] Processing Dataset 2 (Excel Validation): {file_path} ---")

    if not os.path.exists(file_path):
        print(f"❌ Error: ไม่พบไฟล์ '{file_path}'")
        return None

    try:
        # อ่านไฟล์ Excel (ต้องลง pip install openpyxl)
        print("⏳ Reading Excel file... (อาจใช้เวลาสักครู่)")
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # --- Logic การเปลี่ยนชื่อคอลัมน์ที่ยาวๆ ให้สั้นลง ---
        new_cols = {}
        for col in df.columns:
            lower_col = col.lower()
            if 'time' in lower_col: new_cols[col] = 'datetime'
            elif 'air conditioner' in lower_col: new_cols[col] = f'ac_{len(new_cols)}' # เผื่อมีแอร์หลายตัว
            elif 'fridge' in lower_col: new_cols[col] = 'fridge'
            elif 'fan' in lower_col or 'ventilador' in lower_col: new_cols[col] = 'fan'
            elif 'pc' in lower_col: new_cols[col] = 'pc'
            elif 'tv' in lower_col: new_cols[col] = 'tv'
            elif 'lights' in lower_col or 'lampara' in lower_col: new_cols[col] = 'lights'
            elif 'mains' in lower_col: new_cols[col] = 'mains_power'
            elif 'wash' in lower_col or 'lavadora' in lower_col: new_cols[col] = 'washing_machine'
        
        # เปลี่ยนชื่อและ Clean ชื่อที่เหลือ
        df.rename(columns=new_cols, inplace=True)
        df = normalize_columns(df)

        # จัดการเวลา
        if 'datetime' in df.columns:
            # ลบ Timezone ออก (เพราะ Excel นี้เป็น America/Bogota) เพื่อให้ง่ายต่อการ Resample
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            if df['datetime'].dt.tz is not None:
                df['datetime'] = df['datetime'].dt.tz_localize(None)
            df.set_index('datetime', inplace=True)
        
        # Resample เป็นรายชั่วโมง
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.resample("1h").mean().ffill().bfill()

        output_path = os.path.join(OUTPUT_FOLDER, "cleaned_Energy_Validation.csv")
        numeric_df.to_csv(output_path)
        print(f"✅ Saved -> {output_path} | Shape: {numeric_df.shape}")
        return numeric_df

    except Exception as e:
        print(f"❌ Error cleaning Dataset 2: {e}")
        return None

# ==========================================================
# ⚡ DATASET 3: UCI Power (TXT/CSV)
# ==========================================================
def clean_dataset_3_uci(file_path):
    print(f"\n--- [3] Processing Dataset 3 (UCI): {file_path} ---")

    if not os.path.exists(file_path):
        print(f"❌ Error: ไม่พบไฟล์ '{file_path}'")
        return None

    try:
        df = pd.read_csv(file_path, sep=";", na_values=['?', ''], low_memory=False)
        
        df['dt_str'] = df['Date'] + ' ' + df['Time']
        df['Datetime'] = pd.to_datetime(df['dt_str'], dayfirst=True, errors='coerce')
        df.set_index("Datetime", inplace=True)
        df.drop(columns=["Date", "Time", "dt_str"], errors="ignore", inplace=True)

        df = df.astype(float)
        numeric_df = df.resample("1h").mean().ffill().bfill()

        output_path = os.path.join(OUTPUT_FOLDER, "cleaned_UCI_Power.csv")
        numeric_df.to_csv(output_path)
        print(f"✅ Saved -> {output_path} | Shape: {numeric_df.shape}")
        return numeric_df

    except Exception as e:
        print(f"❌ Error cleaning Dataset 3: {e}")
        return None

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    # 1. ไฟล์ HomeC (จาก .rar ที่แตกแล้ว)
    file_1 = "HomeC.csv"  
    
    # 2. ไฟล์ Excel ต้นฉบับ (Dataset 2)
    file_2 = "Energy_Consumption.xlsx"
    
    # 3. ไฟล์ UCI (จาก .rar ที่แตกแล้ว)
    file_3 = "household_power_consumption.txt" 
    
    print("🚀 STARTED: Cleaning Process...")
    
    clean_dataset_1_homec(file_1)
    clean_dataset_2_energy_excel(file_2) # ใช้ฟังก์ชันอ่าน Excel
    clean_dataset_3_uci(file_3)
    
    print("\n🎉 All Done! Files are ready in 'cleaned_data' folder.")