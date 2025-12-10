import pandas as pd
import numpy as np
import os

# ==========================================================
# 📌 กำหนดโฟลเดอร์ Data/Processed (แทน cleaned_data เดิม)
# ==========================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_FOLDER = os.path.join(PROJECT_ROOT, "data", "processed")

os.makedirs(PROCESSED_FOLDER, exist_ok=True)
print(f"📁 Saving processed files to: {PROCESSED_FOLDER}")

# ==========================================================
# 📌 ฟังก์ชันจัดรูปแบบชื่อคอลัมน์
# ==========================================================

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
    print(f"\n--- [1] Processing Dataset 1 (HomeC): {file_path} ---")

    if not os.path.exists(file_path):
        print(f"❌ Error: File not found '{file_path}'")
        return None

    try:
        df = pd.read_csv(file_path, low_memory=False)

        df.columns = [c.replace(" [kW]", "") for c in df.columns]
        df = normalize_columns(df)

        time_col = next((c for c in ["time", "date"] if c in df.columns), None)
        if time_col is None:
            raise ValueError("No time/date column found!")

        if pd.api.types.is_numeric_dtype(df[time_col]):
            df["datetime"] = pd.to_datetime(df[time_col], unit="s")
        else:
            df["datetime"] = pd.to_datetime(df[time_col], errors="coerce")

        df.set_index("datetime", inplace=True)

        numeric_df = df.select_dtypes(include=[np.number]).resample("1H").mean().ffill().bfill()

        cols_drop = ["year", "month", "day", "hour", "minute", "weekofyear", "time", "date", "unnamed_0"]
        numeric_df.drop(columns=[c for c in cols_drop if c in numeric_df.columns], inplace=True)

        output_path = os.path.join(PROCESSED_FOLDER, "cleaned_HomeC.csv")
        numeric_df.to_csv(output_path)

        print(f"✅ Saved → {output_path} | Shape: {numeric_df.shape}")
        return numeric_df

    except Exception as e:
        print(f"❌ Error cleaning Dataset 1: {e}")
        return None


# ==========================================================
# 🔋 DATASET 2: Energy Consumption (Excel)
# ==========================================================

def clean_dataset_2_energy_excel(file_path):
    print(f"\n--- [2] Processing Dataset 2 (Excel): {file_path} ---")

    if not os.path.exists(file_path):
        print(f"❌ Error: File not found '{file_path}'")
        return None

    try:
        print("⏳ Reading Excel...")
        df = pd.read_excel(file_path, engine="openpyxl")

        new_cols = {}
        for col in df.columns:
            lc = col.lower()
            if "time" in lc:
                new_cols[col] = "datetime"
            elif "air conditioner" in lc:
                new_cols[col] = f"ac_{len(new_cols)}"
            elif "fridge" in lc:
                new_cols[col] = "fridge"
            elif "fan" in lc or "ventilador" in lc:
                new_cols[col] = "fan"
            elif "pc" in lc:
                new_cols[col] = "pc"
            elif "tv" in lc:
                new_cols[col] = "tv"
            elif "lights" in lc or "lampara" in lc:
                new_cols[col] = "lights"
            elif "mains" in lc:
                new_cols[col] = "mains_power"
            elif "wash" in lc or "lavadora" in lc:
                new_cols[col] = "washing_machine"

        df.rename(columns=new_cols, inplace=True)
        df = normalize_columns(df)

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df["datetime"] = df["datetime"].dt.tz_localize(None) if df["datetime"].dt.tz is not None else df["datetime"]
            df.set_index("datetime", inplace=True)

        numeric_df = df.select_dtypes(include=[np.number]).resample("1H").mean().ffill().bfill()

        output_path = os.path.join(PROCESSED_FOLDER, "cleaned_Energy_Validation.csv")
        numeric_df.to_csv(output_path)

        print(f"✅ Saved → {output_path} | Shape: {numeric_df.shape}")
        return numeric_df

    except Exception as e:
        print(f"❌ Error cleaning Dataset 2: {e}")
        return None


# ==========================================================
# ⚡ DATASET 3: UCI Household Power
# ==========================================================

def clean_dataset_3_uci(file_path):
    print(f"\n--- [3] Processing Dataset 3 (UCI): {file_path} ---")

    if not os.path.exists(file_path):
        print(f"❌ Error: File not found '{file_path}'")
        return None

    try:
        df = pd.read_csv(file_path, sep=";", na_values=["?", ""], low_memory=False)

        df["dt_str"] = df["Date"] + " " + df["Time"]
        df["Datetime"] = pd.to_datetime(df["dt_str"], dayfirst=True, errors="coerce")

        df.set_index("Datetime", inplace=True)
        df.drop(columns=["Date", "Time", "dt_str"], inplace=True, errors="ignore")

        df = df.astype(float)
        numeric_df = df.resample("1H").mean().ffill().bfill()

        output_path = os.path.join(PROCESSED_FOLDER, "cleaned_UCI_Power.csv")
        numeric_df.to_csv(output_path)

        print(f"✅ Saved → {output_path} | Shape: {numeric_df.shape}")
        return numeric_df

    except Exception as e:
        print(f"❌ Error cleaning Dataset 3: {e}")
        return None


# ==========================================================
# MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":
    file_1 = os.path.join(PROJECT_ROOT, "data", "raw", "HomeC.csv")
    file_2 = os.path.join(PROJECT_ROOT, "data", "raw", "Energy_Consumption.xlsx")
    file_3 = os.path.join(PROJECT_ROOT, "data", "raw", "household_power_consumption.txt")

    print("🚀 START CLEANING PROCESS...")

    clean_dataset_1_homec(file_1)
    clean_dataset_2_energy_excel(file_2)
    clean_dataset_3_uci(file_3)

    print("\n🎉 All Done! Cleaned files saved in data/processed/")
