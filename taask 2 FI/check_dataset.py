# ============================================
# TASK 2 - CHECK DATASET STRUCTURE
# ============================================

print("🔍 TASK 2: CUSTOMER CHURN ANALYSIS - DATASET CHECK")
print("=" * 60)
print()

try:
    import pandas as pd
    print("✓ Pandas loaded successfully!")
except:
    print("✗ Please install pandas first: pip install pandas")
    exit()

# List CSV files in folder
import os
csv_files = [f for f in os.listdir('.') if f.lower().endswith('.csv')]

if csv_files:
    print("CSV files found:")
    for f in csv_files:
        print(f"  - {f}")
else:
    print("No CSV files found. Please put your dataset in this folder.")
    exit()

print()
file_name = input("Enter your dataset filename (e.g., 'telco_churn.csv'): ")

try:
    df = pd.read_csv(file_name)
    print(f"\n✓ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    
    print("\nFIRST 5 ROWS:")
    print(df.head())
    
    print("\nCOLUMN NAMES:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2}. {col}")
    
    print("\nDATASET INFO:")
    print(df.info())
    
    print("\nCHECKING FOR 'Churn' COLUMN:")
    if 'Churn' in df.columns:
        print("✓ 'Churn' column found!")
        print(f"Churn distribution:")
        print(df['Churn'].value_counts())
    else:
        churn_cols = [col for col in df.columns if 'churn' in col.lower()]
        if churn_cols:
            print(f"✓ Found similar column: {churn_cols[0]}")
        else:
            print("✗ No 'Churn' column found. Let me check other possibilities...")
            print("Columns with Yes/No values:")
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].nunique() < 5:
                    print(f"  - {col}: {df[col].unique()}")
    
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 60)
input("Press Enter to see the complete analysis code...")
