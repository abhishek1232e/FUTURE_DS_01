# 1. Open new file in IDLE
# 2. Copy this code:
print("Creating small sample file...")

import pandas as pd

# Read your big file
df = pd.read_csv('cleaned_online_retail.csv')

# Take only first 1000 rows (small enough for GitHub)
small_df = df.head(1000)

# Save as new file
small_df.to_csv('sample_data.csv', index=False)

print("✓ DONE! Created 'sample_data.csv' with 1000 rows")
print(f"Original: {len(df)} rows")
print(f"Sample: {len(small_df)} rows")
