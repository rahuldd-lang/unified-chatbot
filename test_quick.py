#!/usr/bin/env python3
"""Quick sanity check of disaster server CSV loading."""

import sys
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

print("Testing CSV paths...")
csv_1900 = Path(__file__).parent / "DISASTERS" / "1900_2021_DISASTERS.xlsx - emdat data.csv"
csv_1970 = Path(__file__).parent / "DISASTERS" / "1970-2021_DISASTERS.xlsx - emdat data.csv"

print(f"1900 CSV exists: {csv_1900.exists()} -> {csv_1900}")
print(f"1970 CSV exists: {csv_1970.exists()} -> {csv_1970}")

if not csv_1900.exists() or not csv_1970.exists():
    print("ERROR: CSV files not found!")
    sys.exit(1)

print("\nLoading CSV files...")
import pandas as pd

df_1900 = pd.read_csv(csv_1900)
df_1970 = pd.read_csv(csv_1970)

print(f"1900 CSV: {len(df_1900)} rows, cols: {list(df_1900.columns[:5])}")
print(f"1970 CSV: {len(df_1970)} rows, cols: {list(df_1970.columns[:5])}")

print("\nTesting disaster server import...")
try:
    from mcp_servers import disasters_server as ds
    print("✓ disasters_server imported")

    print("\nLoading disaster data...")
    df = ds._load_data()
    print(f"✓ Combined DataFrame: {len(df)} rows")
    print(f"  Columns: {list(df.columns[:8])}")

    print("\nTesting query tools...")
    result = ds.query_disasters_summary_stats()
    import json
    summary = json.loads(result)
    print(f"✓ Global stats: {summary['total_events']} events, {summary['total_deaths']} deaths")

    print("\n✅ All basic checks passed!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
