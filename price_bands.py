#!/usr/bin/env python3
"""
Extract BIDPEROFFER_D band availability for January 2022
for a selected set of DUID prefixes, and save to price_bands/bidperoffer_Jan2022.csv.
"""

import os
import pandas as pd
from nemosis import dynamic_data_compiler

# ─── 1. Configuration ────────────────────────────────────────────────────────

# Local cache for NEMOSIS raw data
CACHE_DIR = "./nemosis_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Output folder
OUTPUT_DIR = "./price_bands"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Time window
START_TIME = "2019/10/01 00:00:00"
END_TIME   = "2019/11/01 00:00:00"

# NEMOSIS table name
TABLE_NAME = "BIDPEROFFER_D"

# Columns to extract
SELECT_COLUMNS = [
    "SETTLEMENTDATE",
    "INTERVAL_DATETIME",
    "DUID",
    "BIDTYPE",
] + [f"BANDAVAIL{i}" for i in range(1, 11)]

# DUID prefixes of interest
PREFIXES = (
    "HPR",
    "DALNTH",
    "LBB",
    "TIB",
    "TORRB",
    "PPCCGT",
    "QPS",
    "BARKIPS",
    "AGLHAL",
    "OSB",
)

# ─── 2. Download & compile the raw CSVs for Jan 2022 ────────────────────────

print(f"Fetching {TABLE_NAME} from {START_TIME} to {END_TIME}…")
df = dynamic_data_compiler(
    start_time        = START_TIME,
    end_time          = END_TIME,
    table_name        = TABLE_NAME,
    raw_data_location = CACHE_DIR,
    select_columns    = SELECT_COLUMNS,
    keep_csv          = True     # cache the CSVs for next time
)

print(f"Retrieved total rows: {len(df):,}")

# ─── 3. Filter by DUID prefix ───────────────────────────────────────────────

mask = df["DUID"].str.startswith(PREFIXES)
df_filtered = df.loc[mask, SELECT_COLUMNS].reset_index(drop=True)

print(f"Rows after filtering by prefixes {PREFIXES}: {len(df_filtered):,}")

# ─── 4. Save result ─────────────────────────────────────────────────────────

out_path = os.path.join(OUTPUT_DIR, "bidperoffer_october2019.csv")
df_filtered.to_csv(out_path, index=False)
print(f"Saved filtered data to {out_path}")

# ─── 5. Show first few rows ────────────────────────────────────────────────

print("\nFirst 5 rows:")
print(df_filtered.head())
