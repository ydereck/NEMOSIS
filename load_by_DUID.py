#!/usr/bin/env python3
"""
Download unit-level cleared energy & FCAS volumes for 18 South-Australian generators
2019-10-01 00:05 → 2020-03-31 00:00 (inclusive).

Requires:  • nemosis  • pandas  • python-dateutil  • (optional) tqdm
"""

import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pandas as pd
from nemosis import dynamic_data_compiler

# ──────────────────────────────────────────────────────────────────────────
# 0.  User settings
# ──────────────────────────────────────────────────────────────────────────
RAW_CACHE = "./NEMOSIS_cache"
os.makedirs(RAW_CACHE, exist_ok=True)

DUIDS = [
    "HPRG1", "HPRL1", "DALNTH01", "DALNTHL1",
    "LBBG1", "LBBL1",
    "TORRB1", "TORRB2", "TORRB3", "TORRB4",
    "PPCCGT",
    "QPS1", "QPS2", "QPS3", "QPS4", "QPS5",
    "BARKIPS1", "AGLHAL", "OSB-AG"
]

START_TS = datetime(2019, 10, 1, 0, 5)  # inclusive
END_TS   = datetime(2020, 3, 31, 0, 0)  # inclusive
# NEMOSIS `end_time` is exclusive, so ask for one extra day
PULL_END = END_TS + timedelta(days=1)

# Columns that hold cleared volumes (MW or MW-equiv.)
FCAS_COLS = [
    "RAISE6SEC", "RAISE60SEC", "RAISE5MIN", "RAISEREG",
    "LOWER6SEC", "LOWER60SEC", "LOWER5MIN", "LOWERREG"
]
NUMERIC_COLS = ["TOTALCLEARED"] + FCAS_COLS

# ──────────────────────────────────────────────────────────────────────────
# 1.  Build a list of monthly [start, end) windows for pulling
# ──────────────────────────────────────────────────────────────────────────
windows = []
cursor = START_TS.replace(day=1, hour=0, minute=0)
while cursor < PULL_END:
    next_month = cursor + relativedelta(months=1)
    windows.append((cursor, next_month))
    cursor = next_month

# ──────────────────────────────────────────────────────────────────────────
# 2.  Pull DISPATCHLOAD chunk-by-chunk
# ──────────────────────────────────────────────────────────────────────────
chunks = []
for s_dt, e_dt in windows:
    s_str = s_dt.strftime("%Y/%m/%d %H:%M:%S")
    e_str = e_dt.strftime("%Y/%m/%d %H:%M:%S")
    print(f"Fetching DISPATCHLOAD {s_str} → {e_str} …")
    df = dynamic_data_compiler(
        table_name="DISPATCHLOAD",
        start_time=s_str,
        end_time=e_str,
        raw_data_location=RAW_CACHE,
        keep_csv=True
    )

    # Keep only the 18 units of interest
    df = df[df["DUID"].isin(DUIDS)]
    # Coerce numeric columns to float32 for memory efficiency
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    chunks.append(df[["SETTLEMENTDATE", "DUID"] + NUMERIC_COLS])

# ──────────────────────────────────────────────────────────────────────────
# 3.  Concatenate & final tidy-up
# ──────────────────────────────────────────────────────────────────────────
raw = pd.concat(chunks, ignore_index=True)

# Restrict to the exact [START_TS, END_TS] inclusive window
mask = (raw["SETTLEMENTDATE"] >= START_TS) & (raw["SETTLEMENTDATE"] <= END_TS)
raw = raw.loc[mask]

# Aggregate by 5-minute interval and DUID (summing in the very rare case of duplicates)
tidy = (
    raw.groupby(["SETTLEMENTDATE", "DUID"], as_index=False)
        .sum(numeric_only=True)
        .rename(columns={"TOTALCLEARED": "ENERGY"})
)

# Optional: sort rows for easier reading
tidy = tidy.sort_values(["SETTLEMENTDATE", "DUID"]).reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────────
# 4.  Save
# ──────────────────────────────────────────────────────────────────────────
out_csv = "dispatchload_unit_energy_fcas_201910-202003.csv"
tidy.to_csv(out_csv, index=False)
print(f"\n✓ Saved {len(tidy):,} rows to {out_csv}")
