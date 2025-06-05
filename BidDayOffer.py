#!/usr/bin/env python3
"""
Download BIDDAYOFFER_D records for a list of DUIDs and save the 10 price-band
columns for each trading day between **1 Nov 2019 and 31 Mar 2020**.

Output: one CSV per calendar month in ./price_bands/, e.g.
    price_bands/biddayoffer_november2019.csv

Dependencies:
    pip install nemosis pandas
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from nemosis import dynamic_data_compiler

# ── 1  Config ──────────────────────────────────────────────────────────────
DUIDS = [
    "HPRG1", "HPRL1", "DALNTH01", "DALNTHL1",
    "LBBG1", "LBBL1",
    "TORRB1", "TORRB2", "TORRB3", "TORRB4",
    "PPCCGT", "QPS1", "QPS2", "QPS3", "QPS4", "QPS5",
    "BARKIPS1", "AGLHAL", "OSB-AG",
]

START_DATE = datetime(2019, 11, 1)
END_DATE   = datetime(2020, 3, 31, 23, 55)

TABLE_NAME = "BIDDAYOFFER_D"
CACHE_DIR  = Path("./nemosis_cache")
OUTPUT_DIR = Path("./price_bands")
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

SELECT_COLUMNS = [
    "SETTLEMENTDATE",    # Trading day date (00:00 local)
    "BIDSETTLEMENTDATE", # Same but optional
    "OFFERDATE",         # Timestamp of file submission
    "DUID",
    "BIDTYPE",           # ENERGY, RAISE6SEC …
] + [f"PRICEBAND{i}" for i in range(1, 11)]

# ── 2  Month iterator helper ──────────────────────────────────────────────

def month_iter(start: datetime, end: datetime):
    cur = start.replace(day=1, hour=0, minute=0, second=0)
    while cur <= end:
        next_month = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield cur, min(next_month - timedelta(minutes=5), end)
        cur = next_month

# ── 3  Download loop ──────────────────────────────────────────────────────
for month_start, month_end in month_iter(START_DATE, END_DATE):
    label = month_start.strftime("%B%Y").lower()  # e.g. "november2019"
    out_path = OUTPUT_DIR / f"biddayoffer_{label}.csv"

    if out_path.exists():
        print(f"⚠️  {out_path.name} exists – skip")
        continue

    print(f"Fetching {TABLE_NAME} for {label} …")
    df = dynamic_data_compiler(
        start_time        = month_start.strftime("%Y/%m/%d %H:%M:%S"),
        end_time          = month_end.strftime("%Y/%m/%d %H:%M:%S"),
        table_name        = TABLE_NAME,
        raw_data_location = str(CACHE_DIR),
        select_columns    = SELECT_COLUMNS,
        keep_csv          = True,
    )

    df = df[df["DUID"].isin(DUIDS)].reset_index(drop=True)
    print(f"Rows kept: {len(df):,}")

    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")

print("\n✅ Finished downloading all months.")
