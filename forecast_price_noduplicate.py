#!/usr/bin/env python3
"""
Fetch and compare 2021 pre-dispatch forecast prices vs. actual dispatch prices
for energy and all FCAS markets, saving deduplicated outputs under a "price_forecast" folder.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import nemosis
from nemseer import download_raw_data, generate_runtimes, compile_data

# ─── 1. Setup ────────────────────────────────────────────────────────────────

# Silence verbose logs
logging.getLogger("nemosis").setLevel(logging.WARNING)
logging.getLogger("nemseer").setLevel(logging.ERROR)

# Analysis window: calendar year 2021
ANALYSIS_START = "2021/01/01 00:00:00"
ANALYSIS_END   = "2022/01/01 00:00:00"

# Local cache directories and output folder
NEMOSIS_CACHE = Path("nemosis_cache")
NEMSEER_CACHE = Path("nemseer_cache")
OUTPUT_DIR    = Path("price_forecast")

for d in (NEMOSIS_CACHE, NEMSEER_CACHE, OUTPUT_DIR):
    d.mkdir(exist_ok=True)

# ─── 2. Cache actual dispatch prices via NEMOSIS ────────────────────────────

print("Caching actual dispatch prices (energy + FCAS)…")
actual_price = nemosis.dynamic_data_compiler(
    start_time        = ANALYSIS_START,
    end_time          = ANALYSIS_END,
    table_name        = "DISPATCHPRICE",
    raw_data_location = str(NEMOSIS_CACHE),
    select_columns    = [
        "SETTLEMENTDATE",
        "REGIONID",
        "INTERVENTION",       # must include for filtering
        "RRP",
        "RAISE6SECRRP", "RAISE60SECRRP", "RAISE5MINRRP", "RAISEREGRRP",
        "LOWER6SECRRP", "LOWER60SECRRP", "LOWER5MINRRP", "LOWERREGRRP"
    ],
    filter_cols       = ["INTERVENTION"],
    filter_values     = ([0],),   # only the base-case dispatch
    fformat           = "parquet",
    keep_csv          = True
)

out_actual = OUTPUT_DIR / "actual_dispatch_price_2021.csv"
actual_price.to_csv(out_actual, index=False)
print(f"Saved actual prices → {out_actual}")

# ─── 3. Download raw forecast price CSVs via NEMSEER ────────────────────────

print("Downloading raw forecast CSVs (PREDISPATCH PRICE & P5MIN REGIONSOLUTION)…")
download_raw_data(
    "PREDISPATCH", "PRICE", str(NEMSEER_CACHE),
    forecasted_start=ANALYSIS_START,
    forecasted_end  =ANALYSIS_END
)
download_raw_data(
    "P5MIN", "REGIONSOLUTION", str(NEMSEER_CACHE),
    forecasted_start=ANALYSIS_START,
    forecasted_end  =ANALYSIS_END
)

# ─── 4. Compile raw forecasts into DataFrames ───────────────────────────────

print("Compiling pre-dispatch (PD) price forecasts…")
pd_run_start, pd_run_end = generate_runtimes(ANALYSIS_START, ANALYSIS_END, "PREDISPATCH")
pd_price = compile_data(
    pd_run_start,
    pd_run_end,
    ANALYSIS_START,
    ANALYSIS_END,
    "PREDISPATCH",
    "PRICE",
    str(NEMSEER_CACHE)
)["PRICE"]

print("Compiling 5-minute (P5MIN) price + FCAS forecasts…")
p5_run_start, p5_run_end = generate_runtimes(ANALYSIS_START, ANALYSIS_END, "P5MIN")
p5_price = compile_data(
    p5_run_start,
    p5_run_end,
    ANALYSIS_START,
    ANALYSIS_END,
    "P5MIN",
    "REGIONSOLUTION",
    str(NEMSEER_CACHE)
)["REGIONSOLUTION"]

# ─── 5. Deduplicate so we keep only the last‐updated record per interval & region ─

# For PD forecasts: group by DATETIME & REGIONID, keep the row with max LASTCHANGED
if "LASTCHANGED" in pd_price.columns:
    pd_price = (
        pd_price
        .sort_values(["DATETIME", "REGIONID", "LASTCHANGED"],
                     ascending=[True, True, False])
        .drop_duplicates(subset=["DATETIME", "REGIONID"], keep="first")
    )

# For P5MIN forecasts: group by INTERVAL_DATETIME & REGIONID, keep max LASTCHANGED
if "LASTCHANGED" in p5_price.columns:
    p5_price = (
        p5_price
        .sort_values(["INTERVAL_DATETIME", "REGIONID", "LASTCHANGED"],
                     ascending=[True, True, False])
        .drop_duplicates(subset=["INTERVAL_DATETIME", "REGIONID"], keep="first")
    )

# ─── 6. Save deduplicated forecast tables ──────────────────────────────────

out_pd = OUTPUT_DIR / "last_forecast_predispatch_price_2021.csv"
pd_price.to_csv(out_pd, index=False)

out_p5 = OUTPUT_DIR / "last_forecast_p5min_price_2021.csv"
p5_price.to_csv(out_p5, index=False)

print(f"Saved deduplicated forecasts →\n  {out_pd}\n  {out_p5}")
