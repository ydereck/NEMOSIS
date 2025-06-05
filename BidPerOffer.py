#!/usr/bin/env python3
"""
Fetch March 2021 BIDPEROFFER_D for DUID='HPRG1', using SETTLEMENTDATE
(after AEMO’s March 2021 format change), by patching the
_data_fetch_loop so it retries with filter_on_settlementdate.
"""

import logging
from pathlib import Path
import pandas as pd

# ─── A. Monkey-patch the internal fetch loop ────────────────────────────
import nemosis.data_fetch_methods as dfm
import nemosis.filters          as flt
from nemosis.filters            import filter_on_settlementdate

# Keep a reference to the original loop
orig_loop = dfm._dynamic_data_fetch_loop

def patched_loop(*args, **kwargs):
    try:
        # First attempt with whatever date_filter was passed in args[6]
        return orig_loop(*args, **kwargs)
    except KeyError as e:
        # If it failed on INTERVAL_DATETIME, retry with settlement-date filter
        if "INTERVAL_DATETIME" in str(e):
            new_args = list(args)
            # args[6] is the date_filter positional argument
            new_args[6] = filter_on_settlementdate
            return orig_loop(*new_args, **kwargs)
        # Otherwise, re-raise
        raise

# Overwrite the internal paging‐and‐filter loop
dfm._dynamic_data_fetch_loop = patched_loop


# ─── B. (Optional) also steer the default primary_date_columns ─────────
import nemosis.defaults as defaults
defaults.primary_date_columns["BIDPEROFFER_D"] = ["SETTLEMENTDATE"]


# ─── C. Silence logs & set up paths ─────────────────────────────────────
from nemosis import dynamic_data_compiler

logging.getLogger("nemosis").setLevel(logging.WARNING)
START, END = "2021/03/01 00:00:00", "2021/04/01 00:00:00"

NEM_CACHE = Path("NEMOSIS_cache"); NEM_CACHE.mkdir(exist_ok=True)
OUT_DIR   = Path("price_bands");    OUT_DIR.mkdir(exist_ok=True)


# ─── D. Fetch the filtered table ─────────────────────────────────────────
df = dynamic_data_compiler(
    start_time        = START,
    end_time          = END,
    table_name        = "BIDPEROFFER_D",
    raw_data_location = str(NEM_CACHE),
    select_columns    = [
        "SETTLEMENTDATE", "DUID", "BIDTYPE", "PERIODID",
        "VERSIONNO", "OFFERDATE", "MAXAVAIL",
        "BANDAVAIL1", "BANDAVAIL2", "BANDAVAIL3",
        "BANDAVAIL4", "BANDAVAIL5", "BANDAVAIL6",
        "BANDAVAIL7", "BANDAVAIL8", "BANDAVAIL9",
        "BANDAVAIL10", "LASTCHANGED"
    ],
    filter_cols       = ["DUID"],
    filter_values     = (["HPRG1"],),
    fformat           = "parquet",
    keep_csv          = True,
    rebuild           = True
)


# ─── E. Save to CSV ───────────────────────────────────────────────────────
out_file = OUT_DIR / "bidperoffer_march2021.csv"
df.to_csv(out_file, index=False)
print(f"Saved {len(df)} rows → {out_file}")
