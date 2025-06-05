"""
prelim_bid_change.py

Create "interval-to-interval bid change" summary for selected SA generators,
Nov-2019 through Mar-2020.

Assumes the folder structure:
  ./price_bands/bidperoffer_november2019.csv
  ./price_bands/bidperoffer_december2019.csv
  ./price_bands/bidperoffer_january2020.csv
  ./price_bands/bidperoffer_february2020.csv
  ./price_bands/bidperoffer_march2020.csv
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ---------------------------------------------------------------------
# 0.  Parameters -------------------------------------------------------
# ---------------------------------------------------------------------
DATA_DIR  = Path("price_bands")
MONTHS    = ["november2019", "december2019",
             "january2020", "february2020", "march2020"]
#MONTHS    = ["december2019", "january2020", "february2020", "march2020"]

DUIDS     = [
    "HPRG1", "HPRL1", "DALNTH01", "DALNTHL1",
    "LBBG1", "LBBL1",
    "TORRB1", "TORRB2", "TORRB3", "TORRB4",
    "PPCCGT",
    "QPS1", "QPS2", "QPS3", "QPS4", "QPS5",
    "BARKIPS1", "AGLHAL", "OSB-AG"
]

BATTERIES = {"HPRG1", "HPRL1", "DALNTH01", "DALNTHL1", "LBBG1", "LBBL1"}
# everything else treated as Thermal

# ---------------------------------------------------------------------
# 1.  Load & concatenate ------------------------------------------------
# ---------------------------------------------------------------------
def read_month(fname: Path) -> pd.DataFrame:
    df = pd.read_csv(fname, parse_dates=["INTERVAL_DATETIME"])
    df = df[df["DUID"].isin(DUIDS)]
    return df

frames = []
for m in MONTHS:
    fpath = DATA_DIR / f"bidperoffer_{m}.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"missing {fpath}")
    frames.append(read_month(fpath))

bids = pd.concat(frames, ignore_index=True)

# Ensure numeric columns really are numeric
avail_cols = [f"BANDAVAIL{i}" for i in range(1, 11)]
bids[avail_cols] = bids[avail_cols].apply(pd.to_numeric, errors="coerce")

# ---------------------------------------------------------------------
# 2.  Flag bid-curve changes -------------------------------------------
# ---------------------------------------------------------------------
bids.sort_values(["DUID", "BIDTYPE", "INTERVAL_DATETIME"], inplace=True)

# compare each row with previous row within DUID×BIDTYPE
diff = bids.groupby(["DUID", "BIDTYPE"])[avail_cols].diff()
bids["Bid_change"] = diff.ne(0).any(axis=1).fillna(0).astype(int)

# ---------------------------------------------------------------------
# 3.  Build summary ----------------------------------------------------
# ---------------------------------------------------------------------
total_intervals = bids["INTERVAL_DATETIME"].nunique()

summary = (bids.groupby(["DUID", "BIDTYPE"])
                 ["Bid_change"]
                 .agg(Intervals      = lambda s: total_intervals,   # same for everyone
                      Num_with_change= "sum")
                 .reset_index())

summary["Frequency"] = summary["Num_with_change"] / summary["Intervals"]

summary["Tech"] = summary["DUID"].apply(
    lambda duid: "Battery" if duid in BATTERIES else "Thermal"
)

# reorder columns
summary = summary[["Tech", "DUID", "BIDTYPE",
                   "Intervals", "Num_with_change", "Frequency"]]
summary.to_csv("bid_change_summary.csv", index=False)
print(summary.head(12))

# ---------------------------------------------------------------------
# 4 . Plot — blue vs orange
# ---------------------------------------------------------------------
mean_freq = (summary.groupby(["DUID", "Tech"])["Frequency"]
                     .mean()
                     .sort_values())          # order for barh

colors = ["tab:blue"  if tech == "Battery"
          else "tab:orange"
          for tech in mean_freq.index.get_level_values("Tech")]

plt.figure(figsize=(10, 6))
ax = mean_freq.plot(kind="barh", color=colors, edgecolor="black")

ax.set_xlabel("Bid-change frequency  (share of 5-min intervals)")
ax.set_title("Interval-to-interval bid-curve changes\nNov-2019 – Mar-2020 (SA)")

# legend
legend_elems = [Patch(facecolor="tab:blue",   label="Battery"),
                Patch(facecolor="tab:orange", label="Thermal")]
ax.legend(handles=legend_elems, loc="lower right")

plt.tight_layout()
plt.savefig("bid_change_frequency.png", dpi=150)
plt.show()
