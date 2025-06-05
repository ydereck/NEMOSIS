import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import numpy as np
from scipy import stats  # for QQ-plot

# ─── A. Locate the price_forecast folder ─────────────────────────────────
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "price_forecast"
if not data_dir.is_dir():
    print(f"ERROR: could not find folder {data_dir}", file=sys.stderr)
    sys.exit(1)

# ─── B. Find the forecast CSVs ───────────────────────────────────────────
forecast_files = sorted(
    p for p in data_dir.iterdir()
    if p.name.startswith("actual_forecast_") and p.suffix.lower() == ".csv"
)
if not forecast_files:
    print("ERROR: no forecast CSVs found!", file=sys.stderr)
    sys.exit(1)

print("Processing these files:")
for p in forecast_files:
    print("  ", p.name)
print()

# ─── C. Read, parse & concatenate ────────────────────────────────────────
frames = []
for fpath in forecast_files:
    print("Reading", fpath.name)
    df = pd.read_csv(fpath, dtype=str)

    # numeric columns
    df["RRP"]    = df["RRP"].astype(float)
    df["LAST_FC_RRP"] = df["LAST_FC_RRP"].astype(float)

    # impose strict "M/D/YYYY H:MM" format
    df["SETTLEMENTDATE"] = pd.to_datetime(
        df["SETTLEMENTDATE"].str.strip(),
        format="%m/%d/%Y %H:%M",
        errors="coerce"
    )
    before = len(df)
    df = df.dropna(subset=["SETTLEMENTDATE"])
    dropped = before - len(df)
    if dropped:
        print(f"  → dropped {dropped} rows due to timestamp parse failures")

    frames.append(df)

data = pd.concat(frames, ignore_index=True)
print(f"\nCombined shape: {data.shape}\n")

# ─── D. Adjust midnight timestamps to previous day ───────────────────────
mask_midnight = (data["SETTLEMENTDATE"].dt.hour == 0) & (data["SETTLEMENTDATE"].dt.minute == 0)
data.loc[mask_midnight, "SETTLEMENTDATE"] -= pd.Timedelta(days=1)

# ─── E. Filter SA1 & compute errors ──────────────────────────────────────
sa = data[data["REGIONID"] == "SA1"].copy()
sa["ERROR"]     = sa["RRP"] - sa["LAST_FC_RRP"]
sa["ABS_ERROR"] = sa["ERROR"].abs()

# ─── F. Group by month for summary series ────────────────────────────────
sa["MONTH"] = sa["SETTLEMENTDATE"].dt.to_period("M")
monthly_mean  = sa.groupby("MONTH")["ABS_ERROR"].mean()
monthly_p50   = sa.groupby("MONTH")["ABS_ERROR"].quantile(0.5)
monthly_p90   = sa.groupby("MONTH")["ABS_ERROR"].quantile(0.9)
monthly_price = sa.groupby("MONTH")["RRP"].mean()

# convert PeriodIndex to Timestamp for plotting
for idx in (monthly_mean, monthly_p50, monthly_p90, monthly_price):
    idx.index = idx.index.to_timestamp()

# ─── G. Plot monthly series ──────────────────────────────────────────────
plt.figure(figsize=(12, 5))
plt.plot(monthly_mean.index,  monthly_mean.values,  marker="o", label="Mean |Error|")
plt.plot(monthly_p50.index,   monthly_p50.values,   marker="o", label="Median |Error|")
plt.plot(monthly_p90.index,   monthly_p90.values,   marker="o", label="90th Pct |Error|")
plt.plot(monthly_price.index, monthly_price.values, linestyle="--", color="black",
         label="Mean Actual Price (RRP)")
plt.title("SA1 – Monthly Forecast Error & Actual Price")
plt.xlabel("Month")
plt.ylabel("Value ($/MWh)")
plt.grid(True)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# # ─── H. Q-Q plot of real forecast errors ─────────────────────────────────
# plt.figure(figsize=(8, 6))
# stats.probplot(sa["ERROR"].dropna(), dist="norm", plot=plt)
# plt.title("SA1 – Q-Q Plot of Forecast Errors (RRP − LAST_FC_RRP)")
# plt.xlabel("Theoretical Quantiles")
# plt.ylabel("Sample Quantiles")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# ─── I. Log-Scale histogram ─────────────────────────
fig, ax = plt.subplots(figsize=(10,6))
# Main histogram, linear scale
ax.hist(sa["ERROR"], bins=100, edgecolor="gray")
ax.set_title("Forecast Error Histogram – Full Range")
ax.set_xlabel("Error ($/MWh)")
ax.set_ylabel("Frequency")
# Switch to log scale on y
ax.set_yscale("log", nonpositive="clip")
# Create an inset for the tails only
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axins = inset_axes(ax, width="40%", height="30%", loc="upper right")
# show only errors beyond ±10 $/MWh
tails = sa["ERROR"][sa["ERROR"].abs() > 10]
axins.hist(tails, bins=50, edgecolor="gray")
axins.set_title("Tails: |Error|>10")
axins.set_xticks([-50, -25, 0, 25, 50])

# ─── K. Percentile table of real forecast errors ─────────────────────────
percentiles = np.arange(0.1, 1.0, 0.1)
pct_values = sa["ERROR"].quantile(percentiles)
pct_df = pd.DataFrame({
    'Percentile': [f"{int(p*100)}th" for p in percentiles],
    'Error ($/MWh)': pct_values.values
})
print("\nForecast Error Percentiles (RRP − LAST_FC_RRP):")
print(pct_df.to_string(index=False))
