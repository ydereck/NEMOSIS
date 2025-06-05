"""
run_stacked_logit_vol.py   –  SA, Nov‑2019 → Mar‑2020
Stacked logit diff‑in‑diff with market dummies, 30‑day revenue share
**plus 24‑hour realised volatility (σ) of each market’s price**
-------------------------------------------------------------------
Folder layout (relative to this file)
  LoadData/
      dispatchload_unit_energy_fcas_201910-202003.csv
  price_forecast/
      actual_forecast_2019.csv
      actual_forecast_2020.csv
  price_bands/
      bidperoffer_<month>.csv  (Nov‑2019 … Mar‑2020)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit

# ------------------------------------------------------------------
# 0.  parameters & constants
# ------------------------------------------------------------------
REGION   = "SA1"
START    = "2019-11-01 00:00:00"
END      = "2020-03-31 23:55:00"
VOL_WIN  = 288        # 24 h = 288×5‑min intervals

DUIDS = [
    "HPRG1","HPRL1","DALNTH01","DALNTHL1","LBBG1","LBBL1",
    "TORRB1","TORRB2","TORRB3","TORRB4","PPCCGT",
    "QPS1","QPS2","QPS3","QPS4","QPS5",
    "BARKIPS1","AGLHAL","OSB-AG"
]

MARKETS = ["ENERGY",
           "RAISE6SEC","RAISE60SEC","RAISE5MIN","RAISEREG",
           "LOWER6SEC","LOWER60SEC","LOWER5MIN","LOWERREG"]

# ------------------------------------------------------------------
# 1.  dispatched MW (energy + FCAS)    ------------------------------
# ------------------------------------------------------------------
load_path = Path("LoadData/dispatchload_unit_energy_fcas_201910-202003.csv")
load = (pd.read_csv(load_path, parse_dates=["SETTLEMENTDATE"])\
          .rename(columns={"SETTLEMENTDATE":"INTERVAL"}))
load = load[load["DUID"].isin(DUIDS)]
load.sort_values(["DUID","INTERVAL"], inplace=True)

# ------------------------------------------------------------------
# 2.  prices & forecasts  -------------------------------------------
# ------------------------------------------------------------------

def prep_price(fname: str):
    df = pd.read_csv(fname, parse_dates=["SETTLEMENTDATE"])
    return df[df["REGIONID"] == REGION]

price_full = pd.concat(
    [prep_price("price_forecast/actual_forecast_2019.csv"),
     prep_price("price_forecast/actual_forecast_2020.csv")],
    ignore_index=True)
price_full.rename(columns={"SETTLEMENTDATE": "INTERVAL"}, inplace=True)
price_full.sort_values("INTERVAL", inplace=True)

# realised volatility (24‑hour rolling std of price *levels*)
for m in MARKETS:
    col = "RRP" if m == "ENERGY" else f"{m}RRP"
    price_full[f"sigma_{m}"] = (price_full[col]
         .rolling(VOL_WIN, min_periods=VOL_WIN//4).std())

# absolute FE and ln|FE|  (needed later)
for m in MARKETS:
    act = "RRP" if m == "ENERGY" else f"{m}RRP"
    fc  = "FC_RRP" if m == "ENERGY" else f"FC_{m}RRP"
    price_full[f"lnFE_{m}"] = np.log((price_full[act]-price_full[fc]).abs()+1e-3)

price = price_full[["INTERVAL"] + [f"lnFE_{m}" for m in MARKETS] +
                   [f"sigma_{m}" for m in MARKETS]].copy()

# ------------------------------------------------------------------
# 2-bis.  summary table: prices for ENERGY + 8 FCAS  ----------------
#     (insert right after price_full.sort_values(...)
# ------------------------------------------------------------------
# keep only the rows inside the study window
mask = (price_full["INTERVAL"] >= START) & (price_full["INTERVAL"] <= END)
price_in_win = price_full.loc[mask].copy()

stats = []
for m in MARKETS:
    col = "RRP" if m == "ENERGY" else f"{m}RRP"
    s = price_in_win[col]
    stats.append(
        s.agg(["mean", "std", "min", "median", "max"]).rename(m)
    )

price_stats = pd.DataFrame(stats)
price_stats.index.name = "Market"
print("\n=== price summary over sample ===")
print(price_stats.round(2))
price_stats.to_csv("price_summary_markets.csv")
print("Saved to 'price_summary_markets.csv'")


# ------------------------------------------------------------------
# 3.  bid‑change flags  ---------------------------------------------
# ------------------------------------------------------------------

def read_month(path: Path):
    df = pd.read_csv(path, parse_dates=["INTERVAL_DATETIME"])
    return df[df["DUID"].isin(DUIDS)]

frames = []
for m in ["november2019","december2019",
          "january2020","february2020","march2020"]:
    frames.append(read_month(Path(f"price_bands/bidperoffer_{m}.csv")))

bids = pd.concat(frames, ignore_index=True)
avail = [f"BANDAVAIL{i}" for i in range(1, 11)]
bids[avail] = bids[avail].apply(pd.to_numeric, errors="coerce")
bids.rename(columns={"INTERVAL_DATETIME":"INTERVAL","BIDTYPE":"Market"}, inplace=True)
bids = bids[bids["Market"].isin(MARKETS)]
bids.sort_values(["DUID","Market","INTERVAL"], inplace=True)

bids["Bid_change"] = (bids.groupby(["DUID","Market"])[avail]
                           .diff().ne(0).any(axis=1).fillna(0).astype(int))
# shift −2 for response at t+2
bids["ATTN_t2"] = bids.groupby(["DUID","Market"])["Bid_change"].shift(-2)
bids.dropna(subset=["ATTN_t2"], inplace=True)

# crop to analysis window
START, END = pd.Timestamp(START), pd.Timestamp(END)
mask_intvl = lambda df: df[(df["INTERVAL"] >= START) & (df["INTERVAL"] <= END)]
load, price, bids = map(mask_intvl, (load, price, bids))

# ------------------------------------------------------------------
# 4.  merge ln|FE| & volatility  ------------------------------------
# ------------------------------------------------------------------
panel = bids.merge(price, on="INTERVAL", how="left")

panel["lnFE_use"] = panel.apply(lambda r: r[f"lnFE_{r['Market']}"], axis=1)
panel["sigma"]     = panel.apply(lambda r: r[f"sigma_{r['Market']}"] , axis=1)
panel["lnSigma"]   = np.log(panel["sigma"].clip(lower=1e-3))

# trim top‑1 % extreme ln|FE| by market
p99 = panel.groupby("Market")["lnFE_use"].transform(lambda s: s.quantile(0.99))
panel = panel[panel["lnFE_use"] <= p99]

# ------------------------------------------------------------------
# 5.  30‑day revenue share per market  ------------------------------
# ------------------------------------------------------------------
act_cols = {"ENERGY":"RRP", **{m:f"{m}RRP" for m in MARKETS if m!="ENERGY"}}
rev_parts = []
for m, col in act_cols.items():
    sub = load[["INTERVAL","DUID", m]].merge(price_full[["INTERVAL", col]], on="INTERVAL")
    sub["Revenue"], sub["Market"] = sub[m]*sub[col]*5/60, m
    rev_parts.append(sub[["INTERVAL","DUID","Market","Revenue"]])
rev = pd.concat(rev_parts, ignore_index=True)
rev.sort_values(["DUID","Market","INTERVAL"], inplace=True)
rev["RollRev"] = rev.groupby(["DUID","Market"])['Revenue'].transform(lambda s: s.rolling(8640, min_periods=1).sum())
rev_total = rev.groupby(["DUID","INTERVAL"])['RollRev'].transform('sum')
rev['Share30'] = (rev['RollRev']/rev_total).fillna(0)

panel = panel.merge(rev[["INTERVAL","DUID","Market","Share30"]], on=["INTERVAL","DUID","Market"], how="left")

# ------------------------------------------------------------------
# 6.  capacity, dummies, interactions  ------------------------------
# ------------------------------------------------------------------
cap = load.groupby("DUID")["ENERGY"].max().reset_index(name="MAXCAP")
panel = panel.merge(cap, on="DUID", how="left")
panel["logCap"] = np.where(panel["MAXCAP"]>0, np.log(panel["MAXCAP"]), 0)

# hour FE
panel["hour"] = panel["INTERVAL"].dt.hour
hour_dum = pd.get_dummies(panel["hour"], prefix="h", drop_first=True, dtype="uint8")

# market dummies (ENERGY baseline)
m_dum = pd.get_dummies(panel["Market"], prefix="M", drop_first=True, dtype="uint8")
panel = pd.concat([panel, m_dum], axis=1)

# base continuous interactions
panel["lnFE:Share"]      = panel["lnFE_use"] * panel["Share30"]
panel["lnSigma:Share"]   = panel["lnSigma"]   * panel["Share30"]
panel["lnSigma:lnFE"]    = panel["lnSigma"]   * panel["lnFE_use"]

# market‑specific interactions
for col in m_dum.columns:
    panel[f"{col}:lnSigma"]        = panel[col] * panel["lnSigma"]
    panel[f"{col}:lnSigma:Share"]  = panel[col] * panel["lnSigma:Share"]
    panel[f"{col}:lnFE"]           = panel[col] * panel["lnFE_use"]
    panel[f"{col}:Share"]          = panel[col] * panel["Share30"]
    panel[f"{col}:lnFE:Share"]     = panel[col] * panel["lnFE:Share"]

# float32 cast
num_cols = ["lnFE_use","Share30","lnFE:Share","lnSigma","lnSigma:Share","lnSigma:lnFE","logCap"]
panel[num_cols] = panel[num_cols].astype("float32")

# ------------------------------------------------------------------
# 7.  build design matrix & run logit  -------------------------------
# ------------------------------------------------------------------
X_parts = [
    panel[["lnFE_use","Share30","lnFE:Share",
           "lnSigma","lnSigma:Share","lnSigma:lnFE","logCap"]],
    panel.filter(like="M_"),
    panel.filter(like=":lnFE"),
    panel.filter(like=":Share"),
    panel.filter(like=":lnSigma"),
    hour_dum,
]
X = pd.concat(X_parts, axis=1, copy=False)
X.insert(0, "const", 1.0, allow_duplicates=False)
X = X.loc[:, ~X.columns.duplicated()].astype("float32", copy=False)

y = panel["ATTN_t2"].astype(int)

# drop rows with NaNs / infs in X or y
mask = X.replace([np.inf,-np.inf], np.nan).dropna().index
X, y = X.loc[mask], y.loc[mask]

# cluster by interval
stacked = Logit(y, X).fit(cov_type="cluster", cov_kwds={"groups": panel.loc[mask, "INTERVAL"]}, disp=False)

print(stacked.summary())

pd.DataFrame({
    "coef": stacked.params,
    "std_err": stacked.bse,
    "z": stacked.tvalues,
    "p_val": stacked.pvalues
}).round(4).to_csv("stacked_logit_coeffs_vol.csv")
print("Saved to 'stacked_logit_coeffs_vol.csv'")
