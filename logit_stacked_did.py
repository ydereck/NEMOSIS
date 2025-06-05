"""
run_stacked_logit.py   –  SA, Nov-2019 → Mar-2020
   stacked logit diff-in-diff with market dummies +
   30-day market-specific revenue share

folder layout (relative to this file)
  LoadData/
      dispatchload_unit_energy_fcas_201910-202003.csv
  price_forecast/
      actual_forecast_2019.csv
      actual_forecast_2020.csv
  price_bands/
      bidperoffer_november2019.csv … bidperoffer_march2020.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit

# -------------------------------------------------------------------
# 0.  parameters & constants
# -------------------------------------------------------------------
REGION  = "SA1"
START   = "2019-11-01 00:00:00"
END     = "2020-03-31 23:55:00"

DUIDS = [
    "HPRG1", "HPRL1", "DALNTH01", "DALNTHL1",
    "LBBG1", "LBBL1",
    "TORRB1", "TORRB2", "TORRB3", "TORRB4",
    "PPCCGT",
    "QPS1", "QPS2", "QPS3", "QPS4", "QPS5",
    "BARKIPS1", "AGLHAL", "OSB-AG"
]

BATTERIES = {"HPRG1","HPRL1","DALNTH01","DALNTHL1","LBBG1","LBBL1"}
MARKETS   = ["ENERGY",
             "RAISE6SEC","RAISE60SEC","RAISE5MIN","RAISEREG",
             "LOWER6SEC","LOWER60SEC","LOWER5MIN","LOWERREG"]

# -------------------------------------------------------------------
# 1.  load dispatched-MW (energy + 8 FCAS) ---------------------------
# -------------------------------------------------------------------
load_path = Path("LoadData/dispatchload_unit_energy_fcas_201910-202003.csv")
load = (pd.read_csv(load_path, parse_dates=["SETTLEMENTDATE"])
          .rename(columns={"SETTLEMENTDATE":"INTERVAL"}))
load = load[load["DUID"].isin(DUIDS)]
load.sort_values(["DUID","INTERVAL"], inplace=True)

# -------------------------------------------------------------------
# 2.  load prices & forecast → |FE| ----------------------------------
# -------------------------------------------------------------------
def prep_price(fname):
    df = pd.read_csv(fname, parse_dates=["SETTLEMENTDATE"])
    return df[df["REGIONID"]==REGION]

price = pd.concat([prep_price("price_forecast/actual_forecast_2019.csv"),
                   prep_price("price_forecast/actual_forecast_2020.csv")],
                  ignore_index=True)

price_full = pd.concat([prep_price("price_forecast/actual_forecast_2019.csv"),
                        prep_price("price_forecast/actual_forecast_2020.csv")],
                       ignore_index=True)
price_full.rename(columns={"SETTLEMENTDATE": "INTERVAL"}, inplace=True)

price = price_full.copy()
for m in MARKETS:
    act = "RRP" if m == "ENERGY" else f"{m}RRP"
    fc  = "FC_RRP" if m == "ENERGY" else f"FC_{m}RRP"
    price[f"AbsFE_{m}"] = (price[act] - price[fc]).abs()+1
    price[f"lnFE_{m}"]  = np.log(price[f"AbsFE_{m}"].clip(lower=1e-3))
price = price[["INTERVAL"] + [f"lnFE_{m}" for m in MARKETS]]

# -------------------------------------------------------------------
# 3.  bid-change flags (ENERGY + 8 FCAS) -----------------------------
# -------------------------------------------------------------------
def read_month(path):
    df = pd.read_csv(path, parse_dates=["INTERVAL_DATETIME"])
    return df[df["DUID"].isin(DUIDS)]

frames=[]
for m in ["november2019","december2019",
          "january2020","february2020","march2020"]:
    frames.append(read_month(Path(f"price_bands/bidperoffer_{m}.csv")))

bids = pd.concat(frames, ignore_index=True)
avail = [f"BANDAVAIL{i}" for i in range(1,11)]
bids[avail] = bids[avail].apply(pd.to_numeric, errors="coerce")
bids.rename(columns={"INTERVAL_DATETIME":"INTERVAL","BIDTYPE":"Market"},
            inplace=True)
bids = bids[bids["Market"].isin(MARKETS)]
bids.sort_values(["DUID","Market","INTERVAL"], inplace=True)

diff = bids.groupby(["DUID","Market"])[avail].diff()
bids["Bid_change"] = diff.ne(0).any(axis=1).fillna(0).astype(int)

# lag –2 → response at τ+2
bids["ATTN_t2"] = (bids.groupby(["DUID","Market"])["Bid_change"]
                        .shift(-2))
bids = bids.dropna(subset=["ATTN_t2"])

# Define start/end as Timestamps
START = pd.Timestamp(START)
END = pd.Timestamp(END)

# Apply to all time-series datasets
load = load[(load["INTERVAL"] >= START) & (load["INTERVAL"] <= END)]
price = price[(price["INTERVAL"] >= START) & (price["INTERVAL"] <= END)]
bids = bids[(bids["INTERVAL"] >= START) & (bids["INTERVAL"] <= END)]


# -------------------------------------------------------------------
# 4.  merge ln|FE| ---------------------------------------------------
# -------------------------------------------------------------------
panel = bids.merge(price, on="INTERVAL", how="left")

def pick_lnfe(row):
    return row[f"lnFE_{row['Market']}"]
panel["lnFE_use"] = panel.apply(pick_lnfe, axis=1)

# -------------------------------------------------------------------
# 4-bis.  trim the 1 % most extreme forecast-error observations
#         (top 1 % of ln|FE| in each market OR overall)
# -------------------------------------------------------------------
# OPTION A – single overall cut-off  (simple, one line)
# p99 = panel["lnFE_use"].quantile(0.99)
# panel = panel[panel["lnFE_use"] <= p99]

# OPTION B – market-specific cut-off  (safer if distributions differ)
p99_by_mkt = (panel.groupby("Market")["lnFE_use"]
                      .transform(lambda s: s.quantile(0.99)))
panel = panel[panel["lnFE_use"] <= p99_by_mkt]

# -------------------------------------------------------------------
# 5.  30-day revenue share per market -------------------------------
# -------------------------------------------------------------------
# per-interval revenue
price_long = pd.melt(price, id_vars=["INTERVAL"],
                     value_vars=[f"lnFE_{m}" for m in MARKETS])  # dummy melt just for INTERVAL list
act_cols = {"ENERGY": "RRP",
            **{m: f"{m}RRP" for m in MARKETS if m != "ENERGY"}}

rev_parts = []
for m, col in act_cols.items():
    sub = (load[["INTERVAL", "DUID", m]]
           .merge(price_full[["INTERVAL", col]], on="INTERVAL"))
    sub["Revenue"] = sub[m] * sub[col] * 5 / 60
    sub["Market"]  = m
    rev_parts.append(sub[["INTERVAL", "DUID", "Market", "Revenue"]])

rev = pd.concat(rev_parts, ignore_index=True)
rev.sort_values(["DUID","Market","INTERVAL"], inplace=True)

# rolling 30-day (30*288=8640)
roll = (rev.groupby(["DUID","Market"])["Revenue"]
            .transform(lambda s: s.rolling(8640, min_periods=1).sum()))
rev["RollRev"] = roll

total = (rev.groupby(["DUID","INTERVAL"])["RollRev"]
             .transform("sum"))
rev["Share30"] = rev["RollRev"] / total
rev["Share30"] = rev["Share30"].fillna(0)

panel = panel.merge(rev[["INTERVAL","DUID","Market","Share30"]],
                    on=["INTERVAL","DUID","Market"], how="left")

# -------------------------------------------------------------------
# 6.  capacity, dummies, interactions -------------------------------
# -------------------------------------------------------------------

cap = load.groupby("DUID")["ENERGY"].max().reset_index(name="MAXCAP")
panel = panel.merge(cap, on="DUID", how="left")
panel["logCap"] = np.where(panel["MAXCAP"]>0,
                           np.log(panel["MAXCAP"]),0)

# hour FE
panel["hour"] = panel["INTERVAL"].dt.hour
hour_dum = pd.get_dummies(panel["hour"], prefix="h", drop_first=True, dtype="uint8")

# market dummies (Energy baseline)
m_dum = pd.get_dummies(panel["Market"], prefix="M", drop_first=True, dtype="uint8")
panel = pd.concat([panel, m_dum], axis=1)

# interactions
for col in m_dum.columns:                      # M_RAISE6SEC …
    panel[f"{col}:lnFE"]   = (panel[col]*panel["lnFE_use"]).astype("float32")
    panel[f"{col}:Share"]  = (panel[col]*panel["Share30"]).astype("float32")
    panel[f"{col}:lnFE:Share"] = (panel[col]*panel["lnFE_use"]*panel["Share30"]).astype("float32")

panel["lnFE:Share"] = panel["lnFE_use"]*panel["Share30"]

# list every continuous column you want in 32-bit floats
num_cols = ["lnFE_use", "Share30", "lnFE:Share", "logCap"]
panel[num_cols] = panel[num_cols].astype("float32")


# design matrix
# X = sm.add_constant(
#         pd.concat([panel[["lnFE_use","Share30","lnFE:Share","logCap"]],
#                    panel.filter(like="M_"),
#                    panel.filter(like=":lnFE"),
#                    panel.filter(like=":Share"),
#                    panel.filter(like=":lnFE:Share"),
#                    hour_dum], axis=1)
#     )
X_parts = [
    panel[["lnFE_use", "Share30", "lnFE:Share", "logCap"]],
    panel.filter(like="M_"),
    panel.filter(like=":lnFE"),
    panel.filter(like=":Share"),
    panel.filter(like=":lnFE:Share"),
    hour_dum
]
# X_parts = [
#     panel[["lnFE_use", "Share30", "lnFE:Share", "logCap"]]
# ]
X = pd.concat(X_parts, axis=1, copy=False)          # no deep copy
X.insert(0, "const", 1.0, allow_duplicates=False)   # manual intercept
X = X.astype("float32", copy=False)                 # keep it light
y = panel["ATTN_t2"]

# drop any residual NaNs / inf
mask = X.replace([np.inf,-np.inf], np.nan).dropna().index
X, y = X.loc[mask], y.loc[mask]
groups = panel.loc[mask,"INTERVAL"]
# %%
# keep only the first occurrence of each column name
X = X.loc[:, ~X.columns.duplicated()]


stacked = sm.Logit(y, X).fit(cov_type="cluster",
                         cov_kwds={"groups": groups},
                         disp=False)



# -------------------------------------------------------------------
# 7.  estimate logit -----------------------------------------------
# -------------------------------------------------------------------
# stacked = Logit(y, X).fit(
#             cov_type="cluster",
#             cov_kwds={"groups": groups},
#             disp=False)
print(stacked.summary())

# Create a DataFrame with coefficients and standard errors
coef_df = pd.DataFrame({
    "coef": stacked.params,
    "std_err": stacked.bse,
    "z": stacked.tvalues,
    "p_value": stacked.pvalues
})

# Optionally round for readability
coef_df = coef_df.round(4)

# Save to CSV
coef_df.to_csv("stacked_logit_coeffs_99th_B.csv")
print("\nCoefficient table saved to 'stacked_logit_coeffs.csv'")

