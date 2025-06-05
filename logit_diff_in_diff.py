"""
run_diff_in_diff.py
SA batteries vs. thermal generators, Nov-2019 – Mar-2020
Menu-Cost / Rational-Inattention tests
---------------------------------------------------------
folder layout (relative to this script):
  LoadData/dispatchload_unit_energy_fcas_201911-202003.csv
  price_forecast/actual_forecast_2019.csv
  price_forecast/actual_forecast_2020.csv
  price_bands/bidperoffer_<month>.csv   (Nov-2019 … Mar-2020)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.discrete.discrete_model import Logit
import statsmodels.api as sm

# -------------------------------------------------------------------
# 0.  PARAMETERS & CONSTANTS
# -------------------------------------------------------------------
REGION       = "SA1"
START, END   = "2019-11-01 00:00:00", "2020-03-31 23:55:00"
DATA_DIR     = Path("LoadData")    # adjust if needed

DUIDS = [
    "HPRG1", "HPRL1", "DALNTH01", "DALNTHL1",
    "LBBG1", "LBBL1",
    "TORRB1", "TORRB2", "TORRB3", "TORRB4",
    "PPCCGT", "QPS1",  "QPS2",   "QPS3",  "QPS4", "QPS5",
    "BARKIPS1", "AGLHAL", "OSB-AG"
]
BATTERIES = {"HPRG1","HPRL1","DALNTH01","DALNTHL1","LBBG1","LBBL1"}

ENERGY_COL   = "RRP"
FORECAST_PRE = "FC_"

FCAS_MARKETS = ["RAISE6SEC","RAISE60SEC","RAISE5MIN","RAISEREG",
                "LOWER6SEC","LOWER60SEC","LOWER5MIN","LOWERREG"]

# -------------------------------------------------------------------
# 1.  LOAD DISPATCHED MW  (energy + FCAS, SA only)
# -------------------------------------------------------------------
load_file = Path("LoadData")/"dispatchload_unit_energy_fcas_201911_202003.csv"
load = (pd.read_csv(load_file, parse_dates=["SETTLEMENTDATE"])
          .rename(columns={"SETTLEMENTDATE":"INTERVAL"}))
load = load[load["DUID"].isin(DUIDS)].copy()
load.sort_values(["DUID","INTERVAL"], inplace=True)

# -------------------------------------------------------------------
# 2.  LOAD PRICES  (actual + PD-5 forecast)  -------------------------
# -------------------------------------------------------------------
price19 = pd.read_csv(Path("price_forecast")/"actual_forecast_2019.csv",
                      parse_dates=["SETTLEMENTDATE"])
price20 = pd.read_csv(Path("price_forecast")/"actual_forecast_2020.csv",
                      parse_dates=["SETTLEMENTDATE"])
price = pd.concat([price19, price20], ignore_index=True)
price = price[price["REGIONID"]==REGION].rename(columns={"SETTLEMENTDATE":"INTERVAL"})

# keep actual + forecast columns
keep_cols = (["INTERVAL"]
             + [ENERGY_COL] + [FORECAST_PRE+ENERGY_COL]
             + [m+"RRP" for m in FCAS_MARKETS]
             + [FORECAST_PRE+m+"RRP" for m in FCAS_MARKETS])
price = price[keep_cols]

# absolute forecast error
price["Abs_FE_E"] = (price[ENERGY_COL] - price[FORECAST_PRE+ENERGY_COL]).abs()
for m in FCAS_MARKETS:
    price[f"Abs_FE_{m}"] = (price[m+"RRP"] - price[FORECAST_PRE+m+"RRP"]).abs()

# -------------------------------------------------------------------
# 3.  BUILD BID-CHANGE FLAGS  ----------------------------------------
# -------------------------------------------------------------------
def build_bidflags(folder: Path, months: list[str])->pd.DataFrame:
    out=[]
    avail_cols=[f"BANDAVAIL{i}" for i in range(1,11)]
    for m in months:
        f=folder/f"bidperoffer_{m}.csv"
        df=pd.read_csv(f, parse_dates=["INTERVAL_DATETIME"])
        df=df[df["DUID"].isin(DUIDS)]
        df=df[["INTERVAL_DATETIME","DUID","BIDTYPE"]+avail_cols]
        df.rename(columns={"INTERVAL_DATETIME":"INTERVAL"}, inplace=True)
        out.append(df)
    bids=pd.concat(out, ignore_index=True)
    bids.sort_values(["DUID","BIDTYPE","INTERVAL"], inplace=True)

    # mark participants: appear at least once in month
    participants = bids.groupby(["DUID","BIDTYPE"]).size().reset_index()
    bids = bids.merge(participants[["DUID","BIDTYPE"]], on=["DUID","BIDTYPE"], how="left")

    # diff across intervals
    diff = bids.groupby(["DUID","BIDTYPE"])[avail_cols].diff().abs()
    bids["Bid_change"] = diff.gt(0).any(axis=1).astype(int)
    return bids[["INTERVAL","DUID","BIDTYPE","Bid_change"]]

months = ["november2019","december2019",
          "january2020","february2020","march2020"]
bidflags = build_bidflags(Path("price_bands"), months)

# align markets naming
bidflags["Market"] = bidflags["BIDTYPE"].replace({"ENERGY":"ENERGY"})
# drop markets not Energy or FCAS of interest
valid_markets = ["ENERGY"] + FCAS_MARKETS
bidflags = bidflags[bidflags["Market"].isin(valid_markets)]

# -------------------------------------------------------------------
# 4.  MERGE (error at τ) with (Bid_change at τ+2) --------------------
# -------------------------------------------------------------------
# merge price errors on τ
panel = bidflags.merge(price, on="INTERVAL", how="left")

# shift bid_change -2 intervals backward so it becomes τ+2
panel.sort_values(["DUID","Market","INTERVAL"], inplace=True)
panel["ATTN_t2"] = (panel.groupby(["DUID","Market"])["Bid_change"]
                          .shift(-2))
panel = panel.dropna(subset=["ATTN_t2"])

# attach capacity (use max observed MW in period)
cap = load.groupby("DUID")["ENERGY"].max().reset_index(name="MAXCAP")
panel = panel.merge(cap, on="DUID", how="left")
panel["logCap"] = np.log(panel["MAXCAP"].replace(0,np.nan)).fillna(0)

# attach Battery dummy
panel["Battery"] = panel["DUID"].isin(BATTERIES).astype(int)

# -------------------------------------------------------------------
# 5.  BUILD ShareE (rolling 24h)  ------------------------------------
# -------------------------------------------------------------------
# revenue by interval
price_cols = {"ENERGY":"RRP", **{m:m+"RRP" for m in FCAS_MARKETS}}
rev_frames=[]
for m,label in price_cols.items():
    sub = (load[["INTERVAL","DUID",m]]
           .merge(price[["INTERVAL",label]], on="INTERVAL"))
    sub["rev"] = sub[m]*sub[label]*5/60
    sub = sub[["INTERVAL","DUID","rev"]].rename(columns={"rev":f"rev_{m}"})
    rev_frames.append(sub)

rev = rev_frames[0]
for f in rev_frames[1:]:
    rev = rev.merge(f, on=["INTERVAL","DUID"], how="outer")
rev.fillna(0, inplace=True)

rev["rev_E"]   = rev["rev_ENERGY"]
rev["rev_F"]   = rev[[f"rev_{m}" for m in FCAS_MARKETS]].sum(axis=1)

# rolling 288-interval sums
rev.sort_values(["DUID","INTERVAL"], inplace=True)
rev["Roll_E"] = (rev.groupby("DUID")["rev_E"]
                   .transform(lambda s: s.rolling(288, min_periods=1).sum()))
rev["Roll_F"] = (rev.groupby("DUID")["rev_F"]
                   .transform(lambda s: s.rolling(288, min_periods=1).sum()))
rev["ShareE"] = rev["Roll_E"] / (rev["Roll_E"]+rev["Roll_F"])
rev.loc[(rev["Roll_E"]+rev["Roll_F"])==0,"ShareE"] = 0.5

# lag by 1 interval
rev["ShareE_lag"] = rev.groupby("DUID")["ShareE"].shift(1)
panel = panel.merge(rev[["INTERVAL","DUID","ShareE_lag"]], on=["INTERVAL","DUID"], how="left")

# -------------------------------------------------------------------
# 6.  ENERGY-ONLY LOGIT  (Menu-Cost / RI) ---------------------------
# -------------------------------------------------------------------
energy = panel[panel["Market"]=="ENERGY"].copy()

# Trim top 1 % extreme |forecast error|
P99 = energy["Abs_FE_E"].quantile(0.99)
energy = energy[energy["Abs_FE_E"] <= P99]

SHIFT = 1                          # > |minimum allowed price| + 1
energy["lnAbs_FE_E"] = np.log(energy["Abs_FE_E"] + SHIFT)

energy["Battery_FE"] = energy["Battery"]*energy["Abs_FE_E"]
energy["lnBattery_FE"] = energy["Battery"]*energy["lnAbs_FE_E"]

X_E = sm.add_constant(
        energy[["lnAbs_FE_E","Battery","lnBattery_FE", "logCap"]]
      )
y_E = energy["ATTN_t2"]

model_E = Logit(y_E, X_E)
res_E = model_E.fit(disp=False,
                    cov_type="cluster",
                    cov_kwds={"groups": energy["INTERVAL"]})
print("\n=== ENERGY-ONLY LOGIT (Menu-cost / RI) ===")
print(res_E.summary())
