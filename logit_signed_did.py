"""
SA batteries vs. thermal generators, Nov‑2019 – Mar‑2020
Menu‑Cost / Rational‑Inattention tests (signed forecast error version)
---------------------------------------------------------------------
Changes relative to the original script:
1)  Trim the 1 % most extreme |forecast‑error| observations.
2)  Use signed forecast errors with piece‑wise log(1+|FE|) terms and
    battery interactions, so slopes may differ for + / − misses and
    for batteries vs. non‑batteries.

Folder layout (relative to this script):
  LoadData/dispatchload_unit_energy_fcas_201911_202003.csv
  price_forecast/actual_forecast_2019.csv
  price_forecast/actual_forecast_2020.csv
  price_bands/bidperoffer_<month>.csv   (Nov‑2019 … Mar‑2020)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit

# -------------------------------------------------------------------
# 0.  PARAMETERS & CONSTANTS
# -------------------------------------------------------------------
REGION       = "SA1"
START, END   = "2019-11-01 00:00:00", "2020-03-31 23:55:00"
DATA_DIR     = Path("LoadData")

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

FCAS_MARKETS = [
    "RAISE6SEC","RAISE60SEC","RAISE5MIN","RAISEREG",
    "LOWER6SEC","LOWER60SEC","LOWER5MIN","LOWERREG"
]

MONTH_FILES = [
    "november2019","december2019",
    "january2020","february2020","march2020"
]

# -------------------------------------------------------------------
# 1.  LOAD DISPATCHED MW  (energy + FCAS, SA only)
# -------------------------------------------------------------------
load_file = DATA_DIR/"dispatchload_unit_energy_fcas_201911_202003.csv"
load = (pd.read_csv(load_file, parse_dates=["SETTLEMENTDATE"])
          .rename(columns={"SETTLEMENTDATE":"INTERVAL"}))
load = load[load["DUID"].isin(DUIDS)].copy()
load.sort_values(["DUID","INTERVAL"], inplace=True)

# -------------------------------------------------------------------
# 2.  LOAD PRICES  (actual + PD‑5 forecast)  -------------------------
# -------------------------------------------------------------------
price19 = pd.read_csv(Path("price_forecast")/"actual_forecast_2019.csv",
                      parse_dates=["SETTLEMENTDATE"])
price20 = pd.read_csv(Path("price_forecast")/"actual_forecast_2020.csv",
                      parse_dates=["SETTLEMENTDATE"])
price = pd.concat([price19, price20], ignore_index=True)
price = price[price["REGIONID"]==REGION].rename(columns={"SETTLEMENTDATE":"INTERVAL"})

# keep actual + forecast columns we need
keep_cols = ["INTERVAL", ENERGY_COL, FORECAST_PRE+ENERGY_COL]
price = price[keep_cols]

# forecast errors (absolute and signed)
price["Abs_FE_E"] = (price[ENERGY_COL] - price[FORECAST_PRE+ENERGY_COL]).abs()
price["FE_E"]      = price[ENERGY_COL] - price[FORECAST_PRE+ENERGY_COL]

# -------------------------------------------------------------------
# 3.  BUILD BID‑CHANGE FLAGS  ----------------------------------------
# -------------------------------------------------------------------

def build_bidflags(folder: Path, months: list[str]) -> pd.DataFrame:
    avail_cols = [f"BANDAVAIL{i}" for i in range(1, 11)]
    out = []
    for m in months:
        f = folder/f"bidperoffer_{m}.csv"
        df = pd.read_csv(f, parse_dates=["INTERVAL_DATETIME"])
        df = df[df["DUID"].isin(DUIDS)]
        df = df[["INTERVAL_DATETIME", "DUID", "BIDTYPE"] + avail_cols]
        df.rename(columns={"INTERVAL_DATETIME": "INTERVAL"}, inplace=True)
        out.append(df)

    bids = pd.concat(out, ignore_index=True)
    bids.sort_values(["DUID", "BIDTYPE", "INTERVAL"], inplace=True)

    # diff across intervals to flag any change
    diff = bids.groupby(["DUID", "BIDTYPE"])[avail_cols].diff().abs()
    bids["Bid_change"] = diff.gt(0).any(axis=1).astype(int)

    return bids[["INTERVAL", "DUID", "BIDTYPE", "Bid_change"]]

bidflags = build_bidflags(Path("price_bands"), MONTH_FILES)

# align market naming (ENERGY only for this script)
bidflags["Market"] = "ENERGY"
# keep only ENERGY bidtype, drop FCAS rows
bidflags = bidflags[bidflags["BIDTYPE"] == "ENERGY"].copy()
bidflags["Market"] = "ENERGY"          # now a safe constant


# -------------------------------------------------------------------
# 4.  MERGE (error at τ) with (Bid_change at τ+2) --------------------
# -------------------------------------------------------------------
panel = bidflags.merge(price, on="INTERVAL", how="left")

# shift bid_change −2 intervals so it reflects response at τ+2
panel.sort_values(["DUID", "Market", "INTERVAL"], inplace=True)
panel["ATTN_t2"] = (panel.groupby(["DUID", "Market"])["Bid_change"].shift(-2))
panel.dropna(subset=["ATTN_t2"], inplace=True)

# attach capacity (max observed MW in period)
cap = load.groupby("DUID")["ENERGY"].max().reset_index(name="MAXCAP")
panel = panel.merge(cap, on="DUID", how="left")
panel["logCap"] = np.log(panel["MAXCAP"].replace(0, np.nan)).fillna(0)

# battery dummy
panel["Battery"] = panel["DUID"].isin(BATTERIES).astype(int)

# -------------------------------------------------------------------
# 5.  BUILD ShareE (rolling 24 h)  -----------------------------------
# -------------------------------------------------------------------
price_E = price[["INTERVAL", ENERGY_COL]].copy()
price_E.rename(columns={ENERGY_COL: "P_E"}, inplace=True)

# interval revenue
rev = (load[["INTERVAL", "DUID", "ENERGY"]]
         .merge(price_E, on="INTERVAL"))
rev["rev_E"] = rev["ENERGY"] * rev["P_E"] * 5/60

# rolling 24 h (288 intervals) revenue share (energy vs. FCAS not needed here)
rev.sort_values(["DUID", "INTERVAL"], inplace=True)
rev["Roll_E"] = rev.groupby("DUID")["rev_E"].transform(lambda s: s.rolling(288, min_periods=1).sum())
# total revenue = Roll_E here (FCAS ignored in this minimal script)
rev["ShareE"] = 1.0
rev["ShareE_lag"] = rev.groupby("DUID")["ShareE"].shift(1)

panel = panel.merge(rev[["INTERVAL", "DUID", "ShareE_lag"]],
                    on=["INTERVAL", "DUID"], how="left")

# -------------------------------------------------------------------
# 6.  ENERGY‑ONLY LOGIT, WITH TRIM & SIGNED FE -----------------------
# -------------------------------------------------------------------
energy = panel.copy()

max_FE_notrim = energy["Abs_FE_E"].max()

# 6‑A  Trim top 1 % extreme |forecast error|
P99 = energy["Abs_FE_E"].quantile(0.99)
energy = energy[energy["Abs_FE_E"] <= P99]

# 6‑B  Build signed log‑magnitude regressors
energy["logMag_FE"] = np.log1p(np.abs(energy["FE_E"]))
energy["FEpos"] = energy["logMag_FE"].where(energy["FE_E"] > 0, 0.0)
energy["FEneg"] = energy["logMag_FE"].where(energy["FE_E"] < 0, 0.0)

# battery interactions
energy["Batt_FEpos"] = energy["Battery"] * energy["FEpos"]
energy["Batt_FEneg"] = energy["Battery"] * energy["FEneg"]

# ------------------------------------------------------------------
# quick sanity-check: largest signed-FE magnitudes (log scale)
# ------------------------------------------------------------------
max_FEpos = energy["FEpos"].max()
max_FEneg = energy["FEneg"].max()          # includes zeros for + surprises
max_FEneg_nz = energy.loc[energy["FEneg"] > 0, "FEneg"].max()

# recover max raw forecast-error magnitudes
# ------------------------------------------------------------------
raw_max = max_FE_notrim
# largest positive surprise
raw_max_pos = np.expm1(max_FEpos)                      # |FE| > 0
# largest negative surprise (keep the sign)
raw_max_neg = -np.expm1(max_FEneg_nz)                  # negative value

print(f"\nLargest no trim FE  (raw) = {raw_max:,.2f} $/MWh")
print(f"\nLargest +FE  (raw) = {raw_max_pos:,.2f} $/MWh")
print(f"Largest –FE  (raw) = {raw_max_neg:,.2f} $/MWh")

# -------------------------------------------------------------------
# 7.  RUN LOGIT ------------------------------------------------------
# -------------------------------------------------------------------
X_E = sm.add_constant(
        energy[["FEpos", "FEneg",
                 "Battery",
                 "Batt_FEpos", "Batt_FEneg",
                 "logCap"]]
      )
y_E = energy["ATTN_t2"].astype(int)

logit_E = Logit(y_E, X_E)
res_E = logit_E.fit(disp=False,
                    cov_type="cluster",
                    cov_kwds={"groups": energy["INTERVAL"]})

print("\n=== ENERGY‑ONLY LOGIT: signed FE with battery interactions ===")
print(res_E.summary())

# %%
# after fitting `res_E`
wald_res = res_E.wald_test("FEpos + Batt_FEpos = FEneg + Batt_FEneg")
print(wald_res)


# optional: save coefficient table
(coef := pd.DataFrame({
        "coef": res_E.params,
        "std_err": res_E.bse,
        "z": res_E.tvalues,
        "p_val": res_E.pvalues
    })).round(4).to_csv("signedFE_logit_coeffs.csv")
print("\nCoefficients saved to 'signedFE_logit_coeffs.csv'")
