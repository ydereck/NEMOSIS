# South Australia Bid‐Offer Analysis

This repository contains data‐pull and analysis scripts for “Scarce Attention at High Frequency,” including bid‐change frequency, forecast‐error diagnostics, and difference‐in‐differences logit regressions.  Each script is briefly described below.

## 1. Data Retrieval

- **`BidDayOffer.py`**  
  Connects to NEMOSIS and fetches daily bidding prices and availability (“price bands”) for each generating unit.  

- **`BidPerOffer.py`**  
  Similar to `BidDayOffer.py`, but iterates through all five‐minute intervals between November 2019 and March 2020.  Outputs `biddayoffer_2019nov_2020mar.csv`, which shows sample price‐band data.

- **`forecast_price_noduplicate.py`**  
  Pulls rolling 5-minute price forecasts (PD-5) for each market from AEMO’s API.  Ensures no duplicate timestamps.  

- **`load_by_DUID.py`**  
  Downloads five‐minute dispatch‐cleared MW for every unit and market.  When combined with actual prices, this generates the revenue time series used in Section 5.

## 2. Descriptive Analysis

- **`Bid_change_count.py`**  
  Reads `biddayoffer_2019nov_2020mar.csv` and counts how many five-minute intervals each unit revised at least one price band.  Produces the descriptive statistics and histograms in Section 4.

- **`forecast_error_plot.py`**  
  Calculates forecast errors (realised price minus PD-5 forecast) for each market, then plots the distributions shown in Section 4.  Requires output from `forecast_price_noduplicate.py`.

## 3. Difference-in-Differences Logit Models

- **`logit_diff_in_diff.py`** (Section 6.1)  
  Runs a simple logit on the energy market only.  
  - Binary outcome: did a unit revise any energy price band two intervals after a PD-5 forecast?  
  - Regressors: ln|Forecast Error|, a battery dummy, interaction term, and lnCapacity.  

- **`logit_signed_did.py`** (Section 6.1, robustness)  
  Implements the same energy-only logit after trimming the top 1 % of \(\lvert\text{FE}\rvert\).  
  Splits \(\ln(1+\lvert\text{FE}\rvert)\) into positive and negative components, each interacted with the battery dummy.  

- **`logit_stacked_did.py`** (Section 6.2)  
  Estimates the core stacked logit across all nine markets (energy + 8 ancillary services).  

- **`logit_stacked_volatility.py`** (Section 6.3)  
  Builds on `logit_stacked_did.py` by adding:
  - A rolling 24-hour realised volatility for each market.  
  - lnVolatility and its interactions with revenue share and ln|FE|.  
  Produces the volatility-augmented coefficients reported in Section 6.3.
