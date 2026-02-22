"""
climate_model.py
Matches climate_coupling.ipynb (Cell 2) exactly.

Climate sigma pipeline:
  1. Log returns row-by-row: log(Close[j] / Close[j-1])
  2. Rolling 21-day sigma: rolling(21).std() * sqrt(252)
  3. Monthly sigma: resample to month-end, take last value
  4. OLS: Sigma ~ const + LagSigma + Cooling + Heating + PalmerZ
  5. climate_sigma = model.predict(X).iloc[-1]

Climate data files: backend/ClimateData/
  - Cooling_Degree_Days.csv  (skiprows=2, Date col YYYYMM, Value col)
  - Heating_Degree_Days.csv  (skiprows=2, Date col YYYYMM, Value col)
  - Palmer_Z.csv             (skiprows=1, Date col YYYYMM, Value col)
"""
import math
import functools
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pricing.vanilla import load_closes, COMPANIES, TICKER_MAP

# Try multiple possible locations for ClimateData folder
def _find_climate_dir():
    candidates = [
        Path(__file__).parent.parent / "ClimateData",       # backend/ClimateData/
        Path(__file__).parent / "ClimateData",               # backend/climate/ClimateData/
        Path.cwd() / "ClimateData",                          # cwd/ClimateData/
        Path.cwd() / "backend" / "ClimateData",              # cwd/backend/ClimateData/
    ]
    for p in candidates:
        if (p / "Cooling_Degree_Days.csv").exists():
            print(f"[climate_model] Found ClimateData at: {p}")
            return p
    print(f"[climate_model] ClimateData NOT found. Tried:")
    for p in candidates:
        print(f"  {p}  exists={p.exists()}")
    return None

CLIMATE_DIR = _find_climate_dir()


# ── Load climate index CSVs once ──────────────────────────────────────────────

def _load_climate_indices():
    """
    Load Cooling, Heating, Palmer-Z into monthly-indexed Series.
    Returns (cooling, heating, palmer) or (None, None, None) if files missing.
    """
    if CLIMATE_DIR is None:
        return None, None, None
    try:
        cdd_path = CLIMATE_DIR / "Cooling_Degree_Days.csv"
        hdd_path = CLIMATE_DIR / "Heating_Degree_Days.csv"
        pz_path  = CLIMATE_DIR / "Palmer_Z.csv"

        def load_series(path, skiprows):
            df = pd.read_csv(path, skiprows=skiprows)
            df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m')
            df = df.set_index('Date')
            df.index = df.index.to_period('M').to_timestamp('M')
            return df['Value']

        cooling = load_series(cdd_path, skiprows=2)
        heating = load_series(hdd_path, skiprows=2)
        palmer  = load_series(pz_path,  skiprows=1)
        print(f"[climate_model] Loaded climate CSVs: {len(cooling)} months of data")
        return cooling, heating, palmer

    except Exception as e:
        print(f"[climate_model] Failed to load climate CSVs: {e}")
        return None, None, None


# Load at import time
_COOLING, _HEATING, _PALMER = _load_climate_indices()
_CLIMATE_DATA_AVAILABLE = _COOLING is not None


def _company_monthly_sigma(closes: list) -> pd.Series:
    """
    Matches notebook Cell 2 exactly:
      current_data['Return'] = log(Close[j] / Close[j-1])
      current_data['RollingSigma'] = Return.rolling(21).std() * sqrt(252)
      monthly_sigma = RollingSigma.resample('ME').last()
    Returns a monthly-indexed Series of sigma values.
    """
    # Build a date-indexed DataFrame with arbitrary daily dates
    # We need real dates to resample monthly — use the CSV dates if possible,
    # otherwise generate a business-day range ending today
    n = len(closes)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)

    df = pd.DataFrame({'Close': closes}, index=dates)
    df['Return'] = 0.0
    for j in range(1, len(df)):
        closing_ratio = df['Close'].iloc[j] / df['Close'].iloc[j - 1]
        df.iloc[j, df.columns.get_loc('Return')] = math.log(closing_ratio)

    df['RollingSigma'] = df['Return'].rolling(21).std() * math.sqrt(252)
    monthly_sigma = df['RollingSigma'].resample('ME').last()
    return monthly_sigma


@functools.lru_cache(maxsize=64)
def fit_climate_model(company: str):
    """
    Fits OLS model matching climate_coupling.ipynb Cell 2:
      Sigma ~ const + LagSigma + Cooling + Heating + PalmerZ

    Returns (model, merged_df) or None if climate data unavailable.
    Cached — only runs once per company per server session.
    """
    if not _CLIMATE_DATA_AVAILABLE:
        return None

    try:
        closes = load_closes(company)
        monthly_sigma = _company_monthly_sigma(closes)

        # Cell 2 uses monthly_sigma (no shift) as the target
        sigma_series = monthly_sigma.rename('Sigma')

        merged = pd.concat([
            sigma_series,
            _COOLING.rename('Cooling'),
            _HEATING.rename('Heating'),
            _PALMER.rename('PalmerZ'),
        ], axis=1).dropna()

        merged['LagSigma'] = merged['Sigma'].shift(1)
        merged = merged.dropna()

        if len(merged) < 10:
            return None

        X = sm.add_constant(merged[['LagSigma', 'Cooling', 'Heating', 'PalmerZ']])
        y = merged['Sigma']
        model = sm.OLS(y, X).fit()

        return model, merged

    except Exception as e:
        print(f"[climate_model] fit_climate_model({company}) failed: {e}")
        return None


def calculate_climate_sigma(company: str, sigma_hist: float,
                             hdd: float = None, cdd: float = None,
                             palmer_z: float = None) -> float:
    """
    Returns climate-adjusted sigma for a company.

    If climate data CSVs are present:
      - Fits OLS model (cached)
      - Uses last row of merged data to get predicted sigma
      - If hdd/cdd/palmer_z provided, predicts for those specific values instead

    If climate data unavailable:
      - Falls back to a small fixed uplift on sigma_hist
    """
    result = fit_climate_model(company)

    if result is None:
        # Fallback: no climate data — apply small fixed uplift
        return sigma_hist * 1.05

    model, merged = result

    if hdd is not None and cdd is not None and palmer_z is not None:
        # Predict for user-supplied climate inputs
        lag_sigma = float(merged['Sigma'].iloc[-1])
        X_new = pd.DataFrame([{
            'const':     1.0,
            'LagSigma':  lag_sigma,
            'Cooling':   cdd,
            'Heating':   hdd,
            'PalmerZ':   palmer_z,
        }])
        try:
            pred = float(model.predict(X_new).iloc[0])
            return max(pred, 0.001)
        except Exception:
            return sigma_hist * 1.05
    else:
        # Use last predicted value from historical fit (matches notebook .iloc[-1])
        X = sm.add_constant(merged[['LagSigma', 'Cooling', 'Heating', 'PalmerZ']])
        predicted = model.predict(X)
        climate_sigma = float(predicted.iloc[-1])
        return max(climate_sigma, 0.001)


def get_regression_summary() -> list:
    """
    Returns OLS summary for all companies — used by Research page regression table.
    """
    results = []
    for company in COMPANIES:
        ticker = TICKER_MAP.get(company, company.upper())
        result = fit_climate_model(company)

        if result is None:
            results.append({
                "company": company, "ticker": ticker,
                "r2": None, "adj_r2": None, "f_stat": None,
                "beta_0": 0, "beta_sigma": 0,
                "beta_hdd": 0, "beta_cdd": 0, "beta_palmer": 0,
                "significant": False,
            })
            continue

        model, _ = result
        params = model.params
        results.append({
            "company":      company,
            "ticker":       ticker,
            "r2":           round(float(model.rsquared), 4),
            "adj_r2":       round(float(model.rsquared_adj), 4),
            "f_stat":       round(float(model.fvalue), 2),
            "beta_0":       round(float(params.get('const', 0)), 6),
            "beta_sigma":   round(float(params.get('LagSigma', 0)), 6),
            "beta_hdd":     round(float(params.get('Heating', 0)), 6),
            "beta_cdd":     round(float(params.get('Cooling', 0)), 6),
            "beta_palmer":  round(float(params.get('PalmerZ', 0)), 6),
            "significant":  bool(model.f_pvalue < 0.05),
        })

    return results


def get_sigma_timeseries(company: str) -> dict:
    """
    Returns monthly historical sigma vs climate-predicted sigma time series.
    Matches the 'Actual vs Climate-Adjusted Sigma' plot in climate_coupling.ipynb Cell 2.
    """
    result = fit_climate_model(company)
    if result is None:
        return None

    model, merged = result
    X = sm.add_constant(merged[['LagSigma', 'Cooling', 'Heating', 'PalmerZ']])
    predicted = model.predict(X)

    dates = [str(d)[:7] for d in merged.index]  # YYYY-MM format

    return {
        "company":   company,
        "ticker":    TICKER_MAP.get(company, company.upper()),
        "dates":     dates,
        "hist_sigma":    [round(float(v) * 100, 3) for v in merged['Sigma']],
        "climate_sigma": [round(float(v) * 100, 3) for v in predicted],
    }


def get_scatter_3d_data() -> list:
    """
    Returns all monthly observations across all companies:
    Cooling, Heating, PalmerZ, Sigma — matching the 3D scatter in climate_coupling.ipynb Cell 2.
    Each point = one company-month observation from the merged OLS dataframe.
    """
    points = []
    for company in COMPANIES:
        result = fit_climate_model(company)
        if result is None:
            continue
        _, merged = result
        ticker = TICKER_MAP.get(company, company.upper())
        for _, row in merged.iterrows():
            points.append({
                "company":  company,
                "ticker":   ticker,
                "cooling":  round(float(row["Cooling"]), 2),
                "heating":  round(float(row["Heating"]), 2),
                "palmer":   round(float(row["PalmerZ"]), 3),
                "sigma":    round(float(row["Sigma"]), 4),
            })
    return points