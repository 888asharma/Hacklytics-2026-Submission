"""
vanilla.py
Refactored from vanilla_pricing.ipynb (Abhinav's branch).
Reads CSVs from Companies/ folder (Shiven's branch).
"""
import math
import os
import csv
from pathlib import Path
from .bs_core import black_scholes_call
import numpy as np


CSV_DIR = Path(__file__).parent.parent / "Companies"

COMPANIES = [
    "amazon", "apple", "bp", "coke", "cop", "corteva",
    "dowjones", "enb", "enph", "eog", "exxon", "fslr",
    "general_mills", "gm", "kmi", "microsoft", "nasdaq",
    "next_era_energy", "pepsi", "plug", "shell", "sp500",
    "target", "tesla", "total_energies", "vde", "walmart",
    "wmb", "xle"
]

TICKER_MAP = {
    "amazon": "AMZN", "apple": "AAPL", "bp": "BP", "coke": "KO",
    "cop": "COP", "corteva": "CTVA", "dowjones": "DJI", "enb": "ENB",
    "enph": "ENPH", "eog": "EOG", "exxon": "XOM", "fslr": "FSLR",
    "general_mills": "GIS", "gm": "GM", "kmi": "KMI", "microsoft": "MSFT",
    "nasdaq": "IXIC", "next_era_energy": "NEE", "pepsi": "PEP", "plug": "PLUG",
    "shell": "SHEL", "sp500": "SPX", "target": "TGT", "tesla": "TSLA",
    "total_energies": "TTE", "vde": "VDE", "walmart": "WMT",
    "wmb": "WMB", "xle": "XLE"
}


def load_closes(company: str) -> list[float]:
    """Load closing prices from company CSV. Returns list sorted oldest → newest."""
    # Try common filename variants — Shiven's files use _us_d suffix e.g. apple_us_d.csv
    # sp500 and dowjones don't have _us_ prefix
    candidates = [
        CSV_DIR / f"{company}_us_d.csv",        # apple_us_d.csv  (main pattern)
        CSV_DIR / f"{company}_d.csv",            # sp500_d.csv, dowjones_d.csv
        CSV_DIR / f"{company}.csv",              # plain fallback
        CSV_DIR / f"{company.upper()}.csv",
        CSV_DIR / f"{company.capitalize()}.csv",
    ]
    path = None
    for c in candidates:
        if c.exists():
            path = c
            break
    if path is None:
        raise FileNotFoundError(f"No CSV found for company '{company}' in {CSV_DIR}")

    closes = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # Normalise headers — handle various column name capitalizations
        for row in reader:
            # Find the Close column regardless of casing
            close_val = None
            for k in row:
                if k.strip().lower() == "close":
                    try:
                        close_val = float(row[k].replace(",", ""))
                    except (ValueError, TypeError):
                        pass
                    break
            if close_val is not None and close_val > 0:
                closes.append(close_val)

    if len(closes) < 2:
        raise ValueError(f"Not enough data in {path} (need at least 2 Close prices)")
    return closes



def calculate_historical_sigma(closes: list[float]) -> float: #Calculate volatility from returns compounded from stock creation date
    arr = np.array(closes, dtype=float)
    log_returns = np.log(arr[1:] / arr[:-1])
    return float(log_returns.std() * math.sqrt(252))



def calculate_vanilla_price(company: str, strike: float, maturity: float, r: float) -> dict:
    """
    Main entry point for vanilla Black-Scholes pricing.
    Returns stock price, historical sigma, option price, and Greeks.
    """
    closes = load_closes(company)
    S = closes[-1]
    prev = closes[-2] if len(closes) >= 2 else S
    sigma = calculate_historical_sigma(closes)
    result = black_scholes_call(S, strike, r, sigma, maturity)

    moneyness_pct = (S - strike) / strike * 100
    if abs(moneyness_pct) < 2:
        moneyness = "ATM"
    elif moneyness_pct > 0:
        moneyness = "ITM"
    else:
        moneyness = "OTM"

    return {
        "company": company,
        "ticker": TICKER_MAP.get(company, company.upper()),
        "stock_price": round(S, 2),
        "prev_price": round(prev, 2),
        "change_pct": round((S - prev) / prev * 100, 3),
        "sigma": round(sigma, 6),
        "option_price": round(result["C"], 4),
        "moneyness": moneyness,
        "moneyness_pct": round(moneyness_pct, 2),
        **result
    }


def get_all_sigmas() -> dict:
    """Return {company: sigma} for all 29 companies (for charts)."""
    result = {}
    for company in COMPANIES:
        try:
            closes = load_closes(company)
            result[company] = round(calculate_historical_sigma(closes), 6)
        except Exception as e:
            result[company] = None
    return result
