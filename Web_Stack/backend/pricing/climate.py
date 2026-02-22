"""
climate.py
Refactored from climate_pricing.ipynb (Abhinav's branch).
Combines vanilla sigma + climate model → climate-adjusted call price.
"""
from pricing.vanilla import load_closes, calculate_historical_sigma
from pricing.bs_core import black_scholes_call
from climate.climate_model import calculate_climate_sigma


def calculate_climate_price(company: str, strike: float, maturity: float, r: float,
                             hdd: float = 1200.0, cdd: float = 600.0,
                             palmer_z: float = -0.5) -> dict:
    """
    Price a call option using climate-adjusted sigma.

    Returns both vanilla and climate pricing for comparison display.
    """
    closes = load_closes(company)
    S = closes[-1]
    sigma_hist = calculate_historical_sigma(closes)
    sigma_climate = calculate_climate_sigma(company, sigma_hist, hdd, cdd, palmer_z)

    vanilla  = black_scholes_call(S, strike, r, sigma_hist, maturity)
    climate  = black_scholes_call(S, strike, r, sigma_climate, maturity)

    abs_diff = climate["C"] - vanilla["C"]
    pct_diff = abs_diff / vanilla["C"] * 100 if vanilla["C"] > 0 else 0.0
    sigma_diff_pp = (sigma_climate - sigma_hist) * 100

    return {
        "company": company,
        "stock_price": round(S, 2),
        "vanilla_sigma": round(sigma_hist, 6),
        "climate_sigma": round(sigma_climate, 6),
        "sigma_diff_pp": round(sigma_diff_pp, 4),
        "vanilla_price": round(vanilla["C"], 4),
        "climate_price": round(climate["C"], 4),
        "abs_diff":  round(abs_diff, 4),
        "pct_diff":  round(pct_diff, 4),
        # Full Greeks for both
        "vanilla":   {k: round(v, 6) for k, v in vanilla.items()},
        "climate_greeks": {k: round(v, 6) for k, v in climate.items()},
    }
