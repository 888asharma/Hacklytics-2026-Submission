"""
bs_core.py
Shared Black-Scholes call pricing function + Greeks.
Extracted from climate_pricing.ipynb (Abhinav's branch).
"""
import math


def _erf(x: float) -> float:
    """Abramowitz & Stegun approximation for erf."""
    a1, a2, a3, a4, a5, p = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return sign * y


def Nc(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1 + _erf(x / math.sqrt(2)))


def Np(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def black_scholes_call(S: float, K: float, r: float, sigma: float, T: float) -> dict:
    """
    Compute Black-Scholes call price and all Greeks.

    Parameters
    ----------
    S     : spot price
    K     : strike price
    r     : risk-free rate (annualised)
    sigma : volatility (annualised)
    T     : time to expiry in years

    Returns
    -------
    dict with keys: C, d1, d2, N1, N2, delta, gamma, vega, theta
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return dict(C=max(0.0, S - K), d1=0.0, d2=0.0,
                    N1=0.5, N2=0.5, delta=0.5, gamma=0.0, vega=0.0, theta=0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    N1, N2 = Nc(d1), Nc(d2)
    C = S * N1 - K * math.exp(-r * T) * N2

    delta = N1
    gamma = Np(d1) / (S * sigma * math.sqrt(T))
    vega  = S * Np(d1) * math.sqrt(T)
    theta = -(S * Np(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * N2

    return dict(
        C=round(C, 6),
        d1=round(d1, 6),
        d2=round(d2, 6),
        N1=round(N1, 6),
        N2=round(N2, 6),
        delta=round(delta, 6),
        gamma=round(gamma, 6),
        vega=round(vega, 6),
        theta=round(theta, 6),
    )
