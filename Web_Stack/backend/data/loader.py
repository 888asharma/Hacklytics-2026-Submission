"""
loader.py
Refactored from data_processing_test.ipynb (Shiven's branch).
Powers the Dashboard page and Research page.

Key params match the notebook exactly:
  r = 0.035, T = 1, K = S0 (ATM)
"""
import math
import numpy as np
from scipy.stats import norm
from pricing.vanilla import load_closes, COMPANIES, TICKER_MAP
from climate.climate_model import calculate_climate_sigma, get_regression_summary

# ── Exact params from data_processing_test.ipynb ──────────────────────────
DASH_R = 0.035
DASH_T = 1


def _bs_atm(S: float, sigma: float) -> float:
    """Black-Scholes ATM call: K=S, r=0.035, T=1. Matches notebook exactly."""
    K = S
    d1 = (math.log(S / K) + (DASH_R + 0.5 * sigma ** 2) * DASH_T) / (sigma * math.sqrt(DASH_T))
    d2 = d1 - sigma * math.sqrt(DASH_T)
    return float(S * norm.cdf(d1) - K * math.exp(-DASH_R * DASH_T) * norm.cdf(d2))


def _sigma_from_closes(closes_arr: np.ndarray, start: int, end: int) -> float:
    """
    Compute annualised sigma from log returns over [start, end] range.
    Matches notebook: csv_data.loc[start:end, 'Return'].std() * sqrt(252)
    """
    log_rets = np.log(closes_arr[start:end+1][1:] / closes_arr[start:end+1][:-1])
    # Use pandas-style std (ddof=1) matching .std() in notebook
    return float(np.std(log_rets, ddof=1) * math.sqrt(252))


def compute_mape(closes: list) -> float:
    """
    Matches custom_range_returns_compounded_check() from the notebook exactly.

    Notebook call: days_bef_start_point = len(csv_data)
    So: start = len(csv_data) - i - len(csv_data) = -i → always < 1 → clamped to 1
    Meaning sigma is always computed from row 1 up to row (len-i).

    For each yearly interval stepping back every 252 days:
      - start = 1 (always, because days_bef_start_point = n)
      - end   = n - i  (pandas .loc[start:end] is inclusive)
      - sigma = std of log returns from index 1..end  (ddof=1, matches pandas .std())
      - S0    = closes[n - i]                         (csv_data['Close'].iloc[-i])
      - K     = S0 (ATM)
      - option_price = BS(S0, K=S0, r=0.035, T=1)
      - expiry_price = closes[n - i + 251]            (csv_data['Close'].iloc[-i + 251])
      - error = (expiry_price - S0 - option_price) / S0 * 100
    Returns average absolute % error across all intervals.
    """
    closes_arr = np.array(closes, dtype=float)
    n = len(closes_arr)
    percentage_list = []

    for i in range(252, n, 252):
        # Exact notebook index mapping
        # csv_data has n rows indexed 0..n-1
        # iloc[-i] = closes_arr[n - i]
        end = n - i          # exclusive in numpy, but notebook uses pandas .loc[1:end] which is inclusive
                             # so we compute log returns for indices 1..end (pandas inclusive = numpy [1:end+1])

        if end < 2:
            continue

        # sigma from log returns index 1..end (pandas loc[1:end] inclusive)
        # numpy equivalent: closes_arr[1:end+1] for prices, then log returns of those
        prices_window = closes_arr[0:end + 1]   # prices from 0..end so log rets cover 1..end
        log_rets = np.log(prices_window[1:] / prices_window[:-1])
        if len(log_rets) < 2:
            continue
        sigma = float(np.std(log_rets, ddof=1) * math.sqrt(252))

        if sigma == 0:
            continue

        S0 = float(closes_arr[n - i])    # csv_data['Close'].iloc[-i]
        K  = S0

        # Exact notebook d1/d2 formula
        d1 = (((DASH_R + 0.5 * sigma ** 2) * DASH_T) + math.log(S0 / K)) / (sigma * math.sqrt(DASH_T))
        d2 = d1 - sigma * math.sqrt(DASH_T)
        option_price = S0 * norm.cdf(d1) - K * math.exp(-DASH_R * DASH_T) * norm.cdf(d2)

        # csv_data['Close'].iloc[-i + 251]
        expiry_idx = n - i + 251
        if expiry_idx >= n:
            continue

        stock_price_expiry = float(closes_arr[expiry_idx])

        stock_price_difference = stock_price_expiry - S0 - option_price
        stock_difference_percentage = stock_price_difference / S0
        percentage_list.append(stock_difference_percentage * 100)

    if not percentage_list:
        return 0.0

    average_abs = sum(abs(x) for x in percentage_list) / len(percentage_list)
    return round(average_abs, 2)


def get_current_day_data(company: str) -> dict:
    """
    Computes dashboard row for a single company.
    Matches data_processing_test.ipynb exactly:
      sigma = std(log_returns) * sqrt(252), K=S0 ATM, r=0.035, T=1
    """
    closes = load_closes(company)
    closes_arr = np.array(closes, dtype=float)

    log_rets = np.log(closes_arr[1:] / closes_arr[:-1])
    log_rets = log_rets[~np.isnan(log_rets)]

    if len(log_rets) < 2:
        raise ValueError(f"Not enough data for {company}")

    sigma_hist = float(log_rets.std() * math.sqrt(252))
    if sigma_hist == 0:
        raise ValueError(f"Sigma is zero for {company}")

    S    = float(closes_arr[-1])
    prev = float(closes_arr[-2]) if len(closes_arr) >= 2 else S

    vanilla_price = _bs_atm(S, sigma_hist)
    sigma_climate = calculate_climate_sigma(company, sigma_hist)
    climate_price = _bs_atm(S, sigma_climate)

    abs_diff = climate_price - vanilla_price
    pct_diff = abs_diff / vanilla_price * 100 if vanilla_price > 0 else 0.0
    avg_abs_return = float(np.mean(np.abs(log_rets)))

    return {
        "company":        company,
        "ticker":         TICKER_MAP.get(company, company.upper()),
        "stock_price":    round(S, 2),
        "prev_price":     round(prev, 2),
        "change_pct":     round((S - prev) / prev * 100, 3),
        "strike":         round(S, 2),
        "sigma_hist":     round(sigma_hist, 6),
        "sigma_climate":  round(sigma_climate, 6),
        "vanilla_price":  round(vanilla_price, 4),
        "climate_price":  round(climate_price, 4),
        "abs_diff":       round(abs_diff, 4),
        "pct_diff":       round(pct_diff, 4),
        "avg_abs_return": round(avg_abs_return * 100, 4),
        "n_obs":          len(closes),
        "r":              DASH_R,
        "T":              DASH_T,
    }


def get_all_companies_data() -> list:
    """Return dashboard data for all 29 companies."""
    results = []
    for company in COMPANIES:
        try:
            results.append(get_current_day_data(company))
        except Exception as e:
            results.append({
                "company": company,
                "ticker":  TICKER_MAP.get(company, "?"),
                "error":   str(e)
            })
    return results


def compute_error_observations(closes: list) -> list:
    """
    Returns the raw list of pct_change values for each yearly window.
    Same loop as compute_mape but returns individual observations
    so the frontend can plot the full distribution per company.
    """
    closes_arr = np.array(closes, dtype=float)
    n = len(closes_arr)
    observations = []

    for i in range(252, n, 252):
        end = n - i
        if end < 2:
            continue
        prices_window = closes_arr[0:end + 1]
        log_rets = np.log(prices_window[1:] / prices_window[:-1])
        if len(log_rets) < 2:
            continue
        sigma = float(np.std(log_rets, ddof=1) * math.sqrt(252))
        if sigma == 0:
            continue

        S0 = float(closes_arr[n - i])
        K  = S0
        d1 = (((DASH_R + 0.5 * sigma ** 2) * DASH_T) + math.log(S0 / K)) / (sigma * math.sqrt(DASH_T))
        d2 = d1 - sigma * math.sqrt(DASH_T)
        option_price = S0 * norm.cdf(d1) - K * math.exp(-DASH_R * DASH_T) * norm.cdf(d2)

        expiry_idx = n - i + 251
        if expiry_idx >= n:
            continue

        stock_price_expiry = float(closes_arr[expiry_idx])
        pct_change = (stock_price_expiry - S0 - option_price) / S0 * 100
        observations.append(pct_change)

    return observations


def _normal_dist_data(values: list, n_bins: int = 20) -> dict:
    """
    Given a list of values, return histogram bins + normal fit params
    for frontend chart rendering. Matches sphinx_option_error_norm_dis.ipynb.
    """
    if len(values) < 3:
        return None
    x = np.array(values, dtype=float)
    mu  = float(x.mean())
    sd  = float(x.std(ddof=1)) if len(x) > 1 else 0.0

    x_min = float(x.min())
    x_max = float(np.percentile(np.abs(x), 99.5)) * 1.05 if len(x) > 0 else float(x.max())
    x_max = max(x_max, float(x.max()))

    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    counts, _ = np.histogram(x, bins=bin_edges, density=True)

    # Normal PDF overlay curve
    xs = np.linspace(x_min, x_max, 120)
    pdf = norm.pdf(xs, loc=mu, scale=sd).tolist() if sd > 0 else [0.0] * 120

    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()

    return {
        "mu":          round(mu, 2),
        "sd":          round(sd, 2),
        "n":           len(values),
        "bin_centers": [round(v, 2) for v in bin_centers],
        "density":     [round(float(v), 6) for v in counts],
        "curve_x":     [round(v, 2) for v in xs.tolist()],
        "curve_y":     [round(v, 6) for v in pdf],
        "mean_line":   round(mu, 2),
    }


def compute_company_distributions() -> list:
    """
    For each company, compute per-yearly-window abs pct change observations,
    then produce normal dist data with and without IQR outliers.
    Matches sphinx_option_error_norm_dis.ipynb and sphinx_option_error_norm_dis_no_outliers.ipynb.
    """
    results = []
    for company in COMPANIES:
        ticker = TICKER_MAP.get(company, company.upper())
        try:
            closes = load_closes(company)
            obs    = compute_error_observations(closes)
            if len(obs) < 3:
                continue

            abs_obs = [abs(v) for v in obs]

            # IQR outlier removal (matching no_outliers notebook)
            q1  = float(np.percentile(abs_obs, 25))
            q3  = float(np.percentile(abs_obs, 75))
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            abs_obs_clean = [v for v in abs_obs if lo <= v <= hi]

            with_out    = _normal_dist_data(abs_obs)
            without_out = _normal_dist_data(abs_obs_clean) if len(abs_obs_clean) >= 3 else None

            results.append({
                "company":          company,
                "ticker":           ticker,
                "with_outliers":    with_out,
                "without_outliers": without_out,
            })
        except Exception:
            continue

    return results




def compute_pct_change_timeseries() -> list:
    """
    Matches sphinx_option_error_norm_dis.ipynb Cell 7.
    For each company: signed pct_change per yearly window, year from CSV Date col,
    IQR outlier flag, top-3 annotate flag, avg_abs.
    """
    import pandas as pd
    from pathlib import Path
    CSV_DIR = Path(__file__).parent.parent / "Companies"

    results = []
    for company in COMPANIES:
        ticker = TICKER_MAP.get(company, company.upper())
        try:
            closes = load_closes(company)
            closes_arr = np.array(closes, dtype=float)
            n = len(closes_arr)

            # Get year for each row index from the CSV Date column
            year_by_idx = {}
            for cand in [
                CSV_DIR / f"{company}_us_d.csv",
                CSV_DIR / f"{company}_d.csv",
                CSV_DIR / f"{company}.csv",
            ]:
                if cand.exists():
                    try:
                        df_dates = pd.read_csv(cand, usecols=[0])
                        parsed = pd.to_datetime(df_dates.iloc[:, 0], errors='coerce')
                        year_by_idx = {i: int(parsed.iloc[i].year) for i in range(len(parsed)) if pd.notna(parsed.iloc[i])}
                    except Exception:
                        pass
                    break

            # Same yearly-window loop as compute_mape but return signed pct_change
            windows = []
            for i in range(252, n, 252):
                end = n - i
                if end < 2:
                    continue
                log_rets = np.log(closes_arr[1:end+1] / closes_arr[0:end])
                if len(log_rets) < 2:
                    continue
                sigma = float(np.std(log_rets, ddof=1) * math.sqrt(252))
                if sigma == 0:
                    continue
                S0 = float(closes_arr[n - i])
                d1 = (((DASH_R + 0.5*sigma**2)*DASH_T) + math.log(S0/S0)) / (sigma*math.sqrt(DASH_T))
                d2 = d1 - sigma*math.sqrt(DASH_T)
                option_price = S0*norm.cdf(d1) - S0*math.exp(-DASH_R*DASH_T)*norm.cdf(d2)
                expiry_idx = n - i + 251
                if expiry_idx >= n:
                    continue
                expiry_price = float(closes_arr[expiry_idx])
                pct_change = (expiry_price - S0 - option_price) / S0 * 100
                window_end_idx = n - i
                windows.append({
                    "window_end_index": window_end_idx,
                    "year":             year_by_idx.get(window_end_idx),
                    "pct_change":       round(pct_change, 3),
                    "abs_pct_change":   round(abs(pct_change), 3),
                })

            if not windows:
                continue

            # Sort chronologically
            windows.sort(key=lambda w: w["year"] if w["year"] is not None else w["window_end_index"])

            # IQR outlier flag on abs_pct_change
            abs_vals = [w["abs_pct_change"] for w in windows]
            q1 = float(np.percentile(abs_vals, 25))
            q3 = float(np.percentile(abs_vals, 75))
            iqr = q3 - q1
            upper = q3 + 1.5 * iqr
            lower = q1 - 1.5 * iqr
            for w in windows:
                w["is_outlier"] = bool(w["abs_pct_change"] > upper or w["abs_pct_change"] < lower)

            # Top-3 outliers by abs value → annotate flag
            outliers_sorted = sorted([w for w in windows if w["is_outlier"]], key=lambda w: w["abs_pct_change"], reverse=True)
            top3_idxs = {w["window_end_index"] for w in outliers_sorted[:3]}
            for w in windows:
                w["annotate"] = w["window_end_index"] in top3_idxs

            avg_abs = round(float(np.mean(abs_vals)), 2)

            results.append({
                "company": company,
                "ticker":  ticker,
                "avg_abs": avg_abs,
                "windows": windows,
            })
        except Exception:
            continue
    return results

def build_research_payload() -> dict:
    """
    Pre-compute everything needed by the Research & Visualization page.
    MAPE now uses custom_range_returns_compounded_check() logic from notebook.
    """
    mape_data    = []
    dist_returns = []
    waveform     = []

    for company in COMPANIES:
        try:
            closes = load_closes(company)
            closes_arr = np.array(closes, dtype=float)
            log_rets = np.log(closes_arr[1:] / closes_arr[:-1])
            log_rets = log_rets[~np.isnan(log_rets)]

            if len(log_rets) < 2:
                continue

            sigma = float(log_rets.std() * math.sqrt(252))
            dist_returns.extend(log_rets.tolist())

            # MAPE — matches custom_range_returns_compounded_check() exactly
            mape = compute_mape(closes)





            mape_data.append({
                "company": company,
                "ticker":  TICKER_MAP.get(company, company.upper()),
                "mape":    mape,
            })

            # Waveform: 30-day rolling vol, flag outliers > 1.5x annual sigma
            window = 30
            step = max(1, len(log_rets) // 29)
            for i in range(window, len(log_rets), step):
                roll = log_rets[i - window:i]
                roll_std = float(np.std(roll) * math.sqrt(252))
                val = max(4.0, min(98.0, roll_std * 100))
                waveform.append({
                    "company":    company,
                    "ticker":     TICKER_MAP.get(company, company.upper()),
                    "value":      round(val, 1),
                    "is_outlier": roll_std > sigma * 1.5,
                })

        except Exception as e:
            mape_data.append({
                "company": company,
                "ticker":  TICKER_MAP.get(company, "?"),
                "mape":    0.0,
            })

    # Normal distribution from actual log returns
    if dist_returns:
        arr    = np.array(dist_returns)
        mean_r = float(arr.mean())
        std_r  = float(arr.std())
        n_bins = 32
        lo, hi = mean_r - 4 * std_r, mean_r + 4 * std_r
        bin_w  = (hi - lo) / n_bins
        bins   = [lo + i * bin_w for i in range(n_bins)]

        counts = [0] * n_bins
        for v in dist_returns:
            idx = int((v - lo) / bin_w)
            if 0 <= idx < n_bins:
                counts[idx] += 1

        inner = [v for v in dist_returns if abs(v - mean_r) <= 2 * std_r]
        counts_no_out = [0] * n_bins
        for v in inner:
            idx = int((v - lo) / bin_w)
            if 0 <= idx < n_bins:
                counts_no_out[idx] += 1

        std_inner = float(np.std(inner)) if inner else 0.0
        returns_dist = {
            "bins":               [round(b * 100, 3) for b in bins],
            "counts":             counts,
            "counts_no_outliers": counts_no_out,
            "std":                round(std_r * 100, 4),
            "std_no_outliers":    round(std_inner * 100, 4),
        }
    else:
        returns_dist = {"bins": [], "counts": [], "counts_no_outliers": [], "std": 0, "std_no_outliers": 0}

    # 3D scatter: HDD x CDD x climate_sigma
    scatter_3d = []
    for i, company in enumerate(COMPANIES):
        try:
            closes  = load_closes(company)
            arr     = np.array(closes, dtype=float)
            sigma_h = float(np.std(np.log(arr[1:] / arr[:-1])) * math.sqrt(252))
            for j in range(8):
                hdd_val = 200 + (j + i * 3) % 20 * 100
                cdd_val = 50  + (j + i * 5) % 18 * 100
                sc = calculate_climate_sigma(company, sigma_h, hdd_val, cdd_val, -0.5)
                scatter_3d.append({"hdd": hdd_val, "cdd": cdd_val, "sigma": round(sc, 4)})
        except Exception:
            pass

    return {
        "mape":         mape_data,
        "returns_dist": returns_dist,
        "waveform":     waveform,
        "scatter_3d":   scatter_3d,
        "regression":   get_regression_summary(),
        "company_distributions": compute_company_distributions(),
        "pct_change_ts":        compute_pct_change_timeseries(),
    }