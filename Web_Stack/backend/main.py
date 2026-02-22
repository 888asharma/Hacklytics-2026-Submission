"""
main.py
ClimaQuant FastAPI backend.
Run with: uvicorn main:app --reload --port 8000
"""
import sys
import os
# Ensure backend/ is on the path so relative imports work
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from pricing.vanilla import calculate_vanilla_price, get_all_sigmas, COMPANIES, TICKER_MAP
from pricing.climate import calculate_climate_price
from pricing.bs_core import black_scholes_call
from data.loader import get_current_day_data, get_all_companies_data, build_research_payload
from climate.climate_model import get_regression_summary

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ClimaQuant API",
    description="Climate-adjusted options pricing — Hacklytics 2026",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache expensive computations at startup
_research_cache: dict = {}
_dashboard_cache: list = []
_sigma_cache: dict = {}


@app.on_event("startup")
async def startup():
    global _research_cache, _dashboard_cache, _sigma_cache
    print("ClimaQuant API starting up — pre-computing caches...")
    try:
        _dashboard_cache = get_all_companies_data()
        print(f"  ✓ Dashboard: {len(_dashboard_cache)} companies loaded")
    except Exception as e:
        print(f"  ✗ Dashboard cache failed: {e}")
    try:
        _sigma_cache = get_all_sigmas()
        print(f"  ✓ Sigmas cached for {len(_sigma_cache)} companies")
    except Exception as e:
        print(f"  ✗ Sigma cache failed: {e}")
    try:
        _research_cache = build_research_payload()
        print(f"  ✓ Research payload built")
    except Exception as e:
        print(f"  ✗ Research cache failed: {e}")
    print("ClimaQuant API ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────────────────────

class VanillaRequest(BaseModel):
    company:  str   = Field(..., example="apple")
    strike:   float = Field(..., example=185.0)
    maturity: float = Field(..., example=0.25, description="Years to expiry")
    r:        float = Field(0.05, example=0.05, description="Risk-free rate")


class ClimateRequest(BaseModel):
    company:  str            = Field(..., example="apple")
    strike:   float          = Field(..., example=185.0)
    maturity: float          = Field(..., example=0.25)
    r:        float          = Field(0.05)
    hdd:      Optional[float] = Field(None, description="Heating degree days — omit to use model-matched NOAA data")
    cdd:      Optional[float] = Field(None, description="Cooling degree days — omit to use model-matched NOAA data")
    palmer_z: Optional[float] = Field(None, description="Palmer-Z drought index — omit to use model-matched NOAA data")


class CustomRequest(BaseModel):
    S:     float = Field(..., example=183.42)
    K:     float = Field(..., example=185.0)
    r:     float = Field(0.05)
    T:     float = Field(..., example=0.25)
    sigma: float = Field(..., example=0.23)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "api": "ClimaQuant v1.0"}


@app.get("/api/companies", tags=["Meta"])
def list_companies():
    """Return the list of supported companies with tickers."""
    return [{"key": c, "ticker": TICKER_MAP.get(c, c.upper())} for c in COMPANIES]


@app.post("/api/vanilla-price", tags=["Pricing"])
def vanilla_price(req: VanillaRequest):
    """
    Compute vanilla Black-Scholes call price using historical volatility.
    Powers: Page 1 (Vanilla B-S)
    """
    if req.company not in COMPANIES:
        raise HTTPException(404, f"Unknown company '{req.company}'. Use GET /api/companies for valid keys.")
    try:
        return calculate_vanilla_price(req.company, req.strike, req.maturity, req.r)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Pricing error: {e}")


@app.post("/api/climate-price", tags=["Pricing"])
def climate_price(req: ClimateRequest):
    """
    Compute climate-adjusted call price.
    Powers: Page 2 (Climate B-S) + Page 4 Calculator climate sigma preset.
    """
    if req.company not in COMPANIES:
        raise HTTPException(404, f"Unknown company '{req.company}'")
    try:
        return calculate_climate_price(
            req.company, req.strike, req.maturity, req.r,
            req.hdd, req.cdd, req.palmer_z
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Climate pricing error: {e}")


@app.get("/api/current-price", tags=["Dashboard"])
def current_price(company: Optional[str] = Query(None, description="Company key; omit for all 29")):
    """
    Dashboard data: stock price, strike, sigma, option price, % change.
    Powers: Page 3 (Dashboard).
    ?company=apple  → single company
    (no param)      → all 29 companies
    """
    if company:
        if company not in COMPANIES:
            raise HTTPException(404, f"Unknown company '{company}'")
        # Check live cache first
        for row in _dashboard_cache:
            if row.get("company") == company:
                return row
        # Fallback: compute on demand
        try:
            return get_current_day_data(company)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            raise HTTPException(500, str(e))
    # All companies — return cache or compute
    if _dashboard_cache:
        return _dashboard_cache
    return get_all_companies_data()


@app.post("/api/custom-price", tags=["Calculator"])
def custom_price(req: CustomRequest):
    """
    Pure Black-Scholes with user-supplied inputs.
    Powers: Page 4 (Calculator).
    """
    return black_scholes_call(req.S, req.K, req.r, req.sigma, req.T)


@app.get("/api/research-data", tags=["Research"])
def research_data():
    """
    Pre-computed research payload: MAPE, distributions, waveform, scatter, regression.
    Powers: Page 5 (Research & Visualization).
    """
    if _research_cache:
        return _research_cache
    # Compute on demand if startup cache missed
    try:
        return build_research_payload()
    except Exception as e:
        raise HTTPException(500, f"Research computation failed: {e}")


@app.get("/api/sigmas", tags=["Meta"])
def all_sigmas():
    """Return historical sigma for all companies (for bar charts)."""
    if _sigma_cache:
        return _sigma_cache
    return get_all_sigmas()


@app.get("/api/sigma-timeseries", tags=["Research"])
def sigma_timeseries(company: str = Query(..., description="Company key e.g. 'apple'")):
    """
    Monthly historical sigma vs climate-predicted sigma time series.
    Matches 'Actual vs Climate-Adjusted Sigma' plot from climate_coupling.ipynb Cell 2.
    Powers: Research page sigma time series chart.
    """
    if company not in COMPANIES:
        raise HTTPException(404, f"Unknown company '{company}'")
    from climate.climate_model import get_sigma_timeseries
    result = get_sigma_timeseries(company)
    if result is None:
        raise HTTPException(503, "Climate data not available — ensure ClimateData/ CSVs are present")
    return result


@app.get("/api/scatter-3d", tags=["Research"])
def scatter_3d():
    """
    Real monthly Cooling × Heating × PalmerZ × Sigma observations across all companies.
    Matches 3D scatter in climate_coupling.ipynb Cell 2.
    """
    from climate.climate_model import get_scatter_3d_data
    return get_scatter_3d_data()


@app.get("/api/debug", tags=["Debug"])
def debug_info():
    """Shows path resolution and data availability — use to diagnose missing ClimateData."""
    from pathlib import Path
    from climate.climate_model import CLIMATE_DIR, _CLIMATE_DATA_AVAILABLE, COMPANIES
    import os
    candidates = [
        Path(__file__).parent / "ClimateData",
        Path(__file__).parent.parent / "ClimateData",
        Path.cwd() / "ClimateData",
        Path.cwd() / "backend" / "ClimateData",
    ]
    return {
        "cwd":                    str(Path.cwd()),
        "main_py_dir":            str(Path(__file__).parent),
        "climate_dir_resolved":   str(CLIMATE_DIR),
        "climate_data_available": _CLIMATE_DATA_AVAILABLE,
        "candidates_checked":     [{"path": str(p), "exists": p.exists()} for p in candidates],
        "companies_count":        len(COMPANIES),
    }