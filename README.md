# ClimaQuant — Climate-Integrated Stock Option Pricing
### Hacklytics 2026 Submission

> *"As digital technology and AI further entrench data from human systems, it is imperative we always account for the natural systems upon which these human systems rely."*

---

## Overview

ClimaQuant is a web application that integrates real-world climate data into stock option pricing strategies. By coupling NOAA climate indices with historical stock data across 29 major companies — spanning energy, food & agriculture, and retail — ClimaQuant allows traders, researchers, and curious users to explore how climate stress affects the true cost of financial risk.

The core question we ask: **what happens to an option's price when you account for drought, extreme heat, or cold snaps?**

---

## Background — What is a Stock Option?

Imagine you want to buy a share of a company worth $100 today, but you're worried the stock might not go up. A **stock option** lets you negotiate to buy that share at a fixed **strike price (K)** at some future **maturation date (T)**. You pay an upfront **option price (C)** for this right.

The industry-standard way to price that option is the **Black-Scholes equation**:

```
C = S·N(d₁) − K·e^(−rT)·N(d₂)

d₁ = [ln(S/K) + (r + σ²/2)·T] / σ√T
d₂ = d₁ − σ√T
```

Where:
- **σ** — volatility (standard deviation of log returns)
- **r** — risk-free rate (expected yield under zero-risk conditions)
- **N()** — cumulative normal distribution applied to d₁ and d₂
- **S** — current stock price, **K** — strike price, **T** — time to expiry

Standard Black-Scholes uses a single fixed historical σ. **ClimaQuant replaces that with a climate-predicted σ**, derived from a multivariate regression against real NOAA climate data — so the option price reflects current climate conditions, not just historical averages.

---

## The Climate Model

For each company, we fit a per-company OLS regression using monthly data:

```
σₜ = β₀ + β₁·σₜ₋₁ + β₂·HDD + β₃·CDD + β₄·PalmerZ + ε
```

Where:
- **σₜ₋₁** — lagged volatility (auto-regression term)
- **HDD** — Heating Degree Days (cold stress → energy demand)
- **CDD** — Cooling Degree Days (heat stress → supply chain pressure)
- **PalmerZ** — Palmer Z Drought Index (negative = drought, positive = wet)

Climate variables are date-aligned to each company's monthly sigma series via `pd.concat`, exactly mirroring the `climate_coupling.ipynb` notebook methodology. The last predicted value from the fitted model is used as **σ_climate** for option pricing.

The **climate premium** — the percentage difference between the climate-adjusted option price and the historical baseline — quantifies the implied market cost of climate risk for each company.

---

## Data Sources

| Data | Source |
|------|--------|
| Stock price histories (29 companies) | [Stooq](https://stooq.com) |
| Heating & Cooling Degree Days | [NOAA](https://www.noaa.gov) |
| Palmer Z Drought Index | [NOAA](https://www.noaa.gov) |

Companies covered span **energy** (ExxonMobil, BP, Shell, ConocoPhillips, NextEra, Enphase, First Solar), **food & agriculture** (Corteva, General Mills, Pepsi, Coca-Cola, Walmart, Target), **tech & markets** (Apple, Microsoft, Amazon, Tesla, Nasdaq, S&P 500, Dow Jones), and more.

---

## Repository Structure — Hacklytics Stack

The full application lives in the `Hacklytics_stack/` folder, which integrates a FastAPI backend and a pure HTML/JS frontend into a single deployable project:

```
Hacklytics_stack/
├── backend/                        # FastAPI Python backend
│   ├── main.py                     # API entrypoint (uvicorn)
│   ├── requirements.txt
│   ├── .python-version             # Python 3.11.9 pin for Render
│   ├── pricing/
│   │   ├── vanilla.py              # Historical Black-Scholes pricing
│   │   ├── climate.py              # Climate-adjusted pricing
│   │   └── bs_core.py              # Core B-S math + Greeks
│   ├── climate/
│   │   └── climate_model.py        # OLS regression, sigma timeseries, 3D scatter
│   ├── data/
│   │   └── loader.py               # CSV loading, MAPE, distributions, research payload
│   ├── Companies/                  # Per-company daily stock price CSVs (Stooq)
│   └── ClimateData/                # NOAA climate CSVs
│       ├── Cooling_Degree_Days.csv
│       ├── Heating_Degree_Days.csv
│       └── Palmer_Z.csv
│
└── frontend/                       # Single-page web application
    └── index.html                  # Full app — vanilla JS + Chart.js, no framework
```

### Backend

Built with **FastAPI** and **uvicorn**. Key endpoints:

| Endpoint | Description |
|----------|-------------|
| `POST /api/vanilla-price` | Historical Black-Scholes call price |
| `POST /api/climate-price` | Climate-adjusted call price |
| `GET /api/current-price` | Dashboard data per company |
| `GET /api/research-data` | Full research payload (MAPE, distributions, regression) |
| `GET /api/sigma-timeseries` | Historical vs climate σ over time |
| `GET /api/scatter-3d` | 3D climate space scatter data |
| `GET /api/debug` | Path resolution diagnostics |

### Frontend

A single `index.html` file — no framework, no build step. Built with vanilla JavaScript and Chart.js. Pages include:

- **Vanilla B-S** — historical volatility calculator with Greeks output and sensitivity curve
- **Climate B-S** — climate-conditioned pricing with automatic NOAA-aligned σ prediction
- **Executive Dashboard** — side-by-side comparison of all 29 companies
- **Option Calculator** — interactive B-S calculator with adjustable S, K, r, T
- **Research & Visualization** — MAPE analysis, normal distributions, σ timeseries, 3D climate scatter, option pricing error waveforms

---

## Key Metrics

**MAPE (Mean Absolute Percentage Error)** measures the average difference between the stock price at maturation and the sum of the strike price plus accumulated option price. The closer to zero, the more accurate the pricing model.

**Climate Premium (%)** is the percentage by which the climate-adjusted option price exceeds (or falls below) the historical baseline — quantifying how much climate risk is being priced into each company's options.

---

## Tools & Technologies

| Layer | Technology |
|-------|-----------|
| Backend framework | FastAPI + uvicorn |
| Numerical computing | NumPy, SciPy, pandas |
| Statistical modeling | statsmodels (OLS) |
| Data sourcing & analysis | SphinxAI |
| Frontend | Vanilla JS, Chart.js |
| Frontend development & integration | Claude AI (Anthropic) |
| Deployment | Render (backend) + Vercel (frontend) |

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/888asharma/Hacklytics-2026-Submission.git
cd Hacklytics-2026-Submission/Hacklytics_stack

# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (separate terminal)
# Just open frontend/index.html in your browser
# or serve with:
cd frontend
python -m http.server 3000
```

Make sure the API URL in `frontend/index.html` is set to `http://localhost:8000` for local development.

---

## Deployment

- **Backend** — deployed on [Render](https://render.com) as a Python web service
- **Frontend** — deployed on [Vercel](https://vercel.com) as a static site
- Live at: `https://climaquant-app.vercel.app`

---

## Team

Built at **Hacklytics 2026** — Georgia Tech's annual data science hackathon.
