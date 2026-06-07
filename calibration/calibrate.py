"""
Kalibracja parametrów (a, σ) modelu Hull-White 1F
=================================================

Metoda: estymacja historyczna z szeregu czasowego krótkiej stopy.

  Δr_i = (θ_i − a·r_i) Δt + σ √Δt · ε_i

Po uproszczeniu (OLS na Δr vs r):
  Δr = α − β·r + ε      →   a ≈ β/Δt,  σ ≈ std(ε)/√Δt

Dodatkowa opcja: krocząca kalibracja (rolling window).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import CALIBRATION_WINDOW


# ── Kalibracja jednorazowa ────────────────────────────────────────────────

def calibrate_hw_historical(
    short_rate: pd.Series, dt: float = 1.0 / 252
) -> dict:
    """
    Estymacja (a, σ) z historycznego szeregu krótkiej stopy.

    Parameters
    ----------
    short_rate : pd.Series  –  proxy krótkiej stopy (np. 3M Treasury)
    dt : float              –  krok czasowy (1/252 = dzienny)

    Returns
    -------
    dict z kluczami: 'a', 'sigma', 'b_long_run', 'residuals_std'
    """
    r = short_rate.dropna().values
    dr = np.diff(r)
    r_lag = r[:-1]

    # OLS: Δr = α + β·r + ε   →   a = −β/dt,  θ̄ ≈ α/dt
    X = np.column_stack([np.ones_like(r_lag), r_lag])
    coeffs, res, _, _ = np.linalg.lstsq(X, dr, rcond=None)
    alpha, beta = coeffs

    a = -beta / dt
    sigma = np.std(dr - X @ coeffs) / np.sqrt(dt)

    # długoterminowa średnia stopy (Vasicek-equivalent)
    b_long_run = alpha / (-beta) if abs(beta) > 1e-12 else r.mean()

    # ograniczenia rozsądności
    a = max(a, 0.001)
    sigma = max(sigma, 1e-6)

    return {
        "a": float(a),
        "sigma": float(sigma),
        "b_long_run": float(b_long_run),
        "residuals_std": float(np.std(dr - X @ coeffs)),
    }


# ── Kalibracja MLE (dokładniejsza) ───────────────────────────────────────

def calibrate_hw_mle(
    short_rate: pd.Series, dt: float = 1.0 / 252
) -> dict:
    """
    Maximum Likelihood Estimation dla procesu Ornsteina-Uhlenbecka.

    Warunkowy rozkład r_{t+Δt} | r_t ~ N(μ, ν²)
      μ = r_t·e^{−aΔt} + b·(1 − e^{−aΔt})
      ν² = σ²·(1 − e^{−2aΔt}) / (2a)
    """
    r = short_rate.dropna().values
    n = len(r) - 1

    def neg_log_lik(params):
        a, sigma, b = params
        if a <= 0 or sigma <= 0:
            return 1e12
        e_adt = np.exp(-a * dt)
        mu = r[:-1] * e_adt + b * (1 - e_adt)
        var = sigma ** 2 * (1 - np.exp(-2 * a * dt)) / (2 * a)
        if var <= 0:
            return 1e12
        ll = -0.5 * n * np.log(2 * np.pi * var) - 0.5 * np.sum(
            (r[1:] - mu) ** 2 / var
        )
        return -ll

    # punkt startowy z OLS
    ols = calibrate_hw_historical(short_rate, dt)
    x0 = [ols["a"], ols["sigma"], ols["b_long_run"]]

    res = minimize(
        neg_log_lik,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8},
    )

    a, sigma, b = res.x
    a = max(a, 0.001)
    sigma = max(sigma, 1e-6)

    return {
        "a": float(a),
        "sigma": float(sigma),
        "b_long_run": float(b),
        "neg_log_lik": float(res.fun),
        "converged": res.success,
    }


# ── Krocząca kalibracja ──────────────────────────────────────────────────

def rolling_calibration(
    short_rate: pd.Series,
    window: int = CALIBRATION_WINDOW,
    method: str = "ols",
    dt: float = 1.0 / 252,
) -> pd.DataFrame:
    """
    Krocząca kalibracja parametrów a i σ w oknie o długości `window`.

    Returns
    -------
    DataFrame z kolumnami: date, a, sigma, b_long_run
    """
    calibrator = calibrate_hw_historical if method == "ols" else calibrate_hw_mle

    dates, a_list, sigma_list, b_list = [], [], [], []

    for end in range(window, len(short_rate)):
        window_data = short_rate.iloc[end - window : end]
        try:
            params = calibrator(window_data, dt)
            dates.append(short_rate.index[end])
            a_list.append(params["a"])
            sigma_list.append(params["sigma"])
            b_list.append(params["b_long_run"])
        except Exception:
            continue

    return pd.DataFrame(
        {"a": a_list, "sigma": sigma_list, "b_long_run": b_list}, index=dates
    )
