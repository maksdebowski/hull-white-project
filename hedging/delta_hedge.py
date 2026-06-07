"""
Symulacja Delta-Hedgingu opcji na ZCB w modelu Hull-White 1F
============================================================

Dwa tryby:
  1. MC backtest   – ścieżki generowane z modelu (test spójności wewnętrznej)
  2. Historyczny   – rzeczywiste dane krzywej dochodowości (test na rynku)

Strategia:
  • Sprzedajemy opcję call na ZCB, otrzymujemy premię C(0)
  • Kupujemy Δ₀ obligacji P(0,S) i finansujemy resztę z rynku pieniężnego
  • W każdym kroku rebalansujemy Δ
  • Na koniec porównujemy wartość portfela z payoffem opcji
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.yield_curve import YieldCurve
from models.hull_white import HullWhite1F
from simulation.monte_carlo import simulate_short_rate


# ══════════════════════════════════════════════════════════════════════════
#  1.  Delta-Hedging na ścieżkach Monte Carlo
# ══════════════════════════════════════════════════════════════════════════


def run_delta_hedge_mc(
    model: HullWhite1F,
    r0: float,
    T: float,
    S: float,
    K: float,
    n_paths: int = 5_000,
    n_steps: int = 52,
    seed: int | None = 42,
) -> dict:
    """
    Backtest hedgingu na ścieżkach MC.

    Returns
    -------
    dict:
      'hedge_errors' : ndarray (n_paths,)  – błąd replikacji
      'option_values': ndarray (n_paths,)  – wartość opcji na koniec
      'portfolio_values': ndarray (n_paths,) – wartość portfela hedgingowego
      'mean_abs_error': float
    """
    t_grid, r_paths = simulate_short_rate(model, r0, T, n_paths, n_steps, seed)
    dt = T / n_steps

    # Cena opcji na starcie
    C0 = model.bond_option_price(0, T, S, K, r0, "call")

    # Alokacja
    cash = np.full(n_paths, C0)  # konto pieniężne
    bond_pos = np.zeros(n_paths)  # ilość obligacji P(t,S)

    for i in range(n_steps):
        t_i = t_grid[i]
        r_i = r_paths[:, i]

        # Cena obligacji bazowej P(t_i, S)
        P_S = model.zcb_price(t_i, S, r_i)

        # Delta (Δ = ∂C/∂P(t,S) ≈ N(h))  —  wektoryzacja
        P_T = model.zcb_price(t_i, T, r_i)
        sp = model.sigma_p(t_i, T, S)

        if sp > 1e-14:
            h = (1 / sp) * np.log(P_S / (K * P_T)) + sp / 2
            from scipy.stats import norm

            delta_new = norm.cdf(h)
        else:
            delta_new = np.where(P_S > K * P_T, 1.0, 0.0)

        # Rebalans
        d_bonds = delta_new - bond_pos
        cash -= d_bonds * P_S  # kupujemy/sprzedajemy obligacje
        cash *= np.exp(r_i * dt)  # cash rośnie po krótkiej stopie

        bond_pos = delta_new

    # Na koniec (t = T)
    r_T = r_paths[:, -1]
    P_TS = model.zcb_price(T, S, r_T)

    portfolio_value = bond_pos * P_TS + cash
    option_payoff = np.maximum(P_TS - K, 0)

    hedge_error = portfolio_value - option_payoff

    return {
        "hedge_errors": hedge_error,
        "option_values": option_payoff,
        "portfolio_values": portfolio_value,
        "mean_abs_error": float(np.mean(np.abs(hedge_error))),
        "std_error": float(np.std(hedge_error)),
        "initial_premium": float(C0),
    }


# ══════════════════════════════════════════════════════════════════════════
#  2.  Delta-Hedging na danych historycznych
# ══════════════════════════════════════════════════════════════════════════


def run_delta_hedge_historical(
    yield_df: pd.DataFrame,
    a: float,
    sigma: float,
    start_idx: int,
    T_years: float = 1.0,
    S_years: float = 5.0,
    K: float = 0.96,
    hedge_freq: str = "weekly",
) -> dict:
    """
    Backtest delta-hedgingu na historycznych krzywych dochodowości.

    Parameters
    ----------
    yield_df  : DataFrame z krzywymi (wiersze = daty, kolumny = tenory)
    a, sigma  : parametry HW
    start_idx : indeks wiersza = data startu
    T_years   : czas do wygaśnięcia opcji
    S_years   : czas do zapadalności obligacji
    K         : strike
    hedge_freq: 'daily' | 'weekly'

    Returns
    -------
    dict z historią hedgingu
    """
    # Liczba kroków
    steps_per_year = 252 if hedge_freq == "daily" else 52
    n_steps = int(T_years * steps_per_year)
    step_size = 1 if hedge_freq == "daily" else 5

    # Sprawdzenie, czy mamy dość danych
    end_idx = start_idx + n_steps * step_size
    if end_idx >= len(yield_df):
        end_idx = len(yield_df) - 1
        n_steps = (end_idx - start_idx) // step_size

    # Budujemy krzywą na datę startową
    row0 = yield_df.iloc[start_idx]
    curve0 = YieldCurve.from_dataframe_row(row0)
    model0 = HullWhite1F(a, sigma, curve0)

    r0 = curve0.f(1e-4)
    C0 = model0.bond_option_price(0, T_years, S_years, K, r0, "call")

    # Historia
    dates = []
    r_hist = []
    delta_hist = []
    option_price_hist = []
    bond_price_hist = []
    portfolio_hist = []

    # Portfel
    cash = C0
    bond_pos = 0.0

    dt = 1.0 / (252 if hedge_freq == "daily" else 52)

    for step in range(n_steps + 1):
        idx = start_idx + step * step_size
        if idx >= len(yield_df):
            break

        row = yield_df.iloc[idx]
        curve = YieldCurve.from_dataframe_row(row)

        t = step * dt
        T_remaining = T_years - t
        S_remaining = S_years - t

        if T_remaining <= 0.01 or S_remaining <= T_remaining:
            break

        model = HullWhite1F(a, sigma, curve)
        r_t = curve.f(1e-4)

        # Ceny obligacji
        P_S = curve.P(S_remaining)
        P_T = curve.P(T_remaining)

        # Cena opcji (od nowa z bieżącej krzywej, t=0 bo recalibrujemy θ)
        option_val = model.bond_option_price(
            0, T_remaining, S_remaining, K, r_t, "call"
        )

        # Delta
        deltas = model.bond_option_delta(0, T_remaining, S_remaining, K, r_t, "call")
        delta_new = deltas["delta_bond"]

        # Rebalans (krok 0 = zakup początkowy, krok >0 = rebalans + odsetki)
        if step > 0:
            cash *= np.exp(r_prev * dt)  # odsetki od poprzedniego kroku

        d_bonds = delta_new - bond_pos
        cash -= d_bonds * P_S
        bond_pos = delta_new
        r_prev = r_t

        # Zapis
        dates.append(yield_df.index[idx])
        r_hist.append(r_t)
        delta_hist.append(delta_new)
        option_price_hist.append(option_val)
        bond_price_hist.append(P_S)
        portfolio_hist.append(bond_pos * P_S + cash)

    # Payoff na koniec
    final_P_S = bond_price_hist[-1] if bond_price_hist else 0
    option_payoff = max(final_P_S - K, 0)
    final_portfolio = portfolio_hist[-1] if portfolio_hist else 0
    hedge_error = final_portfolio - option_payoff

    return {
        "dates": dates,
        "r_history": np.array(r_hist),
        "delta_history": np.array(delta_hist),
        "option_prices": np.array(option_price_hist),
        "bond_prices": np.array(bond_price_hist),
        "portfolio_values": np.array(portfolio_hist),
        "option_payoff": option_payoff,
        "hedge_error": hedge_error,
        "initial_premium": C0,
    }
