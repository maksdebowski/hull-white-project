"""
Symulacja Monte Carlo krótkiej stopy w modelu Hull-White 1F
===========================================================

Schemat Eulera-Maruyamy:
    r_{i+1} = r_i + [θ(t_i) − a·r_i]·Δt + σ·√Δt·Z_i

gdzie Z_i ~ N(0,1).
"""

from __future__ import annotations

import numpy as np

from models.hull_white import HullWhite1F


# ── Symulacja ścieżek ────────────────────────────────────────────────────


def simulate_short_rate(
    model: HullWhite1F,
    r0: float,
    T: float,
    n_paths: int = 10_000,
    n_steps: int = 252,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Symulacja ścieżek krótkiej stopy metodą Eulera-Maruyamy.

    Parameters
    ----------
    model   : HullWhite1F  –  skalibrowany model
    r0      : float        –  bieżąca stopa
    T       : float        –  horyzont symulacji (lata)
    n_paths : int          –  liczba ścieżek MC
    n_steps : int          –  liczba kroków dyskretyzacji
    seed    : int | None   –  ziarno RNG

    Returns
    -------
    t_grid : ndarray, shape (n_steps+1,)
    r_paths : ndarray, shape (n_paths, n_steps+1)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    t_grid = np.linspace(0, T, n_steps + 1)
    r = np.empty((n_paths, n_steps + 1))
    r[:, 0] = r0

    for i in range(n_steps):
        t_i = t_grid[i]
        theta_i = model.theta(t_i)
        Z = rng.standard_normal(n_paths)
        drift = (theta_i - model.a * r[:, i]) * dt
        diffusion = model.sigma * sqrt_dt * Z
        r[:, i + 1] = r[:, i] + drift + diffusion

    return t_grid, r


# ── Wycena opcji na ZCB metodą MC ────────────────────────────────────────


def mc_option_price(
    model: HullWhite1F,
    r0: float,
    T: float,
    S: float,
    K: float,
    n_paths: int = 50_000,
    n_steps: int = 252,
    option_type: str = "call",
    seed: int | None = 42,
) -> dict:
    """
    Cena europejskiej opcji na ZCB metodą Monte Carlo.

    Na każdej ścieżce:
      1. Symuluj r(t) do T
      2. Oblicz P(T,S | r_T) analitycznie z HW
      3. Payoff = max(P(T,S) − K, 0) dla call
      4. Dyskontuj do t=0:  disc = exp(−∫₀ᵀ r(s)ds)  ≈ exp(−Σ r_i Δt)

    Returns
    -------
    dict z kluczami: 'price', 'std_error', 'analytical', 'payoffs'
    """
    t_grid, r_paths = simulate_short_rate(model, r0, T, n_paths, n_steps, seed)
    dt = T / n_steps

    # P(T,S | r_T) – analitycznie z HW
    r_T = r_paths[:, -1]
    P_TS = model.zcb_price(T, S, r_T)

    # Payoff
    if option_type == "call":
        payoffs = np.maximum(P_TS - K, 0)
    else:
        payoffs = np.maximum(K - P_TS, 0)

    # Dyskontowanie: ∫r dt ≈ Σ r_i · Δt (trapezy)
    integral_r = np.trapezoid(r_paths, dx=dt, axis=1)
    disc = np.exp(-integral_r)

    discounted_payoffs = payoffs * disc

    price_mc = np.mean(discounted_payoffs)
    std_err = np.std(discounted_payoffs) / np.sqrt(n_paths)

    # Cena analityczna dla porównania
    price_analytical = model.bond_option_price(0, T, S, K, r0, option_type)

    return {
        "price": float(price_mc),
        "std_error": float(std_err),
        "analytical": float(price_analytical),
        "payoffs": discounted_payoffs,
    }
