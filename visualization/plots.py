"""
Wizualizacja wyników projektu Hull-White 1F
===========================================

Wykresy:
  1. Ewolucja krzywej dochodowości 3D (plotly)
  2. Parametry kalibracji (a, σ) w czasie
  3. P&L portfela hedgingowego
  4. Rozkład payoffów MC
  5. Porównanie HW vs Vasicek
  6. Historia krótkiej stopy
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import MATURITIES, OUTPUT_DIR, YIELD_CURVE_SERIES


def _ensure_output():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
#  1.  Krzywa dochodowości 3D
# ══════════════════════════════════════════════════════════════════════════

def plot_yield_curve_3d(yield_df: pd.DataFrame, save: bool = True) -> go.Figure:
    """
    Wykres powierzchniowy 3D ewolucji krzywej dochodowości (2020–2024).
    Osie: czas × zapadalność × stopa.
    """
    _ensure_output()
    mats = np.array([MATURITIES[k] for k in yield_df.columns if k in MATURITIES])
    cols = [c for c in yield_df.columns if c in MATURITIES]
    Z = yield_df[cols].values * 100  # na procenty

    # Pod-próbkowanie dat (co tydzień) dla wydajności
    step = max(1, len(yield_df) // 260)
    dates_sub = yield_df.index[::step]
    Z_sub = Z[::step]

    date_nums = np.arange(len(dates_sub))

    fig = go.Figure(
        data=[
            go.Surface(
                x=mats,
                y=date_nums,
                z=Z_sub,
                colorscale="Viridis",
                colorbar=dict(title="Stopa [%]"),
            )
        ]
    )

    # Etykiety dat na osi Y
    tick_step = max(1, len(dates_sub) // 8)
    tickvals = date_nums[::tick_step]
    ticktext = [d.strftime("%Y-%m") for d in dates_sub[::tick_step]]

    fig.update_layout(
        title="Ewolucja krzywej dochodowości US Treasury (2020–2024)",
        scene=dict(
            xaxis_title="Zapadalność [lata]",
            yaxis=dict(title="Data", tickvals=tickvals, ticktext=ticktext),
            zaxis_title="Stopa [%]",
            camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8)),
        ),
        width=1000,
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    if save:
        fig.write_html(str(OUTPUT_DIR / "yield_curve_3d.html"))
        try:
            fig.write_image(str(OUTPUT_DIR / "yield_curve_3d.png"), scale=2)
        except Exception:
            pass
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  2.  Parametry kalibracji w czasie
# ══════════════════════════════════════════════════════════════════════════

def plot_calibration_params(
    cal_df: pd.DataFrame, save: bool = True
) -> plt.Figure:
    """Wykresy a(t) i σ(t) z kroczącej kalibracji."""
    _ensure_output()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(cal_df.index, cal_df["a"], color="steelblue", lw=1.2)
    axes[0].set_ylabel("a  (mean-reversion)", fontsize=11)
    axes[0].set_title("Krocząca kalibracja parametrów Hull-White 1F", fontsize=13)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(cal_df.index, cal_df["sigma"], color="firebrick", lw=1.2)
    axes[1].set_ylabel("σ  (zmienność)", fontsize=11)
    axes[1].set_xlabel("Data", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Zaznacz kluczowe daty
    events = {
        "2020-03": "COVID crash",
        "2022-03": "Fed hikes start",
        "2023-07": "Peak rate",
    }
    for ax in axes:
        for date_str, label in events.items():
            try:
                ax.axvline(
                    pd.Timestamp(date_str),
                    color="gray",
                    ls="--",
                    alpha=0.5,
                    lw=0.8,
                )
                ax.text(
                    pd.Timestamp(date_str),
                    ax.get_ylim()[1] * 0.95,
                    f" {label}",
                    fontsize=8,
                    color="gray",
                )
            except Exception:
                pass

    plt.tight_layout()
    if save:
        fig.savefig(str(OUTPUT_DIR / "calibration_params.png"), dpi=150)
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  3.  P&L portfela hedgingowego
# ══════════════════════════════════════════════════════════════════════════

def plot_hedging_pnl(hedge_result: dict, save: bool = True) -> plt.Figure:
    """Wykresy z historycznego backteste'u delta-hedgingu."""
    _ensure_output()
    dates = hedge_result["dates"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Panel 1: wartość portfela vs cena opcji
    axes[0].plot(dates, hedge_result["option_prices"], label="Cena opcji (model)", lw=1.2)
    axes[0].plot(
        dates, hedge_result["portfolio_values"], label="Portfel hedgingowy", lw=1.2, ls="--"
    )
    axes[0].set_ylabel("Wartość")
    axes[0].set_title("Delta-Hedging: portfel vs opcja", fontsize=13)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: delta
    axes[1].plot(dates, hedge_result["delta_history"], color="teal", lw=1)
    axes[1].set_ylabel("Delta (Δ)")
    axes[1].set_title("Pozycja w obligacji (delta)", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: stopa krótka
    axes[2].plot(dates, hedge_result["r_history"] * 100, color="firebrick", lw=1)
    axes[2].set_ylabel("r(t) [%]")
    axes[2].set_xlabel("Data")
    axes[2].set_title("Króka stopa procentowa (proxy)", fontsize=11)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(str(OUTPUT_DIR / "hedging_pnl.png"), dpi=150)
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  4.  Rozkład payoffów / błędów MC
# ══════════════════════════════════════════════════════════════════════════

def plot_mc_distribution(mc_result: dict, save: bool = True) -> plt.Figure:
    """Histogram dyskontowanych payoffów MC + porównanie z ceną analityczną."""
    _ensure_output()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    payoffs = mc_result["payoffs"]

    # Histogram payoffów
    axes[0].hist(payoffs, bins=80, color="steelblue", alpha=0.7, edgecolor="white")
    axes[0].axvline(mc_result["price"], color="red", lw=2, label=f"MC:  {mc_result['price']:.6f}")
    axes[0].axvline(
        mc_result["analytical"],
        color="green",
        lw=2,
        ls="--",
        label=f"Analyt: {mc_result['analytical']:.6f}",
    )
    axes[0].set_xlabel("Dyskontowany payoff")
    axes[0].set_ylabel("Częstość")
    axes[0].set_title("Monte Carlo – rozkład payoffów opcji na ZCB")
    axes[0].legend(fontsize=9)

    # Histogram błędów hedgingu MC (jeśli dostępny)
    axes[1].text(
        0.5, 0.5,
        "Patrz: plot_mc_hedge_errors()",
        ha="center", va="center", fontsize=12, color="gray",
        transform=axes[1].transAxes,
    )
    axes[1].set_title("Błędy replikacji (MC hedge)")

    plt.tight_layout()
    if save:
        fig.savefig(str(OUTPUT_DIR / "mc_distribution.png"), dpi=150)
    return fig


def plot_mc_hedge_errors(hedge_mc: dict, save: bool = True) -> plt.Figure:
    """Histogram błędów replikacji z MC delta-hedgingu."""
    _ensure_output()
    fig, ax = plt.subplots(figsize=(8, 5))

    errors = hedge_mc["hedge_errors"]
    ax.hist(errors, bins=80, color="coral", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", lw=1.5, ls="--")
    ax.set_xlabel("Błąd replikacji (portfel − payoff)")
    ax.set_ylabel("Częstość")
    ax.set_title(
        f"Delta-Hedging MC: MAE={hedge_mc['mean_abs_error']:.6f}, "
        f"σ={hedge_mc['std_error']:.6f}"
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(str(OUTPUT_DIR / "mc_hedge_errors.png"), dpi=150)
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  5.  HW vs Vasicek
# ══════════════════════════════════════════════════════════════════════════

def plot_hw_vs_vasicek(
    maturities: np.ndarray,
    market_rates: np.ndarray,
    hw_rates: np.ndarray,
    vasicek_rates: np.ndarray,
    save: bool = True,
) -> plt.Figure:
    """Porównanie krzywych: rynkowa vs HW vs Vasicek."""
    _ensure_output()
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(maturities, market_rates * 100, "ko-", ms=5, label="Rynek (FRED)", lw=2)
    ax.plot(maturities, hw_rates * 100, "b^--", ms=5, label="Hull-White 1F", lw=1.5)
    ax.plot(maturities, vasicek_rates * 100, "rs--", ms=5, label="Vasicek", lw=1.5)

    ax.set_xlabel("Zapadalność [lata]", fontsize=11)
    ax.set_ylabel("Zero-rate [%]", fontsize=11)
    ax.set_title("Dopasowanie do krzywej dochodowości: HW vs Vasicek", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(str(OUTPUT_DIR / "hw_vs_vasicek.png"), dpi=150)
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  6.  Historia krótkiej stopy
# ══════════════════════════════════════════════════════════════════════════

def plot_short_rate_history(
    short_rate: pd.Series, save: bool = True
) -> plt.Figure:
    """Wykres historycznej krótkiej stopy (3M Treasury)."""
    _ensure_output()
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(short_rate.index, short_rate.values * 100, color="navy", lw=1)
    ax.fill_between(short_rate.index, 0, short_rate.values * 100, alpha=0.1, color="navy")
    ax.set_ylabel("Stopa [%]")
    ax.set_title("Krótka stopa procentowa – 3M US Treasury (2020–2024)", fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(str(OUTPUT_DIR / "short_rate_history.png"), dpi=150)
    return fig
