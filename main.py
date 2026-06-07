"""
Hull-White 1F – Wycena i Delta-Hedging opcji na obligacje
==========================================================

Pipeline:
  1. Pobieranie danych (FRED)
  2. Budowa krzywej dochodowości
  3. Krocząca kalibracja (a, σ)
  4. Wycena opcji na ZCB (analityczna)
  5. Monte Carlo: wycena + weryfikacja
  6. Delta-Hedging: backtest MC + historyczny
  7. Porównanie Hull-White vs Vasicek
  8. Wizualizacja wyników

Uruchomienie:
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Dodaj root projektu do sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BOND_MATURITY,
    HEDGE_FREQUENCY,
    N_PATHS,
    N_STEPS_PER_YEAR,
    NOTIONAL,
    OPTION_MATURITY,
    OUTPUT_DIR,
    STRIKE_PRICE,
)
from data.fred_data import (
    download_yield_curve_data,
    get_short_rate_proxy,
    maturity_array,
)
from data.yield_curve import YieldCurve
from calibration.calibrate import (
    calibrate_hw_historical,
    calibrate_hw_mle,
    rolling_calibration,
)
from models.hull_white import HullWhite1F
from models.vasicek import Vasicek
from simulation.monte_carlo import mc_option_price, simulate_short_rate
from hedging.delta_hedge import run_delta_hedge_mc, run_delta_hedge_historical
from visualization.plots import (
    plot_calibration_params,
    plot_hedging_pnl,
    plot_hw_vs_vasicek,
    plot_mc_distribution,
    plot_mc_hedge_errors,
    plot_short_rate_history,
    plot_yield_curve_3d,
)


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("  HULL-WHITE 1F: Wycena opcji na ZCB i Delta-Hedging (2020-2024)")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════════
    # KROK 1: Pobieranie danych
    # ══════════════════════════════════════════════════════════════════════
    print("\n[1/8] Pobieranie danych z FRED...")
    yield_df = download_yield_curve_data()
    short_rate = get_short_rate_proxy(yield_df)

    print(f"      Okres: {yield_df.index[0].date()} → {yield_df.index[-1].date()}")
    print(f"      Obserwacje: {len(yield_df)}")
    print(f"      Krótka stopa (3M): {short_rate.iloc[-1]*100:.2f}%")

    # ══════════════════════════════════════════════════════════════════════
    # KROK 2: Wizualizacja danych
    # ══════════════════════════════════════════════════════════════════════
    print("\n[2/8] Generowanie wykresów danych...")
    plot_short_rate_history(short_rate)
    fig_3d = plot_yield_curve_3d(yield_df)
    print("      ✓ short_rate_history.png")
    print("      ✓ yield_curve_3d.html + .png")

    # ══════════════════════════════════════════════════════════════════════
    # KROK 3: Kalibracja
    # ══════════════════════════════════════════════════════════════════════
    print("\n[3/8] Kalibracja modelu Hull-White 1F...")

    # (a) Kalibracja na całym okresie
    params_ols = calibrate_hw_historical(short_rate)
    params_mle = calibrate_hw_mle(short_rate)

    print(
        f"      OLS:  a={params_ols['a']:.4f},  σ={params_ols['sigma']:.4f},  "
        f"b={params_ols['b_long_run']:.4f}"
    )
    print(
        f"      MLE:  a={params_mle['a']:.4f},  σ={params_mle['sigma']:.4f},  "
        f"b={params_mle['b_long_run']:.4f}"
    )

    # (b) Krocząca kalibracja
    cal_rolling = rolling_calibration(short_rate, method="ols")
    plot_calibration_params(cal_rolling)
    print(f"      ✓ Krocząca kalibracja: {len(cal_rolling)} punktów")
    print("      ✓ calibration_params.png")

    # Użyjemy parametrów MLE
    a_cal = params_mle["a"]
    sigma_cal = params_mle["sigma"]
    b_cal = params_mle["b_long_run"]

    # ══════════════════════════════════════════════════════════════════════
    # KROK 4: Budowa modelu i wycena opcji
    # ══════════════════════════════════════════════════════════════════════
    print("\n[4/8] Wycena opcji na ZCB...")

    # Krzywa z ostatniego dnia
    curve_last = YieldCurve.from_dataframe_row(yield_df.iloc[-1])
    hw_model = HullWhite1F(a_cal, sigma_cal, curve_last)

    r0 = curve_last.f(1e-4)
    T, S = OPTION_MATURITY, BOND_MATURITY

    # Strike ATM-forward: K = P(0,S) / P(0,T)  →  opcja blisko at-the-money
    K_atm = curve_last.P(S) / curve_last.P(T)
    K = round(K_atm, 4)
    print(f"      ATM-forward strike: K = P(0,S)/P(0,T) = {K_atm:.6f} → K={K}")

    call_hw = hw_model.call_price(T, S, K)
    put_hw = hw_model.put_price(T, S, K)

    print(f"      Parametry: T={T}Y, S={S}Y, K={K}")
    print(f"      r(0) = {r0*100:.2f}%")
    print(f"      P(0,T) = {curve_last.P(T):.6f}")
    print(f"      P(0,S) = {curve_last.P(S):.6f}")
    print(f"      Call HW = {call_hw:.6f}")
    print(f"      Put  HW = {put_hw:.6f}")
    print(f"      Wartość nominalna: Call = {call_hw*NOTIONAL:,.2f} PLN")

    # ══════════════════════════════════════════════════════════════════════
    # KROK 5: Monte Carlo – weryfikacja
    # ══════════════════════════════════════════════════════════════════════
    print("\n[5/8] Monte Carlo – weryfikacja ceny analitycznej...")

    mc = mc_option_price(
        hw_model,
        r0,
        T,
        S,
        K,
        n_paths=N_PATHS,
        n_steps=int(T * N_STEPS_PER_YEAR),
    )
    print(f"      MC price   = {mc['price']:.6f}  ± {mc['std_error']:.6f}")
    print(f"      Analytical = {mc['analytical']:.6f}")
    print(f"      Różnica    = {abs(mc['price']-mc['analytical']):.6f}")

    plot_mc_distribution(mc)
    print("      ✓ mc_distribution.png")

    # ══════════════════════════════════════════════════════════════════════
    # KROK 6: Delta-Hedging (MC)
    # ══════════════════════════════════════════════════════════════════════
    print("\n[6/8] Delta-Hedging – backtest MC...")

    hedge_mc = run_delta_hedge_mc(
        hw_model,
        r0,
        T,
        S,
        K,
        n_paths=5000,
        n_steps=52,  # tygodniowy rebalans
    )
    print(f"      Premia opcji     = {hedge_mc['initial_premium']:.6f}")
    print(f"      MAE (błąd hedge) = {hedge_mc['mean_abs_error']:.6f}")
    print(f"      σ (błąd hedge)   = {hedge_mc['std_error']:.6f}")

    plot_mc_hedge_errors(hedge_mc)
    print("      ✓ mc_hedge_errors.png")

    # ══════════════════════════════════════════════════════════════════════
    # KROK 7: Delta-Hedging (historyczny)
    # ══════════════════════════════════════════════════════════════════════
    print("\n[7/8] Delta-Hedging – backtest historyczny...")

    # Wybierz kilka okien startowych
    start_dates_target = ["2020-06-01", "2021-06-01", "2022-01-03", "2023-01-03"]

    for sd in start_dates_target:
        try:
            idx = yield_df.index.searchsorted(pd.Timestamp(sd))
            if idx >= len(yield_df):
                continue
            hedge_hist = run_delta_hedge_historical(
                yield_df,
                a_cal,
                sigma_cal,
                start_idx=idx,
                T_years=T,
                S_years=S,
                K=K,
                hedge_freq=HEDGE_FREQUENCY,
            )
            print(
                f"      Start {sd}: payoff={hedge_hist['option_payoff']:.6f}, "
                f"hedge_err={hedge_hist['hedge_error']:.6f}, "
                f"premia={hedge_hist['initial_premium']:.6f}"
            )

            if sd == "2022-01-03":
                # Zapisz szczegółowy wykres dla najbardziej interesującego okresu
                plot_hedging_pnl(hedge_hist)
                print("      ✓ hedging_pnl.png (okres 2022)")
        except Exception as exc:
            print(f"      ✗ Start {sd}: {exc}")

    # ══════════════════════════════════════════════════════════════════════
    # KROK 8: Porównanie HW vs Vasicek
    # ══════════════════════════════════════════════════════════════════════
    print("\n[8/8] Porównanie Hull-White vs Vasicek...")

    vasicek = Vasicek(a_cal, b_cal, sigma_cal)

    mats = maturity_array()
    market_rates = np.array([curve_last.zero_rate(m) for m in mats])

    # HW: odtwarza krzywą dokładnie (by design)
    hw_prices = np.array([hw_model.zcb_price(0, m, r0) for m in mats])
    hw_rates = -np.log(hw_prices) / mats

    # Vasicek: nie odtwarza krzywej
    vasicek_rates = vasicek.implied_yield_curve(r0, mats)

    # Ceny opcji
    call_vasicek = vasicek.bond_option_price(0, T, S, K, r0, "call")

    print(f"      Call HW      = {call_hw:.6f}")
    print(f"      Call Vasicek = {call_vasicek:.6f}")
    print(f"      Różnica      = {abs(call_hw - call_vasicek):.6f}")

    # Błąd dopasowania krzywej
    hw_rmse = np.sqrt(np.mean((hw_rates - market_rates) ** 2)) * 10000  # bps
    vas_rmse = np.sqrt(np.mean((vasicek_rates - market_rates) ** 2)) * 10000

    print(f"      RMSE krzywa HW      = {hw_rmse:.1f} bps")
    print(f"      RMSE krzywa Vasicek = {vas_rmse:.1f} bps")

    plot_hw_vs_vasicek(mats, market_rates, hw_rates, vasicek_rates)
    print("      ✓ hw_vs_vasicek.png")

    # ══════════════════════════════════════════════════════════════════════
    # PODSUMOWANIE
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PODSUMOWANIE")
    print("=" * 70)
    print(f"  Model:          Hull-White 1F  (a={a_cal:.4f}, σ={sigma_cal:.4f})")
    print(f"  Okres:          {yield_df.index[0].date()} – {yield_df.index[-1].date()}")
    print(f"  Opcja Call:     T={T}Y, S={S}Y, K={K}")
    print(f"  Cena (analyt.): {call_hw:.6f}")
    print(f"  Cena (MC):      {mc['price']:.6f} ± {mc['std_error']:.6f}")
    print(f"  Hedge MAE (MC): {hedge_mc['mean_abs_error']:.6f}")
    print(f"  Vasicek RMSE:   {vas_rmse:.1f} bps  (HW: {hw_rmse:.1f} bps)")
    print(f"\n  Wyniki zapisano w: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
