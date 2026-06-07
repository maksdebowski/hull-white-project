"""
Konstrukcja ciągłej krzywej dochodowości z dyskretnych danych FRED.

Interpolacja splajnem kubicznym → funkcje:
  P(0,T)  – cena ZCB (discount factor)
  y(T)    – zero-rate (ciągłe składanie)
  f(0,T)  – chwilowa stopa forward
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from config import MATURITIES, YIELD_CURVE_SERIES


class YieldCurve:
    """
    Obiekt reprezentujący krzywą dochodowości na jeden dzień.

    Budowany z wektora par: (zapadalność, zero-rate w ciągłym składaniu).
    """

    def __init__(self, maturities: np.ndarray, zero_rates: np.ndarray):
        """
        Parameters
        ----------
        maturities : array, shape (n,)
            Zapadalności w latach (> 0).
        zero_rates : array, shape (n,)
            Ciągle składane zero-rates (ułamki dziesiętne).
        """
        idx = np.argsort(maturities)
        self.T = np.asarray(maturities, dtype=float)[idx]
        self.y = np.asarray(zero_rates, dtype=float)[idx]

        # Splajn na zero-rates (ekstrapolacja = flat)
        self._y_spline = CubicSpline(
            self.T, self.y, bc_type="clamped", extrapolate=True
        )

    # ── Fabryka z DataFrame (jeden wiersz) ────────────────────────────────

    @classmethod
    def from_dataframe_row(cls, row) -> "YieldCurve":
        """Utwórz YieldCurve z jednego wiersza DataFrame krzywej FRED."""
        mats, rates = [], []
        for label in YIELD_CURVE_SERIES:
            if label in row.index and np.isfinite(row[label]):
                mats.append(MATURITIES[label])
                rates.append(row[label])
        return cls(np.array(mats), np.array(rates))

    # ── Główne funkcje ────────────────────────────────────────────────────

    def zero_rate(self, T: float | np.ndarray) -> float | np.ndarray:
        """Ciągle składana zero-rate y(T)."""
        T = np.asarray(T, dtype=float)
        out = self._y_spline(np.maximum(T, 1e-6))
        return float(out) if out.ndim == 0 else out

    def discount(self, T: float | np.ndarray) -> float | np.ndarray:
        """Cena ZCB  P(0,T) = exp(-y(T)·T)."""
        T = np.asarray(T, dtype=float)
        return np.exp(-self.zero_rate(T) * T)

    def forward_rate(self, T: float | np.ndarray) -> float | np.ndarray:
        """
        Chwilowa stopa forward  f(0,T) = y(T) + T·y'(T)
        (pochodna z -d ln P / dT).
        """
        T = np.asarray(T, dtype=float)
        T_safe = np.maximum(T, 1e-6)
        y_val = self._y_spline(T_safe)
        y_der = self._y_spline(T_safe, 1)  # pierwsza pochodna
        return y_val + T_safe * y_der

    def forward_rate_derivative(self, T: float) -> float:
        """∂f(0,T)/∂T  –  potrzebne do θ(t) w Hull-White."""
        T_safe = max(T, 1e-6)
        y0 = self._y_spline(T_safe)
        y1 = self._y_spline(T_safe, 1)
        y2 = self._y_spline(T_safe, 2)
        return 2 * y1 + T_safe * y2

    # ── Wygodne callable do przekazania do modelu ─────────────────────────

    def P(self, T: float) -> float:
        """Callable: P(0,T)."""
        return float(self.discount(T))

    def f(self, T: float) -> float:
        """Callable: f(0,T)."""
        return float(self.forward_rate(T))

    def __repr__(self) -> str:
        return (
            f"YieldCurve(tenors={len(self.T)}, "
            f"y_short={self.y[0]:.4f}, y_long={self.y[-1]:.4f})"
        )
