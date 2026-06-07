"""
Model Vasicka – benchmark do porównania z Hull-White 1F
=======================================================

Dynamika:
    dr(t) = a·(b − r(t)) dt + σ dW(t)

Kluczowa różnica: b = const ⇒ model NIE odtwarza dokładnie
krzywej rynkowej (w odróżnieniu od HW).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


class Vasicek:
    """
    Parameters
    ----------
    a : float    – mean-reversion speed
    b : float    – długoterminowa średnia stopy
    sigma : float – zmienność
    """

    def __init__(self, a: float, b: float, sigma: float):
        self.a = a
        self.b = b
        self.sigma = sigma

    # ── Bloki ─────────────────────────────────────────────────────────────

    def B(self, t: float, T: float) -> float:
        tau = T - t
        if abs(self.a) < 1e-12:
            return tau
        return (1 - np.exp(-self.a * tau)) / self.a

    def A(self, t: float, T: float) -> float:
        """A(t,T) w Vasicku (analityczna)."""
        a, b, s = self.a, self.b, self.sigma
        Bval = self.B(t, T)
        tau = T - t
        lnA = (Bval - tau) * (a ** 2 * b - s ** 2 / 2) / a ** 2 - (
            s ** 2 * Bval ** 2
        ) / (4 * a)
        return np.exp(lnA)

    # ── Wycena ZCB ───────────────────────────────────────────────────────

    def zcb_price(self, t: float, T: float, r_t: float) -> float:
        return self.A(t, T) * np.exp(-self.B(t, T) * r_t)

    def implied_yield_curve(self, r0: float, maturities: np.ndarray) -> np.ndarray:
        """Zero-rates implikowane przez model (do porównania z rynkiem)."""
        prices = np.array([self.zcb_price(0, T, r0) for T in maturities])
        return -np.log(prices) / maturities

    # ── Opcja na ZCB ─────────────────────────────────────────────────────

    def sigma_p(self, t: float, T: float, S: float) -> float:
        BTS = self.B(T, S)
        return BTS * self.sigma * np.sqrt(
            (1 - np.exp(-2 * self.a * (T - t))) / (2 * self.a)
        )

    def bond_option_price(
        self,
        t: float,
        T: float,
        S: float,
        K: float,
        r_t: float,
        option_type: str = "call",
    ) -> float:
        PtT = self.zcb_price(t, T, r_t)
        PtS = self.zcb_price(t, S, r_t)
        sp = self.sigma_p(t, T, S)

        if sp < 1e-14:
            payoff = PtS - K * PtT
            return max(payoff, 0) if option_type == "call" else max(-payoff, 0)

        h = (1 / sp) * np.log(PtS / (K * PtT)) + sp / 2

        if option_type == "call":
            return PtS * norm.cdf(h) - K * PtT * norm.cdf(h - sp)
        else:
            return K * PtT * norm.cdf(-h + sp) - PtS * norm.cdf(-h)

    def __repr__(self) -> str:
        return f"Vasicek(a={self.a:.4f}, b={self.b:.4f}, σ={self.sigma:.4f})"
