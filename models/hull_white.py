"""
Jednoczynnikowy model Hulla-White'a (HW 1F)
============================================

Dynamika krótkiej stopy:
    dr(t) = [θ(t) − a·r(t)] dt + σ dW(t)

Wzory analityczne:
  • Cena ZCB:           P(t,T) = A(t,T)·exp(−B(t,T)·r(t))
  • Opcja na ZCB:       formuła Jamshidiana
  • θ(t):               wyznaczone tak, by model odtwarzał rynkową krzywą
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


class HullWhite1F:
    """
    Parameters
    ----------
    a : float
        Szybkość powrotu do średniej (mean-reversion speed).
    sigma : float
        Zmienność krótkiej stopy.
    curve : YieldCurve
        Obiekt krzywej dochodowości – musi udostępniać metody:
            .P(T)  → float   (discount factor)
            .f(T)  → float   (instantaneous forward rate)
            .forward_rate_derivative(T) → float
    """

    def __init__(self, a: float, sigma: float, curve):
        self.a = a
        self.sigma = sigma
        self.curve = curve

    # ── θ(t) – dopasowanie do krzywej rynkowej ───────────────────────────

    def theta(self, t: float) -> float:
        """
        θ(t) = ∂f(0,t)/∂t + a·f(0,t) + (σ²/2a)·(1 − e^{−2at})
        """
        a, s = self.a, self.sigma
        dfdt = self.curve.forward_rate_derivative(t)
        f0t = self.curve.f(t)
        return dfdt + a * f0t + (s ** 2 / (2 * a)) * (1 - np.exp(-2 * a * t))

    # ── Bloki budujące: B i A ─────────────────────────────────────────────

    def B(self, t: float, T: float) -> float:
        """B(t,T) = [1 − exp(−a(T−t))] / a"""
        tau = T - t
        if abs(self.a) < 1e-12:
            return tau
        return (1 - np.exp(-self.a * tau)) / self.a

    def lnA(self, t: float, T: float) -> float:
        """
        ln A(t,T) = ln[P^M(0,T) / P^M(0,t)]
                     + B(t,T)·f^M(0,t)
                     − (σ²/4a)·B(t,T)²·(1 − e^{−2at})
        """
        a, s = self.a, self.sigma
        b = self.B(t, T)
        P0T = self.curve.P(T)
        P0t = self.curve.P(t) if t > 1e-8 else 1.0
        f0t = self.curve.f(max(t, 1e-8))
        return np.log(P0T / P0t) + b * f0t - (s ** 2 / (4 * a)) * b ** 2 * (
            1 - np.exp(-2 * a * t)
        )

    # ── Wycena ZCB ───────────────────────────────────────────────────────

    def zcb_price(self, t: float, T: float, r_t: float | np.ndarray):
        """P(t,T | r_t) = A(t,T)·exp(−B(t,T)·r_t)"""
        return np.exp(self.lnA(t, T) - self.B(t, T) * r_t)

    # ── Zmienność ceny obligacji σ_P ──────────────────────────────────────

    def sigma_p(self, t: float, T: float, S: float) -> float:
        """
        σ_P(t,T,S) – zmienność ln P(T,S) warunkowa na info w t.

        σ_P = B(T,S)·σ·√[(1 − e^{−2a(T−t)}) / (2a)]
        """
        a, s = self.a, self.sigma
        BTS = self.B(T, S)
        return BTS * s * np.sqrt((1 - np.exp(-2 * a * (T - t))) / (2 * a))

    # ── Europejska opcja na ZCB (Jamshidian) ─────────────────────────────

    def bond_option_price(
        self,
        t: float,
        T: float,
        S: float,
        K: float,
        r_t: float | None = None,
        option_type: str = "call",
    ) -> float:
        """
        Cena europejskiej opcji na ZCB.

        Parameters
        ----------
        t : czas bieżący
        T : czas wygaśnięcia opcji  (T > t)
        S : czas zapadalności obligacji  (S > T)
        K : cena wykonania
        r_t : bieżąca stopa krótka  (jeśli None → f(0,t))
        option_type : 'call' | 'put'
        """
        if r_t is None:
            r_t = self.curve.f(max(t, 1e-8))

        PtT = self.zcb_price(t, T, r_t)
        PtS = self.zcb_price(t, S, r_t)

        sp = self.sigma_p(t, T, S)
        if sp < 1e-14:
            # opcja wygasła lub brak zmienności
            payoff = max(PtS - K * PtT, 0.0)
            return payoff if option_type == "call" else max(K * PtT - PtS, 0.0)

        h = (1.0 / sp) * np.log(PtS / (K * PtT)) + sp / 2.0

        if option_type == "call":
            return PtS * norm.cdf(h) - K * PtT * norm.cdf(h - sp)
        else:
            return K * PtT * norm.cdf(-h + sp) - PtS * norm.cdf(-h)

    # ── Delta opcji ──────────────────────────────────────────────────────

    def bond_option_delta(
        self,
        t: float,
        T: float,
        S: float,
        K: float,
        r_t: float,
        option_type: str = "call",
    ) -> dict:
        """
        Zwraca słownik z deltami:
          'delta_bond' : ∂C/∂P(t,S)  –  ilość obligacji w hedge'u
          'delta_r'    : ∂C/∂r_t     –  wrażliwość na stopę
        """
        PtT = self.zcb_price(t, T, r_t)
        PtS = self.zcb_price(t, S, r_t)
        sp = self.sigma_p(t, T, S)

        if sp < 1e-14:
            # at expiry
            itm = float(PtS > K * PtT)
            sign = 1.0 if option_type == "call" else -1.0
            return {"delta_bond": sign * itm, "delta_r": 0.0}

        h = (1.0 / sp) * np.log(PtS / (K * PtT)) + sp / 2.0

        BtT = self.B(t, T)
        BtS = self.B(t, S)

        if option_type == "call":
            delta_bond = norm.cdf(h)
            delta_r = -BtS * PtS * norm.cdf(h) + K * BtT * PtT * norm.cdf(h - sp)
        else:
            delta_bond = norm.cdf(h) - 1.0
            delta_r = (
                -BtS * PtS * (norm.cdf(h) - 1.0)
                + K * BtT * PtT * (norm.cdf(h - sp) - 1.0)
            )

        return {"delta_bond": delta_bond, "delta_r": delta_r}

    # ── Wycena opcji na ZCB z t=0 (skrót) ────────────────────────────────

    def call_price(self, T: float, S: float, K: float) -> float:
        """C(0; T, S, K) – cena call z t=0."""
        return self.bond_option_price(0.0, T, S, K, option_type="call")

    def put_price(self, T: float, S: float, K: float) -> float:
        """P(0; T, S, K) – cena put z t=0."""
        return self.bond_option_price(0.0, T, S, K, option_type="put")

    def __repr__(self) -> str:
        return f"HullWhite1F(a={self.a:.4f}, σ={self.sigma:.4f})"
