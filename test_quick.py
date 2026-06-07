"""Szybki test poprawności modelu na syntetycznej krzywej."""
import sys
sys.path.insert(0, ".")

import numpy as np
from data.yield_curve import YieldCurve
from models.hull_white import HullWhite1F
from models.vasicek import Vasicek

# Syntetyczna krzywa: flat 4%
mats = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
rates = np.full_like(mats, 0.04)
curve = YieldCurve(mats, rates)

hw = HullWhite1F(a=0.1, sigma=0.01, curve=curve)
r0 = 0.04

print(f"P(0,1) = {curve.P(1):.6f}")
print(f"P(0,5) = {curve.P(5):.6f}")
print(f"f(0,1) = {curve.f(1):.6f}")
print(f"ZCB HW P(0,5) = {hw.zcb_price(0, 5, r0):.6f}")
print(f"Call(T=1,S=5,K=0.85) = {hw.call_price(1, 5, 0.85):.6f}")
print(f"Put (T=1,S=5,K=0.85) = {hw.put_price(1, 5, 0.85):.6f}")

vas = Vasicek(a=0.1, b=0.04, sigma=0.01)
print(f"Call Vasicek = {vas.bond_option_price(0, 1, 5, 0.85, r0, 'call'):.6f}")

# Sanity checks
assert abs(curve.P(1) - np.exp(-0.04)) < 1e-6, "Discount factor error"
assert hw.call_price(1, 5, 0.85) > 0, "Call price must be positive"
assert hw.put_price(1, 5, 0.85) >= 0, "Put price must be non-negative"

# Put-call parity: C - P = P(0,S) - K*P(0,T)
C = hw.call_price(1, 5, 0.85)
P = hw.put_price(1, 5, 0.85)
parity_lhs = C - P
parity_rhs = curve.P(5) - 0.85 * curve.P(1)
print(f"Put-call parity: C-P={parity_lhs:.6f}, P(S)-K*P(T)={parity_rhs:.6f}, diff={abs(parity_lhs-parity_rhs):.2e}")
assert abs(parity_lhs - parity_rhs) < 1e-6, "Put-call parity violated!"

print("\nWszystkie testy PASSED")
