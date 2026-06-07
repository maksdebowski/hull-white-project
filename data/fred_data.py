"""
Pobieranie i cache'owanie danych z FRED (Federal Reserve Economic Data).
Źródło: https://fred.stlouisfed.org
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fredapi import Fred

from config import (
    DATA_DIR,
    END_DATE,
    FRED_API_KEY,
    MATURITIES,
    SHORT_RATE_TENOR,
    START_DATE,
    YIELD_CURVE_SERIES,
)


def _get_fred() -> Fred:
    if not FRED_API_KEY:
        raise ValueError(
            "Brak klucza API FRED!  Ustaw zmienną FRED_API_KEY w pliku .env\n"
            "Rejestracja (darmowa): https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return Fred(api_key=FRED_API_KEY)


def _cache_path(name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"{name}.pkl"


# ── Krzywa dochodowości ───────────────────────────────────────────────────

def download_yield_curve_data(
    start=None, end=None, *, use_cache: bool = True
) -> pd.DataFrame:
    """
    Pobiera dzienne stopy Treasury z FRED i zwraca DataFrame
    z kolumnami = tenory, wartości w ułamku dziesiętnym (np. 0.05 = 5%).
    """
    cache = _cache_path("yield_curve")
    if use_cache and cache.exists():
        df = pd.read_pickle(cache)
        print(f"[DATA] Wczytano krzywą z cache ({len(df)} wierszy)")
        return df

    start = start or START_DATE
    end = end or END_DATE
    fred = _get_fred()

    frames: dict[str, pd.Series] = {}
    for label, sid in YIELD_CURVE_SERIES.items():
        try:
            s = fred.get_series(sid, start, end)
            frames[label] = s
            print(f"  ✓ {sid:10s} ({label})")
        except Exception as exc:
            print(f"  ✗ {sid:10s} ({label}): {exc}")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    # procentowa → ułamek dziesiętny
    df = df / 100.0

    # wypełnij weekendy/święta + usuń wiersze z brakującymi danymi
    df = df.ffill().dropna()

    df.to_pickle(cache)
    print(f"[DATA] Pobrano i zapisano {len(df)} wierszy krzywej dochodowości")
    return df


# ── Proxy krótkiej stopy r(t) ────────────────────────────────────────────

def get_short_rate_proxy(
    yield_data: pd.DataFrame, tenor: str = SHORT_RATE_TENOR
) -> pd.Series:
    """Zwraca szereg czasowy krótkiej stopy (proxy = stopa 3M Treasury)."""
    if tenor in yield_data.columns:
        return yield_data[tenor].copy().rename("r_t")
    # fallback: najkrótszy dostępny tenor
    return yield_data.iloc[:, 0].copy().rename("r_t")


# ── Pomocnicze ────────────────────────────────────────────────────────────

def maturity_array() -> np.ndarray:
    """Wektor zapadalności (lata) odpowiadający kolumnom krzywej."""
    return np.array([MATURITIES[k] for k in YIELD_CURVE_SERIES])
