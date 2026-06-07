"""
Konfiguracja projektu Hull-White 1F
Wycena opcji na obligacje i symulacja Delta-Hedgingu (2020-2024)
"""
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Ścieżki ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR = PROJECT_ROOT / "data_cache"

# ── FRED API ──────────────────────────────────────────────────────────────
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# ── Okres analizy ─────────────────────────────────────────────────────────
START_DATE = date(2020, 1, 1)
END_DATE = date(2024, 12, 31)

# ── Symbole FRED: krzywa dochodowości US Treasury ─────────────────────────
YIELD_CURVE_SERIES = {
    "1M":  "DGS1MO",
    "3M":  "DGS3MO",
    "6M":  "DGS6MO",
    "1Y":  "DGS1",
    "2Y":  "DGS2",
    "3Y":  "DGS3",
    "5Y":  "DGS5",
    "7Y":  "DGS7",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30",
}

# Odpowiadające zapadalności w latach
MATURITIES = {
    "1M":  1 / 12,
    "3M":  3 / 12,
    "6M":  6 / 12,
    "1Y":  1.0,
    "2Y":  2.0,
    "3Y":  3.0,
    "5Y":  5.0,
    "7Y":  7.0,
    "10Y": 10.0,
    "20Y": 20.0,
    "30Y": 30.0,
}

# ── Parametry opcji na ZCB (domyślne) ────────────────────────────────────
OPTION_MATURITY = 1.0       # T  – czas do wygaśnięcia opcji (lata)
BOND_MATURITY = 5.0         # S  – czas do zapadalności obligacji (lata)
STRIKE_PRICE = 0.96         # K  – cena wykonania (ATM-ish)

# ── Monte Carlo ──────────────────────────────────────────────────────────
N_PATHS = 10_000
N_STEPS_PER_YEAR = 252      # kroki dzienne

# ── Delta-Hedging ────────────────────────────────────────────────────────
HEDGE_FREQUENCY = "weekly"  # 'daily' | 'weekly'
NOTIONAL = 1_000_000        # wartość nominalna portfela

# ── Kalibracja ───────────────────────────────────────────────────────────
CALIBRATION_WINDOW = 60     # okno kroczącej kalibracji (dni robocze)
SHORT_RATE_TENOR = "3M"     # tenor proxy dla krótkiej stopy r(t)
