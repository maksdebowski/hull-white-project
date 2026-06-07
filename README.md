# Hull-White 1F: Wycena opcji na obligacje i Delta-Hedging (2020–2024)

Projekt implementuje **jednoczynnikowy model Hulla-White'a (HW 1F)** i wykorzystuje go do:

- **Wyceny** europejskiej opcji kupna (Call) na obligację zerokuponową (ZCB)
- **Kalibracji** do historycznych danych rynkowych z okresu pandemii i podwyżek stóp
- **Symulacji Monte Carlo** ścieżek krótkiej stopy
- **Backtestingu Delta-Hedgingu** – sprawdzenie skuteczności zabezpieczenia w praktyce
- **Porównania z modelem Vasicka** – dlaczego HW jest lepszy dla instrumentów dłużnych

## Model

$$dr_t = (\theta(t) - a \cdot r_t)\,dt + \sigma\,dW_t$$

gdzie $\theta(t)$ jest wyznaczane tak, by model **idealnie odtwarzał początkową krzywą dochodowości**.

## Struktura projektu

```
hull-white-project/
├── main.py                  # Orkiestrator – uruchamia cały pipeline
├── config.py                # Konfiguracja (parametry, daty, klucze API)
├── requirements.txt         # Zależności Python
├── .env.example             # Szablon pliku .env (klucz FRED API)
│
├── data/
│   ├── fred_data.py         # Pobieranie danych z FRED
│   └── yield_curve.py       # Interpolacja krzywej dochodowości
│
├── models/
│   ├── hull_white.py        # Model Hull-White 1F (analityka + wycena)
│   └── vasicek.py           # Model Vasicka (benchmark)
│
├── calibration/
│   └── calibrate.py         # Kalibracja a, σ (OLS + MLE + rolling)
│
├── simulation/
│   └── monte_carlo.py       # Euler-Maruyama MC + wycena opcji MC
│
├── hedging/
│   └── delta_hedge.py       # Delta-Hedging: MC backtest + historyczny
│
├── visualization/
│   └── plots.py             # Wykresy (3D krzywa, kalibracja, hedging, etc.)
│
└── output/                  # Generowane wykresy i wyniki
```

## Szybki start

### 1. Zainstaluj zależności

```bash
pip install -r requirements.txt
```

### 2. Skonfiguruj klucz FRED API

Zarejestruj się (darmowo) na [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html), a następnie:

```bash
cp .env.example .env
# Wklej swój klucz do .env
```

### 3. Uruchom pipeline

```bash
python main.py
```

## Dane

Źródło: **FRED (Federal Reserve Economic Data)**

| Symbol     | Opis              | Tenor |
| ---------- | ----------------- | ----- |
| DGS1MO     | Treasury 1-Month  | 1M    |
| DGS3MO     | Treasury 3-Month  | 3M    |
| DGS6MO     | Treasury 6-Month  | 6M    |
| DGS1–DGS30 | Treasury 1Y – 30Y | 1–30Y |

## Wyniki (output/)

Po uruchomieniu `main.py` w folderze `output/` pojawią się:

- **yield_curve_3d.html** – interaktywna powierzchnia 3D krzywej dochodowości
- **short_rate_history.png** – historia krótkiej stopy 3M Treasury
- **calibration_params.png** – parametry a(t), σ(t) z kroczącej kalibracji
- **mc_distribution.png** – rozkład payoffów MC vs cena analityczna
- **mc_hedge_errors.png** – histogram błędów replikacji z MC hedge
- **hedging_pnl.png** – historyczny backtest delta-hedgingu (2022)
- **hw_vs_vasicek.png** – porównanie dopasowania krzywej HW vs Vasicek

## Technologie

- Python 3.10+
- NumPy, SciPy, Pandas
- Matplotlib, Plotly
- fredapi (FRED API)
- python-dotenv
