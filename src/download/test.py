import pandas as pd
import yfinance as yf
import time


# CONFIG
CSV_FILE = "sp500_constituents.csv"         # used to fetch the S&P 500 tickers
START_DATE = "2025-01-01"
END_DATE = "2026-01-01"
DELAY = 0.1


# LOAD TICKERS
# ==========================================================

def load_tickers(csv_file: str) -> list:
    df = pd.read_csv(csv_file)
    tickers = df["Symbol"].tolist()         # convert column to list

    # Yahoo formatting fix (e.g. BRK.B → BRK-B)
    tickers = [t.replace(".", "-") for t in tickers]

    print(f"✓ Loaded {len(tickers)} tickers")
    return tickers

load_tickers(CSV_FILE)