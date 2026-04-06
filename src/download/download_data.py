import pandas as pd
import yfinance as yf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# CONFIG
CSV_FILE = ROOT / "data/raw/sp500_constituents.csv"        # used to fetch the S&P 500 tickers
START_DATE = "2000-01-01"
END_DATE = "2026-01-01"
DELAY_BETWEEN_REQUESTS = 0.05


# LOAD TICKERS

def load_tickers(csv_file: str) -> list:
    df = pd.read_csv(csv_file)
    tickers = df["Symbol"].tolist()         # convert column to list

    # Yahoo formatting fix (e.g. BRK.B → BRK-B)
    tickers = [t.replace(".", "-") for t in tickers]

    print(f"✓ Loaded {len(tickers)} tickers")
    return tickers


# DOWNLOAD DATA 

def download_prices(tickers: list, start: str, end: str) -> pd.DataFrame:

    print("Downloading price data from Yahoo Finance...")

    try:
        data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,       # already adjusted prices
        progress=True,
        group_by="ticker",
        threads=True
    )
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return None

    prices_list = []
    failed_tickers = []

    for ticker in tickers:
        try:
            series = data[ticker]["Close"].rename(ticker)
            prices_list.append(series)
        except KeyError:
            failed_tickers.append(ticker)

    if failed_tickers:
        print(f"\n⚠️  {len(failed_tickers)} stocks with no data (delisted/new): {failed_tickers[:5]}...")

    if not prices_list:
        raise ValueError("No data downloaded.")


    prices = pd.concat(prices_list, axis=1)
    # Clean structure
    prices = prices.sort_index()
    prices = prices.dropna(axis=1, how="all")
    prices = prices.astype(float)


    # Quality report
    print(f"\n{'='*60}")
    print(f"✓ Download successful!")
    print(f"{'='*60}")
    print(f"Stocks with data: {len(prices.columns)}/{len(tickers)}")
    print(f"Trading days: {len(prices):,}")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Shape: {prices.shape}")
    
    missing_pct = (prices.isna().sum().sum() / (len(prices) * len(prices.columns))) * 100
    print(f"Data completeness: {100 - missing_pct:.1f}%")
    print(f"{'='*60}\n")

    return prices



if __name__ == "__main__":

    tickers = load_tickers(CSV_FILE)
    prices = download_prices(tickers, start=START_DATE, end=END_DATE)

    if prices is not None:
        prices.to_parquet(ROOT / "data/raw/sp500_prices.parquet", compression="snappy")
        print(f"✓ Saved:sp500_prices.parquet")
        
        prices.to_csv("../../data/raw/sp500_prices.csv")
        print(f"✓ Saved: sp500_prices.csv")

    else:
        print("\n✗ Download failed. Check your connection and try again.")
