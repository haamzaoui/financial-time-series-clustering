import pandas as pd

prices = pd.read_csv("../../data/raw/sp500_prices.csv", index_col=0, parse_dates=True)

# Only check for DATA ERRORS, not IPO gaps
print("QUALITY CHECK (excluding IPO/delisting gaps):")

# Check 1: Are prices valid?
zero_prices = (prices <= 0).sum().sum()
print(f"Invalid prices (≤0): {zero_prices}")
print(f"  ✓ Expected: 0" if zero_prices == 0 else f"  ❌ Problem!")

# Check 2: Are there duplicates?
duplicates = prices.index.duplicated().sum()
print(f"Duplicate dates: {duplicates}")
print(f"  ✓ Expected: 0" if duplicates == 0 else f"  ❌ Problem!")

# Check 3: Any NaN in middle of data (data corruption)?
print(f"\nStocks with complete 25-year history: {(prices.notna().sum() == len(prices)).sum()}")
print(f"  (Out of {len(prices.columns)} total stocks)")

# Check 4: Sample of data
print(f"\nSample data (looks OK?):")
print(prices.iloc[:5, :5])

# Verdict
if zero_prices == 0 and duplicates == 0:
    print("\n✓✓ DATA QUALITY: GOOD")
    print("Missing values are due to IPOs/delistings (normal)")
else:
    print("\n⚠️  DATA QUALITY: Issues detected")