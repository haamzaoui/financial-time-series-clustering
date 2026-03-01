import pandas as pd
import numpy as np

# Load your CSV
prices = pd.read_csv("raw/sp500_prices.csv", index_col=0, parse_dates=True)

# 1. Basic Info
print("="*70)
print("BASIC INFORMATION")
print("="*70)
print(f"Shape: {prices.shape} ({prices.shape[0]} dates × {prices.shape[1]} stocks)")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"Total days: {len(prices):,}")
print(f"Data type: {prices.dtypes.unique()}")

# 2. Missing Values
print("\n" + "="*70)
print("MISSING VALUES")
print("="*70)
missing_total = prices.isna().sum().sum()
missing_pct = (missing_total / (len(prices) * len(prices.columns))) * 100
print(f"Total missing: {missing_total:,} ({missing_pct:.2f}%)")
print(f"Stocks with missing data: {(prices.isna().sum() > 0).sum()} / {len(prices.columns)}")

# Stocks with most missing values
print("\nTop 10 stocks with most missing values:")
missing_by_stock = prices.isna().sum().sort_values(ascending=False)
for stock, count in missing_by_stock.head(10).items():
    pct = (count / len(prices)) * 100
    print(f"  {stock:6s}: {count:5d} ({pct:5.1f}%)")

# 3. Data Completeness
print("\n" + "="*70)
print("DATA COMPLETENESS")
print("="*70)
completeness = (1 - prices.isna().sum() / len(prices)) * 100
print(f"Average completeness: {completeness.mean():.1f}%")
print(f"Min completeness: {completeness.min():.1f}%")
print(f"Max completeness: {completeness.max():.1f}%")

# 4. Price Statistics
print("\n" + "="*70)
print("PRICE STATISTICS")
print("="*70)
print(f"Min price: ${prices.min().min():.2f}")
print(f"Max price: ${prices.max().max():.2f}")
print(f"Mean price: ${prices.mean().mean():.2f}")
print(f"Median price: ${prices.median().median():.2f}")

# 5. Price Range per Stock
print("\n" + "="*70)
print("PRICE RANGE BY STOCK")
print("="*70)
print("Top 10 highest price stocks:")
max_prices = prices.max().sort_values(ascending=False).head(10)
for stock, price in max_prices.items():
    print(f"  {stock:6s}: ${price:,.2f}")

print("\nTop 10 lowest price stocks:")
min_prices = prices.min().sort_values(ascending=True).head(10)
for stock, price in min_prices.items():
    print(f"  {stock:6s}: ${price:,.2f}")

# 6. Check for Duplicates
print("\n" + "="*70)
print("DUPLICATES CHECK")
print("="*70)
duplicates = prices.index.duplicated().sum()
print(f"Duplicate dates: {duplicates}")
if duplicates == 0:
    print("✓ No duplicate dates")

# 7. Check for Gaps
print("\n" + "="*70)
print("TRADING DAY GAPS")
print("="*70)
date_diffs = prices.index.to_series().diff().dt.days
normal_gaps = (date_diffs == 1).sum()  # Normal: weekdays only
weekend_gaps = ((date_diffs == 3) | (date_diffs == 2)).sum()  # Weekend
holiday_gaps = (date_diffs > 3).sum()  # Holidays

print(f"Normal weekday gaps: {normal_gaps:,} (expected ~4,500)")
print(f"Weekend gaps: {weekend_gaps:,} (expected ~1,300)")
print(f"Holiday/multi-day gaps: {holiday_gaps:,} (expected ~30-50)")

# 8. Outliers Check
print("\n" + "="*70)
print("OUTLIERS CHECK")
print("="*70)

# Check for zero or negative prices
zero_prices = (prices <= 0).sum().sum()
print(f"Zero or negative prices: {zero_prices}")
if zero_prices == 0:
    print("✓ No invalid prices")

# Check for extreme price jumps (> 20% in one day)
daily_returns = prices.pct_change()
extreme_jumps = (daily_returns.abs() > 0.20).sum().sum()
print(f"Extreme daily jumps (>20%): {extreme_jumps}")

# 9. Data Type Check
print("\n" + "="*70)
print("DATA TYPE CHECK")
print("="*70)
print(f"Data types: {prices.dtypes.unique()}")
if prices.dtypes.unique()[0] == np.float64:
    print("✓ All prices are numeric (float64)")
else:
    print("⚠️  Some prices might be stored as text")
    non_numeric = prices.apply(pd.to_numeric, errors='coerce').isna().sum().sum()
    print(f"   Non-numeric values: {non_numeric}")

# 10. Sample Data
print("\n" + "="*70)
print("SAMPLE DATA")
print("="*70)
print("\nFirst 5 rows (first 5 stocks):")
print(prices.iloc[:5, :5])
print("\nLast 5 rows (first 5 stocks):")
print(prices.iloc[-5:, :5])

# 11. Summary Report
print("\n" + "="*70)
print("SUMMARY REPORT")
print("="*70)
print(f"✓ Shape: {prices.shape}")
print(f"✓ Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"✓ Data completeness: {completeness.mean():.1f}%")
print(f"✓ Missing values: {missing_pct:.2f}%")
print(f"✓ Invalid prices: {zero_prices}")
print(f"✓ Extreme jumps: {extreme_jumps}")

if missing_pct < 5 and zero_prices == 0 and extreme_jumps < 100:
    print("\n✓✓ DATA QUALITY: GOOD")
else:
    print("\n⚠️  DATA QUALITY: ISSUES DETECTED")