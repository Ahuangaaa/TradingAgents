"""Dev smoke script — requires ``TUSHARE_TOKEN`` / ``TUSHARE_API_KEY`` and network."""
import time

from tradingagents.dataflows.tushare_data import get_tushare_indicators

print("Tushare + stockstats (A-share example, 10-day lookback):")
start_time = time.time()
result = get_tushare_indicators("600519", "macd", "2024-11-01", 10)
end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f}s, length: {len(result)} chars")
print(result)
