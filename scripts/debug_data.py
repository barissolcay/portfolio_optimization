"""
Debug script for data fetching
Bu dosya scripts/ klasöründe bulunur.
"""
import sys
import os

# Proje kökünü path'e ekle (scripts/ klasöründen bir üst dizin)
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.data import fetch_single_stock, fetch_stock_data

print("=" * 50)
print("VERI CEKME DEBUG TESTI")
print("=" * 50)

# Test 1: Tek hisse
print("\n1. Tek hisse testi (AAPL)")
try:
    prices, source = fetch_single_stock("AAPL", "2024-01-01", "2024-12-31")
    print(f"   Kaynak: {source}")
    print(f"   Satır sayısı: {len(prices)}")
    print(f"   İlk tarih: {prices.index.min()}")
    print(f"   Son tarih: {prices.index.max()}")
except Exception as e:
    print(f"   HATA: {type(e).__name__}: {e}")

# Test 2: Birden fazla hisse
print("\n2. Çoklu hisse testi (AAPL, MSFT, GOOGL)")
try:
    stock_prices, benchmark, meta = fetch_stock_data(
        ["AAPL", "MSFT", "GOOGL"],
        "2024-01-01",
        "2024-12-31"
    )
    print(f"   Stock satır: {len(stock_prices)}")
    print(f"   Benchmark satır: {len(benchmark) if benchmark is not None else 'N/A'}")
    print(f"   Meta: {meta}")
except Exception as e:
    print(f"   HATA: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("TEST TAMAMLANDI")
print("=" * 50)
