"""
Demo Veri Oluşturucu
====================
Bu script, yfinance çalışmadığında kullanılmak üzere
gerçekçi demo verileri oluşturur.

Bu dosya scripts/ klasöründe bulunur.
Veriler rastgele ama gerçekçi fiyat hareketleri içerir.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

# cache klasoru (scripts/ klasöründen bir üst dizin)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")

# hisse bilgileri (baslangic fiyatlari ve volatiliteler)
STOCK_INFO = {
    "AAPL": {"start_price": 150, "annual_return": 0.15, "annual_vol": 0.25},
    "MSFT": {"start_price": 300, "annual_return": 0.18, "annual_vol": 0.22},
    "GOOGL": {"start_price": 120, "annual_return": 0.12, "annual_vol": 0.28},
    "AMZN": {"start_price": 140, "annual_return": 0.20, "annual_vol": 0.30},
    "META": {"start_price": 280, "annual_return": 0.25, "annual_vol": 0.35},
    "SPY": {"start_price": 420, "annual_return": 0.10, "annual_vol": 0.15},  # benchmark
}


def generate_realistic_prices(
    start_price: float,
    annual_return: float,
    annual_vol: float,
    n_days: int,
    seed: int = None
) -> np.ndarray:
    """
    Geometric Brownian Motion ile gerçekçi fiyat serisi oluşturur.
    
    Bu model:
    - Rastgele yürüyüş (random walk) içerir
    - Ortalama bir getiri eğilimi vardır
    - Volatilite gerçek piyasa koşullarına yakındır
    """
    if seed is not None:
        np.random.seed(seed)
    
    # gunluk parametreler
    daily_return = annual_return / 252
    daily_vol = annual_vol / np.sqrt(252)
    
    # rastgele soklar
    shocks = np.random.normal(daily_return, daily_vol, n_days)
    
    # log getirilerden fiyat serisi
    log_returns = np.cumsum(shocks)
    prices = start_price * np.exp(log_returns)
    
    return prices


def generate_demo_data(
    start_date: str = "2022-01-03",
    end_date: str = "2024-12-31",
    tickers: list = None
):
    """
    Demo veri seti oluşturur ve cache klasörüne kaydeder.
    """
    if tickers is None:
        tickers = list(STOCK_INFO.keys())
    
    # tarih araligi
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # is gunleri (hafta sonu haric)
    dates = pd.bdate_range(start, end)
    n_days = len(dates)
    
    print(f"Demo veri oluşturuluyor: {n_days} gün, {len(tickers)} hisse")
    
    # cache klasorunu olustur
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    for i, ticker in enumerate(tickers):
        if ticker not in STOCK_INFO:
            print(f"⚠️ {ticker} için bilgi yok, atlanıyor")
            continue
        
        info = STOCK_INFO[ticker]
        
        # fiyat serisi olustur (her hisse icin farkli seed)
        prices = generate_realistic_prices(
            info["start_price"],
            info["annual_return"],
            info["annual_vol"],
            n_days,
            seed=42 + i
        )
        
        # DataFrame olustur
        df = pd.DataFrame({ticker: prices}, index=dates)
        
        # cache'e kaydet
        filename = f"{ticker}_{start_date}_{end_date}.csv"
        filepath = os.path.join(CACHE_DIR, filename)
        df.to_csv(filepath)
        
        print(f"✓ {ticker} kaydedildi: {filename}")
    
    print(f"\n✅ Demo veriler hazır: {CACHE_DIR}")
    return True


if __name__ == "__main__":
    # demo verileri olustur
    generate_demo_data(
        start_date="2022-01-03",
        end_date="2024-12-31",
        tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "META", "SPY"]
    )
