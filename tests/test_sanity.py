"""
Sanity Check Testleri
=====================
Bu testler temel işlevselliği doğrular.
Özellikle log/simple return bileşikleme tutarlılığını kontrol eder.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.returns import calculate_simple_returns, calculate_log_returns
from src.backtest import calculate_metrics, calculate_turnover, calculate_drawdown


def test_simple_return_compounding():
    """
    Simple return ile equity curve doğru hesaplanmalı.
    
    Örnek:
    - Fiyatlar: [100, 110, 121]
    - Simple returns: [0.10, 0.10]
    - Equity: [1.10, 1.21]
    """
    # fiyat serisi olustur
    prices = pd.DataFrame({
        "AAPL": [100.0, 110.0, 121.0]
    }, index=pd.date_range("2024-01-01", periods=3))
    
    # simple returns
    returns = calculate_simple_returns(prices)
    
    # beklenen degerler
    expected_returns = pd.Series([0.10, 0.10], name="AAPL")
    
    # tolerans ile karsilastir
    np.testing.assert_array_almost_equal(
        returns["AAPL"].values,
        expected_returns.values,
        decimal=10
    )
    
    # equity curve hesapla
    equity = (1 + returns["AAPL"]).cumprod()
    expected_equity = pd.Series([1.10, 1.21])
    
    np.testing.assert_array_almost_equal(
        equity.values,
        expected_equity.values,
        decimal=10
    )
    print("✓ Simple return bileşikleme testi geçti!")


def test_log_return_compounding_consistency():
    """
    Log return ile de aynı sonuca ulaşılmalı (farklı formül ile).
    
    Log returns için: equity = exp(cumsum(log_returns))
    """
    prices = pd.DataFrame({
        "AAPL": [100.0, 110.0, 121.0]
    }, index=pd.date_range("2024-01-01", periods=3))
    
    # log returns
    log_returns = calculate_log_returns(prices)
    
    # log return ile equity (DOGRU YONTEM)
    equity_correct = np.exp(log_returns["AAPL"].cumsum())
    
    # beklenen equity
    expected_equity = pd.Series([1.10, 1.21])
    
    np.testing.assert_array_almost_equal(
        equity_correct.values,
        expected_equity.values,
        decimal=10
    )
    print("✓ Log return bileşikleme testi geçti!")


def test_metrics_calculation():
    """
    Metrik hesaplamaları doğru olmalı.
    """
    # degisken getiri serisi (sabit degil, yoksa std=0 olur!)
    np.random.seed(42)
    daily_returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # 252 gun
    
    metrics = calculate_metrics(daily_returns, risk_free_rate=0.0)
    
    # volatilite pozitif olmali
    assert metrics["yillik_volatilite"] > 0, "Volatilite sıfır olamaz"
    
    # toplam getiri hesaplanmali
    assert "toplam_getiri" in metrics
    
    # sharpe hesaplanmali (pozitif veya negatif olabilir)
    assert "sharpe_orani" in metrics
    
    # max drawdown negatif veya sifir olmali
    assert metrics["max_drawdown"] <= 0
    
    print("✓ Metrik hesaplama testi geçti!")


def test_turnover_calculation():
    """
    Turnover hesaplaması doğru olmalı.
    """
    # 3 rebalance, toplam degisim her birinde 0.2
    weights_history = pd.DataFrame({
        "AAPL": [0.5, 0.4, 0.5],
        "MSFT": [0.5, 0.6, 0.5]
    })
    
    turnover = calculate_turnover(weights_history)
    
    # turnover pozitif ve mantikli bir deger olmali
    # diff: row1-row0 = abs(-0.1) + abs(0.1) = 0.2
    # diff: row2-row1 = abs(0.1) + abs(-0.1) = 0.2
    # ortalama = 0.2
    print(f"  Hesaplanan turnover: {turnover:.4f}")
    
    assert turnover >= 0, "Turnover negatif olamaz"
    assert turnover <= 2.0, "Turnover çok yüksek (max 2.0)"


def test_drawdown_calculation():
    """
    Drawdown hesaplaması doğru olmalı.
    """
    # zirve 1.20, sonra 1.00'a düşüş = %16.67 drawdown
    equity = pd.Series([1.00, 1.10, 1.20, 1.00, 1.15])
    
    dd = calculate_drawdown(equity)
    
    # en buyuk drawdown 3. noktada (1.20 -> 1.00)
    max_dd = dd.min()
    expected_dd = (1.00 - 1.20) / 1.20  # -0.1667
    
    assert abs(max_dd - expected_dd) < 0.01
    
    print("✓ Drawdown hesaplama testi geçti!")


if __name__ == "__main__":
    print("=" * 50)
    print("SANITY CHECK TESTLERİ")
    print("=" * 50)
    
    test_simple_return_compounding()
    test_log_return_compounding_consistency()
    test_metrics_calculation()
    test_turnover_calculation()
    test_drawdown_calculation()
    
    print("=" * 50)
    print("✅ TÜM TESTLER GEÇTİ!")
    print("=" * 50)
