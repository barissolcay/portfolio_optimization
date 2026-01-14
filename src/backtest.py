"""
Walk-Forward Backtest Modülü
============================
Bu modül geriye dönük test (backtest) yapar.
Optimizasyon stratejisinin geçmişte nasıl çalışacağını simüle eder.

Önemli: Look-Ahead Bias önlendi!
- Kovaryans ve ortalama SADECE eğitim penceresi ile hesaplanır
- Test dönemi verileri asla eğitim aşamasında kullanılmaz

Backtest Protokolü:
- Eğitim penceresi: varsayılan 252 gün (1 yıl)
- Hold dönemi: varsayılan 21 gün (1 ay)
- Rebalance: Aylık
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Callable, Dict, List
from dataclasses import dataclass

# bu moduldeki diger fonksiyonlari import et
from .returns import (
    calculate_log_returns,
    calculate_simple_returns,
    estimate_covariance_ledoit_wolf,
    calculate_expected_returns,
    TRADING_DAYS_PER_YEAR
)
from .optimize import (
    minimize_variance,
    maximize_sharpe,
    equal_weight_portfolio,
    portfolio_return,
    portfolio_volatility,
    DEFAULT_RISK_FREE_RATE,
    risk_parity_weights
)


# ============================================
# STRES DÖNEMLERİ TANIMLARI
# ============================================

STRESS_PERIODS = {
    "COVID-19 Krizi": {
        "start": "2020-02-19",
        "end": "2020-03-23",
        "description": "COVID-19 pandemisi kaynaklı küresel satış"
    },
    "2022 Ayı Piyasası": {
        "start": "2022-01-03",
        "end": "2022-10-12",
        "description": "Enflasyon ve faiz artışı kaynaklı düşüş"
    },
    "2018 Q4 Düzeltmesi": {
        "start": "2018-10-01",
        "end": "2018-12-24",
        "description": "Fed faiz artışı ve ticaret savaşı endişeleri"
    }
}


def identify_stress_periods_in_data(
    returns: pd.Series,
    stress_periods: dict = None
) -> dict:
    """
    Veri setinde hangi stres dönemlerinin bulunduğunu tespit eder.
    
    Args:
        returns: Getiri serisi
        stress_periods: Stres dönemleri sözlüğü
    
    Returns:
        Veri setinde bulunan stres dönemleri
    """
    if stress_periods is None:
        stress_periods = STRESS_PERIODS
    
    data_start = returns.index.min()
    data_end = returns.index.max()
    
    found_periods = {}
    
    for name, period in stress_periods.items():
        period_start = pd.to_datetime(period["start"])
        period_end = pd.to_datetime(period["end"])
        
        # Dönem veri aralığıyla örtüşüyor mu?
        if period_start <= data_end and period_end >= data_start:
            # Örtüşen kısmı al
            actual_start = max(period_start, data_start)
            actual_end = min(period_end, data_end)
            
            # En az 5 gün veri olmalı
            period_returns = returns[actual_start:actual_end]
            if len(period_returns) >= 5:
                found_periods[name] = {
                    **period,
                    "actual_start": actual_start,
                    "actual_end": actual_end,
                    "n_days": len(period_returns)
                }
    
    return found_periods


def calculate_stress_period_metrics(
    returns: pd.Series,
    period_start: str,
    period_end: str,
    risk_free_rate: float = 0.0
) -> dict:
    """
    Belirli bir dönem için performans metriklerini hesaplar.
    
    Args:
        returns: Getiri serisi
        period_start: Dönem başlangıcı
        period_end: Dönem bitişi
        risk_free_rate: Risksiz faiz oranı
    
    Returns:
        Dönem metrikleri
    """
    start = pd.to_datetime(period_start)
    end = pd.to_datetime(period_end)
    
    period_returns = returns[start:end]
    
    if len(period_returns) == 0:
        return None
    
    total_return = (1 + period_returns).prod() - 1
    volatility = period_returns.std() * np.sqrt(252)
    max_dd = calculate_max_drawdown((1 + period_returns).cumprod())
    worst_day = period_returns.min()
    best_day = period_returns.max()
    
    return {
        "toplam_getiri": total_return,
        "volatilite": volatility,
        "max_drawdown": max_dd,
        "en_kotu_gun": worst_day,
        "en_iyi_gun": best_day,
        "gun_sayisi": len(period_returns)
    }


@dataclass
class BacktestConfig:
    """Backtest konfigürasyon parametreleri."""
    train_window: int = 252  # egitim penceresi (gun)
    hold_period: int = 21    # hold donemi (gun)
    strategy: str = "max_sharpe"  # "min_variance" veya "max_sharpe"
    max_weight: float = 0.30
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE


@dataclass
class BacktestResult:
    """Backtest sonuçları."""
    equity_curve: pd.Series          # portfoy degeri serisi
    daily_returns: pd.Series         # gunluk getiriler
    weights_history: pd.DataFrame    # agirlik gecmisi
    metrics: Dict                    # performans metrikleri
    rebalance_dates: List            # yeniden dengeleme tarihleri
    turnover: float = 0.0            # ortalama turnover
    # YENİ ALANLAR
    transaction_costs: Dict = None   # işlem maliyetleri bilgisi
    net_daily_returns: pd.Series = None  # net getiriler (maliyet düşülmüş)
    net_equity_curve: pd.Series = None   # net equity curve
    net_metrics: Dict = None         # net performans metrikleri


def calculate_portfolio_returns(
    weights: np.ndarray,
    returns: pd.DataFrame
) -> pd.Series:
    """
    Verilen ağırlıklarla portföy getirisini hesaplar.
    
    Formül: r_p = sum(w_i * r_i)
    
    Args:
        weights: Ağırlık vektörü
        returns: Günlük getiri DataFrame'i
    
    Returns:
        Portföy günlük getiri serisi
    """
    # agirliklar ile getirileri carp ve topla
    portfolio_ret = (returns * weights).sum(axis=1)
    return portfolio_ret


def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
    """
    Drawdown (çekilme) serisini hesaplar.
    
    Drawdown: Mevcut değerin, o ana kadarki zirve değerden
    yüzde olarak ne kadar düştüğünü gösterir.
    
    Args:
        equity_curve: Portföy değer serisi (1'den başlar)
    
    Returns:
        Drawdown serisi (negatif değerler)
    """
    # o ana kadarki maksimum deger
    running_max = equity_curve.cummax()
    
    # drawdown = (mevcut - maksimum) / maksimum
    drawdown = (equity_curve - running_max) / running_max
    
    return drawdown


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maksimum drawdown (en büyük düşüş) hesaplar.
    
    Bu metrik, portföyün en kötü döneminde ne kadar
    değer kaybettiğini gösterir.
    
    Args:
        equity_curve: Portföy değer serisi
    
    Returns:
        Maksimum drawdown (negatif bir değer, örn: -0.15 = %15 düşüş)
    """
    drawdown = calculate_drawdown(equity_curve)
    return drawdown.min()


def calculate_turnover(weights_history: pd.DataFrame) -> float:
    """
    Ortalama portföy turnover'ını hesaplar.
    
    Turnover, her rebalance'da yapılan ağırlık değişikliklerinin
    toplamının ortalamasıdır. Yüksek turnover = yüksek işlem maliyeti.
    
    Args:
        weights_history: Ağırlık geçmişi DataFrame'i
    
    Returns:
        Ortalama turnover (0-2 arası, 1 = %100 turnover)
    """
    if len(weights_history) < 2:
        return 0.0
    
    # her satir arasindaki mutlak degisim
    changes = weights_history.diff().abs().sum(axis=1)
    
    # ilk satir NaN olacak, onu atla
    changes = changes.dropna()
    
    if len(changes) == 0:
        return 0.0
    
    return float(changes.mean())


def calculate_metrics(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> Dict:
    """
    Performans metriklerini hesaplar.
    
    Metrikler:
    - Yıllık getiri
    - Yıllık volatilite
    - Sharpe oranı
    - Maksimum drawdown
    - Toplam getiri
    
    Args:
        daily_returns: Günlük getiri serisi
        risk_free_rate: Risksiz faiz oranı (yıllık)
    
    Returns:
        Metrikler sözlüğü
    """
    # toplam getiri (urun seklinde birikimli)
    total_return = (1 + daily_returns).prod() - 1
    
    # yillik getiri
    n_days = len(daily_returns)
    annual_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1
    
    # yillik volatilite
    annual_vol = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    # sharpe orani
    if annual_vol > 0:
        sharpe = (annual_return - risk_free_rate) / annual_vol
    else:
        sharpe = 0
    
    # equity curve olustur ve max drawdown hesapla
    equity = (1 + daily_returns).cumprod()
    max_dd = calculate_max_drawdown(equity)
    
    return {
        "toplam_getiri": total_return,
        "yillik_getiri": annual_return,
        "yillik_volatilite": annual_vol,
        "sharpe_orani": sharpe,
        "max_drawdown": max_dd,
        "islem_gunu_sayisi": n_days
    }


def walk_forward_backtest(
    prices: pd.DataFrame,
    config: BacktestConfig
) -> BacktestResult:
    """
    Walk-forward backtest uygular.
    
    Bu yöntem gerçek dünya koşullarını simüle eder:
    1. Geçmiş verilerle model eğit
    2. Ağırlıkları optimize et
    3. Hold döneminde portföyü uygula
    4. Bir sonraki döneme geç, tekrarla
    
    LOOK-AHEAD BIAS ÖNLEME:
    - Her iterasyonda sadece o ana kadarki veriler kullanılır
    - Gelecek veriler asla eğitim aşamasında kullanılmaz
    
    Args:
        prices: Fiyat DataFrame'i
        config: Backtest konfigürasyonu
    
    Returns:
        BacktestResult objesi
    """
    # getiri hesapla - SIMPLE RETURNS kullaniyoruz (log degil!)
    # Cunku equity curve (1+r).cumprod() ile hesaplaniyor
    returns = calculate_simple_returns(prices)
    
    n_days = len(returns)
    n_assets = len(returns.columns)
    
    # MINIMUM VERI VALIDASYONU
    min_required = config.train_window + config.hold_period * 2
    if n_days < min_required:
        raise ValueError(
            f"Yetersiz veri: {n_days} gün mevcut, en az {min_required} gün gerekli "
            f"(train_window={config.train_window}, hold_period={config.hold_period})"
        )
    
    # sonuclari tutacak listeler
    all_returns = []
    all_weights = []
    rebalance_dates = []
    
    # baslangic noktasi: train_window kadar veri gerekli
    current_idx = config.train_window
    
    print(f"Backtest başlıyor: {n_days} gün, {n_assets} hisse")
    print(f"Eğitim penceresi: {config.train_window} gün, Hold: {config.hold_period} gün")
    
    while current_idx < n_days:
        # EGITIM PENCERESI (sadece gecmis veri!)
        train_start = current_idx - config.train_window
        train_end = current_idx
        
        train_returns = returns.iloc[train_start:train_end]
        
        # kovaryans ve beklenen getiri SADECE egitim verisiyle
        cov_matrix, _ = estimate_covariance_ledoit_wolf(train_returns)
        cov_matrix_annual = cov_matrix * TRADING_DAYS_PER_YEAR
        
        expected_ret = calculate_expected_returns(train_returns)
        
        # optimizasyon
        if config.strategy == "min_variance":
            weights, _, success = minimize_variance(cov_matrix_annual, config.max_weight)
        else:  # max_sharpe
            weights, _, _, success = maximize_sharpe(
                expected_ret,
                cov_matrix_annual,
                config.risk_free_rate,
                config.max_weight
            )
        
        # HOLD DONEMI (test donemi)
        hold_start = current_idx
        hold_end = min(current_idx + config.hold_period, n_days)
        
        hold_returns = returns.iloc[hold_start:hold_end]
        
        # portfoy getirisini hesapla
        portfolio_returns = calculate_portfolio_returns(weights, hold_returns)
        
        # sonuclari kaydet
        all_returns.append(portfolio_returns)
        rebalance_dates.append(returns.index[current_idx])
        
        # agirliklari kaydet
        weight_record = pd.Series(weights, index=returns.columns, name=returns.index[current_idx])
        all_weights.append(weight_record)
        
        # sonraki doneme gec
        current_idx += config.hold_period
    
    # sonuclari birlestir
    daily_returns = pd.concat(all_returns)
    weights_history = pd.DataFrame(all_weights)
    
    # equity curve olustur (1'den baslar)
    equity_curve = (1 + daily_returns).cumprod()
    
    # metrikleri hesapla
    metrics = calculate_metrics(daily_returns, config.risk_free_rate)
    
    # turnover hesapla
    turnover = calculate_turnover(weights_history)
    
    # TRANSACTION COST HESAPLA (YENİ BÖLÜM)
    cost_bps = 10.0  # varsayılan 10 basis points
    
    # İşlem maliyetlerini hesapla
    tx_costs = calculate_total_transaction_costs(weights_history, cost_bps)
    
    # Net getirileri hesapla
    net_returns = calculate_net_returns(
        daily_returns, weights_history, rebalance_dates, cost_bps
    )
    
    # Net equity curve
    net_equity = (1 + net_returns).cumprod()
    
    # Net metrikleri hesapla
    net_metrics = calculate_metrics(net_returns, config.risk_free_rate)
    
    print(f"Backtest tamamlandı: {len(rebalance_dates)} rebalance, turnover: {turnover:.2%}")
    print(f"İşlem maliyeti: toplam {tx_costs['total_cost']*100:.2f}%, "
          f"rebalance başına {tx_costs['avg_cost_per_rebalance']*100:.3f}%")
    
    return BacktestResult(
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        weights_history=weights_history,
        metrics=metrics,
        rebalance_dates=rebalance_dates,
        turnover=turnover,
        # YENİ ALANLAR
        transaction_costs=tx_costs,
        net_daily_returns=net_returns,
        net_equity_curve=net_equity,
        net_metrics=net_metrics
    )


def run_equal_weight_backtest(
    prices: pd.DataFrame,
    config: BacktestConfig
) -> BacktestResult:
    """
    Eşit ağırlıklı strateji ile backtest.
    
    Bu baseline karşılaştırması için kullanılır.
    Optimize edilmiş stratejinin gerçekten daha iyi olup olmadığını
    test etmek için gerekli.
    
    Args:
        prices: Fiyat DataFrame'i
        config: Backtest konfigürasyonu
    
    Returns:
        BacktestResult objesi
    """
    returns = calculate_simple_returns(prices)
    
    n_days = len(returns)
    n_assets = len(returns.columns)
    
    # esit agirlik
    weights = equal_weight_portfolio(n_assets)
    
    all_returns = []
    all_weights = []
    rebalance_dates = []
    
    current_idx = config.train_window
    
    while current_idx < n_days:
        hold_start = current_idx
        hold_end = min(current_idx + config.hold_period, n_days)
        
        hold_returns = returns.iloc[hold_start:hold_end]
        portfolio_returns = calculate_portfolio_returns(weights, hold_returns)
        
        all_returns.append(portfolio_returns)
        rebalance_dates.append(returns.index[current_idx])
        
        weight_record = pd.Series(weights, index=returns.columns, name=returns.index[current_idx])
        all_weights.append(weight_record)
        
        current_idx += config.hold_period
    
    daily_returns = pd.concat(all_returns)
    weights_history = pd.DataFrame(all_weights)
    equity_curve = (1 + daily_returns).cumprod()
    metrics = calculate_metrics(daily_returns, config.risk_free_rate)
    
    return BacktestResult(
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        weights_history=weights_history,
        metrics=metrics,
        rebalance_dates=rebalance_dates
    )


def run_benchmark_backtest(
    benchmark_prices: pd.DataFrame,
    config: BacktestConfig
) -> BacktestResult:
    """
    Benchmark (SPY) için backtest.
    
    Bu piyasa karşılaştırması için kullanılır.
    Portföyümüz piyasayı yeniyor mu sorusuna cevap verir.
    
    Args:
        benchmark_prices: Benchmark fiyat DataFrame'i
        config: Backtest konfigürasyonu
    
    Returns:
        BacktestResult objesi
    """
    returns = calculate_simple_returns(benchmark_prices)
    
    # sadece ilgili dönem
    returns = returns.iloc[config.train_window:]
    
    # benchmark tek hisse, agirlik 1
    weights = np.array([1.0])
    
    daily_returns = returns.iloc[:, 0]  # tek kolon
    equity_curve = (1 + daily_returns).cumprod()
    metrics = calculate_metrics(daily_returns, config.risk_free_rate)
    
    # weights_history benchmark icin anlamsiz ama format icin ekle
    weights_history = pd.DataFrame(
        [[1.0]] * len(returns),
        index=returns.index,
        columns=benchmark_prices.columns
    )
    
    return BacktestResult(
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        weights_history=weights_history,
        metrics=metrics,
        rebalance_dates=[]
    )


def run_risk_parity_backtest(
    prices: pd.DataFrame,
    config: BacktestConfig
) -> BacktestResult:
    """
    Risk Parity stratejisi ile backtest yapar.
    
    Args:
        prices: Fiyat DataFrame'i
        config: Backtest konfigürasyonu
    
    Returns:
        BacktestResult objesi
    """
    returns = calculate_simple_returns(prices)
    n_days = len(returns)
    
    all_returns = []
    all_weights = []
    rebalance_dates = []
    
    current_idx = config.train_window
    
    while current_idx < n_days:
        train_start = current_idx - config.train_window
        train_end = current_idx
        train_returns = returns.iloc[train_start:train_end]
        
        # Kovaryans hesapla
        cov_matrix, _ = estimate_covariance_ledoit_wolf(train_returns)
        cov_matrix_annual = cov_matrix * TRADING_DAYS_PER_YEAR
        
        # Risk Parity ağırlıkları
        weights, _, _ = risk_parity_weights(cov_matrix_annual, config.max_weight)
        
        # Hold dönemi
        hold_start = current_idx
        hold_end = min(current_idx + config.hold_period, n_days)
        hold_returns = returns.iloc[hold_start:hold_end]
        
        portfolio_ret = calculate_portfolio_returns(weights, hold_returns)
        
        all_returns.append(portfolio_ret)
        rebalance_dates.append(returns.index[current_idx])
        
        weight_record = pd.Series(weights, index=returns.columns, name=returns.index[current_idx])
        all_weights.append(weight_record)
        
        current_idx += config.hold_period
    
    daily_returns = pd.concat(all_returns)
    weights_history = pd.DataFrame(all_weights)
    equity_curve = (1 + daily_returns).cumprod()
    metrics = calculate_metrics(daily_returns, config.risk_free_rate)
    turnover = calculate_turnover(weights_history)
    
    # Transaction cost hesapla
    cost_bps = 10.0
    tx_costs = calculate_total_transaction_costs(weights_history, cost_bps)
    net_returns = calculate_net_returns(daily_returns, weights_history, rebalance_dates, cost_bps)
    net_equity = (1 + net_returns).cumprod()
    net_metrics = calculate_metrics(net_returns, config.risk_free_rate)
    
    return BacktestResult(
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        weights_history=weights_history,
        metrics=metrics,
        rebalance_dates=rebalance_dates,
        turnover=turnover,
        transaction_costs=tx_costs,
        net_daily_returns=net_returns,
        net_equity_curve=net_equity,
        net_metrics=net_metrics
    )


def run_backtest_comparison(
    stock_prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    config: BacktestConfig
) -> Dict[str, BacktestResult]:
    """
    Üç stratejiyi karşılaştırmalı olarak backtest eder.
    
    Karşılaştırma:
    1. Optimized: Belirlenen strateji (min_var veya max_sharpe)
    2. Equal-weight: Eşit ağırlıklı baseline
    3. Risk Parity: Risk eşitliği stratejisi
    4. Benchmark (SPY): Piyasa karşılaştırması
    
    Args:
        stock_prices: Hisse fiyatları
        benchmark_prices: Benchmark (SPY) fiyatları
        config: Backtest konfigürasyonu
    
    Returns:
        Sözlük: {strateji_adi: BacktestResult}
    """
    results = {}
    
    print("\n=== Optimized Portfolio ===")
    results["optimized"] = walk_forward_backtest(stock_prices, config)
    
    print("\n=== Equal-Weight Portfolio ===")
    results["equal_weight"] = run_equal_weight_backtest(stock_prices, config)
    
    print("\n=== Risk Parity Portfolio ===")
    results["risk_parity"] = run_risk_parity_backtest(stock_prices, config)
    
    if benchmark_prices is not None:
        print("\n=== Benchmark (SPY) ===")
        results["benchmark"] = run_benchmark_backtest(benchmark_prices, config)
    
    return results


# ============================================
# TRANSACTION COST HESAPLAMA (YENİ FONKSİYONLAR)
# ============================================

def calculate_transaction_cost(
    old_weights: np.ndarray,
    new_weights: np.ndarray,
    cost_bps: float = 10.0  # basis points (10 bps = 0.10%)
) -> float:
    """
    Tek bir rebalance için işlem maliyetini hesaplar.
    
    Formül: cost = sum(|w_new - w_old|) * cost_bps / 10000
    
    Args:
        old_weights: Önceki ağırlıklar
        new_weights: Yeni ağırlıklar
        cost_bps: İşlem maliyeti (basis points, varsayılan 10 bps)
    
    Returns:
        İşlem maliyeti (oran olarak, örn: 0.001 = %0.1)
    """
    if old_weights is None:
        old_weights = np.zeros_like(new_weights)
    
    turnover = np.sum(np.abs(new_weights - old_weights))
    cost = turnover * cost_bps / 10000
    
    return cost


def calculate_total_transaction_costs(
    weights_history: pd.DataFrame,
    cost_bps: float = 10.0
) -> dict:
    """
    Tüm backtest dönemi için toplam işlem maliyetlerini hesaplar.
    
    Args:
        weights_history: Ağırlık geçmişi DataFrame'i
        cost_bps: İşlem maliyeti (basis points)
    
    Returns:
        dict: {
            'total_cost': Toplam maliyet (oran),
            'avg_cost_per_rebalance': Ortalama rebalance maliyeti,
            'n_rebalances': Rebalance sayısı,
            'cost_series': Her rebalance'ın maliyeti
        }
    """
    if len(weights_history) < 2:
        return {
            'total_cost': 0.0,
            'avg_cost_per_rebalance': 0.0,
            'n_rebalances': 0,
            'cost_series': []
        }
    
    costs = []
    
    for i in range(1, len(weights_history)):
        old_w = weights_history.iloc[i-1].values
        new_w = weights_history.iloc[i].values
        cost = calculate_transaction_cost(old_w, new_w, cost_bps)
        costs.append(cost)
    
    # İlk yatırım maliyeti (0'dan başlangıç)
    initial_cost = calculate_transaction_cost(
        np.zeros(len(weights_history.columns)),
        weights_history.iloc[0].values,
        cost_bps
    )
    costs.insert(0, initial_cost)
    
    return {
        'total_cost': sum(costs),
        'avg_cost_per_rebalance': np.mean(costs),
        'n_rebalances': len(costs),
        'cost_series': costs
    }


def calculate_net_returns(
    daily_returns: pd.Series,
    weights_history: pd.DataFrame,
    rebalance_dates: list,
    cost_bps: float = 10.0
) -> pd.Series:
    """
    İşlem maliyetleri düşülmüş net getirileri hesaplar.
    
    Args:
        daily_returns: Brüt günlük getiriler
        weights_history: Ağırlık geçmişi
        rebalance_dates: Rebalance tarihleri
        cost_bps: İşlem maliyeti (basis points)
    
    Returns:
        Net günlük getiriler
    """
    net_returns = daily_returns.copy()
    
    # Her rebalance tarihinde maliyeti düş
    for i, date in enumerate(rebalance_dates):
        if i == 0:
            old_w = np.zeros(len(weights_history.columns))
        else:
            old_w = weights_history.iloc[i-1].values
        
        new_w = weights_history.iloc[i].values
        cost = calculate_transaction_cost(old_w, new_w, cost_bps)
        
        # Rebalance tarihindeki getiriden maliyeti düş
        if date in net_returns.index:
            net_returns.loc[date] -= cost
    
    return net_returns


def compare_metrics(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    """
    Strateji metriklerini karşılaştırmalı tablo olarak döndürür.
    
    Args:
        results: Backtest sonuçları sözlüğü
    
    Returns:
        Karşılaştırma DataFrame'i
    """
    metrics_list = []
    
    for name, result in results.items():
        m = result.metrics.copy()
        m["strateji"] = name
        metrics_list.append(m)
    
    df = pd.DataFrame(metrics_list)
    df = df.set_index("strateji")
    
    # yuzde olarak formatla (gosterim icin)
    return df


# test icin
if __name__ == "__main__":
    print("Backtest modülü test ediliyor...")
    
    # ornek veri olustur
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    
    prices = pd.DataFrame({
        "AAPL": 100 * np.exp(np.random.randn(500).cumsum() * 0.02),
        "MSFT": 200 * np.exp(np.random.randn(500).cumsum() * 0.02),
        "GOOGL": 150 * np.exp(np.random.randn(500).cumsum() * 0.02),
    }, index=dates)
    
    benchmark = pd.DataFrame({
        "SPY": 300 * np.exp(np.random.randn(500).cumsum() * 0.01),
    }, index=dates)
    
    config = BacktestConfig(
        train_window=252,
        hold_period=21,
        strategy="max_sharpe",
        max_weight=0.40,
        risk_free_rate=0.04
    )
    
    results = run_backtest_comparison(prices, benchmark, config)
    
    print("\n=== Karşılaştırma ===")
    comparison = compare_metrics(results)
    print(comparison)
    
    print("\n✓ Test başarılı!")
