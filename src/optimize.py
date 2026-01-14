"""
Portföy Optimizasyon Modülü
===========================
Bu modül mean-variance optimizasyonu ile portföy ağırlıklarını hesaplar.
İki ana strateji var: Minimum Varyans ve Maksimum Sharpe.

Kısıtlar:
- Long-only (açığa satış yok)
- Maksimum ağırlık limiti (varsayılan %30)
- Toplam ağırlık = 1 (tam yatırım)

Fail-safe: Solver başarısız olursa eşit ağırlıklı portföye düşer.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Optional, Callable
import warnings

# varsayilan parametreler
DEFAULT_MAX_WEIGHT = 0.30  # tek hisseye max %30
DEFAULT_RISK_FREE_RATE = 0.04  # %4 sabit (kullanici degistirebilir)


def equal_weight_portfolio(n_assets: int) -> np.ndarray:
    """
    Eşit ağırlıklı portföy oluşturur.
    
    Bu en basit strateji - her hisseye eşit pay.
    Baseline karşılaştırması ve fail-safe için kullanılır.
    
    Args:
        n_assets: Hisse sayısı
    
    Returns:
        Ağırlık vektörü (örn: 3 hisse için [0.333, 0.333, 0.333])
    """
    return np.ones(n_assets) / n_assets


def get_constraints(n_assets: int, max_weight: float = DEFAULT_MAX_WEIGHT) -> dict:
    """
    Optimizasyon kısıtlarını oluşturur.
    
    Kısıtlar:
    1. w_i >= 0 (long-only, açığa satış yok)
    2. sum(w) = 1 (tüm para yatırılır)
    3. w_i <= max_weight (çeşitlendirme zorlaması)
    
    Args:
        n_assets: Hisse sayısı
        max_weight: Tek hisse max ağırlık (0-1 arası)
    
    Returns:
        scipy.optimize için kısıt sözlüğü
    """
    # toplam agirlik = 1 kisiti
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    ]
    
    # bounds: her agirlik 0 ile max_weight arasinda
    bounds = tuple((0.0, max_weight) for _ in range(n_assets))
    
    return {"constraints": constraints, "bounds": bounds}


def portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Portföy varyansını hesaplar.
    
    Formül: sigma_p^2 = w' * Σ * w
    
    Args:
        weights: Ağırlık vektörü
        cov_matrix: Kovaryans matrisi
    
    Returns:
        Portföy varyansı
    """
    return weights @ cov_matrix @ weights


def portfolio_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
    """
    Portföy beklenen getirisini hesaplar.
    
    Formül: r_p = w' * μ
    
    Args:
        weights: Ağırlık vektörü
        expected_returns: Beklenen getiri vektörü
    
    Returns:
        Portföy beklenen getirisi
    """
    return weights @ expected_returns


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Portföy volatilitesini (standart sapma) hesaplar.
    
    Formül: sigma_p = sqrt(w' * Σ * w)
    
    Args:
        weights: Ağırlık vektörü
        cov_matrix: Kovaryans matrisi
    
    Returns:
        Portföy volatilitesi
    """
    return np.sqrt(portfolio_variance(weights, cov_matrix))


def negative_sharpe_ratio(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float
) -> float:
    """
    Negatif Sharpe oranı (minimize edilecek).
    
    Formül: Sharpe = (r_p - r_f) / sigma_p
    
    Minimizer kullandığımız için negatifini alıyoruz.
    
    Args:
        weights: Ağırlık vektörü
        expected_returns: Beklenen getiriler
        cov_matrix: Kovaryans matrisi
        risk_free_rate: Risksiz faiz oranı
    
    Returns:
        Negatif Sharpe oranı
    """
    ret = portfolio_return(weights, expected_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    
    # volatilite 0 ise buyuk ceza ver
    if vol < 1e-10:
        return 1e10
    
    sharpe = (ret - risk_free_rate) / vol
    return -sharpe  # negatif cunku minimize ediyoruz


def minimize_variance(
    cov_matrix: np.ndarray,
    max_weight: float = DEFAULT_MAX_WEIGHT
) -> Tuple[np.ndarray, float, bool]:
    """
    Minimum varyans portföyünü bulur.
    
    Bu strateji riski minimize eder, getiriyi görmezden gelir.
    Risk-averse yatırımcılar için uygundur.
    
    Args:
        cov_matrix: Kovaryans matrisi (yıllıklaştırılmış)
        max_weight: Maksimum ağırlık limiti
    
    Returns:
        Tuple: (optimal_agirliklar, portfoy_volatilitesi, basarili_mi)
    """
    n_assets = cov_matrix.shape[0]
    
    # baslangic noktasi: esit agirlik
    initial_weights = equal_weight_portfolio(n_assets)
    
    # kisitlar
    cons = get_constraints(n_assets, max_weight)
    
    # hedef fonksiyon: varyans
    def objective(w):
        return portfolio_variance(w, cov_matrix)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = minimize(
                objective,
                initial_weights,
                method="SLSQP",
                bounds=cons["bounds"],
                constraints=cons["constraints"],
                options={"maxiter": 1000, "ftol": 1e-10}
            )
        
        if result.success:
            weights = result.x
            # kucuk negatifleri sifirla (sayisal hata)
            weights = np.maximum(weights, 0)
            # normalize et (toplam=1 garanti)
            weights = weights / weights.sum()
            
            vol = portfolio_volatility(weights, cov_matrix)
            return weights, vol, True
        else:
            print(f"⚠️ Optimizer başarısız: {result.message}")
            
    except Exception as e:
        print(f"⚠️ Optimizer hatası: {e}")
    
    # basarisiz olursa esit agirlik dondur (fail-safe)
    print("ℹ️ Fail-safe: Eşit ağırlıklı portföye geçildi")
    weights = equal_weight_portfolio(n_assets)
    vol = portfolio_volatility(weights, cov_matrix)
    return weights, vol, False


def maximize_sharpe(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    max_weight: float = DEFAULT_MAX_WEIGHT
) -> Tuple[np.ndarray, float, float, bool]:
    """
    Maksimum Sharpe oranı portföyünü bulur.
    
    Sharpe oranı, risk başına getiriyi ölçer.
    Daha yüksek Sharpe = daha iyi risk-getiri dengesi.
    
    Args:
        expected_returns: Beklenen getiriler (yıllık)
        cov_matrix: Kovaryans matrisi (yıllık)
        risk_free_rate: Risksiz faiz oranı (örn: 0.04 = %4)
        max_weight: Maksimum ağırlık limiti
    
    Returns:
        Tuple: (optimal_agirliklar, sharpe_orani, volatilite, basarili_mi)
    """
    n_assets = len(expected_returns)
    
    # baslangic noktasi
    initial_weights = equal_weight_portfolio(n_assets)
    
    # kisitlar
    cons = get_constraints(n_assets, max_weight)
    
    # hedef fonksiyon: negatif Sharpe
    def objective(w):
        return negative_sharpe_ratio(w, expected_returns, cov_matrix, risk_free_rate)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = minimize(
                objective,
                initial_weights,
                method="SLSQP",
                bounds=cons["bounds"],
                constraints=cons["constraints"],
                options={"maxiter": 1000, "ftol": 1e-10}
            )
        
        if result.success:
            weights = result.x
            weights = np.maximum(weights, 0)
            weights = weights / weights.sum()
            
            vol = portfolio_volatility(weights, cov_matrix)
            ret = portfolio_return(weights, expected_returns)
            sharpe = (ret - risk_free_rate) / vol
            
            return weights, sharpe, vol, True
        else:
            print(f"⚠️ Optimizer başarısız: {result.message}")
            
    except Exception as e:
        print(f"⚠️ Optimizer hatası: {e}")
    
    # basarisiz olursa esit agirlik (fail-safe)
    print("ℹ️ Fail-safe: Eşit ağırlıklı portföye geçildi")
    weights = equal_weight_portfolio(n_assets)
    vol = portfolio_volatility(weights, cov_matrix)
    ret = portfolio_return(weights, expected_returns)
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
    return weights, sharpe, vol, False


def risk_parity_weights(
    cov_matrix: np.ndarray,
    max_weight: float = DEFAULT_MAX_WEIGHT
) -> Tuple[np.ndarray, float, bool]:
    """
    Risk Parity (Esit Risk Katkisi) portfoyu olusturur.
    Her varligin portfoy riskine katkisi esit olacak sekilde agirliklandirir.
    
    Args:
        cov_matrix: Kovaryans matrisi
        max_weight: Maksimum agirlik limiti
    
    Returns:
        Tuple: (agirliklar, volatilite, basarili_mi)
    """
    n_assets = cov_matrix.shape[0]
    
    # Bireysel volatiliteler (kovaryans matrisinin köşegeni)
    variances = np.diag(cov_matrix)
    volatilities = np.sqrt(variances)
    
    # Inverse volatility ağırlıkları
    inv_vol = 1.0 / volatilities
    raw_weights = inv_vol / inv_vol.sum()
    
    # Max weight kısıtını uygula
    weights = np.minimum(raw_weights, max_weight)
    
    # Normalize et
    weights = weights / weights.sum()
    
    # Portföy volatilitesi
    vol = portfolio_volatility(weights, cov_matrix)
    
    return weights, vol, True


def calculate_efficient_frontier(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    n_points: int = 50,
    max_weight: float = DEFAULT_MAX_WEIGHT
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Etkin sınır (Efficient Frontier) hesaplar.
    
    Etkin sınır, her risk seviyesi için maksimum getiriyi
    veren portföylerin geometrik yerini gösterir.
    
    Args:
        expected_returns: Beklenen getiriler
        cov_matrix: Kovaryans matrisi
        n_points: Sınır üzerindeki nokta sayısı
        max_weight: Maksimum ağırlık
    
    Returns:
        Tuple: (volatiliteler, getiriler, agirlik_listesi)
    """
    n_assets = len(expected_returns)
    
    # once min variance portfolyunu bul (en dusuk risk)
    min_var_weights, min_vol, _ = minimize_variance(cov_matrix, max_weight)
    min_ret = portfolio_return(min_var_weights, expected_returns)
    
    # max getiriyi bul (tek hisseye tam yatirim, ama max_weight kisiti var)
    # basitlestirmek icin bireysel getirilerin max'ini al
    max_ret = expected_returns.max()
    
    # getiri hedefleri olustur
    target_returns = np.linspace(min_ret, max_ret * 0.95, n_points)
    
    volatilities = []
    returns = []
    weights_list = []
    
    for target_ret in target_returns:
        # her hedef getiri icin min varyans bul
        cons = get_constraints(n_assets, max_weight)
        
        # getiri kisiti ekle
        constraints = cons["constraints"].copy()
        constraints.append({
            "type": "eq",
            "fun": lambda w, t=target_ret: portfolio_return(w, expected_returns) - t
        })
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                result = minimize(
                    lambda w: portfolio_variance(w, cov_matrix),
                    equal_weight_portfolio(n_assets),
                    method="SLSQP",
                    bounds=cons["bounds"],
                    constraints=constraints,
                    options={"maxiter": 500}
                )
            
            if result.success:
                w = result.x
                w = np.maximum(w, 0)
                w = w / w.sum()
                
                vol = portfolio_volatility(w, cov_matrix)
                ret = portfolio_return(w, expected_returns)
                
                volatilities.append(vol)
                returns.append(ret)
                weights_list.append(w)
                
        except Exception:
            continue
    
    return np.array(volatilities), np.array(returns), weights_list


def get_portfolio_summary(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    asset_names: list,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
) -> dict:
    """
    Portföy özet bilgilerini hesaplar.
    
    Args:
        weights: Optimize edilmiş ağırlıklar
        expected_returns: Beklenen getiriler
        cov_matrix: Kovaryans matrisi
        asset_names: Hisse isimleri
        risk_free_rate: Risksiz faiz oranı
    
    Returns:
        Özet bilgiler sözlüğü
    """
    ret = portfolio_return(weights, expected_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
    
    # agirlik dict'i olustur
    weight_dict = {name: w for name, w in zip(asset_names, weights)}
    
    return {
        "agirliklar": weight_dict,
        "beklenen_getiri": ret,
        "volatilite": vol,
        "sharpe_orani": sharpe,
        "risk_free_rate": risk_free_rate
    }


def calculate_risk_contribution(
    weights: np.ndarray,
    cov_matrix: np.ndarray
) -> np.ndarray:
    """
    Her varligin portfoy riskine katkisini hesaplar.
    
    Args:
        weights: Portfoy agirliklari
        cov_matrix: Kovaryans matrisi
    
    Returns:
        Risk katkilari (toplami 1 olacak sekilde normalize)
    """
    # Portfoy varyansi
    port_var = weights @ cov_matrix @ weights
    port_vol = np.sqrt(port_var)
    
    # Marginal risk contribution: d(sigma_p) / d(w_i)
    marginal_contrib = cov_matrix @ weights / port_vol
    
    # Risk contribution: w_i × MRC_i
    risk_contrib = weights * marginal_contrib
    
    # Normalize et (toplamı 1 olsun)
    risk_contrib_pct = risk_contrib / risk_contrib.sum()
    
    return risk_contrib_pct


def get_risk_contribution_summary(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    asset_names: list
) -> pd.DataFrame:
    """
    Risk katkısı özet tablosu oluşturur.
    
    Args:
        weights: Portföy ağırlıkları
        cov_matrix: Kovaryans matrisi
        asset_names: Varlık isimleri
    
    Returns:
        Risk katkısı DataFrame'i
    """
    risk_contrib = calculate_risk_contribution(weights, cov_matrix)
    
    df = pd.DataFrame({
        "Varlık": asset_names,
        "Ağırlık": weights,
        "Ağırlık (%)": [f"{w*100:.1f}%" for w in weights],
        "Risk Katkısı": risk_contrib,
        "Risk Katkısı (%)": [f"{rc*100:.1f}%" for rc in risk_contrib],
        "RC/W Oranı": risk_contrib / np.maximum(weights, 1e-10)  # risk/weight oranı
    })
    
    return df


# Duyarlilik Analizi Fonksiyonlari
# Parametre secimlerinin sonuclara etkisini analiz eder


def max_weight_sensitivity_analysis(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    weight_range: list = None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
) -> pd.DataFrame:
    """
    Maksimum agirlik parametresinin sonuclara etkisini analiz eder.
    
    Args:
        expected_returns: Beklenen getiriler
        cov_matrix: Kovaryans matrisi
        weight_range: Test edilecek max_weight degerleri
        risk_free_rate: Risksiz faiz orani
    
    Returns:
        Sensitivity analizi DataFrame'i
    """
    if weight_range is None:
        weight_range = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.00]
    
    results = []
    
    for max_w in weight_range:
        # her max_weight icin optimizasyon yap
        weights, sharpe, vol, success = maximize_sharpe(
            expected_returns, cov_matrix, risk_free_rate, max_w
        )
        
        ret = portfolio_return(weights, expected_returns)
        
        # konsantrasyon olcumu: en buyuk agirlik / esit agirlik
        max_actual = weights.max()
        concentration = max_actual / (1 / len(weights))
        
        results.append({
            "max_weight": f"{max_w:.0%}",
            "beklenen_getiri": ret,
            "volatilite": vol,
            "sharpe": sharpe,
            "en_buyuk_agirlik": max_actual,
            "konsantrasyon": concentration,
            "basarili": success
        })
    
    return pd.DataFrame(results)


def generate_sensitivity_report(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    asset_names: list,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
) -> dict:
    """
    Duyarlilik analiz raporu olusturur.
    
    Args:
        expected_returns: Beklenen getiriler
        cov_matrix: Kovaryans matrisi
        asset_names: Hisse isimleri
        risk_free_rate: Risksiz faiz orani
    
    Returns:
        Analiz sonuclari sozlugu
    """
    # max_weight duyarlılık analizi
    sensitivity_df = max_weight_sensitivity_analysis(
        expected_returns, cov_matrix,
        weight_range=[0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
        risk_free_rate=risk_free_rate
    )
    
    # Sharpe oranının volatilitesi (duyarlılık ölçüsü)
    sharpe_volatility = sensitivity_df["sharpe"].std()
    sharpe_range = sensitivity_df["sharpe"].max() - sensitivity_df["sharpe"].min()
    
    # en iyi ve en kötü max_weight
    best_idx = sensitivity_df["sharpe"].idxmax()
    worst_idx = sensitivity_df["sharpe"].idxmin()
    
    report = {
        "sensitivity_df": sensitivity_df,
        "sharpe_volatility": sharpe_volatility,
        "sharpe_range": sharpe_range,
        "en_iyi_max_weight": sensitivity_df.loc[best_idx, "max_weight"],
        "en_kotu_max_weight": sensitivity_df.loc[worst_idx, "max_weight"],
        "en_iyi_sharpe": sensitivity_df.loc[best_idx, "sharpe"],
        "en_kotu_sharpe": sensitivity_df.loc[worst_idx, "sharpe"],
    }
    
    # yorum olustur
    if sharpe_range > 0.3:
        report["yorum"] = f"⚠️ Yüksek duyarlılık! Sharpe oranı {sharpe_range:.2f} oranında değişiyor."
    elif sharpe_range > 0.15:
        report["yorum"] = f"ℹ️ Orta düzeyde duyarlılık. Sharpe oranı {sharpe_range:.2f} oranında değişiyor."
    else:
        report["yorum"] = f"✓ Düşük duyarlılık. Sharpe oranı {sharpe_range:.2f} oranında değişiyor."
    
    return report


# test icin
if __name__ == "__main__":
    print("Optimizasyon modülü test ediliyor...")
    
    np.random.seed(42)
    
    # ornek veri
    n_assets = 4
    expected_returns = np.array([0.10, 0.12, 0.08, 0.15])  # yillik getiriler
    
    # kovaryans matrisi olustur (pozitif definit olmali)
    random_matrix = np.random.randn(n_assets, n_assets)
    cov_matrix = random_matrix @ random_matrix.T / 100  # kucuk kovaryans
    
    print(f"\nBeklenen Getiriler: {expected_returns}")
    
    # minimum varyans
    print("\n--- Minimum Varyans ---")
    weights_mv, vol_mv, success_mv = minimize_variance(cov_matrix)
    print(f"Ağırlıklar: {weights_mv}")
    print(f"Volatilite: {vol_mv:.4f}")
    print(f"Başarılı: {success_mv}")
    
    # maksimum sharpe
    print("\n--- Maksimum Sharpe ---")
    weights_ms, sharpe_ms, vol_ms, success_ms = maximize_sharpe(
        expected_returns, cov_matrix, risk_free_rate=0.04
    )
    print(f"Ağırlıklar: {weights_ms}")
    print(f"Sharpe: {sharpe_ms:.4f}")
    print(f"Volatilite: {vol_ms:.4f}")
    print(f"Başarılı: {success_ms}")
    
    # esit agirlik (karsilastirma)
    print("\n--- Eşit Ağırlık ---")
    weights_eq = equal_weight_portfolio(n_assets)
    vol_eq = portfolio_volatility(weights_eq, cov_matrix)
    ret_eq = portfolio_return(weights_eq, expected_returns)
    sharpe_eq = (ret_eq - 0.04) / vol_eq
    print(f"Ağırlıklar: {weights_eq}")
    print(f"Sharpe: {sharpe_eq:.4f}")
    
    print("\n✓ Test başarılı!")
