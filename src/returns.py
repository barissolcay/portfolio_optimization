"""
Getiri Hesaplama ve Kovaryans Modülü
====================================
Bu modül fiyat verilerinden getiri hesaplar ve
Ledoit-Wolf shrinkage kovaryans tahmini yapar.

Ekonometrist Kararı:
- Log getiri kullanılıyor (basit getiri yerine)
- Kovaryans için Ledoit-Wolf shrinkage tercih edildi
- Yıllıklaştırma 252 işlem günü üzerinden yapılıyor
"""

import numpy as np
import pandas as pd
import logging
from sklearn.covariance import LedoitWolf
from typing import Tuple

# logging ayarla
logger = logging.getLogger(__name__)

# yilliklastirma icin islem gunu sayisi (ABD borsasi)
TRADING_DAYS_PER_YEAR = 252


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Logaritmik (log) getiri hesaplar.
    
    Formül: r_t = ln(P_t / P_{t-1})
    
    Neden log getiri?
    - Ekonometri literatüründe yaygın tercih
    - Zaman açısından toplanabilir (additivity) özelliği var
    - Normal dağılıma daha yakın davranır
    
    Args:
        prices: Fiyat DataFrame'i (index=tarih, kolonlar=hisseler)
    
    Returns:
        Günlük log getiriler
    """
    # np.log ile dogal logaritma aliyoruz
    # shift(1) bir onceki gunu verir
    log_returns = np.log(prices / prices.shift(1))
    
    # ilk satir NaN olacak (onceki gun yok), onu kaldir
    log_returns = log_returns.dropna()
    
    return log_returns


def calculate_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Basit (aritmetik) getiri hesaplar.
    
    Formül: r_t = (P_t - P_{t-1}) / P_{t-1}
    
    NOT: Bu fonksiyon karşılaştırma için var.
    Ana hesaplamalarda log getiri kullanıyoruz.
    
    Args:
        prices: Fiyat DataFrame'i
    
    Returns:
        Günlük basit getiriler
    """
    simple_returns = prices.pct_change()
    simple_returns = simple_returns.dropna()
    
    return simple_returns


def annualize_return(daily_returns: pd.Series) -> float:
    """
    Günlük getiriyi yıllık getiriye çevirir.
    
    Formül: r_yillik = r_gunluk * 252
    
    NOT: Log getiri için basit çarpım yeterli.
    Basit getiri için (1+r)^252 - 1 kullanılır ama
    biz log getiri kullandığımız için bu formül geçerli.
    
    Args:
        daily_returns: Günlük getiri serisi
    
    Returns:
        Yıllık getiri (örn: 0.15 = %15)
    """
    mean_daily = daily_returns.mean()
    annual_return = mean_daily * TRADING_DAYS_PER_YEAR
    
    return annual_return


def annualize_volatility(daily_returns: pd.Series) -> float:
    """
    Günlük volatiliteyi yıllık volatiliteye çevirir.
    
    Formül: sigma_yillik = sigma_gunluk * sqrt(252)
    
    Bu formül varyansın zamana oranla toplandığı varsayımına dayanır.
    
    Args:
        daily_returns: Günlük getiri serisi
    
    Returns:
        Yıllık volatilite (standart sapma)
    """
    daily_std = daily_returns.std()
    annual_vol = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return annual_vol


def estimate_covariance_sample(returns: pd.DataFrame) -> np.ndarray:
    """
    Örnek (sample) kovaryans matrisi hesaplar.
    
    Bu klasik yöntemdir ama dezavantajı var:
    - Az veri ile güvenilir değil
    - Optimizasyon aşırı ağırlıklara yol açabilir
    
    Biz Ledoit-Wolf kullanıyoruz ama karşılaştırma için bu fonksiyon var.
    
    Args:
        returns: Getiri DataFrame'i
    
    Returns:
        Kovaryans matrisi (NxN numpy array)
    """
    return returns.cov().values


def estimate_covariance_ledoit_wolf(returns: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """
    Ledoit-Wolf shrinkage kovaryans tahmini.
    
    Neden Ledoit-Wolf?
    - Sample kovaryans tahmin hatasına hassas
    - Az veri ile aşırı ağırlıklı portföylere yol açar
    - Ledoit-Wolf, sample kovaryansı yapılandırılmış bir tahminciye "büzer" (shrink)
    - Sonuç daha stabil ve güvenilir ağırlıklar verir
    
    Referans: Ledoit & Wolf (2004) - Journal of Multivariate Analysis
    
    Args:
        returns: Getiri DataFrame'i (satırlar=günler, kolonlar=hisseler)
    
    Returns:
        Tuple: (kovaryans_matrisi, shrinkage_katsayisi)
    """
    # sklearn'in LedoitWolf sinifini kullan
    lw = LedoitWolf()
    lw.fit(returns.values)
    
    # kovaryans matrisi ve shrinkage katsayisi
    cov_matrix = lw.covariance_
    shrinkage = lw.shrinkage_
    
    logger.debug(f"Ledoit-Wolf shrinkage katsayısı: {shrinkage:.4f}")
    
    return cov_matrix, shrinkage


def calculate_expected_returns(returns: pd.DataFrame) -> np.ndarray:
    """
    Beklenen getirileri hesaplar (yıllık).
    
    Basit yöntem: Geçmiş ortalama getiri
    
    NOT: Bu naif bir tahmin. Gerçek dünyada daha sofistike
    yöntemler kullanılır ama MVP için yeterli.
    
    Args:
        returns: Günlük getiri DataFrame'i
    
    Returns:
        Yıllık beklenen getiriler (numpy array)
    """
    # gunluk ortalama al, yilliklastir
    daily_mean = returns.mean().values
    annual_expected = daily_mean * TRADING_DAYS_PER_YEAR
    
    return annual_expected


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Korelasyon matrisi hesaplar.
    
    Korelasyon, iki hissenin birlikte hareket etme eğilimini gösterir.
    -1 ile +1 arasında değer alır.
    
    Args:
        returns: Getiri DataFrame'i
    
    Returns:
        Korelasyon matrisi DataFrame'i
    """
    return returns.corr()


def prepare_returns_for_optimization(
    prices: pd.DataFrame,
    use_log_returns: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Optimizasyon için gerekli tüm hesaplamaları yapar.
    
    Args:
        prices: Fiyat DataFrame'i
        use_log_returns: Log getiri mi kullanılsın? (varsayılan: Evet)
    
    Returns:
        Tuple: (getiriler, beklenen_getiriler, kovaryans_matrisi)
    """
    # getiri hesapla
    if use_log_returns:
        returns = calculate_log_returns(prices)
    else:
        returns = calculate_simple_returns(prices)
    
    # beklenen getiriler (yillik)
    expected_returns = calculate_expected_returns(returns)
    
    # kovaryans (Ledoit-Wolf)
    cov_matrix, _ = estimate_covariance_ledoit_wolf(returns)
    
    # kovaryansı yıllıklaştır (günlük kovaryans * 252)
    cov_matrix_annual = cov_matrix * TRADING_DAYS_PER_YEAR
    
    return returns, expected_returns, cov_matrix_annual


# =============================================================================
# DİNAMİK ANALİZ FONKSİYONLARI (EKONOMETRİST ÖNERİSİ)
# =============================================================================
# Bu fonksiyonlar, ekonometristin önerisiyle eklendi.
# Amacı: Korelasyon ve volatilitenin zaman içinde nasıl değiştiğini analiz etmek.
# Neden önemli: Sabit kovaryans varsayımı gerçek piyasalarda geçerli değildir.
# =============================================================================


def calculate_rolling_volatility(
    returns: pd.DataFrame,
    window: int = 21
) -> pd.DataFrame:
    """
    Rolling (hareketli) volatilite hesaplar.
    
    Bu analiz, volatilitenin zaman içinde nasıl değiştiğini gösterir.
    Piyasa stresi dönemlerinde volatilite yükselir - buna "volatilite
    kümelenmesi" (volatility clustering) denir.
    
    NEDEN ÖNEMLİ?
    - Sabit volatilite varsayımı gerçekçi değil
    - Kriz dönemlerinde risk artar
    - Portföy ağırlıkları buna göre ayarlanmalı
    
    Ekonometrist notu: "Eğer rolling volatilite çok dalgalıysa,
    sabit kovaryans temelli optimizasyon yetersiz kalabilir.
    Bu durumda GARCH tabanlı modeller düşünülmelidir."
    
    Args:
        returns: Getiri DataFrame'i
        window: Hesaplama penceresi (varsayılan 21 gün = 1 ay)
    
    Returns:
        Rolling volatilite DataFrame'i (yıllıklaştırılmış)
    """
    # gunluk standart sapma hesapla
    rolling_std = returns.rolling(window=window).std()
    
    # yilliklastir (gunluk std * sqrt(252))
    rolling_vol = rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return rolling_vol


def calculate_rolling_correlation(
    returns: pd.DataFrame,
    window: int = 60
) -> pd.DataFrame:
    """
    Rolling (hareketli) korelasyon hesaplar.
    
    Bu analiz, hisse senetleri arasındaki korelasyonun zaman içinde
    nasıl değiştiğini gösterir.
    
    NEDEN ÖNEMLİ?
    - Kriz dönemlerinde korelasyonlar yükselir ("correlation breakdown")
    - Normal dönemde düşük korelasyonlu hisseler, krizde birlikte düşer
    - Çeşitlendirme faydası tam olarak anlaşılamaz
    
    Ekonometrist notu: "Eğer kriz dönemlerinde korelasyonlar sıçrıyorsa,
    normal dönem verileriyle yapılan optimizasyon yanıltıcı olabilir.
    Stres testi veya regime-switching modeller düşünülmelidir."
    
    Args:
        returns: Getiri DataFrame'i
        window: Hesaplama penceresi (varsayılan 60 gün = ~3 ay)
    
    Returns:
        Her zaman noktası için ortalama korelasyon serisi
    """
    n_assets = len(returns.columns)
    
    if n_assets < 2:
        # tek hisse varsa korelasyon hesaplanamaz
        return pd.Series(index=returns.index, data=np.nan, name="avg_correlation")
    
    # her zaman noktasi icin ortalama korelasyon hesapla
    avg_correlations = []
    dates = []
    
    for i in range(window, len(returns)):
        # son 'window' gunluk veriyi al
        window_data = returns.iloc[i-window:i]
        
        # korelasyon matrisi hesapla
        corr_matrix = window_data.corr()
        
        # ust ucgen ortalamasi al (diyagonal haric)
        # bu, tum hisse ciftlerinin ortalama korelasyonu
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        avg_corr = corr_matrix.where(mask).stack().mean()
        
        avg_correlations.append(avg_corr)
        dates.append(returns.index[i])
    
    result = pd.Series(avg_correlations, index=dates, name="avg_correlation")
    
    return result


def detect_correlation_regimes(
    rolling_corr: pd.Series,
    high_threshold: float = 0.6,
    low_threshold: float = 0.3
) -> pd.Series:
    """
    Korelasyon rejimlerini tespit eder.
    
    Bu fonksiyon, piyasanın "normal" mi yoksa "stres" döneminde mi
    olduğunu korelasyon seviyelerine göre belirler.
    
    REJİMLER:
    - Düşük korelasyon (< 0.3): Normal dönem, çeşitlendirme etkili
    - Orta korelasyon (0.3-0.6): Geçiş dönemi
    - Yüksek korelasyon (> 0.6): Stres dönemi, dikkatli ol!
    
    Ekonometrist notu: "Yüksek korelasyon dönemlerinde yapılan
    optimizasyon sonuçları daha güvenilirdir çünkü kötü senaryoyu
    yansıtır. Düşük korelasyon dönemlerinde ise çeşitlendirme
    faydası abartılı görünebilir."
    
    Args:
        rolling_corr: Rolling ortalama korelasyon serisi
        high_threshold: Yüksek korelasyon eşiği
        low_threshold: Düşük korelasyon eşiği
    
    Returns:
        Rejim serisi: "düşük", "orta", "yüksek"
    """
    def classify(x):
        if pd.isna(x):
            return np.nan
        elif x >= high_threshold:
            return "yüksek"
        elif x <= low_threshold:
            return "düşük"
        else:
            return "orta"
    
    regimes = rolling_corr.apply(classify)
    regimes.name = "correlation_regime"
    
    return regimes


def generate_correlation_report(returns: pd.DataFrame, window: int = 60) -> dict:
    """
    Kapsamlı korelasyon analiz raporu oluşturur.
    
    Bu rapor, ekonometristin korelasyon dinamikleri hakkındaki
    değerlendirmesini destekler.
    
    Args:
        returns: Getiri DataFrame'i
        window: Rolling window boyutu
    
    Returns:
        Analiz sonuçları sözlüğü
    """
    # rolling korelasyon hesapla
    rolling_corr = calculate_rolling_correlation(returns, window)
    
    # rejim tespiti
    regimes = detect_correlation_regimes(rolling_corr)
    
    # istatistikler
    report = {
        "ortalama_korelasyon": rolling_corr.mean(),
        "min_korelasyon": rolling_corr.min(),
        "max_korelasyon": rolling_corr.max(),
        "volatilite_korelasyon": rolling_corr.std(),  # korelasyonun volatilitesi
        "yuksek_korelasyon_orani": (regimes == "yüksek").mean() * 100,
        "dusuk_korelasyon_orani": (regimes == "düşük").mean() * 100,
        "rolling_corr_series": rolling_corr,
        "regimes_series": regimes
    }
    
    # ekonomik yorum
    if report["yuksek_korelasyon_orani"] > 30:
        report["yorum"] = ("⚠️ Yüksek korelasyon dönemi oranı %30'un üzerinde. "
                          "Çeşitlendirme faydası sınırlı olabilir. "
                          "Ekonometrist notu: Stres dönemi verileri ağırlıklı.")
    elif report["volatilite_korelasyon"] > 0.15:
        report["yorum"] = ("⚠️ Korelasyon volatilitesi yüksek. "
                          "Piyasa rejim değişimleri yaşamış. "
                          "Ekonometrist notu: Sabit kovaryans varsayımı zayıf.")
    else:
        report["yorum"] = ("✓ Korelasyon yapısı nispeten stabil. "
                          "Ekonometrist notu: Optimizasyon sonuçları güvenilir.")
    
    return report


# test icin
if __name__ == "__main__":
    print("Returns modülü test ediliyor...")
    
    # ornek fiyat verisi olustur
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    prices = pd.DataFrame({
        "AAPL": 100 * (1 + np.random.randn(100).cumsum() * 0.02),
        "MSFT": 200 * (1 + np.random.randn(100).cumsum() * 0.02),
    }, index=dates)
    
    # log getiri
    log_ret = calculate_log_returns(prices)
    print("\nLog Getiriler (ilk 5 gün):")
    print(log_ret.head())
    
    # yillik getiri ve volatilite
    for col in log_ret.columns:
        ann_ret = annualize_return(log_ret[col])
        ann_vol = annualize_volatility(log_ret[col])
        print(f"\n{col}: Yıllık getiri={ann_ret:.2%}, Volatilite={ann_vol:.2%}")
    
    # kovaryans
    cov, shrink = estimate_covariance_ledoit_wolf(log_ret)
    print(f"\nKovaryans Matrisi:\n{cov}")
    
    print("\n✓ Test başarılı!")
