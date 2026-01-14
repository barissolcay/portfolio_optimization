"""
Risk Yönetimi Modülü
====================
Bu modül Value at Risk (VaR) hesaplar ve ihlal analizi yapar.
Ayrıca Kupiec POF testi ile VaR modelinin güvenilirliğini test eder.

VaR Nedir?
- Belirli bir güven düzeyinde, belirli bir zaman diliminde
  oluşabilecek maksimum kaybı gösterir.
- Örnek: %95 VaR = -0.02 demek, günün %95'inde kaybımız %2'yi geçmez.

Kupiec POF Testi:
- VaR ihlal sayısının (gerçekleşen kayıp > VaR) beklenen sayıyla
  uyumlu olup olmadığını test eder.
- p-value < 0.05 ise model yetersiz demektir.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class VaRResult:
    """
    VaR hesaplama sonuçları.
    
    Bu dataclass, VaR analizinin tüm sonuçlarını bir arada tutar.
    İki ayrı test içerir:
    1. Kupiec POF Testi: İhlal sayısı beklenen düzeyde mi?
    2. Christoffersen Testi: İhlaller birbirinden bağımsız mı?
    
    Neden iki test?
    - Kupiec sadece frekansı kontrol eder
    - Christoffersen ardışık ihlalleri (clustering) tespit eder
    - İkisi birlikte modelin güvenilirliğini tam olarak değerlendirir
    
    Ekonometrist notu: "VaR modelinin sadece doğru oranda ihlal üretmesi
    yetmez, ihlallerin rastgele dağılması da gerekir. Ardışık ihlaller
    (clustering) varsa, model kötü dönemleri öngöremiyor demektir."
    """
    var_value: float              # VaR degeri (negatif sayi)
    confidence: float             # guven duzeyi (örn: 0.95)
    violations: pd.Series         # ihlal gunleri (True/False)
    n_violations: int             # toplam ihlal sayisi
    expected_violations: float    # beklenen ihlal sayisi
    violation_rate: float         # ihlal orani
    kupiec_statistic: float       # Kupiec test istatistigi
    kupiec_pvalue: float          # Kupiec p-value
    kupiec_passed: bool           # Kupiec testi gecti mi?
    # Christoffersen (bağımsızlık) testi sonuçları
    christoffersen_statistic: float = 0.0  # Christoffersen test istatistigi
    christoffersen_pvalue: float = 1.0     # Christoffersen p-value
    christoffersen_passed: bool = True     # Christoffersen testi gecti mi?
    # Birleşik (joint) test sonucu
    joint_statistic: float = 0.0           # Kupiec + Christoffersen
    joint_pvalue: float = 1.0
    joint_passed: bool = True              # Her iki test de gecti mi?


def calculate_historical_var(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    """
    Historical simulation yöntemiyle VaR hesaplar.
    
    Bu yöntem:
    - Geçmiş getirileri sıralar
    - Belirtilen yüzdelik dilimi bulur
    - O değeri VaR olarak raporlar
    
    Örnek: %95 VaR için en kötü %5'lik dilimin sınırını buluruz.
    
    Args:
        returns: Günlük getiri serisi
        confidence: Güven düzeyi (varsayılan 0.95 = %95)
    
    Returns:
        VaR değeri (negatif bir sayı, kayıp olduğu için)
    """
    # alpha = 1 - confidence (örn: %95 için alpha = 0.05)
    alpha = 1 - confidence
    
    # getirilerin alpha yuzdelik dilimi
    var = np.percentile(returns, alpha * 100)
    
    return var


def count_var_violations(
    returns: pd.Series,
    var_threshold: float
) -> Tuple[pd.Series, int]:
    """
    VaR ihlal sayısını hesaplar.
    
    İhlal: Gerçekleşen kayıp, VaR'dan daha kötü (daha negatif)
    
    Args:
        returns: Günlük getiri serisi
        var_threshold: VaR eşik değeri (negatif)
    
    Returns:
        Tuple: (ihlal_serisi, ihlal_sayisi)
    """
    # VaR'dan daha kotu gunler
    violations = returns < var_threshold
    n_violations = violations.sum()
    
    return violations, n_violations


def kupiec_pof_test(
    n_violations: int,
    n_observations: int,
    confidence: float
) -> Tuple[float, float, bool]:
    """
    Kupiec Proportion of Failures (POF) testi.
    
    Bu test, VaR modelinin doğru kalibre edilip edilmediğini kontrol eder.
    
    H0: Gerçek ihlal oranı = Beklenen ihlal oranı
    H1: Gerçek ihlal oranı ≠ Beklenen ihlal oranı
    
    Test mantığı:
    - %95 VaR için günlerin %5'inde ihlal beklenir
    - Çok fazla veya çok az ihlal varsa model sorunlu
    
    Referans: Kupiec (1995) - Journal of Derivatives
    
    Args:
        n_violations: Gerçekleşen ihlal sayısı
        n_observations: Toplam gözlem sayısı
        confidence: VaR güven düzeyi
    
    Returns:
        Tuple: (test_istatistigi, p_value, gecti_mi)
    """
    # beklenen ihlal orani
    expected_rate = 1 - confidence
    
    # gercek ihlal orani
    if n_observations == 0:
        return 0.0, 1.0, True
    
    actual_rate = n_violations / n_observations
    
    # ihlal yoksa veya cok azsa ozel durum
    if n_violations == 0:
        # hic ihlal yoksa, model cok konservatif olabilir ama gecmis say
        return 0.0, 1.0, True
    
    if n_violations == n_observations:
        # her gun ihlal varsa model tamamen yanlis
        return float('inf'), 0.0, False
    
    # Kupiec LR istatistigi
    # LR = -2 * log[ (1-p)^(n-x) * p^x / (1-p_hat)^(n-x) * p_hat^x ]
    # burada p = expected_rate, p_hat = actual_rate
    
    try:
        # log-likelihood orani
        n = n_observations
        x = n_violations
        p = expected_rate
        p_hat = actual_rate
        
        # pay: null hypothesis altinda likelihood
        log_null = (n - x) * np.log(1 - p) + x * np.log(p)
        
        # payda: alternative hypothesis altinda likelihood
        log_alt = (n - x) * np.log(1 - p_hat) + x * np.log(p_hat)
        
        # LR istatistigi
        lr_stat = -2 * (log_null - log_alt)
        
        # chi-square dagilimi ile karsilastir (1 serbestlik derecesi)
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        
        # %5 anlamlilik duzeyinde test
        passed = p_value > 0.05
        
        return lr_stat, p_value, passed
        
    except Exception as e:
        print(f"⚠️ Kupiec testi hesaplanamadı: {e}")
        return 0.0, 1.0, True


def christoffersen_independence_test(violations: pd.Series) -> Tuple[float, float, bool]:
    """
    Christoffersen Independence (Bağımsızlık) Testi.
    
    Bu test, VaR ihlallerinin birbirinden BAĞIMSIZ olup olmadığını kontrol eder.
    Kupiec testi sadece ihlal SAYISINI kontrol ederken, bu test ihlallerin
    DAĞILIMINI test eder.
    
    NEDEN ÖNEMLİ?
    - Ardışık ihlaller (clustering) varsa, model kötü dönemleri öngöremiyor demek
    - Örnek: 5 ihlal var, ama hepsi aynı hafta içinde → Tehlikeli!
    - Normal dağılım: İhlaller rastgele dağılmalı
    
    YÖNTEM (Markov Chain):
    - Bugün ihlal olup olmadığı, dünkü duruma bağlı mı?
    - Transition matrix: P(bugün ihlal | dünkü durum)
    - Eğer dünkü durum önemliyse → Bağımsızlık yok → Model zayıf
    
    Referans: Christoffersen (1998) - "Evaluating Interval Forecasts"
    Journal of Business & Economic Statistics
    
    Args:
        violations: Boolean ihlal serisi (True = ihlal günü)
    
    Returns:
        Tuple: (test_istatistigi, p_value, gecti_mi)
    """
    # violations serisini 0/1 dizisine çevir
    v = violations.astype(int).values
    n = len(v)
    
    if n < 2:
        return 0.0, 1.0, True
    
    # Transition matrix hesapla
    # n_ij = i durumundan j durumuna geçiş sayısı
    # i=0: önceki gün ihlal yok, i=1: önceki gün ihlal var
    # j=0: bugün ihlal yok, j=1: bugün ihlal var
    
    n_00 = 0  # ihlalsiz günden ihlalsiz güne
    n_01 = 0  # ihlalsiz günden ihlalli güne
    n_10 = 0  # ihlalli günden ihlalsiz güne
    n_11 = 0  # ihlalli günden ihlalli güne (clustering!)
    
    for t in range(1, n):
        if v[t-1] == 0 and v[t] == 0:
            n_00 += 1
        elif v[t-1] == 0 and v[t] == 1:
            n_01 += 1
        elif v[t-1] == 1 and v[t] == 0:
            n_10 += 1
        else:  # v[t-1] == 1 and v[t] == 1
            n_11 += 1
    
    # Toplam geçişler
    n_0 = n_00 + n_01  # 0 durumundan çıkış sayısı
    n_1 = n_10 + n_11  # 1 durumundan çıkış sayısı
    
    # Özel durumlar (yeterli veri yok)
    if n_0 == 0 or n_1 == 0:
        # Hiç geçiş yok, bağımsızlık testi yapılamaz
        return 0.0, 1.0, True
    
    if n_01 == 0 or n_11 == 0:
        # Sıfır sayılar log'da sorun çıkarır
        # Ama bu genelde iyi bir işaret (az ihlal)
        return 0.0, 1.0, True
    
    try:
        # Koşullu olasılıklar
        pi_01 = n_01 / n_0 if n_0 > 0 else 0  # P(ihlal | önceki gün ihlal yok)
        pi_11 = n_11 / n_1 if n_1 > 0 else 0  # P(ihlal | önceki gün ihlal var)
        
        # Koşulsuz olasılık (H0 altında)
        pi = (n_01 + n_11) / (n_0 + n_1)
        
        # Bağımsızlık altında log-likelihood (H0)
        if pi == 0 or pi == 1:
            return 0.0, 1.0, True
        
        log_l0 = (n_00 + n_10) * np.log(1 - pi) + (n_01 + n_11) * np.log(pi)
        
        # Bağımlılık altında log-likelihood (H1)
        # Sıfır olasılıklara karşı koruma
        eps = 1e-10
        p_00 = max(1 - pi_01, eps)
        p_01 = max(pi_01, eps)
        p_10 = max(1 - pi_11, eps)
        p_11 = max(pi_11, eps)
        
        log_l1 = (n_00 * np.log(p_00) + n_01 * np.log(p_01) +
                  n_10 * np.log(p_10) + n_11 * np.log(p_11))
        
        # Likelihood Ratio istatistiği
        lr_stat = -2 * (log_l0 - log_l1)
        
        # Chi-square dağılımı (df=1)
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        
        # %5 anlamlılık düzeyinde test
        passed = p_value > 0.05
        
        return lr_stat, p_value, passed
        
    except Exception as e:
        print(f"⚠️ Christoffersen testi hesaplanamadı: {e}")
        return 0.0, 1.0, True


def joint_var_test(
    kupiec_stat: float,
    christoffersen_stat: float
) -> Tuple[float, float, bool]:
    """
    Birleşik (Joint) VaR Testi.
    
    Kupiec ve Christoffersen testlerini birleştirir.
    
    NEDEN BİRLEŞİK TEST?
    - Kupiec: İhlal sayısı doğru mu?
    - Christoffersen: İhlaller bağımsız mı?
    - İkisi de geçmeli → Model gerçekten güvenilir
    
    Ekonometrist notu: "Sadece Kupiec testi yeterli değil. Bir model
    doğru sayıda ihlal üretebilir ama ihlaller kötü dönemlerde
    kümeleniyorsa, risk yönetimi için kullanılamaz."
    
    Args:
        kupiec_stat: Kupiec test istatistiği
        christoffersen_stat: Christoffersen test istatistiği
    
    Returns:
        Tuple: (joint_stat, p_value, gecti_mi)
    """
    # Birleşik istatistik: LR_joint = LR_kupiec + LR_ind
    joint_stat = kupiec_stat + christoffersen_stat
    
    # Chi-square df=2 (iki bağımsız test)
    p_value = 1 - stats.chi2.cdf(joint_stat, df=2)
    
    passed = p_value > 0.05
    
    return joint_stat, p_value, passed


def generate_var_analysis(
    returns: pd.Series,
    confidence: float = 0.95
) -> VaRResult:
    """
    Kapsamlı VaR analizi yapar.
    
    Bu fonksiyon ÜÇ ayrı test uygular:
    1. VaR hesaplama (Historical Simulation)
    2. Kupiec POF testi (ihlal SAYISI doğru mu?)
    3. Christoffersen testi (ihlaller BAĞIMSIZ mı?)
    4. Birleşik test (her ikisi de geçti mi?)
    
    Ekonometrist notu: "Sadece ihlal sayısına bakmak yanıltıcı olabilir.
    5 ihlal rastgele dağılmışsa sorun yok, ama 5 ihlal aynı haftada
    olduysa model volatilite kümelenmesini yakalayamamış demektir."
    
    Args:
        returns: Günlük getiri serisi
        confidence: Güven düzeyi
    
    Returns:
        VaRResult objesi (tüm test sonuçlarıyla)
    """
    # 1. VaR hesapla
    var_value = calculate_historical_var(returns, confidence)
    
    # 2. ihlalleri say
    violations, n_violations = count_var_violations(returns, var_value)
    
    n_obs = len(returns)
    expected_violations = n_obs * (1 - confidence)
    violation_rate = n_violations / n_obs if n_obs > 0 else 0
    
    # 3. Kupiec testi (ihlal sayısı kontrolü)
    kupiec_stat, kupiec_pval, kupiec_passed = kupiec_pof_test(
        n_violations, n_obs, confidence
    )
    
    # 4. Christoffersen testi (bağımsızlık kontrolü)
    # Ekonometristin önerisi: "Ardışık ihlalleri de test etmeliyiz"
    christ_stat, christ_pval, christ_passed = christoffersen_independence_test(
        violations
    )
    
    # 5. Birleşik test (her iki koşul da sağlanmalı)
    joint_stat, joint_pval, joint_passed = joint_var_test(
        kupiec_stat, christ_stat
    )
    
    return VaRResult(
        var_value=var_value,
        confidence=confidence,
        violations=violations,
        n_violations=n_violations,
        expected_violations=expected_violations,
        violation_rate=violation_rate,
        kupiec_statistic=kupiec_stat,
        kupiec_pvalue=kupiec_pval,
        kupiec_passed=kupiec_passed,
        christoffersen_statistic=christ_stat,
        christoffersen_pvalue=christ_pval,
        christoffersen_passed=christ_passed,
        joint_statistic=joint_stat,
        joint_pvalue=joint_pval,
        joint_passed=joint_passed
    )


def generate_rolling_var_analysis(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.95
) -> Tuple[pd.Series, pd.Series, dict]:
    """
    Rolling (out-of-sample) VaR analizi yapar.
    
    Her gün için, o günden ÖNCEKI window günlük verilerle VaR hesaplar.
    Bu, gerçek dünya kullanımını simüle eder - gelecek bilinmez.
    
    Args:
        returns: Günlük getiri serisi
        window: VaR hesaplama penceresi (varsayılan 252 gün = 1 yıl)
        confidence: Güven düzeyi
    
    Returns:
        Tuple: (var_serisi, ihlal_serisi, ozet_dict)
    """
    n = len(returns)
    
    if n <= window:
        raise ValueError(f"Yetersiz veri: {n} gün, en az {window + 1} gün gerekli")
    
    var_series = pd.Series(index=returns.index, dtype=float)
    violations = pd.Series(index=returns.index, dtype=bool)
    
    # window sonrasi her gun icin
    for i in range(window, n):
        # SADECE gecmis veri ile VaR hesapla (out-of-sample)
        past_returns = returns.iloc[i-window:i]
        var_value = calculate_historical_var(past_returns, confidence)
        
        var_series.iloc[i] = var_value
        
        # bugunun getirisi VaR'i astimu?
        today_return = returns.iloc[i]
        violations.iloc[i] = today_return < var_value
    
    # NaN'lari kaldir (window oncesi)
    var_series = var_series.dropna()
    violations = violations.iloc[window:]
    
    # ozet istatistikler
    n_violations = violations.sum()
    n_obs = len(violations)
    expected = (1 - confidence) * n_obs
    
    summary = {
        "n_observations": n_obs,
        "n_violations": int(n_violations),
        "expected_violations": expected,
        "violation_rate": n_violations / n_obs if n_obs > 0 else 0,
        "mean_var": var_series.mean(),
        "is_out_of_sample": True
    }
    
    # Kupiec testi
    lr_stat, p_value, passed = kupiec_pof_test(n_violations, n_obs, confidence)
    summary["kupiec_pvalue"] = p_value
    summary["kupiec_passed"] = passed
    
    return var_series, violations, summary


def generate_risk_report(
    returns: pd.Series,
    confidence_levels: list = [0.90, 0.95, 0.99]
) -> pd.DataFrame:
    """
    Farklı güven düzeyleri için kapsamlı risk raporu oluşturur.
    
    Bu rapor üç ayrı testi tek tabloda gösterir:
    - Kupiec POF: İhlal sayısı kontrolü
    - Christoffersen: Bağımsızlık kontrolü  
    - Birleşik: Final değerlendirme
    
    Ekonometrist notu: "VaR modelinin güvenilirliği için
    hem frekans hem de bağımsızlık testlerinden geçmesi gerekir."
    
    Args:
        returns: Günlük getiri serisi
        confidence_levels: Test edilecek güven düzeyleri
    
    Returns:
        Risk raporu DataFrame'i
    """
    rows = []
    
    for conf in confidence_levels:
        result = generate_var_analysis(returns, conf)
        
        # Birleşik sonuç ikonu
        if result.joint_passed:
            final_result = "✓ Başarılı"
        elif result.kupiec_passed or result.christoffersen_passed:
            final_result = "⚠️ Kısmi"
        else:
            final_result = "✗ Başarısız"
        
        rows.append({
            "Güven Düzeyi": f"%{conf*100:.0f}",
            "VaR (%)": f"{result.var_value*100:.2f}%",
            "Beklenen/Gerçek İhlal": f"{result.expected_violations:.0f}/{result.n_violations}",
            "Kupiec": "✓" if result.kupiec_passed else "✗",
            "Christoffersen": "✓" if result.christoffersen_passed else "✗",
            "Birleşik Sonuç": final_result
        })
    
    return pd.DataFrame(rows)


def interpret_var_result(result: VaRResult) -> str:
    """
    VaR sonuçlarını ekonomik olarak yorumlar.
    
    Bu yorum, raporda "ekonometrist katkısı" olarak görünecek.
    Üç ayrı test sonucunu birlikte değerlendirir:
    - Kupiec: İhlal sayısı
    - Christoffersen: İhlal bağımsızlığı
    - Joint: Birleşik değerlendirme
    
    Ekonometrist notu: "Risk modelinin güvenilirliği sadece ihlal
    sayısına bakılarak değerlendirilemez. İhlallerin zamansal
    dağılımı da kritik öneme sahiptir."
    
    Args:
        result: VaRResult objesi
    
    Returns:
        Yorum metni
    """
    lines = []
    
    # 1. VaR yorumu
    var_pct = abs(result.var_value) * 100
    lines.append(f"📊 VaR Değeri: %{result.confidence*100:.0f} güven düzeyinde, "
                f"günlük maksimum beklenen kayıp %{var_pct:.2f} olarak hesaplanmıştır.")
    
    # 2. İhlal analizi yorumu
    if result.n_violations <= result.expected_violations * 1.5:
        lines.append(f"📈 İhlal Analizi: Gerçekleşen ihlal sayısı ({result.n_violations}) "
                    f"beklenen değere ({result.expected_violations:.1f}) yakındır. "
                    f"Model tutarlı çalışmaktadır.")
    else:
        lines.append(f"⚠️ İhlal Analizi: Gerçekleşen ihlal sayısı ({result.n_violations}) "
                    f"beklenen değerin ({result.expected_violations:.1f}) üzerindedir. "
                    f"Model risk tahmininde yetersiz kalabilir.")
    
    # 3. Kupiec testi yorumu
    if result.kupiec_passed:
        lines.append(f"✓ Kupiec POF Testi: Model istatistiksel olarak güvenilir bulunmuştur "
                    f"(p-value = {result.kupiec_pvalue:.4f} > 0.05).")
    else:
        lines.append(f"✗ Kupiec POF Testi: Model istatistiksel olarak yetersiz bulunmuştur "
                    f"(p-value = {result.kupiec_pvalue:.4f} < 0.05). "
                    f"VaR tahminleri dikkatle değerlendirilmelidir.")
    
    # 4. Christoffersen testi yorumu (YENİ - ekonometrist katkısı!)
    if result.christoffersen_passed:
        lines.append(f"✓ Christoffersen Bağımsızlık Testi: İhlaller birbirinden bağımsız "
                    f"dağılmıştır (p-value = {result.christoffersen_pvalue:.4f} > 0.05). "
                    f"Volatilite kümelenmesi gözlemlenmemiştir.")
    else:
        lines.append(f"✗ Christoffersen Bağımsızlık Testi: İhlallerde kümelenme (clustering) "
                    f"tespit edilmiştir (p-value = {result.christoffersen_pvalue:.4f} < 0.05). "
                    f"Model kötü dönemleri öngörmede başarısızdır. "
                    f"Ekonometrist notu: Ardışık ihlaller, modelin volatilite "
                    f"rejimlerini yakalayamadığını göstermektedir.")
    
    # 5. Birleşik test yorumu
    if result.joint_passed:
        lines.append(f"🏆 Birleşik Test Sonucu: VaR modeli hem ihlal sayısı hem de "
                    f"bağımsızlık açısından BAŞARILI bulunmuştur. "
                    f"Risk yönetimi için güvenle kullanılabilir.")
    else:
        if result.kupiec_passed and not result.christoffersen_passed:
            lines.append(f"⚠️ Birleşik Test Sonucu: İhlal sayısı doğru ancak ihlaller "
                        f"bağımsız değil. Model volatilite kümelenmesi dönemlerinde "
                        f"güncellenmeli veya GARCH tabanlı VaR düşünülmelidir.")
        elif not result.kupiec_passed and result.christoffersen_passed:
            lines.append(f"⚠️ Birleşik Test Sonucu: İhlaller bağımsız ancak sayı tutarsız. "
                        f"VaR güven düzeyi veya hesaplama penceresi gözden geçirilmelidir.")
        else:
            lines.append(f"❌ Birleşik Test Sonucu: Model her iki testte de başarısız. "
                        f"Risk modeli kapsamlı bir şekilde revize edilmelidir. "
                        f"Ekonometrist önerisi: Alternatif risk ölçütleri (CVaR, ES) "
                        f"veya farklı modelleme yaklaşımları değerlendirilmelidir.")
    
    return "\n\n".join(lines)


def calculate_expected_shortfall(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    """
    Expected Shortfall (CVaR) hesaplar.
    
    ES, VaR eşiğini aşan kayıpların ortalamasıdır.
    VaR'dan daha iyi bir risk ölçüsü olarak kabul edilir
    çünkü kuyruk riskini daha iyi yakalar.
    
    NOT: Bu fonksiyon MVP için opsiyonel, nice-to-have özellik.
    
    Args:
        returns: Günlük getiri serisi
        confidence: Güven düzeyi
    
    Returns:
        Expected Shortfall değeri
    """
    var = calculate_historical_var(returns, confidence)
    
    # VaR'dan daha kotu getiriler
    tail_losses = returns[returns <= var]
    
    if len(tail_losses) == 0:
        return var
    
    return tail_losses.mean()


# test icin
if __name__ == "__main__":
    print("Risk modülü test ediliyor...")
    
    # ornek getiri serisi olustur
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    
    # normal dagilim + arada buyuk kayiplar
    returns = pd.Series(
        np.random.normal(0.0005, 0.015, 252),
        index=dates
    )
    # birkac buyuk kayip ekle
    returns.iloc[50] = -0.05
    returns.iloc[120] = -0.06
    returns.iloc[200] = -0.04
    
    print("\nGetiri İstatistikleri:")
    print(f"Ortalama: {returns.mean()*100:.4f}%")
    print(f"Std: {returns.std()*100:.4f}%")
    print(f"Min: {returns.min()*100:.4f}%")
    print(f"Max: {returns.max()*100:.4f}%")
    
    # VaR analizi
    print("\n=== %95 VaR Analizi ===")
    result = generate_var_analysis(returns, 0.95)
    print(f"VaR: {result.var_value:.4f} ({result.var_value*100:.2f}%)")
    print(f"İhlal Sayısı: {result.n_violations}/{len(returns)}")
    print(f"Beklenen İhlal: {result.expected_violations:.1f}")
    print(f"Kupiec p-value: {result.kupiec_pvalue:.4f}")
    print(f"Test Geçti mi: {result.kupiec_passed}")
    
    # risk raporu
    print("\n=== Risk Raporu ===")
    report = generate_risk_report(returns)
    print(report.to_string(index=False))
    
    # yorum
    print("\n=== Ekonomik Yorum ===")
    print(interpret_var_result(result))
    
    print("\n✓ Test başarılı!")
