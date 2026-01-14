"""
Veri Çekme ve Önbellekleme Modülü
=================================
Bu modül Yahoo Finance'den hisse senedi verilerini çeker,
önbelleğe alır ve eksik günleri doldurur.

Önemli: SPY (S&P 500 ETF) benchmark olarak her zaman otomatik çekilir.
"""

import os
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Cache klasörünün yolu (proje kök dizinine göre)
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")

# Benchmark sembolü - her zaman otomatik çekilecek
BENCHMARK_SYMBOL = "SPY"


@dataclass
class DataResult:
    """Veri çekme sonucu - demo data kullanıldığında flag döner."""
    stock_prices: pd.DataFrame
    benchmark_prices: Optional[pd.DataFrame]
    is_demo_data: bool = False  # demo/synthetic veri mi?
    failed_tickers: List[str] = None  # çekilemeyen ticker'lar
    
    def __post_init__(self):
        if self.failed_tickers is None:
            self.failed_tickers = []



def _ensure_cache_dir():
    """
    Cache klasörü yoksa oluşturur.
    İlk çalıştırmada klasör olmayabilir diye kontrol ediyoruz.
    """
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)


def _exponential_backoff(func, max_retries: int = 5, base_delay: float = 1.0):
    """
    Rate limit durumunda üstel geri çekilme stratejisi.
    
    Nasıl çalışır:
    - İlk hata: 1 saniye bekle
    - İkinci hata: 2 saniye bekle
    - Üçüncü hata: 4 saniye bekle
    - vs...
    
    Args:
        func: Çalıştırılacak fonksiyon
        max_retries: Maksimum deneme sayısı
        base_delay: Başlangıç bekleme süresi (saniye)
    
    Returns:
        Fonksiyonun sonucu
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                # her denemede bekleme süresini 2 katına çıkar
                wait_time = base_delay * (2 ** attempt)
                print(f"Hata oluştu, {wait_time:.1f} saniye bekleniyor... (Deneme {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
    
    # tüm denemeler başarısız olduysa son hatayı fırlat
    raise last_exception


def _get_cache_filename(ticker: str, start_date: str, end_date: str) -> str:
    """
    Cache dosyası için benzersiz isim oluşturur.
    Örnek: AAPL_2023-01-01_2024-01-01.csv
    """
    # tarih formatını düzelt (/ yerine - kullan)
    start_clean = start_date.replace("/", "-")
    end_clean = end_date.replace("/", "-")
    return f"{ticker}_{start_clean}_{end_clean}.csv"


def save_to_cache(df: pd.DataFrame, ticker: str, start_date: str, end_date: str) -> str:
    """
    DataFrame'i CSV olarak cache klasörüne kaydeder.
    
    Args:
        df: Kaydedilecek veri
        ticker: Hisse senedi sembolü
        start_date: Başlangıç tarihi
        end_date: Bitiş tarihi
    
    Returns:
        Kaydedilen dosyanın yolu
    """
    _ensure_cache_dir()
    filename = _get_cache_filename(ticker, start_date, end_date)
    filepath = os.path.join(CACHE_DIR, filename)
    df.to_csv(filepath)
    print(f"✓ {ticker} verisi önbelleğe kaydedildi: {filename}")
    return filepath


@dataclass
class CacheAnalysis:
    """Cache analiz sonucu - eksik veri bilgisini tutar."""
    data: Optional[pd.DataFrame]  # Cache'den okunan veri
    coverage: float               # Kapsama oranı (0-1)
    cache_start: Optional[datetime] = None
    cache_end: Optional[datetime] = None
    missing_start: Optional[datetime] = None  # Eksik başlangıç tarihi
    missing_end: Optional[datetime] = None    # Eksik bitiş tarihi
    missing_days: int = 0                     # Eksik gün sayısı
    is_complete: bool = False                 # Tam kapsama var mı?


def analyze_cache(ticker: str, start_date: str, end_date: str) -> CacheAnalysis:
    """
    Cache durumunu analiz eder ve eksik veri bilgisini döndürür.
    
    Bu fonksiyon:
    - Cache'de ne kadar veri olduğunu kontrol eder
    - Eksik tarih aralığını tespit eder
    - Kapsama oranını hesaplar
    
    Args:
        ticker: Hisse senedi sembolü
        start_date: İstenen başlangıç tarihi
        end_date: İstenen bitiş tarihi
    
    Returns:
        CacheAnalysis objesi
    """
    _ensure_cache_dir()
    
    requested_start = pd.to_datetime(start_date)
    requested_end = pd.to_datetime(end_date)
    expected_days = len(pd.bdate_range(requested_start, requested_end))
    
    # Boş sonuç
    empty_result = CacheAnalysis(
        data=None, coverage=0.0, missing_start=requested_start,
        missing_end=requested_end, missing_days=expected_days
    )
    
    if expected_days == 0:
        return empty_result
    
    # En iyi cache adayını bul
    best_df = None
    best_coverage = 0.0
    best_cache_start = None
    best_cache_end = None
    
    try:
        cache_files = [f for f in os.listdir(CACHE_DIR) 
                       if f.startswith(f"{ticker}_") and f.endswith(".csv")]
    except Exception:
        return empty_result
    
    for fname in cache_files:
        fpath = os.path.join(CACHE_DIR, fname)
        
        try:
            df = pd.read_csv(fpath, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index)
            
            if len(df) == 0:
                continue
            
            cache_start = df.index.min()
            cache_end = df.index.max()
            
            # İstenen aralıkla kesişen veriyi al
            filtered = df[(df.index >= requested_start) & (df.index <= requested_end)]
            if len(filtered) == 0:
                continue
            
            coverage = len(filtered) / expected_days
            
            if coverage > best_coverage:
                best_coverage = coverage
                best_df = filtered
                best_cache_start = cache_start
                best_cache_end = cache_end
                
        except Exception:
            continue
    
    if best_df is None:
        return empty_result
    
    # Eksik tarih aralığını hesapla
    missing_start = None
    missing_end = None
    missing_days = 0
    
    if best_coverage < 1.0:
        # Başta mı eksik?
        if best_cache_start > requested_start:
            missing_start = requested_start
            missing_end = best_cache_start - timedelta(days=1)
            missing_days += len(pd.bdate_range(missing_start, missing_end))
        
        # Sonda mı eksik?
        if best_cache_end < requested_end:
            if missing_start is None:
                missing_start = best_cache_end + timedelta(days=1)
            missing_end = requested_end
            missing_days += len(pd.bdate_range(best_cache_end + timedelta(days=1), requested_end))
    
    return CacheAnalysis(
        data=best_df,
        coverage=best_coverage,
        cache_start=best_cache_start,
        cache_end=best_cache_end,
        missing_start=missing_start,
        missing_end=missing_end,
        missing_days=missing_days,
        is_complete=(best_coverage >= 0.9999)
    )


def load_from_cache(ticker: str, start_date: str, end_date: str, 
                    allow_partial: bool = True) -> Optional[pd.DataFrame]:
    """
    Cache'den veri okumaya çalışır.
    
    AKILLI CACHE STRATEJİSİ:
    ========================
    1. TAM EŞLEŞME: Aynı tarih aralığı için daha önce çekilmiş veri varsa kullan
    2. TAM KAPSAMA: Daha geniş aralıklı cache varsa, isteneni filtrele
    3. KISMİ KAPSAMA (%90+): allow_partial=True ise kabul et
    
    Args:
        ticker: Hisse senedi sembolü
        start_date: Başlangıç tarihi (YYYY-MM-DD)
        end_date: Bitiş tarihi (YYYY-MM-DD)
        allow_partial: %90+ kısmi kapsamayı kabul et (varsayılan: True)
    
    Returns:
        DataFrame veya None (cache yetersizse)
    """
    analysis = analyze_cache(ticker, start_date, end_date)
    
    if analysis.data is None:
        return None
    
    # Tam kapsama
    if analysis.is_complete:
        print(f"✓ {ticker} cache tam kapsıyor ({len(analysis.data)} gün)")
        return analysis.data
    
    # Kısmi kapsama
    if analysis.coverage >= 0.90:
        if allow_partial:
            print(f"✓ {ticker} cache %{analysis.coverage*100:.0f} kapsıyor ({len(analysis.data)} gün)")
            if analysis.missing_days > 0:
                print(f"  ℹ️ Eksik: {analysis.missing_start.strftime('%Y-%m-%d') if analysis.missing_start else '?'} → "
                      f"{analysis.missing_end.strftime('%Y-%m-%d') if analysis.missing_end else '?'} ({analysis.missing_days} iş günü)")
            return analysis.data
        else:
            print(f"ℹ️ {ticker} cache %{analysis.coverage*100:.0f} kapsıyor ama tam veri istendi")
            return None
    
    # Yetersiz kapsama
    if analysis.coverage > 0:
        print(f"ℹ️ {ticker} cache sadece %{analysis.coverage*100:.0f} kapsıyor, canlı veri denenecek")
    
    return None


def _fetch_live_data(ticker: str, start_date: str, end_date: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Yahoo Finance'den canlı veri çeker (internal helper).
    
    Args:
        ticker: Hisse senedi sembolü
        start_date: Başlangıç tarihi
        end_date: Bitiş tarihi
    
    Returns:
        Tuple: (DataFrame veya None, kaynak)
    """
    try:
        print(f"⏳ {ticker} canlı veri indiriliyor ({start_date} → {end_date})...")
        
        def _download():
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                raise ValueError(f"{ticker} için veri bulunamadı!")
            return data
        
        data = _exponential_backoff(_download, max_retries=3, base_delay=1.0)
        
        # yfinance v1.0: 'Adj Close' kaldırıldı, artık 'Close' kullanılıyor
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                close_prices = data['Close']
            elif 'Adj Close' in data.columns.get_level_values(0):
                close_prices = data['Adj Close']
            else:
                close_prices = data.iloc[:, 0]
        else:
            if 'Close' in data.columns:
                close_prices = data[['Close']]
            elif 'Adj Close' in data.columns:
                close_prices = data[['Adj Close']]
            else:
                close_prices = data.iloc[:, :1]
        
        # tek kolon varsa DataFrame olarak tut
        if isinstance(close_prices, pd.Series):
            close_prices = close_prices.to_frame(name=ticker)
        else:
            close_prices.columns = [ticker]
        
        if len(close_prices) > 0:
            print(f"✓ {ticker} canlı veri alındı: {len(close_prices)} gün")
            return close_prices, "live"
        
        return None, "empty"
        
    except Exception as e:
        print(f"⚠️ {ticker} canlı veri hatası: {e}")
        return None, "error"


def _filter_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    DataFrame'i belirtilen tarih aralığına filtreler.
    Demo verileri daha geniş bir aralık içerebilir.
    """
    # index'in datetime oldugunu garanti et
    df.index = pd.to_datetime(df.index)
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # tarih araligi filtrele
    mask = (df.index >= start) & (df.index <= end)
    filtered = df.loc[mask]
    
    if len(filtered) == 0:
        print(f"⚠️ Belirtilen tarih aralığında veri yok, tüm veri kullanılıyor")
        return df
    
    return filtered


def load_csv_fallback(filepath: str) -> pd.DataFrame:
    """
    Kullanıcının manuel olarak yüklediği CSV dosyasını okur.
    Bu fonksiyon API çalışmadığında yedek olarak kullanılır.
    
    Args:
        filepath: CSV dosyasının tam yolu
    
    Returns:
        DataFrame (index=tarih, kolonlar=hisse sembolleri)
    """
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"✓ CSV dosyası yüklendi: {filepath}")
    return df


def fetch_single_stock(
    ticker: str, 
    start_date: str, 
    end_date: str,
    force_live: bool = False,      # False = cache öncelikli (varsayılan)
    force_complete: bool = False   # True = eksik veriyi tamamla
) -> Tuple[pd.DataFrame, str]:
    """
    Tek bir hisse senedi için veri çeker.
    
    VERİ ÇEKME STRATEJİSİ:
    ======================
    1. Cache'e bak (akıllı cache - tam/kısmi kapsama kontrolü)
    2. Cache yetersizse → Canlı veri çek (yfinance)
    3. Canlı da başarısızsa → Cache fallback (varsa)
    4. Hiçbiri olmazsa → Sentetik veri üret (demo amaçlı)
    
    YENİ: INCREMENTAL FETCH (force_complete=True)
    =============================================
    - Cache'de eksik olan kısmı tespit et
    - SADECE eksik tarihleri çek (API tasarrufu)
    - Eski ve yeni veriyi birleştir
    - Güncellenmiş cache'i kaydet
    
    Args:
        ticker: Hisse senedi sembolü (örn: "AAPL")
        start_date: Başlangıç tarihi (YYYY-MM-DD)
        end_date: Bitiş tarihi (YYYY-MM-DD)
        force_live: True ise cache'i atla, direkt canlı dene
        force_complete: True ise eksik veriyi tamamla (incremental fetch)
    
    Returns:
        Tuple: (fiyat_df, veri_kaynagi)
        veri_kaynagi: "live", "cache", "cache+live" veya "synthetic"
    """
    data_source = "unknown"
    close_prices = None
    
    # 1. CACHE ANALİZİ
    if not force_live:
        analysis = analyze_cache(ticker, start_date, end_date)
        
        # Tam kapsama - direkt dön
        if analysis.is_complete and analysis.data is not None:
            return analysis.data, "cache"
        
        # Kısmi kapsama
        if analysis.data is not None and analysis.coverage >= 0.90:
            # Kullanıcı tam veri istemiyorsa, kısmi kabul et
            if not force_complete:
                print(f"✓ {ticker} cache %{analysis.coverage*100:.0f} kapsıyor ({len(analysis.data)} gün)")
                if analysis.missing_days > 0:
                    print(f"  ℹ️ Eksik: {analysis.missing_days} iş günü "
                          f"({analysis.missing_end.strftime('%Y-%m-%d') if analysis.missing_end else '?'} tarihine kadar)")
                return analysis.data, "cache"
            
            # INCREMENTAL FETCH: Sadece eksik kısmı çek
            if analysis.missing_end is not None and analysis.missing_days > 0:
                print(f"🔄 {ticker} için eksik {analysis.missing_days} gün çekiliyor...")
                
                # Eksik kısım sonda ise (en yaygın durum)
                if analysis.cache_end and analysis.cache_end < pd.to_datetime(end_date):
                    try:
                        # Sadece eksik tarihleri çek
                        incremental_start = (analysis.cache_end + timedelta(days=1)).strftime('%Y-%m-%d')
                        incremental_data, inc_source = _fetch_live_data(ticker, incremental_start, end_date)
                        
                        if incremental_data is not None and len(incremental_data) > 0:
                            # Eski ve yeni veriyi birleştir
                            combined = pd.concat([analysis.data, incremental_data])
                            combined = combined[~combined.index.duplicated(keep='last')]
                            combined = combined.sort_index()
                            
                            # Güncellenmiş cache'i kaydet
                            save_to_cache(combined, ticker, start_date, end_date)
                            
                            print(f"✓ {ticker} güncellendi: {len(analysis.data)} + {len(incremental_data)} = {len(combined)} gün")
                            return combined, "cache+live"
                    except Exception as e:
                        print(f"⚠️ {ticker} incremental fetch başarısız: {e}")
                        # Hata durumunda mevcut cache'i kullan
                        return analysis.data, "cache"
        
        # %90'ın altında ama bir miktar veri var - fallback için sakla
        cached_fallback = analysis.data if analysis.data is not None else None
    else:
        cached_fallback = None
    
    # 2. TAM CANLI VERİ ÇEK
    try:
        print(f"⏳ {ticker} canlı veri indiriliyor...")
        
        def _download():
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                raise ValueError(f"{ticker} için veri bulunamadı!")
            return data
        
        data = _exponential_backoff(_download, max_retries=3, base_delay=1.0)
        
        # yfinance v1.0: 'Adj Close' kaldırıldı, artık 'Close' kullanılıyor
        # Sütunlar MultiIndex formatında: ('Close', 'AAPL')
        if isinstance(data.columns, pd.MultiIndex):
            # MultiIndex durumu - ('Close', ticker) formatında
            if 'Close' in data.columns.get_level_values(0):
                close_prices = data['Close']
            elif 'Adj Close' in data.columns.get_level_values(0):
                close_prices = data['Adj Close']
            else:
                # İlk sayısal kolonu al
                close_prices = data.iloc[:, 0]
        else:
            # Düz kolonlar
            if 'Close' in data.columns:
                close_prices = data[['Close']]
            elif 'Adj Close' in data.columns:
                close_prices = data[['Adj Close']]
            else:
                close_prices = data.iloc[:, :1]
        
        # tek kolon varsa DataFrame olarak tut
        if isinstance(close_prices, pd.Series):
            close_prices = close_prices.to_frame(name=ticker)
        else:
            close_prices.columns = [ticker]
        
        # Tarih aralığı validasyonu
        if len(close_prices) > 0:
            actual_start = close_prices.index.min()
            actual_end = close_prices.index.max()
            requested_start = pd.to_datetime(start_date)
            requested_end = pd.to_datetime(end_date)
            
            # Minimum veri kontrolü
            expected_days = len(pd.bdate_range(requested_start, requested_end))
            actual_days = len(close_prices)
            coverage = actual_days / expected_days if expected_days > 0 else 0
            
            if coverage >= 0.50:  # En az %50 veri varsa kabul et
                data_source = "live"
                
                if coverage < 0.90:
                    print(f"⚠️ {ticker} kısmi veri alındı: {actual_days}/{expected_days} gün (%{coverage*100:.0f})")
                else:
                    print(f"✓ {ticker} canlı veri alındı: {actual_days} gün")
                
                # Başarılı canlı veriyi cache'e kaydet
                save_to_cache(close_prices, ticker, start_date, end_date)
                return close_prices, data_source
            else:
                print(f"⚠️ {ticker} canlı veri yetersiz: sadece {actual_days} gün ({coverage*100:.0f}%)")
        else:
            print(f"⚠️ {ticker} canlı veri boş")
        
    except Exception as e:
        print(f"⚠️ {ticker} canlı veri hatası: {e}")
    
    # 3. CACHE'E BAK (canlı başarısız olursa fallback)
    # Not: force_live=True durumunda cache henüz kontrol edilmedi
    if force_live:
        cached = load_from_cache(ticker, start_date, end_date)
        if cached is not None and len(cached) > 0:
            data_source = "cache"
            print(f"ℹ️ {ticker} cache'den yüklendi (fallback): {len(cached)} gün")
            return cached, data_source
    
    # 4. SENTETİK VERİ ÜRET (son çare - sadece demo amaçlı!)
    try:
        print(f"🔧 {ticker} için sentetik veri üretiliyor (DEMO - gerçek veri değil!)...")
        
        # scripts klasöründen import
        import sys
        project_root = os.path.dirname(os.path.dirname(__file__))
        scripts_path = os.path.join(project_root, "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        
        from generate_demo_data import generate_realistic_prices, STOCK_INFO
        
        # tarih araligi
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.bdate_range(start, end)
        n_days = len(dates)
        
        if n_days == 0:
            raise ValueError(f"Geçersiz tarih aralığı: {start_date} - {end_date}")
        
        # hisse bilgisi varsa kullan, yoksa varsayılan değerler
        if ticker in STOCK_INFO:
            info = STOCK_INFO[ticker]
        else:
            # varsayılan parametreler
            info = {"start_price": 100, "annual_return": 0.10, "annual_vol": 0.25}
            print(f"ℹ️ {ticker} için varsayılan parametreler kullanılıyor")
        
        # fiyat serisi üret
        prices = generate_realistic_prices(
            info["start_price"],
            info["annual_return"],
            info["annual_vol"],
            n_days,
            seed=hash(ticker) % 1000  # her ticker için tutarlı seed
        )
        
        close_prices = pd.DataFrame({ticker: prices}, index=dates)
        data_source = "synthetic"
        
        print(f"⚠️ {ticker} SENTETİK veri üretildi: {len(close_prices)} gün (GERÇEK VERİ DEĞİL!)")
        
        return close_prices, data_source
        
    except Exception as synth_error:
        print(f"❌ {ticker} sentetik veri üretilemedi: {synth_error}")
        raise ValueError(
            f"{ticker} için veri alınamadı! "
            f"Canlı, cache veya sentetik veri üretilemedi."
        )


def fetch_stock_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    include_benchmark: bool = True,
    force_complete: bool = False  # YENİ: Eksik veriyi tamamla
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Birden fazla hisse senedi için veri çeker.
    
    ÖNEMLİ: Benchmark (SPY) her zaman otomatik olarak çekilir!
    Bu sayede backtest aşamasında karşılaştırma yapabiliriz.
    
    Args:
        tickers: Hisse senedi sembolleri listesi (örn: ["AAPL", "MSFT", "GOOGL"])
        start_date: Başlangıç tarihi (YYYY-MM-DD)
        end_date: Bitiş tarihi (YYYY-MM-DD)
        include_benchmark: Benchmark (SPY) dahil edilsin mi? (varsayılan: True)
        force_complete: Eksik veriyi tamamla (incremental fetch)
    
    Returns:
        Tuple: (hisse_fiyatlari, benchmark_fiyatlari, meta_bilgi)
        meta_bilgi: {"sources": {ticker: source}, "failed": [ticker_list], "missing_info": {...}}
    """
    all_prices = []
    data_sources = {}
    failed_tickers = []
    missing_info = {}  # Eksik veri bilgisi
    
    # kullanıcının seçtiği hisseleri çek
    for ticker in tickers:
        ticker = ticker.strip().upper()  # boşlukları temizle, büyük harfe çevir
        
        # SPY zaten benchmark olarak çekileceği için listeye ekleme
        if ticker == BENCHMARK_SYMBOL and include_benchmark:
            continue
        
        # Önce cache analizi yap (eksik bilgi için)
        if not force_complete:
            analysis = analyze_cache(ticker, start_date, end_date)
            if analysis.missing_days > 0 and analysis.coverage >= 0.90:
                missing_info[ticker] = {
                    "coverage": analysis.coverage,
                    "missing_days": analysis.missing_days,
                    "missing_end": analysis.missing_end.strftime('%Y-%m-%d') if analysis.missing_end else None
                }
            
        try:
            prices, source = fetch_single_stock(ticker, start_date, end_date, 
                                                force_complete=force_complete)
            all_prices.append(prices)
            data_sources[ticker] = source
        except Exception as e:
            print(f"⚠️ {ticker} için veri çekilemedi: {e}")
            failed_tickers.append(ticker)
    
    # hisseleri birleştir
    if not all_prices:
        raise ValueError("Hiçbir hisse için veri çekilemedi!")
    
    stock_prices = pd.concat(all_prices, axis=1)
    
    # benchmark (SPY) çek
    benchmark_prices = None
    if include_benchmark:
        try:
            benchmark_prices, bench_source = fetch_single_stock(
                BENCHMARK_SYMBOL, start_date, end_date, force_complete=force_complete
            )
            data_sources[BENCHMARK_SYMBOL] = bench_source
        except Exception as e:
            print(f"⚠️ Benchmark ({BENCHMARK_SYMBOL}) çekilemedi: {e}")
    
    # eksik günleri hizala
    stock_prices = align_and_fill(stock_prices)
    if benchmark_prices is not None:
        benchmark_prices = align_and_fill(benchmark_prices)
    
    # stock ve benchmark'i ortak tarihlere hizala
    stock_prices, benchmark_prices = align_dates(stock_prices, benchmark_prices)
    
    # meta bilgi
    meta_info = {
        "sources": data_sources,
        "failed": failed_tickers,
        "all_live": all(s == "live" for s in data_sources.values()),
        "any_cache": any(s == "cache" for s in data_sources.values()),
        "any_incremental": any(s == "cache+live" for s in data_sources.values()),
        "missing_info": missing_info  # Eksik veri bilgisi (kullanıcıya gösterilecek)
    }
    
    return stock_prices, benchmark_prices, meta_info


def align_and_fill(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """
    Eksik günleri doldurur ve tarihleri hizalar.
    
    Forward-fill yöntemi:
    - Bir gün için veri yoksa, önceki günün değerini kullanır
    - Bu borsa tatil günleri için mantıklı bir yaklaşım
    
    Args:
        df: Fiyat DataFrame'i
        method: Doldurma yöntemi ("ffill" = önceki değer)
    
    Returns:
        Eksik değerleri doldurulmuş DataFrame
    """
    # once forward fill yap
    df = df.ffill()
    
    # baştaki NaN'lar için backward fill yap (ilk günler veri yoksa)
    df = df.bfill()
    
    # hala NaN varsa (tüm satır boşsa) o satırı sil
    df = df.dropna()
    
    return df


def align_dates(
    stock_prices: pd.DataFrame,
    benchmark_prices: Optional[pd.DataFrame]
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Stock ve benchmark verilerini ortak tarihlere hizalar.
    
    Bu fonksiyon, farklı tarih aralıklarına sahip verilerin
    doğru karşılaştırılmasını sağlar.
    
    Args:
        stock_prices: Hisse fiyatları DataFrame'i
        benchmark_prices: Benchmark fiyatları DataFrame'i (opsiyonel)
    
    Returns:
        Tuple: (hizalanmış_stock, hizalanmış_benchmark)
    """
    if benchmark_prices is None:
        return stock_prices, None
    
    # ortak tarihleri bul
    common_dates = stock_prices.index.intersection(benchmark_prices.index)
    
    if len(common_dates) == 0:
        print("⚠️ Stock ve benchmark arasında ortak tarih bulunamadı!")
        return stock_prices, benchmark_prices
    
    # kaybedilen gun sayisini raporla
    stock_lost = len(stock_prices) - len(common_dates)
    bench_lost = len(benchmark_prices) - len(common_dates)
    
    if stock_lost > 0 or bench_lost > 0:
        print(f"ℹ️ Tarih hizalama: {len(common_dates)} ortak gün "
              f"(stock: -{stock_lost}, benchmark: -{bench_lost})")
    
    # ortak tarihlere filtrele
    aligned_stock = stock_prices.loc[common_dates]
    aligned_benchmark = benchmark_prices.loc[common_dates]
    
    return aligned_stock, aligned_benchmark


def get_cached_tickers() -> List[str]:
    """
    Cache klasöründeki mevcut ticker'ları listeler.
    Kullanıcıya hangi verilerin hazır olduğunu göstermek için.
    
    Returns:
        Ticker sembolleri listesi
    """
    _ensure_cache_dir()
    
    tickers = set()
    for filename in os.listdir(CACHE_DIR):
        if filename.endswith(".csv"):
            # dosya adından ticker'ı çıkart (ilk _ öncesi)
            ticker = filename.split("_")[0]
            tickers.add(ticker)
    
    return sorted(list(tickers))


def clear_cache():
    """
    Tüm cache dosyalarını siler.
    Dikkatli kullan - tüm önbellek silinir!
    """
    _ensure_cache_dir()
    
    count = 0
    for filename in os.listdir(CACHE_DIR):
        if filename.endswith(".csv"):
            filepath = os.path.join(CACHE_DIR, filename)
            os.remove(filepath)
            count += 1
    
    print(f"✓ {count} cache dosyası silindi")


# test icin
if __name__ == "__main__":
    # basit test
    print("Veri modülü test ediliyor...")
    
    tickers = ["AAPL", "MSFT"]
    start = "2023-01-01"
    end = "2024-01-01"
    
    prices, benchmark = fetch_stock_data(tickers, start, end)
    
    print("\nHisse Fiyatları:")
    print(prices.head())
    
    print("\nBenchmark (SPY):")
    print(benchmark.head())
    
    print("\n✓ Test başarılı!")
