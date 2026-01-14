# Akıllı Portföy Optimizasyon & Risk Dashboard

Modern Portföy Teorisi (MPT) tabanlı, mean-variance optimizasyonu yapan ve walk-forward backtest ile sonuçları doğrulayan bir **Streamlit dashboard** uygulaması.

## 🎯 Proje Amacı

Kullanıcının seçtiği hisse senedi evreninde:
- **Mean-Variance Optimizasyonu** ile optimal portföy ağırlıkları hesaplamak
- **Walk-Forward Backtest** ile geçmiş performansı test etmek
- **VaR/CVaR Analizi** ve istatistiksel testlerle risk değerlendirmesi yapmak
- **Stres Testi** ile kriz dönemlerinde portföy davranışını analiz etmek
- Tüm süreci **tekrarlanabilir**, **görsel** ve **açıklayıcı** bir dashboard'da sunmak

## ✨ Özellikler

### Optimizasyon
- **Minimum Varyans** ve **Maksimum Sharpe** stratejileri
- **Risk Parity** stratejisi (eşit risk katkısı)
- **Ledoit-Wolf shrinkage** kovaryans tahmini (daha güvenilir ağırlıklar)
- **Risk Katkısı (Risk Contribution)** analizi
- Long-only kısıtı ve maksimum ağırlık limiti
- Etkin Sınır (Efficient Frontier) görselleştirmesi
- **"Neden bu hisse seçildi?"** açıklamaları

### Backtest
- **Walk-Forward metodolojisi** (look-ahead bias önlenir)
- 4 yönlü karşılaştırma: Optimize / Eşit Ağırlık / Risk Parity / SPY Benchmark
- **Transaction Cost (İşlem Maliyeti)** hesaplaması
- Brüt vs Net performans karşılaştırması
- Performans metrikleri: Sharpe, Volatilite, Max Drawdown

### Risk Analizi
- **Historical VaR** hesaplaması
- **CVaR (Expected Shortfall)** - VaR aşıldığında beklenen kayıp
- **Kupiec POF testi** (ihlal sayısı kontrolü)
- **Christoffersen bağımsızlık testi** (ihlal kümelenmesi kontrolü)
- **Stres Testi** - COVID, 2022 düşüşü gibi kriz dönemleri analizi
- Otomatik ekonomik yorum üretimi

### Dinamik Analiz
- **Rolling korelasyon** analizi ve rejim tespiti
- **Rolling volatilite** takibi
- **Parametre duyarlılık** analizi

### Kullanıcı Deneyimi
- **Özet Sonuç Paneli** - "Ne yapmalısın?" tek bakışta
- **Terimler Sözlüğü** - Her metriğin açıklaması
- **Neden bu hisse alındı/alınmadı?** açıklamaları
- **Modüler ve temiz kod yapısı** (components.py)

## 🛠️ Teknolojiler

| Kategori | Araç |
|----------|------|
| Veri Çekme | yfinance |
| Hesaplama | pandas, numpy, scipy |
| Kovaryans | scikit-learn (Ledoit-Wolf) |
| Dashboard | Streamlit |
| Grafikler | Plotly |

## 📁 Proje Yapısı

```
ekofin_project/
├── app/
│   ├── main.py          # Streamlit dashboard ana dosyası
│   └── components.py    # UI bileşenleri (grafikler, paneller)
├── src/
│   ├── data.py          # Veri çekme, cache, incremental fetch
│   ├── returns.py       # Getiri hesaplama, kovaryans, rolling analiz
│   ├── optimize.py      # Mean-variance & risk parity optimizasyon
│   ├── backtest.py      # Walk-forward backtest, stres testi
│   └── risk.py          # VaR, CVaR, Kupiec, Christoffersen testleri
├── data/
│   └── cache/           # Önbellek dosyaları
├── scripts/
│   ├── generate_demo_data.py  # Demo veri üretici
│   └── debug_data.py          # Debug scripti
├── tests/
│   └── test_sanity.py   # Temel testler
├── report/              # Raporlar için klasör
├── requirements.txt     # Python bağımlılıkları
└── README.md
```

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimler
- Python 3.9+
- pip

### 2. Kurulum

```bash
# Proje klasörüne git
cd ekofin_project

# Sanal ortam oluştur (önerilen)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### 3. Çalıştırma

```bash
streamlit run app/main.py
```

Tarayıcıda `http://localhost:8501` adresinde açılacaktır.

## 📊 Kullanım

1. **Sol panelden** hisse sembollerini girin (örn: AAPL, MSFT, GOOGL)
2. **Tarih aralığı** seçin (en az 1.5 yıl önerilir)
3. **Strateji ve parametreleri** ayarlayın
4. **"Analizi Başlat"** butonuna tıklayın

### Örnek Hisseler (ABD)
- **Teknoloji:** AAPL, MSFT, GOOGL, AMZN, META, NVDA
- **Finans:** JPM, BAC, GS, V, MA
- **Sağlık:** JNJ, UNH, PFE
- **ETF:** SPY (benchmark olarak otomatik eklenir)

## 🔧 Teknik Detaylar

### Metodoloji

**Getiri Hesaplama:**
- Logaritmik (log) getiri kullanılır
- Yıllıklaştırma: 252 işlem günü

**Kovaryans Tahmini:**
- Ledoit-Wolf shrinkage (sample covariance yerine)
- Daha stabil ağırlıklar, daha az aşırı pozisyon

**Backtest:**
- Walk-forward: 252 gün eğitim, 21 gün hold
- Her rebalance'da kovaryans sadece eğitim verisiyle hesaplanır

**Risk:**
- Historical simulation VaR
- Kupiec POF: İhlal sayısı testi
- Christoffersen: İhlal bağımsızlığı testi

### Fail-Safe Mekanizmaları
- Optimizer başarısız olursa → Eşit ağırlıklı portföy
- API rate limit → Exponential backoff + cache fallback
- Veri yoksa → Sentetik veri üretimi (demo amaçlı)

## 📈 Çıktılar

- **Özet Sonuç Paneli** - "Ne yapmalısın?" tek bakışta
- **Portföy ağırlıkları** (CSV/JSON export)
- **"Neden bu hisse seçildi?"** açıklamaları
- **Performans metrikleri** tablosu (brüt & net)
- **Equity curve** grafiği (4 strateji karşılaştırması)
- **Drawdown** analizi
- **VaR/CVaR ihlal** grafiği
- **Risk katkısı** analizi ve RC/W oranları
- **Stres testi** - kriz dönemleri performansı
- **Korelasyon matrisi** ve rolling analiz
- **Duyarlılık analizi** raporu

## ⚠️ Önemli Notlar

1. **Geçmiş performans geleceği garanti etmez**
2. Veri kaynağı Yahoo Finance (ücretsiz, araştırma amaçlı)
3. Transaction cost dahil (%0.1 varsayılan)
4. Sadece ABD hisse senetleri (USD)

## 📚 Referanslar

1. **Markowitz, H. (1952)** - Modern Portföy Teorisi
2. **Ledoit, O. & Wolf, M. (2004)** - Shrinkage kovaryans
3. **Kupiec, P.H. (1995)** - VaR ihlal testi
4. **Christoffersen, P.F. (1998)** - Bağımsızlık testi

## 📝 Lisans

Bu proje eğitim ve araştırma amaçlı geliştirilmiştir.
