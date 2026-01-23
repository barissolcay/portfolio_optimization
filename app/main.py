"""
Akıllı Portföy Optimizasyon & Risk Dashboard
=============================================
Streamlit uygulaması - temiz ve modüler yapı.

Kullanım: streamlit run app/main.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from io import StringIO
from datetime import datetime, timedelta
import json

# Kendi modüllerimiz
from src.data import fetch_stock_data, get_cached_tickers, BENCHMARK_SYMBOL
from src.returns import (
    calculate_log_returns, estimate_covariance_ledoit_wolf,
    calculate_expected_returns, calculate_correlation_matrix,
    generate_correlation_report, calculate_rolling_volatility,
    TRADING_DAYS_PER_YEAR
)
from src.optimize import (
    minimize_variance, maximize_sharpe, equal_weight_portfolio,
    calculate_efficient_frontier, get_risk_contribution_summary,
    generate_sensitivity_report, DEFAULT_RISK_FREE_RATE
)
from src.backtest import (
    BacktestConfig, walk_forward_backtest, run_backtest_comparison,
    calculate_drawdown, identify_stress_periods_in_data, calculate_stress_period_metrics
)
from src.risk import (
    generate_var_analysis, generate_risk_report, interpret_var_result,
    calculate_expected_shortfall
)

# UI bileşenleri
from app.components import (
    render_summary_panel, render_backtest_chart, render_drawdown_chart,
    render_weights_chart, render_risk_contribution_chart, render_var_chart,
    render_efficient_frontier, render_correlation_heatmap, render_rolling_correlation_chart,
    render_metrics_table, render_metrics_explanation, render_stress_test_panel,
    render_sensitivity_chart, render_rolling_volatility_chart
)


# =====================
# CACHE FONKSİYONLARI
# =====================

@st.cache_data(show_spinner="Veri indiriliyor...")
def cached_fetch_data(tickers_str: str, start_date: str, end_date: str, force_complete: bool = False):
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    return fetch_stock_data(tickers, start_date, end_date, force_complete=force_complete)


@st.cache_data(show_spinner="Optimizasyon yapılıyor...")
def cached_optimize(_hash: str, prices_json: str, strategy: str, max_weight: float, risk_free_rate: float):
    prices = pd.read_json(StringIO(prices_json))
    returns = calculate_log_returns(prices)
    expected_ret = calculate_expected_returns(returns)
    cov_matrix, shrinkage = estimate_covariance_ledoit_wolf(returns)
    cov_annual = cov_matrix * TRADING_DAYS_PER_YEAR
    
    if strategy == "min_variance":
        weights, vol, success = minimize_variance(cov_annual, max_weight)
        sharpe = None
    else:
        weights, sharpe, vol, success = maximize_sharpe(expected_ret, cov_annual, risk_free_rate, max_weight)
    
    return weights, vol, sharpe, success, expected_ret, cov_annual, shrinkage


@st.cache_data(show_spinner="Backtest yapılıyor...")
def cached_backtest(_hash: str, stock_json: str, bench_json: str, train: int, hold: int, strategy: str, max_w: float, rf: float):
    stock_prices = pd.read_json(StringIO(stock_json))
    bench_prices = pd.read_json(StringIO(bench_json)) if bench_json else None
    
    config = BacktestConfig(train_window=train, hold_period=hold, strategy=strategy, max_weight=max_w, risk_free_rate=rf)
    results = run_backtest_comparison(stock_prices, bench_prices, config)
    
    # Serialize
    serialized = {}
    for name, res in results.items():
        serialized[name] = {
            "equity_curve": res.equity_curve.to_json(),
            "daily_returns": res.daily_returns.to_json(),
            "metrics": res.metrics
        }
    return serialized


# =====================
# SAYFA AYARLARI
# =====================

st.set_page_config(page_title="Portföy Dashboard", page_icon="📊", layout="wide")
st.title("📊 Portföy Optimizasyon Dashboard")

# =====================
# SIDEBAR
# =====================

with st.sidebar:
    st.header("⚙️ Parametreler")
    
    # Hisse seçimi
    default_tickers = "AAPL, MSFT, GOOGL, AMZN, META"
    tickers_input = st.text_input(
        "Hisseler", 
        value=default_tickers, 
        help="ABD borsasından hisse sembolleri. Virgülle ayırın. Örn: AAPL, MSFT, GOOGL"
    )
    
    cached = get_cached_tickers()
    if cached:
        st.caption(f"💾 Cache: {', '.join(cached[:5])}" + ("..." if len(cached) > 5 else ""))
    
    # Tarih
    st.subheader("📅 Tarih Aralığı")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Başlangıç", 
            value=datetime.now() - timedelta(days=3*365),
            help="Analiz başlangıç tarihi. En az 1 yıl veri önerilir."
        )
    with col2:
        end_date = st.date_input(
            "Bitiş", 
            value=datetime.now() - timedelta(days=1),
            help="Analiz bitiş tarihi. Bugünün verisi tamamlanmamış olabilir."
        )
    
    # Strateji
    st.subheader("🎯 Optimizasyon")
    strategy = st.selectbox(
        "Strateji",
        options=["max_sharpe", "min_variance"],
        format_func=lambda x: "Max Sharpe" if x == "max_sharpe" else "Min Varyans",
        help="""
        **Max Sharpe:** Risk-getiri dengesini optimize eder. Daha yüksek getiri için biraz risk alır.
        
        **Min Varyans:** Sadece riski minimize eder. Daha güvenli ama getiri düşük olabilir.
        """
    )
    
    max_weight = st.slider(
        "Max Ağırlık", 0.10, 1.00, 0.30, 0.05,
        help="""
        Tek bir hisseye verilebilecek maksimum ağırlık.
        
        • **%10-20:** Çok çeşitlendirilmiş
        • **%25-35:** Dengeli (önerilen)
        • **%50+:** Konsantre portföy, yüksek risk
        """
    )
    
    risk_free_rate = st.number_input(
        "Risksiz Faiz", 0.00, 0.20, 0.04, 0.005, format="%.3f",
        help="""
        Sharpe oranı hesaplamasında kullanılan risksiz getiri (yıllık).
        
        ABD Hazine tahvili faizi referans alınır (~%4-5).
        Bu oranın üzerinde getiri "risk primi" olarak kabul edilir.
        """
    )
    
    # Backtest
    st.subheader("🔄 Backtest Ayarları")
    train_window = st.number_input(
        "Eğitim Penceresi (gün)", 60, 504, 252, 21,
        help="""
        Portföy ağırlıklarını hesaplamak için kullanılan geçmiş veri süresi.
        
        • **252 gün** = ~1 yıl (standart)
        • **126 gün** = ~6 ay (daha reaktif)
        
        Kısa pencere = hızlı adaptasyon ama gürültüye duyarlı.
        """
    )
    
    hold_period = st.number_input(
        "Hold Periyodu (gün)", 5, 63, 21,
        help="""
        Portföyü yeniden dengelemeden önce tutma süresi.
        
        • **21 gün** = ~1 ay (standart kurumsal tercih)
        • **5 gün** = Haftalık (yüksek işlem maliyeti!)
        
        Kısa hold = daha sık işlem = daha fazla maliyet.
        """
    )
    
    # VaR
    st.subheader("⚠️ Risk Ayarları")
    var_confidence = st.selectbox(
        "VaR Güven Düzeyi", 
        [0.90, 0.95, 0.99], 
        index=1, 
        format_func=lambda x: f"%{x*100:.0f}",
        help="""
        Value at Risk (VaR) için güven düzeyi.
        
        • **%95:** 100 günde 5 gün bu kayıp aşılabilir (standart)
        • **%99:** 100 günde 1 gün bu kayıp aşılabilir (muhafazakar)
        
        Yüksek güven = daha büyük VaR değeri.
        """
    )
    
    # Veri ayarları
    st.subheader("💾 Veri Ayarları")
    force_complete = st.checkbox(
        "🔄 Güncel Veri Zorla", 
        help="""
        **Açık:** Cache'de eksik tarihler varsa internetten indirir.
        
        **Kapalı:** Cache yeterliyse olduğu gibi kullanır (hızlı).
        """
    )
    
    st.divider()
    run_button = st.button("🚀 Analizi Başlat", type="primary", use_container_width=True)
    
    if run_button:
        st.session_state["run"] = True


# =====================
# ANA PANEL
# =====================

if run_button or st.session_state.get("run"):
    try:
        # 1. VERİ ÇEK
        stock_prices, bench_prices, meta = cached_fetch_data(
            tickers_input,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            force_complete
        )
        
        if stock_prices.empty:
            st.error("Veri çekilemedi!")
            st.stop()
        
        actual_start = stock_prices.index.min().strftime("%Y-%m-%d")
        actual_end = stock_prices.index.max().strftime("%Y-%m-%d")
        
        # Veri durumu - kullanıcıya net bilgi ver
        sources = meta.get("sources", {})
        synthetic_tickers = [t for t, s in sources.items() if s == "synthetic"]
        cache_tickers = [t for t, s in sources.items() if s == "cache"]
        live_tickers = [t for t, s in sources.items() if s == "live"]
        
        # Ana durum mesajı
        st.success(f"✅ **{len(stock_prices.columns)} hisse**, **{len(stock_prices)} gün** ({actual_start} → {actual_end})")
        
        # Veri kaynağı detayları
        with st.expander("📁 Veri Kaynakları (Detay)", expanded=bool(synthetic_tickers)):
            if cache_tickers:
                st.info(f"💾 **Cache'den:** {', '.join(cache_tickers)}")
            if live_tickers:
                st.success(f"🌐 **Canlı (API):** {', '.join(live_tickers)}")
            if synthetic_tickers:
                st.error(f"""
                ⚠️ **SENTETİK VERİ (GERÇEK DEĞİL!):** {', '.join(synthetic_tickers)}
                
                Yahoo Finance API'den veri çekilemedi. Bu hisseler için **rastgele üretilmiş demo veri** kullanılıyor.
                
                **Nedenleri:**
                - Yahoo Finance geçici olarak erişilemez
                - Rate limit (çok fazla istek)
                - İnternet bağlantısı sorunu
                
                **Çözüm:** Biraz bekleyip tekrar dene veya farklı hisseler seç.
                """)
        
        # Eksik veri uyarısı
        missing = meta.get("missing_info", {})
        if missing and not force_complete:
            total_missing = sum(m.get("missing_days", 0) for m in missing.values())
            if total_missing > 0:
                st.warning(f"⚠️ Cache'de {total_missing} gün eksik. 'Güncel Veri Zorla' ile tamamlayabilirsin.")
        
        # JSON & Hash
        stock_json = stock_prices.to_json()
        bench_json = bench_prices.to_json() if bench_prices is not None else None
        prices_hash = hashlib.sha256((stock_json + "|" + (bench_json or "")).encode()).hexdigest()[:16]
        
        # 2. OPTİMİZASYON
        weights, vol, sharpe, success, expected_ret, cov_annual, shrinkage = cached_optimize(
            prices_hash, stock_json, strategy, max_weight, risk_free_rate
        )
        
        # 3. BACKTEST
        backtest_results = cached_backtest(
            prices_hash, stock_json, bench_json, train_window, hold_period, strategy, max_weight, risk_free_rate
        )
        
        # =====================
        # 🎯 ÖZET PANEL (EN ÖNEMLİ!)
        # =====================
        
        render_summary_panel(
            backtest_results, weights, stock_prices.columns.tolist(),
            strategy, actual_start, actual_end,
            synthetic_tickers=synthetic_tickers
        )
        
        st.divider()
        
        # =====================
        # DETAYLI ANALİZLER
        # =====================
        
        st.header("📊 Detaylı Analizler")
        
        # Tab'lar ile organize et
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Backtest", "⚠️ Risk", "🎯 Optimizasyon", "🔗 Korelasyon"])
        
        # TAB 1: BACKTEST
        with tab1:
            st.markdown("""
            **Bu grafik ne gösteriyor?** Farklı stratejilerin geçmişteki performansını karşılaştırır.
            
            Her çizgi, 1 birimlik yatırımın zaman içinde nasıl değiştiğini gösterir.
            Yukarı giden çizgi = kazanç, aşağı giden = kayıp.
            """)
            st.plotly_chart(render_backtest_chart(backtest_results, BENCHMARK_SYMBOL), use_container_width=True)
            
            st.markdown(render_metrics_explanation())
            
            show_net = st.checkbox("🔍 İşlem Maliyetlerini Dahil Et (Net Performans)", value=False)
            st.dataframe(render_metrics_table(backtest_results, BENCHMARK_SYMBOL, show_net=show_net), use_container_width=True)
            
            if show_net:
                costs = backtest_results["optimized"].get("transaction_costs", {})
                if costs:
                    st.info(f"💾 **İşlem Maliyeti Özeti (Optimize):** Toplam %{costs['total_cost']*100:.2f} maliyet, {costs['n_rebalances']} adet rebalance işlemi.")
            
            with st.expander("📉 Drawdown Analizi (En Kötü Dönemler)"):
                st.markdown("""
                **Drawdown nedir?** Portföyün zirvesinden ne kadar düştüğünü gösterir.
                
                Örnek: -%20 drawdown = En yüksek noktadan %20 düşüş yaşandı.
                Bu grafik, yatırımcının "en zor dönemlerde" ne kadar kayıp yaşayacağını gösterir.
                """)
                st.plotly_chart(render_drawdown_chart(backtest_results, calculate_drawdown, BENCHMARK_SYMBOL), use_container_width=True)
            
            with st.expander("🔥 Stres Testi (Kriz Dönemleri Analizi)"):
                st.markdown("""
                **Stres Testi nedir?** Portföyün geçmişteki büyük kriz dönemlerinde (COVID-19, 2022 Ayı Piyasası vb.) nasıl davrandığını ölçer.
                
                Bu analiz, portföyün "en zor zamanlarda" ne kadar dayanıklı olduğunu görmenizi sağlar.
                """)
                
                # Optimized getiri serisini al
                opt_returns_bt = pd.read_json(StringIO(backtest_results["optimized"]["daily_returns"]), typ="series")
                
                # Stres dönemlerini veri içinde bul
                stress_periods = identify_stress_periods_in_data(opt_returns_bt)
                
                # Her dönem için metrikleri hesapla
                stress_results = {}
                for name, info in stress_periods.items():
                    metrics = calculate_stress_period_metrics(
                        opt_returns_bt, 
                        info["actual_start"], 
                        info["actual_end"],
                        risk_free_rate=risk_free_rate
                    )
                    if metrics:
                        metrics["description"] = info["description"]
                        stress_results[name] = metrics
                
                render_stress_test_panel(stress_results)
        
        # TAB 2: RISK
        with tab2:
            st.markdown("""
            **Value at Risk (VaR)** günlük maksimum beklenen kaybı gösterir.
            
            Örnek: VaR = -%2.5 ve %95 güven → "100 günün 95'inde kayıp %2.5'i geçmez"
            """)
            
            opt_returns = pd.read_json(StringIO(backtest_results["optimized"]["daily_returns"]), typ="series")
            var_result = generate_var_analysis(opt_returns, var_confidence)
            cvar_value = calculate_expected_shortfall(opt_returns, var_confidence)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "VaR (Günlük)", 
                    f"{var_result.var_value*100:.2f}%",
                    help=f"Günlük maksimum beklenen kayıp (%{var_confidence*100:.0f} güven)"
                )
            with col2:
                st.metric(
                    "CVaR (ES)", 
                    f"{cvar_value*100:.2f}%",
                    help="VaR aşıldığında ortalama kayıp. Daha muhafazakar ölçü."
                )
            with col3:
                st.metric(
                    "İhlal Sayısı", 
                    f"{var_result.n_violations}/{len(opt_returns)}",
                    help=f"VaR'ın aşıldığı gün sayısı. Beklenen: {var_result.expected_violations:.0f}"
                )
            
            st.markdown("""
            **Grafik Açıklaması:**
            - 🔵 Mavi çizgi: Günlük portföy getirileri
            - 🔴 Kırmızı kesikli: VaR eşiği (maksimum beklenen kayıp)
            - ❌ Kırmızı X'ler: VaR'ın aşıldığı günler (ihlaller)
            """)
            st.plotly_chart(render_var_chart(opt_returns, var_result, cvar_value, var_confidence), use_container_width=True)
            
            with st.expander("📋 Detaylı Risk Raporu"):
                st.markdown("Farklı güven düzeylerinde VaR değerleri ve test sonuçları:")
                st.dataframe(generate_risk_report(opt_returns), use_container_width=True)
                st.info(interpret_var_result(var_result))
        
        # TAB 3: OPTİMİZASYON
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Portföy Ağırlıkları")
                st.markdown("*Her hisseye yatırılacak yüzde. Toplam %100.*")
                weight_df = pd.DataFrame({
                    "Hisse": stock_prices.columns,
                    "Ağırlık": [round(w, 4) for w in weights],
                    "Yüzde": [f"%{w*100:.1f}" for w in weights]
                })
                st.dataframe(weight_df, use_container_width=True)
                st.plotly_chart(render_weights_chart(weight_df), use_container_width=True)
            
            with col2:
                st.markdown("### Risk Katkısı Analizi")
                st.markdown("""
                *Her hissenin portföy riskine katkısı.*
                
                **RC/W Oranı:** Risk Katkısı / Ağırlık
                - **>1:** Hisse, ağırlığından fazla risk taşıyor ⚠️
                - **<1:** Hisse, ağırlığından az risk taşıyor ✓
                """)
                risk_df = get_risk_contribution_summary(weights, cov_annual, stock_prices.columns.tolist())
                st.dataframe(risk_df, use_container_width=True)
                st.plotly_chart(render_risk_contribution_chart(risk_df), use_container_width=True)
            
            with st.expander("📈 Etkin Sınır (Risk-Getiri Uzayı)"):
                st.markdown("""
                **Bu grafik ne gösteriyor?**
                
                - **Mavi çizgi:** Her risk seviyesinde elde edilebilecek maksimum getiri
                - **⭐ Kırmızı yıldız:** Seçilen optimal portföy
                - **💎 Yeşil elmas:** Eşit ağırlıklı portföy
                - **Açık mavi noktalar:** Bireysel hisseler
                
                İdeal portföy, çizginin üzerinde veya yakınında olmalı.
                """)
                st.plotly_chart(
                    render_efficient_frontier(
                        expected_ret, cov_annual, weights, vol,
                        stock_prices.columns.tolist(),
                        calculate_efficient_frontier, equal_weight_portfolio, max_weight
                    ),
                    use_container_width=True
                )
            
            with st.expander("🧪 Duyarlılık Analizi (Model Kararlılığı)"):
                st.markdown("""
                **Duyarlılık Analizi nedir?** "Max Ağırlık" parametresini değiştirdiğinizde portföyün ne kadar değiştiğini ölçer.
                
                Eğer küçük bir değişim çok büyük fark yaratıyorsa, model kararsız olabilir. Stabil modellerde eğri pürüzsüzdür.
                """)
                
                sens_report = generate_sensitivity_report(
                    expected_ret, cov_annual, 
                    stock_prices.columns.tolist(),
                    risk_free_rate=risk_free_rate
                )
                
                # Grafik
                st.plotly_chart(render_sensitivity_chart(sens_report["sensitivity_df"]), use_container_width=True)
                
                # Yorum
                st.info(f"💡 **Analiz Notu:** {sens_report['yorum']}")
                st.write(f"Sharpe oranı değişim aralığı: {sens_report['sharpe_range']:.3f}")
        
        # TAB 4: KORELASYON
        with tab4:
            st.markdown("""
            ### Korelasyon Matrisi
            
            **Bu ne gösteriyor?** Hisselerin birlikte nasıl hareket ettiğini gösterir.
            
            | Değer | Anlam | Çeşitlendirme |
            |-------|-------|---------------|
            | **+1 (koyu mavi)** | Aynı yönde hareket | ❌ Faydasız |
            | **0 (beyaz)** | Bağımsız hareket | ✓ İdeal |
            | **-1 (koyu kırmızı)** | Zıt yönde hareket | ✓✓ Mükemmel hedge |
            
            **İyi bir portföyde** hisseler arası korelasyon düşük olmalı.
            Yüksek korelasyonlu hisseler birlikte düşer, çeşitlendirme işe yaramaz.
            """)
            
            returns = calculate_log_returns(stock_prices)
            corr_matrix = calculate_correlation_matrix(returns)
            st.plotly_chart(render_correlation_heatmap(corr_matrix), use_container_width=True)
            
            # Ortalama korelasyon
            corr_values = corr_matrix.values
            avg_corr = np.mean(corr_values[np.triu_indices_from(corr_values, k=1)])
            
            if avg_corr > 0.7:
                st.warning(f"⚠️ Ortalama korelasyon yüksek ({avg_corr:.2f}). Çeşitlendirme sınırlı olabilir.")
            elif avg_corr > 0.5:
                st.info(f"ℹ️ Ortalama korelasyon: {avg_corr:.2f} (orta düzey)")
            else:
                st.success(f"✓ Ortalama korelasyon düşük ({avg_corr:.2f}). İyi çeşitlendirme!")
            
            # --- YENİ: ROLLING KORELASYON ANALİZİ ---
            st.divider()
            st.subheader("📈 Dinamik Korelasyon Analizi")
            st.markdown("""
            **Neden Önemli?** Korelasyonlar sabit değildir. Kriz dönemlerinde hisseler arasındaki korelasyon genellikle artar.
            Bu grafik, 63 günlük hareketli pencerelerle ortalama korelasyonun zaman içindeki değişimini gösterir.
            """)
            
            corr_report = generate_correlation_report(returns, window=63)
            
            # Grafik
            st.plotly_chart(render_rolling_correlation_chart(corr_report["rolling_corr_series"]), use_container_width=True)
            
            # Rapor Metrikleri
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Ortalama", f"{corr_report['ortalama_korelasyon']:.2f}")
            with col2:
                st.metric("Minimum", f"{corr_report['min_korelasyon']:.2f}")
            with col3:
                st.metric("Maksimum", f"{corr_report['max_korelasyon']:.2f}")
            with col4:
                st.metric("Yüksek Kor. Oranı", f"%{corr_report['yuksek_korelasyon_orani']:.0f}")
            
            st.info(f"💡 **Analiz Notu:** {corr_report['yorum']}")
            
            # --- YENİ: ROLLING VOLATİLİTE ---
            st.divider()
            st.subheader("📈 Dinamik Volatilite (Risk) Analizi")
            st.markdown("""
            **Bu grafik ne gösteriyor?** Hisse senetlerinin risk seviyelerinin (volatilite) zaman içindeki değişimini gösterir.
            Yukarı giden çizgiler riskin arttığını, aşağı gidenler ise piyasanın sakinleştiğini gösterir.
            """)
            
            rolling_vol = calculate_rolling_volatility(returns, window=21)
            st.plotly_chart(render_rolling_volatility_chart(rolling_vol), use_container_width=True)
        
        # =====================
        # EXPORT
        # =====================
        
        with st.expander("💾 Sonuçları İndir"):
            col1, col2 = st.columns(2)
            with col1:
                csv = weight_df.to_csv(index=False)
                st.download_button("📥 Ağırlıklar (CSV)", csv, "weights.csv", "text/csv")
            with col2:
                export = {
                    "parametreler": {"tickers": tickers_input, "strategy": strategy, "max_weight": max_weight},
                    "agirliklar": dict(zip(stock_prices.columns.tolist(), [float(w) for w in weights])),
                    "performans": backtest_results["optimized"]["metrics"]
                }
                st.download_button("📥 Sonuçlar (JSON)", json.dumps(export, indent=2), "results.json", "application/json")
        
        st.success("✅ Analiz tamamlandı!")
        
    except Exception as e:
        st.error(f"Hata: {e}")
        st.exception(e)

else:
    st.info("👈 Parametreleri ayarla ve **Analizi Başlat** butonuna tıkla.")
    
    st.markdown("""
    ### Hızlı Başlangıç
    1. **Hisseler:** ABD hisse sembolleri (örn: AAPL, MSFT)
    2. **Tarih:** En az 1 yıl veri önerilir
    3. **Strateji:** Max Sharpe veya Min Varyans
    4. **Analizi Başlat** 🚀
    """)
