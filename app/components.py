"""
Dashboard UI Bileşenleri
========================
Bu dosya Streamlit dashboard için yeniden kullanılabilir UI bileşenlerini içerir.
main.py dosyasını temiz tutmak için buraya ayrıldı.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO


def render_summary_panel(
    backtest_results: dict,
    weights: np.ndarray,
    stock_columns: list,
    strategy: str,
    actual_start: str,
    actual_end: str,
    initial_investment: float = 10000,
    synthetic_tickers: list = None
):
    """
    🎯 ÖZET SONUÇ PANELİ - En önemli kısım!
    Kullanıcıya "Ee sonuç ne?" sorusunun cevabını verir.
    """
    st.header("🎯 SONUÇ: Ne Yapmalısın?")
    
    # Sentetik veri uyarısı - en üstte göster!
    if synthetic_tickers:
        st.error(f"""
        ⚠️ **DİKKAT: DEMO VERİ KULLANILIYOR!**
        
        **{', '.join(synthetic_tickers)}** için gerçek veri çekilemedi. Bu hisseler sentetik (rastgele üretilmiş) veri içeriyor.
        
        **Bu sonuçlar gerçeği yansıtmaz!** Sadece sistemin nasıl çalıştığını göstermek içindir.
        """)
    
    # Metrikleri al
    opt_metrics = backtest_results["optimized"]["metrics"]
    eq_metrics = backtest_results["equal_weight"]["metrics"]
    bench_metrics = backtest_results.get("benchmark", {}).get("metrics", {})
    rp_metrics = backtest_results.get("risk_parity", {}).get("metrics", {})
    
    # Getiriler
    opt_return = opt_metrics.get("toplam_getiri", 0)
    eq_return = eq_metrics.get("toplam_getiri", 0)
    bench_return = bench_metrics.get("toplam_getiri", 0) if bench_metrics else 0
    rp_return = rp_metrics.get("toplam_getiri", 0) if rp_metrics else 0
    
    # Simülasyon hesapla
    opt_final = initial_investment * (1 + opt_return)
    eq_final = initial_investment * (1 + eq_return)
    bench_final = initial_investment * (1 + bench_return)
    rp_final = initial_investment * (1 + rp_return)
    
    # 💰 BÜYÜK ÖZET KARTLARI
    st.markdown("### 💰 Eğer Geçmişte Yatırım Yapsaydın...")
    st.caption(f"*{actual_start} - {actual_end} arasında {initial_investment:,.0f}₺ yatırsaydın:*")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        profit = opt_final - initial_investment
        st.metric(
            "🎯 Optimize Portföy",
            f"{opt_final:,.0f}₺",
            delta=f"{profit:+,.0f}₺ ({opt_return*100:+.1f}%)",
            delta_color="normal" if profit >= 0 else "inverse"
        )
    
    with col2:
        profit = eq_final - initial_investment
        st.metric(
            "⚖️ Eşit Ağırlık",
            f"{eq_final:,.0f}₺",
            delta=f"{profit:+,.0f}₺ ({eq_return*100:+.1f}%)",
            delta_color="normal" if profit >= 0 else "inverse"
        )
    
    with col3:
        profit = rp_final - initial_investment
        st.metric(
            "🔄 Risk Parity",
            f"{rp_final:,.0f}₺",
            delta=f"{profit:+,.0f}₺ ({rp_return*100:+.1f}%)",
            delta_color="normal" if profit >= 0 else "inverse"
        )
    
    with col4:
        if bench_return:
            profit = bench_final - initial_investment
            st.metric(
                "📈 Piyasa (SPY)",
                f"{bench_final:,.0f}₺",
                delta=f"{profit:+,.0f}₺ ({bench_return*100:+.1f}%)",
                delta_color="normal" if profit >= 0 else "inverse"
            )
        else:
            st.metric("📈 Piyasa (SPY)", "Veri yok")
    
    # 📦 HER STRATEJİNİN İÇERİĞİ
    st.markdown("---")
    st.markdown("### 📦 Stratejilerin Portföy Dağılımları")
    st.caption("*Her strateji parayı nasıl dağıtıyor?*")
    
    # Ağırlıkları hazırla
    rounded_weights = [round(w, 4) for w in weights]
    n_stocks = len(stock_columns)
    eq_weights_list = [round(1.0/n_stocks, 4)] * n_stocks  # Eşit ağırlık
    
    # Risk Parity ağırlıklarını backtest sonuçlarından çıkaramıyoruz, 
    # ama açıklama verebiliriz
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**🎯 Optimize Portföy**")
        # En yüksek ağırlıklı 3 hisseyi göster
        sorted_idx = sorted(range(len(rounded_weights)), key=lambda i: rounded_weights[i], reverse=True)
        for i in sorted_idx[:3]:
            if rounded_weights[i] >= 0.01:
                st.write(f"• {stock_columns[i]}: %{rounded_weights[i]*100:.0f}")
        others = sum(1 for w in rounded_weights if w < 0.01)
        if others > 0:
            st.caption(f"*({others} hisse portföye dahil edilmedi)*")
    
    with col2:
        st.markdown("**⚖️ Eşit Ağırlık**")
        eq_pct = 100 / n_stocks
        for ticker in stock_columns[:3]:
            st.write(f"• {ticker}: %{eq_pct:.0f}")
        if n_stocks > 3:
            st.caption(f"*... ve {n_stocks - 3} hisse daha (hepsi %{eq_pct:.0f})*")
    
    with col3:
        st.markdown("**🔄 Risk Parity**")
        st.caption("Her hissenin **eşit risk katkısı** yapması hedeflenir.")
        st.write("• Volatil hisseler → Düşük ağırlık")
        st.write("• Stabil hisseler → Yüksek ağırlık")
    
    with col4:
        st.markdown("**📈 Piyasa (SPY)**")
        st.caption("S&P 500 ETF - 500 büyük ABD şirketi")
        st.write("• Tek varlık: SPY")
        st.write("• Pasif yatırım")
        st.write("• Benchmark olarak kullanılır")
    
    # 🏆 KAZANAN STRATEJİ - Dürüst Değerlendirme
    st.markdown("---")
    
    # Risk metrikleri
    opt_sharpe = opt_metrics.get("sharpe_orani", 0)
    eq_sharpe = eq_metrics.get("sharpe_orani", 0)
    bench_sharpe = bench_metrics.get("sharpe_orani", 0) if bench_metrics else 0
    rp_sharpe = rp_metrics.get("sharpe_orani", 0) if rp_metrics else 0
    
    opt_dd = abs(opt_metrics.get("max_drawdown", 0))
    eq_dd = abs(eq_metrics.get("max_drawdown", 0))
    bench_dd = abs(bench_metrics.get("max_drawdown", 0)) if bench_metrics else 0
    rp_dd = abs(rp_metrics.get("max_drawdown", 0)) if rp_metrics else 0
    
    opt_vol = opt_metrics.get("yillik_volatilite", 0)
    eq_vol = eq_metrics.get("yillik_volatilite", 0)
    
    strategies = {
        "Optimize Portföy": opt_return,
        "Eşit Ağırlık": eq_return,
        "Risk Parity": rp_return,
        "Piyasa (SPY)": bench_return
    }
    best_strategy = max(strategies, key=strategies.get)
    best_return = strategies[best_strategy]
    
    beat_market = opt_return > bench_return if bench_return else True
    beat_equal = opt_return > eq_return
    
    # Sharpe bazlı en iyi
    sharpe_dict = {
        "Optimize Portföy": opt_sharpe,
        "Eşit Ağırlık": eq_sharpe,
        "Risk Parity": rp_sharpe,
        "Piyasa (SPY)": bench_sharpe
    }
    best_risk_adjusted = max(sharpe_dict, key=sharpe_dict.get)
    
    # Drawdown bazlı en güvenli
    dd_dict = {
        "Optimize Portföy": opt_dd,
        "Eşit Ağırlık": eq_dd,
        "Risk Parity": rp_dd,
        "Piyasa (SPY)": bench_dd
    }
    safest = min(dd_dict, key=dd_dict.get)
    
    # ===== DÜRÜST DEĞERLENDİRME =====
    st.markdown("### 📊 Strateji Karşılaştırması")
    
    # Karşılaştırma tablosu
    comparison_data = {
        "Strateji": ["🎯 Optimize", "⚖️ Eşit Ağırlık", "🔄 Risk Parity", "📈 Piyasa (SPY)"],
        "Getiri": [f"%{opt_return*100:.1f}", f"%{eq_return*100:.1f}", f"%{rp_return*100:.1f}", f"%{bench_return*100:.1f}"],
        "Sharpe": [f"{opt_sharpe:.2f}", f"{eq_sharpe:.2f}", f"{rp_sharpe:.2f}", f"{bench_sharpe:.2f}"],
        "Max Düşüş": [f"-%{opt_dd*100:.1f}", f"-%{eq_dd*100:.1f}", f"-%{rp_dd*100:.1f}", f"-%{bench_dd*100:.1f}"],
        "Volatilite": [f"%{opt_vol*100:.1f}", f"%{eq_vol*100:.1f}", f"%{rp_metrics.get('yillik_volatilite', 0)*100:.1f}", f"%{bench_metrics.get('yillik_volatilite', 0)*100:.1f}" if bench_metrics else "-"]
    }
    st.dataframe(pd.DataFrame(comparison_data).set_index("Strateji"), use_container_width=True)
    
    # ===== KAZANANLAR =====
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💰 En Çok Kazandıran", best_strategy, f"%{best_return*100:.1f}")
    
    with col2:
        st.metric("📈 En İyi Risk/Getiri", best_risk_adjusted, f"Sharpe: {sharpe_dict[best_risk_adjusted]:.2f}")
    
    with col3:
        st.metric("🛡️ En Güvenli", safest, f"Max DD: -%{dd_dict[safest]*100:.1f}")
    
    # ===== DÜRÜST YORUM =====
    st.markdown("---")
    st.markdown("### 💡 Dürüst Değerlendirme")
    
    # Optimize portföy kazandı mı?
    if best_strategy == "Optimize Portföy" and best_risk_adjusted == "Optimize Portföy":
        st.success("""
        ✅ **Optimizasyon işe yaradı!** Hem en yüksek getiri hem en iyi risk-ayarlı performans.
        
        Bu dönemde matematiksel optimizasyon, basit stratejilerden daha iyi sonuç verdi.
        """)
    
    elif best_strategy != "Optimize Portföy" and best_risk_adjusted == "Optimize Portföy":
        st.info(f"""
        🎯 **{best_strategy} daha çok kazandırdı ANCAK...**
        
        Optimize portföy **risk-ayarlı bazda** (Sharpe) daha iyi! Yani:
        - Aynı risk için daha fazla getiri, veya
        - Aynı getiri için daha az risk
        
        **Neden önemli?** Yüksek getiri her zaman iyi değil - ne kadar risk aldığın önemli.
        Sharpe oranı = (Getiri - Risksiz Faiz) / Risk
        
        **Optimize:** Sharpe {opt_sharpe:.2f}, Volatilite %{opt_vol*100:.1f}
        **{best_strategy}:** Sharpe {sharpe_dict[best_strategy]:.2f}
        """)
    
    elif best_strategy != "Optimize Portföy" and safest == "Optimize Portföy":
        st.info(f"""
        🛡️ **{best_strategy} daha çok kazandırdı ANCAK...**
        
        Optimize portföy **en güvenli** seçenek! Kriz dönemlerinde daha az düştü:
        - Optimize Max Drawdown: **-%{opt_dd*100:.1f}**
        - {best_strategy} Max Drawdown: **-%{dd_dict[best_strategy]*100:.1f}**
        
        **Neden önemli?** -%40 düşen portföyü %67 artışla telafi etmen lazım.
        Daha az düşmek, uzun vadede daha değerli olabilir.
        """)
    
    elif best_strategy != "Optimize Portföy":
        # Optimize portföy hiçbir kategoride kazanmadı - dürüst ol!
        st.warning(f"""
        ⚠️ **Bu dönemde optimizasyon pek işe yaramadı!**
        
        **{best_strategy}** hem daha çok kazandırdı hem de risk metrikleri benzer veya daha iyi.
        
        **Neden oldu?**
        - Mean-variance optimizasyonu **geçmiş veriye** dayanır
        - Geçmişteki en iyi hisseler gelecekte de en iyi olmayabilir (mean reversion)
        - Seçilen hisse evreni çok homojen olabilir (hepsi teknoloji gibi)
        - Bu dönemde piyasa koşulları basit stratejileri desteklemiş olabilir
        
        **Bu normal mi?** EVET! Akademik çalışmalar gösteriyor ki:
        - %30-50 dönemde basit stratejiler optimizasyonu yener
        - Ledoit-Wolf shrinkage bunu azaltır ama tamamen önlemez
        - Önemli olan **uzun vadeli** ve **farklı piyasa koşullarında** test
        
        **Ne yapmalısın?**
        1. Daha uzun dönem test et (en az 5-10 yıl)
        2. Farklı hisse evrenleri dene (sektör çeşitliliği)
        3. Kriz dönemlerini içeren tarih aralığı seç (2020, 2022)
        """)
    
    # Risk parity öne çıktı mı?
    if best_strategy == "Risk Parity" or best_risk_adjusted == "Risk Parity":
        st.info("""
        💡 **Risk Parity dikkat çekici!** Bu strateji:
        - Her hisseye "eşit risk" verir (eşit para değil)
        - Volatil hisselere daha az, stabil hisselere daha çok yatırır
        - Genelde düşüşlerde daha iyi korur
        """)
    
    # 🔍 NEDEN BU DAĞILIM? AÇIKLAMA (Expander içinde)
    weight_df = pd.DataFrame({
        "Hisse": stock_columns,
        "Ağırlık": rounded_weights,
    }).sort_values("Ağırlık", ascending=False)
    zero_weights = weight_df[weight_df["Ağırlık"] < 0.01]
    
    with st.expander("🔍 Optimize Portföy Neden Bu Şekilde Dağıtıldı?"):
        strategy_name = "Max Sharpe" if strategy == "max_sharpe" else "Min Varyans"
        
        st.markdown(f"""
        **Kullanılan Strateji: {strategy_name}**
        
        {"**Max Sharpe** stratejisi, risk başına en yüksek getiriyi hedefler. Sistem, geçmiş verilerden her hissenin beklenen getirisini ve riskini hesapladı. Yüksek getiri/risk oranına sahip hisseler daha fazla ağırlık aldı." if strategy == "max_sharpe" else "**Min Varyans** stratejisi, portföy riskini (dalgalanmayı) minimize etmeyi hedefler. Düşük volatiliteli ve birbirleriyle düşük korelasyonlu hisseler tercih edildi."}
        """)
        
        # En çok alınan hisse açıklaması
        top_stock = weight_df.iloc[0]
        st.success(f"""
        🏆 **{top_stock['Hisse']}** en yüksek ağırlığı aldı (%{top_stock['Ağırlık']*100:.0f})
        
        Çünkü: {"Risk-getiri dengesi (Sharpe oranı) diğerlerinden daha iyi." if strategy == "max_sharpe" else "Volatilitesi düşük ve/veya diğer hisselerle korelasyonu düşük."}
        """)
        
        # Hiç alınmayan hisseler açıklaması
        if len(zero_weights) > 0:
            excluded = ", ".join(zero_weights["Hisse"].tolist())
            st.warning(f"""
            ⚠️ **Portföye dahil edilmeyen hisseler:** {excluded}
            
            **Neden alınmadı?**
            {"Bu hisselerin risk-getiri oranı (Sharpe) diğerlerinden düşük. Yani aynı getiri için daha fazla risk taşıyorlar veya aynı risk için daha az getiri sağlıyorlar." if strategy == "max_sharpe" else "Bu hisseler yüksek volatiliteye sahip veya portföydeki diğer hisselerle yüksek korelasyonlu. Dahil edilseler portföy riski artardı."}
            """)
    
    # Risk uyarısı
    max_dd = abs(opt_metrics.get('max_drawdown', 0)) * 100
    if max_dd > 20:
        st.error(f"""
        ⚠️ **Risk Uyarısı:** Bu portföy geçmişte **%{max_dd:.0f} düşüş** yaşadı!
        
        Yani {initial_investment:,.0f}₺ yatırım yapıldığında, bir noktada {initial_investment * max_dd / 100:,.0f}₺ 
        geçici kayıp yaşanabilir. Bu risk toleransınıza uygun mu değerlendirin.
        """)
    elif max_dd > 10:
        st.warning(f"⚠️ **Dikkat:** Geçmişte %{max_dd:.0f} düşüş yaşandı. Kısa vadeli dalgalanmalar beklenebilir.")
    
    # 📖 Terimler Sözlüğü
    with st.expander("📖 Terimler Ne Anlama Geliyor?"):
        st.markdown(f"""
        ### Terimler Sözlüğü
        
        | Terim | Açıklama |
        |-------|----------|
        | **Sharpe Oranı** | Risk başına getiri. (Getiri - Risksiz Faiz) / Risk. **1+ iyi**, **2+ çok iyi**. |
        | **Volatilite** | Fiyat dalgalanması (yıllık %). Yüksek = riskli. |
        | **Max Drawdown** | Zirveden en dip noktaya düşüş. -%20'den fazlası dikkat gerektirir. |
        | **VaR (Value at Risk)** | Belirli güven düzeyinde günlük maksimum kayıp tahmini. |
        | **CVaR (Expected Shortfall)** | VaR aşıldığında ortalama kayıp. Daha muhafazakar risk ölçüsü. |
        | **Korelasyon** | İki hissenin birlikte hareket etme eğilimi (-1 ile +1 arası). |
        | **Risk Katkısı** | Her hissenin portföy riskine katkısı (%). |
        | **RC/W Oranı** | Risk Katkısı / Ağırlık. >1 ise hisse ağırlığından fazla risk taşıyor. |
        | **Risk Parity** | Her hissenin eşit risk katkısı yapmasını hedefleyen strateji. |
        | **Mean-Variance** | Markowitz'in getiri-varyans optimizasyonu (bu sistemin temeli). |
        
        ---
        
        ### Optimizasyon Neden Her Zaman Kazanamaz?
        
        Mean-variance optimizasyonu **geçmiş veriye** dayanır. Problemler:
        
        1. **Tahmin hatası:** Geçmiş getiriler geleceği tahmin etmez
        2. **Aşırı uyum (overfitting):** Geçmişe çok iyi uyan portföy gelecekte kötü olabilir
        3. **Parametre duyarlılığı:** Küçük değişiklikler büyük ağırlık farkları yaratır
        
        **Ledoit-Wolf shrinkage** bu sorunları azaltır ama tamamen çözmez.
        
        ---
        
        ### Bu Portföyün Risk Profili
        
        - **Yıllık Volatilite:** %{opt_vol*100:.1f} — Portföy değeri yılda bu kadar dalgalanabilir
        - **Max Drawdown:** %{max_dd:.1f} — En kötü dönemde bu kadar düştü
        - **Sharpe Oranı:** {opt_sharpe:.2f} — {"İyi risk-getiri dengesi ✓" if opt_sharpe >= 1 else "Orta düzey" if opt_sharpe >= 0.5 else "Düşük ⚠️"}
        """)


def render_backtest_chart(backtest_results: dict, benchmark_symbol: str = "SPY"):
    """Equity curve ve performans grafiği."""
    colors = {"optimized": "blue", "equal_weight": "green", "risk_parity": "orange", "benchmark": "gray"}
    names = {"optimized": "Optimize", "equal_weight": "Eşit Ağırlık", "risk_parity": "Risk Parity", "benchmark": f"Benchmark ({benchmark_symbol})"}
    
    fig = go.Figure()
    
    for strat_name, data in backtest_results.items():
        equity = pd.read_json(StringIO(data["equity_curve"]), typ="series")
        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            name=names.get(strat_name, strat_name),
            line=dict(color=colors.get(strat_name, "purple"))
        ))
    
    fig.update_layout(
        title="Portföy Değeri (1 birim başlangıç)",
        xaxis_title="Tarih",
        yaxis_title="Değer",
        hovermode="x unified"
    )
    
    return fig


def render_drawdown_chart(backtest_results: dict, calculate_drawdown_func, benchmark_symbol: str = "SPY"):
    """Drawdown grafiği."""
    colors = {"optimized": "blue", "equal_weight": "green", "risk_parity": "orange", "benchmark": "gray"}
    names = {"optimized": "Optimize", "equal_weight": "Eşit Ağırlık", "risk_parity": "Risk Parity", "benchmark": f"Benchmark ({benchmark_symbol})"}
    
    fig = go.Figure()
    
    for strat_name, data in backtest_results.items():
        equity = pd.read_json(StringIO(data["equity_curve"]), typ="series")
        dd = calculate_drawdown_func(equity)
        fig.add_trace(go.Scatter(
            x=dd.index,
            y=dd.values * 100,
            mode="lines",
            name=names.get(strat_name, strat_name),
            fill="tozeroy",
            line=dict(color=colors.get(strat_name, "purple"))
        ))
    
    fig.update_layout(
        title="Drawdown (Zirveden Düşüş)",
        xaxis_title="Tarih",
        yaxis_title="Drawdown (%)",
        hovermode="x unified"
    )
    
    return fig


def render_weights_chart(weight_df: pd.DataFrame):
    """Ağırlık dağılımı bar chart."""
    fig = px.bar(
        weight_df,
        x="Hisse",
        y="Ağırlık",
        color="Ağırlık",
        color_continuous_scale="Blues",
        title="Portföy Ağırlık Dağılımı"
    )
    fig.update_layout(showlegend=False)
    return fig


def render_risk_contribution_chart(risk_contrib_df: pd.DataFrame):
    """Risk katkısı pasta grafiği."""
    fig = px.pie(
        risk_contrib_df,
        values="Risk Katkısı",
        names="Varlık",
        title="Risk Dağılımı",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    return fig


def render_var_chart(opt_returns: pd.Series, var_result, cvar_value: float, confidence: float):
    """VaR ihlal grafiği."""
    fig = go.Figure()
    
    # Getiriler
    fig.add_trace(go.Scatter(
        x=opt_returns.index,
        y=opt_returns.values * 100,
        mode="lines",
        name="Günlük Getiri",
        line=dict(color="blue", width=1)
    ))
    
    # VaR çizgisi
    fig.add_hline(
        y=var_result.var_value * 100,
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR: {var_result.var_value*100:.2f}%"
    )
    
    # CVaR çizgisi
    fig.add_hline(
        y=cvar_value * 100,
        line_dash="dot",
        line_color="darkred",
        annotation_text=f"CVaR: {cvar_value*100:.2f}%"
    )
    
    # İhlaller
    violation_dates = opt_returns[var_result.violations].index
    violation_values = opt_returns[var_result.violations].values * 100
    
    fig.add_trace(go.Scatter(
        x=violation_dates,
        y=violation_values,
        mode="markers",
        name="İhlal",
        marker=dict(size=8, color="red", symbol="x")
    ))
    
    fig.update_layout(
        title=f"VaR İhlal Analizi (%{confidence*100:.0f})",
        xaxis_title="Tarih",
        yaxis_title="Getiri (%)",
        hovermode="x unified"
    )
    
    return fig


def render_efficient_frontier(
    expected_ret: np.ndarray,
    cov_annual: np.ndarray,
    weights: np.ndarray,
    vol: float,
    stock_columns: list,
    calculate_frontier_func,
    equal_weight_func,
    max_weight: float
):
    """Etkin sınır grafiği."""
    vols, rets, _ = calculate_frontier_func(expected_ret, cov_annual, n_points=30, max_weight=max_weight)
    
    fig = go.Figure()
    
    # Etkin sınır
    fig.add_trace(go.Scatter(
        x=vols, y=rets,
        mode="lines",
        name="Etkin Sınır",
        line=dict(color="blue", width=2)
    ))
    
    # Bireysel hisseler
    individual_vols = np.sqrt(np.diag(cov_annual))
    fig.add_trace(go.Scatter(
        x=individual_vols, y=expected_ret,
        mode="markers+text",
        name="Hisseler",
        text=stock_columns,
        textposition="top center",
        marker=dict(size=10, color="lightblue")
    ))
    
    # Optimal portföy
    port_ret = np.dot(weights, expected_ret)
    fig.add_trace(go.Scatter(
        x=[vol], y=[port_ret],
        mode="markers",
        name="Optimal",
        marker=dict(size=15, color="red", symbol="star")
    ))
    
    # Eşit ağırlık
    eq_weights = equal_weight_func(len(weights))
    eq_vol = np.sqrt(eq_weights @ cov_annual @ eq_weights)
    eq_ret = np.dot(eq_weights, expected_ret)
    fig.add_trace(go.Scatter(
        x=[eq_vol], y=[eq_ret],
        mode="markers",
        name="Eşit Ağırlık",
        marker=dict(size=12, color="green", symbol="diamond")
    ))
    
    fig.update_layout(
        title="Risk-Getiri Uzayı",
        xaxis_title="Volatilite",
        yaxis_title="Beklenen Getiri"
    )
    
    return fig


def render_correlation_heatmap(corr_matrix: pd.DataFrame):
    """Korelasyon matrisi ısı haritası."""
    fig = px.imshow(
        corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        zmin=-1,
        zmax=1,
        text_auto=".2f"
    )
    fig.update_layout(title="Korelasyon Matrisi")
    return fig


def render_metrics_table(backtest_results: dict, benchmark_symbol: str = "SPY"):
    """Performans metrikleri tablosu."""
    names = {"optimized": "Optimize", "equal_weight": "Eşit Ağırlık", "risk_parity": "Risk Parity", "benchmark": f"Benchmark ({benchmark_symbol})"}
    
    rows = []
    for strat_name, data in backtest_results.items():
        m = data["metrics"].copy()
        m["strateji"] = names.get(strat_name, strat_name)
        rows.append(m)
    
    df = pd.DataFrame(rows).set_index("strateji")
    
    # Formatla
    display_df = df.copy()
    display_df["toplam_getiri"] = display_df["toplam_getiri"].apply(lambda x: f"{x*100:.2f}%")
    display_df["yillik_getiri"] = display_df["yillik_getiri"].apply(lambda x: f"{x*100:.2f}%")
    display_df["yillik_volatilite"] = display_df["yillik_volatilite"].apply(lambda x: f"{x*100:.2f}%")
    display_df["sharpe_orani"] = display_df["sharpe_orani"].apply(lambda x: f"{x:.3f}")
    display_df["max_drawdown"] = display_df["max_drawdown"].apply(lambda x: f"{x*100:.2f}%")
    
    display_df.columns = ["Toplam", "Yıllık", "Volatilite", "Sharpe", "MaxDD", "Gün"]
    
    return display_df


def render_metrics_explanation():
    """Metrikler için açıklama metni."""
    return """
    **Tablo Kolonları:**
    - **Toplam:** Tüm dönem boyunca toplam getiri
    - **Yıllık:** Yıllıklaştırılmış getiri (bileşik)
    - **Volatilite:** Yıllık risk (standart sapma). Düşük = daha stabil
    - **Sharpe:** Risk başına getiri. **1+ iyi**, **2+ çok iyi**
    - **MaxDD:** En kötü dönemdeki düşüş. **-%20'den fazlası dikkat!**
    - **Gün:** Toplam işlem günü sayısı
    """
