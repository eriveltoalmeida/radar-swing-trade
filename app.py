import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime

# ==========================================
# CONFIGURAÇÃO DE TELA MÓVEL
# ==========================================
st.set_page_config(
    page_title="Radar B3",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# PARÂMETROS GERAIS
# ==========================================
CAPITAL_TOTAL = 20000.00
RISCO_MAX_TRADE = 0.01
ALOCACAO_MAX_POSICAO = 0.20
MIN_VOL_FINANCEIRO_DIA = 5_000_000
ADX_MIN = 20
JANELA_FORCA_RELATIVA = 21

TICKERS = sorted(list(set([
    "ABEV3.SA", "ALOS3.SA", "AZZA3.SA", "B3SA3.SA", "BBAS3.SA", "BBDC4.SA", "BBSE3.SA", "BPAC11.SA",
    "BRAP4.SA", "BRAV3.SA", "CMIG4.SA", "CPFE3.SA", "CSAN3.SA", "CYRE3.SA", "EGIE3.SA", "EQTL3.SA",
    "GGBR4.SA", "ITSA4.SA", "ITUB4.SA", "KLBN11.SA", "LREN3.SA", "MGLU3.SA", "PETR4.SA", "PRIO3.SA",
    "RADL3.SA", "RAIL3.SA", "RENT3.SA", "SANB11.SA", "SBSP3.SA", "SUZB3.SA", "TAEE11.SA", "TOTS3.SA",
    "UGPA3.SA", "VALE3.SA", "VIVT3.SA", "WEGE3.SA"
])))

# ==========================================
# FUNÇÕES DE CÁLCULO QUANTITATIVO
# ==========================================
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # ATR & ADX
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift()).abs()
    tr3 = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / df["ATR"])
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / df["ATR"])
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df["ADX"] = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    df["VOL_SMA_20"] = df["Volume"].rolling(window=20).mean()
    df["MAX_20"] = df["High"].shift(1).rolling(window=20).max()
    return df

@st.cache_data(ttl=3600)  # Guarda os dados em cache por 1 hora para economizar tráfego no celular
def executar_radar():
    # Contexto Ibov
    ibov = yf.download("^BVSP", period="1y", interval="1d", progress=False)
    if isinstance(ibov.columns, pd.MultiIndex):
        close_ibov = ibov["Close"].iloc[:, 0].dropna()
    else:
        close_ibov = ibov["Close"].dropna()
    
    preco_ibov = float(close_ibov.iloc[-1])
    sma50_ibov = float(close_ibov.rolling(50).mean().iloc[-1])
    retorno_ibov_21d = ((preco_ibov - float(close_ibov.iloc[-(JANELA_FORCA_RELATIVA+1)])) / float(close_ibov.iloc[-(JANELA_FORCA_RELATIVA+1)])) * 100
    regime_ibov = "ALTA" if preco_ibov > sma50_ibov else "BAIXA"

    setups, radar = [], []
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.dropna()
            if len(df) < 150: continue
            df = calcular_indicadores(df)
            
            ult = df.iloc[-1]
            ant = df.iloc[-2]
            preco = float(ult["Close"])
            atr = float(ult["ATR"])
            vol_fin = preco * float(ult["VOL_SMA_20"])
            ret_ativo = ((preco - float(df["Close"].iloc[-(JANELA_FORCA_RELATIVA+1)])) / float(df["Close"].iloc[-(JANELA_FORCA_RELATIVA+1)])) * 100
            rs_ibov = round(ret_ativo - retorno_ibov_21d, 2)

            tend_alta = preco > float(ult["SMA_200"]) and float(ult["EMA_21"]) > float(ult["SMA_50"])
            dist_mme21 = abs(float(ult["Low"]) - float(ult["EMA_21"])) / float(ult["EMA_21"])
            
            setup_nome = None
            gatilho, stop = None, None

            if vol_fin >= MIN_VOL_FINANCEIRO_DIA and tend_alta and float(ult["ADX"]) > ADX_MIN:
                # Setup Pullback
                if dist_mme21 <= 0.025 and preco > float(ult["EMA_21"]):
                    setup_nome = "Pullback MME21"
                    gatilho = round(float(ult["High"]) + (0.10 * atr), 2)
                    stop = round(min(float(ult["Low"]), float(ant["Low"])) - (1.0 * atr), 2)
                # Setup Breakout
                elif float(ult["High"]) >= float(ult["MAX_20"]) and float(ult["Volume"]) > float(ult["VOL_SMA_20"]):
                    setup_nome = "Breakout 20d"
                    gatilho = round(float(ult["High"]) + (0.10 * atr), 2)
                    stop = round(min(float(ult["Low"]), float(ant["Low"])) - (1.0 * atr), 2)

            # Se armou setup dentro do risco
            if setup_nome and gatilho and stop:
                risco_acao = gatilho - stop
                risco_pct = (risco_acao / gatilho) * 100
                if 0 < risco_pct <= 8.0:
                    qtd = min(int((CAPITAL_TOTAL * RISCO_MAX_TRADE) // risco_acao), int((CAPITAL_TOTAL * ALOCACAO_MAX_POSICAO) // gatilho))
                    if qtd > 0:
                        setups.append({
                            "Ticker": ticker.replace(".SA", ""),
                            "Setup": setup_nome,
                            "Gatilho": gatilho,
                            "Stop Loss": stop,
                            "Alvo (2:1)": round(gatilho + (2 * risco_acao), 2),
                            "Risco (%)": round(risco_pct, 1),
                            "Qtd": qtd,
                            "Total (R$)": round(qtd * gatilho, 2)
                        })

            radar.append({
                "Ticker": ticker.replace(".SA", ""),
                "Preço": round(preco, 2),
                "RSI": round(float(ult["RSI"]), 1),
                "ADX": round(float(ult["ADX"]), 1),
                "RS vs IBOV (%)": rs_ibov,
                "Tendência": "Alta" if tend_alta else "Neutra/Baixa"
            })
        except Exception:
            continue

    return regime_ibov, preco_ibov, pd.DataFrame(setups), pd.DataFrame(radar)

# ==========================================
# INTERFACE DO APP (STREAMLIT)
# ==========================================
st.title("📱 Radar Swing Trade B3")

with st.spinner("Atualizando cotações..."):
    regime_ibov, preco_ibov, df_setups, df_radar = executar_radar()

# Cartões de Métricas no topo da tela do celular
col1, col2 = st.columns(2)
with col1:
    cor_regime = "green" if regime_ibov == "ALTA" else "red"
    st.metric("Regime Ibov", f"{regime_ibov}", delta=f"{preco_ibov:,.0f} pts")
with col2:
    st.metric("Capital Alocado", f"R$ {CAPITAL_TOTAL:,.2f}")

st.write("---")

# Abas pensadas para navegação rápida pelo polegar
tab_setups, tab_radar = st.tabs(["🎯 Setups Armados", "📊 Radar de Ações"])

with tab_setups:
    st.subheader("Gatilhos para o Pregão")
    if df_setups.empty:
        st.info("Nenhum setup qualificado com risco controlado hoje.")
    else:
        # Exibe em cartões individuais mais amigáveis para celular do que tabelas largas
        for _, s in df_setups.iterrows():
            with st.container():
                st.markdown(f"### **{s['Ticker']}** · `{s['Setup']}`")
                c1, c2, c3 = st.columns(3)
                c1.metric("Gatilho", f"R$ {s['Gatilho']:.2f}")
                c2.metric("Stop", f"R$ {s['Stop Loss']:.2f}")
                c3.metric("Alvo", f"R$ {s['Alvo (2:1)']:.2f}")
                st.caption(f"Operação sugerida: **{s['Qtd']} ações** (~R$ {s['Total (R$)']:,.2f}) | Risco: {s['Risco (%)']}%")
                st.divider()

with tab_radar:
    st.subheader("Monitoramento de Papéis")
    busca = st.text_input("Filtrar Ticker:", "").upper()
    if busca:
        df_radar = df_radar[df_radar["Ticker"].str.contains(busca)]
    st.dataframe(df_radar, use_container_width=True, hide_index=True)

if st.button("🔄 Forçar Atualização"):
    st.cache_data.clear()
    st.rerun()