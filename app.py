import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ==========================================
# CONFIGURAÇÃO DE TELA MÓVEL & TEMA NATIVO
# ==========================================
st.set_page_config(
    page_title="Radar B3",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 2rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }
    header, footer { visibility: hidden !important; height: 0px !important; }
    
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #11141C;
        border: 1px solid #1E2330;
        border-radius: 10px;
        padding: 8px 14px;
        margin-bottom: 10px;
    }
    .top-bar .title { font-size: 15px; font-weight: 700; color: #FFFFFF; }
    .top-bar .badge-alta { background: #00875A; color: white; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: bold; }
    .top-bar .badge-baixa { background: #DE350B; color: white; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: bold; }

    .trade-card {
        background: #161A23;
        border: 1px solid #232838;
        border-left: 4px solid #00D084;
        border-radius: 8px;
        padding: 9px 12px;
        margin-bottom: 8px;
    }
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .ticker-name { font-size: 16px; font-weight: 800; color: #FFFFFF; }
    .setup-pill { background: #1C2B22; color: #00D084; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 600; }
    
    /* Tags de Status da Entrada */
    .status-aguardando { background: #2B2610; color: #FFD000; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 700; }
    .status-comprar { background: #004D2E; color: #00FF9D; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 700; }
    .status-esticado { background: #4D1212; color: #FF6666; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 700; }
    
    /* Grade de Métricas ajustada para 4 colunas */
    .metrics-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr 1fr;
        gap: 4px;
        background: #0D1017;
        padding: 6px 6px;
        border-radius: 6px;
        text-align: center;
    }
    .m-item .m-label { font-size: 9px; color: #8A94A6; text-transform: uppercase; }
    .m-item .m-val { font-size: 12.5px; font-weight: 700; color: #F1F3F5; }
    
    .card-footer {
        display: flex;
        justify-content: space-between;
        margin-top: 5px;
        font-size: 11px;
        color: #8A94A6;
    }
    .card-footer span b { color: #D0D5DD; }
</style>
""", unsafe_allow_html=True)

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

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

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

@st.cache_data(ttl=900)  # Reduzido para 15 min de cache para acompanhar o pregão
def executar_radar():
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
            rs_ibov = round(ret_ativo - retorno_ibov_21d, 1)

            tend_alta = preco > float(ult["SMA_200"]) and float(ult["EMA_21"]) > float(ult["SMA_50"])
            dist_mme21 = abs(float(ult["Low"]) - float(ult["EMA_21"])) / float(ult["EMA_21"])
            
            setup_nome = None
            gatilho, stop = None, None

            if vol_fin >= MIN_VOL_FINANCEIRO_DIA and tend_alta and float(ult["ADX"]) > ADX_MIN:
                # Pullback MME21
                if dist_mme21 <= 0.025 and preco > float(ult["EMA_21"]):
                    setup_nome = "Pullback MME21"
                    gatilho = round(float(ult["High"]) + (0.10 * atr), 2)
                    stop = round(min(float(ult["Low"]), float(ant["Low"])) - (1.0 * atr), 2)
                # Breakout 20d
                elif float(ult["High"]) >= float(ult["MAX_20"]) and float(ult["Volume"]) > float(ult["VOL_SMA_20"]):
                    setup_nome = "Breakout 20d"
                    gatilho = round(float(ult["High"]) + (0.10 * atr), 2)
                    stop = round(min(float(ult["Low"]), float(ant["Low"])) - (1.0 * atr), 2)

            if setup_nome and gatilho and stop:
                risco_acao = gatilho - stop
                risco_pct = (risco_acao / gatilho) * 100
                if 0 < risco_pct <= 8.0:
                    qtd = min(int((CAPITAL_TOTAL * RISCO_MAX_TRADE) // risco_acao), int((CAPITAL_TOTAL * ALOCACAO_MAX_POSICAO) // gatilho))
                    if qtd > 0:
                        # Avaliação do Ponto de Entrada vs Cotação Atual
                        dist_gatilho_pct = ((preco - gatilho) / gatilho) * 100
                        if preco < gatilho:
                            status_txt = "AGUARDANDO"
                            status_css = "status-aguardando"
                        elif preco >= gatilho and dist_gatilho_pct <= 1.0:
                            status_txt = "ENTRAR AGORA"
                            status_css = "status-comprar"
                        else:
                            status_txt = "ESTICOU"
                            status_css = "status-esticado"

                        setups.append({
                            "Ticker": ticker.replace(".SA", ""),
                            "Setup": setup_nome,
                            "PrecoAtual": preco,
                            "Gatilho": gatilho,
                            "Stop": stop,
                            "Alvo": round(gatilho + (2 * risco_acao), 2),
                            "Risco": round(risco_pct, 1),
                            "Qtd": qtd,
                            "Total": round(qtd * gatilho, 2),
                            "StatusTxt": status_txt,
                            "StatusCss": status_css,
                            "DistGatilho": dist_gatilho_pct
                        })

            radar.append({
                "Papel": ticker.replace(".SA", ""),
                "Preço": round(preco, 2),
                "RSI": round(float(ult["RSI"]), 0),
                "ADX": round(float(ult["ADX"]), 0),
                "RS Ibov": f"{rs_ibov:+}%"
            })
        except Exception:
            continue

    return regime_ibov, preco_ibov, pd.DataFrame(setups), pd.DataFrame(radar)

# ==========================================
# RENDERIZAÇÃO
# ==========================================
regime_ibov, preco_ibov, df_setups, df_radar = executar_radar()

badge_classe = "badge-alta" if regime_ibov == "ALTA" else "badge-baixa"
st.markdown(f"""
<div class="top-bar">
    <div class="title">⚡ Radar Swing Trade</div>
    <div class="{badge_classe}">IBOV: {regime_ibov} ({preco_ibov:,.0f})</div>
</div>
""", unsafe_allow_html=True)

aba = st.radio("visão", ["🎯 Setups Armados", "📊 Radar Geral"], horizontal=True, label_visibility="collapsed")

if aba == "🎯 Setups Armados":
    if df_setups.empty:
        st.info("Nenhum setup qualificado hoje.")
    else:
        for _, s in df_setups.iterrows():
            st.markdown(f"""
            <div class="trade-card">
                <div class="card-header">
                    <div>
                        <span class="ticker-name">{s['Ticker']}</span>
                        <span class="{s['StatusCss']}">{s['StatusTxt']}</span>
                    </div>
                    <span class="setup-pill">{s['Setup']}</span>
                </div>
                <div class="metrics-grid">
                    <div class="m-item">
                        <div class="m-label">Atual</div>
                        <div class="m-val" style="color: #61A5FF;">R$ {s['PrecoAtual']:.2f}</div>
                    </div>
                    <div class="m-item">
                        <div class="m-label">Gatilho</div>
                        <div class="m-val">R$ {s['Gatilho']:.2f}</div>
                    </div>
                    <div class="m-item">
                        <div class="m-label">Stop</div>
                        <div class="m-val" style="color: #FF5C5C;">R$ {s['Stop']:.2f}</div>
                    </div>
                    <div class="m-item">
                        <div class="m-label">Alvo 2:1</div>
                        <div class="m-val" style="color: #00D084;">R$ {s['Alvo']:.2f}</div>
                    </div>
                </div>
                <div class="card-footer">
                    <span>Lote: <b>{s['Qtd']} ações</b> (R$ {s['Total']:,.0f})</span>
                    <span>Dist. Gatilho: <b>{s['DistGatilho']:+.1f}%</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.dataframe(df_radar, use_container_width=True, hide_index=True, height=400)

if st.button("🔄 Atualizar Cotações", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
