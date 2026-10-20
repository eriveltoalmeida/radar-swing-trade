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
    /* Ajuste de margens do container móvel */
    .block-container {
        padding-top: 0.6rem !important;
        padding-bottom: 2rem !important;
        padding-left: 0.7rem !important;
        padding-right: 0.7rem !important;
        background-color: #131722;
    }
    header, footer { visibility: hidden !important; height: 0px !important; }
    
    /* Barra Superior Sóbria */
    .top-bar-clean {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #1E222D;
        border: 1px solid #2A2E39;
        border-radius: 6px;
        padding: 7px 12px;
        margin-bottom: 9px;
    }
    .top-bar-clean .label { 
        font-size: 13px; 
        font-weight: 600; 
        color: #D1D4DC; 
        letter-spacing: 0.3px;
    }
    .top-bar-clean .badge-alta { 
        background: #1B382B; 
        color: #26A69A; 
        border: 1px solid #26A69A40; 
        padding: 2px 8px; 
        border-radius: 4px; 
        font-size: 11px; 
        font-weight: 600; 
    }
    .top-bar-clean .badge-baixa { 
        background: #3B2026; 
        color: #EF5350; 
        border: 1px solid #EF535040; 
        padding: 2px 8px; 
        border-radius: 4px; 
        font-size: 11px; 
        font-weight: 600; 
    }

    /* Cartão do Setup */
    .trade-card {
        background: #1E222D;
        border: 1px solid #2A2E39;
        border-left: 3px solid #4B5E78;
        border-radius: 6px;
        padding: 9px 12px;
        margin-bottom: 8px;
    }
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .ticker-name { 
        font-size: 16px; 
        font-weight: 700; 
        color: #F0F3FA; 
        letter-spacing: 0.5px;
    }
    .setup-pill { 
        background: #2A2E39; 
        color: #B2B5BE; 
        padding: 2px 7px; 
        border-radius: 4px; 
        font-size: 11px; 
        font-weight: 500; 
    }
    
    /* Tags de Status Sóbrias */
    .status-aguardando { 
        background: #332B1A; 
        color: #D4A347; 
        border: 1px solid #D4A34730; 
        padding: 1px 6px; 
        border-radius: 4px; 
        font-size: 10px; 
        font-weight: 600; 
    }
    .status-comprar { 
        background: #1B382B; 
        color: #26A69A; 
        border: 1px solid #26A69A40; 
        padding: 1px 6px; 
        border-radius: 4px; 
        font-size: 10px; 
        font-weight: 600; 
    }
    .status-esticado { 
        background: #3B2026; 
        color: #EF5350; 
        border: 1px solid #EF535040; 
        padding: 1px 6px; 
        border-radius: 4px; 
        font-size: 10px; 
        font-weight: 600; 
    }
    
    /* Grade de Métricas Neutra */
    .metrics-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr 1fr;
        gap: 4px;
        background: #171B26;
        border: 1px solid #232733;
        padding: 6px;
        border-radius: 5px;
        text-align: center;
    }
    .m-item .m-label { 
        font-size: 9px; 
        color: #787B86; 
        text-transform: uppercase; 
        font-weight: 500;
        margin-bottom: 2px;
    }
    .m-item .m-val { 
        font-size: 12.5px; 
        font-weight: 600; 
        color: #D1D4DC; 
    }
    
    /* Rodapé do Card */
    .card-footer {
        display: flex;
        justify-content: space-between;
        margin-top: 5px;
        font-size: 11px;
        color: #787B86;
    }
    .card-footer span b { 
        color: #B2B5BE; 
        font-weight: 600;
    }
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

TICKERS = sorted(list(set
