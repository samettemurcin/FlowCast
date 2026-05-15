"""
FlowCast — Streamlit financial intelligence platform.
"""

from __future__ import annotations

import html
import io
import math
import textwrap
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts as ch
from data_processor import load_and_process
from forecaster import ForecastMethod, get_forecast, stl_decomposition_monthly

st.set_page_config(
    page_title="FlowCast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

TOP_NAV_PAGES = (
    ("home", "🏠", "Home"),
    ("dashboard", "📊", "Dashboard"),
    ("spending", "💸", "Spending"),
    ("forecast", "🔮", "Forecast"),
    ("ai", "💡", "Insights"),
    ("export", "📥", "Export"),
)

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Base */
html, body, .main, * {
    font-family: 'Inter', sans-serif !important;
}
/* Flush layout — remove Streamlit default top gap for hidden header */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
[data-testid="stAppViewContainer"] > .main,
section.main,
section.main > div,
[data-testid="stMain"],
[data-testid="stMain"] > div,
[data-testid="stMainBlockContainer"],
.main .block-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
.main .block-container {
    background: #F8F9FF;
    padding: 0 28px 28px !important;
    max-width: 100% !important;
}
/* Collapse empty Streamlit markdown wrappers (CSS inject, nav anchor) */
[data-testid="stMain"] [data-testid="element-container"]:has(style),
[data-testid="stMain"] [data-testid="element-container"]:has(.fc-top-nav-start),
[data-testid="stMain"] [data-testid="element-container"]:has(.fc-page-marker) {
    margin: 0 !important;
    padding: 0 !important;
    min-height: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    line-height: 0 !important;
}
[data-testid="stMain"] [data-testid="stMarkdown"]:has(style),
[data-testid="stMain"] [data-testid="stMarkdown"]:has(.fc-top-nav-start),
[data-testid="stMain"] [data-testid="stMarkdown"]:has(.fc-page-marker) {
    margin: 0 !important;
    padding: 0 !important;
    min-height: 0 !important;
}
/* Tighter vertical rhythm in main column */
[data-testid="stMain"] [data-testid="stVerticalBlock"] {
    gap: 0.5rem !important;
}
[data-testid="stMain"] [data-testid="element-container"] {
    margin-bottom: 0 !important;
}

/* Page background */
.stApp {
    background: #F8F9FF !important;
}

/* Hide broken Material icon label on sidebar collapse */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
    display: none !important;
}

/* KPI CARDS */
.kpi-card {
    background: white;
    border-radius: 20px;
    padding: 24px 20px;
    box-shadow: 0 2px 20px rgba(108,99,255,0.08);
    border-top: 4px solid;
    height: 100%;
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(108,99,255,0.15);
}
.kpi-label {
    font-size: 12px;
    font-weight: 600;
    color: #9090B0;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 34px;
    font-weight: 800;
    color: #1A1A2E;
    line-height: 1.1;
    margin-bottom: 6px;
}
.kpi-sub {
    font-size: 13px;
    color: #00B894;
    font-weight: 500;
}
.kpi-sub.down { color: #E17055; }

/* SECTION HEADERS */
.section-header {
    font-size: 24px;
    font-weight: 700;
    color: #1A1A2E;
    margin: 32px 0 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-sub {
    font-size: 14px;
    color: #9090B0;
    margin-top: -12px;
    margin-bottom: 20px;
}

/* CHART CONTAINERS */
.chart-wrap {
    background: white;
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

/* INSIGHT CARDS */
.insight-item {
    background: white;
    border-left: 4px solid #6C63FF;
    border-radius: 0 16px 16px 0;
    padding: 16px 20px;
    margin-bottom: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.insight-item.warning {
    border-left-color: #FFB347;
    background: #FFFBF0;
}
.insight-item.success {
    border-left-color: #00B894;
    background: #F0FFF8;
}
.insight-icon {
    font-size: 20px;
    margin-right: 8px;
}
.insight-title {
    font-size: 15px;
    font-weight: 700;
    color: #1A1A2E;
}
.insight-desc {
    font-size: 14px;
    color: #5A5A7A;
    line-height: 1.6;
    margin-top: 4px;
}

/* UPLOAD AREA */
.upload-panel-wrap {
    background: white;
    border-radius: 20px;
    padding: 8px 8px 4px;
    box-shadow: 0 4px 24px rgba(108,99,255,0.10);
    border: 1px solid #E8E6FF;
    margin-bottom: 8px;
}
.upload-panel-head {
    text-align: center;
    padding: 20px 16px 8px;
}
.upload-panel-head .upload-icon { font-size: 40px; margin-bottom: 8px; }
.upload-panel-head .upload-title {
    font-size: 17px; font-weight: 700; color: #1A1A2E; margin-bottom: 6px;
}
.upload-panel-head .upload-sub {
    font-size: 14px; color: #9090B0; line-height: 1.5;
}
.upload-panel-wrap [data-testid="stFileUploader"] {
    background: #F8F7FF !important;
    border: 2px dashed #A09BFF !important;
    border-radius: 16px !important;
    padding: 20px 16px !important;
    transition: border-color 0.2s, background 0.2s !important;
}
.upload-panel-wrap [data-testid="stFileUploader"]:hover {
    border-color: #6C63FF !important;
    background: #F0EEFF !important;
}
/* Hide duplicate label / fix "uploadUpload" overlap */
.upload-panel-wrap [data-testid="stFileUploader"] > label,
.upload-panel-wrap [data-testid="stFileUploader"] label[data-testid="stWidgetLabel"] {
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}
.upload-panel-wrap [data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
}
.upload-panel-wrap [data-testid="stFileUploaderDropzone"] > div {
    padding: 0 !important;
}
.upload-panel-wrap [data-testid="stFileUploaderDropzone"] small {
    color: #9090B0 !important;
    font-size: 13px !important;
}
.upload-panel-wrap [data-testid="stFileUploaderDropzone"] button {
    background: white !important;
    border: 2px solid #6C63FF !important;
    color: #6C63FF !important;
    border-radius: 50px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 8px 24px !important;
    margin-top: 8px !important;
}
.upload-panel-wrap [data-testid="stFileUploaderDropzone"] button:hover {
    background: #6C63FF !important;
    color: white !important;
}
.upload-panel-wrap [data-testid="stFileUploaderDropzone"] button p,
.upload-panel-wrap [data-testid="stFileUploaderDropzone"] button div {
    font-size: 14px !important;
    line-height: 1.4 !important;
}
/* TIPS / RECOMMENDATIONS */
.tips-box {
    border-radius: 0 16px 16px 0;
    padding: 16px 20px;
    margin: 0 0 24px 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
.tips-box .tips-title {
    font-size: 15px; font-weight: 700; color: #1A1A2E; margin-bottom: 10px;
}
.tips-box ul.tips-list {
    margin: 0; padding-left: 20px; color: #5A5A7A;
    font-size: 14px; line-height: 1.7;
}
.tips-box ul.tips-list li { margin-bottom: 4px; }
/* STEP CARDS on home */
.step-card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    border-left: 4px solid #6C63FF;
    box-shadow: 0 2px 12px rgba(108,99,255,0.08);
    height: 100%;
}
.step-card .step-num {
    font-size: 12px; font-weight: 700; color: #6C63FF;
    text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 6px;
}
.step-card .step-title { font-size: 16px; font-weight: 700; color: #1A1A2E; margin-bottom: 8px; }
.step-card .step-desc { font-size: 14px; color: #5A5A7A; line-height: 1.6; }

/* HOME HERO */
.fc-hero {
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    margin: 0 0 28px 0;
    padding: 44px 36px 36px;
    text-align: center;
    background: linear-gradient(135deg, #5B54E8 0%, #7C4DFF 42%, #0D9488 100%);
    box-shadow: 0 20px 60px rgba(91, 84, 232, 0.28);
}
.fc-hero::before,
.fc-hero::after {
    content: "";
    position: absolute;
    border-radius: 50%;
    background: rgba(255,255,255,0.07);
    pointer-events: none;
}
.fc-hero::before { width: 280px; height: 280px; top: -120px; right: -80px; }
.fc-hero::after { width: 200px; height: 200px; bottom: -90px; left: -60px; }
.fc-hero-inner { position: relative; z-index: 1; }
.fc-hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.16);
    color: white;
    padding: 6px 16px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 16px;
    border: 1px solid rgba(255,255,255,0.22);
    letter-spacing: 0.3px;
}
.fc-hero-logo {
    color: white;
    font-size: clamp(56px, 9vw, 92px);
    font-weight: 800;
    margin: 0 0 8px;
    letter-spacing: -3px;
    line-height: 0.95;
    text-shadow: 0 4px 24px rgba(0,0,0,0.15);
}
.fc-hero-tagline {
    color: rgba(255,255,255,0.92);
    font-size: clamp(17px, 2.2vw, 22px);
    font-weight: 400;
    margin: 0 auto 28px;
    max-width: 620px;
    line-height: 1.55;
}
.fc-hero-pills {
    display: flex;
    gap: 10px;
    justify-content: center;
    flex-wrap: wrap;
    margin-bottom: 32px;
}
.fc-hero-pill {
    background: rgba(255,255,255,0.16);
    color: white;
    padding: 8px 18px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid rgba(255,255,255,0.22);
}
.fc-hero-directions {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 18px;
    padding: 20px 24px;
    text-align: left;
    max-width: 900px;
    margin: 0 auto;
    backdrop-filter: blur(8px);
}
.fc-hero-directions-title {
    color: white;
    font-size: 14px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 14px;
    opacity: 0.95;
}
.fc-hero-steps {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
}
@media (max-width: 900px) {
    .fc-hero-steps { grid-template-columns: 1fr; }
}
.fc-hero-step {
    background: rgba(255,255,255,0.14);
    border-radius: 14px;
    padding: 14px 16px;
    border: 1px solid rgba(255,255,255,0.18);
}
.fc-hero-step-num {
    font-size: 11px;
    font-weight: 800;
    color: rgba(255,255,255,0.75);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 6px;
}
.fc-hero-step-title {
    color: white;
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 4px;
}
.fc-hero-step-desc {
    color: rgba(255,255,255,0.85);
    font-size: 13px;
    line-height: 1.45;
}

/* BUTTONS */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 12px 28px !important;
    transition: all 0.2s !important;
    border: 2px solid #6C63FF !important;
    color: #6C63FF !important;
    background: white !important;
}
.stButton > button:hover {
    background: #6C63FF !important;
    color: white !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(108,99,255,0.3) !important;
}

/* PRIMARY BUTTON */
.primary-btn button {
    background: linear-gradient(135deg,
      #6C63FF, #8B5CF6) !important;
    color: white !important;
    border-color: transparent !important;
    box-shadow: 0 4px 16px rgba(108,99,255,0.35) !important;
}

/* METRICS */
[data-testid="stMetric"] {
    background: white;
    border-radius: 16px;
    padding: 16px 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
[data-testid="stMetricValue"] {
    font-size: 30px !important;
    font-weight: 800 !important;
    color: #1A1A2E !important;
}
[data-testid="stMetricLabel"] {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #9090B0 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border-radius: 16px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
}

/* TABS */
.stTabs [data-testid="stTab"] {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #6C63FF !important;
}

/* SELECT/SLIDER labels */
.stSelectbox label, .stSlider label,
.stRadio label {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #5A5A7A !important;
}

/* General text bigger */
p {
    font-size: 15px !important;
    line-height: 1.7 !important;
    color: #444466 !important;
}

/* Hide Streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    height: 0 !important;
    min-height: 0 !important;
    visibility: hidden !important;
    display: none !important;
}

/* Inner page intro strip */
.fc-page-intro {
    background: white;
    border-radius: 16px;
    padding: 18px 22px;
    margin-bottom: 20px;
    box-shadow: 0 2px 14px rgba(108,99,255,0.07);
    border: 1px solid #ECECF8;
}
.fc-page-intro h2 {
    margin: 0 0 4px !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #1A1A2E !important;
}
.fc-page-intro p {
    margin: 0 !important;
    font-size: 14px !important;
    color: #5A5A7A !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #F8F9FF; }
::-webkit-scrollbar-thumb {
    background: #D0CEFF;
    border-radius: 3px;
}

.alert-warn {
    background: #FFFBF0;
    border-left: 4px solid #FFB347;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 14px;
    color: #664400;
}
.alert-good {
    background: #F0FFF8;
    border-left: 4px solid #00B894;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 14px;
    color: #006644;
}

.fc-top-nav-start,
.fc-page-marker {
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Top navigation bar — next element-container after nav anchor */
[data-testid="stMain"] [data-testid="element-container"]:has(.fc-top-nav-start) + [data-testid="element-container"] {
    background: white !important;
    border-bottom: 2px solid #F0F0F8 !important;
    margin: 0 -28px 1rem -28px !important;
    padding: 8px 20px 6px !important;
    box-shadow: 0 2px 12px rgba(108,99,255,0.06) !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 100 !important;
}
[data-testid="stMain"] [data-testid="element-container"]:has(.fc-top-nav-start) + [data-testid="element-container"] [data-testid="stHorizontalBlock"] {
    gap: 6px !important;
    align-items: center !important;
}
[data-testid="stMain"] [data-testid="element-container"]:has(.fc-top-nav-start) + [data-testid="element-container"] [data-testid="column"] {
    min-width: 0 !important;
}
[data-testid="stMain"] [data-testid="element-container"]:has(.fc-top-nav-start) + [data-testid="element-container"] .stButton > button {
    border-radius: 50px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    width: 100% !important;
    min-height: 42px !important;
    max-height: 42px !important;
    padding: 8px 12px !important;
    white-space: nowrap !important;
    line-height: 1.2 !important;
    transition: all 0.15s ease !important;
}
[data-testid="stMain"] [data-testid="element-container"]:has(.fc-top-nav-start) + [data-testid="element-container"] .stButton > button p,
[data-testid="stMain"] [data-testid="element-container"]:has(.fc-top-nav-start) + [data-testid="element-container"] .stButton > button div {
    white-space: nowrap !important;
    font-size: 13px !important;
    line-height: 1.2 !important;
}
[data-testid="stMain"] [data-testid="element-container"]:has(.fc-top-nav-start) + [data-testid="element-container"] [data-testid="baseButton-secondary"] {
    background: #F8F7FF !important;
    color: #6C63FF !important;
    border: 2px solid #E8E6FF !important;
}
[data-testid="stMain"] [data-testid="element-container"]:has(.fc-top-nav-start) + [data-testid="element-container"] [data-testid="baseButton-secondary"]:hover {
    background: #6C63FF !important;
    color: white !important;
    border-color: #6C63FF !important;
}
[data-testid="stMain"] [data-testid="element-container"]:has(.fc-top-nav-start) + [data-testid="element-container"] [data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #6C63FF, #8B5CF6) !important;
    color: white !important;
    border: 2px solid transparent !important;
    box-shadow: 0 4px 16px rgba(108,99,255,0.35) !important;
}
</style>
"""

SIDEBAR_CSS = """
<style>
/* SIDEBAR: Clean white with light border */
section[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #EEEEF5 !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.04) !important;
}

/* Sidebar text dark */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: #1A1A2E !important;
}

/* FlowCast title in sidebar */
section[data-testid="stSidebar"] h1 {
    background: linear-gradient(135deg, #6C63FF, #00C9A7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 28px !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}

/* Sidebar buttons */
section[data-testid="stSidebar"] .stButton > button {
    background: #F8F7FF !important;
    color: #6C63FF !important;
    border: 1.5px solid #E8E6FF !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 16px !important;
    width: 100% !important;
    text-align: left !important;
    margin-bottom: 4px !important;
    transition: all 0.15s ease !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #6C63FF !important;
    color: white !important;
    border-color: #6C63FF !important;
    transform: translateX(4px) !important;
}

/* Active page button */
.nav-active button {
    background: linear-gradient(135deg, #6C63FF, #8B5CF6) !important;
    color: white !important;
    border-color: transparent !important;
    box-shadow: 0 4px 12px rgba(108,99,255,0.3) !important;
}

/* Mode buttons */
.mode-personal button {
    background: linear-gradient(135deg, #6C63FF, #8B5CF6) !important;
    color: white !important;
    border-color: transparent !important;
}
.mode-business button {
    background: linear-gradient(135deg, #0066FF, #0099CC) !important;
    color: white !important;
    border-color: transparent !important;
}

/* Sidebar divider */
section[data-testid="stSidebar"] hr {
    border-color: #EEEEF5 !important;
    margin: 12px 0 !important;
}

/* Data loaded badge */
.data-badge {
    background: #E8FFF5;
    border: 1px solid #00B894;
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 13px;
    color: #006644;
    font-weight: 500;
}
</style>
"""

def render_home_hero() -> None:
    """Large FlowCast branding + quick-start directions on Home."""
    st.markdown(
        """
<div class="fc-hero">
  <div class="fc-hero-inner">
    <div class="fc-hero-badge">✨ Financial Intelligence Platform</div>
    <h1 class="fc-hero-logo">FlowCast</h1>
    <p class="fc-hero-tagline">
      Upload your bank export, choose Personal or Business mode, and explore dashboards,
      forecasts, and plain-language insights in minutes.
    </p>
    <div class="fc-hero-pills">
      <span class="fc-hero-pill">🧠 Smart insights</span>
      <span class="fc-hero-pill">📊 Rich dashboards</span>
      <span class="fc-hero-pill">🔮 Forecasting</span>
      <span class="fc-hero-pill">📤 One-click export</span>
    </div>
    <div class="fc-hero-directions">
      <div class="fc-hero-directions-title">How to get started</div>
      <div class="fc-hero-steps">
        <div class="fc-hero-step">
          <div class="fc-hero-step-num">Step 1</div>
          <div class="fc-hero-step-title">👤 Pick your mode</div>
          <div class="fc-hero-step-desc">Personal for spending habits, or Business for revenue &amp; cash flow.</div>
        </div>
        <div class="fc-hero-step">
          <div class="fc-hero-step-num">Step 2</div>
          <div class="fc-hero-step-title">📂 Upload your file</div>
          <div class="fc-hero-step-desc">Drop a CSV or Excel bank export—we map dates and amounts automatically.</div>
        </div>
        <div class="fc-hero-step">
          <div class="fc-hero-step-num">Step 3</div>
          <div class="fc-hero-step-title">📊 Explore &amp; forecast</div>
          <div class="fc-hero-step-desc">Use the top navigation for Dashboard, Spending, Forecast, and Insights.</div>
        </div>
      </div>
    </div>
  </div>
</div>
        """.strip(),
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    defaults = {
        "page": "home",
        "mode": "personal",
        "df_raw": None,
        "df_monthly": None,
        "summary": None,
        "file_name": None,
        "file_bytes": None,
        "forecast_df": None,
        "fc_prophet": None,
        "fc_arima": None,
        "fc_fingerprint": None,
        "home_mode_selected": False,
        "fc_horizon": 6,
        "fc_ci": 0.95,
        "fc_season": "auto",
        "fc_whatif": 0.0,
        "fc_view": "compare",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _html(markup: str) -> None:
    """Render raw HTML (always uses unsafe_allow_html=True)."""
    st.markdown(textwrap.dedent(markup).strip(), unsafe_allow_html=True)


def _inject_css(page: str) -> None:
    """Single style block + page theme (one markdown node = less top gap)."""
    page_theme = ""
    if page == "home":
        page_theme = """
        <style>
        section.main .block-container {
            background:
                radial-gradient(ellipse 80% 50% at 10% -10%, rgba(108,99,255,0.18), transparent 55%),
                radial-gradient(ellipse 60% 40% at 95% 5%, rgba(0,201,167,0.12), transparent 50%),
                linear-gradient(180deg, #ECEEFF 0%, #F4F5FF 28%, #F8F9FF 100%) !important;
        }
        </style>
        """
    st.markdown(
        f"{GLOBAL_CSS}{SIDEBAR_CSS}{page_theme}"
        f'<div class="fc-page-marker" data-page="{html.escape(page)}"></div>',
        unsafe_allow_html=True,
    )


def section_header(icon: str, title: str, subtitle: str = "") -> None:
    sub_html = ""
    if subtitle:
        sub_html = (
            f'<div style="font-size:14px;color:#9090B0;margin-top:4px;">'
            f"{html.escape(subtitle)}</div>"
        )
    block = (
        f'<div style="margin:28px 0 16px;">'
        f'<div style="font-size:22px;font-weight:700;color:#1A1A2E;display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
        f'<span style="font-size:26px;">{icon}</span> {html.escape(title)}</div>'
        f"{sub_html}"
        f'<div style="height:3px;background:linear-gradient(90deg,#6C63FF,#00C9A7,transparent);border-radius:2px;margin-top:10px;"></div>'
        f"</div>"
    )
    st.markdown(block, unsafe_allow_html=True)


def tips_box(title: str, items: list[str], *, variant: str = "info") -> None:
    colors = {
        "info": ("#6C63FF", "#F8F7FF"),
        "success": ("#00B894", "#F0FFF8"),
        "warn": ("#FFB347", "#FFFBF0"),
    }
    border, bg = colors.get(variant, colors["info"])
    lis = "".join(f"<li>{html.escape(t)}</li>" for t in items)
    st.markdown(
        f'<div class="tips-box" style="border-left:4px solid {border};background:{bg};">'
        f'<div class="tips-title">{html.escape(title)}</div>'
        f'<ul class="tips-list">{lis}</ul></div>',
        unsafe_allow_html=True,
    )


def _render_upload_zone() -> Any:
    """Professional upload panel; empty label avoids duplicate Upload text."""
    st.markdown(
        '<div class="upload-panel-wrap"><div class="upload-panel-head">'
        '<div class="upload-icon">📂</div>'
        '<div class="upload-title">Drag & drop your file here</div>'
        '<div class="upload-sub">CSV or Excel · bank export with date & amount columns</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        " ",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
        key="flowcast_main_upload",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return uploaded


def kpi_card(
    label: str,
    value: str,
    sub: str = "",
    color: str = "#6C63FF",
    up: bool = True,
) -> None:
    arrow = "↑" if up else "↓"
    sub_color = "#00B894" if up else "#E17055"
    _html(
        f"""
    <div style="
        background: white;
        border-radius: 20px;
        padding: 24px 20px;
        box-shadow: 0 2px 20px rgba(108,99,255,0.08);
        border-top: 4px solid {color};
        height: 100%;
    ">
        <div style="
            font-size: 12px; font-weight: 600;
            color: #9090B0; text-transform: uppercase;
            letter-spacing: 0.8px; margin-bottom: 10px;
        ">{html.escape(label)}</div>
        <div style="
            font-size: 32px; font-weight: 800;
            color: #1A1A2E; line-height: 1.1;
            margin-bottom: 8px;
        ">{value}</div>
        {f'<div style="font-size:13px; font-weight:500; color:{sub_color};">{arrow} {html.escape(sub)}</div>' if sub else ''}
    </div>
    """)


PAGE_INTROS: dict[str, tuple[str, str, str]] = {
    "dashboard": ("📊", "Dashboard", "KPIs, trends, and monthly breakdowns at a glance."),
    "spending": ("💸", "Spending", "Category breakdowns, patterns, and where your money goes."),
    "forecast": ("🔮", "Forecast", "Project future cash flow with confidence intervals."),
    "ai": ("💡", "Insights", "Plain-language recommendations from your data."),
    "export": ("📤", "Export", "Download charts, forecasts, and reports in one click."),
}


def render_page_intro(page: str) -> None:
    if page not in PAGE_INTROS:
        return
    icon, title, subtitle = PAGE_INTROS[page]
    st.markdown(
        f'<div class="fc-page-intro">'
        f'<h2>{icon} {html.escape(title)}</h2>'
        f'<p>{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def render_top_nav() -> None:
    current = st.session_state.get("page", "home")
    _html('<div class="fc-top-nav-start" aria-hidden="true"></div>')
    cols = st.columns(len(TOP_NAV_PAGES))
    for idx, (key, icon, name) in enumerate(TOP_NAV_PAGES):
        with cols[idx]:
            if st.button(
                f"{icon} {name}",
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if current == key else "secondary",
            ):
                st.session_state.page = key
                st.rerun()



def fmt_money(x: float | None, *, compact: bool = False) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    x = float(x)
    ax = abs(x)
    if compact and ax >= 1_000_000:
        return f"${x/1_000_000:,.1f}M"
    if compact and ax >= 1_000:
        return f"${x/1_000:,.1f}K"
    return f"${x:,.2f}"


def fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if isinstance(x, float) and math.isinf(x):
        return "∞%"
    return f"{x:+.1f}%"


def _safe_plot(fn: Callable[..., go.Figure], *args: Any, **kwargs: Any) -> go.Figure | None:
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _plot_with_caption(fn: Callable[..., go.Figure], caption: str, *args: Any, **kwargs: Any) -> None:
    _html('<div class="chart-wrap">')
    fig = _safe_plot(fn, *args, **kwargs)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        st.caption(caption)
    else:
        st.warning("Chart could not be drawn for this data.")
    _html("</div>")


def _empty_state() -> None:
    _html(
        """
<div style="
  max-width:560px;margin:3rem auto;padding:2.5rem 2rem;text-align:center;
  background:linear-gradient(135deg,#F8F7FF,#F0FFFE);
  border-radius:24px;box-shadow:0 8px 32px rgba(108,99,255,0.15);
  border:1px solid #E8E8F8;
">
  <div style="font-size:3.5rem;margin-bottom:12px;">📂</div>
  <h2 style="color:#1A1A2E;font-size:1.5rem;margin-bottom:8px;">No data loaded yet</h2>
  <p style="color:#444466;font-size:16px;line-height:1.6;">
    Upload a bank export on <b>Home</b> to unlock dashboards, spending views, forecasts, and exports.
  </p>
</div>
        """,
    )
    if st.button("Go to Home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()


def _has_data() -> bool:
    return st.session_state.df_monthly is not None and st.session_state.df_raw is not None


def _reprocess_from_bytes() -> None:
    b = st.session_state.file_bytes
    if not b:
        return
    fn = st.session_state.file_name or "data.csv"
    with st.spinner("Analyzing your data... 🧠"):
        df_m, df_r, summary, mode = load_and_process(
            io.BytesIO(b), mode=st.session_state.mode, filename=fn
        )
    st.session_state.df_monthly = df_m
    st.session_state.df_raw = df_r
    st.session_state.summary = summary
    st.session_state.forecast_df = None
    st.session_state.fc_prophet = None
    st.session_state.fc_arima = None
    st.session_state.fc_fingerprint = None


def _forecast_fingerprint() -> str:
    return "|".join(
        str(st.session_state.get(k, ""))
        for k in ("mode", "fc_horizon", "fc_ci", "fc_season", "fc_whatif", "fc_view")
    )


def _clear_forecasts_if_stale() -> None:
    cur = _forecast_fingerprint()
    if st.session_state.get("fc_fingerprint") != cur:
        st.session_state.forecast_df = None
        st.session_state.fc_prophet = None
        st.session_state.fc_arima = None


def _holdout_pred(monthly: pd.DataFrame, method: ForecastMethod, **kw: Any) -> np.ndarray | None:
    if len(monthly) < 6:
        return None
    train = monthly.iloc[:-3].reset_index(drop=True)
    act = monthly.iloc[-3:]["y"].to_numpy(dtype=float)
    try:
        fc, _ = get_forecast(train, method=method, periods=3, **kw)
        return fc["yhat"].to_numpy(dtype=float)[:3]
    except Exception:
        return None


def _accuracy_block(monthly: pd.DataFrame, method: ForecastMethod, **kw: Any) -> tuple[float | None, ...]:
    if len(monthly) < 6:
        return (None,) * 4
    y = monthly.iloc[-3:]["y"].to_numpy(dtype=float)
    pred = _holdout_pred(monthly, method, **kw)
    if pred is None:
        return (None,) * 4
    mae = float(np.mean(np.abs(y - pred)))
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    mask = np.abs(y) > 1e-9
    mape = float(np.mean(np.abs((y[mask] - pred[mask]) / y[mask])) * 100.0) if mask.any() else None
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    return mae, rmse, mape, r2


def _sidebar() -> None:
    _html("<h1>FlowCast</h1>")
    st.caption("Financial Intelligence Platform")
    st.divider()

    mode = st.session_state.mode
    c1, c2 = st.columns(2)
    with c1:
        pers = st.button("👤 Personal", use_container_width=True, key="mode_personal")
    with c2:
        biz = st.button("🏢 Business", use_container_width=True, key="mode_business")
    if mode == "personal":
        _html(
            """
<style>
section[data-testid="stSidebar"] [data-testid="column"]:first-of-type .stButton > button {
    background: linear-gradient(135deg, #6C63FF, #8B5CF6) !important;
    color: white !important;
    border-color: transparent !important;
}
</style>
""",
        )
    elif mode == "business":
        _html(
            """
<style>
section[data-testid="stSidebar"] [data-testid="column"]:nth-of-type(2) .stButton > button {
    background: linear-gradient(135deg, #0066FF, #0099CC) !important;
    color: white !important;
    border-color: transparent !important;
}
</style>
""",
        )
    if pers:
        st.session_state.mode = "personal"
        if st.session_state.file_bytes:
            _reprocess_from_bytes()
        st.rerun()
    if biz:
        st.session_state.mode = "business"
        if st.session_state.file_bytes:
            _reprocess_from_bytes()
        st.rerun()

    st.divider()
    if st.session_state.file_name:
        name = html.escape(st.session_state.file_name)
        _html(
            f'<div class="data-badge">✅ {name}</div>',
        )
    if st.button("🗑 Clear data", use_container_width=True, type="secondary"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        _init_state()
        st.rerun()


def page_home() -> None:
    render_home_hero()

    tips_box(
        "Quick tip",
        [
            "No bank file handy? Download the sample CSV below to explore every page instantly.",
            "You can switch Personal ↔ Business anytime from the sidebar or the cards below.",
        ],
        variant="success",
    )

    section_header("👤", "Choose your mode")
    pc, bc = st.columns(2)
    with pc:
        _html(
            """
<div style="
    background: linear-gradient(135deg, #6C63FF, #8B5CF6);
    border-radius: 20px; padding: 32px; color: white;
    height: 100%;
">
    <div style="font-size: 48px; margin-bottom: 12px;">👤</div>
    <h3 style="color:white; font-size:24px; margin-bottom:8px;">
      Personal Mode</h3>
    <p style="color:rgba(255,255,255,0.85); font-size:15px;
      line-height:1.6;">
      Track your spending habits, find savings opportunities,
      and forecast your personal financial future.
    </p>
    <ul style="color:rgba(255,255,255,0.85);
      font-size:14px; margin-top:16px; padding-left:20px;">
        <li>Monthly spending breakdown</li>
        <li>Category analysis</li>
        <li>Budget health score</li>
        <li>Savings recommendations</li>
    </ul>
</div>
""",
        )
        if st.button("Select Personal Mode →", key="sel_personal", use_container_width=True):
            st.session_state.mode = "personal"
            st.session_state.home_mode_selected = True
            if st.session_state.file_bytes:
                _reprocess_from_bytes()
            st.rerun()
    with bc:
        _html(
            """
<div style="
    background: linear-gradient(135deg, #0066FF, #0099CC);
    border-radius: 20px; padding: 32px; color: white;
    height: 100%;
">
    <div style="font-size: 48px; margin-bottom: 12px;">🏢</div>
    <h3 style="color:white; font-size:24px; margin-bottom:8px;">
      Business Mode</h3>
    <p style="color:rgba(255,255,255,0.85); font-size:15px;
      line-height:1.6;">
      Revenue tracking, cash flow, P&amp;L signals, and burn-rate visibility in one place.
    </p>
    <ul style="color:rgba(255,255,255,0.85);
      font-size:14px; margin-top:16px; padding-left:20px;">
        <li>Revenue vs expense trends</li>
        <li>Monthly cash position</li>
        <li>Category-level business spend</li>
        <li>Forward-looking scenarios</li>
    </ul>
</div>
""",
        )
        if st.button("Select Business Mode →", key="sel_business", use_container_width=True):
            st.session_state.mode = "business"
            st.session_state.home_mode_selected = True
            if st.session_state.file_bytes:
                _reprocess_from_bytes()
            st.rerun()

    section_header(
        "📂",
        "Upload Your Financial Data",
        "Works with any bank export. Supports CSV and Excel files.",
    )

    tips_box(
        "Tips for best results",
        [
            "Export 3–12 months of transactions so trends and forecasts are meaningful.",
            "Include a category column when your bank provides one—unlocks richer charts.",
            "Use CSV for fastest uploads; Excel (.xlsx) works too.",
        ],
    )

    uc1, uc2, uc3 = st.columns([1, 2, 1])
    with uc2:
        uploaded_file = _render_upload_zone()
        try:
            with open("sample_data.csv", "rb") as f:
                st.download_button(
                    "⬇️ Download sample CSV to try FlowCast",
                    f.read(),
                    file_name="sample_data.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        except OSError:
            pass

    if uploaded_file is not None:
        st.session_state.file_bytes = uploaded_file.getvalue()
        st.session_state.file_name = uploaded_file.name
        with st.spinner("Analyzing your data... 🧠"):
            _reprocess_from_bytes()

    s1, s2, s3 = st.columns(3)
    with s1:
        _html(
            '<div class="step-card"><div class="step-num">Step 1</div><div class="step-title">📥 Export from your bank</div><div class="step-desc">Download a CSV or Excel statement from your banking app.</div></div>',
        )
    with s2:
        _html(
            '<div class="step-card"><div class="step-num">Step 2</div><div class="step-title">📤 Upload here</div><div class="step-desc">Drop your file in the panel above — we detect columns automatically.</div></div>',
        )
    with s3:
        _html(
            '<div class="step-card"><div class="step-num">Step 3</div><div class="step-title">✨ Get insights</div><div class="step-desc">Explore dashboards, forecasts, and plain-language recommendations.</div></div>',
        )
    if st.session_state.df_monthly is not None:
        s = st.session_state.summary or {}
        dm = st.session_state.df_monthly
        n_m = len(dm)
        pct = s.get("pct_change")

        section_header("⚡", "Quick snapshot")
        if st.session_state.mode == "personal":
            pc1, pc2, pc3, pc4 = st.columns(4)
            up_pct = isinstance(pct, float) and not math.isnan(pct) and pct >= 0
            with pc1:
                kpi_card("💰 Total spent", fmt_money(s.get("total")), f"over {n_m} months", "#6C63FF", True)
            with pc2:
                kpi_card("📅 Monthly average", fmt_money(s.get("mean")), "vs earlier months", "#00B894", up_pct)
            with pc3:
                kpi_card(
                    "🔺 Peak month",
                    html.escape(str(s.get("max_month", ""))),
                    fmt_money(s.get("max_val")),
                    "#FFB347",
                    True,
                )
            with pc4:
                kpi_card(
                    "📉 3‑mo change",
                    html.escape(str(s.get("trend", "")).title()),
                    fmt_pct(float(pct)) if isinstance(pct, (int, float)) else "—",
                    "#FF6584",
                    up_pct,
                )
        else:
            tr = float(s.get("total_revenue") or 0)
            te = float(s.get("total_expenses") or 0)
            net = float(s.get("net_total") or s.get("total") or 0)
            burn = "Healthy" if net >= 0 else ("Watch closely" if net > -0.1 * max(te, 1) else "Attention needed")
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                kpi_card("💵 Total revenue", fmt_money(tr), "", "#00B894", True)
            with b2:
                kpi_card("💸 Total expenses", fmt_money(te), "", "#E17055", False)
            with b3:
                kpi_card("📊 Net cash flow", fmt_money(net), "", "#00B894" if net >= 0 else "#E17055", net >= 0)
            with b4:
                kpi_card("🔥 Cash signal", html.escape(burn), "", "#FFB347", net >= 0)

        _plot_with_caption(
            ch.plot_area_trend,
            "Look for months that sit above or below the curve—those are when spending sped up or slowed down.",
            dm.rename(columns={"ds": "ds", "y": "y"}),
            title="Monthly trend (sparkline view)",
        )

        _html("<br>")
        _html('<div class="primary-btn">')
        if st.button("→ Explore Dashboard", type="primary", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
        _html('</div>')


def page_dashboard() -> None:
    if not _has_data():
        _empty_state()
        return
    s = st.session_state.summary or {}
    dm = st.session_state.df_monthly
    raw = st.session_state.df_raw
    y_series = dm["y"].to_numpy(dtype=float)
    spend_variation = float(np.std(y_series)) if len(y_series) else 0.0

    tips_box(
        "How to read this page",
        [
            "KPI cards summarize your full date window—use them as your headline numbers.",
            "Compare the trend line with the donut to see if one month or one category drives change.",
            "High month-to-month swing means budgeting with a buffer may help.",
        ],
    )

    section_header("📊", "Overview")
    r1 = st.columns(4)
    cards = [
        ("Total in window", s.get("total"), "#6C63FF", None),
        ("Average per month", s.get("mean"), "#00C9A7", None),
        ("Typical month (median)", s.get("median"), "#FFB347", None),
        (
            "How much months swing",
            spend_variation,
            "#6C63FF",
            "Higher = more up-and-down between months.",
        ),
    ]
    for (lab, val, col, sub), c in zip(cards, r1):
        with c:
            disp = fmt_money(float(val)) if val is not None else "—"
            kpi_card(lab, disp, sub or "", col, True)

    left, right = st.columns([0.58, 0.42])
    with left:
        title = "Monthly spending trend" if st.session_state.mode == "personal" else "Monthly net trend"
        _plot_with_caption(
            ch.plot_area_trend,
            "Each point is one month—use this line to see the direction of your cash flow.",
            dm,
            title=title,
        )
    with right:
        try:
            cat_ok = (raw["category"].astype(str).str.strip().ne("")) & (
                raw["category"].astype(str).str.lower().ne("nan")
            )
            if cat_ok.any():
                gcat = raw.loc[cat_ok].copy()
                gcat["_a"] = gcat["amount"].abs()
                g = gcat.groupby("category", as_index=False)["_a"].sum()
                g = g.rename(columns={"category": "cat", "_a": "val"}).sort_values("val", ascending=False).head(8)
                _plot_with_caption(
                    ch.plot_donut,
                    "Slices show where dollars went—big slices are categories to revisit first.",
                    g["cat"].tolist(),
                    g["val"].tolist(),
                    title="Share by category",
                )
            else:
                total = float(raw["amount"].abs().sum())
                _plot_with_caption(
                    ch.plot_donut,
                    "Add a category column in your file to split this donut by category.",
                    ["Total"],
                    [total],
                    title="Total",
                )
        except Exception as e:
            st.warning(f"Category chart skipped: {e}")

    a, b, c = st.columns(3)
    with a:
        section_header("📊", "Monthly totals (bars)")
        bar_title = "Monthly spending" if st.session_state.mode == "personal" else "Monthly net"
        _plot_with_caption(
            ch.plot_monthly_bars,
            "Taller bars mean a heavier month; green is at or above your average month, red is below.",
            dm,
            title=bar_title,
        )
    with b:
        section_header("📈", "Latest vs prior month")
        _html('<div class="chart-wrap">')
        if len(dm) >= 2:
            last = float(dm["y"].iloc[-1])
            prev = float(dm["y"].iloc[-2])
            fig = go.Figure()
            fig.add_trace(
                go.Bar(name="Prior month", x=["Months"], y=[prev], marker_color="#94a3b8", hovertemplate="Prior<br>$%{y:,.2f}<extra></extra>")
            )
            bar_col = "#00B894" if last <= prev else "#E17055"
            fig.add_trace(
                go.Bar(name="Latest month", x=["Months"], y=[last], marker_color=bar_col, hovertemplate="Latest<br>$%{y:,.2f}<extra></extra>")
            )
            fig.update_layout(barmode="group", paper_bgcolor="#fff", plot_bgcolor="#fff", font=dict(family="Inter, sans-serif", size=13))
            fig.update_yaxes(tickprefix="$")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Compare the two rightmost months: a higher latest bar usually means more cash went out (personal) or weaker net (business).")
        else:
            st.info("Need at least two months to compare.")
        _html("</div>")
    with c:
        section_header("🎯", "Spending rhythm")
        vol = float(np.std(y_series) / (np.mean(np.abs(y_series)) + 1e-9))
        score = int(np.clip(vol * 80, 0, 100))
        _plot_with_caption(
            ch.plot_gauge,
            "Higher scores mean month‑to‑month totals jump around more—worth a steady budget if you want calm cash flow.",
            float(score),
            "Month-to-month bumpiness",
            max_val=100.0,
        )

    if st.session_state.mode == "personal":
        section_header("📋", "Simple report card")
        cv = float(np.std(y_series) / (np.mean(y_series) + 1e-9)) if len(y_series) else 0
        grade_cons = "Strong" if cv < 0.15 else ("Good" if cv < 0.25 else ("Fair" if cv < 0.4 else "Shaky"))
        _html(
            f'<div class="insight-item"><div class="insight-title">Consistency</div>'
            f'<div class="insight-desc">{html.escape(grade_cons)} — how steady your monthly totals look.</div></div>',
        )
        _html(
            f'<div class="insight-item"><div class="insight-title">Direction</div>'
            f'<div class="insight-desc">Recent pattern: <b>{html.escape(str(s.get("trend","")).title())}</b>.</div></div>',
        )
    else:
        section_header("💼", "Business pulse")
        if s.get("has_drcr"):
            _html(
                '<div class="alert-good">Debit/credit column detected—we split revenue and spend automatically.</div>',
            )
        _plot_with_caption(
            ch.plot_monthly_bars,
            "Same monthly bars as above, focused on net so you can pair with your runway planning.",
            dm,
            title="Net by month",
        )


def page_spending() -> None:
    if not _has_data():
        _empty_state()
        return
    raw = st.session_state.df_raw
    tips_box(
        "Recommendations",
        [
            "Filter by date range to focus on a trip, quarter, or pay cycle.",
            "Treemap shows where dollars go; calendar heatmap shows when you spend.",
            "Check unusual months at the bottom for one-off spikes worth investigating.",
        ],
    )

    if st.session_state.mode == "business":
        section_header("💼", "Business spending view")
        st.caption("Treemap and trends use credits for revenue and debits for expenses when your file marks Dr/Cr.")
        t = raw.copy()
        t["amt"] = t["amount"].astype(float)
        if st.session_state.summary.get("has_drcr"):
            exp = t.loc[t["drcr"] == "Db"].copy()
            rev = t.loc[t["drcr"] == "Cr"].copy()
            c1, c2 = st.columns(2)
            with c1:
                if exp["category"].astype(str).str.len().sum() > 0:
                    _plot_with_caption(
                        ch.plot_treemap,
                        "Box size = dollars—big boxes are spend categories to watch in supplier review.",
                        exp,
                        "category",
                        "amount",
                        title="Expense treemap",
                    )
                else:
                    st.info("Add a category column for a richer treemap.")
            with c2:
                rv = rev.groupby(rev["date"].dt.to_period("M").dt.to_timestamp(), as_index=False)["amt"].sum()
                rv.columns = ["ds", "y"]
                _plot_with_caption(
                    ch.plot_area_trend,
                    "This line is revenue by month—spot dips early.",
                    rv,
                    title="Monthly revenue",
                )
        else:
            _html(
                '<div class="alert-warn">Add a debit/credit column for the best business breakdown.</div>',
            )
        return

    section_header("💸", "Where does your money go?")
    dmin, dmax = pd.to_datetime(raw["date"]).min(), pd.to_datetime(raw["date"]).max()
    dr = st.date_input("Date range", value=(dmin.date(), dmax.date()))
    if isinstance(dr, tuple) and len(dr) == 2:
        m1, m2 = dr
        filt = (pd.to_datetime(raw["date"]).dt.date >= m1) & (pd.to_datetime(raw["date"]).dt.date <= m2)
        tx = raw.loc[filt].copy()
    else:
        tx = raw.copy()

    c1, c2, c3 = st.columns(3)
    with c1:
        if tx["category"].astype(str).str.len().sum() > 0:
            _plot_with_caption(
                ch.plot_treemap,
                "Treemap groups spend—double-click a slice in Plotly to zoom when many labels overlap.",
                tx,
                "category",
                "amount",
                title="Categories",
            )
        else:
            st.info("Add a category column to unlock the category treemap.")
    with c2:
        m = tx.assign(merchant=tx["category"].where(tx["category"].astype(str).str.strip().ne(""), "Unknown"))
        m["_a"] = m["amount"].abs()
        g = m.groupby("merchant", as_index=False)["_a"].sum().rename(columns={"_a": "val"})
        _plot_with_caption(
            ch.plot_horizontal_bars,
            "Longest bars are labels (category or merchant) that absorbed the most dollars in your filter.",
            g,
            "merchant",
            "val",
            title="Top labels by dollars",
        )
    with c3:
        if tx["category"].astype(str).str.len().sum() > 0:
            cnt = tx.groupby("category", as_index=False).size()
            cnt.columns = ["cat", "n"]
            _plot_with_caption(
                ch.plot_donut,
                "This counts transactions, not dollars—use it to see noisy categories you swipe in often.",
                cnt["cat"].tolist(),
                cnt["n"].astype(float).tolist(),
                title="Transaction counts",
            )

    section_header("📅", "When do you spend?")
    u, v = st.columns(2)
    with u:
        _plot_with_caption(
            ch.plot_heatmap_calendar,
            "Darker cells mean more dollars that calendar day; look for streaks (weekends, pay cycles).",
            tx,
            "date",
            "amount",
            title="Calendar heatmap",
        )
    with v:
        _plot_with_caption(
            ch.plot_day_of_week,
            "Taller bars mean higher average dollars on that weekday—useful for weekly wallet checks.",
            tx,
            "date",
            "amount",
        )

    section_header("💳", "Transactions & amounts")
    x, y = st.columns(2)
    with x:
        _plot_with_caption(
            ch.plot_histogram,
            "The shape shows typical purchase sizes—watch a long tail to the right for big occasional hits.",
            tx,
            "amount",
        )
    with y:
        _html('<div class="chart-wrap">')
        _html('<div class="insight-title" style="padding:8px 12px;">Top 10 transactions (by size)</div>')
        dd = tx.assign(abs_amt=tx["amount"].abs()).sort_values("abs_amt", ascending=False).head(10)
        show = dd[["date", "category", "amount"]].copy()
        show["date"] = pd.to_datetime(show["date"]).dt.strftime("%Y-%m-%d")
        show["amount"] = show["amount"].map(lambda v: fmt_money(float(v)))
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption("Large one-off rows here deserve a second look—subscriptions, travel, or fees often hide in the top few lines.")
        _html("</div>")

    section_header("⚠️", "Months that look unusually high")
    dm = st.session_state.df_monthly
    yv = dm["y"].to_numpy(dtype=float)
    mu, sd = float(np.mean(yv)), float(np.std(yv))
    thr = mu + 1.5 * sd
    bad = dm[dm["y"] > thr]
    if bad.empty:
        _html(
            '<div class="alert-good">No months stood out as extreme spikes versus your typical rhythm.</div>',
        )
    else:
        for _, r in bad.iterrows():
            pct = (float(r["y"]) - mu) / (mu + 1e-9) * 100.0
            _html(
                f'<div class="alert-warn"><b>{pd.Timestamp(r["ds"]).strftime("%Y-%m")}</b> — '
                f'{fmt_money(float(r["y"]))} ({pct:+.0f}% vs your typical month)</div>',
            )


def page_forecast() -> None:
    if not _has_data():
        _empty_state()
        return
    st.session_state.setdefault("fc_view", "compare")
    st.session_state.setdefault("fc_horizon", 6)
    st.session_state.setdefault("fc_ci", 0.95)
    st.session_state.setdefault("fc_season", "auto")
    st.session_state.setdefault("fc_whatif", 0.0)

    tips_box(
        "Forecast tips",
        [
            "Run forecast after uploading data; change horizon and seasonality, then tap Run forecast.",
            "Compare Prophet vs ARIMA on the Compare view—Prophet handles seasonality, ARIMA recent momentum.",
            "Use what-if slider to stress-test a spending increase before it happens.",
        ],
    )

    section_header("🔮", "Forecast lab")
    view = st.radio("Pick a model view", ["Prophet", "ARIMA", "Compare both"], horizontal=True)
    st.session_state.fc_view = {"Prophet": "prophet", "ARIMA": "arima", "Compare both": "compare"}[view]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.fc_horizon = st.slider("Months ahead", 3, 24, int(st.session_state.fc_horizon))
    with c2:
        ci_label = st.selectbox("Forecast band width", ["80%", "90%", "95%"], index=2)
        st.session_state.fc_ci = {"80%": 0.80, "90%": 0.90, "95%": 0.95}[ci_label]
    with c3:
        seas = st.selectbox("Seasonality feel", ["Auto", "Weekly", "Monthly", "Yearly"], index=0)
        st.session_state.fc_season = {"Auto": "auto", "Weekly": "weekly", "Monthly": "monthly", "Yearly": "yearly"}[seas]

    st.session_state.fc_whatif = st.slider("What-if: nudge history up/down (%)", -30, 30, int(st.session_state.fc_whatif))

    cur_fp = _forecast_fingerprint()
    old_fp = st.session_state.get("fc_fingerprint")
    _clear_forecasts_if_stale()

    dm = st.session_state.df_monthly.copy()
    dm["y"] = dm["y"].astype(float) * (1.0 + st.session_state.fc_whatif / 100.0)
    kw = dict(interval_width=float(st.session_state.fc_ci), seasonality=st.session_state.fc_season)  # type: ignore[arg-type]

    run_clicked = st.button("Run forecast", type="primary")
    need_run = run_clicked or (old_fp != cur_fp) or (st.session_state.fc_prophet is None)
    if need_run:
        with st.spinner("Running forecast model... 🔮"):
            try:
                p_df, _ = get_forecast(dm, "prophet", st.session_state.fc_horizon, **kw)
                a_df, _ = get_forecast(dm, "arima", st.session_state.fc_horizon, **kw)
                st.session_state.fc_prophet = p_df
                st.session_state.fc_arima = a_df
                st.session_state.forecast_df = p_df
                st.session_state.fc_fingerprint = cur_fp
            except Exception as e:
                st.error(str(e))

    p_df = st.session_state.fc_prophet
    a_df = st.session_state.fc_arima
    if p_df is None or a_df is None:
        st.info("Tap **Run forecast** to generate the next months and chart.")
        return

    if st.session_state.fc_view == "compare":
        _plot_with_caption(
            ch.plot_forecast,
            "Purple band = likely range; solid purple line = middle guess; dashed line marks where history ends and forecast begins.",
            dm,
            p_df,
            title="Forecast",
            forecast_arima=a_df,
        )
    elif st.session_state.fc_view == "prophet":
        _plot_with_caption(
            ch.plot_forecast,
            "Same reading as above but only the primary model curve.",
            dm,
            p_df,
            title="Prophet",
            forecast_arima=None,
        )
    else:
        _plot_with_caption(
            ch.plot_forecast,
            "ARIMA focuses on recent momentum—compare with Prophet in **Compare both**.",
            dm,
            a_df,
            title="ARIMA",
            forecast_arima=None,
        )

    mcols = st.columns(4)
    mae_p, rmse_p, mape_p, r2_p = _accuracy_block(dm, "prophet", **kw)
    mae_a, rmse_a, mape_a, r2_a = _accuracy_block(dm, "arima", **kw)
    use_mae, use_rmse, use_mape, use_r2 = (
        (mae_p, rmse_p, mape_p, r2_p) if st.session_state.fc_view != "arima" else (mae_a, rmse_a, mape_a, r2_a)
    )
    labels_plain = [
        "Average prediction error ($)",
        "Typical big miss ($)",
        "Typical % gap vs actual (test months)",
        "How well the curve hugs history (0–1)",
    ]
    vals = [use_mae, use_rmse, use_mape, use_r2]
    fc_colors = ["#6C63FF", "#00B894", "#FFB347", "#8B5CF6"]
    for lab, val, col, color in zip(labels_plain, vals, mcols, fc_colors):
        with col:
            if val is None:
                vv = "—"
            elif "% gap" in lab.lower():
                vv = f"{float(val):.1f}%"
            elif "0–1" in lab or "(0–1)" in lab:
                vv = f"{float(val):.3f}"
            else:
                vv = fmt_money(float(val))
            kpi_card(lab, vv, "", color, True)

    st.caption(
        f"What-if applies {st.session_state.fc_whatif:+.0f}% to history before training. "
        f"Rough next-6-month middle forecast (Prophet): {fmt_money(float(p_df['yhat'].head(6).sum()))}."
    )

    section_header("📈", "Seasonal pattern (STL)")
    with st.spinner("Building seasonal view... 💡"):
        try:
            dec = stl_decomposition_monthly(dm)
            _plot_with_caption(
                ch.plot_decomposition,
                "Trend = slow drift, seasonal = repeating month pattern, residual = what is left—big residual spikes are surprises.",
                dec,
            )
        except Exception as e:
            st.info(f"Seasonal view unavailable: {e}")

    with st.expander("Forecast table"):
        show = p_df.copy()
        if st.session_state.fc_view == "compare":
            show = show.merge(a_df.rename(columns={"yhat": "yhat_arima"}), on="ds", how="outer")
        st.dataframe(show, use_container_width=True)


def page_ai() -> None:
    if not _has_data():
        _empty_state()
        return
    with st.spinner("Building insights... 💡"):
        s = st.session_state.summary or {}
        dm = st.session_state.df_monthly
        raw = st.session_state.df_raw
        y = dm["y"].to_numpy(dtype=float)
        score = 50.0
        tips_box(
            "How to use insights",
            [
                "Health score blends trend, consistency, and surprise months—not a credit score.",
                "Act on one Pay attention item per week for steady progress.",
                "Correlation chart only appears when your file has multiple numeric columns.",
            ],
        )

        if len(y) >= 2:
            slope = np.polyfit(np.arange(len(y)), y, 1)[0]
            t_pts = 25 * (1.0 if slope <= 0 else max(0.0, 1.0 - min(slope / (np.mean(y) + 1e-9), 1.0)))
            cv = float(np.std(y) / (np.mean(y) + 1e-9))
            c_pts = max(0.0, 25 - min(cv * 80, 25))
            mu = float(np.mean(y))
            sd = float(np.std(y)) or 1.0
            ano = int(np.sum(y > mu + 1.5 * sd))
            a_pts = max(0.0, 25 - min(ano * 8, 25))
            b_pts = 25.0
            score = float(np.clip(t_pts + c_pts + a_pts + b_pts, 0, 100))

        _plot_with_caption(
            ch.plot_gauge,
            "Green zone = comfortable mix of steadiness; orange and pink mean volatility or surprise months showed up.",
            score,
            "Financial health score",
            max_val=100.0,
        )

        section_header("💡", "Plain-language snapshot")
        variation_txt = fmt_money(float(s.get("std", 0)))
        insights = [
            ("📈", "Direction", f"Trend is **{html.escape(str(s.get('trend','')))}** with **{fmt_pct(float(s.get('pct_change')))}** comparing first vs last three-month averages."),
            ("🔺", "Largest month", f"**{html.escape(str(s.get('max_month','')))}** reached **{fmt_money(float(s.get('max_val',0)))}**."),
            ("🏷️", "Most common label", f"**{html.escape(str(s.get('most_common_category') or 'n/a'))}**."),
            ("🔁", "Activity", f"**{int(s.get('total_transactions',0))}** transactions; about **{fmt_money(float(s.get('avg_per_transaction',0)))}** per swipe on average."),
            ("📉", "Month-to-month swing", f"Typical up/down swing in monthly totals: **{variation_txt}** (how much your spending varies month to month)."),
            ("💡", "Quick win", "Trim ~10% from your biggest category for an easy savings lever."),
        ]
        for icon, title, body in insights:
            _html(
                f'<div class="insight-item"><div class="insight-title">{icon} {html.escape(title)}</div>'
                f'<div class="insight-desc">{body}</div></div>',
            )

        section_header("💡", "Ideas to try")
        _html(
            '<div class="insight-item warning"><div class="insight-title">Pay attention</div>'
            '<div class="insight-desc">Review your highest month and set a simple monthly ceiling.</div></div>',
        )
        _html(
            '<div class="insight-item warning"><div class="insight-title">Watch</div>'
            '<div class="insight-desc">Each Friday, compare week-to-date spend vs last week.</div></div>',
        )
        _html(
            '<div class="insight-item success"><div class="insight-title">Low effort</div>'
            '<div class="insight-desc">Pick one subscription to audit—often a few dollars hide there.</div></div>',
        )

        num = raw.select_dtypes(include=[np.number])
        if num.shape[1] >= 2:
            _plot_with_caption(
                ch.plot_correlation,
                "Darker squares show pairs of numeric columns that move together in your export.",
                num,
                title="Number relationships",
            )

        section_header("✨", "Did you know?")
        days = (pd.to_datetime(raw["date"]).max() - pd.to_datetime(raw["date"]).min()).days + 1
        st.write(f"- Per day over your date window: **{fmt_money(float(raw['amount'].abs().sum() / max(days, 1)))}**")
        raw2 = raw.copy()
        raw2["_dow"] = pd.to_datetime(raw2["date"]).dt.day_name()
        topd = raw2.groupby("_dow")["amount"].apply(lambda x: x.abs().sum()).idxmax()
        st.write(f"- Busiest weekday by total dollars: **{topd}**")


def page_export() -> None:
    if not _has_data():
        _empty_state()
        return
    tips_box(
        "Export guide",
        [
            "Monthly CSV is best for spreadsheets; Excel bundles all sheets for sharing.",
            "Run Forecast first if you want forecast rows in the Excel workbook.",
            "Insights TXT is a plain summary you can paste into email or notes.",
        ],
    )

    section_header("📥", "Export center")
    c1, c2 = st.columns(2)
    with c1:
        section_header("📄", "Monthly CSV")
        st.download_button(
            "Download monthly CSV",
            st.session_state.df_monthly.to_csv(index=False).encode("utf-8"),
            file_name="flowcast_monthly.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        section_header("🔮", "Forecast CSV")
        if st.session_state.fc_prophet is not None:
            st.download_button(
                "Download forecast CSV",
                st.session_state.fc_prophet.to_csv(index=False).encode("utf-8"),
                file_name="flowcast_forecast.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.caption("Run a forecast first on the Forecast page.")

    c3, c4 = st.columns(2)
    with c3:
        section_header("💡", "Insights text file")
        lines = [f"{k}: {v}" for k, v in (st.session_state.summary or {}).items()]
        st.download_button("Download insights TXT", "\n".join(lines).encode("utf-8"), file_name="insights.txt", use_container_width=True)
    with c4:
        section_header("📊", "Excel workbook")
        buf = io.BytesIO()
        try:
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                st.session_state.df_monthly.to_excel(w, sheet_name="Monthly", index=False)
                st.session_state.df_raw.to_excel(w, sheet_name="Transactions", index=False)
                if st.session_state.fc_prophet is not None:
                    st.session_state.fc_prophet.to_excel(w, sheet_name="Forecast", index=False)
            st.download_button(
                "Download Excel",
                buf.getvalue(),
                file_name="flowcast_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Excel export needs openpyxl: {e}")


def main() -> None:
    _init_state()
    page = st.session_state.page
    with st.sidebar:
        _sidebar()
    _inject_css(page)
    render_top_nav()
    if page == "home":
        page_home()
    else:
        render_page_intro(page)
        if page == "dashboard":
            page_dashboard()
        elif page == "spending":
            page_spending()
        elif page == "forecast":
            page_forecast()
        elif page == "ai":
            page_ai()
        else:
            page_export()


if __name__ == "__main__":
    main()
