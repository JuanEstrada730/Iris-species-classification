# ============================================================
#  Proyect.py  —  Iris Species Classification Dashboard
#  Universidad de la Costa | Data Mining
#  Profesor: José Escorcia-Gutierrez, Ph.D.
#  Departamento de Ciencias de la Computación y Electrónica
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Classification | CUC Data Mining",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ──────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background: #f5f7fa; }
[data-testid="stSidebar"]          { background: #1a2332; }
[data-testid="stSidebar"] *        { color: #e2e8f0 !important; }
[data-testid="stSidebar"] hr       { border-color: #334155 !important; }

/* ── Metric cards ───────────────────────────────────────────── */
.metric-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 18px 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
    display: flex;
    align-items: center;
    gap: 16px;
    border-left: 4px solid;
    margin-bottom: 4px;
}
.metric-card.blue   { border-color: #3b82f6; }
.metric-card.green  { border-color: #10b981; }
.metric-card.amber  { border-color: #f59e0b; }
.metric-card.purple { border-color: #8b5cf6; }
.metric-card.slate  { border-color: #64748b; }

.metric-icon svg { width: 34px; height: 34px; stroke-width: 1.7; fill: none; }
.metric-card.blue   .metric-icon svg { stroke: #3b82f6; }
.metric-card.green  .metric-icon svg { stroke: #10b981; }
.metric-card.amber  .metric-icon svg { stroke: #f59e0b; }
.metric-card.purple .metric-icon svg { stroke: #8b5cf6; }
.metric-card.slate  .metric-icon svg { stroke: #64748b; }

.metric-label {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .8px; color: #64748b;
}
.metric-value { font-size: 26px; font-weight: 700; color: #1e293b; line-height: 1.1; }

/* ── Section headers ────────────────────────────────────────── */
.section-header {
    display: flex; align-items: center; gap: 10px;
    border-bottom: 2px solid #e2e8f0; padding-bottom: 8px;
    margin: 24px 0 16px;
}
.section-header svg { width: 20px; height: 20px; stroke: #3b82f6;
                       stroke-width: 1.8; fill: none; }
.section-header h3  { margin: 0; font-size: 16px; color: #1e293b; font-weight: 700; }

/* ── Species result box ─────────────────────────────────────── */
.species-result {
    border-radius: 12px; padding: 22px 24px; text-align: center;
    margin: 8px 0; color: #ffffff;
}
.species-name    { font-size: 24px; font-weight: 700; font-style: italic; }
.confidence-text { font-size: 13px; opacity: .85; margin-top: 4px; }

/* ── Probability bars ───────────────────────────────────────── */
.prob-row     { margin: 7px 0; }
.prob-row-top { display: flex; justify-content: space-between; }
.prob-label   { font-size: 12px; color: #475569; margin-bottom: 2px; }
.prob-pct     { font-size: 12px; font-weight: 600; color: #1e293b; }
.prob-bar-bg  { background: #e2e8f0; border-radius: 6px; height: 9px; }
.prob-bar     { border-radius: 6px; height: 9px; }

/* ── Info banners ───────────────────────────────────────────── */
.info-blue   { background:#eff6ff; border-left:4px solid #3b82f6; padding:11px 14px;
               border-radius:0 8px 8px 0; font-size:13px; color:#1e40af; margin:12px 0; }
.info-green  { background:#f0fdf4; border-left:4px solid #10b981; padding:11px 14px;
               border-radius:0 8px 8px 0; font-size:13px; color:#065f46; margin:12px 0; }
.info-purple { background:#faf5ff; border-left:4px solid #8b5cf6; padding:11px 14px;
               border-radius:0 8px 8px 0; font-size:13px; color:#5b21b6; margin:12px 0; }

/* ── Page title ─────────────────────────────────────────────── */
.page-title { color: #1e293b; font-size: 22px; font-weight: 700; margin: 0; }
.page-sub   { color: #64748b; font-size: 14px; margin-top: 2px; margin-bottom: 20px; }

/* ── Hide Streamlit chrome ──────────────────────────────────── */
footer    { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SVG ICON LIBRARY
# ─────────────────────────────────────────────────────────────────────────────
ICONS = {
    # Bullseye  → Accuracy
    "accuracy": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
        <circle cx="12" cy="12" r="10"/>
        <circle cx="12" cy="12" r="6"/>
        <circle cx="12" cy="12" r="2"/>
    </svg>""",
    # Funnel    → Precision
    "precision": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
        <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>
    </svg>""",
    # Refresh   → Recall
    "recall": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
        <polyline points="1 4 1 10 7 10"/>
        <path d="M3.51 15a9 9 0 1 0 .49-3.96"/>
    </svg>""",
    # Bar chart → F1
    "f1": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
        <line x1="18" y1="20" x2="18" y2="10"/>
        <line x1="12" y1="20" x2="12" y2="4"/>
        <line x1="6"  y1="20" x2="6"  y2="14"/>
        <line x1="2"  y1="20" x2="22" y2="20"/>
    </svg>""",
    # Database  → data
    "data": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
        <ellipse cx="12" cy="5" rx="9" ry="3"/>
        <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>
        <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
    </svg>""",
    # CPU       → model
    "model": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
        <rect x="4" y="4" width="16" height="16" rx="2"/>
        <rect x="9" y="9" width="6"  height="6"/>
        <line x1="9"  y1="1"  x2="9"  y2="4"/>
        <line x1="15" y1="1"  x2="15" y2="4"/>
        <line x1="9"  y1="20" x2="9"  y2="23"/>
        <line x1="15" y1="20" x2="15" y2="23"/>
        <line x1="20" y1="9"  x2="23" y2="9"/>
        <line x1="20" y1="14" x2="23" y2="14"/>
        <line x1="1"  y1="9"  x2="4"  y2="9"/>
        <line x1="1"  y1="14" x2="4"  y2="14"/>
    </svg>""",
    # Search    → predict
    "predict": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
        <circle cx="11" cy="11" r="8"/>
        <line x1="21" y1="21" x2="16.65" y2="16.65"/>
    </svg>""",
    # Compass   → explore
    "explore": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
        <circle cx="12" cy="12" r="10"/>
        <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>
    </svg>""",
    # Layers    → chart / eda
    "chart": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
        <polygon points="12 2 2 7 12 12 22 7 12 2"/>
        <polyline points="2 17 12 22 22 17"/>
        <polyline points="2 12 12 17 22 12"/>
    </svg>""",
    # Grid      → samples
    "samples": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
        <rect x="3"  y="3"  width="7" height="7"/>
        <rect x="14" y="3"  width="7" height="7"/>
        <rect x="14" y="14" width="7" height="7"/>
        <rect x="3"  y="14" width="7" height="7"/>
    </svg>""",
    # Checkmark → no-missing
    "check": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
        <polyline points="22 4 12 14.01 9 11.01"/>
    </svg>""",
}

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    "Sepal Length (cm)", "Sepal Width (cm)",
    "Petal Length (cm)", "Petal Width (cm)",
]
TARGET_NAMES  = ["Iris setosa", "Iris versicolor", "Iris virginica"]
SPECIES_COLORS = {
    "Iris setosa":     "#3b82f6",
    "Iris versicolor": "#10b981",
    "Iris virginica":  "#f59e0b",
}

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def section_header(icon_key: str, title: str):
    st.markdown(f"""
    <div class="section-header">
        <span class="metric-icon">{ICONS[icon_key]}</span>
        <h3>{title}</h3>
    </div>""", unsafe_allow_html=True)


def metric_card(color: str, icon_key: str, label: str, value: str) -> str:
    return f"""
    <div class="metric-card {color}">
        <div class="metric-icon">{ICONS[icon_key]}</div>
        <div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
    </div>"""


def apply_layout(fig: go.Figure, height: int = 400):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#1e293b",
        font_family="'Inter', 'Segoe UI', sans-serif",
        margin=dict(t=30, b=20, l=20, r=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA & MODEL  (cached so they only run once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df   = iris.frame.copy()
    df.columns = FEATURES + ["Species"]
    df["Species Name"] = df["Species"].map(dict(enumerate(TARGET_NAMES)))
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df[FEATURES].values
    y = df["Species"].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=None,
        min_samples_split=2, min_samples_leaf=1,
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_sc    = cross_val_score(clf, X_sc, y, cv=cv, scoring="accuracy")

    metrics = {
        "accuracy":  accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, average="weighted"),
        "recall":    recall_score(y_te, y_pred, average="weighted"),
        "f1":        f1_score(y_te, y_pred, average="weighted"),
        "cv_mean":   cv_sc.mean(),
        "cv_std":    cv_sc.std(),
    }
    cm       = confusion_matrix(y_te, y_pred)
    feat_imp = clf.feature_importances_
    return clf, scaler, metrics, cm, feat_imp


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:20px 0 10px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="58" height="58" viewBox="0 0 60 60">
                <circle cx="30" cy="30" r="7" fill="#8b5cf6"/>
                <ellipse cx="30" cy="13" rx="5.5" ry="9" fill="#a78bfa" opacity=".85"/>
                <ellipse cx="30" cy="47" rx="5.5" ry="9" fill="#a78bfa" opacity=".85"/>
                <ellipse cx="13" cy="30" rx="9" ry="5.5" fill="#a78bfa" opacity=".85"/>
                <ellipse cx="47" cy="30" rx="9" ry="5.5" fill="#a78bfa" opacity=".85"/>
                <ellipse cx="18" cy="18" rx="5.5" ry="9"
                         transform="rotate(45 18 18)"  fill="#c4b5fd" opacity=".7"/>
                <ellipse cx="42" cy="18" rx="5.5" ry="9"
                         transform="rotate(-45 42 18)" fill="#c4b5fd" opacity=".7"/>
                <ellipse cx="18" cy="42" rx="5.5" ry="9"
                         transform="rotate(-45 18 42)" fill="#c4b5fd" opacity=".7"/>
                <ellipse cx="42" cy="42" rx="5.5" ry="9"
                         transform="rotate(45 42 42)"  fill="#c4b5fd" opacity=".7"/>
            </svg>
            <div style="font-size:16px; font-weight:700; color:#e2e8f0; margin-top:8px;">
                Iris Classification
            </div>
            <div style="font-size:11px; color:#94a3b8; margin-top:2px;">
                Data Mining &nbsp;&middot;&nbsp; CUC
            </div>
        </div>
        <hr/>
        <div style="font-size:10px; font-weight:700; color:#94a3b8;
                    letter-spacing:.7px; margin-bottom:6px;">NAVIGATION</div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "nav",
            ["Data Overview", "Exploratory Analysis",
             "Model Performance", "Prediction"],
            label_visibility="collapsed",
        )

        st.markdown("""<hr/>
        <div style="font-size:10px; font-weight:700; color:#94a3b8;
                    letter-spacing:.7px; margin-bottom:6px;">ALGORITHM</div>
        <div style="font-size:12px; color:#cbd5e1; line-height:1.8;">
            Random Forest Classifier<br/>
            <span style="color:#64748b;">n_estimators = 200<br/>
            max_depth = None<br/>
            Criterion = Gini impurity</span>
        </div>
        <hr/>
        <div style="font-size:10px; font-weight:700; color:#94a3b8;
                    letter-spacing:.7px; margin-bottom:6px;">VALIDATION</div>
        <div style="font-size:12px; color:#cbd5e1; line-height:1.8;">
            Stratified K-Fold (k = 5)<br/>
            <span style="color:#64748b;">Test split = 25 %<br/>
            random_state = 42</span>
        </div>
        <hr/>
        <div style="font-size:10px; color:#475569; line-height:1.6;">
            Universidad de la Costa<br/>
            Ing. de Sistemas &mdash; Semestre VI<br/>
            Minería de Datos &middot; 2025-I
        </div>
        """, unsafe_allow_html=True)

    return page


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 1 — DATA OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
def page_data_overview(df: pd.DataFrame):
    st.markdown('<p class="page-title">Data Understanding</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Structural overview of the Iris dataset: '
        'descriptive statistics, class balance, and feature distributions per species.</p>',
        unsafe_allow_html=True,
    )

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    for col, (clr, ico, lbl, val) in zip(
        [c1, c2, c3, c4],
        [("blue",   "samples", "Total Samples", "150"),
         ("green",  "explore", "Features",      "4"),
         ("amber",  "model",   "Target Classes","3"),
         ("purple", "check",   "Missing Values","0")],
    ):
        with col:
            st.markdown(metric_card(clr, ico, lbl, val), unsafe_allow_html=True)

    # Descriptive statistics
    section_header("data", "Descriptive Statistics")
    st.dataframe(
        df[FEATURES].describe().round(4)
        .style
        .format("{:.4f}"),
        use_container_width=True,
    )

    # Class balance
    section_header("chart", "Class Distribution")
    balance = df["Species Name"].value_counts().reset_index()
    balance.columns = ["Species", "Count"]
    fig_bal = px.bar(
        balance, x="Species", y="Count",
        color="Species", color_discrete_map=SPECIES_COLORS,
        text="Count",
    )
    fig_bal.update_traces(textposition="outside")
    fig_bal.update_layout(showlegend=False, xaxis_title="", yaxis_title="Samples")
    apply_layout(fig_bal, height=300)
    st.plotly_chart(fig_bal, use_container_width=True)

    # Feature distributions — histogram
    section_header("chart", "Feature Distributions by Species")
    fig_hist = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f.replace(" (cm)", "") for f in FEATURES],
        vertical_spacing=0.16, horizontal_spacing=0.10,
    )
    for i, feat in enumerate(FEATURES):
        r, c = divmod(i, 2)
        for sp, clr in SPECIES_COLORS.items():
            vals = df[df["Species Name"] == sp][feat]
            fig_hist.add_trace(
                go.Histogram(
                    x=vals, name=sp, marker_color=clr,
                    opacity=0.70, showlegend=(i == 0),
                    histnorm="probability density", nbinsx=14,
                ),
                row=r + 1, col=c + 1,
            )
    apply_layout(fig_hist, height=490)
    fig_hist.update_layout(
        barmode="overlay",
        legend=dict(orientation="h", y=1.06, x=0.5,
                    xanchor="center", font_size=12),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Box plots
    section_header("chart", "Box Plots by Species")
    fig_box = make_subplots(
        rows=1, cols=4,
        subplot_titles=[f.replace(" (cm)", "") for f in FEATURES],
        horizontal_spacing=0.06,
    )
    for i, feat in enumerate(FEATURES):
        for sp, clr in SPECIES_COLORS.items():
            vals = df[df["Species Name"] == sp][feat]
            fig_box.add_trace(
                go.Box(y=vals, name=sp, marker_color=clr,
                       showlegend=(i == 0), boxmean=True),
                row=1, col=i + 1,
            )
    apply_layout(fig_box, height=390)
    fig_box.update_layout(
        boxmode="group",
        legend=dict(orientation="h", y=1.06, x=0.5,
                    xanchor="center", font_size=12),
    )
    st.plotly_chart(fig_box, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 2 — EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def page_eda(df: pd.DataFrame):
    st.markdown('<p class="page-title">Exploratory Data Analysis</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Correlation structure and multi-dimensional feature '
        'relationships that reveal discriminative patterns across the three species.</p>',
        unsafe_allow_html=True,
    )

    # Correlation heatmap
    section_header("explore", "Pearson Correlation Matrix")
    corr  = df[FEATURES].corr()
    short = [f.replace(" (cm)", "") for f in FEATURES]
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=short, y=short,
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}", textfont_size=14,
    ))
    apply_layout(fig_corr, height=360)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown(
        '<div class="info-blue"><strong>Observation:</strong> Petal length and petal width '
        'exhibit a strong positive correlation (r&nbsp;&asymp;&nbsp;0.96), making them the '
        'most discriminative features for classification. Sepal width shows a weak or '
        'negative correlation with the remaining variables.</div>',
        unsafe_allow_html=True,
    )

    # Scatter matrix
    section_header("chart", "Scatter Matrix (Pair Plot)")
    fig_sm = px.scatter_matrix(
        df, dimensions=FEATURES, color="Species Name",
        color_discrete_map=SPECIES_COLORS, opacity=0.72,
        labels={f: f.replace(" (cm)", "") for f in FEATURES},
    )
    fig_sm.update_traces(diagonal_visible=False, marker_size=4)
    apply_layout(fig_sm, height=590)
    fig_sm.update_layout(legend=dict(title="Species", font_size=12))
    st.plotly_chart(fig_sm, use_container_width=True)

    # Interactive 2D scatter
    section_header("explore", "Interactive 2D Scatter with Marginals")
    ca, cb = st.columns(2)
    with ca:
        x_ax = st.selectbox("X axis", FEATURES, index=2, key="eda_x")
    with cb:
        y_ax = st.selectbox("Y axis", FEATURES, index=3, key="eda_y")

    fig_2d = px.scatter(
        df, x=x_ax, y=y_ax, color="Species Name",
        color_discrete_map=SPECIES_COLORS, symbol="Species Name",
        opacity=0.85, marginal_x="histogram", marginal_y="box",
    )
    fig_2d.update_traces(marker_size=7, selector=dict(type="scatter"))
    apply_layout(fig_2d, height=510)
    st.plotly_chart(fig_2d, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
def page_model(df: pd.DataFrame, clf, scaler, metrics: dict, cm, feat_imp):
    st.markdown('<p class="page-title">Model Performance</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Evaluation of the Random Forest classifier — '
        'stratified 75/25 train-test split and 5-fold cross-validation.</p>',
        unsafe_allow_html=True,
    )

    # Metric cards
    section_header("model", "Evaluation Metrics")
    c1, c2, c3, c4 = st.columns(4)
    for col, (clr, ico, lbl, key) in zip(
        [c1, c2, c3, c4],
        [("blue",   "accuracy",  "Accuracy",  "accuracy"),
         ("green",  "precision", "Precision", "precision"),
         ("amber",  "recall",    "Recall",    "recall"),
         ("purple", "f1",        "F1 Score",  "f1")],
    ):
        with col:
            st.markdown(
                metric_card(clr, ico, lbl, f"{metrics[key]:.4f}"),
                unsafe_allow_html=True,
            )

    st.markdown(
        f'<div class="info-green"><strong>Cross-Validation (5-fold):</strong> '
        f'Mean Accuracy = <strong>{metrics["cv_mean"]:.4f}</strong> '
        f'&nbsp;&plusmn;&nbsp; {metrics["cv_std"]:.4f} &mdash; '
        f'the low standard deviation confirms stable generalisation across all folds.</div>',
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns(2)

    # Confusion matrix
    with col_l:
        section_header("chart", "Confusion Matrix")
        labels_s = ["Setosa", "Versicolor", "Virginica"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=labels_s, y=labels_s,
            colorscale=[[0, "#eff6ff"], [1, "#1e40af"]],
            text=cm, texttemplate="<b>%{text}</b>",
            textfont_size=20, showscale=False,
        ))
        fig_cm.update_layout(
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
        )
        apply_layout(fig_cm, height=340)
        st.plotly_chart(fig_cm, use_container_width=True)

    # Feature importance
    with col_r:
        section_header("chart", "Feature Importance (Mean Gini Decrease)")
        feat_short = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
        order      = np.argsort(feat_imp)
        bar_colors = ["#3b82f6" if i == order[-1] else "#93c5fd" for i in range(4)]
        fig_fi = go.Figure(go.Bar(
            x=feat_imp[order],
            y=[feat_short[i] for i in order],
            orientation="h",
            marker_color=[bar_colors[i] for i in order],
            text=[f"{feat_imp[i]:.4f}" for i in order],
            textposition="outside",
        ))
        fig_fi.update_xaxes(
            range=[0, max(feat_imp) * 1.28],
            showgrid=True, gridcolor="#f1f5f9",
        )
        fig_fi.update_layout(
            xaxis_title="Mean Decrease in Gini Impurity",
            showlegend=False,
        )
        apply_layout(fig_fi, height=340)
        st.plotly_chart(fig_fi, use_container_width=True)

    # Per-class breakdown
    section_header("model", "Per-Class Performance Breakdown")
    X_sc = scaler.transform(df[FEATURES].values)
    _, X_te, _, y_te = train_test_split(
        X_sc, df["Species"].values, test_size=0.25,
        random_state=42, stratify=df["Species"].values,
    )
    y_pred = clf.predict(X_te)

    rows = []
    for i, name in enumerate(TARGET_NAMES):
        rows.append({
            "Species":   name,
            "Precision": precision_score(y_te, y_pred, average=None)[i],
            "Recall":    recall_score(y_te, y_pred, average=None)[i],
            "F1 Score":  f1_score(y_te, y_pred, average=None)[i],
            "Support":   int((y_te == i).sum()),
        })
    per_df = pd.DataFrame(rows)
    st.dataframe(
        per_df.style
            .format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1 Score": "{:.4f}"}),
        use_container_width=True, hide_index=True,
    )

    # Radar chart
    section_header("explore", "Metrics Radar — Per Species")
    categories = ["Precision", "Recall", "F1 Score"]
    fig_radar  = go.Figure()
    for row, clr in zip(rows, SPECIES_COLORS.values()):
        vals = [row[c] for c in categories] + [row[categories[0]]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=categories + [categories[0]],
            fill="toself", name=row["Species"],
            line_color=clr, fillcolor=clr, opacity=0.25,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        legend=dict(font_size=12),
    )
    apply_layout(fig_radar, height=420)
    st.plotly_chart(fig_radar, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 4 — PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def page_prediction(df: pd.DataFrame, clf, scaler):
    st.markdown('<p class="page-title">Species Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Input four flower measurements to obtain a real-time '
        'species prediction, class probabilities, and a 3D visualisation of the '
        'sample relative to the full dataset.</p>',
        unsafe_allow_html=True,
    )

    col_in, col_res = st.columns([1, 1.8])

    with col_in:
        section_header("predict", "Flower Measurements")
        sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
        sw = st.slider("Sepal Width (cm)",  1.5, 5.0, 3.0, 0.1)
        pl = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
        pw = st.slider("Petal Width (cm)",  0.1, 2.6, 1.2, 0.1)
        st.button("Run Prediction", type="primary", use_container_width=True)

    # Live prediction (updates on every slider move)
    sample = np.array([[sl, sw, pl, pw]])
    pred   = clf.predict(scaler.transform(sample))[0]
    proba  = clf.predict_proba(scaler.transform(sample))[0]

    sp_name  = TARGET_NAMES[pred]
    sp_color = list(SPECIES_COLORS.values())[pred]
    conf     = proba[pred]

    with col_res:
        section_header("model", "Prediction Result")
        st.markdown(f"""
        <div class="species-result"
             style="background: linear-gradient(135deg, {sp_color}bb 0%, {sp_color} 100%);">
            <div style="font-size:11px; opacity:.8; letter-spacing:1px;
                        text-transform:uppercase; margin-bottom:4px;">
                Predicted Species
            </div>
            <div class="species-name"><em>{sp_name}</em></div>
            <div class="confidence-text">Confidence: {conf:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:14px;'>", unsafe_allow_html=True)
        for i, (name, p) in enumerate(zip(TARGET_NAMES, proba)):
            bar_clr = list(SPECIES_COLORS.values())[i]
            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-row-top">
                    <span class="prob-label"><em>{name}</em></span>
                    <span class="prob-pct">{p:.1%}</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar"
                         style="width:{p*100:.1f}%; background:{bar_clr};"></div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 3D scatter
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("chart", "3D Feature Space — Sample Position")

    c3a, c3b, c3c = st.columns(3)
    with c3a:
        ax_x = st.selectbox("X axis", FEATURES, index=2, key="p3x")
    with c3b:
        ax_y = st.selectbox("Y axis", FEATURES, index=3, key="p3y")
    with c3c:
        ax_z = st.selectbox("Z axis", FEATURES, index=0, key="p3z")

    inp = {FEATURES[0]: sl, FEATURES[1]: sw, FEATURES[2]: pl, FEATURES[3]: pw}

    fig_3d = go.Figure()
    for sp_nm, sp_cl in SPECIES_COLORS.items():
        sp_df = df[df["Species Name"] == sp_nm]
        fig_3d.add_trace(go.Scatter3d(
            x=sp_df[ax_x], y=sp_df[ax_y], z=sp_df[ax_z],
            mode="markers",
            marker=dict(size=5, color=sp_cl, opacity=0.55),
            name=sp_nm,
        ))

    fig_3d.add_trace(go.Scatter3d(
        x=[inp[ax_x]], y=[inp[ax_y]], z=[inp[ax_z]],
        mode="markers+text",
        marker=dict(size=11, color="#ef4444", symbol="diamond",
                    line=dict(color="white", width=2)),
        text=["New Sample"],
        textposition="top center",
        textfont=dict(size=12, color="#ef4444"),
        name="New Sample",
    ))

    fig_3d.update_layout(
        height=560,
        scene=dict(
            xaxis_title=ax_x.replace(" (cm)", ""),
            yaxis_title=ax_y.replace(" (cm)", ""),
            zaxis_title=ax_z.replace(" (cm)", ""),
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#e2e8f0"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#e2e8f0"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#e2e8f0"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font_size=12, bordercolor="#e2e8f0", borderwidth=1),
        font_color="#1e293b",
        margin=dict(t=20, b=20, l=10, r=10),
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown(
        '<div class="info-purple"><strong>Note:</strong> The red diamond marker '
        'represents the input sample projected into the selected 3D feature subspace. '
        'Its proximity to a species cluster provides visual corroboration of the '
        "classifier's prediction. Use the axis selectors to explore different "
        'feature combinations.</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    df = load_data()
    clf, scaler, metrics, cm, feat_imp = train_model(df)
    page = render_sidebar()

    if   page == "Data Overview":       page_data_overview(df)
    elif page == "Exploratory Analysis": page_eda(df)
    elif page == "Model Performance":    page_model(df, clf, scaler, metrics, cm, feat_imp)
    elif page == "Prediction":           page_prediction(df, clf, scaler)


if __name__ == "__main__":
    main()
