from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Genome Data Clustering Studio",
    page_icon="GC",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

PLOT_FILES = [
    "05_umap_basic.png",
    "06_k_selection.png",
    "07_kmeans_clusters.png",
    "09_expression_heatmap.png",
    "10_top_genes.png",
    "13_confusion_matrix.png",
    "14_feature_importance.png",
    "15_prediction_confidence.png",
]

SUBTYPE_COPY = {
    0: "Subtype 0 - high-separation program",
    1: "Subtype 1 - immune-active program",
    2: "Subtype 2 - stromal-rich program",
    3: "Subtype 3 - mixed phenotype",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

        :root {
            --bg: #f4efe7;
            --panel: rgba(255, 252, 247, 0.86);
            --panel-strong: #fffdf9;
            --text: #132a2a;
            --muted: #5b6e6d;
            --accent: #0f766e;
            --accent-2: #d97706;
            --accent-3: #1d4ed8;
            --border: rgba(19, 42, 42, 0.10);
            --shadow: 0 24px 50px rgba(26, 38, 38, 0.08);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(217, 119, 6, 0.12), transparent 30%),
                radial-gradient(circle at top right, rgba(15, 118, 110, 0.14), transparent 34%),
                linear-gradient(180deg, #f8f3eb 0%, #eef6f4 52%, #f5efe6 100%);
            color: var(--text);
            font-family: 'Manrope', sans-serif;
        }

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }

        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif !important;
            color: var(--text);
            letter-spacing: -0.02em;
        }

        section[data-testid="stSidebar"] {
            background: rgba(19, 42, 42, 0.94);
            color: #f7f6f2;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        section[data-testid="stSidebar"] * {
            color: #f7f6f2 !important;
        }

        .sidebar-brand {
            background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 22px;
            padding: 1rem 1rem 0.95rem 1rem;
            margin-bottom: 1rem;
        }

        .sidebar-brand-kicker {
            font-size: 0.74rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: rgba(247, 246, 242, 0.68);
            font-weight: 800;
        }

        .sidebar-brand-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.3rem;
            font-weight: 700;
            margin-top: 0.35rem;
            color: #ffffff;
        }

        .sidebar-brand-copy {
            font-size: 0.9rem;
            line-height: 1.5;
            color: rgba(247, 246, 242, 0.76);
            margin-top: 0.5rem;
        }

        .sidebar-stack {
            display: grid;
            gap: 0.8rem;
            margin: 0.5rem 0 1rem 0;
        }

        .sidebar-metric-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.12), rgba(255,255,255,0.04));
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 0.9rem 1rem;
            box-shadow: 0 16px 28px rgba(0, 0, 0, 0.12);
        }

        .sidebar-metric-label {
            font-size: 0.74rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 800;
            color: rgba(247, 246, 242, 0.68);
        }

        .sidebar-metric-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.45rem;
            line-height: 1.1;
            font-weight: 700;
            color: #ffffff;
            margin-top: 0.25rem;
        }

        .sidebar-metric-copy {
            font-size: 0.84rem;
            line-height: 1.45;
            color: rgba(247, 246, 242, 0.72);
            margin-top: 0.35rem;
        }

        .sidebar-section-title {
            font-size: 0.76rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 800;
            color: rgba(247, 246, 242, 0.62);
            margin: 1rem 0 0.55rem 0;
        }

        .sidebar-list-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 0.85rem 0.95rem;
        }

        .sidebar-list-item {
            font-size: 0.9rem;
            color: rgba(247, 246, 242, 0.84);
            padding: 0.32rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }

        .sidebar-list-item:last-child {
            border-bottom: none;
            padding-bottom: 0;
        }

        .hero-card {
            background: linear-gradient(135deg, rgba(19, 42, 42, 0.96), rgba(15, 118, 110, 0.88));
            border-radius: 28px;
            padding: 2rem 2.1rem;
            box-shadow: var(--shadow);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: #f8fafc;
            overflow: hidden;
            position: relative;
        }

        .hero-card:after {
            content: "";
            position: absolute;
            inset: auto -40px -50px auto;
            width: 180px;
            height: 180px;
            background: radial-gradient(circle, rgba(255,255,255,0.2), transparent 68%);
        }

        .eyebrow {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.12);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.9rem;
        }

        .hero-title {
            font-size: 2.7rem;
            line-height: 1.03;
            margin: 0;
            color: #fcfdfc;
        }

        .hero-copy {
            color: rgba(248, 250, 252, 0.85);
            font-size: 1.02rem;
            margin-top: 0.9rem;
            max-width: 760px;
        }

        .mini-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 1.35rem;
        }

        .mini-pill {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 0.9rem 1rem;
        }

        .mini-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: rgba(248, 250, 252, 0.68);
        }

        .mini-value {
            font-size: 1.28rem;
            font-weight: 800;
            color: #ffffff;
        }

        .section-label {
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--accent);
            font-weight: 800;
            margin-top: 0.5rem;
            margin-bottom: 0.2rem;
        }

        .card {
            background: var(--panel);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 1.15rem 1.2rem;
            box-shadow: var(--shadow);
        }

        .chart-card {
            border-radius: 28px 28px 0 0;
            padding: 1.15rem 1.2rem 0.9rem 1.2rem;
            margin-bottom: 0;
            border: 1px solid rgba(19, 42, 42, 0.12);
            border-bottom: none;
            position: relative;
            overflow: hidden;
        }

        .chart-card:before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 4px;
            background: linear-gradient(90deg, rgba(255,255,255,0.0), rgba(255,255,255,0.8), rgba(255,255,255,0.0));
            opacity: 0.8;
        }

        .chart-kicker {
            font-size: 0.76rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--accent);
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .chart-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.2rem;
            color: var(--text);
            margin-bottom: 0.2rem;
        }

        .chart-copy {
            color: var(--muted);
            font-size: 0.92rem;
            margin-bottom: 0.8rem;
        }

        .chart-theme-teal {
            background: linear-gradient(135deg, rgba(15, 118, 110, 0.16), rgba(255,255,255,0.96) 58%, rgba(15, 118, 110, 0.08));
        }

        .chart-theme-amber {
            background: linear-gradient(135deg, rgba(217, 119, 6, 0.16), rgba(255,255,255,0.96) 58%, rgba(245, 158, 11, 0.08));
        }

        .chart-theme-blue {
            background: linear-gradient(135deg, rgba(29, 78, 216, 0.16), rgba(255,255,255,0.96) 58%, rgba(59, 130, 246, 0.08));
        }

        .chart-theme-sage {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.14), rgba(255,255,255,0.96) 58%, rgba(20, 184, 166, 0.08));
        }

        .chart-theme-rose {
            background: linear-gradient(135deg, rgba(244, 63, 94, 0.14), rgba(255,255,255,0.96) 58%, rgba(251, 113, 133, 0.08));
        }

        .table-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(246, 242, 235, 0.88));
            border: 1px solid rgba(19, 42, 42, 0.12);
            border-radius: 26px;
            padding: 1rem;
            box-shadow: 0 20px 38px rgba(24, 38, 38, 0.08);
            margin-bottom: 1rem;
        }

        .table-shell {
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(244, 239, 231, 0.92));
            border: 1px solid rgba(19, 42, 42, 0.12);
            border-top: none;
            border-radius: 0 0 26px 26px;
            padding: 0.1rem 1rem 1rem 1rem;
            box-shadow: 0 24px 48px rgba(24, 38, 38, 0.10);
            margin-top: -1rem;
            margin-bottom: 1.2rem;
        }

        .download-tile {
            background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(242, 249, 247, 0.86));
            border: 1px solid rgba(19, 42, 42, 0.12);
            border-radius: 24px;
            padding: 1rem 1rem 0.75rem 1rem;
            box-shadow: 0 18px 36px rgba(24, 38, 38, 0.08);
            min-height: 170px;
        }

        .download-kicker {
            font-size: 0.74rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--accent);
            font-weight: 800;
            margin-bottom: 0.25rem;
        }

        .download-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.1rem;
            color: var(--text);
            margin-bottom: 0.35rem;
        }

        .download-copy {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.5;
            margin-bottom: 0.9rem;
        }

        .insight-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.88), rgba(250,248,243,0.92));
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            min-height: 148px;
            box-shadow: var(--shadow);
        }

        .insight-card h4 {
            margin: 0 0 0.45rem 0;
            font-size: 0.88rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .insight-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: var(--text);
            margin: 0 0 0.3rem 0;
        }

        .insight-copy {
            font-size: 0.93rem;
            color: var(--muted);
            margin: 0;
        }

        div[data-testid="stMetric"] {
            background: var(--panel-strong);
            border: 1px solid var(--border);
            padding: 1rem;
            border-radius: 20px;
            box-shadow: var(--shadow);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.6rem;
            background: rgba(255,255,255,0.45);
            border: 1px solid rgba(19, 42, 42, 0.10);
            padding: 0.45rem;
            border-radius: 999px;
            width: fit-content;
            box-shadow: 0 12px 26px rgba(24, 38, 38, 0.06);
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.0);
            border-radius: 999px;
            border: 1px solid transparent;
            padding: 0.55rem 1.1rem;
            font-weight: 800;
            color: var(--muted);
            transition: all 0.2s ease;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #123434, #0f766e) !important;
            color: #ffffff !important;
            border-color: rgba(15, 118, 110, 0.22) !important;
            box-shadow: 0 12px 24px rgba(15, 118, 110, 0.24);
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255,255,255,0.68);
            color: var(--text);
        }

        .stPlotlyChart {
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(244, 239, 231, 0.92));
            border: 1px solid rgba(19, 42, 42, 0.12);
            border-top: none;
            border-radius: 0 0 28px 28px;
            padding: 0.35rem 1rem 1rem 1rem;
            margin-bottom: 1.4rem;
            box-shadow: 0 24px 48px rgba(24, 38, 38, 0.10);
        }

        .stPlotlyChart > div {
            border-radius: 20px;
            overflow: hidden;
        }

        .stDownloadButton button, .stButton button {
            border-radius: 999px !important;
            border: 1px solid rgba(15, 118, 110, 0.16) !important;
            background: linear-gradient(135deg, #0f766e, #115e59) !important;
            color: white !important;
            font-weight: 700 !important;
            padding: 0.55rem 1rem !important;
        }

        .stFileUploader > div {
            background: rgba(255,255,255,0.55);
            border-radius: 22px;
            border: 1px dashed rgba(15, 118, 110, 0.35);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_plot(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.88)",
        font=dict(family="Manrope, sans-serif", color="#173131"),
        margin=dict(l=24, r=24, t=56, b=24),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.65)",
        ),
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="rgba(19, 42, 42, 0.18)",
        gridcolor="rgba(19, 42, 42, 0.08)",
        zeroline=False,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="rgba(19, 42, 42, 0.18)",
        gridcolor="rgba(19, 42, 42, 0.08)",
        zeroline=False,
    )
    return fig


def render_chart_card(
    title: str,
    subtitle: str,
    fig: go.Figure,
    kicker: str = "Analytics",
    theme: str = "teal",
) -> None:
    st.markdown(
        f"""
        <div class="chart-card chart-theme-{theme}">
            <div class="chart-kicker">{kicker}</div>
            <div class="chart-title">{title}</div>
            <div class="chart-copy">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(style_plot(fig), use_container_width=True)


def render_table_card(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="table-card">
            <div class="chart-kicker">Breakdown</div>
            <div class="chart-title">{title}</div>
            <div class="chart-copy">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_table_card(title: str, subtitle: str) -> None:
    render_table_card(title, subtitle)
    st.markdown('<div class="table-shell">', unsafe_allow_html=True)


def close_data_table_card() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_download_tile(title: str, subtitle: str, kicker: str = "Export") -> None:
    st.markdown(
        f"""
        <div class="download-tile">
            <div class="download-kicker">{kicker}</div>
            <div class="download-title">{title}</div>
            <div class="download-copy">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def close_download_tile() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


@st.cache_data
def load_feature_names() -> List[str]:
    path = RESULTS_DIR / "gene_names.txt"
    if path.exists():
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return []


@st.cache_resource
def load_models() -> Dict[str, object]:
    model_paths = {
        "scaler": MODELS_DIR / "scaler.pkl",
        "pca": MODELS_DIR / "pca.pkl",
        "classifier": MODELS_DIR / "classifier.pkl",
    }
    missing = [name for name, path in model_paths.items() if not path.exists()]
    if missing:
        return {}
    return {name: joblib.load(path) for name, path in model_paths.items()}


@st.cache_data
def load_data() -> Dict[str, object]:
    cluster_path = RESULTS_DIR / "cluster_assignments.csv"
    prediction_path = RESULTS_DIR / "prediction_results.csv"
    genes_path = RESULTS_DIR / "top_genes_per_cluster.csv"

    if not cluster_path.exists():
        return {"ready": False}

    cluster_df = pd.read_csv(cluster_path)
    cluster_df["subtype_name"] = cluster_df["final_subtype"].apply(lambda x: f"Subtype {x}")
    cluster_df["cell_index"] = cluster_df["cell_index"].astype(str)

    prediction_df = pd.read_csv(prediction_path) if prediction_path.exists() else pd.DataFrame()
    top_genes_df = pd.read_csv(genes_path) if genes_path.exists() else pd.DataFrame()

    subtype_counts = (
        cluster_df["subtype_name"]
        .value_counts()
        .sort_index()
        .rename_axis("Subtype")
        .reset_index(name="Cells")
    )

    confidence_summary = pd.DataFrame()
    if not prediction_df.empty:
        prediction_df["predicted_name"] = prediction_df["predicted_subtype"].apply(lambda x: f"Subtype {x}")
        prediction_df["true_name"] = prediction_df["true_subtype"].apply(lambda x: f"Subtype {x}")
        confidence_summary = (
            prediction_df.groupby("predicted_name", as_index=False)
            .agg(
                mean_confidence=("confidence", "mean"),
                min_confidence=("confidence", "min"),
                max_confidence=("confidence", "max"),
            )
        )
        confidence_summary.columns = ["Subtype", "Mean Confidence", "Min Confidence", "Max Confidence"]

    return {
        "ready": True,
        "cluster_df": cluster_df,
        "prediction_df": prediction_df,
        "top_genes_df": top_genes_df,
        "subtype_counts": subtype_counts,
        "confidence_summary": confidence_summary,
        "feature_names": load_feature_names(),
    }


def build_excel_download(data: Dict[str, object], session_history: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        data["cluster_df"].to_excel(writer, sheet_name="cluster_assignments", index=False)
        if not data["prediction_df"].empty:
            data["prediction_df"].to_excel(writer, sheet_name="prediction_history", index=False)
        if not data["top_genes_df"].empty:
            data["top_genes_df"].to_excel(writer, sheet_name="top_genes", index=False)
        if not session_history.empty:
            session_history.to_excel(writer, sheet_name="session_predictions", index=False)
    return output.getvalue()


def render_hero(data: Dict[str, object]) -> None:
    cluster_df = data["cluster_df"]
    subtype_count = cluster_df["final_subtype"].nunique()
    hero_html = f"""
    <div class="hero-card">
        <div class="eyebrow">Genomic Intelligence Studio</div>
        <h1 class="hero-title">Genome clustering results presented like a real analytics product.</h1>
        <p class="hero-copy">
            Explore discovered subtypes, inspect model behavior, review saved prediction history,
            and export analysis-ready tables from one Streamlit workspace.
        </p>
        <div class="mini-grid">
            <div class="mini-pill">
                <div class="mini-label">Cells Profiled</div>
                <div class="mini-value">{len(cluster_df):,}</div>
            </div>
            <div class="mini-pill">
                <div class="mini-label">Subtypes Detected</div>
                <div class="mini-value">{subtype_count}</div>
            </div>
            <div class="mini-pill">
                <div class="mini-label">Model Features</div>
                <div class="mini-value">{len(data["feature_names"]):,}</div>
            </div>
        </div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)


def render_sidebar(data: Dict[str, object]) -> None:
    cluster_df = data["cluster_df"]
    prediction_df = data["prediction_df"]
    subtype_count = cluster_df["final_subtype"].nunique()
    median_confidence = (
        f"{prediction_df['confidence'].median() * 100:.1f}%"
        if not prediction_df.empty
        else "N/A"
    )
    accuracy = (
        f"{prediction_df['correct'].mean() * 100:.1f}%"
        if not prediction_df.empty
        else "N/A"
    )
    session_count = len(st.session_state.get("session_predictions", []))

    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-kicker">Genome Studio</div>
            <div class="sidebar-brand-title">Clustering Command Center</div>
            <div class="sidebar-brand-copy">
                A polished workspace for subtype exploration, patient scoring, and export-ready genomics reporting.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        f"""
        <div class="sidebar-stack">
            <div class="sidebar-metric-card">
                <div class="sidebar-metric-label">Cells Profiled</div>
                <div class="sidebar-metric-value">{len(cluster_df):,}</div>
                <div class="sidebar-metric-copy">Saved observations available for exploration in the dashboard.</div>
            </div>
            <div class="sidebar-metric-card">
                <div class="sidebar-metric-label">Detected Subtypes</div>
                <div class="sidebar-metric-value">{subtype_count}</div>
                <div class="sidebar-metric-copy">Distinct groups surfaced by the current clustering outputs.</div>
            </div>
            <div class="sidebar-metric-card">
                <div class="sidebar-metric-label">Median Confidence</div>
                <div class="sidebar-metric-value">{median_confidence}</div>
                <div class="sidebar-metric-copy">Typical certainty level across saved classifier predictions.</div>
            </div>
            <div class="sidebar-metric-card">
                <div class="sidebar-metric-label">Model Accuracy</div>
                <div class="sidebar-metric-value">{accuracy}</div>
                <div class="sidebar-metric-copy">Historic match rate between saved predictions and subtype labels.</div>
            </div>
            <div class="sidebar-metric-card">
                <div class="sidebar-metric-label">Session Runs</div>
                <div class="sidebar-metric-value">{session_count}</div>
                <div class="sidebar-metric-copy">New patient scoring runs recorded in this active Streamlit session.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('<div class="sidebar-section-title">Workspace Highlights</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div class="sidebar-list-card">
            <div class="sidebar-list-item">Interactive subtype dashboards</div>
            <div class="sidebar-list-item">Patient upload and prediction lab</div>
            <div class="sidebar-list-item">Session history tracking</div>
            <div class="sidebar-list-item">CSV and Excel export center</div>
            <div class="sidebar-list-item">Saved result figure archive</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(data: Dict[str, object]) -> None:
    cluster_df = data["cluster_df"]
    prediction_df = data["prediction_df"]
    subtype_counts = data["subtype_counts"]
    confidence_summary = data["confidence_summary"]

    st.markdown('<div class="section-label">Overview</div>', unsafe_allow_html=True)
    st.subheader("Dataset health and platform summary")

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Total cells", f"{len(cluster_df):,}")
    with metric_cols[1]:
        st.metric("Detected subtypes", cluster_df["final_subtype"].nunique())
    with metric_cols[2]:
        median_conf = (
            f"{prediction_df['confidence'].median() * 100:.1f}%"
            if not prediction_df.empty
            else "N/A"
        )
        st.metric("Median confidence", median_conf)
    with metric_cols[3]:
        accuracy = (
            f"{prediction_df['correct'].mean() * 100:.1f}%"
            if not prediction_df.empty
            else "N/A"
        )
        st.metric("Prediction accuracy", accuracy)

    info_cols = st.columns(3)
    subtype_leader = subtype_counts.sort_values("Cells", ascending=False).iloc[0]
    confidence_floor = (
        prediction_df["confidence"].min() * 100 if not prediction_df.empty else None
    )
    with info_cols[0]:
        st.markdown(
            f"""
            <div class="insight-card">
                <h4>Dominant subtype</h4>
                <p class="insight-value">{subtype_leader['Subtype']}</p>
                <p class="insight-copy">{subtype_leader['Cells']:,} cells assigned in the saved clustering outputs.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_cols[1]:
        copy = (
            f"Lowest saved confidence is {confidence_floor:.1f}% across historic predictions."
            if confidence_floor is not None
            else "Prediction history becomes available after the saved classifier exports results."
        )
        st.markdown(
            f"""
            <div class="insight-card">
                <h4>Risk watch</h4>
                <p class="insight-value">{prediction_df['correct'].sum() if not prediction_df.empty else 0:,}</p>
                <p class="insight-copy">{copy}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_cols[2]:
        st.markdown(
            """
            <div class="insight-card">
                <h4>Experience</h4>
                <p class="insight-value">Website-style</p>
                <p class="insight-copy">Interactive sections, analytics storytelling, and direct data export in one app.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    fig = px.scatter(
        cluster_df,
        x="umap_x",
        y="umap_y",
        color="subtype_name",
        hover_data=["cell_index", "kmeans_subtype", "hier_subtype"],
        title="UMAP landscape of final subtype assignments",
        color_discrete_sequence=px.colors.qualitative.Set2,
        opacity=0.78,
    )
    fig.update_traces(marker=dict(size=4, line=dict(width=0)))
    render_chart_card(
        "UMAP landscape of final subtype assignments",
        "Navigate the global cell map with subtype coloring and per-cell hover details in a larger, presentation-ready panel.",
        fig,
        kicker="Landscape",
        theme="teal",
    )

    fig = px.bar(
        subtype_counts,
        x="Cells",
        y="Subtype",
        orientation="h",
        title="Cell volume by subtype",
        color="Subtype",
        color_discrete_sequence=px.colors.qualitative.Safe,
        text="Cells",
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    render_chart_card(
        "Cell volume by subtype",
        "Quickly compare how the discovered clusters distribute across the full cohort with a cleaner ranked breakdown.",
        fig,
        kicker="Distribution",
        theme="amber",
    )

    if not confidence_summary.empty:
        fig = px.scatter(
            confidence_summary,
            x="Mean Confidence",
            y="Subtype",
            size="Max Confidence",
            color="Subtype",
            error_x=confidence_summary["Max Confidence"] - confidence_summary["Min Confidence"],
            title="Confidence band by subtype",
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig.update_layout(xaxis_tickformat=".0%")
        render_chart_card(
            "Confidence band by subtype",
            "A quick read on average certainty and spread for each discovered subtype across saved predictions.",
            fig,
            kicker="Reliability",
            theme="blue",
        )


def render_cluster_explorer(data: Dict[str, object]) -> None:
    cluster_df = data["cluster_df"]
    prediction_df = data["prediction_df"]
    top_genes_df = data["top_genes_df"]

    st.markdown('<div class="section-label">Explorer</div>', unsafe_allow_html=True)
    st.subheader("Interactive cluster and model explorer")

    subtype_options = sorted(cluster_df["subtype_name"].unique().tolist())
    selected = st.multiselect(
        "Filter visible subtypes",
        subtype_options,
        default=subtype_options,
    )
    filtered = cluster_df[cluster_df["subtype_name"].isin(selected)].copy()

    fig = px.density_heatmap(
        filtered,
        x="umap_x",
        y="umap_y",
        nbinsx=45,
        nbinsy=45,
        title="Spatial density across the filtered subtype set",
        color_continuous_scale="Tealgrn",
    )
    render_chart_card(
        "Spatial density across the filtered subtype set",
        "Use the subtype filter to focus the density map on the patterns you want to inspect.",
        fig,
        kicker="Explorer",
        theme="blue",
    )

    preview = (
        top_genes_df[top_genes_df["cluster"].isin([int(x.split()[-1]) for x in selected])]
        if not top_genes_df.empty
        else pd.DataFrame()
    )
    if not preview.empty:
        preview = preview.sort_values(["cluster", "abs_fold_change"], ascending=[True, False]).groupby("cluster").head(5)
        fig = px.bar(
            preview,
            x="abs_fold_change",
            y="gene",
            color="cluster",
            facet_row="cluster",
            title="Top marker genes by selected cluster",
            height=max(420, 180 * preview["cluster"].nunique()),
        )
        fig.for_each_annotation(lambda a: a.update(text=a.text.replace("cluster=", "Subtype ")))
        render_chart_card(
            "Top marker genes by selected cluster",
            "Review the strongest expression signals backing each visible subtype.",
            fig,
            kicker="Markers",
            theme="sage",
        )
    else:
        st.info("Marker gene previews will appear here when `results/top_genes_per_cluster.csv` is available.")

    if not prediction_df.empty:
        fig = px.histogram(
            prediction_df,
            x="confidence",
            color="predicted_name",
            nbins=30,
            barmode="overlay",
            title="Confidence distribution across saved predictions",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_layout(xaxis_tickformat=".0%")
        render_chart_card(
            "Confidence distribution across saved predictions",
            "Spot where the classifier is decisive and where certainty starts to thin out.",
            fig,
            kicker="Model Quality",
            theme="rose",
        )

        scorecard = (
            prediction_df.groupby("predicted_name", as_index=False)
            .agg(
                observations=("predicted_name", "size"),
                accuracy=("correct", "mean"),
                avg_confidence=("confidence", "mean"),
            )
            .sort_values("observations", ascending=False)
        )
        scorecard["accuracy"] = (scorecard["accuracy"] * 100).round(1)
        scorecard["avg_confidence"] = (scorecard["avg_confidence"] * 100).round(1)
        render_table_card(
            "Subtype scorecard",
            "A compact summary of volume, accuracy, and average confidence by predicted subtype.",
        )
        st.dataframe(
            scorecard,
            use_container_width=True,
            hide_index=True,
            column_config={
                "accuracy": st.column_config.ProgressColumn("Accuracy %", min_value=0, max_value=100),
                "avg_confidence": st.column_config.ProgressColumn("Avg confidence %", min_value=0, max_value=100),
            },
        )


def normalise_uploaded_patient(uploaded_file, feature_names: List[str]) -> Tuple[pd.Series, pd.DataFrame]:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        source = pd.read_excel(uploaded_file)
    else:
        source = pd.read_csv(uploaded_file)

    working = source.copy()
    working.columns = [str(col).strip() for col in working.columns]
    feature_set = set(feature_names)

    if {"gene", "expression"}.issubset({col.lower() for col in working.columns}):
        lower_map = {col.lower(): col for col in working.columns}
        series = (
            working[[lower_map["gene"], lower_map["expression"]]]
            .dropna()
            .drop_duplicates(subset=[lower_map["gene"]], keep="last")
            .set_index(lower_map["gene"])[lower_map["expression"]]
        )
        series.index = series.index.astype(str)
        aligned = pd.to_numeric(series, errors="coerce").reindex(feature_names).fillna(0.0)
        return aligned.astype(float), source.head(10)

    matching_cols = [col for col in working.columns if col in feature_set]
    if matching_cols:
        row = working.iloc[0]
        aligned = pd.to_numeric(row.reindex(feature_names), errors="coerce").fillna(0.0)
        return aligned.astype(float), source.head(10)

    flat = pd.to_numeric(working.squeeze(), errors="coerce")
    if getattr(flat, "ndim", 1) == 1 and len(flat) == len(feature_names):
        aligned = pd.Series(flat.to_numpy(), index=feature_names, dtype=float)
        return aligned, source.head(10)

    raise ValueError(
        "Unsupported file layout. Use either a `gene,expression` file, a single-row file with feature columns, "
        f"or a vector with exactly {len(feature_names)} values."
    )


def predict_patient(feature_vector: pd.Series, models: Dict[str, object]) -> Dict[str, object]:
    scaler = models["scaler"]
    pca = models["pca"]
    clf = models["classifier"]

    x = np.log1p(feature_vector.to_numpy(dtype=float))
    x = scaler.transform([x])
    x = pca.transform(x)

    predicted = int(clf.predict(x)[0])
    probabilities = clf.predict_proba(x)[0]
    class_labels = [int(label) for label in clf.classes_]
    prob_map = {f"Subtype {label}": float(prob) for label, prob in zip(class_labels, probabilities)}

    return {
        "predicted_subtype": predicted,
        "predicted_label": f"Subtype {predicted}",
        "confidence": float(probabilities.max()),
        "probabilities": prob_map,
        "clinical_note": SUBTYPE_COPY.get(predicted, "Saved classifier subtype identified from uploaded sample."),
    }


def render_prediction_lab(data: Dict[str, object]) -> None:
    st.markdown('<div class="section-label">Prediction Lab</div>', unsafe_allow_html=True)
    st.subheader("Upload a new patient file and generate website-style prediction output")

    models = load_models()
    feature_names = data["feature_names"]

    if "session_predictions" not in st.session_state:
        st.session_state["session_predictions"] = []

    if not models or not feature_names:
        st.warning("Saved models or feature names are missing. Run the pipeline before using the prediction workspace.")
        return

    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown(
            """
            <div class="card">
                <h3 style="margin-top:0;">Upload patient expression data</h3>
                <p style="color:#5b6e6d;">
                    Accepted formats: CSV or Excel. You can upload a <code>gene,expression</code> table,
                    a single-row table with matching feature columns, or a plain vector with the saved feature count.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader("Patient file", type=["csv", "xlsx", "xls"], key="patient_upload")

        if uploaded is not None:
            try:
                feature_vector, preview = normalise_uploaded_patient(uploaded, feature_names)
                st.dataframe(preview, use_container_width=True, hide_index=True)
                patient_id = st.text_input("Patient ID", value=Path(uploaded.name).stem)
                if st.button("Run subtype prediction", use_container_width=True):
                    result = predict_patient(feature_vector, models)
                    record = {
                        "patient_id": patient_id,
                        "predicted_subtype": result["predicted_label"],
                        "confidence_percent": round(result["confidence"] * 100, 2),
                        "clinical_note": result["clinical_note"],
                    }
                    record.update({name: round(prob * 100, 2) for name, prob in result["probabilities"].items()})
                    st.session_state["session_predictions"].append(record)
                    st.session_state["latest_prediction"] = result
                    st.session_state["latest_patient_id"] = patient_id
            except Exception as exc:
                st.error(str(exc))

    with right:
        latest = st.session_state.get("latest_prediction")
        latest_patient = st.session_state.get("latest_patient_id", "Not predicted yet")
        if latest:
            st.markdown(
                f"""
                <div class="card">
                    <div class="section-label" style="margin-top:0;">Latest result</div>
                    <h2 style="margin-top:0;">{latest['predicted_label']}</h2>
                    <p style="font-size:1.05rem;color:#173131;"><strong>Patient:</strong> {latest_patient}</p>
                    <p style="font-size:1.05rem;color:#173131;"><strong>Confidence:</strong> {latest['confidence'] * 100:.2f}%</p>
                    <p style="color:#5b6e6d;">{latest['clinical_note']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            probs = pd.DataFrame(
                {
                    "Subtype": list(latest["probabilities"].keys()),
                    "Probability": list(latest["probabilities"].values()),
                }
            )
            fig = px.bar(
                probs,
                x="Subtype",
                y="Probability",
                color="Subtype",
                title="Probability profile",
                text_auto=".1%",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig.update_layout(yaxis_tickformat=".0%")
            render_chart_card(
                "Probability profile",
                "See how the uploaded patient scores across the available subtype classes.",
                fig,
                kicker="Prediction",
                theme="blue",
            )
        else:
            st.info("Run a prediction to populate the live result card and probability profile.")


def render_history_and_downloads(data: Dict[str, object]) -> None:
    st.markdown('<div class="section-label">History and Exports</div>', unsafe_allow_html=True)
    st.subheader("Saved history, session activity, and one-click downloads")

    session_history = pd.DataFrame(st.session_state.get("session_predictions", []))
    history_tabs = st.tabs(["Saved Model History", "Session Prediction History", "Download Center", "Visual Archive"])

    with history_tabs[0]:
        prediction_df = data["prediction_df"]
        if prediction_df.empty:
            st.info("No saved prediction history was found in `results/prediction_results.csv`.")
        else:
            preview = prediction_df.head(1000).copy()
            if "confidence" in preview.columns:
                preview["confidence"] = (preview["confidence"] * 100).round(1)
            render_data_table_card(
                "Saved model history",
                "Historic classifier outputs from the project results folder, trimmed to a fast preview.",
            )
            st.dataframe(preview, use_container_width=True, hide_index=True)
            close_data_table_card()
            st.caption("Showing the first 1,000 rows for performance. Use the download buttons for the full file.")

    with history_tabs[1]:
        if session_history.empty:
            st.info("Predictions you run in this session will be stored here.")
        else:
            render_data_table_card(
                "Session prediction history",
                "Fresh predictions generated during this current Streamlit session.",
            )
            st.dataframe(session_history, use_container_width=True, hide_index=True)
            close_data_table_card()
            render_download_tile(
                "Session history CSV",
                "Export the current session's patient predictions for follow-up reporting or review.",
                kicker="Session Export",
            )
            st.download_button(
                "Download session history CSV",
                session_history.to_csv(index=False).encode("utf-8"),
                file_name="session_prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
            close_download_tile()

    with history_tabs[2]:
        download_cols = st.columns(3)
        excel_bytes = build_excel_download(data, session_history)
        with download_cols[0]:
            render_download_tile(
                "Cluster assignments CSV",
                "Raw clustering output for downstream analysis, validation, or custom visualizations.",
            )
            st.download_button(
                "Cluster assignments CSV",
                data["cluster_df"].to_csv(index=False).encode("utf-8"),
                file_name="cluster_assignments.csv",
                mime="text/csv",
                use_container_width=True,
            )
            close_download_tile()
        with download_cols[1]:
            prediction_payload = (
                data["prediction_df"].to_csv(index=False).encode("utf-8")
                if not data["prediction_df"].empty
                else b""
            )
            render_download_tile(
                "Prediction history CSV",
                "Saved model predictions with confidence values and subtype probabilities from the results folder.",
            )
            st.download_button(
                "Prediction history CSV",
                prediction_payload,
                file_name="prediction_results.csv",
                mime="text/csv",
                use_container_width=True,
                disabled=data["prediction_df"].empty,
            )
            close_download_tile()
        with download_cols[2]:
            render_download_tile(
                "Full Excel workbook",
                "Bundle cluster assignments, prediction history, top genes, and session outputs into one workbook.",
                kicker="Workbook",
            )
            st.download_button(
                "Full Excel workbook",
                excel_bytes,
                file_name="genome_dashboard_exports.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            close_download_tile()

        if not data["top_genes_df"].empty:
            render_download_tile(
                "Top genes CSV",
                "Export marker genes by cluster for interpretation, reporting, or annotation workflows.",
                kicker="Feature Export",
            )
            st.download_button(
                "Top genes CSV",
                data["top_genes_df"].to_csv(index=False).encode("utf-8"),
                file_name="top_genes_per_cluster.csv",
                mime="text/csv",
                use_container_width=True,
            )
            close_download_tile()

    with history_tabs[3]:
        available_plots = [name for name in PLOT_FILES if (RESULTS_DIR / name).exists()]
        if not available_plots:
            st.info("No result figures are available in the `results/` directory yet.")
        else:
            gallery_cols = st.columns(2)
            for idx, name in enumerate(available_plots):
                with gallery_cols[idx % 2]:
                    st.image(str(RESULTS_DIR / name), caption=name.replace(".png", "").replace("_", " "))


def main() -> None:
    inject_css()

    data = load_data()
    if not data["ready"]:
        st.warning("Pipeline outputs are missing. Run `python src/run_pipeline.py` first.")
        st.code("python src/run_pipeline.py")
        return

    render_sidebar(data)

    render_hero(data)
    st.write("")

    top_tabs = st.tabs(["Overview", "Dashboards", "Prediction Lab", "History and Downloads"])
    with top_tabs[0]:
        render_overview(data)
    with top_tabs[1]:
        render_cluster_explorer(data)
    with top_tabs[2]:
        render_prediction_lab(data)
    with top_tabs[3]:
        render_history_and_downloads(data)


if __name__ == "__main__":
    main()
