"""
ADNI/PPMI Hybrid Quantum-Classical ML Benchmarking Tool
=========================================================
100% LOCAL – No external API calls. All computation via local Python libraries.
ADNI/PPMI Data Use Agreement compliant: no data leaves this machine.

DEMO_MODE=1 env var: disables CSV upload (synthetic data only).
Set this on any cloud host to prevent accidental real-data submission.
"""
import os
import io
import json
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from quantum_pipeline import (
    DataProcessor, PennyLaneVQC, evaluate_classifier, run_benchmark
)
from sklearn.model_selection import train_test_split

# ── Demo / compliance mode ─────────────────────────────────────────────────────
DEMO_MODE = os.environ.get("DEMO_MODE", "0") == "1"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantumMed · ADNI/PPMI Benchmark",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

:root {
    --bg: #0d1117;
    --surface: #161b22;
    --surface2: #1c2230;
    --border: #30363d;
    --accent: #7c5cbf;
    --accent2: #58a6ff;
    --quantum: #a371f7;
    --classical: #3fb950;
    --warning: #e3b341;
    --text: #e6edf3;
    --muted: #8b949e;
}

.stApp { background: var(--bg); color: var(--text); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.card-accent {
    border-left: 3px solid var(--quantum);
}

/* Metric pills */
.metric-row { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }
.metric-pill {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 18px;
    min-width: 130px;
    text-align: center;
}
.metric-pill .label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.8px; }
.metric-pill .value { font-size: 22px; font-weight: 700; color: var(--accent2); margin-top: 4px; }
.metric-pill.q .value { color: var(--quantum); }
.metric-pill.c .value { color: var(--classical); }

/* Header banner */
.hero {
    background: linear-gradient(135deg, #1a1040 0%, #0d1117 50%, #0a1628 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 80% 50%, rgba(124,92,191,0.12) 0%, transparent 70%);
}
.hero h1 { font-size: 26px; font-weight: 700; margin: 0; color: var(--text); }
.hero p { color: var(--muted); margin: 6px 0 0; font-size: 14px; }
.compliance-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(63,185,80,0.12); border: 1px solid rgba(63,185,80,0.35);
    color: #3fb950; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;
    margin-top: 10px;
}

/* Step badges */
.step-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px; background: var(--quantum); border-radius: 50%;
    color: white; font-weight: 700; font-size: 13px; margin-right: 8px;
}
.section-title { font-size: 16px; font-weight: 600; color: var(--text); margin: 0; display: flex; align-items: center; }

/* Tables */
.dataframe { border-radius: 8px; overflow: hidden; }

/* Warning box */
.warn-box {
    background: rgba(227,179,65,0.1);
    border: 1px solid rgba(227,179,65,0.4);
    border-radius: 8px; padding: 12px 16px;
    color: var(--warning); font-size: 13px;
}

/* Custom scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>⚛️ QuantumMed — ADNI / PPMI Benchmark Suite</h1>
  <p>Hybrid Quantum-Classical ML · Variational Quantum Classifier (PennyLane) vs Random Forest vs Logistic Regression</p>
  <span class="compliance-badge">🔒 100 % Local · ADNI/PPMI DUA Compliant · No External API Calls</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.divider()

    if DEMO_MODE:
        st.markdown("**📂 Data**")
        st.info("🔒 **Demo mode** — CSV upload disabled on this hosted instance.\nRun locally to load real ADNI/PPMI data.")
        uploaded_file = None
        folder_path = ""
    else:
        st.markdown("**📂 Data**")
        upload_mode = st.radio("Input mode", ["Upload CSV", "Load from folder path"], label_visibility="collapsed")
        uploaded_file = None
        folder_path = ""
        if upload_mode == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        else:
            folder_path = st.text_input("Folder path (first CSV found is loaded)", placeholder="C:/data/ADNI_export/")

    st.divider()
    st.markdown("**🔬 Preprocessing**")
    imputer_strategy = st.selectbox("Missing value strategy", ["mean", "median", "most_frequent"])
    n_components = st.slider("PCA components (= qubits)", min_value=2, max_value=8, value=4, step=1)
    test_size = st.slider("Test split %", min_value=10, max_value=40, value=20, step=5) / 100

    st.divider()
    st.markdown("**⚛️ VQC Settings**")
    n_layers = st.slider("Entangler layers", min_value=1, max_value=4, value=2)
    max_steps = st.slider("Optimisation steps", min_value=10, max_value=100, value=30, step=5)

    st.divider()
    st.markdown("**🌲 Classical Settings**")
    rf_estimators = st.slider("RF estimators", min_value=50, max_value=300, value=100, step=50)

    st.divider()
    st.markdown("**💾 Output**")
    output_dir = st.text_input("Output folder", value="./results")

    st.markdown("""
    <div style='margin-top:20px; padding:10px; background:rgba(124,92,191,0.1); border-radius:8px; font-size:11px; color:#8b949e;'>
    ⚠️ This tool operates 100% offline.<br>
    No data is transmitted to any server.<br>
    ADNI/PPMI DUA compliant.
    </div>
    """, unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded uploaded CSV — {df.shape[0]} rows × {df.shape[1]} columns")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

elif folder_path.strip():
    try:
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        if csv_files:
            first_csv = os.path.join(folder_path, csv_files[0])
            df = pd.read_csv(first_csv)
            st.success(f"✅ Loaded `{csv_files[0]}` from folder — {df.shape[0]} rows × {df.shape[1]} columns")
        else:
            st.warning("No CSV files found in the specified folder.")
    except Exception as e:
        st.error(f"Could not read folder: {e}")

# ── Demo mode (synthetic data) ─────────────────────────────────────────────────
if df is None:
    with st.container():
        if DEMO_MODE:
            msg = "🔒 <b>Demo instance</b> — Running with synthetic ADNI-style data only. No real participant data is accepted here. <a href='https://github.com/Victorraj020/Quantum-_extension-' style='color:#58a6ff;'>Clone &amp; run locally</a> to use real ADNI/PPMI exports."
        else:
            msg = "📋 No data loaded yet. Use the sidebar to upload a CSV or point to a folder. Click <b>Load demo data</b> below to test with a synthetic ADNI-style dataset (random data, not real participant records)."
        st.markdown(f'<div class="warn-box">{msg}</div>', unsafe_allow_html=True)

    if st.button("🧪 Load synthetic demo data (ADNI-style)", use_container_width=True):
        np.random.seed(42)
        n = 180
        feature_names = [
            "Hippocampus_Volume", "Entorhinal_Thickness", "FDG_PET_SUVr", "AV45_Amyloid_PET",
            "CSF_Abeta42", "CSF_pTau181", "CSF_tTau", "MoCA_Score", "MMSE_Score",
            "UPDRS_Part3_Motor", "DatSCAN_SBR", "Age"
        ]
        X = np.random.randn(n, len(feature_names))
        mask = np.random.rand(*X.shape) < 0.05
        X[mask] = np.nan
        logits = 0.8*X[:,0] - 0.7*X[:,1] + 0.9*X[:,2] - 0.6*X[:,4] + np.random.randn(n)*0.5
        labels = np.where(logits > 0, "AD", "CN")
        df = pd.DataFrame(X, columns=feature_names)
        df["Diagnosis_Group"] = labels
        st.session_state["demo_df"] = df
        st.rerun()

if "demo_df" in st.session_state and df is None:
    df = st.session_state["demo_df"]

# ── Column selection ───────────────────────────────────────────────────────────
if df is not None:
    st.markdown("---")
    st.markdown('<p class="section-title"><span class="step-badge">1</span> Select Features & Target</p>', unsafe_allow_html=True)

    with st.container():
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.markdown("**Data preview** (first 5 rows)")
            st.dataframe(df.head(), use_container_width=True)
        with col_right:
            st.markdown(f"**Shape:** `{df.shape[0]} × {df.shape[1]}`")
            dtypes = df.dtypes.value_counts().to_dict()
            for k, v in dtypes.items():
                st.markdown(f"- `{k}`: {v} cols")
            missing = df.isnull().sum().sum()
            st.markdown(f"- Missing values: **{missing}**")

    all_cols = list(df.columns)
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

    target_col = st.selectbox(
        "🎯 Target / Label column",
        options=all_cols,
        index=len(all_cols) - 1,
        help="The column you want to predict (e.g. Diagnosis_Group)"
    )

    default_features = [c for c in numeric_cols if c != target_col]
    feature_cols = st.multiselect(
        "📊 Feature columns (numeric only)",
        options=[c for c in all_cols if c != target_col],
        default=default_features[:12],
        help="Select the input features. Non-numeric columns will cause errors."
    )

    if feature_cols:
        classes_preview = df[target_col].dropna().unique()
        n_classes = len(classes_preview)
        mode_label = "Binary" if n_classes == 2 else f"Multi-class ({n_classes} classes)"
        st.info(f"🏷️ Detected **{mode_label}** task — classes: `{sorted([str(c) for c in classes_preview])}`")

# ── Run benchmark ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title"><span class="step-badge">2</span> Run Benchmark</p>', unsafe_allow_html=True)

    can_run = len(feature_cols) >= n_components
    if not can_run:
        st.warning(f"⚠️ Select at least **{n_components}** feature columns to match PCA components ({n_components}).")

    if st.button("🚀 Run Full Benchmark (Classical + VQC)", use_container_width=True, disabled=not can_run):
        config = {
            "imputer_strategy": imputer_strategy,
            "n_components": n_components,
            "vqc_n_layers": n_layers,
            "vqc_max_steps": max_steps,
            "rf_n_estimators": rf_estimators,
            "test_size": test_size,
            "random_state": 42
        }

        progress_bar = st.progress(0)
        status_txt = st.empty()

        try:
            status_txt.markdown("⏳ Preprocessing & PCA...")
            progress_bar.progress(10)

            status_txt.markdown("🌲 Training Random Forest...")
            progress_bar.progress(25)

            status_txt.markdown("📈 Training Logistic Regression...")
            progress_bar.progress(40)

            status_txt.markdown(f"⚛️ Training VQC ({n_layers} layers, {max_steps} steps) — this may take a minute...")
            progress_bar.progress(55)

            benchmark_df, loadings_df, rf_importance, out_path = run_benchmark(
                df, feature_cols, target_col, config, output_dir
            )

            progress_bar.progress(90)
            status_txt.markdown("📊 Generating plots & saving artefacts...")
            time.sleep(0.3)
            progress_bar.progress(100)
            status_txt.empty()

            st.session_state["results"] = {
                "benchmark_df": benchmark_df,
                "loadings_df": loadings_df,
                "rf_importance": rf_importance,
                "output_dir": out_path,
                "feature_cols": feature_cols,
                "target_col": target_col,
                "n_components": n_components,
            }
            st.rerun()

        except Exception as e:
            progress_bar.empty()
            status_txt.empty()
            st.error(f"❌ Benchmark failed: {e}")
            st.exception(e)

# ── Results ────────────────────────────────────────────────────────────────────
if "results" in st.session_state:
    res = st.session_state["results"]
    bdf = res["benchmark_df"]
    loadings_df = res["loadings_df"]
    rf_importance = res["rf_importance"]
    out_path = res["output_dir"]
    feature_cols = res["feature_cols"]
    n_components = res["n_components"]

    st.markdown("---")
    st.markdown('<p class="section-title"><span class="step-badge">3</span> Results</p>', unsafe_allow_html=True)

    # ── Per-model metric cards
    model_colors = {
        "Random Forest": "#3fb950",
        "Logistic Regression": "#58a6ff",
        "Variational Quantum Classifier (VQC)": "#a371f7"
    }
    model_icons = {
        "Random Forest": "🌲",
        "Logistic Regression": "📈",
        "Variational Quantum Classifier (VQC)": "⚛️"
    }

    for model_name in bdf.index:
        row = bdf.loc[model_name]
        color = model_colors.get(model_name, "#ffffff")
        icon = model_icons.get(model_name, "🔬")

        st.markdown(f"""
        <div class="card" style="border-left: 3px solid {color};">
          <div style="font-size:15px; font-weight:600; color:{color};">{icon} {model_name}</div>
          <div class="metric-row">
            <div class="metric-pill"><div class="label">Accuracy</div><div class="value" style="color:{color};">{row.get('Accuracy', 0):.3f}</div></div>
            <div class="metric-pill"><div class="label">Sensitivity</div><div class="value" style="color:{color};">{row.get('Sensitivity', 0):.3f}</div></div>
            <div class="metric-pill"><div class="label">Specificity</div><div class="value" style="color:{color};">{row.get('Specificity', 0):.3f}</div></div>
            <div class="metric-pill"><div class="label">Precision</div><div class="value" style="color:{color};">{row.get('Precision', 0):.3f}</div></div>
            <div class="metric-pill"><div class="label">F1-Score</div><div class="value" style="color:{color};">{row.get('F1-Score', 0):.3f}</div></div>
            <div class="metric-pill"><div class="label">ROC-AUC</div><div class="value" style="color:{color};">{row.get('ROC-AUC', 0):.3f}</div></div>
            <div class="metric-pill"><div class="label">PR-AUC</div><div class="value" style="color:{color};">{row.get('PR-AUC', 0):.3f}</div></div>
            <div class="metric-pill"><div class="label">Train Time</div><div class="value" style="color:{color};">{row.get('Training Time (s)', 0):.1f}s</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Comparison chart
    st.markdown("#### 📊 Benchmark Comparison Chart")
    metrics_to_plot = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1-Score", "ROC-AUC", "PR-AUC"]
    plot_df = bdf[metrics_to_plot]

    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0d1117')
    ax.set_facecolor('#161b22')

    x = np.arange(len(metrics_to_plot))
    bar_w = 0.25
    colors = ['#3fb950', '#58a6ff', '#a371f7']
    for i, (model, color) in enumerate(zip(plot_df.index, colors)):
        offset = (i - 1) * bar_w
        bars = ax.bar(x + offset, plot_df.loc[model].values, bar_w, label=model, color=color, alpha=0.9, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, color='#e6edf3', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", color='#8b949e')
    ax.tick_params(colors='#8b949e')
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#30363d', zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.legend(facecolor='#1c2230', edgecolor='#30363d', labelcolor='#e6edf3', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Explainability
    st.markdown("---")
    st.markdown('<p class="section-title"><span class="step-badge">4</span> Explainability</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🌲 RF Feature Importance (PCA space)", "🔭 PCA Component Loadings (original features)"])

    with tab1:
        fig2, ax2 = plt.subplots(figsize=(7, 3.5), facecolor='#0d1117')
        ax2.set_facecolor('#161b22')
        comps = rf_importance.index.tolist()
        imps = rf_importance["RF_Importance"].values
        bars = ax2.barh(comps, imps, color='#3fb950', alpha=0.85)
        ax2.set_xlabel("Importance", color='#8b949e')
        ax2.tick_params(colors='#8b949e')
        ax2.set_title("Random Forest: Importance by PCA Component", color='#e6edf3', fontsize=11)
        for spine in ax2.spines.values():
            spine.set_edgecolor('#30363d')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # Top features per top PC
        top_pc_idx = int(np.argmax(imps))
        top_pc = comps[top_pc_idx]
        st.markdown(f"**Most important PC: `{top_pc}`** — top contributing original features:")
        pc_row = loadings_df.loc[top_pc].abs().sort_values(ascending=False).head(5)
        for feat, coef in pc_row.items():
            bar_len = int(coef * 30)
            st.markdown(f"`{feat}` {'█' * bar_len} `{coef:.3f}`")

    with tab2:
        fig3, ax3 = plt.subplots(figsize=(10, 4), facecolor='#0d1117')
        ax3.set_facecolor('#161b22')
        im = ax3.imshow(loadings_df.values, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        cbar = fig3.colorbar(im, ax=ax3)
        cbar.ax.yaxis.set_tick_params(color='#8b949e')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#8b949e')
        ax3.set_yticks(range(n_components))
        ax3.set_yticklabels(loadings_df.index, color='#e6edf3')
        ax3.set_xticks(range(len(feature_cols)))
        ax3.set_xticklabels(feature_cols, rotation=55, ha='right', color='#e6edf3', fontsize=8)
        ax3.set_title("PCA Loadings: Components vs Original Features", color='#e6edf3', fontsize=11)
        for spine in ax3.spines.values():
            spine.set_edgecolor('#30363d')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    # ── Download section
    st.markdown("---")
    st.markdown('<p class="section-title"><span class="step-badge">5</span> Download Results</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        csv_bytes = bdf.to_csv().encode()
        st.download_button("📥 Benchmark Metrics CSV", data=csv_bytes,
                           file_name="benchmark_metrics.csv", mime="text/csv", use_container_width=True)
    with col2:
        ld_bytes = loadings_df.to_csv().encode()
        st.download_button("📥 PCA Loadings CSV", data=ld_bytes,
                           file_name="pca_loadings.csv", mime="text/csv", use_container_width=True)
    with col3:
        ri_bytes = rf_importance.to_csv().encode()
        st.download_button("📥 RF Importance CSV", data=ri_bytes,
                           file_name="rf_importance.csv", mime="text/csv", use_container_width=True)

    st.success(f"✅ All outputs saved to: `{os.path.abspath(out_path)}`")

    with st.expander("📋 Run summary JSON"):
        summary_path = os.path.join(out_path, "run_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                st.json(json.load(f))
