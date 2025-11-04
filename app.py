import streamlit as st
import pandas as pd
import os
import time
import tempfile
import traceback
from automl_agent import AutoMLAgent
from plots import (
    plot_correlation_heatmap,
    plot_pairwise,
    plot_target_distribution,
    plot_feature_importance,
    plot_model_comparison,
    plot_shap_summary,
)

# Page configuration
st.set_page_config(page_title="Devika — AI Data Scientist Agent", layout="wide", page_icon="🤖")
st.markdown(
    """
    <style>
    :root {
      --bg: #060913;
      --glass: rgba(255,255,255,0.02);
      --muted: #9fbcd8;
      --neon: #17a2ff;
      --accent: linear-gradient(90deg,#17a2ff,#6b2cff);
    }
    .stApp { background: linear-gradient(180deg,#04050a,#071226); color: #dff6ff; }
    .glass { background: var(--glass); border-radius: 12px; padding: 14px; border: 1px solid rgba(255,255,255,0.03); }
    .sidebar .stImage > img { border-radius:50%; border:4px solid rgba(23,162,255,0.12); box-shadow:0 10px 40px rgba(23,162,255,0.08); display:block; margin:auto; }
    .aria-name { text-align:center; font-weight:700; color:#eaf8ff; margin-top:8px; font-size:18px; text-shadow:0 0 12px rgba(23,162,255,0.06); }
    .aria-role { text-align:center; color:var(--muted); font-size:12px; margin-bottom:6px; }
    .neon-card {
        background: linear-gradient(90deg, rgba(23,162,255,0.08), rgba(107,44,255,0.02));
        border-radius: 10px;
        padding: 12px;
        border: 1px solid rgba(23,162,255,0.06);
    }
    .neon-card h3 { margin:0; color: #dff9ff; }
    .neon-card .meta { color: #bfeaff; margin-top:6px; font-weight:600; }
    .stat-inline { display:inline-block; min-width:120px; text-align:center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar (centered avatar) ----------
# ---------- Sidebar (centered avatar) ----------
with st.sidebar:
    avatar_path = "assets/avatar.png"
    if os.path.exists(avatar_path):
        st.image(avatar_path, width=170)
    else:
        st.markdown(
            "<div style='width:170px;height:170px;border-radius:50%;background:linear-gradient(90deg,#17a2ff,#6b2cff);margin:12px auto;box-shadow:0 12px 40px rgba(23,162,255,0.08)'></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='aria-name'>Devika</div>", unsafe_allow_html=True)
    st.markdown("<div class='aria-role'>AI Data Scientist Agent </div>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📂 Upload")
    uploaded_file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"])
    st.markdown("---")

    st.markdown("### ⚙ Options")
    enable_llm = st.checkbox("Enable LLM insights (optional)")
    llm_api_key = None
    if enable_llm:
        llm_api_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("---")

    st.markdown("### 🚀 Run")
    run_button = st.button("Run AutoML", key="run_main")


st.markdown("<h1 style='margin:6px 0 6px 0;color:#dff9ff;'>Devika — AI Data Scientist Agent</h1>", unsafe_allow_html=True)
st.markdown("<div style='color:#9fbcd8;margin-bottom:12px;'>Automated EDA · AutoML · Explainability</div>", unsafe_allow_html=True)

# ---------- Data loader helper ----------
def load_csv_any_delimiter(uploaded):
    try:
        uploaded.seek(0)
        return pd.read_csv(uploaded)
    except Exception:
        try:
            uploaded.seek(0)
            return pd.read_excel(uploaded)
        except Exception:
            try:
                uploaded.seek(0)
                return pd.read_csv(uploaded, sep=None, engine="python")
            except Exception:
                return None

if uploaded_file is None:
    st.info("Upload a CSV or Excel file from the sidebar to begin.")
    st.stop()
df = load_csv_any_delimiter(uploaded_file)
if df is None or df.shape[1] == 0:
    st.error("Unable to parse the uploaded file. Ensure it's a valid CSV/XLSX with a header row.")
    st.stop()
# normalize columns
df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=True)
# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Visual Insights", "🤖 Model Results", "🔍 Explainability"])
# ---------- Suggest target ----------
def suggest_target_column(df_local):
    candidates = [c for c in df_local.columns if c.strip().lower() in ("target","label","y","outcome","survived")]
    if candidates:
        return candidates[0]
    for c in df_local.columns:
        try:
            if df_local[c].nunique() == 2:
                return c
        except Exception:
            continue
    num_cols = df_local.select_dtypes(include=["number"]).columns.tolist()
    return num_cols[-1] if num_cols else df_local.columns[0]

# ---------- Tab1: full dataset & dataset summary card ----------
with tab1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📋 Full dataset")
    st.dataframe(df, use_container_width=True)
    rows = df.shape[0]
    cols = df.shape[1]
    missing = int(df.isna().sum().sum())
    st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
    st.markdown("<h3>📊 Dataset Summary</h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='meta'><span class='stat-inline'><strong>{rows}</strong><br>Rows</span> <span class='stat-inline'><strong>{cols}</strong><br>Columns</span> <span class='stat-inline'><strong>{missing}</strong><br>Missing Values</span></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("🎯 Target selection")
    suggested = suggest_target_column(df)
    st.markdown(f"*Suggested target:* **`{suggested}`**")
    try:
        default_idx = list(df.columns).index(suggested)
    except Exception:
        default_idx = 0
    target_col = st.selectbox("Select target column (override if needed)", df.columns, index=default_idx)

    st.markdown("---")
    st.download_button("📥 Download CSV", data=df.to_csv(index=False), file_name="dataset.csv")
    st.markdown("</div>", unsafe_allow_html=True)

#  Run AutoML
if run_button:
    if (target_col is None) or (target_col not in df.columns):
        st.error("Please select a valid target column before running.")
        st.stop()
    prog = st.progress(0)
    status = st.empty()
    try:
        # small staged progress
        prog.progress(5); time.sleep(0.04)
        prog.progress(12); time.sleep(0.05)
        status.info("Cleaning & imputing data...")
        prog.progress(25); time.sleep(0.06)

        # run training inside spinner (correct usage)
        with st.spinner("Aria is training models and analyzing your data..."):
            agent = AutoMLAgent(df, target_col, llm_api_key if enable_llm else None)
            result = agent.run()

        # finalize progress
        status.empty()
        prog.progress(85); time.sleep(0.08)
        prog.progress(100); time.sleep(0.06)

        # present best model + score
        best_name = result.get("best_model_name", "N/A")
        best_score = result.get("best_score", None)
        problem_type = result.get("problem_type", None)
        if problem_type == "classification":
            metric_label = "Accuracy"
        elif problem_type == "regression":
            metric_label = "RMSE"
        else:
            metric_label = "Score"

        try:
            display_score = f"{best_score:.4f}" if best_score is not None else "N/A"
        except Exception:
            display_score = str(best_score)

        st.success(f"🎉 Your data is analyzed successfully! — Best model: **{best_name}** — {metric_label}: **{display_score}**")
        st.balloons()

        # ---------- Tab2: Visual Insights ----------
        with tab2:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.subheader("Correlation heatmap")
            try:
                fig = plot_correlation_heatmap(result["processed_df"])
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, key="corr_heatmap_v1")
                else:
                    st.info("No numeric columns for correlation heatmap.")
            except Exception:
                st.info("Heatmap rendering failed.")

            st.markdown("---")
            st.subheader("Pairwise plots (first numeric features)")
            try:
                fig = plot_pairwise(result["processed_df"], result["target_column"])
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, key="pairwise_v1")
                else:
                    st.info("Pairwise plot not available.")
            except Exception:
                st.info("Pairwise plotting failed.")
            st.markdown("</div>", unsafe_allow_html=True)

        # ---------- Tab3: Model Results ----------
        with tab3:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.subheader("Model leaderboard")
            try:
                fig = plot_model_comparison(result.get("model_performance", {}))
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, key="model_comp_v1")
                else:
                    st.info("Model comparison not available.")
            except Exception:
                st.info("Model comparison failed.")

            st.markdown("---")
            st.subheader("Feature importance (best model)")
            try:
                fig = plot_feature_importance(result.get("feature_importances", {}))
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, key="feat_imp_v1")
                else:
                    st.info("Feature importance not available.")
            except Exception:
                st.info("Feature importance failed.")

            st.markdown("---")
            st.markdown(
                f"""
                <div style='padding:10px;border-radius:8px;background:linear-gradient(90deg, rgba(23,162,255,0.03), rgba(107,44,255,0.02));'>
                    <strong>Best model:</strong> {best_name} &nbsp; • &nbsp;
                    <strong>{metric_label}:</strong> {display_score}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("---")
            st.subheader("Training details")
            st.json(result.get("training_details", {}))
            if result.get("trained_pipeline") is not None:
                try:
                    import joblib
                    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
                    joblib.dump(result["trained_pipeline"], tmpf.name)
                    with open(tmpf.name, "rb") as fh:
                        st.download_button("Download trained model (.pkl)", data=fh, file_name="trained_model.pkl")
                except Exception:
                    st.info("Model export not available.")
            st.markdown("</div>", unsafe_allow_html=True)
        with tab4:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.subheader("Explainability")

            shap_info = result.get("shap", None)
            if shap_info is not None and shap_info.get("shap_values", None) is not None:
                # try to render SHAP plot (plots.py handles backend & failures)
                try:
                    ok = plot_shap_summary(shap_info)
                    if ok is False or ok is None:
                        raise Exception("shap plot not rendered")
                except Exception:
                    st.info("SHAP computed but rendering failed. Showing feature importances instead.")
                    fig = plot_feature_importance(result.get("feature_importances", {}))
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="shap_fallback_imp_v1")
            else:
                st.info("SHAP not available or not produced. Showing feature importances and short insights.")
                fig = plot_feature_importance(result.get("feature_importances", {}))
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="shap_fallback_imp_v1")
                top_feats = result.get("feature_importances", {})
                if top_feats:
                    st.markdown("**Top features:**")
                    for f, v in sorted(top_feats.items(), key=lambda x: -abs(x[1]))[:8]:
                        st.markdown(f"- `{f}` : {v:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        status.empty()
        prog.empty()
        st.error("AutoML run failed: " + str(e))
        st.write(traceback.format_exc())

else:
    with tab2:
        st.info("▶ Run analysis to generate visual insights.")
    with tab3:
        st.info("▶ Run analysis to show model results.")
    with tab4:
        st.info("▶ Run analysis to enable explainability.")

# footer
st.markdown("<div style='text-align:center;color:#9fbcd8;padding-top:14px;'>Made with 💙 by Devika — Your AI Data Scientist</div>", unsafe_allow_html=True)
