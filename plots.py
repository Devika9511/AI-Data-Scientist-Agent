# plots.py — plotting helpers (Plotly + SHAP-safe)
import plotly.express as px
import numpy as np
import streamlit as st

# Try import shap & matplotlib only if available
try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

def plot_correlation_heatmap(df):
    """Return plotly heatmap figure for numeric correlations or None if not possible"""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] == 0:
        return None
    corr = num_df.corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
    fig.update_layout(template="plotly_dark", height=600)
    return fig

def plot_pairwise(df, target_col, max_features=6):
    """Return a scatter matrix for first numeric features (if available)"""
    num_df = df.select_dtypes(include=[np.number])
    cols = num_df.columns.tolist()[:max_features]
    if len(cols) == 0:
        return None
    try:
        if target_col in df.columns:
            plot_df = df[cols + [target_col]] if target_col not in cols else df[cols]
        else:
            plot_df = df[cols]
        fig = px.scatter_matrix(plot_df, dimensions=cols, title="Pairwise (numeric features)")
        fig.update_layout(template="plotly_dark", height=700)
        return fig
    except Exception:
        return None

def plot_target_distribution(df, target_col, problem_type="classification"):
    """Return histogram for the target column"""
    if target_col not in df.columns:
        return None
    if problem_type == "classification":
        fig = px.histogram(df, x=target_col, title="Target Distribution")
    else:
        fig = px.histogram(df, x=target_col, nbins=40, title="Target Distribution")
    fig.update_layout(template="plotly_dark", height=400)
    return fig

def plot_feature_importance(feat_imp, top_n=20):
    """feat_imp: dict {feature: importance}"""
    if not feat_imp:
        return None
    items = sorted(feat_imp.items(), key=lambda x: -abs(x[1]))[:top_n]
    names = [i[0] for i in items]
    vals = [i[1] for i in items]
    fig = px.bar(x=vals, y=names, orientation="h", title="Feature importance")
    fig.update_layout(template="plotly_dark", yaxis={'categoryorder':'total ascending'}, height=600)
    return fig

def plot_model_comparison(perf_dict):
    """perf_dict: {model_name: score}"""
    if not perf_dict:
        return None
    names = list(perf_dict.keys())
    scores = [perf_dict[k] for k in names]
    df = {"model": names, "score": scores}
    fig = px.bar(df, x="model", y="score", title="Model comparison (higher is better)")
    fig.update_layout(template="plotly_dark", height=400)
    return fig

def plot_shap_summary(shap_dict):
    """
    Attempt to render SHAP summary plot using matplotlib backend.
    shap_dict expected shape: {'explainer': explainer, 'shap_values': shap_values, 'feature_names': feature_names}
    Returns True if plotted, False otherwise.
    """
    if not SHAP_AVAILABLE:
        return False

    try:
        shap_values = shap_dict.get("shap_values", None)
        feature_names = shap_dict.get("feature_names", None)

        if shap_values is None:
            return False

        # shap_values may be list (multi-class) -> pick index 1 or mean
        vals = shap_values
        if isinstance(shap_values, list):
            idx = 1 if len(shap_values) > 1 else 0
            vals = shap_values[idx]

        plt.figure(figsize=(8,6))
        try:
            shap.summary_plot(vals, features=None if feature_names is None else None, feature_names=feature_names, show=False)
        except Exception:
            shap.summary_plot(vals, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
        plt.clf()
        return True
    except Exception:
        return False
