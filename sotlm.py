# app.py
# Robust Streamlit SoTL dashboard with ML explorer
# GUARANTEED: no invalid widget bounds, no duplicate keys

import re
import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.sparse import hstack

st.set_page_config(page_title="AI Workflow SoTL Dashboard", layout="wide")

DEFAULT_PATH = "AI workflow student submissions - Sheet1.csv"

# --------------------------------------------------
# Utilities
# --------------------------------------------------
def normalize_colname(c):
    return re.sub(r"\s+", " ", str(c)).strip()

def detect_text_columns(df, min_len=40):
    cols = []
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].dropna().astype(str)
            if len(s) >= 5 and s.str.len().mean() >= min_len:
                cols.append(c)
    return cols

def detect_step_columns(df):
    return [
        c for c in df.columns
        if re.search(r"\bStep\s*\d+", str(c), re.I)
        or "Five steps in the project development workflow" in str(c)
    ]

def detect_likert_columns(df):
    cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.85 and s.nunique() <= 10:
            cols.append(c)
    return cols

@st.cache_data(show_spinner=False)
def load_df(file_bytes=None, path=None):
    if file_bytes is not None:
        return pd.read_csv(io.BytesIO(file_bytes))
    if path is not None:
        return pd.read_csv(path)
    raise ValueError("No CSV provided")

# --------------------------------------------------
# Load data
# --------------------------------------------------
st.title("Student AI Workflow — Exploratory + ML Dashboard")

with st.sidebar:
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="upload_csv")
    use_default = st.checkbox("Use default CSV", value=(uploaded is None), key="use_default")

try:
    if uploaded:
        df = load_df(file_bytes=uploaded.getvalue())
    else:
        df = load_df(path=DEFAULT_PATH if use_default else None)
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

df = df.copy()
df.columns = [normalize_colname(c) for c in df.columns]

if "respondent_id" not in df.columns:
    df.insert(0, "respondent_id", np.arange(1, len(df) + 1))

text_cols   = detect_text_columns(df)
step_cols   = detect_step_columns(df)
likert_cols = detect_likert_columns(df)

tabs = st.tabs(["Overview", "Likert", "Workflow", "Text mining", "ML explorer", "Export"])
tab_overview, tab_likert, tab_workflow, tab_text, tab_ml, tab_export = tabs

# --------------------------------------------------
# Overview
# --------------------------------------------------
with tab_overview:
    st.metric("Responses", len(df))
    st.metric("Columns", df.shape[1])
    st.metric("Text columns", len(text_cols))

    st.subheader("Preview")
    st.dataframe(df.head(25), use_container_width=True)

# --------------------------------------------------
# Likert
# --------------------------------------------------
with tab_likert:
    if not likert_cols:
        st.info("No numeric Likert-style columns detected.")
    else:
        col = st.selectbox("Likert item", likert_cols, key="likert_col")
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) >= 2:
            st.bar_chart(s.value_counts().sort_index())
            st.write(f"Median: {s.median():.2f}")
        else:
            st.warning("Not enough numeric data.")

# --------------------------------------------------
# Workflow
# --------------------------------------------------
with tab_workflow:
    if not step_cols:
        st.info("No workflow columns detected.")
    else:
        step = st.selectbox("Workflow step", step_cols, key="workflow_step")
        s = df[step].dropna().astype(str)
        st.dataframe(
            pd.DataFrame({
                "respondent_id": df.loc[s.index, "respondent_id"],
                "response": s
            }),
            use_container_width=True,
            height=500
        )

# --------------------------------------------------
# Text mining (SAFE)
# --------------------------------------------------
with tab_text:
    if not text_cols:
        st.info("No long-form text columns detected.")
    else:
        text_col = st.selectbox("Text column", text_cols, key="tm_text_col")
        texts = df[text_col].fillna("").astype(str)
        texts = texts[texts.str.len() > 0]

        if len(texts) < 10:
            st.warning("Too few responses for text mining.")
        else:
            max_sample = len(texts)
            sample_n = st.slider(
                "Sample size",
                min_value=10,
                max_value=max_sample,
                value=min(200, max_sample),
                step=10,
                key="tm_sample_n"
            )

            k = st.slider(
                "Number of clusters",
                min_value=2,
                max_value=min(10, sample_n),
                value=min(5, sample_n),
                step=1,
                key="tm_k"
            )

            rng = np.random.default_rng(42)
            idx = rng.choice(texts.index, size=sample_n, replace=False)
            texts_sample = texts.loc[idx]

            vect = TfidfVectorizer(stop_words="english", min_df=2, max_features=3000)
            X = vect.fit_transform(texts_sample)

            km = KMeans(n_clusters=k, n_init="auto", random_state=42)
            labels = km.fit_predict(X)

            st.bar_chart(pd.Series(labels).value_counts().sort_index())

            chosen = st.selectbox(
                "Inspect cluster",
                options=sorted(set(labels)),
                key="tm_cluster"
            )

            cluster_idx = texts_sample.index[labels == chosen]
            show_n = min(30, len(cluster_idx))

            st.dataframe(
                pd.DataFrame({
                    "respondent_id": df.loc[cluster_idx, "respondent_id"],
                    "response": df.loc[cluster_idx, text_col]
                }).head(show_n),
                use_container_width=True,
                height=500
            )

# --------------------------------------------------
# ML explorer (BULLETPROOF)
# --------------------------------------------------
with tab_ml:
    st.subheader("ML Explorer (unsupervised, exploratory)")

    feature_mode = st.radio(
        "Feature space",
        ["Text only", "Workflow only", "Text + Workflow"],
        horizontal=True,
        key="ml_mode"
    )

    X_blocks = []
    labels = []

    # Text features
    if feature_mode in ["Text only", "Text + Workflow"] and text_cols:
        text_col = st.selectbox("Text column", text_cols, key="ml_text_col")
        all_text = df[text_col].fillna("").astype(str)
        vect = TfidfVectorizer(stop_words="english", min_df=2, max_features=3000)
        X_text = vect.fit_transform(all_text)
        X_blocks.append(X_text)
        labels.extend(vect.get_feature_names_out())

    # Workflow features
    if feature_mode in ["Workflow only", "Text + Workflow"] and step_cols:
        steps = st.multiselect(
            "Workflow columns",
            step_cols,
            default=step_cols[:3],
            key="ml_steps"
        )
        if steps:
            wf = df[steps].notna().astype(int).values
            X_blocks.append(wf)
            labels.extend(steps)

    if not X_blocks:
        st.info("Select at least one feature source.")
        st.stop()

    X = hstack(X_blocks) if len(X_blocks) > 1 else X_blocks[0]
    n_rows = X.shape[0]

    sample_n = st.slider(
        "Sample size",
        min_value=10,
        max_value=n_rows,
        value=min(200, n_rows),
        step=10,
        key="ml_sample_n"
    )

    k = st.slider(
        "Number of clusters",
        min_value=2,
        max_value=min(10, sample_n),
        value=min(5, sample_n),
        step=1,
        key="ml_k"
    )

    rng = np.random.default_rng(42)
    idx = rng.choice(np.arange(n_rows), size=sample_n, replace=False)
    X_sub = X[idx]

    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    clusters = km.fit_predict(X_sub)

    st.bar_chart(pd.Series(clusters).value_counts().sort_index())

    chosen = st.selectbox(
        "Inspect cluster",
        options=sorted(set(clusters)),
        key="ml_cluster"
    )

    members = idx[clusters == chosen]
    show_n = min(30, len(members))

    if text_cols:
        st.dataframe(
            df.loc[members, ["respondent_id", text_cols[0]]].head(show_n),
            use_container_width=True,
            height=500
        )

# --------------------------------------------------
# Export
# --------------------------------------------------
with tab_export:
    cols = st.multiselect(
        "Columns to export (long format)",
        options=[c for c in df.columns if c != "respondent_id"],
        default=step_cols[:5] if step_cols else df.columns[1:6].tolist(),
        key="export_cols"
    )

    long_df = df[["respondent_id"] + cols].melt(
        id_vars="respondent_id",
        var_name="item",
        value_name="response"
    )

    st.dataframe(long_df.head(200), use_container_width=True)

    st.download_button(
        "Download tidy_long.csv",
        data=long_df.to_csv(index=False).encode("utf-8"),
        file_name="tidy_long.csv",
        mime="text/csv",
        key="export_btn"
    )
