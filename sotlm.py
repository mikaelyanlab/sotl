# app.py
# Streamlit dashboard + ML explorer for student AI workflow CSV
# SoTL-safe, exploratory, and robust to small datasets

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
# Helper functions
# --------------------------------------------------
def normalize_colname(c):
    return re.sub(r"\s+", " ", str(c)).strip()

def detect_text_columns(df, min_avg_len=40):
    cols = []
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].dropna().astype(str)
            if len(s) > 0 and s.str.len().mean() >= min_avg_len:
                cols.append(c)
    return cols

def detect_step_columns(df):
    cols = []
    for c in df.columns:
        if re.search(r"\bStep\s*\d+", str(c), re.IGNORECASE):
            cols.append(c)
        elif "Five steps in the project development workflow" in str(c):
            cols.append(c)
    return cols

def detect_likertish_columns(df):
    cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.9:
            uniq = s.dropna().unique()
            if len(uniq) <= 10:
                cols.append(c)
    return cols

# --------------------------------------------------
# Load data
# --------------------------------------------------
st.title("Student AI Workflow — Exploratory + ML Dashboard")

with st.sidebar:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    use_default = st.checkbox("Use default CSV", value=(uploaded is None))

@st.cache_data
def load_df(file_bytes=None, path=None):
    if file_bytes is not None:
        return pd.read_csv(io.BytesIO(file_bytes))
    if path is not None:
        return pd.read_csv(path)
    raise ValueError("No data source provided")

try:
    if uploaded:
        df = load_df(uploaded.getvalue())
    else:
        df = load_df(path=DEFAULT_PATH if use_default else None)
except Exception as e:
    st.error(f"Could not load CSV: {e}")
    st.stop()

df.columns = [normalize_colname(c) for c in df.columns]

if "respondent_id" not in df.columns:
    df.insert(0, "respondent_id", range(1, len(df) + 1))

text_cols = detect_text_columns(df)
step_cols = detect_step_columns(df)
likert_cols = detect_likertish_columns(df)

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab_overview, tab_likert, tab_steps, tab_text, tab_ml = st.tabs(
    ["Overview", "Likert", "Workflow", "Text Mining", "ML Explorer"]
)

# --------------------------------------------------
# Overview
# --------------------------------------------------
with tab_overview:
    st.metric("Responses", len(df))
    st.metric("Columns", len(df.columns))
    st.metric("Text columns", len(text_cols))

    st.subheader("Missingness")
    miss = df.isna().mean().sort_values(ascending=False)
    st.dataframe(miss.to_frame("fraction missing"), height=350)

    st.subheader("Preview")
    st.dataframe(df.head(20), height=350)

# --------------------------------------------------
# Likert
# --------------------------------------------------
with tab_likert:
    if not likert_cols:
        st.info("No Likert-style numeric columns detected.")
    else:
        col = st.selectbox("Select Likert item", likert_cols)
        counts = df[col].value_counts().sort_index()
        st.bar_chart(counts)
        st.write(f"Median: {df[col].median():.2f}")

# --------------------------------------------------
# Workflow
# --------------------------------------------------
with tab_steps:
    if not step_cols:
        st.info("No workflow step columns detected.")
    else:
        step = st.selectbox("Workflow step", step_cols)
        s = df[step].dropna().astype(str)

        query = st.text_input("Search responses")
        if query:
            s = s[s.str.contains(query, case=False)]

        st.dataframe(
            pd.DataFrame({
                "respondent_id": df.loc[s.index, "respondent_id"],
                "response": s
            }),
            height=500
        )

# --------------------------------------------------
# Text Mining
# --------------------------------------------------
with tab_text:
    if not text_cols:
        st.info("No long-form text columns detected.")
    else:
        text_col = st.selectbox("Text column", text_cols)
        texts = df[text_col].dropna().astype(str)
        texts = texts[texts.str.len() > 0]

        if len(texts) < 10:
            st.warning("Too few responses for text mining.")
        else:
            vect = TfidfVectorizer(
                stop_words="english",
                min_df=2,
                max_features=2000
            )
            X = vect.fit_transform(texts)
            scores = np.asarray(X.mean(axis=0)).ravel()
            terms = np.array(vect.get_feature_names_out())
            top = terms[np.argsort(scores)[::-1][:25]]

            st.write("Top TF-IDF terms:")
            st.write(", ".join(top))

# --------------------------------------------------
# ML Explorer (FIXED)
# --------------------------------------------------
with tab_ml:
    st.subheader("ML Explorer (Unsupervised Pattern Discovery)")

    feature_mode = st.radio(
        "Feature space",
        ["Text only", "Workflow only", "Text + Workflow"],
        horizontal=True
    )

    # ---- Text features ----
    text_feature_col = None
    if feature_mode in ["Text only", "Text + Workflow"]:
        if not text_cols:
            st.warning("No text columns available.")
        else:
            text_feature_col = st.selectbox("Text column", text_cols)

    # ---- Workflow features ----
    selected_steps = []
    if feature_mode in ["Workflow only", "Text + Workflow"]:
        if not step_cols:
            st.warning("No workflow columns available.")
        else:
            selected_steps = st.multiselect(
                "Workflow columns",
                step_cols,
                default=step_cols[:3]
            )

    X_blocks = []
    labels = []

    # Text block
    if text_feature_col:
        texts = df[text_feature_col].fillna("").astype(str)
        vect = TfidfVectorizer(
            stop_words="english",
            min_df=2,
            max_features=3000
        )
        X_text = vect.fit_transform(texts)
        X_blocks.append(X_text)
        labels.extend(vect.get_feature_names_out())

    # Workflow block
    if selected_steps:
        wf = df[selected_steps].notna().astype(int).values
        X_blocks.append(wf)
        labels.extend(selected_steps)

    if not X_blocks:
        st.info("Select at least one feature source.")
        st.stop()

    X = hstack(X_blocks) if len(X_blocks) > 1 else X_blocks[0]

    # ---- SAFE sample size slider ----
    n = X.shape[0]
    min_n = 10
    max_n = max(min_n, n)
    default_n = max(min_n, min(1000, n))

    sample_n = st.slider(
        "Sample size (for speed)",
        min_value=min_n,
        max_value=max_n,
        value=default_n,
        step=10
    )

    idx = np.random.RandomState(42).choice(n, sample_n, replace=False)
    X_sub = X[idx]

    k = st.slider("Number of clusters", 2, 12, 5)

    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    clusters = km.fit_predict(X_sub)

    st.bar_chart(pd.Series(clusters).value_counts().sort_index())

    chosen = st.selectbox("Inspect cluster", sorted(set(clusters)))
    rows = idx[clusters == chosen]

    st.write(f"Cluster {chosen} — {len(rows)} responses")

    if text_feature_col:
        st.dataframe(
            df.loc[rows, ["respondent_id", text_feature_col]].head(30),
            height=400
        )

    st.caption(
        "Clusters are exploratory analytic profiles, not measures of learning or performance."
    )
