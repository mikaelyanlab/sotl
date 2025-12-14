# app.py
# SoTL AI Workflow Dashboard (student-level rows)
# Dataset confirmed: 45 students (rows), first column = anonymized IDs (A..SS)

import io
import re
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.sparse import hstack, issparse

st.set_page_config(page_title="SoTL AI Workflow Dashboard", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def normalize_colname(c: str) -> str:
    return re.sub(r"\s+", " ", str(c)).strip()


def clean_text(x) -> str:
    """Turn NaN/None/'None'/'nan' etc into empty string; strip whitespace."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"none", "nan", "na", "n/a"}:
        return ""
    return s


def nonempty(series: pd.Series) -> pd.Series:
    """Boolean mask: response exists and isn't just None/nan/empty."""
    s = series.apply(clean_text)
    return s.str.len() > 0


@st.cache_data(show_spinner=False)
def load_csv(uploaded_bytes: bytes | None, default_path: str | None) -> pd.DataFrame:
    if uploaded_bytes is not None:
        return pd.read_csv(io.BytesIO(uploaded_bytes))
    if default_path is not None:
        return pd.read_csv(default_path)
    raise ValueError("No CSV provided.")


def safe_slider_int(label, min_v, max_v, default_v, step, key):
    """Guarantee Streamlit slider contract: min <= value <= max."""
    if max_v < min_v:
        max_v = min_v
    default_v = max(min_v, min(int(default_v), int(max_v)))
    return st.slider(label, min_value=int(min_v), max_value=int(max_v), value=int(default_v), step=int(step), key=key)


# -----------------------------
# Load data
# -----------------------------
st.title("Student AI Workflow — Text Mining + ML Explorer")

DEFAULT_PATH = "sheet1.csv"  # keep your CSV in repo root next to app.py

with st.sidebar:
    st.header("Data input")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="upload_csv_v1")
    use_default = st.checkbox("Use default sheet1.csv in repo", value=(uploaded is None), key="use_default_v1")

try:
    df_raw = load_csv(uploaded.getvalue() if uploaded else None, DEFAULT_PATH if use_default else None)
except Exception as e:
    st.error("Failed to load CSV.")
    st.exception(e)
    st.stop()

df = df_raw.copy()
df.columns = [normalize_colname(c) for c in df.columns]
df = df.reset_index(drop=True)

with st.sidebar:
    st.header("ID column")
    id_col = st.selectbox("Select student ID column", df.columns.tolist(), index=0, key="id_col_v1")

df["student_id"] = df[id_col].astype(str).str.strip().replace({"nan": "", "None": ""}).fillna("")
blank = df["student_id"].str.len().eq(0)
if blank.any():
    df.loc[blank, "student_id"] = [f"STU_{i:03d}" for i in np.where(blank)[0]]

# All prompt columns (everything except the ID col and derived student_id)
prompt_cols = [c for c in df.columns if c not in {id_col, "student_id"}]

tabs = st.tabs(["Overview", "Explore", "Text mining", "ML explorer", "Export"])
tab_overview, tab_explore, tab_text, tab_ml, tab_export = tabs


# -----------------------------
# Overview
# -----------------------------
with tab_overview:
    st.metric("Students (rows)", len(df))
    st.metric("Prompt columns", len(prompt_cols))

    st.subheader("Preview")
    st.dataframe(df[["student_id"] + prompt_cols[:5]].head(20), use_container_width=True)

    st.subheader("Column emptiness")
    emptiness = []
    for c in prompt_cols:
        emptiness.append((c, float((~nonempty(df[c])).mean())))
    miss = pd.DataFrame(emptiness, columns=["column", "fraction_empty"]).sort_values("fraction_empty", ascending=False)
    st.dataframe(miss, use_container_width=True, height=420)


# -----------------------------
# Explore (simple browsing/search)
# -----------------------------
with tab_explore:
    st.subheader("Browse responses by prompt")

    col = st.selectbox("Prompt column", prompt_cols, key="explore_col_v1")
    s = df[col].apply(clean_text)

    only_nonempty = st.checkbox("Show only non-empty responses", value=True, key="explore_nonempty_v1")
    if only_nonempty:
        mask = s.str.len() > 0
    else:
        mask = pd.Series(True, index=df.index)

    q = st.text_input("Search (case-insensitive contains)", value="", key="explore_search_v1").strip()
    if q:
        mask = mask & s.str.contains(q, case=False, na=False)

    out = pd.DataFrame({"student_id": df.loc[mask, "student_id"], "response": s.loc[mask]})
    st.write(f"Rows shown: {len(out)}")
    st.dataframe(out, use_container_width=True, height=520)


# -----------------------------
# Text mining (TF-IDF + KMeans) on a single chosen prompt column
# -----------------------------
with tab_text:
    st.subheader("Text mining (TF-IDF + KMeans)")
    text_col = st.selectbox("Text column to analyze", prompt_cols, key="tm_textcol_v2")

    texts_all = df[text_col].apply(clean_text)
    mask = texts_all.str.len() > 0
    texts = texts_all.loc[mask]

    if len(texts) < 10:
        st.warning(f"Too few non-empty responses in this column for clustering (non-empty = {len(texts)}).")
    else:
        sample_n = safe_slider_int(
            "Sample size (speed)",
            min_v=10,
            max_v=len(texts),
            default_v=min(45, len(texts)),
            step=1,
            key="tm_sample_v2",
        )

        k = safe_slider_int(
            "Number of clusters (k)",
            min_v=2,
            max_v=min(12, sample_n),
            default_v=min(6, min(12, sample_n)),
            step=1,
            key="tm_k_v2",
        )

        top_terms_n = safe_slider_int(
            "Top TF-IDF terms to display",
            min_v=10,
            max_v=60,
            default_v=25,
            step=5,
            key="tm_terms_v2",
        )

        try:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(texts.index.to_numpy(), size=int(sample_n), replace=False)
            texts_sample = texts.loc[sample_idx]

            vect = TfidfVectorizer(stop_words="english", min_df=2, max_features=4000, ngram_range=(1, 2))
            X = vect.fit_transform(texts_sample)

            mean_scores = np.asarray(X.mean(axis=0)).ravel()
            terms = np.array(vect.get_feature_names_out())
            top_idx = np.argsort(mean_scores)[::-1][:int(top_terms_n)]
            st.markdown("**Top TF-IDF terms:**")
            st.write(", ".join(terms[top_idx]))

            km = KMeans(n_clusters=int(k), n_init="auto", random_state=42)
            labels = km.fit_predict(X)

            st.markdown("**Cluster sizes:**")
            st.bar_chart(pd.Series(labels).value_counts().sort_index())

            chosen = st.selectbox("Inspect cluster", sorted(set(labels)), key="tm_cluster_v2")
            cluster_rows = texts_sample.index[labels == chosen]

            out = pd.DataFrame({
                "student_id": df.loc[cluster_rows, "student_id"],
                "response": df.loc[cluster_rows, text_col].apply(clean_text),
            })

            st.write(f"Cluster {chosen}: {len(out)} students")
            st.dataframe(out, use_container_width=True, height=520)

        except Exception as e:
            st.error("Text mining failed.")
            st.exception(e)


# -----------------------------
# ML explorer (Text / Workflow / Text+Workflow)
# Workflow columns are treated as binary: non-empty response or not
# -----------------------------
with tab_ml:
    st.subheader("ML explorer (unsupervised)")

    mode = st.radio(
        "Feature space",
        ["Text only", "Workflow only", "Text + Workflow"],
        horizontal=True,
        key="ml_mode_v3",
    )

    selected_text_col = None
    selected_workflow_cols = []

    if mode in ["Text only", "Text + Workflow"]:
        selected_text_col = st.selectbox("Text column", prompt_cols, key="ml_textcol_v3")

    if mode in ["Workflow only", "Text + Workflow"]:
        selected_workflow_cols = st.multiselect(
            "Workflow columns (binary presence/absence)",
            options=prompt_cols,
            default=[],
            key="ml_workflowcols_v3",
        )

    X_blocks = []
    feature_labels = []

    # Text block
    if selected_text_col is not None:
        all_text = df[selected_text_col].apply(clean_text)

        # If too sparse, fail early with a clear message
        if (all_text.str.len() > 0).sum() < 10:
            st.warning("Selected text column has too few non-empty responses for TF-IDF (need ~10+).")
        else:
            vect = TfidfVectorizer(stop_words="english", min_df=2, max_features=4000, ngram_range=(1, 2))
            X_text = vect.fit_transform(all_text)
            X_blocks.append(X_text)
            feature_labels.extend(list(vect.get_feature_names_out()))

    # Workflow block: binary presence/absence
    if selected_workflow_cols:
        wf = np.column_stack([nonempty(df[c]).astype(int).to_numpy() for c in selected_workflow_cols])
        X_blocks.append(wf)
        feature_labels.extend(selected_workflow_cols)

    if not X_blocks:
        st.info("Choose a text column and/or at least one workflow column.")
        st.stop()

    X = hstack(X_blocks) if len(X_blocks) > 1 else X_blocks[0]

    # Critical: stable sparse format + robust row slicing
    if issparse(X):
        X = X.tocsr()

    n_rows = X.shape[0]
    sample_n = safe_slider_int("Sample size (speed)", 10, n_rows, min(45, n_rows), 1, key="ml_sample_v3")
    k = safe_slider_int("Number of clusters (k)", 2, min(12, sample_n), min(5, min(12, sample_n)), 1, key="ml_k_v3")

    try:
        rng = np.random.default_rng(42)
        idx = rng.choice(np.arange(n_rows), size=int(sample_n), replace=False).astype(int)
        X_sub = X[idx, :]  # robust for CSR + dense

        km = KMeans(n_clusters=int(k), n_init="auto", random_state=42)
        clusters = km.fit_predict(X_sub)

        st.markdown("**Cluster sizes:**")
        st.bar_chart(pd.Series(clusters).value_counts().sort_index())

        chosen = st.selectbox("Inspect cluster", sorted(set(clusters)), key="ml_cluster_v3")
        members = idx[clusters == chosen]

        # Display members
        show_cols = ["student_id"]
        if selected_text_col is not None:
            show_cols.append(selected_text_col)

        out = df.iloc[members][show_cols].copy()
        if selected_text_col is not None:
            out[selected_text_col] = out[selected_text_col].apply(clean_text)

        st.write(f"Cluster {chosen}: {len(out)} students (showing all; max 45 anyway)")
        st.dataframe(out, use_container_width=True, height=520)

    except Exception as e:
        st.error("ML explorer failed.")
        st.exception(e)


# -----------------------------
# Export
# -----------------------------
with tab_export:
    st.subheader("Export tidy long format (student_id × item × response)")

    cols = st.multiselect(
        "Columns to export",
        options=prompt_cols,
        default=prompt_cols[:8],
        key="export_cols_v1",
    )

    long_df = df[["student_id"] + cols].melt(
        id_vars="student_id",
        var_name="item",
        value_name="response",
    )
    long_df["response"] = long_df["response"].apply(clean_text)

    st.dataframe(long_df.head(200), use_container_width=True, height=520)
    st.download_button(
        "Download tidy_long.csv",
        data=long_df.to_csv(index=False).encode("utf-8"),
        file_name="tidy_long.csv",
        mime="text/csv",
        key="export_btn_v1",
    )
