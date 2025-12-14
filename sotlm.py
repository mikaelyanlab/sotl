# app.py
# Streamlit dashboard + ML explorer for student AI workflow CSV
# Robust for Streamlit Cloud: unique widget keys + safe bounds

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


# ----------------------------
# Helpers
# ----------------------------
def normalize_colname(c: str) -> str:
    return re.sub(r"\s+", " ", str(c)).strip()


def detect_text_columns(df: pd.DataFrame, min_avg_len: int = 40) -> list[str]:
    """Heuristic: long average string length suggests open-ended response columns."""
    cols = []
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].dropna().astype(str)
            if len(s) == 0:
                continue
            if s.str.len().mean() >= min_avg_len:
                cols.append(c)
    return cols


def detect_step_columns(df: pd.DataFrame) -> list[str]:
    """Detect columns that look like 'Step 1...' or contain the 'Five steps...' phrase."""
    cols = []
    for c in df.columns:
        name = str(c)
        if re.search(r"\bStep\s*\d+", name, flags=re.IGNORECASE):
            cols.append(c)
        elif "Five steps in the project development workflow" in name:
            cols.append(c)
    return cols


def detect_likertish_columns(df: pd.DataFrame) -> list[str]:
    """Detect Likert-ish columns by numeric coercion + small unique value count."""
    cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if len(s.dropna()) == 0:
            continue
        if s.notna().mean() > 0.85:
            uniq = np.sort(s.dropna().unique())
            if len(uniq) <= 12 and np.min(uniq) >= 0 and np.max(uniq) <= 10:
                cols.append(c)
    return cols


def safe_slider_bounds(n: int, default: int = 1000, min_allowed: int = 10):
    """Return (min_value, max_value, value) always consistent for Streamlit slider."""
    if n <= 0:
        return (min_allowed, min_allowed, min_allowed)
    min_v = min_allowed
    max_v = max(min_allowed, n)
    val = max(min_allowed, min(default, n))
    return (min_v, max_v, val)


@st.cache_data(show_spinner=False)
def load_df(file_bytes: bytes | None, path: str | None) -> pd.DataFrame:
    if file_bytes is not None:
        return pd.read_csv(io.BytesIO(file_bytes))
    if path is not None:
        return pd.read_csv(path)
    raise ValueError("No data source provided.")


# ----------------------------
# UI: Data load
# ----------------------------
st.title("Student AI Workflow — Exploratory + ML Dashboard")

with st.sidebar:
    st.header("Data input")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="sidebar_upload_csv")
    use_default = st.checkbox("Use default CSV filename", value=(uploaded is None), key="sidebar_use_default")


try:
    if uploaded is not None:
        df = load_df(uploaded.getvalue(), None)
    else:
        df = load_df(None, DEFAULT_PATH if use_default else None)
except Exception as e:
    st.error(f"Could not load CSV: {e}")
    st.stop()

df = df.copy()
df.columns = [normalize_colname(c) for c in df.columns]

if "respondent_id" not in df.columns:
    df.insert(0, "respondent_id", np.arange(1, len(df) + 1))

text_cols = detect_text_columns(df)
step_cols = detect_step_columns(df)
likert_cols = detect_likertish_columns(df)

tab_overview, tab_likert, tab_workflow, tab_text, tab_ml, tab_export = st.tabs(
    ["Overview", "Likert", "Workflow", "Text mining", "ML explorer", "Export"]
)


# ----------------------------
# Overview tab
# ----------------------------
with tab_overview:
    c1, c2, c3 = st.columns(3)
    c1.metric("Responses (rows)", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Detected text columns", f"{len(text_cols):,}")

    st.subheader("Detected column groups")
    st.write("**Likert-ish:**", likert_cols if likert_cols else "None detected")
    st.write("**Workflow/step columns:**", step_cols if step_cols else "None detected")
    st.write("**Open-text columns:**", text_cols if text_cols else "None detected")

    st.subheader("Missingness (fraction missing)")
    miss = df.isna().mean().sort_values(ascending=False)
    st.dataframe(miss.to_frame("fraction_missing"), use_container_width=True, height=350)

    st.subheader("Preview")
    st.dataframe(df.head(30), use_container_width=True, height=350)


# ----------------------------
# Likert tab
# ----------------------------
with tab_likert:
    st.subheader("Likert / ordinal distributions")

    if not likert_cols:
        st.info("No numeric Likert-like columns were auto-detected. If you have labeled Likert text, we can add manual selection.")
    else:
        col = st.selectbox("Select Likert item", likert_cols, key="likert_select_item")
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            st.warning("No numeric values in this column after coercion.")
        else:
            counts = s.value_counts().sort_index()
            st.bar_chart(counts)
            st.write(f"Median: {float(s.median()):.2f} | IQR: {float(s.quantile(0.75) - s.quantile(0.25)):.2f}")
            st.dataframe(counts.reset_index().rename(columns={"index": "value", col: "count"}),
                         use_container_width=True)


# ----------------------------
# Workflow tab
# ----------------------------
with tab_workflow:
    st.subheader("Workflow responses explorer")

    if not step_cols:
        st.info("No workflow/step columns detected.")
    else:
        step = st.selectbox("Select workflow step column", step_cols, key="workflow_select_step")
        s = df[step].dropna().astype(str)
        query = st.text_input("Search within responses", value="", key="workflow_search")

        if query.strip():
            s = s[s.str.contains(query.strip(), case=False, na=False)]

        show_n = st.slider("Show first N matches", 10, max(10, min(500, len(s))), min(50, max(10, len(s))),
                           key="workflow_show_n")

        out = pd.DataFrame({
            "respondent_id": df.loc[s.index, "respondent_id"].astype(int).values,
            "response": s.values
        }).head(int(show_n))

        st.dataframe(out, use_container_width=True, height=520)


# ----------------------------
# Text mining tab (TF-IDF + clustering) — FIXED keys + SAFE sample bounds
# ----------------------------
with tab_text:
    st.subheader("Open-text mining (TF-IDF + clustering)")

    if not text_cols:
        st.info("No long-form text columns detected by heuristic.")
    else:
        text_col = st.selectbox("Text column", text_cols, key="textmining_select_textcol")
        texts = df[text_col].fillna("").astype(str)
        texts = texts[texts.str.len() > 0]

        if len(texts) < 10:
            st.warning("Too few non-empty responses in this column for stable text mining (need ~10+).")
        else:
            # SAFE bounds for sample size
            min_v, max_v, val = safe_slider_bounds(len(texts), default=1000, min_allowed=10)
            sample_n = st.slider(
                "Sample size for text mining (speed)",
                min_value=min_v,
                max_value=max_v,
                value=val,
                step=10,
                key="textmining_sample_n"
            )

            n_terms = st.slider("Top TF-IDF terms", 10, 60, 25, 5, key="textmining_n_terms")
            k = st.slider("Number of clusters (KMeans)", 2, 12, 6, 1, key="textmining_k")

            # Sample deterministically for stability
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(texts.index.values, size=int(sample_n), replace=False)
            texts_sample = texts.loc[sample_idx]

            vect = TfidfVectorizer(stop_words="english", min_df=2, max_features=4000, ngram_range=(1, 2))
            X = vect.fit_transform(texts_sample)

            # TF-IDF top terms (global)
            mean_scores = np.asarray(X.mean(axis=0)).ravel()
            terms = np.array(vect.get_feature_names_out())
            top_idx = np.argsort(mean_scores)[::-1][:int(n_terms)]
            top_terms = terms[top_idx]
            st.markdown("**Top TF-IDF terms:**")
            st.write(", ".join(top_terms))

            # KMeans clustering on sampled text
            km = KMeans(n_clusters=int(k), n_init="auto", random_state=42)
            labels = km.fit_predict(X)

            counts = pd.Series(labels).value_counts().sort_index()
            st.bar_chart(counts)

            chosen_cluster = st.selectbox("Inspect cluster", sorted(counts.index.tolist()),
                                          key="textmining_inspect_cluster")

            cluster_rows = texts_sample.index[labels == chosen_cluster]
            show_cluster_n = st.slider("Show responses in cluster", 10, min(200, len(cluster_rows)),
                                       min(30, len(cluster_rows)), 10,
                                       key="textmining_show_cluster_n")

            st.dataframe(
                pd.DataFrame({
                    "respondent_id": df.loc[cluster_rows, "respondent_id"].astype(int).values,
                    "response": df.loc[cluster_rows, text_col].astype(str).values
                }).head(int(show_cluster_n)),
                use_container_width=True,
                height=520
            )


# ----------------------------
# ML explorer tab — FIXED keys + SAFE sample bounds + PCA optional
# ----------------------------
with tab_ml:
    st.subheader("ML Explorer (Unsupervised pattern discovery)")

    feature_mode = st.radio(
        "Feature space",
        ["Text only", "Workflow only", "Text + Workflow"],
        horizontal=True,
        key="ml_feature_mode"
    )

    text_feature_col = None
    if feature_mode in ["Text only", "Text + Workflow"]:
        if not text_cols:
            st.warning("No text columns detected.")
        else:
            # IMPORTANT: unique key (different from text mining tab)
            text_feature_col = st.selectbox("Text column", text_cols, key="ml_select_textcol")

    selected_steps = []
    if feature_mode in ["Workflow only", "Text + Workflow"]:
        if not step_cols:
            st.warning("No workflow/step columns detected.")
        else:
            selected_steps = st.multiselect(
                "Workflow columns (binary: response present vs missing)",
                step_cols,
                default=step_cols[:3] if len(step_cols) >= 3 else step_cols,
                key="ml_select_steps"
            )

    # Build feature matrix
    X_blocks = []
    feature_labels = []

    if text_feature_col is not None:
        all_text = df[text_feature_col].fillna("").astype(str)
        vect = TfidfVectorizer(stop_words="english", min_df=2, max_features=4000, ngram_range=(1, 2))
        X_text = vect.fit_transform(all_text)
        X_blocks.append(X_text)
        feature_labels.extend(list(vect.get_feature_names_out()))

    if selected_steps:
        wf = df[selected_steps].notna().astype(int).values
        X_blocks.append(wf)
        feature_labels.extend(selected_steps)

    if not X_blocks:
        st.info("Select at least one feature source.")
        st.stop()

    X = hstack(X_blocks) if len(X_blocks) > 1 else X_blocks[0]

    # Sample size (for speed) — SAFE bounds
    n_rows = X.shape[0]
    min_v, max_v, val = safe_slider_bounds(n_rows, default=1000, min_allowed=10)
    sample_n = st.slider(
        "Sample size (for speed)",
        min_value=min_v,
        max_value=max_v,
        value=val,
        step=10,
        key="ml_sample_n"
    )

    rng = np.random.RandomState(42)
    idx = rng.choice(np.arange(n_rows), size=int(sample_n), replace=False)
    X_sub = X[idx]

    k = st.slider("Number of clusters (k)", 2, 12, 5, 1, key="ml_k")
    show_pca = st.checkbox("Show PCA scatter (2D)", value=True, key="ml_show_pca")

    km = KMeans(n_clusters=int(k), n_init="auto", random_state=42)
    clusters = km.fit_predict(X_sub)

    st.markdown("### Cluster sizes")
    counts = pd.Series(clusters).value_counts().sort_index()
    st.bar_chart(counts)

    chosen = st.selectbox("Inspect cluster", sorted(counts.index.tolist()), key="ml_inspect_cluster")
    members = idx[clusters == chosen]

    st.write(f"Cluster {chosen} — {len(members)} respondents")

    # Top features
    centroid = km.cluster_centers_[chosen]
    top_idx = np.argsort(centroid)[::-1][:25]
    top_feats = [feature_labels[i] for i in top_idx if i < len(feature_labels)]
    st.markdown("**Top centroid features:**")
    st.write(", ".join(top_feats) if top_feats else "No features available.")

    # Show representative responses
    if text_feature_col is not None:
        st.markdown("**Representative responses (first 30):**")
        st.dataframe(
            df.loc[members, ["respondent_id", text_feature_col]].head(30),
            use_container_width=True,
            height=420
        )

    # PCA visualization (optional)
    if show_pca:
        st.markdown("### PCA view (approximate)")
        try:
            X_dense = X_sub.toarray() if hasattr(X_sub, "toarray") else np.asarray(X_sub)
            if X_dense.shape[1] >= 2:
                pca = PCA(n_components=2, random_state=42)
                coords = pca.fit_transform(X_dense)
                viz = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "cluster": clusters})
                st.scatter_chart(viz, x="PC1", y="PC2", color="cluster")
            else:
                st.info("Not enough feature dimensions for PCA.")
        except Exception as e:
            st.warning(f"PCA failed (often due to huge feature matrices): {e}")

    st.caption("Interpret clusters as exploratory analytic profiles (strategies/roles), not measures of learning.")


# ----------------------------
# Export tab
# ----------------------------
with tab_export:
    st.subheader("Export a tidy long table for coding / analysis")

    default_cols = step_cols[:10] if step_cols else [c for c in df.columns if c not in ["respondent_id"]][:10]
    selected_cols = st.multiselect(
        "Columns to include as items (long format)",
        options=[c for c in df.columns if c != "respondent_id"],
        default=default_cols,
        key="export_select_cols"
    )

    long_df = df[["respondent_id"] + selected_cols].melt(
        id_vars=["respondent_id"],
        var_name="item",
        value_name="response"
    )

    st.dataframe(long_df.head(200), use_container_width=True, height=420)

    st.download_button(
        "Download tidy_long.csv",
        data=long_df.to_csv(index=False).encode("utf-8"),
        file_name="tidy_long.csv",
        mime="text/csv",
        key="export_download_button"
    )
