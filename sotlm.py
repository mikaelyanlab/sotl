# app.py
# Robust Streamlit SoTL dashboard + text mining + ML explorer
# Built for your dataset structure:
# - Rows = students (45)
# - First column = anonymized student IDs (A..Z, AA..SS)
# - Remaining columns = responses to prompts
#
# Fixes included:
# - Uses true student ID column (no fake respondent_id)
# - Unique widget keys everywhere (prevents DuplicateElementId)
# - Safe slider bounds everywhere (prevents ValueBelowMinError / StreamlitAPIException)
# - Robust sparse indexing (CSR + X[idx, :]) (prevents TypeError at X_sub = X[idx])
# - Treats "None"/"nan"/"" as empty responses consistently
# - ML tab displays the selected text column (not text_cols[0]) and uses iloc correctly

import io
import re
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from scipy.sparse import hstack, issparse

st.set_page_config(page_title="AI Workflow SoTL Dashboard", layout="wide")

DEFAULT_PATH = "sheet1.csv"  # <-- put your CSV next to app.py in the repo


# -----------------------------
# Helpers
# -----------------------------
def normalize_colname(c: str) -> str:
    return re.sub(r"\s+", " ", str(c)).strip()


def to_clean_str(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"none", "nan", "na", "n/a"}:
        return ""
    return s


def nonempty_mask(series: pd.Series) -> pd.Series:
    s = series.apply(to_clean_str)
    return s.str.len() > 0


def detect_text_columns(df: pd.DataFrame, min_avg_len: int = 40) -> list[str]:
    cols = []
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].dropna().astype(str)
            if len(s) == 0:
                continue
            # ignore short “categorical-ish” columns that happen to be object
            if s.str.len().mean() >= min_avg_len:
                cols.append(c)
    return cols


def detect_step_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        name = str(c)
        if re.search(r"\bStep\s*\d+", name, flags=re.IGNORECASE):
            cols.append(c)
        elif "workflow" in name.lower() and "step" in name.lower():
            cols.append(c)
    return cols


def detect_likertish_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.85 and s.nunique(dropna=True) <= 12:
            cols.append(c)
    return cols


def safe_slider_int(label: str, min_v: int, max_v: int, default_v: int, step: int, key: str) -> int:
    # Guarantee Streamlit slider contract: min <= default <= max
    if max_v < min_v:
        max_v = min_v
    default_v = max(min_v, min(default_v, max_v))
    return st.slider(label, min_value=min_v, max_value=max_v, value=default_v, step=step, key=key)


@st.cache_data(show_spinner=False)
def load_df(file_bytes: bytes | None, path: str | None) -> pd.DataFrame:
    if file_bytes is not None:
        return pd.read_csv(io.BytesIO(file_bytes))
    if path is not None:
        return pd.read_csv(path)
    raise ValueError("No data source provided.")


# -----------------------------
# Load data
# -----------------------------
st.title("Student AI Workflow — Text Mining + ML Explorer")

with st.sidebar:
    st.header("Data input")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="upload_csv")
    use_default = st.checkbox("Use default CSV in repo", value=(uploaded is None), key="use_default_csv")

try:
    if uploaded is not None:
        df_raw = load_df(uploaded.getvalue(), None)
    else:
        df_raw = load_df(None, DEFAULT_PATH if use_default else None)
except Exception as e:
    st.error(f"Could not load CSV: {e}")
    st.stop()

df = df_raw.copy()
df.columns = [normalize_colname(c) for c in df.columns]
df = df.reset_index(drop=True)

# Student ID column selection (default: first column)
with st.sidebar:
    st.header("ID settings")
    id_col_default = df.columns[0]
    id_col = st.selectbox("Student ID column", df.columns.tolist(), index=0, key="id_col_select")
    st.caption("This should be your anonymized student name column (A..Z, AA..SS).")

df["student_id"] = df[id_col].astype(str).str.strip()
df["student_id"] = df["student_id"].replace({"nan": "", "None": ""})
df["student_id"] = df["student_id"].fillna("")
# If any blank IDs exist, make them stable
blank_mask = df["student_id"].astype(str).str.strip().eq("")
if blank_mask.any():
    df.loc[blank_mask, "student_id"] = [f"STU_{i:03d}" for i in np.where(blank_mask)[0]]

# Candidate column groups
all_cols = [c for c in df.columns if c not in {"student_id"}]
text_cols = [c for c in detect_text_columns(df[all_cols]) if c != id_col]
step_cols = [c for c in detect_step_columns(df[all_cols]) if c != id_col]
likert_cols = [c for c in detect_likertish_columns(df[all_cols]) if c != id_col]

tabs = st.tabs(["Overview", "Likert", "Workflow", "Text mining", "ML explorer", "Export"])
tab_overview, tab_likert, tab_workflow, tab_text, tab_ml, tab_export = tabs


# -----------------------------
# Overview
# -----------------------------
with tab_overview:
    c1, c2, c3 = st.columns(3)
    c1.metric("Students (rows)", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Detected long-text cols", f"{len(text_cols):,}")

    st.subheader("Column groups detected")
    st.write("**ID column:**", id_col)
    st.write("**Workflow/Step-like columns:**", step_cols if step_cols else "None detected")
    st.write("**Likert-ish numeric columns:**", likert_cols if likert_cols else "None detected")
    st.write("**Long-text columns:**", text_cols if text_cols else "None detected")

    st.subheader("Preview")
    preview_cols = ["student_id"] + [c for c in df.columns if c not in {"student_id"}][:6]
    st.dataframe(df[preview_cols].head(25), use_container_width=True, height=350)

    st.subheader("Missingness (fraction empty by column)")
    emptiness = {}
    for c in df.columns:
        if c == "student_id":
            continue
        emptiness[c] = (~nonempty_mask(df[c])).mean()
    miss_df = pd.DataFrame({"column": list(emptiness.keys()), "fraction_empty": list(emptiness.values())}).sort_values(
        "fraction_empty", ascending=False
    )
    st.dataframe(miss_df, use_container_width=True, height=420)


# -----------------------------
# Likert
# -----------------------------
with tab_likert:
    st.subheader("Likert / numeric ordinal distributions")
    if not likert_cols:
        st.info("No numeric Likert-ish columns detected (small discrete numeric ranges).")
    else:
        col = st.selectbox("Select item", likert_cols, key="likert_item_select")
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            st.warning("No numeric values found in this column.")
        else:
            counts = s.value_counts().sort_index()
            st.bar_chart(counts)
            st.write(f"Median: {float(s.median()):.2f} | IQR: {float(s.quantile(0.75) - s.quantile(0.25)):.2f}")
            st.dataframe(counts.reset_index().rename(columns={"index": "value", col: "count"}), use_container_width=True)


# -----------------------------
# Workflow
# -----------------------------
with tab_workflow:
    st.subheader("Workflow / Step response explorer")
    if not step_cols:
        st.info("No workflow/step columns detected.")
    else:
        step = st.selectbox("Select workflow/step column", step_cols, key="workflow_step_select")
        s_clean = df[step].apply(to_clean_str)
        mask = s_clean.str.len() > 0

        query = st.text_input("Search within responses", value="", key="workflow_search")
        if query.strip():
            mask = mask & s_clean.str.contains(query.strip(), case=False, na=False)

        out = pd.DataFrame({"student_id": df.loc[mask, "student_id"], "response": s_clean.loc[mask]})
        max_show = max(10, min(300, len(out)))
        show_n = safe_slider_int("Show first N matches", 10, max_show, min(50, max_show), 10, key="workflow_show_n")

        st.write(f"Matching responses: {len(out)}")
        st.dataframe(out.head(int(show_n)), use_container_width=True, height=520)


# -----------------------------
# Text mining (TF-IDF + KMeans) on ONE selected text column
# -----------------------------
with tab_text:
    st.subheader("Text mining (TF-IDF + KMeans)")

    if not text_cols:
        st.info("No long-text columns detected by heuristic.")
    else:
        text_col = st.selectbox("Text column", text_cols, key="tm_text_col_select")
        texts = df[text_col].apply(to_clean_str)
        texts = texts[texts.str.len() > 0]

        if len(texts) < 10:
            st.warning("Too few non-empty responses in this column for clustering (need ~10+).")
        else:
            # Sampling
            sample_n = safe_slider_int(
                "Sample size",
                min_v=10,
                max_v=len(texts),
                default_v=min(200, len(texts)),
                step=5,
                key="tm_sample_n",
            )
            # K must be <= sample_n
            k = safe_slider_int(
                "Number of clusters (k)",
                min_v=2,
                max_v=min(15, sample_n),
                default_v=min(6, min(15, sample_n)),
                step=1,
                key="tm_k",
            )
            top_terms_n = safe_slider_int(
                "Top TF-IDF terms",
                min_v=10,
                max_v=60,
                default_v=25,
                step=5,
                key="tm_top_terms",
            )

            rng = np.random.default_rng(42)
            sample_idx = rng.choice(texts.index.to_numpy(), size=int(sample_n), replace=False)
            texts_sample = texts.loc[sample_idx]

            vect = TfidfVectorizer(stop_words="english", min_df=2, max_features=4000, ngram_range=(1, 2))
            X = vect.fit_transform(texts_sample)

            # Top TF-IDF terms (global)
            mean_scores = np.asarray(X.mean(axis=0)).ravel()
            terms = np.array(vect.get_feature_names_out())
            top_idx = np.argsort(mean_scores)[::-1][:int(top_terms_n)]
            st.markdown("**Top TF-IDF terms:**")
            st.write(", ".join(terms[top_idx]))

            km = KMeans(n_clusters=int(k), n_init="auto", random_state=42)
            labels = km.fit_predict(X)

            st.markdown("**Cluster sizes:**")
            st.bar_chart(pd.Series(labels).value_counts().sort_index())

            chosen = st.selectbox("Inspect cluster", options=sorted(set(labels)), key="tm_cluster_select")
            cluster_rows = texts_sample.index[labels == chosen]
            cluster_out = pd.DataFrame(
                {"student_id": df.loc[cluster_rows, "student_id"], "response": df.loc[cluster_rows, text_col].apply(to_clean_str)}
            )

            # No fragile slider here; just show up to 50 or cluster size
            show_n = min(50, len(cluster_out))
            st.write(f"Cluster {chosen}: {len(cluster_out)} students (showing {show_n})")
            st.dataframe(cluster_out.head(show_n), use_container_width=True, height=520)


# -----------------------------
# ML explorer (Text / Workflow / Text+Workflow)
# -----------------------------
with tab_ml:
    st.subheader("ML Explorer (unsupervised, exploratory)")

    feature_mode = st.radio(
        "Feature space",
        ["Text only", "Workflow only", "Text + Workflow"],
        horizontal=True,
        key="ml_feature_mode",
    )

    # Selections
    selected_text_col = None
    selected_steps = []

    if feature_mode in ["Text only", "Text + Workflow"]:
        if not text_cols:
            st.warning("No text columns available for ML.")
        else:
            selected_text_col = st.selectbox("Text column", text_cols, key="ml_text_col_select")

    if feature_mode in ["Workflow only", "Text + Workflow"]:
        if not step_cols:
            st.warning("No workflow columns available for ML.")
        else:
            selected_steps = st.multiselect(
                "Workflow columns",
                options=step_cols,
                default=step_cols[:3],
                key="ml_steps_select",
            )

    # Build X blocks
    X_blocks = []
    feature_labels = []

    # Text block
    if selected_text_col is not None:
        all_text = df[selected_text_col].apply(to_clean_str)
        vect = TfidfVectorizer(stop_words="english", min_df=2, max_features=4000, ngram_range=(1, 2))
        X_text = vect.fit_transform(all_text)
        X_blocks.append(X_text)
        feature_labels.extend(list(vect.get_feature_names_out()))

    # Workflow block (binary non-empty)
    if selected_steps:
        wf = np.column_stack([nonempty_mask(df[c]).astype(int).to_numpy() for c in selected_steps])
        X_blocks.append(wf)
        feature_labels.extend(selected_steps)

    if not X_blocks:
        st.info("Select at least one feature source to run ML.")
        st.stop()

    X = hstack(X_blocks) if len(X_blocks) > 1 else X_blocks[0]

    # IMPORTANT: ensure stable indexing for sparse matrices
    if issparse(X):
        X = X.tocsr()

    n_rows = X.shape[0]

    sample_n = safe_slider_int(
        "Sample size (for speed)",
        min_v=10,
        max_v=n_rows,
        default_v=min(200, n_rows),
        step=5,
        key="ml_sample_n",
    )

    # k must be <= sample_n
    k = safe_slider_int(
        "Number of clusters (k)",
        min_v=2,
        max_v=min(15, sample_n),
        default_v=min(5, min(15, sample_n)),
        step=1,
        key="ml_k",
    )

    show_pca = st.checkbox("Show PCA scatter (2D)", value=True, key="ml_show_pca")

    rng = np.random.default_rng(42)
    idx = rng.choice(np.arange(n_rows), size=int(sample_n), replace=False).astype(int)
    X_sub = X[idx, :]  # robust for csr + dense

    km = KMeans(n_clusters=int(k), n_init="auto", random_state=42)
    clusters = km.fit_predict(X_sub)

    st.markdown("### Cluster sizes")
    st.bar_chart(pd.Series(clusters).value_counts().sort_index())

    chosen = st.selectbox("Inspect cluster", options=sorted(set(clusters)), key="ml_cluster_select")
    members = idx[clusters == chosen]

    # Top centroid features
    st.markdown("### Top centroid features (interpretive hints)")
    centroid = km.cluster_centers_[chosen]
    top_idx = np.argsort(centroid)[::-1][:25]
    top_feats = [feature_labels[i] for i in top_idx if i < len(feature_labels)]
    st.write(", ".join(top_feats) if top_feats else "No features available.")

    # Show member rows (student-level, because rows = students)
    st.markdown("### Cluster members (sampled)")
    base_cols = ["student_id"]
    if selected_text_col is not None:
        base_cols.append(selected_text_col)

    out_df = df.iloc[members][base_cols].copy()
    if selected_text_col is not None:
        out_df[selected_text_col] = out_df[selected_text_col].apply(to_clean_str)

    show_n = min(50, len(out_df))
    st.write(f"Cluster {chosen}: {len(out_df)} students (showing {show_n})")
    st.dataframe(out_df.head(show_n), use_container_width=True, height=520)

    # PCA visualization
    if show_pca:
        st.markdown("### PCA view (approximate)")
        try:
            X_dense = X_sub.toarray() if issparse(X_sub) else np.asarray(X_sub)
            if X_dense.shape[1] >= 2:
                pca = PCA(n_components=2, random_state=42)
                coords = pca.fit_transform(X_dense)
                viz = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "cluster": clusters})
                st.scatter_chart(viz, x="PC1", y="PC2", color="cluster")
            else:
                st.info("Not enough feature dimensions for PCA.")
        except Exception as e:
            st.warning(f"PCA failed: {e}")

    st.caption("These clusters are exploratory analytic profiles (strategy/role patterns), not measures of learning.")


# -----------------------------
# Export
# -----------------------------
with tab_export:
    st.subheader("Export tidy long table (student_id × item × response)")

    selectable = [c for c in df.columns if c not in {"student_id"}]
    default_cols = step_cols[:8] if step_cols else selectable[:8]

    cols = st.multiselect(
        "Columns to export",
        options=selectable,
        default=default_cols,
        key="export_cols_select",
    )

    long_df = df[["student_id"] + cols].melt(
        id_vars="student_id",
        var_name="item",
        value_name="response",
    )
    long_df["response"] = long_df["response"].apply(to_clean_str)

    st.dataframe(long_df.head(200), use_container_width=True, height=520)

    st.download_button(
        "Download tidy_long.csv",
        data=long_df.to_csv(index=False).encode("utf-8"),
        file_name="tidy_long.csv",
        mime="text/csv",
        key="export_download_btn",
    )
