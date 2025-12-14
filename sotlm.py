# app.py
# Streamlit dashboard + ML explorer for student AI workflow CSV (SoTL-friendly, exploratory)

import re
import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from scipy.sparse import hstack  # requires scipy

st.set_page_config(page_title="AI Workflow SoTL Dashboard", layout="wide")

# If you run locally, keep the CSV in the same folder as this app.py
DEFAULT_PATH = "AI workflow student submissions - Sheet1.csv"


# ----------------------------
# Helpers
# ----------------------------
def normalize_colname(c: str) -> str:
    return re.sub(r"\s+", " ", str(c)).strip()


def detect_text_columns(df: pd.DataFrame, min_avg_len: int = 40) -> list[str]:
    """Heuristic: long average string length suggests open-ended response columns."""
    text_cols = []
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].dropna().astype(str)
            if len(s) == 0:
                continue
            if s.str.len().mean() >= min_avg_len:
                text_cols.append(c)
    return text_cols


LIKERT_KEYWORDS = [
    "How comfortable", "To what extent", "extent", "comfortable",
    "responsible", "effective", "comfort", "recommend"
]


def detect_likertish_columns(df: pd.DataFrame) -> list[str]:
    """Detect Likert-ish columns by name keywords OR numeric small-range integers."""
    candidates = []
    for c in df.columns:
        name = str(c)
        if any(k.lower() in name.lower() for k in LIKERT_KEYWORDS):
            candidates.append(c)
            continue

        s = df[c].dropna()
        if len(s) == 0:
            continue

        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().mean() > 0.9:
            unique = sorted(s_num.dropna().unique())
            if len(unique) <= 10 and (min(unique) >= 0) and (max(unique) <= 10):
                candidates.append(c)

    # Deduplicate, preserve order
    out, seen = [], set()
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def detect_step_columns(df: pd.DataFrame) -> list[str]:
    """Detect columns that look like 'Step 1 ...' or 'Five steps ...' workflow items."""
    step_cols = []
    for c in df.columns:
        name = str(c)
        if re.search(r"\bStep\s*\d+", name, flags=re.IGNORECASE):
            step_cols.append(c)
        elif "Five steps in the project development workflow" in name:
            step_cols.append(c)
    return step_cols


def tidy_long(df: pd.DataFrame, id_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    out = df[id_cols + value_cols].copy()
    long = out.melt(id_vars=id_cols, value_vars=value_cols, var_name="item", value_name="response")
    long["response"] = long["response"].astype(str)
    long["is_missing"] = long["response"].isin(["nan", "None", ""])
    return long


def top_tfidf_terms(texts: pd.Series, n_terms: int = 20, max_features: int = 5000) -> pd.DataFrame:
    vect = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=3
    )
    X = vect.fit_transform(texts)
    mean_scores = np.asarray(X.mean(axis=0)).ravel()
    terms = np.array(vect.get_feature_names_out())
    top_idx = np.argsort(mean_scores)[::-1][:n_terms]
    return pd.DataFrame({"term": terms[top_idx], "mean_tfidf": mean_scores[top_idx]})


def cluster_texts(texts: pd.Series, k: int = 6, max_features: int = 8000):
    vect = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=3
    )
    X = vect.fit_transform(texts)
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)

    terms = np.array(vect.get_feature_names_out())
    centroids = km.cluster_centers_
    top_terms = {}
    for i in range(k):
        top_idx = np.argsort(centroids[i])[::-1][:12]
        top_terms[i] = ", ".join(terms[top_idx])

    return labels, top_terms


# ----------------------------
# Load data
# ----------------------------
st.title("Student AI Workflow Dataset — Exploratory + ML Explorer")

with st.sidebar:
    st.header("Data input")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    use_example = st.checkbox("Use local default CSV filename", value=(uploaded is None))

@st.cache_data(show_spinner=False)
def load_df(file_bytes: bytes | None, path: str | None):
    if file_bytes is not None:
        return pd.read_csv(io.BytesIO(file_bytes))
    if path is not None:
        return pd.read_csv(path)
    raise ValueError("No data provided")

try:
    if uploaded is not None:
        df = load_df(uploaded.getvalue(), None)
    else:
        df = load_df(None, DEFAULT_PATH if use_example else None)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

df.columns = [normalize_colname(c) for c in df.columns]

# Create respondent_id if not present
df = df.copy()
if "respondent_id" not in df.columns:
    df.insert(0, "respondent_id", np.arange(1, len(df) + 1))

# Detect column groups
text_cols = detect_text_columns(df)
likert_cols = detect_likertish_columns(df)
step_cols = detect_step_columns(df)

# ----------------------------
# Tabs
# ----------------------------
tab_overview, tab_likert, tab_steps, tab_text, tab_ml, tab_export = st.tabs(
    ["Overview", "Likert / Ordinal", "Workflow Steps", "Open-Text Mining", "ML Explorer", "Export / Tidy"]
)

# ----------------------------
# Overview
# ----------------------------
with tab_overview:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Text-heavy columns (heuristic)", f"{len(text_cols):,}")

    st.subheader("Missingness by column")
    miss = df.isna().mean().sort_values(ascending=False)
    miss_df = pd.DataFrame({"column": miss.index, "missing_fraction": miss.values})
    st.dataframe(miss_df, use_container_width=True, height=420)

    st.subheader("Column groups detected")
    st.write("**Workflow / step columns (detected):**")
    st.write(step_cols if step_cols else "None detected.")
    st.write("**Likert-ish columns (detected):**")
    st.write(likert_cols if likert_cols else "None detected.")
    st.write("**Open-text columns (detected):**")
    st.write(text_cols if text_cols else "None detected.")

    st.subheader("Preview")
    st.dataframe(df.head(30), use_container_width=True, height=350)

# ----------------------------
# Likert / Ordinal
# ----------------------------
with tab_likert:
    st.subheader("Likert / ordinal distributions (auto-detected)")
    if not likert_cols:
        st.info("No likert-ish columns detected. If your scale items are labeled differently, we can add a manual selector.")
    else:
        col = st.selectbox("Select item", likert_cols)
        s = df[col].dropna()

        # Try numeric
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().mean() > 0.9:
            counts = s_num.value_counts().sort_index()
            st.bar_chart(counts)
            st.write(f"Median: {float(s_num.median()):.2f} | IQR: {float(s_num.quantile(0.75) - s_num.quantile(0.25)):.2f}")
            counts_df = counts.reset_index()
            counts_df.columns = ["value", "count"]
        else:
            counts = s.astype(str).value_counts()
            st.bar_chart(counts)
            counts_df = counts.reset_index()
            counts_df.columns = ["value", "count"]

        st.write("Raw value counts:")
        st.dataframe(counts_df, use_container_width=True)

# ----------------------------
# Workflow Steps
# ----------------------------
with tab_steps:
    st.subheader("Workflow step explorer")
    if not step_cols:
        st.info("No workflow-step columns detected. We can switch this tab to a manual column selector if needed.")
    else:
        step = st.selectbox("Select step / workflow question", step_cols)
        s = df[step].dropna().astype(str)

        c1, c2 = st.columns([2, 1])
        with c1:
            query = st.text_input("Search within responses (case-insensitive)", value="")
        with c2:
            show_n = st.number_input("Show N responses", min_value=10, max_value=500, value=50, step=10)

        if query.strip():
            mask = s.str.contains(query.strip(), case=False, na=False)
            s2 = s[mask]
        else:
            s2 = s

        st.write(f"Showing {min(show_n, len(s2))} of {len(s2)} responses for: **{step}**")
        out_df = pd.DataFrame({
            "respondent_id": df.loc[s2.index, "respondent_id"].astype(int).values,
            "response": s2.values
        }).head(int(show_n))
        st.dataframe(out_df, use_container_width=True, height=520)

# ----------------------------
# Open-Text Mining
# ----------------------------
with tab_text:
    st.subheader("Open-text mining (keywords + clustering)")
    if not text_cols:
        st.info("No long-form text columns detected by heuristic. You can lower the min_avg_len threshold in detect_text_columns().")
    else:
        text_col = st.selectbox("Select open-text column", text_cols)
        texts = df[text_col].dropna().astype(str)
        texts = texts[texts.str.len() > 0]

        c1, c2, c3 = st.columns(3)
        with c1:
            n_terms = st.number_input("Top TF-IDF terms", min_value=10, max_value=60, value=25, step=5)
        with c2:
            k = st.number_input("Number of clusters (KMeans)", min_value=2, max_value=12, value=6, step=1)
        with c3:
            sample_n = st.number_input(
                "Sample size for clustering (speed)",
                min_value=200,
                max_value=int(min(2000, len(texts))) if len(texts) > 0 else 200,
                value=int(min(1000, len(texts))) if len(texts) > 0 else 200,
                step=100
            )

        if len(texts) == 0:
            st.warning("Selected column has no non-empty text.")
        else:
            texts_sample = texts.sample(int(sample_n), random_state=42) if len(texts) > sample_n else texts

            st.markdown("### Top keywords (TF-IDF)")
            try:
                kw = top_tfidf_terms(texts_sample, n_terms=int(n_terms))
                st.dataframe(kw, use_container_width=True, height=300)
            except Exception as e:
                st.warning(f"TF-IDF failed (often due to too much missing/short text): {e}")

            st.markdown("### Response clusters (interpret as usage/meaning profiles, not truth)")
            try:
                labels, top_terms = cluster_texts(texts_sample, k=int(k))
                clustered = pd.DataFrame({
                    "respondent_id": texts_sample.index.map(lambda i: int(df.loc[i, "respondent_id"])),
                    "cluster": labels,
                    "response": texts_sample.values
                })
                cluster_counts = clustered["cluster"].value_counts().sort_index()
                st.bar_chart(cluster_counts)

                st.markdown("**Cluster topic hints (top terms):**")
                for i in range(int(k)):
                    st.write(f"- Cluster {i}: {top_terms.get(i, '')}")

                chosen_cluster = st.selectbox("Inspect a cluster", list(range(int(k))))
                show_cluster_n = st.slider("Show responses", min_value=10, max_value=200, value=30, step=10)
                st.dataframe(
                    clustered[clustered["cluster"] == chosen_cluster].head(int(show_cluster_n)),
                    use_container_width=True,
                    height=420
                )
            except Exception as e:
                st.warning(f"Clustering failed: {e}")

# ----------------------------
# ML Explorer (CSV-level ML: text only / workflow only / combined)
# ----------------------------
with tab_ml:
    st.subheader("Machine Learning Explorer (Pattern Discovery)")

    st.markdown(
        """
        This panel uses **unsupervised** machine learning to surface latent patterns in the CSV.
        The goal is **exploratory structure discovery** (strategies/profiles), not prediction of outcomes.
        """
    )

    st.markdown("### 1) Choose feature space")
    feature_mode = st.radio(
        "Feature set",
        ["Text only", "Workflow only", "Text + Workflow"],
        horizontal=True
    )

    # Text feature selection
    text_feature_col = None
    if feature_mode in ["Text only", "Text + Workflow"]:
        if not text_cols:
            st.warning("No text columns detected for text-based ML. Adjust the text detection heuristic or select Workflow only.")
        else:
            text_feature_col = st.selectbox("Open-text column for ML", text_cols)

    # Workflow feature selection
    selected_steps = []
    if feature_mode in ["Workflow only", "Text + Workflow"]:
        if not step_cols:
            st.warning("No workflow step columns detected. Adjust step detection or use Text only.")
        else:
            selected_steps = st.multiselect(
                "Workflow/step columns (binary: response present vs missing)",
                step_cols,
                default=step_cols[:4] if len(step_cols) >= 4 else step_cols
            )

    # Build feature matrix
    X_blocks = []
    feature_labels = []

    # Text block (sparse)
    if feature_mode in ["Text only", "Text + Workflow"] and text_feature_col:
        texts = df[text_feature_col].fillna("").astype(str)

        vect = TfidfVectorizer(
            stop_words="english",
            min_df=3,
            max_features=3000,
            ngram_range=(1, 2)
        )
        X_text = vect.fit_transform(texts)
        X_blocks.append(X_text)
        feature_labels.extend(list(vect.get_feature_names_out()))

    # Workflow block (dense -> convert to sparse implicitly via hstack)
    if feature_mode in ["Workflow only", "Text + Workflow"] and selected_steps:
        wf = df[selected_steps].notna().astype(int).values
        X_blocks.append(wf)
        feature_labels.extend(selected_steps)

    if len(X_blocks) == 0:
        st.info("Select at least one feature source (text column and/or workflow columns).")
    else:
        # Combine blocks
        if len(X_blocks) == 1:
            X = X_blocks[0]
        else:
            X = hstack(X_blocks)

        st.markdown("### 2) Choose clustering + visualization settings")
        k = st.slider("Number of clusters (k)", min_value=2, max_value=12, value=5)
        do_pca = st.checkbox("Show PCA scatter (2D)", value=True)

        # Fit clustering
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(X)

        df_ml = df.copy()
        df_ml["cluster"] = labels

        st.markdown("### 3) Cluster overview")
        counts = df_ml["cluster"].value_counts().sort_index()
        st.bar_chart(counts)

        st.markdown("### 4) Inspect and interpret clusters")
        chosen_cluster = st.selectbox("Select cluster", sorted(df_ml["cluster"].unique()))
        cluster_df = df_ml[df_ml["cluster"] == chosen_cluster]

        st.write(f"**Cluster {chosen_cluster}** — {len(cluster_df)} respondents")

        # Representative responses
        if text_feature_col:
            st.markdown("#### Representative responses (top 30)")
            st.dataframe(
                cluster_df[["respondent_id", text_feature_col]].head(30),
                use_container_width=True,
                height=420
            )

        # Top features defining cluster (centroid)
        st.markdown("#### Features most associated with this cluster (centroid top weights)")
        centroid = km.cluster_centers_[chosen_cluster]

        # If X is sparse from TF-IDF, centroid aligns with feature_labels length
        top_idx = np.argsort(centroid)[::-1][:25]
        top_feats = [feature_labels[i] for i in top_idx if i < len(feature_labels)]
        st.write(", ".join(top_feats) if top_feats else "No features available.")

        # PCA visualization
        if do_pca:
            st.markdown("### 5) Low-dimensional view (PCA)")
            # PCA needs dense array; ok for moderate datasets/features
            X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
            if X_dense.shape[1] < 2:
                st.info("Not enough feature dimensions for PCA.")
            else:
                pca = PCA(n_components=2, random_state=42)
                coords = pca.fit_transform(X_dense)

                viz_df = pd.DataFrame({
                    "PC1": coords[:, 0],
                    "PC2": coords[:, 1],
                    "cluster": labels,
                    "respondent_id": df["respondent_id"].astype(int).values
                })
                st.scatter_chart(viz_df, x="PC1", y="PC2", color="cluster")

        st.caption(
            "Interpret clusters as *analytic profiles* of AI-use language/workflow, not as objective categories or measures of learning."
        )

# ----------------------------
# Export / Tidy
# ----------------------------
with tab_export:
    st.subheader("Export a tidy long table for coding / analysis (R-friendly)")

    default_value_cols = step_cols[:10] if step_cols else [c for c in df.columns if c not in ["respondent_id"]][:10]
    value_cols = st.multiselect(
        "Select columns to include as items",
        options=[c for c in df.columns if c != "respondent_id"],
        default=default_value_cols
    )
    id_cols = ["respondent_id"]

    long_df = tidy_long(df, id_cols=id_cols, value_cols=value_cols)
    st.dataframe(long_df.head(200), use_container_width=True, height=420)

    csv_bytes = long_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download tidy_long.csv",
        data=csv_bytes,
        file_name="tidy_long.csv",
        mime="text/csv"
    )

st.caption(
    "SoTL note: use this dashboard to map workflows/roles/strategies. Avoid outcome claims unless your design supports them."
)
