"""Streamlit dashboard for experiment visualization."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from arachne.tracking.tracker import load_all_experiments

MODELS_DIR = Path("models")

st.set_page_config(
    page_title="Arachne — Table Classification",
    page_icon="🕷",
    layout="wide",
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

@st.cache_data(ttl=30)
def get_experiments() -> list[dict]:
    return load_all_experiments(MODELS_DIR)


def experiments_to_df(experiments: list[dict]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        test = exp.get("test_metrics", {})
        cv = exp.get("cv_results", {})
        rows.append({
            "ID": exp.get("experiment_id", ""),
            "Name": exp.get("experiment_name", ""),
            "Timestamp": exp.get("timestamp", "")[:19].replace("T", " "),
            "Accuracy": test.get("accuracy", None),
            "Macro F1": test.get("macro_f1", None),
            "Weighted F1": test.get("weighted_f1", None),
            "CV Accuracy": cv.get("mean_accuracy", None),
            "CV Std": cv.get("std_accuracy", None),
            "Duration (s)": exp.get("duration_seconds", None),
            "Status": exp.get("status", ""),
            "_path": exp.get("_path", ""),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

st.sidebar.title("🕷 Arachne")
page = st.sidebar.radio("Navigation", ["Overview", "Experiment Details", "Compare"])
st.sidebar.markdown("---")
if st.sidebar.button("Refresh experiments"):
    st.cache_data.clear()

# ──────────────────────────────────────────────
# Page: Overview
# ──────────────────────────────────────────────

if page == "Overview":
    st.title("Experiments Overview")

    experiments = get_experiments()
    if not experiments:
        st.warning(f"No experiments found in `{MODELS_DIR}/`. Run a training first.")
        st.code("python scripts/train.py --config configs/experiments/tfidf_logistic.yaml")
        st.stop()

    df = experiments_to_df(experiments)

    # Summary KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total experiments", len(df))
    best_acc = df["Accuracy"].max()
    col2.metric("Best accuracy", f"{best_acc:.4f}" if pd.notna(best_acc) else "N/A")
    best_name = df.loc[df["Accuracy"].idxmax(), "Name"] if pd.notna(best_acc) else "N/A"
    col3.metric("Best model", best_name)

    st.markdown("---")

    # Main table
    st.subheader("All runs")
    display_cols = ["Name", "Timestamp", "Accuracy", "Macro F1", "CV Accuracy", "CV Std", "Duration (s)", "Status"]
    st.dataframe(
        df[display_cols].style.format({
            "Accuracy": "{:.4f}",
            "Macro F1": "{:.4f}",
            "CV Accuracy": "{:.4f}",
            "CV Std": "{:.4f}",
            "Duration (s)": "{:.1f}",
        }, na_rep="—").highlight_max(subset=["Accuracy", "Macro F1"], color="#c6efce"),
        use_container_width=True,
    )

    # Accuracy bar chart
    if df["Accuracy"].notna().any():
        st.subheader("Accuracy comparison")
        fig = px.bar(
            df.dropna(subset=["Accuracy"]).sort_values("Accuracy", ascending=True),
            x="Accuracy", y="Name", orientation="h",
            color="Accuracy", color_continuous_scale="Blues",
            text="Accuracy",
        )
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(height=max(300, len(df) * 50), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# Page: Experiment Details
# ──────────────────────────────────────────────

elif page == "Experiment Details":
    st.title("Experiment Details")

    experiments = get_experiments()
    if not experiments:
        st.warning("No experiments found.")
        st.stop()

    df = experiments_to_df(experiments)
    selected_id = st.selectbox("Select experiment", df["ID"].tolist())

    exp = next(e for e in experiments if e["experiment_id"] == selected_id)
    exp_path = Path(exp["_path"])

    col1, col2, col3, col4 = st.columns(4)
    test = exp.get("test_metrics", {})
    col1.metric("Accuracy", f"{test.get('accuracy', 0):.4f}")
    col2.metric("Macro F1", f"{test.get('macro_f1', 0):.4f}")
    col3.metric("Weighted F1", f"{test.get('weighted_f1', 0):.4f}")
    col4.metric("Duration", f"{exp.get('duration_seconds', 0):.1f}s")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Per-class metrics", "Confusion matrix", "CV results", "Config"])

    with tab1:
        per_class = test.get("per_class", {})
        if per_class:
            pc_df = pd.DataFrame(per_class).T.reset_index().rename(columns={"index": "class"})
            fig = px.bar(
                pc_df.melt(id_vars="class", value_vars=["precision", "recall", "f1"]),
                x="class", y="value", color="variable", barmode="group",
                labels={"value": "Score", "class": "Class", "variable": "Metric"},
                color_discrete_sequence=["#4C72B0", "#DD8452", "#55A868"],
            )
            fig.update_layout(yaxis_range=[0, 1.05])
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                pd.DataFrame(per_class).T.style.format("{:.4f}", subset=["precision", "recall", "f1"]),
                use_container_width=True,
            )

    with tab2:
        cm_path = exp_path / "plots" / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path))
        else:
            st.info("Confusion matrix image not found.")

    with tab3:
        cv = exp.get("cv_results", {})
        if cv:
            col1, col2 = st.columns(2)
            col1.metric("Mean accuracy", f"{cv.get('mean_accuracy', 0):.4f}")
            col2.metric("Std", f"{cv.get('std_accuracy', 0):.4f}")

            fold_scores = cv.get("fold_scores", [])
            if fold_scores:
                fig = px.bar(
                    x=list(range(1, len(fold_scores) + 1)),
                    y=fold_scores,
                    labels={"x": "Fold", "y": "Accuracy"},
                    title="Cross-validation fold scores",
                )
                fig.add_hline(y=cv["mean_accuracy"], line_dash="dash", line_color="red",
                              annotation_text=f"Mean: {cv['mean_accuracy']:.4f}")
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cross-validation results for this experiment.")

    with tab4:
        config_path = exp_path / "config.yaml"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                st.code(f.read(), language="yaml")
        else:
            st.json(exp)


# ──────────────────────────────────────────────
# Page: Compare
# ──────────────────────────────────────────────

elif page == "Compare":
    st.title("Compare Experiments")

    experiments = get_experiments()
    if len(experiments) < 2:
        st.warning("You need at least 2 experiments to compare.")
        st.stop()

    df = experiments_to_df(experiments)
    selected_ids = st.multiselect(
        "Select experiments to compare",
        df["ID"].tolist(),
        default=df["ID"].tolist()[:min(4, len(df))],
    )

    if len(selected_ids) < 2:
        st.info("Select at least 2 experiments.")
        st.stop()

    selected_exps = [e for e in experiments if e["experiment_id"] in selected_ids]
    compare_df = df[df["ID"].isin(selected_ids)]

    st.subheader("Global metrics comparison")
    metrics_cols = ["Accuracy", "Macro F1", "Weighted F1", "CV Accuracy"]
    available = [c for c in metrics_cols if compare_df[c].notna().any()]

    fig = go.Figure()
    for _, row in compare_df.iterrows():
        fig.add_trace(go.Bar(
            name=row["Name"],
            x=available,
            y=[row[m] for m in available],
            text=[f"{row[m]:.4f}" if pd.notna(row[m]) else "N/A" for m in available],
            textposition="outside",
        ))
    fig.update_layout(barmode="group", yaxis_range=[0, 1.1], height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Per-class F1 comparison")
    all_classes = set()
    for exp in selected_exps:
        all_classes.update(exp.get("test_metrics", {}).get("per_class", {}).keys())
    all_classes = sorted(all_classes)

    if all_classes:
        pc_rows = []
        for exp in selected_exps:
            pc = exp.get("test_metrics", {}).get("per_class", {})
            for cls in all_classes:
                pc_rows.append({
                    "Experiment": exp["experiment_name"],
                    "Class": cls,
                    "F1": pc.get(cls, {}).get("f1", None),
                })
        pc_df = pd.DataFrame(pc_rows)
        fig2 = px.bar(
            pc_df.dropna(subset=["F1"]),
            x="Class", y="F1", color="Experiment", barmode="group",
        )
        fig2.update_layout(yaxis_range=[0, 1.1])
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Raw metrics table")
    st.dataframe(
        compare_df[["Name", "Timestamp", "Accuracy", "Macro F1", "Weighted F1", "CV Accuracy", "Duration (s)"]].style.format({
            "Accuracy": "{:.4f}",
            "Macro F1": "{:.4f}",
            "Weighted F1": "{:.4f}",
            "CV Accuracy": "{:.4f}",
            "Duration (s)": "{:.1f}",
        }, na_rep="—"),
        use_container_width=True,
    )
