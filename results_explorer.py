"""Read-only Streamlit explorer for committed KNN result tables."""

from pathlib import Path

import pandas as pd
import streamlit as st

from analysis import available_metrics, load_csv_results, rank_results, summarize_results


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

st.set_page_config(page_title="KNN Experiment Explorer", page_icon="📊", layout="wide")
st.title("KNN Experiment Explorer")
st.caption("MNIST and letter-recognition results from committed experiment artifacts")
st.info(
    "This is a read-only portfolio explorer. It compares recorded CSV outputs and does not retrain models, "
    "download data, or provide a prediction service."
)

try:
    results = load_csv_results(RESULTS_DIR)
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

summary = summarize_results(results)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Recorded rows", summary["rows"])
col2.metric("Best accuracy", f"{summary['best_accuracy']:.2%}")
col3.metric("Best metric", summary["best_metric"].title())
col4.metric("Best K", summary["best_k"])

st.sidebar.header("Filters")
metrics = list(available_metrics(results))
selected_metrics = st.sidebar.multiselect("Distance metric", metrics, default=metrics)
k_values = sorted(int(value) for value in results["K"].unique())
selected_k = st.sidebar.multiselect("K value", k_values, default=k_values)

filtered = results[
    results["Metric"].isin(selected_metrics) & results["K"].isin(selected_k)
].copy()
if filtered.empty:
    st.warning("No recorded rows match the selected filters.")
    st.stop()

st.subheader("Accuracy comparison")
chart = filtered.pivot_table(index="K", columns="Metric", values="Accuracy", aggfunc="max")
st.line_chart(chart)

st.subheader("Ranked experiment results")
display = rank_results(filtered).copy()
display["Metric"] = display["Metric"].str.title()
display["Accuracy"] = display["Accuracy"].map(lambda value: f"{value:.2%}")
st.dataframe(
    display[["Source", "Metric", "K", "Accuracy"]],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Artifact notes")
st.markdown(
    "The consolidated project retains the original notebooks, CSV/XLSX exports, and written report. "
    "The accuracy values shown here are recorded outputs from the source experiments; reproduce the notebooks "
    "with documented data paths and preprocessing before making a new benchmark claim."
)
