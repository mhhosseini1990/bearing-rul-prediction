"""Six-page fixed48 direct-RUL research dashboard."""
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from results import discover_results  # noqa: E402

FINAL = ROOT / "outputs/final/model/fixed48_direct_final"
st.set_page_config(page_title="Bearing RUL Research", layout="wide")


@st.cache_data
def csv(path):
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


state = discover_results(FINAL)
page = st.sidebar.radio("Research workspace", ["Project Overview", "Dataset Explorer", "RUL Prediction",
                                                "Training Evidence", "Maintenance Replay", "Research Results"])
st.sidebar.caption("Saved exploratory evidence; not live industrial monitoring")
st.sidebar.success(f"Verified final checkpoints: {state['verified_count']}/30")

if page == "Project Overview":
    st.title("Fixed48 direct bearing-RUL prediction")
    st.write("LSTM and GRU use identical fixed48 inputs and validation-only checkpoint selection.")
    a, b, c, d = st.columns(4)
    a.metric("XJTU bearings", 15); b.metric("Fixed features", 48)
    c.metric("Verified checkpoints", f"{state['verified_count']}/30")
    d.metric("Completed runs", f"{state['completed_count']}/30")
    if state["complete"]:
        st.success("Final experiment artifacts are complete: all 30 fold/model/seed runs are provenance-verified.")
    else:
        st.warning("Final experiment is incomplete or contains invalid artifacts. Available verified results are shown below.")
        if state["missing"]:
            st.write("Missing runs:", ", ".join(f"{f}/{m}/seed{s}" for f, m, s in state["missing"]))
        if state["invalid"]:
            st.dataframe(pd.DataFrame(state["invalid"]), hide_index=True)
    if not state["model_summary"].empty:
        st.subheader("LSTM versus GRU — bearing-macro test metrics")
        st.dataframe(state["model_summary"], hide_index=True)
        metric = state["model_summary"].melt(
            id_vars="model", value_vars=["mean_macro_mae_minutes", "mean_macro_rmse_minutes"],
            var_name="metric", value_name="minutes")
        st.plotly_chart(px.bar(metric, x="model", y="minutes", color="metric", barmode="group"),
                        width="stretch")
    st.caption("Primary scores use complete held-out bearings and the common observation range step ≥ 29.")
elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    data = csv(ROOT / "data/processed/xjtu_fixed_features_48.csv")
    if data.empty: st.error("Canonical fixed48 data are unavailable.")
    else:
        bearing = st.selectbox("Bearing", sorted(data.bearing_id.unique()))
        feature = st.selectbox("Feature", list(data.columns[-48:]))
        st.plotly_chart(px.line(data[data.bearing_id == bearing], x="step", y=feature), width="stretch")
elif page == "RUL Prediction":
    st.title("Direct RUL Prediction")
    pred = csv(FINAL / "predictions_common_range.csv")
    if pred.empty: st.warning("Final direct-RUL predictions have not been generated.")
    else:
        model = st.selectbox("Architecture", sorted(pred.model.unique()))
        seed = st.selectbox("Seed", sorted(pred.seed.unique()))
        bearing = st.selectbox("Bearing", sorted(pred.bearing_id.unique()))
        view = pred[(pred.model == model) & (pred.seed == seed) & (pred.bearing_id == bearing)]
        plot = view.melt("step", ["rul_minutes", "predicted_rul_minutes"],
                         var_name="series", value_name="minutes")
        st.plotly_chart(px.line(plot, x="step", y="minutes", color="series"), width="stretch")
        st.caption(f"{len(view)} common-range observations; no future vibration measurement is an input.")
elif page == "Training Evidence":
    st.title("Training and Validation Evidence")
    runs = state["runs"]
    curves = state["learning_curves"]
    if runs.empty:
        st.warning("Per-run training evidence is unavailable.")
    else:
        architecture = st.selectbox("Architecture", sorted(runs.model.unique()), key="training_model")
        view = runs[runs.model == architecture]
        st.dataframe(view[["fold", "seed", "sequence_length", "best_epoch",
                           "validation_macro_mae_minutes", "test_common_macro_mae_minutes",
                           "test_common_macro_rmse_minutes"]], hide_index=True)
        st.plotly_chart(px.box(view, x="fold", y="validation_macro_mae_minutes", points="all",
                              title="Validation-only checkpoint-selection error"), width="stretch")
    if not curves.empty:
        fold = st.selectbox("Learning-curve fold", sorted(curves.fold.unique()))
        selected = curves[curves.fold == fold]
        st.plotly_chart(px.line(selected, x="epoch", y="val_macro_mae_minutes",
                                color="model", line_dash="seed"), width="stretch")
elif page == "Maintenance Replay":
    st.title("Hypothetical Maintenance Replay")
    summary = csv(ROOT / "outputs/final/maintenance_replay/strategy_summary.csv")
    if summary.empty:
        st.warning("Maintenance replay results are unavailable; no values are inferred from model metrics.")
    else:
        scenario = st.selectbox("Cost scenario", sorted(summary.cost_scenario.unique()))
        view = summary[summary.cost_scenario == scenario]
        st.dataframe(view, hide_index=True)
        st.plotly_chart(px.bar(view, x="strategy", y="total_simulated_cost_units"), width="stretch")
        st.caption("Unitless assumed costs; no validated industrial savings claim.")
else:
    st.title("Final LSTM versus GRU Results")
    if state["model_summary"].empty:
        st.warning("No valid model summary is available.")
    else:
        st.dataframe(state["model_summary"], hide_index=True)
        metric = state["model_summary"].melt(
            id_vars="model", value_vars=["mean_macro_mae_minutes", "mean_macro_rmse_minutes"],
            var_name="metric", value_name="minutes")
        st.plotly_chart(px.bar(metric, x="model", y="minutes", color="metric", barmode="group"),
                        width="stretch")
    if not state["fold_metrics"].empty:
        fold_mean = state["fold_metrics"].groupby(["fold", "model"], as_index=False)[
            ["macro_mae_minutes", "macro_rmse_minutes"]].mean()
        st.subheader("Fold-level performance")
        st.plotly_chart(px.bar(fold_mean, x="fold", y="macro_mae_minutes", color="model",
                               barmode="group"), width="stretch")
    if not state["seed_metrics"].empty:
        st.subheader("Seed-level variability")
        st.plotly_chart(px.scatter(state["seed_metrics"], x="seed", y="macro_mae_minutes",
                                   color="model", symbol="model"), width="stretch")
        st.dataframe(state["seed_metrics"], hide_index=True)
    st.error("Outer XJTU results are exploratory because earlier work inspected outer-test outcomes.")
