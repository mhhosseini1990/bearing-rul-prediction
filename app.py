import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Bearing Predictive Maintenance", layout="wide")
st.title("Predictive Maintenance of Rolling Bearings")


# 1. Data / model layer 
def get_defect_probability(n_points: int, seed: int = 42) -> pd.DataFrame:
    """Simulates a bearing degrading over time: probability rises like an S-curve."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_points)
    p = 1 / (1 + np.exp(-(t - 0.7 * n_points) / (0.08 * n_points)))
    p = np.clip(p + rng.normal(0, 0.02, n_points), 0, 1)
    return pd.DataFrame({"time": t, "p_defect": p})


# 2. Sidebar controls 
st.sidebar.header("Settings")
n_points = st.sidebar.slider("Number of time steps", 50, 1000, 300)
threshold = st.sidebar.slider("Alarm threshold", 0.1, 0.99, 0.80)

df = get_defect_probability(n_points)

# 3. KPIs
current_p = df["p_defect"].iloc[-1]
col1, col2 = st.columns(2)
col1.metric("Current defect probability", f"{current_p:.1%}")
col2.metric("Status", "⚠️ MAINTENANCE NEEDED" if current_p >= threshold else "✅ Healthy")

# 4. Chart: X = time, Y = probability of defect 
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["time"], y=df["p_defect"],
                         mode="lines", name="P(defect)"))
fig.add_hline(y=threshold, line_dash="dash", line_color="red",
              annotation_text="Alarm threshold")
fig.update_layout(
    xaxis_title="Time (cycles / hours)",
    yaxis_title="Probability of defect",
    yaxis_range=[0, 1],
    height=500,
)
st.plotly_chart(fig, use_container_width=True)