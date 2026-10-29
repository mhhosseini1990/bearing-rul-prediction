"""Interactive dashboard for saved XJTU-SY LSTM/GRU experiment results.

Run from the repository root: python -m streamlit run app.py
No synthetic probabilities, retraining, or live inference are performed here.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / 'outputs'
st.set_page_config(page_title='Bearing RUL | LSTM vs GRU', page_icon='⚙️', layout='wide')


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_optional(folder: Path, name: str) -> pd.DataFrame | None:
    path = folder / name
    if not path.is_file():
        return None
    try:
        return read_csv(str(path))
    except Exception as exc:
        st.warning(f'Cannot read {path.name}: {exc}')
        return None


def find_column(frame: pd.DataFrame, *options: str) -> str | None:
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    return next((lookup[x.lower()] for x in options if x.lower() in lookup), None)


def prediction_frame(frame: pd.DataFrame, model: str) -> pd.DataFrame | None:
    aliases = {
        'bearing_id': ('bearing_id', 'bearing', 'bearing_name', 'test_bearing', 'unit_id'),
        'step': ('step', 'target_step', 'snapshot_index', 'snapshot_step', 'time_step'),
        'actual': ('actual_rul_minutes', 'true_rul_minutes', 'rul_true_minutes', 'rul_actual_minutes', 'y_true_minutes', 'actual_rul'),
        'predicted': ('predicted_rul_minutes', 'prediction_minutes', 'pred_rul_minutes', 'rul_pred_minutes', 'y_pred_minutes', 'predicted_rul'),
    }
    columns = {key: find_column(frame, *values) for key, values in aliases.items()}
    missing = [key for key, value in columns.items() if value is None]
    if missing:
        st.error(f'{model} predictions: missing columns {missing}. Available: {list(frame.columns)}')
        return None
    result = frame.rename(columns={v: k for k, v in columns.items()})[list(aliases)].copy()
    for col in ('step', 'actual', 'predicted'):
        result[col] = pd.to_numeric(result[col], errors='coerce')
    result = result.dropna(subset=['bearing_id', 'step', 'actual', 'predicted'])
    result['model'] = model
    return result.sort_values(['bearing_id', 'step'])


st.title('⚙️ Predictive Maintenance of Rolling Bearings')
st.caption('XJTU-SY • Wavelet features • LSTM vs GRU • Offline historical evaluation')

runs = sorted(p.name for p in OUTPUTS.iterdir() if p.is_dir()) if OUTPUTS.exists() else []
if not runs:
    st.error('No experiment outputs found. Expected: outputs/main/evaluation/')
    st.stop()
run_name = st.sidebar.selectbox('Experiment run', runs, index=runs.index('main') if 'main' in runs else 0)
base = OUTPUTS / run_name
evaluation = base / 'evaluation'
training = base / 'training'
replay = base / 'maintenance_replay'

st.sidebar.info('This dashboard reads saved experiment results. It does not calculate defect probabilities or run live sensor inference.')
page = st.sidebar.radio('Dashboard section', ['Overview', 'Model comparison', 'RUL predictions', 'Maintenance replay', 'Training analysis'])

comparison = load_optional(evaluation, 'comparison.csv')
per_bearing = load_optional(evaluation, 'per_bearing_metrics.csv')
predictions = []
for model in ('LSTM', 'GRU'):
    raw = load_optional(evaluation, f'{model.lower()}_predictions.csv')
    if raw is not None:
        normalized = prediction_frame(raw, model)
        if normalized is not None:
            predictions.append(normalized)
preds = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()

if page == 'Overview':
    st.subheader('Project overview')
    st.write('Predict remaining useful life (RUL) in minutes from historical vibration measurements using eight wavelet-energy features and 20-snapshot sequences.')
    a, b, c, d = st.columns(4)
    a.metric('Dataset', 'XJTU-SY')
    b.metric('Bearings', '15')
    c.metric('Train / Val / Test', '9 / 3 / 3')
    d.metric('Models', 'LSTM + GRU')
    st.markdown('**Processing pipeline**')
    st.code('Vibration CSVs → db4 DWT (8 features) → train-only scaling → 20-step windows → LSTM / GRU → RUL (min) → maintenance replay', language='text')
    st.info('Historical replay is an illustrative offline decision-support simulation, not verified factory deployment or cost savings.')
    st.markdown('**Files available for selected run**')
    st.dataframe(pd.DataFrame({'File': ['Evaluation comparison', 'Per-bearing metrics', 'LSTM predictions', 'GRU predictions', 'Replay events', 'Replay summary'], 'Available': [(evaluation / 'comparison.csv').exists(), (evaluation / 'per_bearing_metrics.csv').exists(), (evaluation / 'lstm_predictions.csv').exists(), (evaluation / 'gru_predictions.csv').exists(), (replay / 'replay_events.csv').exists(), (replay / 'replay_summary.csv').exists()]}), hide_index=True)

elif page == 'Model comparison':
    st.subheader('Model performance on held-out bearings')
    if comparison is not None:
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        model_col = find_column(comparison, 'model')
        mae_col = find_column(comparison, 'mae_minutes', 'mae', 'overall_mae_minutes')
        rmse_col = find_column(comparison, 'rmse_minutes', 'rmse', 'overall_rmse_minutes')
        if model_col and mae_col and rmse_col:
            tidy = comparison.melt(id_vars=model_col, value_vars=[mae_col, rmse_col], var_name='Metric', value_name='Minutes')
            st.plotly_chart(px.bar(tidy, x='Metric', y='Minutes', color=model_col, barmode='group', title='Test MAE and RMSE (minutes)'), use_container_width=True)
    else:
        st.warning('Missing evaluation/comparison.csv')
    if per_bearing is not None:
        st.markdown('**Performance by bearing**')
        st.dataframe(per_bearing, use_container_width=True, hide_index=True)
        model_col = find_column(per_bearing, 'model')
        bearing_col = find_column(per_bearing, 'bearing_id', 'bearing')
        mae_col = find_column(per_bearing, 'mae_minutes', 'mae')
        if model_col and bearing_col and mae_col:
            st.plotly_chart(px.bar(per_bearing, x=bearing_col, y=mae_col, color=model_col, barmode='group', title='MAE by test bearing'), use_container_width=True)
    st.caption('Overall scores weight each prediction window; bearing histories have different lengths.')

elif page == 'RUL predictions':
    st.subheader('Actual vs predicted remaining useful life')
    if preds.empty:
        st.warning('No valid prediction CSV files found in evaluation/.')
    else:
        bearings = sorted(preds['bearing_id'].unique())
        selected = st.selectbox('Test bearing', bearings)
        chosen_models = st.multiselect('Models', sorted(preds['model'].unique()), default=sorted(preds['model'].unique()))
        subset = preds[(preds['bearing_id'] == selected) & (preds['model'].isin(chosen_models))]
        fig = go.Figure()
        if not subset.empty:
            actual = subset.sort_values('step').drop_duplicates('step')
            fig.add_trace(go.Scatter(x=actual['step'], y=actual['actual'], mode='lines', name='Actual RUL', line=dict(color='black', width=3)))
            for model, rows in subset.groupby('model'):
                fig.add_trace(go.Scatter(x=rows['step'], y=rows['predicted'], mode='lines', name=f'{model} predicted RUL'))
        fig.update_layout(xaxis_title='Snapshot index (1 minute between snapshots)', yaxis_title='RUL (minutes)', height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.caption('Each point uses the last snapshot of a past-only 20-snapshot window. These are saved test predictions, not new uploaded-signal inference.')

elif page == 'Maintenance replay':
    st.subheader('Historical maintenance replay')
    st.info('Saved simulation: threshold 120 min; 3 consecutive predictions; minimum useful lead 30 min. Settings are fixed for this saved run.')
    summary = load_optional(replay, 'replay_summary.csv')
    events = load_optional(replay, 'replay_events.csv')
    timeline = load_optional(replay, 'replay_timeline.csv')
    if summary is None or events is None:
        st.warning('Replay results missing. Run: python scripts/run_maintenance_replay.py --run-name ' + run_name)
    else:
        st.markdown('**Per-model summary**')
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.markdown('**First confirmed alert per bearing**')
        st.dataframe(events, use_container_width=True, hide_index=True)
        model_col = find_column(events, 'model')
        bearing_col = find_column(events, 'bearing_id')
        lead_col = find_column(events, 'actual_lead_time_minutes')
        if model_col and bearing_col and lead_col:
            fig = px.bar(events, x=bearing_col, y=lead_col, color=model_col, barmode='group', title='Actual lead time at first alert')
            fig.add_hline(y=30, line_dash='dash', annotation_text='Minimum useful lead (30 min)')
            st.plotly_chart(fig, use_container_width=True)
        if timeline is not None:
            with st.expander('Inspect replay timeline'):
                st.dataframe(timeline, use_container_width=True, hide_index=True)
        st.caption('Early / timely / late are defined by the replay rules; they do not establish real-world maintenance effectiveness.')

elif page == 'Training analysis':
    st.subheader('Training vs validation loss')
    model = st.selectbox('Model', ['LSTM', 'GRU'])
    history = load_optional(training, f'{model.lower()}_history.csv')
    if history is None:
        st.warning(f'Missing training/{model.lower()}_history.csv')
    else:
        epoch = find_column(history, 'epoch')
        train = find_column(history, 'train_mse_scaled', 'train_loss')
        val = find_column(history, 'validation_mse_scaled', 'val_mse_scaled', 'val_loss')
        if epoch and train and val:
            chart = go.Figure()
            chart.add_trace(go.Scatter(x=history[epoch], y=history[train], name='Training MSE', mode='lines+markers'))
            chart.add_trace(go.Scatter(x=history[epoch], y=history[val], name='Validation MSE', mode='lines+markers'))
            best = history.loc[pd.to_numeric(history[val], errors='coerce').idxmin()]
            chart.add_vline(x=float(best[epoch]), line_dash='dash', annotation_text=f'Best epoch {int(best[epoch])}')
            chart.update_layout(xaxis_title='Epoch', yaxis_title='MSE (scaled RUL)', height=500)
            st.plotly_chart(chart, use_container_width=True)
            st.metric('Best validation epoch', int(best[epoch]))
            st.metric('Best validation MSE (scaled)', f'{float(best[val]):.4f}')
        else:
            st.error(f'Unexpected history columns: {list(history.columns)}')
        st.dataframe(history, use_container_width=True, hide_index=True)
    st.caption('A rising validation loss with falling training loss is consistent with overfitting, but may also reflect differences between bearing histories.')
