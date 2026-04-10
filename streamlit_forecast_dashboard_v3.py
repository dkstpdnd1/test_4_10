import math
from pathlib import Path
from datetime import timedelta

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="LSTM 예측 결과 대시보드 v3", layout="wide")
st.title("LSTM 예측 결과 대시보드 v3")
st.markdown("예측값, baseline, 실제값을 함께 비교합니다.")

palette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

def apply_common_layout(fig, title_text, y_title, y_max, day_start, day_end):
    tick_hours = [0, 3, 6, 9, 12, 15, 18, 21, 24]
    tick_values = [day_start + timedelta(hours=h) for h in tick_hours]
    tick_text = [f"{h:02d}:00" if h < 24 else "24:00" for h in tick_hours]

    fig.update_layout(
        title=title_text,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(
            title="시간",
            range=[day_start, day_end],
            tickmode="array",
            tickvals=tick_values,
            ticktext=tick_text,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.10)",
            showline=True,
            linecolor="black",
            linewidth=1.2,
            mirror=True
        ),
        yaxis=dict(
            title=y_title,
            range=[0, y_max],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.10)",
            showline=True,
            linecolor="black",
            linewidth=1.2,
            mirror=True
        )
    )
    return fig

@st.cache_data
def load_result_files(result_dir: str):
    result_path = Path(result_dir)
    required = [
        "compare_pred_vs_actual.csv",
        "pred_people_wait.csv",
        "actual_people_wait.csv",
        "baseline_people_wait.csv",
        "overall_metrics.csv",
        "area_metrics.csv"
    ]

    for f in required:
        if not (result_path / f).exists():
            raise FileNotFoundError(f"필수 파일 없음: {f}")

    compare_df = pd.read_csv(result_path / "compare_pred_vs_actual.csv")
    pred_df = pd.read_csv(result_path / "pred_people_wait.csv")
    actual_df = pd.read_csv(result_path / "actual_people_wait.csv")
    baseline_df = pd.read_csv(result_path / "baseline_people_wait.csv")
    overall_df = pd.read_csv(result_path / "overall_metrics.csv")
    area_metrics_df = pd.read_csv(result_path / "area_metrics.csv")

    history_df = None
    hist_path = result_path / "training_history.csv"
    if hist_path.exists():
        history_df = pd.read_csv(hist_path)

    blend_df = None
    blend_path = result_path / "blend_alpha.csv"
    if blend_path.exists():
        blend_df = pd.read_csv(blend_path)

    for df in [compare_df, pred_df, actual_df, baseline_df]:
        df["actual_time"] = pd.to_datetime(df["actual_time"], errors="coerce")

    return compare_df, pred_df, actual_df, baseline_df, overall_df, area_metrics_df, history_df, blend_df

st.sidebar.header("결과 폴더")
result_dir = st.sidebar.text_input("결과 폴더", value="forecast_output_v3")

try:
    compare_df, pred_df, actual_df, baseline_df, overall_df, area_metrics_df, history_df, blend_df = load_result_files(result_dir)
except Exception as e:
    st.error(f"결과 로드 실패: {e}")
    st.stop()

target_date = pd.to_datetime(overall_df.loc[0, "target_date"]).date()
day_start = pd.to_datetime(target_date)
day_end = day_start + timedelta(days=1)

st.subheader("전체 성능 비교")
c1, c2, c3, c4 = st.columns(4)
c1.metric("People MAE (Pred)", f"{overall_df.loc[0, 'people_MAE_pred']:.3f}")
c2.metric("People MAE (Baseline)", f"{overall_df.loc[0, 'people_MAE_baseline']:.3f}")
c3.metric("Wait MAE (Pred)", f"{overall_df.loc[0, 'wait_MAE_pred']:.3f}")
c4.metric("Wait MAE (Baseline)", f"{overall_df.loc[0, 'wait_MAE_baseline']:.3f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("People RMSE (Pred)", f"{overall_df.loc[0, 'people_RMSE_pred']:.3f}")
c6.metric("People RMSE (Baseline)", f"{overall_df.loc[0, 'people_RMSE_baseline']:.3f}")
c7.metric("Wait RMSE (Pred)", f"{overall_df.loc[0, 'wait_RMSE_pred']:.3f}")
c8.metric("Wait RMSE (Baseline)", f"{overall_df.loc[0, 'wait_RMSE_baseline']:.3f}")

if history_df is not None and not history_df.empty:
    st.subheader("학습 Loss")
    fig_hist = go.Figure()
    if "loss" in history_df.columns:
        fig_hist.add_trace(go.Scatter(
            x=list(range(1, len(history_df) + 1)),
            y=history_df["loss"],
            mode="lines",
            name="loss"
        ))
    if "mae" in history_df.columns:
        fig_hist.add_trace(go.Scatter(
            x=list(range(1, len(history_df) + 1)),
            y=history_df["mae"],
            mode="lines",
            name="mae"
        ))
    fig_hist.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_hist, use_container_width=True)

areas = sorted(compare_df["area"].dropna().astype(str).unique().tolist())
default_areas = [a for a in areas if a != "Outside"]
if not default_areas:
    default_areas = areas

selected_areas = st.sidebar.multiselect(
    "표시 구역",
    options=areas,
    default=default_areas[:6] if len(default_areas) >= 6 else default_areas
)

if not selected_areas:
    st.warning("구역을 하나 이상 선택하세요.")
    st.stop()

plot_pred = pred_df[pred_df["area"].isin(selected_areas)].copy()
plot_actual = actual_df[actual_df["area"].isin(selected_areas)].copy()
plot_baseline = baseline_df[baseline_df["area"].isin(selected_areas)].copy()
plot_compare = compare_df[compare_df["area"].isin(selected_areas)].copy()

st.subheader("구역별 이용객 수: 예측 vs baseline vs 실제")
fig_people = go.Figure()

for i, area in enumerate(selected_areas):
    color = palette[i % len(palette)]

    sub_actual = plot_actual[plot_actual["area"] == area].sort_values("actual_time")
    sub_pred = plot_pred[plot_pred["area"] == area].sort_values("actual_time")
    sub_base = plot_baseline[plot_baseline["area"] == area].sort_values("actual_time")

    fig_people.add_trace(go.Scatter(
        x=sub_actual["actual_time"], y=sub_actual["num_people"],
        mode="lines", name=f"{area} 실제",
        line=dict(color=color, width=2)
    ))
    fig_people.add_trace(go.Scatter(
        x=sub_pred["actual_time"], y=sub_pred["num_people"],
        mode="lines", name=f"{area} 예측",
        line=dict(color=color, width=2, dash="dash")
    ))
    fig_people.add_trace(go.Scatter(
        x=sub_base["actual_time"], y=sub_base["num_people"],
        mode="lines", name=f"{area} baseline",
        line=dict(color=color, width=1, dash="dot")
    ))

people_y_max = max(
    float(plot_actual["num_people"].max()) if not plot_actual.empty else 1.0,
    float(plot_pred["num_people"].max()) if not plot_pred.empty else 1.0,
    float(plot_baseline["num_people"].max()) if not plot_baseline.empty else 1.0,
    1.0
)
fig_people = apply_common_layout(
    fig_people,
    f"{target_date} 구역별 이용객 수 비교",
    "인원 수",
    math.ceil(people_y_max * 1.1),
    day_start,
    day_end
)
st.plotly_chart(fig_people, use_container_width=True)

st.subheader("구역별 대기시간: 예측 vs baseline vs 실제")
fig_wait = go.Figure()

for i, area in enumerate(selected_areas):
    color = palette[i % len(palette)]

    sub_actual = plot_actual[plot_actual["area"] == area].sort_values("actual_time")
    sub_pred = plot_pred[plot_pred["area"] == area].sort_values("actual_time")
    sub_base = plot_baseline[plot_baseline["area"] == area].sort_values("actual_time")

    fig_wait.add_trace(go.Scatter(
        x=sub_actual["actual_time"], y=sub_actual["wait_time_min"],
        mode="lines", name=f"{area} 실제",
        line=dict(color=color, width=2)
    ))
    fig_wait.add_trace(go.Scatter(
        x=sub_pred["actual_time"], y=sub_pred["wait_time_min"],
        mode="lines", name=f"{area} 예측",
        line=dict(color=color, width=2, dash="dash")
    ))
    fig_wait.add_trace(go.Scatter(
        x=sub_base["actual_time"], y=sub_base["wait_time_min"],
        mode="lines", name=f"{area} baseline",
        line=dict(color=color, width=1, dash="dot")
    ))

wait_y_max = max(
    float(plot_actual["wait_time_min"].max()) if not plot_actual.empty else 1.0,
    float(plot_pred["wait_time_min"].max()) if not plot_pred.empty else 1.0,
    float(plot_baseline["wait_time_min"].max()) if not plot_baseline.empty else 1.0,
    1.0
)
fig_wait = apply_common_layout(
    fig_wait,
    f"{target_date} 구역별 대기시간 비교",
    "대기시간 (분)",
    round(wait_y_max * 1.15, 2),
    day_start,
    day_end
)
st.plotly_chart(fig_wait, use_container_width=True)

st.subheader("구역별 성능 지표")
st.dataframe(area_metrics_df, use_container_width=True)

if blend_df is not None and not blend_df.empty:
    st.subheader("구역별 Blend Alpha")
    st.dataframe(blend_df.sort_values("area"), use_container_width=True)

st.subheader("마지막 시점 비교")
last_ts = plot_compare["actual_time"].max()
last_compare = (
    plot_compare[plot_compare["actual_time"] == last_ts][[
        "area",
        "num_people_pred", "num_people_actual", "num_people_baseline",
        "wait_time_min_pred", "wait_time_min_actual", "wait_time_min_baseline",
        "abs_err_people", "abs_err_people_baseline",
        "abs_err_wait", "abs_err_wait_baseline"
    ]]
    .sort_values("area")
    .copy()
)
st.dataframe(last_compare, use_container_width=True)

st.subheader("상세 비교")
detail_df = plot_compare.copy()
detail_df["actual_time"] = detail_df["actual_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
detail_df = detail_df.sort_values(["actual_time", "area"])
st.dataframe(detail_df, use_container_width=True)

csv_data = detail_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="상세 비교 CSV 다운로드",
    data=csv_data,
    file_name=f"compare_pred_vs_actual_{target_date}.csv",
    mime="text/csv"
)
